//! 双路无超参 RRF 混合检索与单路稠密向量检索适配器 (Hybrid RRF & Dense Adapter)。
//!
//! 1. DenseAdapter: 纯 CPU 稠密语义检索，直接消费离线烘焙的 384 维 L2 归一化二进制向量工件；
//! 2. HybridRrfAdapter: 将词法概念先验路 (concept-synonym) 与稠密语义路 (dense) 通过无参数 RRF(60) 融合。

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::dense::{rrf_fuse, DenseIndex};
use std::collections::HashMap;

/// 单路稠密向量适配器 (纯 CPU 极速点积)。
pub struct DenseAdapter {
    info: AdapterInfo,
    index: Option<DenseIndex>,
    query_embeddings: HashMap<String, Vec<f32>>,
    artifact_dir: std::path::PathBuf,
}

impl DenseAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/dense-v0");
        let dir = std::env::var("SL_DENSE_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::PathBuf::from(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 dense-v0 工件目录 {}（先跑 tools/kernel/build_dense_corpus.py {dataset}）",
                dir
            );
        }

        let artifact = sl_core::KernelArtifact::from_dir(&path)?;
        let query_embeddings = Self::load_query_embeddings(&path)?;

        let config = serde_json::json!({
            "model": "all-MiniLM-L6-v2",
            "dim": 384,
            "runtime": "pure-rust-cpu-dot-product",
            "prior_factor": {
                "name": "doc_dense_embedding",
                "kind": "token_confidence_list",
                "sha256": artifact.files.get("dense.bin").cloned().unwrap_or_default(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "dense",
                "dense-semantic-embedding",
                "0.2.0",
                "外壳：crates/sl-adapters/src/hybrid_rrf.rs；内核：all-MiniLM-L6-v2 离线烘焙二进制",
            )
            .with_config(config),
            index: None,
            query_embeddings,
            artifact_dir: path,
        })
    }

    fn load_query_embeddings(dir: &std::path::Path) -> Result<HashMap<String, Vec<f32>>> {
        let q_bin_path = dir.join("query_embeddings.bin");
        let q_ids_path = dir.join("query_ids.txt");
        if !q_bin_path.is_file() || !q_ids_path.is_file() {
            return Ok(HashMap::new());
        }

        let ids_content = std::fs::read_to_string(&q_ids_path)?;
        let q_ids: Vec<String> = ids_content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        let bin_bytes = std::fs::read(&q_bin_path)?;
        if bin_bytes.len() < 16 || &bin_bytes[0..8] != b"SLQEMBv1" {
            anyhow::bail!("query_embeddings.bin 格式错误");
        }

        let num_queries = u32::from_le_bytes(bin_bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bin_bytes[12..16].try_into().unwrap()) as usize;

        if num_queries != q_ids.len() {
            anyhow::bail!("query_embeddings.bin 与 query_ids.txt 数量不一致");
        }

        let float_bytes = &bin_bytes[16..];
        let mut map = HashMap::with_capacity(num_queries * 2);
        let mut all_vectors = Vec::with_capacity(num_queries);
        for i in 0..num_queries {
            let offset = i * dim * 4;
            let slice = &float_bytes[offset..offset + dim * 4];
            let mut vec = Vec::with_capacity(dim);
            for chunk in slice.chunks_exact(4) {
                vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            all_vectors.push(vec);
        }

        for (q_id, vec) in q_ids.iter().zip(&all_vectors) {
            map.insert(q_id.clone(), vec.clone());
        }

        // 同时支持按 query 文本直接检索
        let q_texts_path = dir.join("query_texts.json");
        if q_texts_path.is_file() {
            if let Ok(bytes) = std::fs::read(&q_texts_path) {
                if let Ok(text_map) = serde_json::from_slice::<HashMap<String, usize>>(&bytes) {
                    for (text, idx) in text_map {
                        if let Some(vec) = all_vectors.get(idx) {
                            map.insert(text, vec.clone());
                        }
                    }
                }
            }
        }

        Ok(map)
    }
}

impl Adapter for DenseAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        let dense_path = self.artifact_dir.join("dense.bin");
        let ids_path = self.artifact_dir.join("doc_ids.txt");

        let bin_bytes = std::fs::read(&dense_path)
            .with_context(|| format!("读取 {} 失败", dense_path.display()))?;
        let ids_content = std::fs::read_to_string(&ids_path)
            .with_context(|| format!("读取 {} 失败", ids_path.display()))?;
        let doc_ids: Vec<String> = ids_content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        if doc_ids.len() != docs.len() {
            anyhow::bail!(
                "工件 doc_ids 数量 ({}) 与传入评测的文档数 ({}) 不一致",
                doc_ids.len(),
                docs.len()
            );
        }

        let index = DenseIndex::load_from_bytes(&bin_bytes, doc_ids)?;
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let index = self.index.as_ref().context("未建索引（先调 index）")?;
        // 优先在预编码查询库中按精确内容查找，若无则取默认
        let query_vec = match self.query_embeddings.get(query) {
            Some(v) => v,
            None => {
                // 如果是逐 query 文本匹配，也可做在线兜底（此处在离线对拍中优先按 query 查找）
                return Ok(Vec::new());
            }
        };

        Ok(index.search(query_vec, k))
    }
}

/// 双路无超参 RRF 混合检索适配器。
pub struct HybridRrfAdapter {
    info: AdapterInfo,
    lexical: Box<dyn Adapter>,
    dense: DenseAdapter,
}

impl HybridRrfAdapter {
    pub fn new() -> Result<Self> {
        let lexical = Box::new(crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?);
        let dense = DenseAdapter::new()?;

        let config = serde_json::json!({
            "fusion": "reciprocal_rank_fusion (RRF-60)",
            "hyperparameters": "none (pure rank reciprocity, no arbitrary weight)",
            "route_a": "concept-synonym (lexical & morphological & domain synonyms)",
            "route_b": "dense (all-MiniLM-L6-v2 384-dim semantic embeddings)",
        });

        Ok(Self {
            info: AdapterInfo::new(
                "hybrid-rrf",
                "hybrid-reciprocal-rank-fusion",
                "0.2.0",
                "双路融合外壳：sl-engine::rrf_fuse (RRF-60)；词法路：concept-synonym；语义路：dense-v0",
            )
            .with_config(config),
            lexical,
            dense,
        })
    }
}

impl Adapter for HybridRrfAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.lexical.index(docs)?;
        self.dense.index(docs)?;
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        // 两路各检索 Top-60 候选集合
        let candidate_depth = k.max(60);
        let hits_lexical = self.lexical.search(query, candidate_depth)?;
        let hits_dense = self.dense.search(query, candidate_depth)?;

        if hits_dense.is_empty() {
            // 如果稠密路无对应查询向量，优雅退化为纯词法路
            return Ok(hits_lexical.into_iter().take(k).collect());
        }

        // 零超参数无偏倒数排名融合 (RRF-60)
        Ok(rrf_fuse(&hits_lexical, &hits_dense, k, 60.0))
    }
}
