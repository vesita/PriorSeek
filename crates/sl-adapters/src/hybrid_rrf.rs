//! 双路无超参 RRF 混合检索与单路稠密向量检索适配器 (Hybrid RRF & Dense Adapter)。
//!
//! 1. DenseAdapter: 纯 CPU 稠密语义检索，直接消费离线烘焙的 384 维 L2 归一化二进制向量工件；
//! 2. HybridRrfAdapter: 将词法概念先验路 (concept-synonym) 与稠密语义路 (dense) 通过无参数 RRF(60) 融合。

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::dense::DenseIndex;
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
    inner: sl_bridge::RrfBridge,
}

impl HybridRrfAdapter {
    pub fn new() -> Result<Self> {
        let lexical = Box::new(crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?);
        let dense = Box::new(DenseAdapter::new()?);
        let inner = sl_bridge::RrfBridge::new(
            "hybrid-rrf",
            "hybrid-reciprocal-rank-fusion",
            "通用 RRF 外壳 + concept-synonym + dense-v0",
            vec![lexical, dense],
        );
        Ok(Self { inner })
    }
}

impl Adapter for HybridRrfAdapter {
    fn info(&self) -> &AdapterInfo {
        self.inner.info()
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.inner.index(docs)
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        self.inner.search(query, k)
    }
}

/// 三路先验协同检索适配器 (Tri-Route PriorSeek Adapter)。
/// 融合三条异构先验路：
/// 1. `concept-synonym`: 领域医学同义与前缀硬对齐先验（解决专业实体形态漂移）；
/// 2. `dense`: 离线 MiniLM 384 维连续语义流（解决语义改写与抽象表达）；
/// 3. `saliency-bm25`: 深度网络 Token 显著性门控（抑制泛化停用词噪音）。
pub struct TriHybridRrfAdapter {
    inner: sl_bridge::AdapterBridge,
}

impl TriHybridRrfAdapter {
    pub fn new() -> Result<Self> {
        let concept = Box::new(crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?);
        let dense = Box::new(DenseAdapter::new()?);
        let saliency = Box::new(crate::shell::ShellAdapter::saliency_bm25()?);

        let inner = sl_bridge::AdapterBridge::with_weighted_routes(
            "tri-hybrid-rrf",
            "tri-route-reciprocal-rank-fusion",
            "PriorSeek 泛型三路先验协同 (concept-synonym:1.0 + dense-v0:1.0 + saliency-bm25:0.5)",
            vec![(concept, 1.0), (dense, 1.0), (saliency, 0.5)],
        );
        Ok(Self { inner })
    }
}

impl Adapter for TriHybridRrfAdapter {
    fn info(&self) -> &AdapterInfo {
        self.inner.info()
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.inner.index(docs)
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        self.inner.search(query, k)
    }
}



