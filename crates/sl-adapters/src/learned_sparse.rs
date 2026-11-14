//! 神经稀疏先验检索适配器 (Learned Sparse Adapter)。
//!
//! 消费离线神经词表投影先验工件 (learned-sparse-v1, 基于 SPLADE 架构)：
//! - 离线工件：文档与查询的离散带权稀疏向量 (sparse.bin / query_sparse.json)；
//! - 在线外壳：纯 Rust 实现的带权倒排索引 (SparseInvertedIndex)，纯 CPU 运行，极速带权累加；
//! - 零超参数：无 BM25 经验参数，无启发式加权衰减，全部语义词权均由神经内核离线确定。

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::SparseInvertedIndex;
use std::collections::HashMap;

/// 单路神经稀疏先验检索适配器
pub struct LearnedSparseAdapter {
    info: AdapterInfo,
    index: Option<SparseInvertedIndex>,
    query_sparse: HashMap<String, Vec<(u32, f32)>>,
    artifact_dir: std::path::PathBuf,
}

impl LearnedSparseAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/learned-sparse-v1");
        let dir = std::env::var("SL_LEARNED_SPARSE_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::PathBuf::from(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 learned-sparse-v1 工件目录 {}（先跑 tools/kernel/build_learned_sparse.py {dataset}）",
                dir
            );
        }

        let artifact = sl_core::KernelArtifact::from_dir(&path)?;
        let query_sparse = Self::load_query_sparse(&path)?;

        let config = serde_json::json!({
            "model": "naver/splade-cocondenser-ensembledistil",
            "vocab_size": 30522,
            "runtime": "pure-rust-cpu-sparse-inverted-index",
            "prior_factor": {
                "name": "doc_sparse_vector",
                "kind": "token_sparse_weights",
                "sha256": artifact.files.get("sparse.bin").cloned().unwrap_or_default(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "learned-sparse",
                "learned-sparse-splade",
                "0.1.0",
                "外壳：crates/sl-adapters/src/learned_sparse.rs；内核：SPLADE 离线烘焙神经稀疏词权",
            )
            .with_config(config),
            index: None,
            query_sparse,
            artifact_dir: path,
        })
    }

    fn load_query_sparse(dir: &std::path::Path) -> Result<HashMap<String, Vec<(u32, f32)>>> {
        let mut map = HashMap::new();

        // 优先加载 query_sparse_by_text.json（按文本直接映射）
        let text_path = dir.join("query_sparse_by_text.json");
        if text_path.is_file() {
            let bytes = std::fs::read(&text_path)
                .with_context(|| format!("读取 {} 失败", text_path.display()))?;
            let raw: HashMap<String, Vec<(u32, f32)>> = serde_json::from_slice(&bytes)
                .with_context(|| format!("解析 {} 失败", text_path.display()))?;
            for (k, v) in raw {
                map.insert(k, v);
            }
        }

        // 同时支持按 query_id 映射
        let id_path = dir.join("query_sparse.json");
        if id_path.is_file() {
            let bytes = std::fs::read(&id_path)
                .with_context(|| format!("读取 {} 失败", id_path.display()))?;
            let raw: HashMap<String, Vec<(u32, f32)>> = serde_json::from_slice(&bytes)
                .with_context(|| format!("解析 {} 失败", id_path.display()))?;
            for (k, v) in raw {
                map.insert(k, v);
            }
        }

        Ok(map)
    }
}

impl Adapter for LearnedSparseAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        let sparse_path = self.artifact_dir.join("sparse.bin");
        let ids_path = self.artifact_dir.join("doc_ids.txt");

        let bin_bytes = std::fs::read(&sparse_path)
            .with_context(|| format!("读取 {} 失败", sparse_path.display()))?;
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

        let index = SparseInvertedIndex::from_bytes(&bin_bytes, doc_ids)?;
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("稀疏索引尚未构建"))?;

        let query_terms = self
            .query_sparse
            .get(query)
            .ok_or_else(|| anyhow::anyhow!("未找到 query '{}' 的预烘焙稀疏词权向量", query))?;

        Ok(index.search(query_terms, k))
    }
}

/// 神经双路混合检索适配器 (Neural Hybrid: Learned Sparse + Dense)。
///
/// 完全基于离线深度学习内核产出的两类先验因子：
/// 1. 神经稀疏词权先验 (learned-sparse, 基于 SPLADE 30522 维词表投影)；
/// 2. 连续语义向量先验 (dense, 基于 MiniLM 384 维向量)；
/// 两路均在离线预计算完毕，外壳使用纯 CPU 无超参 RRF(60) 融合。
pub struct NeuralHybridAdapter {
    inner: sl_bridge::RrfBridge,
}

impl NeuralHybridAdapter {
    pub fn new() -> Result<Self> {
        let sparse = Box::new(LearnedSparseAdapter::new()?);
        let dense = Box::new(crate::hybrid_rrf::DenseAdapter::new()?);
        let inner = sl_bridge::RrfBridge::new(
            "neural-hybrid",
            "neural-hybrid-fusion",
            "纯深度学习先验双路融合：learned-sparse(SPLADE) + dense(MiniLM)",
            vec![sparse, dense],
        );
        Ok(Self { inner })
    }
}

impl Adapter for NeuralHybridAdapter {
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
