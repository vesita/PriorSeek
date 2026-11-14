//! 双语解耦神经稀疏检索适配器 (Bilingual Sparse Adapter)。
//!
//! 消费离线语系特化扩散工件 (bilingual-sparse-v1)：
//! - 中文语料使用中文专属扩散头（naver/neuclir22-splade-zh，32000 维）；
//! - 英文语料使用英文专属扩散头（naver/splade-cocondenser-ensembledistil，30522 维）；
//! - 附带中英双向概念映射枢纽 (BilingualLexicon)；
//! - 外壳执行纯 CPU 带权倒排累加，零超参，零运行时神经网络计算。

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::{BilingualLexicon, SparseInvertedIndex};
use std::collections::HashMap;

pub struct BilingualSparseAdapter {
    info: AdapterInfo,
    index: Option<SparseInvertedIndex>,
    query_sparse: HashMap<String, Vec<(u32, f32)>>,
    lexicon: Option<BilingualLexicon>,
    artifact_dir: std::path::PathBuf,
}

impl BilingualSparseAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/bilingual-sparse-v1");
        let dir = std::env::var("SL_BILINGUAL_SPARSE_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::PathBuf::from(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 bilingual-sparse-v1 工件目录 {}（先跑 tools/kernel/build_bilingual_sparse.py {dataset}）",
                dir
            );
        }

        let artifact = sl_core::KernelArtifact::from_dir(&path)?;
        let query_sparse = Self::load_query_sparse(&path)?;
        let lexicon = Self::load_lexicon(&path)?;

        let config = serde_json::json!({
            "framework": "bilingual-token-hub + language-specific-diffusion",
            "runtime": "pure-rust-cpu-sparse-inverted-index",
            "prior_factor": {
                "name": "doc_sparse_vector",
                "kind": "token_confidence_list",
                "sha256": artifact.files.get("sparse.bin").cloned().unwrap_or_default(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "bilingual-sparse",
                "bilingual-sparse-splade",
                "0.1.0",
                "外壳：crates/sl-adapters/src/bilingual_sparse.rs；内核：双语解耦特化扩散",
            )
            .with_config(config),
            index: None,
            query_sparse,
            lexicon,
            artifact_dir: path,
        })
    }

    fn load_lexicon(dir: &std::path::Path) -> Result<Option<BilingualLexicon>> {
        let path = dir.join("bilingual_lexicon.json");
        if path.is_file() {
            let content = std::fs::read_to_string(&path)?;
            let lex = BilingualLexicon::from_json(&content)?;
            Ok(Some(lex))
        } else {
            Ok(None)
        }
    }

    fn load_query_sparse(dir: &std::path::Path) -> Result<HashMap<String, Vec<(u32, f32)>>> {
        let mut map = HashMap::new();

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

    pub fn lexicon(&self) -> Option<&BilingualLexicon> {
        self.lexicon.as_ref()
    }
}

impl Adapter for BilingualSparseAdapter {
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
