//! 神经编译稀疏索引适配器 (Learned Sparse Lexical Routing)。
//!
//! 在离线索引期将内核产出的文档级高置信度语义扩展词项 (Document Expansion)
//! 直接“编译”进倒排索引（PositionalIndex），线上运行时 100% 纯 CPU，零神经网络计算。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::bow::BowScorer;
use sl_engine::index::PositionalIndex;
use std::collections::HashMap;
use std::sync::Arc;

pub struct SparseRouteAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    scorer: Option<BowScorer>,
    /// doc_id -> [(语义扩展词, 置信度)]
    expansions: HashMap<String, Vec<(String, f64)>>,
    vocabulary: HashMap<String, u32>,
    /// 扩展词阻尼衰减权重
    alpha: f64,
    /// 最小准入置信度阈值
    min_conf: f64,
}

impl SparseRouteAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/sparse-route-v1");
        let dir = std::env::var("SL_SPARSE_ROUTE_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 sparse-route 工件目录 {}（先跑 tools/kernel/build_sparse_routing.py {dataset}）",
                dir
            );
        }
        let alpha = std::env::var("SL_SPARSE_ALPHA")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.3);
        let min_conf = std::env::var("SL_SPARSE_MIN_CONF")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.25);

        let (expansions, sha256) = Self::load_expansions(path)?;

        let config = serde_json::json!({
            "alpha": alpha,
            "min_conf": min_conf,
            "prior_factor": {
                "name": "doc_expansion",
                "kind": "token_confidence_list",
                "sha256": sha256,
                "docs_with_expansions": expansions.len(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "sparse-route",
                "sparse-lexical-routing",
                "0.2.0",
                "外壳：crates/sl-adapters/src/sparse_route.rs；内核编译：tools/kernel/build_sparse_routing.py",
            )
            .with_config(config),
            index: None,
            scorer: None,
            expansions,
            vocabulary: HashMap::new(),
            alpha,
            min_conf,
        })
    }

    /// 从工件目录加载并校验 doc_expansions.json。
    fn load_expansions(
        dir: &std::path::Path,
    ) -> Result<(HashMap<String, Vec<(String, f64)>>, String)> {
        let artifact = sl_core::KernelArtifact::from_dir(dir)?;
        let payload_path = dir.join("doc_expansions.json");
        let bytes = std::fs::read(&payload_path)
            .with_context(|| format!("读取 {} 失败", payload_path.display()))?;

        // 校验 sha256
        let expected = artifact
            .files
            .get("doc_expansions.json")
            .ok_or_else(|| anyhow::anyhow!("工件缺少 doc_expansions.json 哈希登记"))?;
        let actual = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };
        if actual != *expected {
            anyhow::bail!(
                "{} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）",
                payload_path.display()
            );
        }

        let parsed: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("解析 {} 失败", payload_path.display()))?;
        let mut map = HashMap::new();
        if let Some(entries) = parsed.get("entries").and_then(|v| v.as_object()) {
            for (doc_id, rows) in entries {
                let mut list = Vec::new();
                if let Some(rows) = rows.as_array() {
                    for row in rows {
                        let Some(pair) = row.as_array() else { continue };
                        let (Some(word), Some(score)) = (
                            pair.first().and_then(|v| v.as_str()),
                            pair.get(1).and_then(|v| v.as_f64()),
                        ) else {
                            continue;
                        };
                        list.push((word.to_string(), score));
                    }
                }
                map.insert(doc_id.clone(), list);
            }
        }
        Ok((map, actual))
    }
}

impl Adapter for SparseRouteAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.vocabulary.clear();

        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());

        for doc in docs {
            // 1. 原文词项
            let raw_terms = bm25_terms(&doc.full_text());
            let mut ids: Vec<u32> = Vec::with_capacity(raw_terms.len() + 16);

            for term in raw_terms {
                let next = self.vocabulary.len() as u32;
                let id = *self.vocabulary.entry(term).or_insert(next);
                ids.push(id);
            }

            // 2. 神经编译：将离线语义扩展词项无缝追加到文档倒排索引
            if let Some(ext_list) = self.expansions.get(&doc.id) {
                for (ext_word, conf) in ext_list {
                    if *conf >= self.min_conf && self.alpha > 0.0 {
                        let next = self.vocabulary.len() as u32;
                        let ext_id = *self.vocabulary.entry(ext_word.clone()).or_insert(next);
                        ids.push(ext_id);
                    }
                }
            }

            encoded.push((doc.id.clone(), ids));
        }

        let index = Arc::new(PositionalIndex::build(encoded));
        self.scorer = Some(BowScorer::new(Arc::clone(&index)));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let scorer = self.scorer.as_ref().context("未建索引（先调 index）")?;
        let terms = bm25_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let weighted: Vec<(u32, f64)> = terms
            .into_iter()
            .filter_map(|term| self.vocabulary.get(&term).map(|&id| (id, 1.0)))
            .collect();

        if weighted.is_empty() {
            return Ok(Vec::new());
        }

        // 线上完全是纯 CPU 倒排查表，零神经网络开销
        Ok(scorer.search(&weighted, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_route_compiles_expansion_terms_into_index() {
        let mut adapter = SparseRouteAdapter {
            info: AdapterInfo::new("test", "test", "0.2.0", "test"),
            index: None,
            scorer: None,
            expansions: HashMap::new(),
            vocabulary: HashMap::new(),
            alpha: 0.5,
            min_conf: 0.2,
        };

        // 模拟文档 d1 原文只有 "heart attack"，但离线语义扩展了 "myocardial"
        let mut ext = HashMap::new();
        ext.insert("d1".to_string(), vec![("myocardial".to_string(), 0.9)]);
        adapter.expansions = ext;

        let docs = vec![Doc {
            id: "d1".to_string(),
            title: "Heart Attack".to_string(),
            text: "Patient suffered severe chest pain.".to_string(),
        }];

        adapter.index(&docs).unwrap();

        // 查询 "myocardial"（原文完全没有出现这个词）
        let hits = adapter.search("myocardial", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "d1");
        assert!(hits[0].1 > 0.0, "通过离线神经编译，文档成功被同义词命中！");
    }
}
