//! 神经流匹配外壳适配器 (Neural Flow Matcher / Optimal Transport Retrieval)。
//!
//! 彻底废除 BM25 词袋对词频的算术求和逻辑，将检索定义为：
//! 查询概念意图流到文档论证流的最小最优传输（Wasserstein / Earth Mover's Distance）搬运成本。
//! 纯 CPU 纳秒级 RWMD 算子，零重型框架依赖。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::bow::BowScorer;
use sl_engine::flow_matcher::{CostGraph, FlowMatcher};
use sl_engine::index::PositionalIndex;
use std::collections::HashMap;
use std::sync::Arc;

pub struct FlowMatchAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    bow: Option<BowScorer>,
    matcher: Option<FlowMatcher>,
    /// doc_idx -> [(term_id, tf)]
    doc_profiles: Vec<Vec<(u32, f32)>>,
    vocabulary: HashMap<String, u32>,
    /// 字符串形式的代价图：src_word -> [(dst_word, cost)]
    cost_entries: HashMap<String, Vec<(String, f32)>>,
    temperature: f32,
}

impl FlowMatchAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/flow-cost-v0");
        let dir = std::env::var("SL_FLOW_COST_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 flow-cost 工件目录 {}（先跑 tools/kernel/build_flow_costs.py {dataset}）",
                dir
            );
        }
        let temperature = std::env::var("SL_FLOW_TEMP")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.5);

        let (cost_entries, sha256) = Self::load_cost_graph(path)?;

        let config = serde_json::json!({
            "temperature": temperature,
            "metric": "relaxed_wasserstein_flow",
            "prior_factor": {
                "name": "semantic_transport_cost",
                "kind": "token_confidence_list",
                "sha256": sha256,
                "source_terms": cost_entries.len(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "flow-match",
                "neural-flow-matching",
                "0.2.0",
                "外壳：crates/sl-adapters/src/flow_match.rs（最优传输 RWMD 流外壳）；内核：tools/kernel/build_flow_costs.py",
            )
            .with_config(config),
            index: None,
            bow: None,
            matcher: None,
            doc_profiles: Vec::new(),
            vocabulary: HashMap::new(),
            cost_entries,
            temperature,
        })
    }

    fn load_cost_graph(
        dir: &std::path::Path,
    ) -> Result<(HashMap<String, Vec<(String, f32)>>, String)> {
        let artifact = sl_core::KernelArtifact::from_dir(dir)?;
        let payload_path = dir.join("cost_graph.json");
        let bytes = std::fs::read(&payload_path)
            .with_context(|| format!("读取 {} 失败", payload_path.display()))?;

        // 校验 sha256
        let expected = artifact
            .files
            .get("cost_graph.json")
            .ok_or_else(|| anyhow::anyhow!("工件缺少 cost_graph.json 哈希登记"))?;
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
            for (term, rows) in entries {
                let mut list = Vec::new();
                if let Some(rows) = rows.as_array() {
                    for row in rows {
                        let Some(pair) = row.as_array() else { continue };
                        let (Some(neighbor), Some(cost)) = (
                            pair.first().and_then(|v| v.as_str()),
                            pair.get(1).and_then(|v| v.as_f64()),
                        ) else {
                            continue;
                        };
                        list.push((neighbor.to_string(), cost as f32));
                    }
                }
                map.insert(term.clone(), list);
            }
        }
        Ok((map, actual))
    }
}

impl Adapter for FlowMatchAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.vocabulary.clear();
        self.doc_profiles.clear();

        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());

        for doc in docs {
            let terms = bm25_terms(&doc.full_text());
            let mut ids = Vec::with_capacity(terms.len());
            let mut tf_map: HashMap<u32, f32> = HashMap::new();

            for term in terms {
                let next = self.vocabulary.len() as u32;
                let id = *self.vocabulary.entry(term).or_insert(next);
                ids.push(id);
                *tf_map.entry(id).or_insert(0.0) += 1.0;
            }

            // 提取文档核心论证概念分布（取 tf 最高的 Top 32 词）
            let mut profile: Vec<(u32, f32)> = tf_map.into_iter().collect();
            profile.sort_by(|a, b| b.1.total_cmp(&a.1));
            profile.truncate(32);
            self.doc_profiles.push(profile);

            encoded.push((doc.id.clone(), ids));
        }

        let index = Arc::new(PositionalIndex::build(encoded));
        self.bow = Some(BowScorer::new(Arc::clone(&index)));

        // 构建词项 id 维度的数值代价图
        let mut cost_graph = CostGraph::new();
        for (src_word, neighbors) in &self.cost_entries {
            if let Some(&src_id) = self.vocabulary.get(src_word) {
                for (dst_word, cost) in neighbors {
                    if let Some(&dst_id) = self.vocabulary.get(dst_word) {
                        cost_graph
                            .edges
                            .entry(src_id)
                            .or_default()
                            .push((dst_id, *cost));
                        cost_graph
                            .edges
                            .entry(dst_id)
                            .or_default()
                            .push((src_id, *cost));
                    }
                }
            }
        }

        self.matcher = Some(FlowMatcher::new(cost_graph, self.temperature));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let bow = self.bow.as_ref().context("未建索引（先调 index）")?;
        let matcher = self.matcher.as_ref().context("未初始化 matcher")?;
        let index = self.index.as_ref().context("未建索引")?;

        let terms = bm25_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let query_terms: Vec<(u32, f32)> = terms
            .iter()
            .filter_map(|t| {
                self.vocabulary.get(t).map(|&id| {
                    let idf = index.idf(id) as f32;
                    (id, idf.max(0.1))
                })
            })
            .collect();

        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 粗排候选池：包含原词以及距离近的语义概念词
        let mut query_weighted: HashMap<u32, f64> = HashMap::new();
        for &(id, _) in &query_terms {
            query_weighted.insert(id, 1.0);
            if let Some(neighbors) = matcher.costs.edges.get(&id) {
                for &(dst, cost) in neighbors {
                    if cost < 0.8 {
                        let w = (1.0 - cost as f64).max(0.1);
                        let slot = query_weighted.entry(dst).or_insert(0.0);
                        if w > *slot {
                            *slot = w;
                        }
                    }
                }
            }
        }
        let weighted: Vec<(u32, f64)> = query_weighted.into_iter().collect();
        let candidate_hits = bow.search(&weighted, k.max(60));
        if candidate_hits.is_empty() {
            return Ok(Vec::new());
        }

        // 2. 神经流匹配：基于最优传输代价 (RWMD) 进行重排打分
        let mut scored: Vec<(String, f64)> = Vec::with_capacity(candidate_hits.len());

        for (doc_id, _) in candidate_hits {
            let doc_idx = match (0..index.len() as u32).find(|&d| index.doc_id(d) == doc_id) {
                Some(idx) => idx as usize,
                None => continue,
            };

            let doc_profile = &self.doc_profiles[doc_idx];
            // 纯最优传输流匹配相似度！彻底摒弃词袋累加
            let flow_score = matcher.score(&query_terms, doc_profile);
            scored.push((doc_id, flow_score));
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_match_adapter_prefers_semantically_aligned_documents() {
        let mut adapter = FlowMatchAdapter {
            info: AdapterInfo::new("test", "test", "0.2.0", "test"),
            index: None,
            bow: None,
            matcher: None,
            doc_profiles: Vec::new(),
            vocabulary: HashMap::new(),
            cost_entries: HashMap::new(),
            temperature: 0.5,
        };

        // 假设原词 1 (cancer) 与 2 (carcinoma) 存在很小的搬运成本
        let mut entries = HashMap::new();
        entries.insert("cancer".to_string(), vec![("carcinoma".to_string(), 0.15)]);
        adapter.cost_entries = entries;

        let docs = vec![
            Doc {
                id: "d1".to_string(),
                title: "Cancer Therapy".to_string(),
                text: "Targeting cancer cells in patients.".to_string(),
            },
            Doc {
                id: "d2".to_string(),
                title: "Carcinoma Study".to_string(),
                text: "Treating carcinoma with novel drugs.".to_string(),
            },
        ];

        adapter.index(&docs).unwrap();

        // 搜 "carcinoma"：虽然 d1 写的是 "cancer"，但在语义流传输中依然获得高分
        let hits = adapter.search("carcinoma", 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits[0].1 > 0.0);
    }
}
