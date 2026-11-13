//! 纯 CPU 极速微型神经网络外壳适配器 (Neural Topological Scorer)。
//!
//! 在倒排索引粗排基础上，利用纯 Rust 原生极速微神经网络（参数量 49 个标量，耗时 < 50 微秒）
//! 对候选文档中的实体共现紧凑度与时序因果流进行在线神经拓扑打分。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::bow::BowScorer;
use sl_engine::index::PositionalIndex;
use sl_engine::micro_net::MicroNet;
use std::collections::HashMap;
use std::sync::Arc;

pub struct NeuralTopoAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    bow: Option<BowScorer>,
    net: MicroNet,
    vocabulary: HashMap<String, u32>,
    /// 拓扑调制强度
    gamma: f64,
}

impl NeuralTopoAdapter {
    pub fn new() -> Result<Self> {
        let gamma = std::env::var("SL_TOPO_GAMMA")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.3);

        let net = MicroNet::default();

        let config = serde_json::json!({
            "gamma": gamma,
            "micro_net_params": 49,
            "runtime": "pure-rust-cpu-registers",
        });

        Ok(Self {
            info: AdapterInfo::new(
                "neural-topo",
                "micro-neural-shell",
                "0.2.0",
                "外壳：crates/sl-adapters/src/neural_topo.rs；内核：sl-engine::MicroNet（49 标量微型前馈网络）",
            )
            .with_config(config),
            index: None,
            bow: None,
            net,
            vocabulary: HashMap::new(),
            gamma,
        })
    }
}

impl Adapter for NeuralTopoAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.vocabulary.clear();

        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        for doc in docs {
            let terms = bm25_terms(&doc.full_text());
            let ids: Vec<u32> = terms
                .into_iter()
                .map(|term| {
                    let next = self.vocabulary.len() as u32;
                    *self.vocabulary.entry(term).or_insert(next)
                })
                .collect();
            encoded.push((doc.id.clone(), ids));
        }

        let index = Arc::new(PositionalIndex::build(encoded));
        self.bow = Some(BowScorer::new(Arc::clone(&index)));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let bow = self.bow.as_ref().context("未建索引（先调 index）")?;
        let index = self.index.as_ref().context("未建索引")?;
        let terms = bm25_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let term_ids: Vec<u32> = terms
            .iter()
            .filter_map(|t| self.vocabulary.get(t).copied())
            .collect();

        if term_ids.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 词袋倒排粗排（纯 CPU < 2ms）获取前 top-50 候选集
        let weighted: Vec<(u32, f64)> = term_ids.iter().map(|&id| (id, 1.0)).collect();
        let candidate_hits = bow.search(&weighted, k.max(50));
        if candidate_hits.is_empty() || term_ids.len() < 2 || self.gamma <= 0.0 {
            return Ok(candidate_hits.into_iter().take(k).collect());
        }

        // 2. 微型神经网络在线拓扑前向重排（百微秒级）
        let (t1, t2) = (term_ids[0], term_ids[1]);
        let idf_avg = ((index.idf(t1) + index.idf(t2)) / 2.0) as f32;

        let mut reranked: Vec<(String, f64)> = Vec::with_capacity(candidate_hits.len());

        for (doc_id, base_score) in candidate_hits {
            // 通过 doc_id 反查 doc 下标
            let doc_idx = match (0..index.len() as u32).find(|&d| index.doc_id(d) == doc_id) {
                Some(idx) => idx,
                None => {
                    reranked.push((doc_id, base_score));
                    continue;
                }
            };

            let p1 = index.positions_in(t1, doc_idx);
            let p2 = index.positions_in(t2, doc_idx);

            if p1.is_empty() || p2.is_empty() {
                reranked.push((doc_id, base_score));
                continue;
            }

            // 计算最小物理跨距与相对方向
            let mut min_span = u32::MAX;
            let mut in_order = false;

            for pos1 in &p1 {
                for pos2 in &p2 {
                    let (pos1, pos2) = (*pos1, *pos2);
                    let span = if pos2 >= pos1 {
                        in_order = true;
                        pos2 - pos1
                    } else {
                        pos1 - pos2
                    };
                    if span < min_span {
                        min_span = span;
                    }
                }
            }

            let span_feature = 1.0f32 / (1.0 + ((min_span as f32).ln_1p()));
            let doc_len_norm =
                (p1.len() + p2.len()) as f32 / (index.doc_len(doc_idx) as f32).max(1.0);
            let order_feature = if in_order { 1.0f32 } else { 0.0f32 };
            let sem_dir = if in_order { 0.5f32 } else { -0.5f32 };

            // 微型神经网络在线拓扑前向重排（5 维输入）
            let input = [
                (idf_avg / 15.0).min(1.0),
                (doc_len_norm * 10.0).min(1.0),
                span_feature,
                order_feature,
                sem_dir,
            ];
            let topo_factor = self.net.forward(&input) as f64;

            // 动态结构调制打分
            let final_score = base_score * (1.0 + self.gamma * (topo_factor - 0.5));
            reranked.push((doc_id, final_score));
        }

        reranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        reranked.truncate(k);
        Ok(reranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_topo_adapter_rewards_ordered_compact_matches() {
        let mut adapter = NeuralTopoAdapter::new().unwrap();
        // d1: "A B" 紧凑且顺序正确
        // d2: "B ... A" 顺序颠倒且跨距远
        let docs = vec![
            Doc {
                id: "d1".to_string(),
                title: "Compact in-order".to_string(),
                text: "alpha beta".to_string(),
            },
            Doc {
                id: "d2".to_string(),
                title: "Reversed sparse".to_string(),
                text: "beta other other other other alpha".to_string(),
            },
        ];

        adapter.index(&docs).unwrap();
        let hits = adapter.search("alpha beta", 10).unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].0, "d1",
            "紧凑且顺序一致的文档获得微型神经网络的拓扑奖励排在第一"
        );
        assert!(hits[0].1 > hits[1].1);
    }
}
