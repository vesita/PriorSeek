//! 双统计单元外壳（`dual-unit`）：长文档按段取最大，短文档按整篇求和。
//!
//! 依据：
//! - §43 实测：段落级 Score-Max 在 arguana（长论证）上 +0.0017、在 scifact（短摘要）上 −0.0343；
//! - 文献里 Score-Max / 分段聚合本来就是为长文档设计的（LSR 长文档复现研究）；
//! - 触发条件必须是**可观测的结构量**（C2）：这里用"文档长度 > 1.5 × 语料平均长度"，
//!   阈值来自语料统计，不是调出来的魔数。
//!
//! 打分公式与 BM25 完全相同；区别只在"用哪个统计单元算分"。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::{BowScorer, PositionalIndex, WeightedTerm};
use std::collections::HashMap;
use std::sync::Arc;

pub struct DualUnitAdapter {
    info: AdapterInfo,
    doc_index: Option<Arc<PositionalIndex>>,
    doc_scorer: Option<BowScorer>,
    passage_index: Option<Arc<PositionalIndex>>,
    passage_scorer: Option<BowScorer>,
    passage_doc: HashMap<String, String>,
    vocab: HashMap<String, u32>,
    /// 段落窗口（token 数）。
    window: usize,
    /// 判定"长文档"的阈值：`ratio × 语料平均长度`。
    long_doc_ratio: f64,
    /// 语料平均长度（建索引时算出）。
    avg_len: f64,
}

impl DualUnitAdapter {
    pub fn new() -> Result<Self> {
        let window: usize = std::env::var("SL_PASSAGE_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(60);
        let long_doc_ratio: f64 = std::env::var("SL_LONG_DOC_RATIO")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(1.5);
        let config = serde_json::json!({
            "algorithm": "dual-unit（短文档整篇 BM25 / 长文档段落 Score-Max）",
            "window": window,
            "long_doc_ratio": long_doc_ratio,
            "trigger": "文档长度 > ratio × 语料平均长度（语料统计，非拟合）",
            "prior_factor": null,
        });
        Ok(Self {
            info: AdapterInfo::new(
                "dual-unit",
                "dual-statistical-unit",
                "0.1.0",
                "外壳：crates/sl-adapters/src/dual_unit.rs（统计单元由文档长度触发）",
            )
            .with_config(config),
            doc_index: None,
            doc_scorer: None,
            passage_index: None,
            passage_scorer: None,
            passage_doc: HashMap::new(),
            vocab: HashMap::new(),
            window,
            long_doc_ratio,
            avg_len: 0.0,
        })
    }
}

impl Adapter for DualUnitAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut doc_level: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        let mut passage_level: Vec<(String, Vec<u32>)> = Vec::new();
        self.passage_doc.clear();
        let mut total_tokens = 0usize;

        for doc in docs {
            let tokens: Vec<u32> = bm25_terms(&doc.full_text())
                .into_iter()
                .map(|term| {
                    let next = vocab.len() as u32;
                    *vocab.entry(term).or_insert(next)
                })
                .collect();
            total_tokens += tokens.len();
            doc_level.push((doc.id.clone(), tokens.clone()));
            if tokens.is_empty() {
                continue;
            }
            let stride = (self.window / 2).max(1);
            let mut start = 0usize;
            let mut ordinal = 0usize;
            loop {
                let end = (start + self.window).min(tokens.len());
                let passage_id = format!("{}#{}", doc.id, ordinal);
                self.passage_doc.insert(passage_id.clone(), doc.id.clone());
                passage_level.push((passage_id, tokens[start..end].to_vec()));
                if end >= tokens.len() {
                    break;
                }
                start += stride;
                ordinal += 1;
            }
        }

        self.avg_len = total_tokens as f64 / docs.len().max(1) as f64;
        self.vocab = vocab;
        let doc_index = Arc::new(PositionalIndex::build(doc_level));
        let passage_index = Arc::new(PositionalIndex::build(passage_level));
        self.doc_scorer = Some(BowScorer::with_delta(Arc::clone(&doc_index), 0.0)?);
        self.passage_scorer = Some(BowScorer::with_delta(Arc::clone(&passage_index), 0.0)?);
        self.doc_index = Some(doc_index);
        self.passage_index = Some(passage_index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let doc_scorer = self
            .doc_scorer
            .as_ref()
            .context("dual-unit 未建索引（先调 index）")?;
        let passage_scorer = self
            .passage_scorer
            .as_ref()
            .context("dual-unit 未建索引（先调 index）")?;
        let doc_index = self.doc_index.as_ref().context("dual-unit 缺文档索引")?;

        let mut seen = std::collections::HashSet::new();
        let terms: Vec<WeightedTerm> = bm25_terms(query)
            .into_iter()
            .filter(|term| seen.insert(term.clone()))
            .filter_map(|term| self.vocab.get(&term).map(|id| (*id, 1.0)))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let pool = (k * 8).max(50);
        let doc_hits = doc_scorer.search(&terms, pool);
        // 段落臂只对"长文档"生效，所以这里必须多取候补，否则长文档可能进不了池。
        let passage_hits = passage_scorer.search(&terms, pool * 4);
        let mut passage_best: HashMap<String, f64> = HashMap::new();
        for (passage_id, score) in passage_hits {
            let Some(doc_id) = self.passage_doc.get(&passage_id).cloned() else {
                continue;
            };
            let slot = passage_best.entry(doc_id).or_insert(f64::NEG_INFINITY);
            if score > *slot {
                *slot = score;
            }
        }

        let threshold = self.long_doc_ratio * self.avg_len;
        let mut merged: HashMap<String, f64> = HashMap::new();
        for (doc_id, score) in doc_hits {
            merged.insert(doc_id, score);
        }
        // 只在"长文档"上让段落臂参与；短文档保持整篇求和（§43 的负结果）。
        for (doc_id, score) in passage_best {
            let doc_pos = (0..doc_index.len()).find(|d| doc_index.doc_id(*d as u32) == doc_id);
            let Some(doc_pos) = doc_pos else { continue };
            let len = doc_index.doc_len(doc_pos as u32) as f64;
            if len <= threshold {
                continue;
            }
            let slot = merged.entry(doc_id).or_insert(f64::NEG_INFINITY);
            if score > *slot {
                *slot = score;
            }
        }

        let mut results: Vec<(String, f64)> = merged
            .into_iter()
            .filter(|(_, score)| score.is_finite())
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        Ok(results)
    }
}
