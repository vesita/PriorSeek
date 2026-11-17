//! 伪相关反馈（PRF / Rocchio）外壳：首轮 BM25 取 top-k，用命中文档的内核词权
//! 构造查询侧扩展，再打第二轮。
//!
//! 与既有扩展方向的区别（`docs/RESULTS.md` §13 / §23）：
//! - §13 的扩展词来自**语料共现**（PPMI），与本次查询无关；
//! - §23 用首轮置信度做**手工阈值门**，扩展词来源仍是 PPMI；
//! - 本适配器的扩展词来自**本次查询实际命中的文档**，天然是查询条件化的——
//!   这正是 §13.3 里 oracle 上限（+2.36pp）指向的缺口。
//!
//! 参数（`top_docs` / `exp_terms` / `exp_weight`）先按文献常用值固定，
//! 不做 test 集调参；要调就调 train split 并预注册。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::{BowScorer, PositionalIndex, WeightedTerm};
use std::collections::HashMap;
use std::sync::Arc;

pub struct PrfAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    scorer: Option<BowScorer>,
    /// 文档 token 序列（与索引同序），PRF 需要用它们统计扩展词。
    doc_tokens: Vec<Vec<u32>>,
    /// 词表：token 字符串 → id（与索引共用同一份映射）。
    vocab: HashMap<String, u32>,
    top_docs: usize,
    exp_terms: usize,
    exp_weight: f64,
}

impl PrfAdapter {
    pub fn new() -> Result<Self> {
        let top_docs: usize = std::env::var("SL_PRF_TOP_DOCS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let exp_terms: usize = std::env::var("SL_PRF_TERMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let exp_weight: f64 = std::env::var("SL_PRF_WEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.4);

        let config = serde_json::json!({
            "algorithm": "prf-rocchio",
            "top_docs": top_docs,
            "exp_terms": exp_terms,
            "exp_weight": exp_weight,
            "prior_factor": null,
        });

        Ok(Self {
            info: AdapterInfo::new(
                "prf",
                "prf-rocchio",
                "0.1.0",
                "外壳：crates/sl-adapters/src/prf.rs（BM25 首轮 + 命中文档词权扩展 + 二轮）",
            )
            .with_config(config),
            index: None,
            scorer: None,
            doc_tokens: Vec::new(),
            vocab: HashMap::new(),
            top_docs,
            exp_terms,
            exp_weight,
        })
    }

    fn id_of(&mut self, term: &str) -> u32 {
        if let Some(&id) = self.vocab.get(term) {
            return id;
        }
        let next = self.vocab.len() as u32;
        self.vocab.insert(term.to_string(), next);
        next
    }

    /// 首轮命中文档 → 扩展词（按 BM25 分加权求和，取 top `exp_terms`）。
    fn expand(&mut self, query: &str, base_hits: &[(String, f64)]) -> Vec<WeightedTerm> {
        let Some(index) = self.index.as_ref() else {
            return Vec::new();
        };
        let mut weights: HashMap<u32, f64> = HashMap::new();
        for (doc_id, score) in base_hits.iter().take(self.top_docs) {
            let Some(doc_pos) = (0..index.len()).find(|d| index.doc_id(*d as u32) == doc_id) else {
                continue;
            };
            let tokens = &self.doc_tokens[doc_pos];
            if tokens.is_empty() {
                continue;
            }
            for &term in tokens {
                *weights.entry(term).or_insert(0.0) += score.max(0.0);
            }
        }
        // 原查询词不重复计入。
        let mut original: Vec<u32> = bm25_terms(query)
            .into_iter()
            .filter_map(|term| self.vocab.get(&term).copied())
            .collect();
        original.sort_unstable();
        original.dedup();

        let mut candidates: Vec<(u32, f64)> = weights
            .into_iter()
            .filter(|(term, _)| !original.contains(term))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(self.exp_terms);

        let max = candidates
            .first()
            .map(|(_, weight)| *weight)
            .filter(|weight| *weight > 0.0)
            .unwrap_or(1.0);
        candidates
            .into_iter()
            .map(|(term, weight)| (term, self.exp_weight * weight / max))
            .collect()
    }
}

impl Adapter for PrfAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for doc in docs {
            let ids = bm25_terms(&doc.full_text())
                .into_iter()
                .map(|term| {
                    let next = vocab.len() as u32;
                    *vocab.entry(term).or_insert(next)
                })
                .collect();
            encoded.push((doc.id.clone(), ids));
        }
        let doc_tokens: Vec<Vec<u32>> = encoded.iter().map(|(_, ids)| ids.clone()).collect();
        let index = Arc::new(PositionalIndex::build(encoded));
        let scorer = BowScorer::with_delta(Arc::clone(&index), 0.0)?;
        self.index = Some(index);
        self.scorer = Some(scorer);
        self.doc_tokens = doc_tokens;
        self.vocab = vocab;
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let base_terms: Vec<WeightedTerm> = {
            let mut seen = std::collections::HashSet::new();
            bm25_terms(query)
                .into_iter()
                .filter(|term| seen.insert(term.clone()))
                .map(|term| {
                    let next = self.vocab.len() as u32;
                    let id = *self.vocab.entry(term).or_insert(next);
                    (id, 1.0)
                })
                .collect()
        };
        let scorer = self.scorer.as_ref().context("prf 未建索引（先调 index）")?;
        let base_hits = scorer.search(&base_terms, self.top_docs.max(k));

        let expansion = self.expand(query, &base_hits);
        if expansion.is_empty() {
            return Ok(base_hits.into_iter().take(k).collect());
        }
        let mut combined = base_terms;
        combined.extend(expansion);
        let scorer = self.scorer.as_ref().context("prf 未建索引（先调 index）")?;
        Ok(scorer.search(&combined, k))
    }
}
