//! 段落级证据外壳（`passage`）：文档的分数 = 它**最好的一段**的 BM25 分数。
//!
//! 动机（来自 §12 / §38 的负结果）：把整篇文档当一个词袋求和，会让长文档里的
//! 一段强证据被全文长度摊薄；而"有序对齐"这类几何因子又已被证伪。
//! 段落最大值是另一种解释："一篇文档至少有一段真的在讲查询这件事"，
//! 它不需要任何拟合系数，也不改变 BM25 的打分公式，只改变**统计单元**。
//!
//! 结构参数（窗口 `W`、步长 `S`）是切分规则，不是权重；默认 60/30，
//! 可用 `SL_PASSAGE_WINDOW` / `SL_PASSAGE_STRIDE` 改，用于消融。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::{BowScorer, PositionalIndex};
use std::collections::HashMap;
use std::sync::Arc;

pub struct PassageAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    scorer: Option<BowScorer>,
    /// 段落 id → 文档 id
    passage_doc: HashMap<String, String>,
    /// 词表：token → id（查询期编码用，与建索引同一份映射）
    vocab: HashMap<String, u32>,
    window: usize,
    stride: usize,
}

impl PassageAdapter {
    pub fn new() -> Result<Self> {
        let window: usize = std::env::var("SL_PASSAGE_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(60);
        let stride: usize = std::env::var("SL_PASSAGE_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(30);
        let config = serde_json::json!({
            "algorithm": "passage-max-bm25",
            "window": window,
            "stride": stride,
            "prior_factor": null,
        });
        Ok(Self {
            info: AdapterInfo::new(
                "passage",
                "passage-max",
                "0.1.0",
                "外壳：crates/sl-adapters/src/passage.rs（段落级统计单元 + 每文档取最大段分）",
            )
            .with_config(config),
            index: None,
            scorer: None,
            passage_doc: HashMap::new(),
            vocab: HashMap::new(),
            window,
            stride,
        })
    }
}

impl Adapter for PassageAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut passages: Vec<(String, Vec<u32>)> = Vec::new();
        self.passage_doc.clear();

        for doc in docs {
            let tokens: Vec<u32> = bm25_terms(&doc.full_text())
                .into_iter()
                .map(|term| {
                    let next = vocab.len() as u32;
                    *vocab.entry(term).or_insert(next)
                })
                .collect();
            if tokens.is_empty() {
                continue;
            }
            let mut start = 0usize;
            let mut ordinal = 0usize;
            loop {
                let end = (start + self.window).min(tokens.len());
                let piece = tokens[start..end].to_vec();
                let passage_id = format!("{}#{}", doc.id, ordinal);
                self.passage_doc
                    .insert(passage_id.clone(), doc.id.clone());
                passages.push((passage_id, piece));
                if end >= tokens.len() {
                    break;
                }
                start += self.stride;
                ordinal += 1;
            }
        }

        self.vocab = vocab;
        let index = Arc::new(PositionalIndex::build(passages));
        let scorer = BowScorer::with_delta(Arc::clone(&index), 0.0)?;
        self.index = Some(index);
        self.scorer = Some(scorer);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let scorer = self.scorer.as_ref().context("passage 未建索引（先调 index）")?;
        let mut seen = std::collections::HashSet::new();
        let terms: Vec<sl_engine::WeightedTerm> = bm25_terms(query)
            .into_iter()
            .filter(|term| seen.insert(term.clone()))
            .filter_map(|term| self.vocab.get(&term).map(|id| (*id, 1.0)))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        // 先多取段落候选，再按文档取最大段分。
        let pool = (k * 8).max(50);
        let hits = scorer.search(&terms, pool);
        let mut best: HashMap<String, f64> = HashMap::new();
        for (passage_id, score) in hits {
            let Some(doc_id) = self.passage_doc.get(&passage_id).cloned() else {
                continue;
            };
            let slot = best.entry(doc_id).or_insert(f64::NEG_INFINITY);
            if score > *slot {
                *slot = score;
            }
        }
        let mut docs: Vec<(String, f64)> = best.into_iter().collect();
        docs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        docs.truncate(k);
        Ok(docs)
    }
}
