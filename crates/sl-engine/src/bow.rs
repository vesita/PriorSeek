//! 词袋基线打分（外壳里的「退化端点」），支持**带权查询词**。
//!
//! 带权查询是「先验因子」的消费方式：内核产出 `{(token, confidence)}`，外壳把它和原词一起
//! 送进打分，每个查询词的贡献乘以自己的权重。权重 1.0 的路径与既有 `bm25` 逐位一致。
//!
//! 它同时是两条门的载体：
//!
//! 1. **等价性门 G1**：权重全 1、查询词去重后，与既有 `bm25` 适配器逐查询一致；
//! 2. 消融对照：同一套索引与 IDF，只看出权重带来的差异。

use crate::index::PositionalIndex;
use std::sync::Arc;

/// BM25 的两个参数（与既有 `bm25` 适配器同值，改动需同步）。
pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;

/// 带权查询词：`(token id, 权重)`。原词权重 1.0，扩展词权重 = 置信度。
pub type WeightedTerm = (u32, f64);

#[derive(Debug)]
pub struct BowScorer {
    index: Arc<PositionalIndex>,
}

impl BowScorer {
    pub fn new(index: Arc<PositionalIndex>) -> Self {
        Self { index }
    }

    pub fn index(&self) -> &PositionalIndex {
        &self.index
    }

    /// BM25 词袋分：`Σ w_q(t) · idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·len/avg_len))`。
    ///
    /// 同一个 token 重复出现时取**最大权重**（与"查询词去重"的语义一致）。
    pub fn score_doc(&self, query: &[WeightedTerm], doc: u32) -> f64 {
        let avg_len = self.index.avg_len();
        let len = self.index.doc_len(doc) as f64;
        let mut seen: Vec<(u32, f64)> = Vec::with_capacity(query.len());
        for &(term, weight) in query {
            match seen.iter_mut().find(|(t, _)| *t == term) {
                Some(slot) => {
                    if weight > slot.1 {
                        slot.1 = weight;
                    }
                }
                None => seen.push((term, weight)),
            }
        }

        let mut score = 0.0;
        for (term, weight) in seen {
            if weight == 0.0 {
                continue;
            }
            let tf = self.index.positions_in(term, doc).len() as f64;
            if tf <= 0.0 {
                continue;
            }
            let idf = self.index.idf(term);
            if idf <= 0.0 {
                continue;
            }
            let denom = tf + K1 * (1.0 - B + B * len / avg_len);
            score += weight * idf * tf * (K1 + 1.0) / denom;
        }
        score
    }

    pub fn search(&self, query: &[WeightedTerm], k: usize) -> Vec<(String, f64)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        let terms: Vec<u32> = query.iter().map(|(term, _)| *term).collect();
        let mut scored: Vec<(u32, f64)> = self
            .index
            .candidates(&terms)
            .into_iter()
            .map(|doc| (doc, self.score_doc(query, doc)))
            .filter(|(_, score)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(doc, score)| (self.index.doc_id(doc).to_string(), score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(terms: &[u32]) -> Vec<WeightedTerm> {
        terms.iter().map(|t| (*t, 1.0)).collect()
    }

    #[test]
    fn score_matches_hand_computation() {
        let index = PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 1, 2]),
            ("d2".to_string(), vec![2, 3]),
            ("d3".to_string(), vec![3]),
        ]);
        let scorer = BowScorer::new(Arc::new(index));
        let idf = (1.0f64 + (3.0 - 1.0 + 0.5) / (1.0 + 0.5)).ln();
        let tf = 2.0f64;
        let denom = tf + K1 * (1.0 - B + B * 3.0 / 2.0);
        let expected = idf * tf * (K1 + 1.0) / denom;
        assert!((scorer.score_doc(&unit(&[1]), 0) - expected).abs() < 1e-12);
        assert_eq!(scorer.score_doc(&unit(&[99]), 0), 0.0);
    }

    #[test]
    fn query_weight_scales_the_term_contribution() {
        let index = PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 2]),
            ("d2".to_string(), vec![1]),
        ]);
        let scorer = BowScorer::new(Arc::new(index));
        let full = scorer.score_doc(&[(1, 1.0)], 0);
        let half = scorer.score_doc(&[(1, 0.5)], 0);
        assert!((half - full * 0.5).abs() < 1e-12, "权重应线性缩放该项贡献：{half} vs {full}");
        assert_eq!(scorer.score_doc(&[(1, 0.0)], 0), 0.0, "权重 0 等价于没这个词");
    }

    #[test]
    fn duplicate_terms_take_max_weight() {
        let index = PositionalIndex::build(vec![("d1".to_string(), vec![1])]);
        let scorer = BowScorer::new(Arc::new(index));
        let a = scorer.score_doc(&[(1, 0.3), (1, 0.9)], 0);
        let b = scorer.score_doc(&[(1, 0.9)], 0);
        assert!((a - b).abs() < 1e-12, "重复词项应取最大权重而不是相加");
    }

    #[test]
    fn search_orders_by_score_then_doc_id() {
        let index = PositionalIndex::build(vec![
            ("a".to_string(), vec![1]),
            ("b".to_string(), vec![1]),
        ]);
        let scorer = BowScorer::new(Arc::new(index));
        let hits = scorer.search(&unit(&[1]), 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "a");
        assert_eq!(hits[1].0, "b");
    }
}
