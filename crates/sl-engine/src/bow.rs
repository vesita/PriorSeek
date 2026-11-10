//! 词袋基线打分（外壳里的「退化端点」，用于等价性验证与消融对照）。
//!
//! 它不是"另一个算法"，而是 OAC 在 `ordered = false`、`K ≡ 1`、且按 BM25 方式做
//! 词频饱和与长度归一时的**退化形态**。存在的目的是两条：
//!
//! 1. **等价性门 G1**：外壳用这份代码算出的分数必须与既有 `bm25` 适配器逐查询一致
//!    （证明外壳是忠实的，不是另写了一个"差不多的"）；
//! 2. 消融对照：同一套索引、同一份 IDF，只把"有序/跨度"关掉，看差多少。

use crate::index::PositionalIndex;
use std::sync::Arc;

/// BM25 的两个参数（与既有 `bm25` 适配器同值，改动需同步）。
pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;

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

    /// BM25 词袋分：`Σ idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·len/avg_len))`。
    pub fn score_doc(&self, query: &[u32], doc: u32) -> f64 {
        let avg_len = self.index.avg_len();
        let len = self.index.doc_len(doc) as f64;
        // 查询词去重（与既有 `bm25` 适配器一致：那边用 BTreeSet 收集查询词项）。
        let mut seen: Vec<u32> = Vec::with_capacity(query.len());
        let mut score = 0.0;
        for term in query {
            if seen.contains(term) {
                continue;
            }
            seen.push(*term);
            let tf = self.index.positions_in(*term, doc).len() as f64;
            if tf <= 0.0 {
                continue;
            }
            let idf = self.index.idf(*term);
            if idf <= 0.0 {
                continue;
            }
            let denom = tf + K1 * (1.0 - B + B * len / avg_len);
            score += idf * tf * (K1 + 1.0) / denom;
        }
        score
    }

    pub fn search(&self, query: &[u32], k: usize) -> Vec<(String, f64)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(u32, f64)> = self
            .index
            .candidates(query)
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

    #[test]
    fn score_matches_hand_computation() {
        let index = PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 1, 2]),
            ("d2".to_string(), vec![2, 3]),
            ("d3".to_string(), vec![3]),
        ]);
        let scorer = BowScorer::new(Arc::new(index));
        // d1: tf(1)=2, len=3, avg=2.5；idf(1) = ln(1 + (3-1+0.5)/(1+0.5)) = ln(2.6667)
        let idf = (1.0f64 + (3.0 - 1.0 + 0.5) / (1.0 + 0.5)).ln();
        let tf = 2.0f64;
        let denom = tf + K1 * (1.0 - B + B * 3.0 / 2.0);
        let expected = idf * tf * (K1 + 1.0) / denom;
        assert!((scorer.score_doc(&[1], 0) - expected).abs() < 1e-12);
        // 未命中词项 ⇒ 0
        assert_eq!(scorer.score_doc(&[99], 0), 0.0);
    }

    #[test]
    fn search_orders_by_score_then_doc_id() {
        let index = PositionalIndex::build(vec![
            ("a".to_string(), vec![1]),
            ("b".to_string(), vec![1]),
        ]);
        let scorer = BowScorer::new(Arc::new(index));
        let hits = scorer.search(&[1], 10);
        assert_eq!(hits.len(), 2);
        // 同分（同长度、同 tf、同 idf）⇒ 按 doc id 升序
        assert_eq!(hits[0].0, "a");
        assert_eq!(hits[1].0, "b");
    }
}
