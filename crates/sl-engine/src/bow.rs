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

#[derive(Debug, Clone)]
pub struct BowScorer {
    index: Arc<PositionalIndex>,
    delta: f64,
    /// 词项先验显著性因子 S(t) in [0, 1]（可选，None 则回退到基准参数）。
    saliency: Option<std::collections::HashMap<u32, f64>>,
    /// 动态饱和度调节强度 lambda (k1_eff = K1 * (1.0 + lambda * S(t)))。
    lambda_k1: f64,
}

impl BowScorer {
    pub fn new(index: Arc<PositionalIndex>) -> Self {
        Self {
            index,
            delta: 0.0,
            saliency: None,
            lambda_k1: 0.0,
        }
    }

    /// 构造带 BM25+ 词频下界偏置 `delta` 的打分器。
    ///
    /// 强制点：`delta` 必须是非负有限数；负数或 NaN 在构造期立即报错。
    pub fn with_delta(index: Arc<PositionalIndex>, delta: f64) -> anyhow::Result<Self> {
        if !delta.is_finite() || delta < 0.0 {
            anyhow::bail!("delta 必须为非负有限数，收到 {delta}");
        }
        Ok(Self {
            index,
            delta,
            saliency: None,
            lambda_k1: 0.0,
        })
    }

    /// 构造带先验显著性因子 S(t) 调制的打分器。
    pub fn with_saliency(
        index: Arc<PositionalIndex>,
        delta: f64,
        saliency: std::collections::HashMap<u32, f64>,
        lambda_k1: f64,
    ) -> anyhow::Result<Self> {
        if !delta.is_finite() || delta < 0.0 {
            anyhow::bail!("delta 必须为非负有限数，收到 {delta}");
        }
        if !lambda_k1.is_finite() || lambda_k1 < 0.0 {
            anyhow::bail!("lambda_k1 必须为非负有限数，收到 {lambda_k1}");
        }
        Ok(Self {
            index,
            delta,
            saliency: Some(saliency),
            lambda_k1,
        })
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn index(&self) -> &PositionalIndex {
        &self.index
    }

    /// BM25 词袋分（支持 BM25+ 下界偏置与先验 Saliency 调制）：
    /// `Σ w_q(t) · idf(t) · [ tf·(k1_eff+1) / (tf + k1_eff·(1−b+b·len/avg_len)) + delta_eff ]`。
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
            let (k1_eff, delta_eff) = match &self.saliency {
                Some(map) => {
                    let s = map.get(&term).copied().unwrap_or(0.0);
                    (K1 * (1.0 + self.lambda_k1 * s), self.delta * s)
                }
                None => (K1, self.delta),
            };
            let denom = tf + k1_eff * (1.0 - B + B * len / avg_len);
            let tf_comp = tf * (k1_eff + 1.0) / denom + delta_eff;
            score += weight * idf * tf_comp;
        }
        score
    }

    pub fn search(&self, query: &[WeightedTerm], k: usize) -> Vec<(String, f64)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        // 倒排累加：查询项每条 posting 只扫一次；筛选内核的回报评估会重复调用本方法，
        // 因此不能对每个候选文档再逐项反查 positions（那会退化成候选数 × 查询词数扫描）。
        let mut terms: Vec<WeightedTerm> = Vec::with_capacity(query.len());
        for &(term, weight) in query {
            match terms.iter_mut().find(|(t, _)| *t == term) {
                Some(slot) if weight > slot.1 => slot.1 = weight,
                Some(_) => {}
                None => terms.push((term, weight)),
            }
        }
        let avg_len = self.index.avg_len();
        let mut scores: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
        for (term, query_weight) in terms {
            if query_weight <= 0.0 {
                continue;
            }
            let idf = self.index.idf(term);
            if idf <= 0.0 {
                continue;
            }
            let (k1_eff, delta_eff) = match &self.saliency {
                Some(map) => {
                    let s = map.get(&term).copied().unwrap_or(0.0);
                    (K1 * (1.0 + self.lambda_k1 * s), self.delta * s)
                }
                None => (K1, self.delta),
            };
            let occurrences = self.index.occurrences(term);
            let mut cursor = 0usize;
            while cursor < occurrences.len() {
                let doc = occurrences[cursor].doc;
                let start = cursor;
                while cursor < occurrences.len() && occurrences[cursor].doc == doc {
                    cursor += 1;
                }
                let tf = (cursor - start) as f64;
                let len = self.index.doc_len(doc) as f64;
                let denom = tf + k1_eff * (1.0 - B + B * len / avg_len);
                let tf_comp = tf * (k1_eff + 1.0) / denom + delta_eff;
                *scores.entry(doc).or_insert(0.0) += query_weight * idf * tf_comp;
            }
        }
        let mut scored: Vec<(u32, f64)> = scores
            .into_iter()
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
    fn delta_zero_matches_traditional_bm25() {
        let index = Arc::new(PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 1, 2]),
            ("d2".to_string(), vec![2, 3]),
            ("d3".to_string(), vec![3]),
        ]));
        let default_scorer = BowScorer::new(Arc::clone(&index));
        let zero_scorer = BowScorer::with_delta(Arc::clone(&index), 0.0).unwrap();
        for doc in 0..3 {
            let s1 = default_scorer.score_doc(&unit(&[1, 2]), doc);
            let s2 = zero_scorer.score_doc(&unit(&[1, 2]), doc);
            assert_eq!(
                s1.to_bits(),
                s2.to_bits(),
                "delta=0 必须与默认 BM25 逐位一致"
            );
        }
    }

    #[test]
    fn delta_bonus_increases_score_monotonically() {
        let index = Arc::new(PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 2]),
            ("d2".to_string(), vec![3]),
        ]));
        let s_base = BowScorer::new(Arc::clone(&index));
        let s_plus = BowScorer::with_delta(Arc::clone(&index), 1.0).unwrap();
        // 命中文档得分单调增加
        assert!(s_plus.score_doc(&unit(&[1]), 0) > s_base.score_doc(&unit(&[1]), 0));
        // 未命中文档严格保持 0.0
        assert_eq!(s_plus.score_doc(&unit(&[1]), 1), 0.0);
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
        assert!(
            (half - full * 0.5).abs() < 1e-12,
            "权重应线性缩放该项贡献：{half} vs {full}"
        );
        assert_eq!(
            scorer.score_doc(&[(1, 0.0)], 0),
            0.0,
            "权重 0 等价于没这个词"
        );
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
        let index =
            PositionalIndex::build(vec![("a".to_string(), vec![1]), ("b".to_string(), vec![1])]);
        let scorer = BowScorer::new(Arc::new(index));
        let hits = scorer.search(&unit(&[1]), 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "a");
        assert_eq!(hits[1].0, "b");
    }

    #[test]
    fn saliency_modulation_suppresses_low_saliency_delta() {
        let index = Arc::new(PositionalIndex::build(vec![("d1".to_string(), vec![1, 2])]));
        let mut map = std::collections::HashMap::new();
        map.insert(1, 1.0); // term 1: 高显著性 S=1.0
        map.insert(2, 0.0); // term 2: 低显著性 S=0.0
        let scorer = BowScorer::with_saliency(Arc::clone(&index), 1.0, map, 0.5).unwrap();

        let s1 = scorer.score_doc(&unit(&[1]), 0);
        let s2 = scorer.score_doc(&unit(&[2]), 0);

        let s_base1 = BowScorer::new(Arc::clone(&index)).score_doc(&unit(&[1]), 0);
        let s_base2 = BowScorer::new(Arc::clone(&index)).score_doc(&unit(&[2]), 0);

        // term 1 获得了 delta 加成和 k1 放大
        assert!(s1 > s_base1);
        // term 2 (S=0.0) 的分数与 base 逐位一致，没有任何 delta 假阳性加分！
        assert_eq!(s2.to_bits(), s_base2.to_bits());
    }
}
