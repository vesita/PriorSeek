//! **SGL：因子统合打分（Sequential Generative Likelihood）**。
//!
//! 定义见 [`docs/PREREG-sgl.md`](../../../docs/PREREG-sgl.md)。外壳只有一条公式：
//!
//! ```text
//! score(q, d) = Σ_{t ∈ q ∩ d} ev(t, d)                   词项因子（默认 = BM25 term weight）
//!             + Σ_{相邻查询对} max(0, gap(g))              邻近因子（同向相邻，几何间距似然比）
//! ```
//!
//! 两个因子直接相加，没有融合系数；`ev` 的默认实现与 `bm25` 适配器逐位一致
//! （实测：关掉邻近因子时 nDCG@10 = 0.6647，与 bm25 相同）。
//! **默认 `gap_term = false`**：只启用已过门槛的因子；邻近因子在 scifact 上实测为负收益。
//!
//! **实测结论（scifact，300 计分查询）**：
//! - 邻近因子是**负收益**（0.6647 → 0.6474）；DPH 词项因子也是负收益（0.6293）；
//! - **概念因子（内核产出）是正收益**（0.6647 → 0.6788），外壳公式不变、只换因子；
//! - 所以 SGL 的默认配置是"BM25 词项因子 + 邻近关"，概念因子走 `SL_SGL_CONCEPT_ARTIFACT`。
//!
//! 完整数据见 [`docs/RESULTS.md`](../../../docs/RESULTS.md) §38。
//!
//! 三种退化端点（消融用）：
//!
//! | 设定 | 退化成 |
//! |---|---|
//! | `gap_term = false` | BM25（逐位一致） |
//! | `term_term = false` | 纯邻近结构（覆盖数 + 邻近加成） |
//! | 两者都关 | 命中词项个数 |

use crate::bow::WeightedTerm;
use crate::index::PositionalIndex;
use std::sync::Arc;

/// 词项因子的实现。默认 `Bm25`（与 `bm25` 适配器逐位一致）；`Dph` 是参数自由的
/// DFR 随机性模型（Bose-Einstein），不含 k1/b，用来验证"换一个词项因子"能否独立成立。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermFactor {
    /// BM25 的对数几率饱和形式（有 k1/b，作为默认与对照）。
    Bm25,
    /// DFR 的 DPH：`(f1 + f2) / (f + 0.5)`，f1 为信息量、f2 为 Laplacian 修正，无自由参数。
    Dph,
}

impl TermFactor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Dph => "dph",
        }
    }
}

/// SGL 的结构参数。只有"允许跨多远"这一个旋钮，且它是结构约束不是打分权重。
#[derive(Debug, Clone, PartialEq)]
pub struct SglParams {
    /// 词项因子实现（默认 BM25 形式；DPH 无自由参数）。
    pub term_factor: TermFactor,
    /// 两个相邻命中之间允许的最大位置间隔；`None` = 不截断（默认）。
    pub window: Option<u32>,
    /// 是否计入间距证据（关掉 = 词项证据覆盖）。
    pub gap_term: bool,
    /// 是否计入词项证据（关掉 = 每命中记 1 分）。
    pub term_term: bool,
}

impl Default for SglParams {
    fn default() -> Self {
        Self {
            term_factor: TermFactor::Bm25,
            window: None,
            // 默认只启用**已过门槛**的因子：词项因子（与 bm25 逐位一致）。
            // 邻近因子在 scifact 上是负收益（0.6647 → 0.6474），默认关闭；
            // 换语料要用时显式打开（SL_SGL_GAP_TERM=true），并重新预注册。
            gap_term: false,
            term_term: true,
        }
    }
}

impl SglParams {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "term_factor": self.term_factor.as_str(),
            "window": self.window,
            "gap_term": self.gap_term,
            "term_term": self.term_term,
            "formula": "max over alignment [ Σ ev(t,d) + Σ gap(g) ]；ev=ln((tf+1)/(len·df/N+1))，gap=ln(r/u)+(g-1)·ln((1-r)/(1-u))",
        })
    }
}

/// 因子统合打分器：持有索引与结构参数，对候选文档给出最优对齐的对数似然。
#[derive(Debug)]
pub struct SglScorer {
    index: Arc<PositionalIndex>,
    params: SglParams,
    /// 每个查询下标已处理事件的 `(文档位置, 最优值)`，位置升序。
    history: Vec<Vec<(u32, f64)>>,
}

impl SglScorer {
    pub fn new(index: Arc<PositionalIndex>, params: SglParams) -> Self {
        Self {
            index,
            params,
            history: Vec::new(),
        }
    }

    pub fn index(&self) -> &PositionalIndex {
        &self.index
    }

    pub fn params(&self) -> &SglParams {
        &self.params
    }

    /// 文档在查询下的 SGL 分数。
    ///
    /// 两项因子直接相加，没有对齐约束（把"词序"当硬约束已被证伪：
    /// 有序对齐在 scifact 上从 0.6647 掉到 0.5427，见 docs/ALGORITHM.md §4）：
    ///
    /// ```text
    /// score = Σ_{t ∈ q ∩ d} ev(t, d)                      词项证据（覆盖）
    ///       + Σ_{相邻查询对 (t_i, t_{i+1}) 都在 d 里}      邻近加成（无序，只加不减）
    ///           max(0, gap(min|pos_i − pos_j|))
    /// ```
    pub fn score_doc(&mut self, query: &[WeightedTerm], doc: u32) -> f64 {
        let len = self.index.doc_len(doc);
        if query.is_empty() || len == 0 {
            return 0.0;
        }

        // 查询词去重（保序），并记录每个词在文档里的出现位置。
        let mut seen: Vec<u32> = Vec::new();
        let mut positions: Vec<Vec<u32>> = Vec::new();
        for (term, weight) in query {
            if *weight == 0.0 || seen.contains(term) {
                continue;
            }
            let mut occ = self.index.positions_in(*term, doc);
            if occ.is_empty() {
                continue;
            }
            occ.sort_unstable();
            seen.push(*term);
            positions.push(occ);
        }
        if seen.is_empty() {
            return 0.0;
        }

        let len_f = len as f64;
        let m = seen.len();
        let r = (m as f64 / len_f).min(1.0 - 1e-9);
        let u = 1.0 / len_f;
        let log_ratio = (r / u).ln();
        let log_survive = ((1.0 - r) / (1.0 - u)).ln();
        let gap_value = |gap: u32| -> f64 {
            (log_ratio + f64::from(gap.saturating_sub(1)) * log_survive).max(0.0)
        };

        let mut score = 0.0f64;
        for (slot, &term) in seen.iter().enumerate() {
            let weight = query
                .iter()
                .find(|(candidate, _)| *candidate == term)
                .map(|(_, w)| *w)
                .unwrap_or(1.0);
            if self.params.term_term {
                score += weight * self.log_evidence(term, doc, len);
            } else {
                score += weight;
            }
            // 与下一个查询词（如果也在文档里）的最近距离，作为邻近加成。
            if self.params.gap_term {
                if let Some(next_positions) = positions.get(slot + 1) {
                    let mut best_gap: Option<u32> = None;
                    for &a in &positions[slot] {
                        for &b in next_positions {
                            // 只认同向（查询顺序）的相邻出现：反向相邻不计邻近。
                            if b <= a {
                                continue;
                            }
                            let gap = b - a;
                            if best_gap.map_or(true, |current| gap < current) {
                                best_gap = Some(gap);
                            }
                        }
                    }
                    if let Some(gap) = best_gap {
                        score += gap_value(gap);
                    }
                }
            }
        }
        score
    }

    /// 词项证据：**内核缺席时的默认因子**（BM25 的对数几率形式）。
    ///
    /// 实测教训：直接用语料背景率的 Dirichlet 对数似然比当词权，动态范围被压缩
    /// （稀有词上限约 1.2 nats，BM25 的 idf 可达 6.8），nDCG 掉到 0.62。
    /// 所以默认因子沿用 BM25 的 term weight —— 它本来就是"文档内出现"相对"随机文档"的
    /// 对数几率的一种饱和形式；接了内核之后，这一项整体换成内核产出的词权表即可。
    fn log_evidence(&self, term: u32, doc: u32, len: usize) -> f64 {
        match self.params.term_factor {
            TermFactor::Bm25 => self.bm25_evidence(term, doc, len),
            TermFactor::Dph => self.dph_evidence(term, doc, len),
        }
    }

    /// DFR 的 DPH（Amati 2006）：参数自由，只有 f、dl、avg_dl、N、F 五个统计量。
    ///
    /// `f1 = f·ln(1 + (avg_dl/dl)·(N/F))`（信息量），`f2 = ½·ln(2π f (1 − f/dl))`（Laplacian），
    /// `score = (f1 + f2) / (f + 0.5)`。用自然对数，与邻近因子同为 nats。
    fn dph_evidence(&self, term: u32, doc: u32, len: usize) -> f64 {
        let f = self.index.positions_in(term, doc).len() as f64;
        if f <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let dl = len as f64;
        let n = self.index.len() as f64;
        let collection_f = self.index.occurrences(term).len() as f64;
        if n <= 0.0 || collection_f <= 0.0 || dl <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let avg = self.index.avg_len().max(1.0);
        let f1 = f * (1.0 + (avg / dl) * (n / collection_f)).ln();
        let tail = (1.0 - f / dl).max(1e-9);
        let f2 = 0.5 * (2.0 * std::f64::consts::PI * f * tail).ln();
        (f1 + f2) / (f + 0.5)
    }

    fn bm25_evidence(&self, term: u32, doc: u32, len: usize) -> f64 {
        let tf = self.index.positions_in(term, doc).len() as f64;
        if tf <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let n = self.index.len() as f64;
        let df = f64::from(self.index.doc_freq(term));
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
        if idf <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let avg_len = self.index.avg_len().max(1.0);
        let k1 = 1.5;
        let b = 0.75;
        let denom = tf + k1 * (1.0 - b + b * (len as f64) / avg_len);
        idf * tf * (k1 + 1.0) / denom
    }

    /// 检索：候选集 = 命中任一词项的文档，逐个打 SGL 分。
    pub fn search(&mut self, query: &[WeightedTerm], k: usize) -> Vec<(String, f64)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        let terms: Vec<u32> = query.iter().map(|(term, _)| *term).collect();
        let candidates = self.index.candidates(&terms);
        let mut scored: Vec<(u32, f64)> = candidates
            .into_iter()
            .map(|doc| {
                let score = self.score_doc(query, doc);
                (doc, score)
            })
            .filter(|(_, score)| score.is_finite())
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

    fn build(docs: Vec<(&str, Vec<u32>)>) -> Arc<PositionalIndex> {
        Arc::new(PositionalIndex::build(
            docs.into_iter()
                .map(|(id, tokens)| (id.to_string(), tokens)),
        ))
    }

    /// 手算校验：三篇文档、查询 [1,2]，按公式逐项算出期望值。
    ///
    /// 语料：d0=[1,2]（len 2）、d1=[1,9,2]（len 3）、d2=[2,1]（len 2），avg_len=7/3。
    /// 两个词项 df 都是 3、N=3 ⇒ idf = ln(1 + 0.5/3.5)。
    /// tf=1、len=2 ⇒ 饱和项 = 1·2.5 / (1 + 1.5·(0.25 + 0.75·2/(7/3)))。
    /// 邻近：m=2、len=2 ⇒ r→1、u=0.5 ⇒ ln(r/u) = ln 2。
    #[test]
    fn hand_computed_score_matches_formula() {
        let index = build(vec![
            ("d0", vec![1, 2]),
            ("d1", vec![1, 9, 2]),
            ("d2", vec![2, 1]),
        ]);
        let mut scorer = SglScorer::new(
            Arc::clone(&index),
            SglParams {
                gap_term: true,
                ..Default::default()
            },
        );
        let score = scorer.score_doc(&unit(&[1, 2]), 0);

        let idf = (1.0f64 + 0.5 / 3.5).ln();
        let saturation = 1.0 * 2.5 / (1.0 + 1.5 * (0.25 + 0.75 * 2.0 / (7.0 / 3.0)));
        let expected = 2.0 * idf * saturation + 2.0f64.ln();
        assert!(
            (score - expected).abs() < 1e-6,
            "score={score} expected={expected}"
        );
    }

    /// 覆盖度：命中两个词的文档必须高于只命中一个词的文档（等长）。
    #[test]
    fn coverage_beats_partial_match_on_equal_length_documents() {
        let mut docs: Vec<(String, Vec<u32>)> = vec![
            ("both".to_string(), vec![1, 2, 9]),
            ("only-one".to_string(), vec![1, 9, 9]),
        ];
        for filler in 0..20u32 {
            docs.push((format!("filler-{filler}"), vec![100 + filler, 200 + filler]));
        }
        let index = Arc::new(PositionalIndex::build(docs));
        let mut scorer = SglScorer::new(Arc::clone(&index), SglParams::default());
        let both = scorer.score_doc(&unit(&[1, 2]), 0);
        let one = scorer.score_doc(&unit(&[1, 2]), 1);
        assert!(both > one, "覆盖更多查询词应更高：{both} vs {one}");
    }

    /// 邻近：等长文档里，两个词相邻的必须高于被撑开的。
    #[test]
    fn proximity_beats_spread_on_equal_length_documents() {
        let mut docs: Vec<(String, Vec<u32>)> = vec![
            ("tight".to_string(), vec![1, 2, 9, 9]),
            ("loose".to_string(), vec![1, 9, 9, 2]),
        ];
        for filler in 0..20u32 {
            docs.push((format!("filler-{filler}"), vec![100 + filler, 200 + filler]));
        }
        let index = Arc::new(PositionalIndex::build(docs));
        let mut scorer = SglScorer::new(
            Arc::clone(&index),
            SglParams {
                gap_term: true,
                ..Default::default()
            },
        );
        let tight = scorer.score_doc(&unit(&[1, 2]), 0);
        let loose = scorer.score_doc(&unit(&[1, 2]), 1);
        assert!(tight > loose, "紧邻应高于分散：{tight} vs {loose}");
    }

    /// 空查询 / 词项都不在文档里 ⇒ 0（不是负分，也不是 −inf）。
    #[test]
    fn empty_query_and_missing_terms_score_zero() {
        let index = build(vec![("d0", vec![1, 2])]);
        let mut scorer = SglScorer::new(Arc::clone(&index), SglParams::default());
        assert_eq!(scorer.score_doc(&[], 0), 0.0);
        assert_eq!(scorer.score_doc(&unit(&[42]), 0), 0.0);
    }

    /// 关掉间距项：等长且覆盖相同的两篇必须同分（邻近加成消失）。
    #[test]
    fn disabling_gap_term_leaves_only_term_evidence() {
        let mut docs: Vec<(String, Vec<u32>)> = vec![
            ("tight".to_string(), vec![1, 2, 9, 9]),
            ("loose".to_string(), vec![1, 9, 9, 2]),
        ];
        for filler in 0..20u32 {
            docs.push((format!("filler-{filler}"), vec![100 + filler, 200 + filler]));
        }
        let index = Arc::new(PositionalIndex::build(docs));
        let mut scorer = SglScorer::new(
            Arc::clone(&index),
            SglParams {
                gap_term: false,
                ..Default::default()
            },
        );
        let tight = scorer.score_doc(&unit(&[1, 2]), 0);
        let loose = scorer.score_doc(&unit(&[1, 2]), 1);
        assert!((tight - loose).abs() < 1e-9, "关掉间距项后应同分：{tight} vs {loose}");
    }

    /// 两项都关：分数 = 文档里出现的查询词个数。
    #[test]
    fn disabling_both_terms_degenerates_to_coverage_count() {
        let index = build(vec![
            ("three", vec![1, 2, 3]),
            ("two", vec![1, 9, 2]),
        ]);
        let mut scorer = SglScorer::new(
            Arc::clone(&index),
            SglParams {
                gap_term: false,
                term_term: false,
                ..Default::default()
            },
        );
        let query = unit(&[1, 2, 3]);
        assert!((scorer.score_doc(&query, 0) - 3.0).abs() < 1e-9);
        assert!((scorer.score_doc(&query, 1) - 2.0).abs() < 1e-9);
    }
}
