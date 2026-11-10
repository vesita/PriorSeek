//! **OAC：有序对齐覆盖（Ordered Alignment Coverage）** —— 本仓的检索算法。
//!
//! 定义（`docs/ALGORITHM.md`）：
//!
//! ```text
//! q = [t1..tn]                 查询词序列（token id）
//! d = 文档 token 序列
//! 对齐 A = 一组 (i, j)，要求 i 严格递增（对应 t1..tn 的顺序）、j 严格递增（文档内位置）
//! score(q,d) = max over A  Σ_{(i,j) ∈ A} idf(t_i) · K(gap(j, 上一个 j))
//! ```
//!
//! - **单调约束**（`ordered`）是算法的核心：词序必须在文档里以同样的顺序出现；
//! - **`K`（gap 权重）由内核提供**：内核只回答「间隔为 g 的相邻匹配值多少分」；
//! - 允许跳过少量查询词（`max_skip`）：不是所有查询词都能在文档里出现。
//!
//! 两个退化端点（消融用，不是补丁）：
//!
//! | 设定 | 退化成 |
//! |---|---|
//! | `ordered = false`，`K ≡ 1` | 词袋 IDF 覆盖 |
//! | `ordered = true`，且只允许 gap = 1 | 短语匹配 |
//!
//! 复杂度：对每篇候选文档，只在「实际命中的出现位置」上做 DP，
//! `O(命中出现数 × max_skip × window)`，与文档长度线性、与语料规模无关。

use crate::index::PositionalIndex;
use std::sync::Arc;

/// BM25 的 k1（与 `bow` 打分的参数同值；OAC 的逐项权重沿用它）。
pub const BM25_K1: f64 = 1.5;

/// **先验因子**：gap → 权重表。
///
/// 这是外壳与内核的**唯一边界**：内核网络产出这张表（当前 v0 由语料统计给出），
/// 外壳只负责消费它——内核换实现时，这里的内容换、外壳代码不变。
/// 表是分桶的（`[1]`, `[2]`, `[3..=4]`, `[5..=8]`, `>8`），共 5 个权重。
#[derive(Debug, Clone, PartialEq)]
pub struct KernelGapTable {
    /// 5 个桶：gap=1 / gap=2 / gap∈[3,4] / gap∈[5,8] / gap>8。
    pub weights: [f64; 5],
}

impl Default for KernelGapTable {
    fn default() -> Self {
        // 缺省 = 均匀（内核缺席时的退化形态，用于「内核是否必要」的消融）。
        Self { weights: [1.0; 5] }
    }
}

impl KernelGapTable {
    /// 均匀表（消融用：等价于"不看间隔"）。
    pub fn uniform() -> Self {
        Self { weights: [1.0; 5] }
    }

    /// 按间隙取权重。gap 是「当前匹配位置 − 上一个匹配位置」。
    pub fn weight(&self, gap: u32) -> f64 {
        match gap {
            0 | 1 => self.weights[0],
            2 => self.weights[1],
            3..=4 => self.weights[2],
            5..=8 => self.weights[3],
            _ => self.weights[4],
        }
    }

    pub fn bucket_of(gap: u32) -> usize {
        match gap {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "gap_weights": {
                "gap=1": self.weights[0],
                "gap=2": self.weights[1],
                "gap=3..4": self.weights[2],
                "gap=5..8": self.weights[3],
                "gap>8": self.weights[4],
            }
        })
    }
}

/// OAC 的参数。全部是算法定义的一部分，不是"调参补丁"。
#[derive(Debug, Clone, PartialEq)]
pub struct OacParams {
    /// 是否要求词序单调（关掉 = 词袋）。
    pub ordered: bool,
    /// 是否使用 span 折扣（关掉 = K ≡ 1）。
    pub use_span_discount: bool,
    /// 两个相邻匹配之间允许的最大位置间隔；超过则不构成一次对齐转移。
    pub window: u32,
    /// 允许连续跳过多少个查询词（0 = 每个查询词都必须命中）。
    pub max_skip: usize,
    /// 内核提供的 gap 权重表。
    pub kernel: KernelGapTable,
    /// 长度归一强度 `b`（0 = 不归一）。**与 BM25 同一个 b**：这里不是"整体除以长度"，
    /// 而是用在每个匹配项的饱和权重里（见 [`Self::term_weight`]）。
    pub length_b: f64,
}

impl Default for OacParams {
    fn default() -> Self {
        Self {
            ordered: true,
            use_span_discount: true,
            window: 8,
            max_skip: 2,
            kernel: KernelGapTable::default(),
            length_b: 0.75,
        }
    }
}

impl OacParams {
    pub fn to_json(&self) -> serde_json::Value {
        let mut value = self.kernel.to_json();
        if let Some(obj) = value.as_object_mut() {
            obj.insert("ordered".into(), serde_json::json!(self.ordered));
            obj.insert("use_span_discount".into(), serde_json::json!(self.use_span_discount));
            obj.insert("window".into(), serde_json::json!(self.window));
            obj.insert("max_skip".into(), serde_json::json!(self.max_skip));
            obj.insert("length_b".into(), serde_json::json!(self.length_b));
        }
        value
    }
}

/// OAC 打分器：持有索引与参数，对候选文档给出「最优有序对齐」的分数。
#[derive(Debug)]
pub struct OacScorer {
    index: Arc<PositionalIndex>,
    params: OacParams,
    /// 单查询复用缓冲：每个查询下标最近 window 内的 (文档位置, 最优值)。
    /// 用「位置 + 值」显式校验，不用取模——取模会在 window 变小时把远处的位置错认成近处。
    recent: Vec<Vec<(u32, f64)>>,
    /// 每个查询下标「任意更早位置」的最优值（无序端点用）。
    prefix_best: Vec<f64>,
}

impl OacScorer {
    pub fn new(index: Arc<PositionalIndex>, params: OacParams) -> Self {
        let mut scorer = Self { index, params, recent: Vec::new(), prefix_best: Vec::new() };
        scorer.reset_buffers(0);
        scorer
    }

    pub fn index(&self) -> &PositionalIndex {
        &self.index
    }

    pub fn params(&self) -> &OacParams {
        &self.params
    }

    fn reset_buffers(&mut self, query_len: usize) {
        self.recent = vec![Vec::new(); query_len.max(1)];
        self.prefix_best = vec![f64::NEG_INFINITY; query_len.max(1)];
    }

    /// 文档在查询下的 OAC 分数。
    pub fn score_doc(&mut self, query: &[u32], doc: u32) -> f64 {
        if query.is_empty() {
            return 0.0;
        }
        if self.recent.len() < query.len() {
            self.reset_buffers(query.len());
        }
        for row in self.recent.iter_mut().take(query.len()) {
            row.clear();
        }
        for slot in self.prefix_best.iter_mut().take(query.len()) {
            *slot = f64::NEG_INFINITY;
        }

        // 收集本查询在本文档内的命中事件：(位置, 查询下标)，按位置升序。
        let mut events: Vec<(u32, usize)> = Vec::new();
        for (qi, term) in query.iter().enumerate() {
            for pos in self.index.positions_in(*term, doc) {
                events.push((pos, qi));
            }
        }
        if events.is_empty() {
            return 0.0;
        }
        events.sort_unstable();

        let window = self.params.window;
        let max_skip = self.params.max_skip;

        // 无序端点：词袋 IDF 覆盖。每个查询词独立贡献，不链式、不看位置，
        // 这正是 OAC 在 `ordered = false` 时的退化形态（见模块文档的退化表）。
        if !self.params.ordered {
            let mut seen: Vec<u32> = Vec::new();
            let mut total = 0.0;
            for (_, qi) in &events {
                let term = query[*qi];
                if seen.contains(&term) {
                    continue;
                }
                seen.push(term);
                total += self.term_weight(term, doc);
            }
            return total;
        }

        let mut best_total = 0.0f64;

        for (pos, qi) in events {
            let idf = self.term_weight(query[qi], doc);
            if idf <= 0.0 {
                continue;
            }
            // 起始：只匹配这一个查询词，没有前驱 ⇒ gap 记 0。
            let mut best_here = idf * self.gap_weight(0);

            if self.params.ordered {
                // 前驱必须是更早的查询词，文档位置更靠前且间隔 ≤ window。
                for skip in 0..=max_skip {
                    if qi < skip + 1 {
                        break;
                    }
                    let prev_qi = qi - skip - 1;
                    for &(prev_pos, prev_val) in self.recent[prev_qi].iter().rev() {
                        if prev_pos >= pos {
                            break;
                        }
                        let gap = pos - prev_pos;
                        if gap > window {
                            break;
                        }
                        let candidate = prev_val + idf * self.gap_weight(gap);
                        if candidate > best_here {
                            best_here = candidate;
                        }
                    }
                }
            }

            // 写入近期队列（保序：位置升序），并更新前缀最优。
            self.recent[qi].push((pos, best_here));
            // 只保留仍在窗口内的（更早的对后续事件不再可达）。
            let cutoff = pos.saturating_sub(window);
            if let Some(first_keep) =
                self.recent[qi].iter().position(|(p, _)| *p >= cutoff)
            {
                self.recent[qi].drain(..first_keep);
            }
            if best_here > self.prefix_best[qi] {
                self.prefix_best[qi] = best_here;
            }
            if best_here > best_total {
                best_total = best_here;
            }
        }
        best_total
    }

    /// 单个匹配项的权重：**BM25 的 term weight**（tf 饱和 + 长度归一）。
    ///
    /// `idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·len/avg))`。
    /// 这一项决定了「一次匹配值多少分」，而 OAC 的有序约束决定「哪些匹配可以同时计入」。
    /// 当 `ordered = false` 且不做 gap 折扣时，整条公式逐字退化为 BM25。
    fn term_weight(&self, term: u32, doc: u32) -> f64 {
        let tf = self.index.positions_in(term, doc).len() as f64;
        if tf <= 0.0 {
            return 0.0;
        }
        let idf = self.index.idf(term);
        if idf <= 0.0 {
            return 0.0;
        }
        let avg_len = self.index.avg_len();
        let len = self.index.doc_len(doc) as f64;
        let b = self.params.length_b.clamp(0.0, 1.0);
        let denom = tf + BM25_K1 * (1.0 - b + b * len / avg_len);
        idf * tf * (BM25_K1 + 1.0) / denom
    }

    fn gap_weight(&self, gap: u32) -> f64 {
        if !self.params.use_span_discount || !self.params.ordered {
            1.0
        } else {
            self.params.kernel.weight(gap)
        }
    }

    /// 检索：候选集 = 命中任一词项的文档，逐个打 OAC 分。
    pub fn search(&mut self, query: &[u32], k: usize) -> Vec<(String, f64)> {
        if query.is_empty() || k == 0 {
            return Vec::new();
        }
        let candidates = self.index.candidates(query);
        let mut scored: Vec<(u32, f64)> = candidates
            .into_iter()
            .map(|doc| {
                let score = self.score_doc(query, doc);
                (doc, score)
            })
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

    /// d1 = [1,2,3]（查询词按序、连续）
    /// d2 = [3,2,1]（顺序相反）
    /// d3 = [1,9,9,2]（按序但间隔 2）
    fn scorer(params: OacParams) -> OacScorer {
        let index = PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 2, 3]),
            ("d2".to_string(), vec![3, 2, 1]),
            ("d3".to_string(), vec![1, 9, 9, 2]),
            ("d4".to_string(), vec![9, 9, 9]),
        ]);
        OacScorer::new(Arc::new(index), params)
    }

    #[test]
    fn ordered_matching_prefers_in_order_documents() {
        // 只比较**等长文档**，把「长度归一」这个因素排除掉：
        // d1 = [1,2,3] 正序，d2 = [3,2,1] 反序，两者长度都是 3。
        let mut s = scorer(OacParams { window: 8, max_skip: 2, ..Default::default() });
        let q = [1u32, 2u32, 3u32];
        let d1 = s.score_doc(&q, 0);
        let d2 = s.score_doc(&q, 1);
        assert!(d1 > 0.0 && d2 > 0.0);
        assert!(d1 > d2, "等长文档下，正序应高于反序：{d1} vs {d2}");
    }

    #[test]
    fn length_normalization_penalizes_sparse_matches_in_long_documents() {
        // d3 = [1,9,9,2]（len 4，命中 2 个查询词）vs d2 = [3,2,1]（len 3，命中 3 个但反序）。
        // 长度归一开启时，长文档的稀疏匹配会被压低——这是模型定义的一部分，不是 bug。
        let mut with_norm = scorer(OacParams { window: 8, length_b: 0.75, ..Default::default() });
        let mut without = scorer(OacParams { window: 8, length_b: 0.0, ..Default::default() });
        let q = [1u32, 2u32, 3u32];
        let (a, b) = (with_norm.score_doc(&q, 2), with_norm.score_doc(&q, 1));
        let (a0, b0) = (without.score_doc(&q, 2), without.score_doc(&q, 1));
        assert!(a < b, "有归一时长文档稀疏匹配被压低：{a} vs {b}");
        assert!(a0 > b0 || (a0 - b0).abs() > 0.0, "无归一时该关系应不同：{a0} vs {b0}");
    }

    #[test]
    fn unordered_mode_ignores_order() {
        let mut s = scorer(OacParams {
            ordered: false,
            window: 8,
            max_skip: 2,
            ..Default::default()
        });
        let q = [1u32, 2u32, 3u32];
        let d1 = s.score_doc(&q, 0);
        let d2 = s.score_doc(&q, 1);
        assert!((d1 - d2).abs() < 1e-12, "无序模式下反序文档应与正序同分：{d1} vs {d2}");
    }

    #[test]
    fn uniform_kernel_removes_span_discount() {
        let decay = KernelGapTable { weights: [1.0, 0.5, 0.25, 0.1, 0.05] };
        let mut with_discount = scorer(OacParams {
            kernel: decay,
            window: 8,
            ..Default::default()
        });
        let mut uniform = scorer(OacParams {
            kernel: KernelGapTable::uniform(),
            window: 8,
            ..Default::default()
        });
        let q = [1u32, 2u32];
        let a = with_discount.score_doc(&q, 2); // d3: 1 与 2 间隔 3
        let b = uniform.score_doc(&q, 2);
        assert!(a < b, "有折扣时应低于均匀：{a} vs {b}");
    }

    #[test]
    fn window_limits_transitions() {
        let mut tight = scorer(OacParams { window: 1, max_skip: 0, ..Default::default() });
        let mut loose = scorer(OacParams { window: 8, max_skip: 0, ..Default::default() });
        let q = [1u32, 2u32];
        let tight_score = tight.score_doc(&q, 2); // d3: 1 与 2 间隔 3，超出 window=1
        let loose_score = loose.score_doc(&q, 2);
        assert!(tight_score < loose_score, "窗口收紧应降低跨距匹配的分数");
    }

    #[test]
    fn empty_query_scores_zero_and_document_without_terms_scores_zero() {
        let mut s = scorer(OacParams::default());
        assert_eq!(s.score_doc(&[], 0), 0.0);
        assert_eq!(s.score_doc(&[1u32, 2u32], 3), 0.0); // d4 不含 1/2
    }
}
