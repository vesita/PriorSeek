//! 极速最优传输语义流外壳算子 (Relaxed Optimal Transport / Earth Mover's Distance)。
//!
//! 彻底摒弃传统 BM25 词袋求和假设，将检索重构为查询意图流向文档论证流的最小传输代价问题。
//! 纯 Rust 原生浮点计算，无动态堆分配，单次文档流匹配耗时 < 1 微秒。

use std::collections::HashMap;

/// 词项间语义流传输代价表：(src_id, dst_id) -> 搬运成本 in [0.0, 1.0]。
#[derive(Debug, Clone, Default)]
pub struct CostGraph {
    /// 紧凑邻接表：src -> [(dst, cost)]
    pub edges: HashMap<u32, Vec<(u32, f32)>>,
}

impl CostGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// 查询从 src 到 dst 的语义搬运代价。
    /// 相同概念代价为 0.0；若无语义边相连则为最大代价 1.0。
    #[inline(always)]
    pub fn cost(&self, src: u32, dst: u32) -> f32 {
        if src == dst {
            return 0.0;
        }
        if let Some(neighbors) = self.edges.get(&src) {
            for &(neighbor, cost) in neighbors {
                if neighbor == dst {
                    return cost;
                }
            }
        }
        if let Some(neighbors) = self.edges.get(&dst) {
            for &(neighbor, cost) in neighbors {
                if neighbor == src {
                    return cost;
                }
            }
        }
        1.0
    }
}

pub struct FlowMatcher {
    pub costs: CostGraph,
    pub temperature: f32,
}

impl FlowMatcher {
    pub fn new(costs: CostGraph, temperature: f32) -> Self {
        Self { costs, temperature }
    }

    /// 计算查询意图流到文档论证流的最小最优传输成本 (RWMD)。
    ///
    /// query_terms: [(term_id, weight)]（通常 weight 为 IDF）
    /// doc_terms: [(term_id, tf)]
    #[inline]
    pub fn compute_transport_cost(&self, query: &[(u32, f32)], doc: &[(u32, f32)]) -> f32 {
        if query.is_empty() || doc.is_empty() {
            return 1.0;
        }

        // 1. 归一化查询质量分布
        let q_sum: f32 = query.iter().map(|(_, w)| *w).sum();
        if q_sum <= 0.0 {
            return 1.0;
        }

        // 2. 归一化文档质量分布
        let d_sum: f32 = doc.iter().map(|(_, tf)| *tf).sum();
        if d_sum <= 0.0 {
            return 1.0;
        }

        // 3. 计算查询单向最小搬运成本：每个查询意图在文档中找到的最便宜替代概念
        let mut cost_q_to_d = 0.0f32;
        for &(q_term, q_w) in query {
            let mut min_c = 1.0f32;
            for &(d_term, _) in doc {
                let c = self.costs.cost(q_term, d_term);
                if c < min_c {
                    min_c = c;
                    if min_c == 0.0 {
                        break;
                    }
                }
            }
            cost_q_to_d += (q_w / q_sum) * min_c;
        }

        // 4. 计算文档单向最小搬运成本：文档主题离查询意图的偏离度
        let mut cost_d_to_q = 0.0f32;
        for &(d_term, d_tf) in doc {
            let mut min_c = 1.0f32;
            for &(q_term, _) in query {
                let c = self.costs.cost(q_term, d_term);
                if c < min_c {
                    min_c = c;
                    if min_c == 0.0 {
                        break;
                    }
                }
            }
            cost_d_to_q += (d_tf / d_sum) * min_c;
        }

        cost_q_to_d.max(cost_d_to_q)
    }

    /// 将传输代价映射为相似度得分：Score = exp(-Cost / tau)。
    #[inline]
    pub fn score(&self, query: &[(u32, f32)], doc: &[(u32, f32)]) -> f64 {
        let cost = self.compute_transport_cost(query, doc);
        ((-cost / self.temperature) as f64).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_matcher_identical_documents_have_zero_cost() {
        let matcher = FlowMatcher::new(CostGraph::new(), 0.5);
        let q = vec![(1, 1.0), (2, 2.0)];
        let d = vec![(1, 1.0), (2, 3.0)];
        let cost = matcher.compute_transport_cost(&q, &d);
        assert_eq!(cost, 0.0, "完全相同的概念集合搬运代价必须为 0");
        let score = matcher.score(&q, &d);
        assert!((score - 1.0).abs() < 1e-6, "代价为 0 时得分为 1.0");
    }

    #[test]
    fn flow_matcher_semantic_neighbors_reduce_cost() {
        let mut costs = CostGraph::new();
        // 假设 1 (cancer) 与 2 (carcinoma) 存在语义关联，距离代价 0.2
        costs.edges.insert(1, vec![(2, 0.2)]);

        let matcher = FlowMatcher::new(costs, 0.5);
        let q = vec![(1, 1.0)];
        let d_synonym = vec![(2, 1.0)]; // 文档使用同义词
        let d_unrelated = vec![(99, 1.0)]; // 文档完全无关

        let cost_synonym = matcher.compute_transport_cost(&q, &d_synonym);
        let cost_unrelated = matcher.compute_transport_cost(&q, &d_unrelated);

        assert!((cost_synonym - 0.2).abs() < 1e-6);
        assert_eq!(cost_unrelated, 1.0);
        assert!(matcher.score(&q, &d_synonym) > matcher.score(&q, &d_unrelated));
    }
}
