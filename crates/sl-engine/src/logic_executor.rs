//! 无参数化逻辑执行外壳 (Parameterless Logic Executor)。
//!
//! 外壳内部零硬编码超参数（无 k1, 无 b, 无 magic weights）。
//! 外壳纯粹作为物理逻辑约束谓词执行引擎：
//! 1. MatchAnchor: 基于相对篇幅的自然饱和锚点匹配；
//! 2. CheckCoverage: 基于先验显著性权重的完整性门控；
//! 3. VerifyCausalFlow: 基于实体位置物理跨距与方向的因果流验证。

use std::collections::HashMap;

/// 查询级逻辑约束描述符（由内核离线或在线编译产出）。
#[derive(Debug, Clone)]
pub struct QueryConstraint {
    /// 锚点词列表：[(term_id, saliency_weight, idf)]
    pub anchors: Vec<(u32, f32, f32)>,
    /// 实体对因果时序流约束：[(term_1, term_2, flow_strength)]
    pub causal_flows: Vec<(u32, u32, f32)>,
    /// 饱和项量纲标定因子（由内核编译注入，默认 2.0）
    pub saturation_scale: f32,
    /// 软门控阻尼系数（由内核编译注入，默认 1.0）
    pub gate_softness: f32,
}

impl Default for QueryConstraint {
    fn default() -> Self {
        Self {
            anchors: Vec::new(),
            causal_flows: Vec::new(),
            saturation_scale: 2.0,
            gate_softness: 1.0,
        }
    }
}

/// 谓词执行证据解释记录（完全可解释）。
#[derive(Debug, Clone)]
pub struct ExecutionExplanation {
    pub raw_anchor_score: f64,
    pub coverage_ratio: f32,
    pub coverage_gate: f32,
    pub flow_multiplier: f64,
    pub final_score: f64,
}

pub struct LogicExecutor;

impl LogicExecutor {
    /// 对单篇文档执行逻辑约束图验证与物理打分。
    /// 完全由物理输入与约束图驱动，外壳无任何硬编码常量。
    pub fn execute(
        constraint: &QueryConstraint,
        doc_tf: &HashMap<u32, u32>,
        doc_positions: impl Fn(u32) -> Vec<u32>,
        doc_len: usize,
        avg_len: f64,
    ) -> (f64, ExecutionExplanation) {
        if constraint.anchors.is_empty() || doc_len == 0 {
            return (
                0.0,
                ExecutionExplanation {
                    raw_anchor_score: 0.0,
                    coverage_ratio: 0.0,
                    coverage_gate: 0.0,
                    flow_multiplier: 1.0,
                    final_score: 0.0,
                },
            );
        }

        let len_ratio = (doc_len as f64) / avg_len.max(1.0);

        // 1. 谓词一：MatchAnchor（自然物理饱和）
        let mut raw_anchor_score = 0.0f64;
        let mut total_saliency = 0.0f32;
        let mut matched_saliency = 0.0f32;

        for &(term_id, saliency, idf) in &constraint.anchors {
            total_saliency += saliency;
            if let Some(&tf) = doc_tf.get(&term_id) {
                if tf > 0 {
                    matched_saliency += saliency;
                    let tf_f = tf as f64;
                    // 自然物理饱和（由内核注入量纲缩放）：tf * scale / (tf + len_ratio)
                    let sat = (tf_f * constraint.saturation_scale as f64) / (tf_f + len_ratio);
                    raw_anchor_score += (saliency as f64) * (idf as f64) * sat;
                }
            }
        }

        if raw_anchor_score <= 0.0 || total_saliency <= 0.0 {
            return (
                0.0,
                ExecutionExplanation {
                    raw_anchor_score: 0.0,
                    coverage_ratio: 0.0,
                    coverage_gate: 0.0,
                    flow_multiplier: 1.0,
                    final_score: 0.0,
                },
            );
        }

        // 2. 谓词二：CheckCoverage（平滑软门控：覆盖率越高越强，但绝不直接腰斩正样本）
        let coverage_ratio = (matched_saliency / total_saliency).min(1.0);
        let coverage_gate =
            (1.0 + constraint.gate_softness * coverage_ratio) / (1.0 + constraint.gate_softness);

        // 3. 谓词三：VerifyCausalFlow（因果时序流奖励）
        let mut flow_bonus = 0.0f64;
        for &(t1, t2, strength) in &constraint.causal_flows {
            if strength <= 0.0 {
                continue;
            }
            let p1 = doc_positions(t1);
            let p2 = doc_positions(t2);
            if p1.is_empty() || p2.is_empty() {
                continue;
            }

            let mut min_span = u32::MAX;
            let mut in_order = false;

            for &pos1 in &p1 {
                for &pos2 in &p2 {
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

            if in_order {
                let compact_factor = 1.0f64 / (1.0 + ((min_span as f64).ln_1p()));
                flow_bonus += (strength as f64) * compact_factor;
            }
        }

        let flow_multiplier = 1.0 + flow_bonus;
        let final_score = raw_anchor_score * (coverage_gate as f64) * flow_multiplier;

        (
            final_score,
            ExecutionExplanation {
                raw_anchor_score,
                coverage_ratio,
                coverage_gate,
                flow_multiplier,
                final_score,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logic_executor_rewards_high_coverage_and_penalizes_low_coverage() {
        let mut constraint = QueryConstraint::default();
        constraint.anchors = vec![(1, 1.0, 5.0), (2, 1.0, 5.0)]; // 两个核心词
        constraint.gate_softness = 1.0;

        let mut doc_full = HashMap::new();
        doc_full.insert(1, 2);
        doc_full.insert(2, 2);

        let mut doc_half = HashMap::new();
        doc_half.insert(1, 4); // 只命中词 1，词 2 完全缺失

        let (score_full, exp_full) =
            LogicExecutor::execute(&constraint, &doc_full, |_| vec![], 100, 100.0);
        let (score_half, exp_half) =
            LogicExecutor::execute(&constraint, &doc_half, |_| vec![], 100, 100.0);

        assert_eq!(exp_full.coverage_gate, 1.0);
        assert!(exp_half.coverage_gate < 1.0, "未完全覆盖的文档门控平滑受抑");
        assert!(score_full > score_half);
    }
}
