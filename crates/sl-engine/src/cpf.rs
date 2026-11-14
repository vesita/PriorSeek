//! 全息相干势能场算子 (Coherent Potential Field, CPF) 引擎核心定义
//! 
//! 本模块定义了 CPF 的基础数据结构。CPF 放弃传统的“向量化”或“倒排词化”割裂方案，
//! 将文档处理为“多极局域势场”，查询处理为“引力相干探针组”。

/// 一个全息文档势场由一组局域势能极点组成。
/// 每个极点记录概念槽位 ID、局域语义向量表征与局域质量因子。
pub struct DocPotentialField {
    pub poles: Vec<PotentialPole>,
}

pub struct PotentialPole {
    pub concept_id: u32,
    pub semantic_vector: Vec<f32>, // 归一化语义表征
    pub local_mass: f32,          // 局域重要度/质量
}

/// 查询引力相干探针组。
/// 每个探针由一个概念 ID、语义表征与全局特异度权重构成。
pub struct QueryProbeSet {
    pub probes: Vec<QueryProbe>,
}

pub struct QueryProbe {
    pub concept_id: u32,
    pub semantic_vector: Vec<f32>,
    pub specificity_weight: f32, // 对应原先的 w_j (IDF 深度先验)
}

/// 核心算子 Trait
pub trait CpfOperator {
    /// 计算查询在特定文档势场中的总相干能量：
    /// TotalEnergy = sum_{j in q} w_j * ln(1.0 + max_{i in d} (Equiv(c_j, c_i) * exp(gamma * <q_j, v_i>)))
    fn compute_total_energy(&self, query: &QueryProbeSet, doc: &DocPotentialField, gamma: f32) -> f32;
}

pub struct CpfEngine {
    pub gamma: f32,
}

impl CpfOperator for CpfEngine {
    fn compute_total_energy(&self, query: &QueryProbeSet, doc: &DocPotentialField, gamma: f32) -> f32 {
        let mut total_energy = 0.0;

        for probe in &query.probes {
            let mut max_local_phi = 0.0;
            
            // 在文档的所有局域势极点中寻找与该探针最相干的极点
            for pole in &doc.poles {
                if pole.concept_id == probe.concept_id {
                    // 相干度核：余弦相似度指数映射 exp(gamma * <q, v>)
                    let dot = dot_product(&probe.semantic_vector, &pole.semantic_vector);
                    let phi = (gamma * dot).exp();
                    if phi > max_local_phi {
                        max_local_phi = phi;
                    }
                }
            }

            // 累加该探针贡献的对数相干势能：w_j * ln(1.0 + Phi)
            if max_local_phi > 0.0 {
                total_energy += probe.specificity_weight * (1.0 + max_local_phi).ln();
            }
        }
        
        total_energy
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpf_monopoly_suppression() {
        let engine = CpfEngine { gamma: 2.0 };

        // 查询由两个概念探针构成: 概念 10 (IDF=5.0) + 概念 20 (IDF=5.0)
        let query = QueryProbeSet {
            probes: vec![
                QueryProbe {
                    concept_id: 10,
                    semantic_vector: vec![1.0, 0.0],
                    specificity_weight: 5.0,
                },
                QueryProbe {
                    concept_id: 20,
                    semantic_vector: vec![0.0, 1.0],
                    specificity_weight: 5.0,
                },
            ],
        };

        // 文档 A: 均匀覆盖两个概念 (概念 10 和 概念 20 均存在且语义一致)
        let doc_a = DocPotentialField {
            poles: vec![
                PotentialPole {
                    concept_id: 10,
                    semantic_vector: vec![1.0, 0.0],
                    local_mass: 1.0,
                },
                PotentialPole {
                    concept_id: 20,
                    semantic_vector: vec![0.0, 1.0],
                    local_mass: 1.0,
                },
            ],
        };

        // 文档 B: 单词垄断 (概念 10 出现多次，但完全缺失概念 20)
        let doc_b = DocPotentialField {
            poles: vec![
                PotentialPole {
                    concept_id: 10,
                    semantic_vector: vec![1.0, 0.0],
                    local_mass: 5.0,
                },
                PotentialPole {
                    concept_id: 10,
                    semantic_vector: vec![1.0, 0.0],
                    local_mass: 5.0,
                },
            ],
        };

        let energy_a = engine.compute_total_energy(&query, &doc_a, 2.0);
        let energy_b = engine.compute_total_energy(&query, &doc_b, 2.0);

        // 核心物理验证：均匀覆盖命题的文档 A，能量必然碾压单词垄断的文档 B
        assert!(energy_a > energy_b, "均匀覆盖文档能量 ({energy_a}) 必须高于单词垄断文档 ({energy_b})");
    }

    #[test]
    fn test_cpf_semantic_alignment_boost() {
        let engine = CpfEngine { gamma: 2.0 };

        let query = QueryProbeSet {
            probes: vec![QueryProbe {
                concept_id: 1,
                semantic_vector: vec![1.0, 0.0],
                specificity_weight: 2.0,
            }],
        };

        // 语义完全对齐文档 (dot = 1.0)
        let doc_aligned = DocPotentialField {
            poles: vec![PotentialPole {
                concept_id: 1,
                semantic_vector: vec![1.0, 0.0],
                local_mass: 1.0,
            }],
        };

        // 语义垂直正交文档 (dot = 0.0)
        let doc_orthogonal = DocPotentialField {
            poles: vec![PotentialPole {
                concept_id: 1,
                semantic_vector: vec![0.0, 1.0],
                local_mass: 1.0,
            }],
        };

        let energy_aligned = engine.compute_total_energy(&query, &doc_aligned, 2.0);
        let energy_ortho = engine.compute_total_energy(&query, &doc_orthogonal, 2.0);

        assert!(energy_aligned > energy_ortho);
    }
}

