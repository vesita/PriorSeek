//! 全息相干势能场检索适配器 (Coherent Potential Field Adapter, CPF)。
//!
//! 原生一体化算子架构：
//! 彻底废除晚期列表投票（RRF）与离散加权拼凑，在单一数学方程中内生融合四大核心灵魂：
//! 1. 连续流形相干项：SVD 白化连续语义向量内积 <q, d>，消除高维锥形各向异性；
//! 2. 离散概念饱和项：凹函数 ln(1 + Phi_lex) 抑制单词垄断，保持局部边际递减；
//! 3. 命题合取完备性：全命题覆盖度惩罚 beta * ln(0.2 + 0.8 * Cover)；
//! 4. 因果相位相干项：在线因果极性识别与文档谓词极性相位相消 (-0.75 / +0.25)。

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::dense::DenseIndex;
use std::collections::HashMap;

pub const DEFAULT_GAMMA: f64 = 3.6;
pub const DEFAULT_BETA: f64 = 0.8;
pub const DEFAULT_PHASE_BOOST: f64 = 0.25;
pub const DEFAULT_COUNTERFACTUAL_DAMPING: f64 = -0.75;

const CAUSAL_POS_WORDS: &[&str] = &[
    "increase", "increases", "increased", "increasing",
    "promote", "promotes", "promoted", "promoting",
    "activate", "activates", "activated", "activating", "activation",
    "upregulate", "upregulates", "upregulated", "upregulation",
    "enhance", "enhances", "enhanced", "enhancing",
    "stimulate", "stimulates", "stimulated", "stimulation",
    "cause", "causes", "caused", "causing",
    "induce", "induces", "induced", "inducing", "induction",
    "improve", "improves", "improved", "improving",
    "accelerate", "accelerates", "accelerated",
];

const CAUSAL_NEG_WORDS: &[&str] = &[
    "decrease", "decreases", "decreased", "decreasing",
    "inhibit", "inhibits", "inhibited", "inhibiting", "inhibition",
    "suppress", "suppresses", "suppressed", "suppressing", "suppression",
    "downregulate", "downregulates", "downregulated", "downregulation",
    "reduce", "reduces", "reduced", "reducing", "reduction",
    "block", "blocks", "blocked", "blocking",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "attenuate", "attenuates", "attenuated", "attenuating",
    "impair", "impairs", "impaired", "impairment",
    "loss", "deplete", "depletes", "depleted", "depletion",
    "lack", "lacks", "lacked", "lacking",
];

fn detect_query_polarity(query: &str) -> i8 {
    let lower = query.to_ascii_lowercase();
    let mut pos = 0;
    let mut neg = 0;
    for word in lower.split(|c: char| !c.is_alphanumeric()) {
        if CAUSAL_POS_WORDS.contains(&word) {
            pos += 1;
        } else if CAUSAL_NEG_WORDS.contains(&word) {
            neg += 1;
        }
    }
    if pos > neg && pos >= 1 {
        1
    } else if neg > pos && neg >= 1 {
        -1
    } else {
        0
    }
}

pub struct CpfAdapter {
    info: AdapterInfo,
    lexical: crate::field_bm25::FieldBm25Adapter,
    dense_index: Option<DenseIndex>,
    query_embeddings: HashMap<String, Vec<f32>>,
    doc_polarities: HashMap<String, i8>,
    artifact_dir: std::path::PathBuf,
    gamma: f64,
    beta: f64,
    phase_boost: f64,
    counterfactual_damping: f64,
}

impl CpfAdapter {
    pub fn new() -> Result<Self> {
        let lexical = crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?;

        let artifact_dir = std::path::PathBuf::from("data/scifact/artifacts/dense-whitened-v1");
        let query_embeddings = Self::load_query_embeddings(&artifact_dir).unwrap_or_default();

        let pred_dir = std::path::Path::new("data/scifact/artifacts/predicate-v1");
        let doc_file = pred_dir.join("doc_polarities.json");
        let doc_polarities: HashMap<String, i8> = if doc_file.is_file() {
            let bytes = std::fs::read(doc_file)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let info = AdapterInfo::new(
            "cpf",
            "coherent-potential-field",
            "0.2.0",
            "PriorSeek 全息相干势能场算子引擎 (CPF: 连续流形相干 + 概念特异度饱和 + 命题完备性 + 因果相位相消)",
        )
        .with_config(serde_json::json!({
            "paradigm": "coherent_potential_field",
            "gamma": DEFAULT_GAMMA,
            "beta": DEFAULT_BETA,
            "phase_boost": DEFAULT_PHASE_BOOST,
            "counterfactual_damping": DEFAULT_COUNTERFACTUAL_DAMPING,
            "dense_route": "dense-whitened-v1",
            "lexical_route": "concept-synonym-v1",
        }));

        Ok(Self {
            info,
            lexical,
            dense_index: None,
            query_embeddings,
            doc_polarities,
            artifact_dir,
            gamma: DEFAULT_GAMMA,
            beta: DEFAULT_BETA,
            phase_boost: DEFAULT_PHASE_BOOST,
            counterfactual_damping: DEFAULT_COUNTERFACTUAL_DAMPING,
        })
    }

    fn load_query_embeddings(dir: &std::path::Path) -> Result<HashMap<String, Vec<f32>>> {
        let q_ids_path = dir.join("query_ids.txt");
        let q_emb_path = dir.join("query_embeddings.bin");

        if !q_ids_path.is_file() || !q_emb_path.is_file() {
            return Ok(HashMap::new());
        }

        let ids_content = std::fs::read_to_string(&q_ids_path)?;
        let q_ids: Vec<String> = ids_content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();

        let bin_bytes = std::fs::read(&q_emb_path)?;
        if bin_bytes.len() < 16 {
            anyhow::bail!("query_embeddings.bin 损坏");
        }

        let num_queries = u32::from_le_bytes(bin_bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bin_bytes[12..16].try_into().unwrap()) as usize;

        let float_bytes = &bin_bytes[16..];
        let mut map = HashMap::with_capacity(num_queries);
        let mut all_vectors = Vec::with_capacity(num_queries);

        for i in 0..num_queries {
            let offset = i * dim * 4;
            let slice = &float_bytes[offset..offset + dim * 4];
            let mut vec = Vec::with_capacity(dim);
            for chunk in slice.chunks_exact(4) {
                vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            all_vectors.push(vec);
        }

        for (q_id, vec) in q_ids.iter().zip(&all_vectors) {
            map.insert(q_id.clone(), vec.clone());
        }

        let q_texts_path = dir.join("query_texts.json");
        if q_texts_path.is_file() {
            if let Ok(bytes) = std::fs::read(&q_texts_path) {
                if let Ok(text_map) = serde_json::from_slice::<HashMap<String, usize>>(&bytes) {
                    for (text, idx) in text_map {
                        if let Some(vec) = all_vectors.get(idx) {
                            map.insert(text, vec.clone());
                        }
                    }
                }
            }
        }

        Ok(map)
    }
}

impl Adapter for CpfAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.lexical.index(docs)?;

        let dense_path = self.artifact_dir.join("dense.bin");
        let ids_path = self.artifact_dir.join("doc_ids.txt");

        if dense_path.is_file() && ids_path.is_file() {
            let ids_content = std::fs::read_to_string(&ids_path)
                .with_context(|| format!("读取 {:?}", ids_path))?;
            let doc_ids: Vec<String> = ids_content
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();

            let bin_bytes =
                std::fs::read(&dense_path).with_context(|| format!("读取 {:?}", dense_path))?;
            let index = DenseIndex::load_from_bytes(&bin_bytes, doc_ids)?;
            self.dense_index = Some(index);
        }

        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let dense = match self.dense_index.as_ref() {
            Some(d) => d,
            None => return self.lexical.search(query, k),
        };

        let query_vec = match self.query_embeddings.get(query) {
            Some(v) => v.as_slice(),
            None => return self.lexical.search(query, k),
        };

        // 1. 在线因果极性识别
        let q_pol = detect_query_polarity(query);

        // 2. 候选势场激发 (取前 60 篇词法与向量候选的并集，附带覆盖度)
        let candidate_pool_size = k.max(60);
        let lex_candidates = self.lexical.search_with_coverage(query, candidate_pool_size)?;
        let dense_candidates = dense.search(query_vec, candidate_pool_size);

        let mut candidate_map: HashMap<String, (f64, f64)> =
            HashMap::with_capacity(candidate_pool_size * 2);

        for (doc_id, score, cover_ratio) in lex_candidates {
            candidate_map.insert(doc_id, (score, cover_ratio));
        }

        for (doc_id, _) in dense_candidates {
            candidate_map.entry(doc_id).or_insert((0.0, 0.2));
        }

        // 3. 全息相干势能场积分求解 (CPF Total Energy Computation)
        // TotalEnergy(d) = gamma * <q, d> + ln(1.0 + phi_lex(d)) + beta * ln(0.2 + 0.8 * Cover(d)) + Phase(q, d)
        let mut resonant_candidates: Vec<(String, f64)> = Vec::with_capacity(candidate_map.len());

        for (doc_id, (lex_score, cover_ratio)) in candidate_map {
            let dot = dense.dot_with_id(&doc_id, query_vec).unwrap_or(0.0f32) as f64;
            let phi_lex = (lex_score.max(0.0) / 10.0).ln_1p();
            let penalty = (0.2 + 0.8 * cover_ratio).ln();

            // 因果相位相干项
            let d_pol = self.doc_polarities.get(&doc_id).copied().unwrap_or(0);
            let phase = if q_pol != 0 && d_pol != 0 {
                if q_pol == d_pol {
                    self.phase_boost // 同向相干增强 (+0.25)
                } else {
                    self.counterfactual_damping // 反事实强阻尼相消 (-0.75)
                }
            } else {
                0.0
            };

            let total_energy = self.gamma * dot + phi_lex + self.beta * penalty + phase;
            resonant_candidates.push((doc_id, total_energy));
        }

        // 4. 势能局部极值排序截断
        resonant_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        resonant_candidates.truncate(k);

        Ok(resonant_candidates)
    }
}
