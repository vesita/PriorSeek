//! 三元组因果拓扑检索适配器 (Triplet Causal Topology Adapter)。
//!
//! 核心范式转移：
//! 从“连续统计词频”走向“离散因果拓扑演绎”。
//! 在高维流形共振的基础上，挂载 (Subject, Relation, Object) 因果拓扑图：
//! 1. 同向事实证明：关系谓词与实体对完全一致，激发逻辑证明增益 (+0.40)；
//! 2. 反事实因果绝缘：关系谓词与实体对直接对立冲突，实施物理级相干绝缘截断 (-2.50)！

use anyhow::Result;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::HashMap;

pub struct TripletCausalAdapter {
    inner: crate::resonance::ResonanceAdapter,
    /// 实体对 -> [(doc_id, relation)]
    entity_pair_map: HashMap<(String, String), Vec<(String, String)>>,
    /// qid -> [S, R, O]
    query_triplets: HashMap<String, Vec<String>>,
    text_to_qid: HashMap<String, String>,
}

impl TripletCausalAdapter {
    pub fn new() -> Result<Self> {
        let mut inner = crate::resonance::ResonanceAdapter::whitened()?;
        inner.info = AdapterInfo::new(
            "triplet-causal",
            "triplet-causal",
            "0.2.0",
            "PriorSeek 三元组因果拓扑检索 (白化流形共振 + (S,R,O) 逻辑因果拓扑绝缘截断)",
        )
        .with_config(serde_json::json!({
            "paradigm": "causal_triplet_topology_deduction",
            "gamma": crate::resonance::DEFAULT_GAMMA,
            "beta": 0.8,
            "proof_boost": 0.40,
            "counterfactual_insulation": -2.50,
            "triplets_artifact": "triplets-v1",
        }));

        let trip_dir = std::path::Path::new("data/scifact/artifacts/triplets-v1");
        let corpus_file = trip_dir.join("corpus_triplets.json");
        let query_file = trip_dir.join("query_triplets.json");

        // 构建实体对因果图倒排
        let mut entity_pair_map: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
        if corpus_file.is_file() {
            let bytes = std::fs::read(&corpus_file)?;
            let raw_corpus: HashMap<String, Vec<Vec<String>>> = serde_json::from_slice(&bytes)?;
            for (doc_id, triplets) in raw_corpus {
                for t in triplets {
                    if t.len() == 3 {
                        let subj = t[0].clone();
                        let rel = t[1].clone();
                        let obj = t[2].clone();
                        entity_pair_map
                            .entry((subj, obj))
                            .or_default()
                            .push((doc_id.clone(), rel));
                    }
                }
            }
        }

        let query_triplets: HashMap<String, Vec<String>> = if query_file.is_file() {
            let bytes = std::fs::read(&query_file)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let mut text_to_qid = HashMap::new();
        let queries_file = std::path::Path::new("data/scifact/queries.jsonl");
        if queries_file.is_file() {
            if let Ok(content) = std::fs::read_to_string(queries_file) {
                for line in content.lines() {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                        if let (Some(id), Some(text)) = (val["id"].as_str(), val["text"].as_str()) {
                            text_to_qid.insert(text.to_string(), id.to_string());
                        }
                    }
                }
            }
        }

        Ok(Self {
            inner,
            entity_pair_map,
            query_triplets,
            text_to_qid,
        })
    }
}

impl Adapter for TripletCausalAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.inner.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.inner.index(docs)
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        if self.inner.dense_index.is_none() {
            return self.inner.lexical.search(query, k);
        }
        let query_vec = match self.inner.query_embeddings.get(query) {
            Some(v) => v.clone(),
            None => return self.inner.lexical.search(query, k),
        };

        // 查找查询的因果三元组
        let qid = self.text_to_qid.get(query).map(|s| s.as_str()).unwrap_or(query);
        let q_triplet = self.query_triplets.get(qid).cloned();

        // 阶段一：激发候选势场池
        let candidate_pool_size = k.max(60);
        let lex_candidates = self.inner.lexical.search_with_coverage(query, candidate_pool_size)?;
        let dense = self.inner.dense_index.as_ref().unwrap();
        let dense_candidates = dense.search(&query_vec, candidate_pool_size);

        let mut candidate_map: HashMap<String, (f64, f64)> =
            HashMap::with_capacity(candidate_pool_size * 2);

        for (doc_id, score, cover_ratio) in lex_candidates {
            candidate_map.insert(doc_id, (score, cover_ratio));
        }

        for (doc_id, _) in dense_candidates {
            candidate_map.entry(doc_id).or_insert((0.0, 0.2));
        }

        // 阶段二：流形共振积分 + 逻辑级三元组因果绝缘/证明算子
        let mut resonant_candidates: Vec<(String, f64)> = Vec::with_capacity(candidate_map.len());

        for (doc_id, (lex_score, cover_ratio)) in candidate_map {
            let dot = dense.dot_with_id(&doc_id, &query_vec).unwrap_or(0.0f32) as f64;
            let phi_lex = (lex_score.max(0.0) / 10.0).ln_1p();
            let penalty = (0.2 + 0.8 * cover_ratio).ln();

            // 核心逻辑因果判定
            let mut delta_triplet = 0.0;
            if let Some(t) = &q_triplet {
                let s_q = &t[0];
                let r_q = &t[1];
                let o_q = &t[2];

                // 查文档是否记载了该 (S, O) 实体对
                if let Some(doc_relations) = self.entity_pair_map.get(&(s_q.clone(), o_q.clone())) {
                    for (d_id, r_d) in doc_relations {
                        if d_id == &doc_id {
                            if r_d == r_q {
                                // 逻辑完全同向证实！
                                delta_triplet += 0.40;
                            } else if (r_q == "PROMOTES" && r_d == "INHIBITS")
                                || (r_q == "INHIBITS" && r_d == "PROMOTES")
                            {
                                // 反事实对立冲突，实施物理级绝缘相消！
                                delta_triplet -= 2.50;
                            }
                        }
                    }
                }
            }

            let log_energy = self.inner.gamma * dot + phi_lex + self.inner.beta * penalty + delta_triplet;
            resonant_candidates.push((doc_id, log_energy));
        }

        // 阶段三：能量局部极值排序截断
        resonant_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        resonant_candidates.truncate(k);

        Ok(resonant_candidates)
    }
}
