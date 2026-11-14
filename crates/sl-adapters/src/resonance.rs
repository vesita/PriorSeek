//! 先验引导的语义流形共振检索适配器 (Prior-Guided Semantic Manifold Resonance Adapter)。
//!
//! 范式转移：彻底摒弃输出列表末端的离散投票 (RRF / 加权合并)，
//! 重构为高维流形上的“离线概念先验势场”与“连续语义场”的非线性相干共振算子。
//!
//! 物理数学模型：
//! Energy(d, q) = (1.0 + phi_lex(d, q)) * exp( gamma * phi_dense(d, q) ) * (0.2 + 0.8 * Cover(d, q))^beta
//! 在对数势能空间中计算：
//! LogScore(d, q) = gamma * <q_dense, d_dense> + ln(1.0 + phi_lex(d, q)) + beta * ln(0.2 + 0.8 * Cover(d, q))

use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::dense::DenseIndex;
use std::collections::HashMap;

/// 默认流形连续场耦合常数（平衡连续角动量与离散概念势能）
pub const DEFAULT_GAMMA: f64 = 3.6;

pub struct ResonanceAdapter {
    info: AdapterInfo,
    lexical: crate::field_bm25::FieldBm25Adapter,
    dense_index: Option<DenseIndex>,
    query_embeddings: HashMap<String, Vec<f32>>,
    artifact_dir: std::path::PathBuf,
    gamma: f64,
    beta: f64,
}

impl ResonanceAdapter {
    /// 基础先验流形共振算子 (beta = 0.0)
    pub fn new() -> Result<Self> {
        let lexical = crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?;

        let artifact_dir = std::path::PathBuf::from("data/scifact/artifacts/dense-v0");
        let query_embeddings = Self::load_query_embeddings(&artifact_dir).unwrap_or_default();

        let info = AdapterInfo::new(
            "resonance",
            "manifold-resonance",
            "0.2.0",
            "PriorSeek 语义流形共振算子 (相干场耦合: (1+phi_lex)*exp(gamma*phi_dense))",
        )
        .with_config(serde_json::json!({
            "paradigm": "coherent_manifold_resonance",
            "gamma": DEFAULT_GAMMA,
            "beta": 0.0,
            "lexical_route": "concept-synonym-v1",
            "dense_route": "dense-v0 (384-dim continuous manifold)",
        }));

        Ok(Self {
            info,
            lexical,
            dense_index: None,
            query_embeddings,
            artifact_dir,
            gamma: DEFAULT_GAMMA,
            beta: 0.0,
        })
    }

    /// 合取命题覆盖度相干流形共振算子 (beta = 0.8)
    pub fn conjunctive() -> Result<Self> {
        let mut adapter = Self::new()?;
        adapter.beta = 0.8;
        adapter.info = AdapterInfo::new(
            "resonance-conjunctive",
            "manifold-resonance",
            "0.2.0",
            "PriorSeek 合取命题覆盖度相干流形共振 (破除单专有名词垄断欺骗)",
        )
        .with_config(serde_json::json!({
            "paradigm": "conjunctive_manifold_resonance",
            "gamma": DEFAULT_GAMMA,
            "beta": 0.8,
            "lexical_route": "concept-synonym-v1 (with_coverage)",
            "dense_route": "dense-v0 (384-dim continuous manifold)",
        }));
        Ok(adapter)
    }

    /// 各向异性流形白化投影 + 合取命题覆盖度相干流形共振算子
    pub fn whitened() -> Result<Self> {
        let lexical = crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?;

        let artifact_dir = std::path::PathBuf::from("data/scifact/artifacts/dense-whitened-v1");
        let query_embeddings = Self::load_query_embeddings(&artifact_dir).unwrap_or_default();

        let info = AdapterInfo::new(
            "resonance-whitened",
            "manifold-resonance",
            "0.2.0",
            "PriorSeek 各向异性流形白化投影 + 合取覆盖共振算子 (SVD白化度规消解锥形各向异性)",
        )
        .with_config(serde_json::json!({
            "paradigm": "anisotropic_whitened_manifold_resonance",
            "gamma": DEFAULT_GAMMA,
            "beta": 0.8,
            "lexical_route": "concept-synonym-v1 (with_coverage)",
            "dense_route": "dense-whitened-v1 (SVD-whitened 384-dim continuous manifold)",
        }));

        Ok(Self {
            info,
            lexical,
            dense_index: None,
            query_embeddings,
            artifact_dir,
            gamma: DEFAULT_GAMMA,
            beta: 0.8,
        })
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
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

impl Adapter for ResonanceAdapter {
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

        // 阶段一：激发候选势场池 (取前 60~80 篇词法与向量候选的并集，附带覆盖度)
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

        // 阶段二：相干共振算子推演 (Resonance Operator Execution)
        // LogScore(d) = gamma * <q, d> + ln(1.0 + phi_lex(d)) + beta * ln(0.2 + 0.8 * Cover(d))
        let mut resonant_candidates: Vec<(String, f64)> = Vec::with_capacity(candidate_map.len());

        for (doc_id, (lex_score, cover_ratio)) in candidate_map {
            let dot = dense.dot_with_id(&doc_id, query_vec).unwrap_or(0.0f32) as f64;

            // 离散概念势能做非线性平滑
            let phi_lex = (lex_score.max(0.0) / 10.0).ln_1p();

            // 合取覆盖惩罚: ln(0.2 + 0.8 * cover)
            let penalty = if self.beta > 0.0 {
                self.beta * (0.2 + 0.8 * cover_ratio).ln()
            } else {
                0.0
            };

            let log_resonance = self.gamma * dot + phi_lex + penalty;

            resonant_candidates.push((doc_id, log_resonance));
        }

        // 阶段三：能量局部极值排序截断
        resonant_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        resonant_candidates.truncate(k);

        Ok(resonant_candidates)
    }
}

/// 谓词论元因果相干流形共振适配器 (Structural Resonance Adapter)
///
/// 核心理论：
/// 解决传统向量内积与词法在因果方向上的结构盲目性（A promotes B vs A inhibits B 余弦高度重合）。
/// 引入离线提取的因果谓词极性拓扑先验 (+1 诱导 / -1 抑制 / 0 中性)；
/// 当且仅当文档因果方向与查询命题一致时产生正向共振 (+0.15)，极性矛盾时触发反事实阻尼衰减 (-0.30)。
pub struct StructuralResonanceAdapter {
    inner: ResonanceAdapter,
    doc_polarities: HashMap<String, i8>,
    query_polarities: HashMap<String, i8>,
    text_to_qid: HashMap<String, String>,
}

impl StructuralResonanceAdapter {
    pub fn new() -> Result<Self> {
        let mut inner = ResonanceAdapter::whitened()?;
        inner.info = AdapterInfo::new(
            "structural-resonance",
            "structural-resonance",
            "0.2.0",
            "PriorSeek 谓词论元因果相干流形共振 (白化流形 + 合取覆盖 + 谓词因果极性相干)",
        )
        .with_config(serde_json::json!({
            "paradigm": "causal_structural_manifold_resonance",
            "gamma": DEFAULT_GAMMA,
            "beta": 0.8,
            "dense_route": "dense-whitened-v1",
            "predicate_route": "predicate-v1 (+1 induction / -1 suppression)",
        }));

        let pred_dir = std::path::Path::new("data/scifact/artifacts/predicate-v1");
        let doc_file = pred_dir.join("doc_polarities.json");
        let query_file = pred_dir.join("query_polarities.json");

        let doc_polarities: HashMap<String, i8> = if doc_file.is_file() {
            let bytes = std::fs::read(doc_file)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let query_polarities: HashMap<String, i8> = if query_file.is_file() {
            let bytes = std::fs::read(query_file)?;
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
            doc_polarities,
            query_polarities,
            text_to_qid,
        })
    }
}

impl Adapter for StructuralResonanceAdapter {
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

        let dense = match self.inner.dense_index.as_ref() {
            Some(d) => d,
            None => return self.inner.lexical.search(query, k),
        };

        let query_vec = match self.inner.query_embeddings.get(query) {
            Some(v) => v.as_slice(),
            None => return self.inner.lexical.search(query, k),
        };

        // 识别查询的因果极性先验
        let qid = self.text_to_qid.get(query).map(|s| s.as_str()).unwrap_or(query);
        let q_pol = self.query_polarities.get(qid).copied().unwrap_or(0);

        // 阶段一：激发候选势场池
        let candidate_pool_size = k.max(60);
        let lex_candidates = self.inner.lexical.search_with_coverage(query, candidate_pool_size)?;
        let dense_candidates = dense.search(query_vec, candidate_pool_size);

        let mut candidate_map: HashMap<String, (f64, f64)> =
            HashMap::with_capacity(candidate_pool_size * 2);

        for (doc_id, score, cover_ratio) in lex_candidates {
            candidate_map.insert(doc_id, (score, cover_ratio));
        }

        for (doc_id, _) in dense_candidates {
            candidate_map.entry(doc_id).or_insert((0.0, 0.2));
        }

        // 阶段二：相干共振算子推演 + 谓词因果极性相干调制
        let mut resonant_candidates: Vec<(String, f64)> = Vec::with_capacity(candidate_map.len());

        for (doc_id, (lex_score, cover_ratio)) in candidate_map {
            let dot = dense.dot_with_id(&doc_id, query_vec).unwrap_or(0.0f32) as f64;
            let phi_lex = (lex_score.max(0.0) / 10.0).ln_1p();
            let penalty = (0.2 + 0.8 * cover_ratio).ln();

            // 谓词结构相干增益 / 反事实阻尼相消
            let d_pol = self.doc_polarities.get(&doc_id).copied().unwrap_or(0);
            let delta_struct = if q_pol != 0 && d_pol != 0 {
                if q_pol == d_pol {
                    0.15 // 同向因果共振
                } else {
                    -0.30 // 反事实极性相消
                }
            } else {
                0.0
            };

            let log_resonance = self.inner.gamma * dot + phi_lex + self.inner.beta * penalty + delta_struct;
            resonant_candidates.push((doc_id, log_resonance));
        }

        resonant_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        resonant_candidates.truncate(k);

        Ok(resonant_candidates)
    }
}

