//! 因子消费外壳：把「同一次打分」暴露在两条**独立的**因子/选项轴上，便于单变量对照。
//!
//! | 轴 | 取值 | 消费的因子 | 会不会改变索引 |
//! |---|---|---|---|
//! | 概念归并 | `None` / `token → 规范形式` | `kind = concept` | 会（文档与查询 token 都先归并） |
//! | 字段加权 | `Off` / `Uniform(beta)` / `Factor{beta, conf}` | `kind = field` | 不会 |
//!
//! 两条轴的四种组合就是 `bm25` / `concept` / `uniform` / `concept+uniform`，其余变量全同。
//!
//! # 打分
//!
//! ```text
//! c(t)         = 概念归并后的规范形式（无映射时 c(t) = t）
//! tf_eff(t, d) = tf_full(c(t), d) + beta · conf_title(t) · tf_title(c(t), d)
//! score(d)     = Σ_t idf(c(t)) · tf_eff·(k1+1) / (tf_eff + k1·(1−b+b·len_full/avg_len))
//! ```
//!
//! 两条轴都关时与 `bm25` 适配器逐查询一致——由 `equivalent_to_bm25_without_factor`
//! 与 `concept_mode_with_empty_map_matches_bm25` 两条测试守着。
//!
//! 为什么是这个形状：`bm25` 把标题和正文拼成一个字段（`Doc::full_text`），于是「标题命中」
//! 与「正文命中」同价，且同一概念的不同形态（`cell` / `cells`）被当成两个独立词项。
//! 因子分别提供这两件事的事实，外壳只负责怎么用。

use crate::tokenize::bm25_terms;
use anyhow::Context;
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::index::PositionalIndex;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// 与 `bm25` 适配器同值（`sl-engine::bow::{K1, B}`）。
const K1: f64 = 1.5;
const B: f64 = 0.75;

/// 标题命中的放大方式。
///
/// 三者共用同一条打分路径，区别只在 `boost` 怎么来——这样对照实验是单变量的。
#[derive(Debug, Clone, PartialEq)]
pub enum FieldWeight {
    /// 不放大：等价于 `bm25`。
    Off,
    /// 统一放大：标题命中一律按 `beta` 放大，不依赖任何因子。
    /// 这是「字段信息本身值不值得用」的下界对照。
    Uniform(f64),
    /// 因子驱动：按 `beta · conf_title(t)` 放大，`conf` 来自内核产出的因子。
    Factor {
        beta: f64,
        conf: HashMap<String, f64>,
    },
}

impl FieldWeight {
    fn beta(&self) -> f64 {
        match self {
            Self::Off => 0.0,
            Self::Uniform(beta) => *beta,
            Self::Factor { beta, .. } => *beta,
        }
    }

    fn boost(&self, term: &str) -> f64 {
        match self {
            Self::Off => 0.0,
            Self::Uniform(beta) => *beta,
            Self::Factor { beta, conf } => beta * conf.get(term).copied().unwrap_or(0.0),
        }
    }

    fn factor_size(&self) -> usize {
        match self {
            Self::Factor { conf, .. } => conf.len(),
            _ => 0,
        }
    }
}

pub struct FieldBm25Adapter {
    info: AdapterInfo,
    /// 全文索引，词表与 `bm25` 同源（同一分词器、同一 idf 定义）。
    full: Arc<PositionalIndex>,
    /// 标题索引，与 `full` 共用同一套 token id。
    title: Arc<PositionalIndex>,
    weight: FieldWeight,
    vocabulary: HashMap<String, u32>,
    /// 概念归并：`token → 规范形式`。空 map = 不归并。
    ///
    /// 它同时作用于文档侧与查询侧——只映射一侧会让两侧词表错位，命中直接归零。
    concept: HashMap<String, String>,
    /// BM25+ 词频下界偏置（默认 0.0 即传统 BM25）。
    delta: f64,
    /// 是否开启双层谓词感知索引模式 (Tiered Predicate-Aware Index)
    tiered_mode: bool,
    /// Tier-1 核心谓词词表
    tier1_terms: std::collections::HashSet<u32>,
}

impl FieldBm25Adapter {
    /// 不放大标题命中、不归并概念：与 `bm25` 逐查询一致。
    pub fn new() -> Self {
        Self::with_weight(FieldWeight::Off, None)
    }

    /// 纯 BM25F：标题命中按统一 `beta` 放大。用于判断「字段信息本身」是否值得用。
    pub fn uniform(beta: f64) -> Self {
        Self::with_weight(FieldWeight::Uniform(beta), None)
    }

    /// 构造外壳。`factor` 为空时退化为 `FieldWeight::Off`，避免「有超参没因子」的静默放大。
    pub fn with_factor(
        factor: HashMap<String, f64>,
        beta: f64,
        factor_sha256: Option<String>,
    ) -> Self {
        let weight = if factor.is_empty() {
            FieldWeight::Off
        } else {
            FieldWeight::Factor { beta, conf: factor }
        };
        Self::with_weight(weight, factor_sha256)
    }

    /// 两条轴一起设：概念归并 + 字段加权方式。
    pub fn with_concept_and_weight(
        concept: HashMap<String, String>,
        concept_sha256: Option<String>,
        weight: FieldWeight,
    ) -> Self {
        let mut adapter = Self::with_weight(weight, None);
        adapter.concept = concept;
        let entries = adapter.concept.len();
        if entries > 0 {
            if let Some(config) = adapter.info.config.as_object_mut() {
                config.insert(
                    "prior_factor_concept".into(),
                    serde_json::json!({
                        "name": "token_concept",
                        "kind": "concept",
                        "entries": entries,
                        "sha256": concept_sha256,
                    }),
                );
            }
        }
        adapter
    }

    /// 设置 BM25+ 词频下界偏置 delta。
    pub fn with_delta(mut self, delta: f64) -> Self {
        if delta.is_finite() && delta >= 0.0 {
            self.delta = delta;
            if let Some(config) = self.info.config.as_object_mut() {
                config.insert("delta".into(), serde_json::json!(delta));
            }
        }
        self
    }

    /// 概念归并后的规范形式；没有映射时原样返回。
    fn map(&self, term: &str) -> String {
        match self.concept.get(term) {
            Some(canonical) => canonical.clone(),
            None => term.to_string(),
        }
    }

    fn with_weight(weight: FieldWeight, factor_sha256: Option<String>) -> Self {
        let delta: f64 = match std::env::var("SL_DELTA") {
            Ok(raw) => {
                let val: f64 = raw.trim().parse().unwrap_or(0.0);
                if val.is_finite() && val >= 0.0 {
                    val
                } else {
                    0.0
                }
            }
            Err(_) => 0.0,
        };
        let mut config = serde_json::json!({
            "k1": K1,
            "b": B,
            "beta": weight.beta(),
            "mode": match &weight {
                FieldWeight::Off => "off",
                FieldWeight::Uniform(_) => "uniform",
                FieldWeight::Factor { .. } => "factor",
            },
            "prior_factor": if weight.factor_size() == 0 {
                serde_json::Value::Null
            } else {
                serde_json::json!({
                    "name": "token_field",
                    "kind": "field",
                    "entries": weight.factor_size(),
                    "sha256": factor_sha256,
                })
            },
        });
        if delta > 0.0 {
            if let Some(obj) = config.as_object_mut() {
                obj.insert("delta".into(), serde_json::json!(delta));
            }
        }
        Self {
            info: AdapterInfo::new(
                "field-bm25",
                "factor-shell",
                "0.2.0",
                "外壳：crates/sl-adapters/src/field_bm25.rs（PositionalIndex + 字段 tf 加权）；因子：tools/kernel/build_field.py",
            )
            .with_config(config),
            full: Arc::new(PositionalIndex::default()),
            title: Arc::new(PositionalIndex::default()),
            weight,
            vocabulary: HashMap::new(),
            concept: HashMap::new(),
            delta,
            tiered_mode: false,
            tier1_terms: std::collections::HashSet::new(),
        }
    }
}

/// 从工件目录读取因子：校验 `KernelArtifact` 准入、`prior_factor` 契约与 payload 摘要。
///
/// 返回 `(token → 标题出现率, payload sha256)`。供 `from_env` 与评测脚本共用，
/// 保证两条路径读的是同一份数据、走同一套校验。
pub fn load_factor_dir(dir: &std::path::Path) -> anyhow::Result<(HashMap<String, f64>, String)> {
    let artifact = sl_core::KernelArtifact::from_dir(dir)?;
    let spec = artifact
        .prior_factor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("工件 {} 没有登记 prior_factor", dir.display()))?;
    if spec.kind != sl_core::FactorKind::Field {
        anyhow::bail!(
            "工件 {} 的 prior_factor.kind = {:?}；field-bm25 只消费 kind = field 的因子",
            dir.display(),
            spec.kind
        );
    }

    let payload_path = dir.join("fields.json");
    let payload = std::fs::read(&payload_path).with_context(|| {
        format!(
            "读取 {} 失败（先跑 tools/kernel/build_field.py）",
            payload_path.display()
        )
    })?;
    let expected = artifact
        .files
        .get("fields.json")
        .ok_or_else(|| anyhow::anyhow!("工件 {} 没有登记 fields.json 的摘要", dir.display()))?;
    let actual = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&payload);
        format!("{:x}", hasher.finalize())
    };
    if actual != *expected {
        anyhow::bail!(
            "{} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）——因子被改过",
            payload_path.display()
        );
    }

    let parsed: serde_json::Value = serde_json::from_slice(&payload)
        .with_context(|| format!("解析 {} 失败", payload_path.display()))?;
    let mut factor = HashMap::new();
    if let Some(entries) = parsed.get("entries").and_then(|value| value.as_object()) {
        for (token, rows) in entries {
            let Some(rows) = rows.as_array() else {
                continue;
            };
            for row in rows {
                let Some(pair) = row.as_array() else { continue };
                if pair.first().and_then(|value| value.as_str()) != Some("title") {
                    continue;
                }
                if let Some(conf) = pair.get(1).and_then(|value| value.as_f64()) {
                    factor.insert(token.clone(), conf);
                }
            }
        }
    }
    if factor.is_empty() {
        anyhow::bail!("工件 {} 的 fields.json 没有任何 title 条目", dir.display());
    }
    Ok((factor, actual))
}

/// 读工件里的一份 payload 并对照 manifest 校验摘要。返回 `(内容, 实际 sha256)`。
///
/// 所有因子读取都走这里：摘要不符就报错，不能「读到了就算数」。
fn read_verified_payload(
    dir: &std::path::Path,
    artifact: &sl_core::KernelArtifact,
    file_name: &str,
) -> anyhow::Result<(Vec<u8>, String)> {
    let path = dir.join(file_name);
    let payload = std::fs::read(&path).with_context(|| {
        format!(
            "读取 {} 失败（先跑 tools/kernel/build_*.py）",
            path.display()
        )
    })?;
    let expected = artifact
        .files
        .get(file_name)
        .ok_or_else(|| anyhow::anyhow!("工件 {} 没有登记 {file_name} 的摘要", dir.display()))?;
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&payload);
    let actual = format!("{:x}", hasher.finalize());
    if actual != *expected {
        anyhow::bail!(
            "{} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）——因子被改过",
            path.display()
        );
    }
    Ok((payload, actual))
}

/// 从工件目录读取 `kind = concept` 的因子：`token → 规范形式`。
///
/// 载荷形态与 field / expansion 同构（`{"entries": {token: [[目标, 置信度]]}}`），
/// 因此外壳只多一条查表路径，不多一套解析。置信度目前恒为 1.0（硬归并）；
/// 外壳只取第一个目标，取几个、要不要按置信度加权属于外壳超参。
pub fn load_concept_dir(
    dir: &std::path::Path,
) -> anyhow::Result<(HashMap<String, String>, String)> {
    let artifact = sl_core::KernelArtifact::from_dir(dir)?;
    let spec = artifact
        .prior_factor
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("工件 {} 没有登记 prior_factor", dir.display()))?;
    if spec.kind != sl_core::FactorKind::Concept {
        anyhow::bail!(
            "工件 {} 的 prior_factor.kind = {:?}；概念归并只消费 kind = concept 的因子",
            dir.display(),
            spec.kind
        );
    }

    let (payload, sha256) = read_verified_payload(dir, &artifact, "concepts.json")?;
    let parsed: serde_json::Value = serde_json::from_slice(&payload)
        .with_context(|| format!("解析 {}/concepts.json 失败", dir.display()))?;
    let mut concept = HashMap::new();
    if let Some(entries) = parsed.get("entries").and_then(|value| value.as_object()) {
        for (token, rows) in entries {
            let Some(rows) = rows.as_array() else {
                continue;
            };
            // 硬归并：只取第一个目标。软归并（多目标 + 置信度）留待外壳有明确用法时再接。
            if let Some(canonical) = rows
                .first()
                .and_then(|row| row.as_array())
                .and_then(|pair| pair.first())
                .and_then(|value| value.as_str())
            {
                concept.insert(token.clone(), canonical.to_string());
            }
        }
    }
    if concept.is_empty() {
        anyhow::bail!("工件 {} 的 concepts.json 没有任何条目", dir.display());
    }
    Ok((concept, sha256))
}

/// `concept-bm25` 的因子来源：`SL_CONCEPT_ARTIFACT` > `SL_DATA_DIR`/`SL_DATASET`/artifacts/concept-v0。
fn resolve_concept_dir() -> anyhow::Result<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("SL_CONCEPT_ARTIFACT") {
        return Ok(std::path::PathBuf::from(dir));
    }
    let data_dir = std::env::var("SL_DATA_DIR").unwrap_or_else(|_| "data".to_string());
    let dataset = std::env::var("SL_DATASET").map_err(|_| {
        anyhow::anyhow!(
            "concept-bm25 需要概念因子工件：设置 SL_CONCEPT_ARTIFACT，或由 `sl run` 提供 SL_DATA_DIR/SL_DATASET。\n构建工件：uv run python tools/kernel/build_concept.py <dataset>"
        )
    })?;
    Ok(std::path::Path::new(&data_dir)
        .join(dataset)
        .join("artifacts")
        .join("concept-v0"))
}

impl FieldBm25Adapter {
    /// 从 `<dataset_dir>/artifacts/<id>` 读取概念归并因子。
    pub fn load_concept_named(
        dataset_dir: &std::path::Path,
        artifact_id: &str,
    ) -> anyhow::Result<(HashMap<String, String>, String)> {
        load_concept_dir(&dataset_dir.join("artifacts").join(artifact_id))
    }

    /// `concept-bm25`：概念归并 + 可选字段加权（`SL_FIELD_BETA`，默认 0）。
    pub fn from_env_concept() -> anyhow::Result<Self> {
        let (concept, sha256) = load_concept_dir(&resolve_concept_dir()?)?;
        let beta = match std::env::var("SL_FIELD_BETA") {
            Ok(raw) => {
                let value: f64 = raw
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("SL_FIELD_BETA 不是数字：{raw:?}"))?;
                if !value.is_finite() || value < 0.0 {
                    anyhow::bail!("SL_FIELD_BETA 必须是非负有限数：{raw:?}");
                }
                value
            }
            Err(_) => 0.0,
        };
        let weight = if beta > 0.0 {
            FieldWeight::Uniform(beta)
        } else {
            FieldWeight::Off
        };
        let mut adapter = Self::with_concept_and_weight(concept, Some(sha256), weight);
        adapter.info.name = "concept-bm25".to_string();
        adapter.info.family = "factor-shell".to_string();
        Ok(adapter)
    }

    /// 按指定工件 ID 构造概念归并适配器。
    pub fn from_named_concept(artifact_id: &str, adapter_name: &str) -> anyhow::Result<Self> {
        let data_dir = std::env::var("SL_DATA_DIR").unwrap_or_else(|_| "data".to_string());
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let dir = std::path::Path::new(&data_dir)
            .join(dataset)
            .join("artifacts")
            .join(artifact_id);
        let (concept, sha256) = load_concept_dir(&dir)?;
        let mut adapter = Self::with_concept_and_weight(concept, Some(sha256), FieldWeight::Off);
        adapter.info.name = adapter_name.to_string();
        adapter.info.family = "factor-shell".to_string();
        Ok(adapter)
    }

    /// 双层分层加速版同义概念适配器 (Tiered Predicate Index)。
    pub fn tiered_concept() -> anyhow::Result<Self> {
        let mut adapter = Self::from_named_concept("concept-synonym-v1", "tiered-concept")?;
        adapter.tiered_mode = true;
        if let Some(config) = adapter.info.config.as_object_mut() {
            config.insert("tiered_mode".into(), serde_json::json!(true));
        }
        Ok(adapter)
    }
}

impl FieldBm25Adapter {
    /// 从 `<dataset_dir>/artifacts/field-v0` 读取因子，供评测脚本在同一进程内扫多个 beta。
    pub fn load_factor(
        dataset_dir: &std::path::Path,
    ) -> anyhow::Result<(HashMap<String, f64>, String)> {
        Self::load_factor_named(dataset_dir, "field-v0")
    }
    /// 按工件 id 读取因子：对照不同 `conf` 定义时要用不同的工件。
    pub fn load_factor_named(
        dataset_dir: &std::path::Path,
        artifact_id: &str,
    ) -> anyhow::Result<(HashMap<String, f64>, String)> {
        load_factor_dir(&dataset_dir.join("artifacts").join(artifact_id))
    }

    /// 从工件目录加载因子并构造外壳。
    ///
    /// 路径解析同 `seq`：`SL_FIELD_ARTIFACT` 优先，否则 `SL_DATA_DIR`/`SL_DATASET`/artifacts/field-v0。
    /// 工件必须通过 `KernelArtifact` 的准入校验（含 `prior_factor` 契约），且 kind 必须是 `field`。
    pub fn from_env() -> anyhow::Result<Self> {
        let dir = resolve_artifact_dir()?;
        let (factor, sha256) = load_factor_dir(&dir)?;

        let beta = match std::env::var("SL_FIELD_BETA") {
            Ok(raw) => {
                let value: f64 = raw
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("SL_FIELD_BETA 不是数字：{raw:?}"))?;
                if !value.is_finite() || value < 0.0 {
                    anyhow::bail!("SL_FIELD_BETA 必须是非负有限数：{raw:?}");
                }
                value
            }
            Err(_) => 1.0,
        };

        Ok(Self::with_factor(factor, beta, Some(sha256)))
    }

    /// 高级检索：在返回 BM25 分数的同时，返回文档对查询词项的覆盖比例 `[0.0, 1.0]`
    pub fn search_with_coverage(
        &mut self,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<(String, f64, f64)>> {
        let query_terms: BTreeSet<String> = bm25_terms(query)
            .into_iter()
            .map(|term| self.map(&term))
            .collect();
        if query_terms.is_empty() || self.full.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let num_terms = query_terms.len() as f64;
        let avg_len = self.full.avg_len();
        let mut scores: Vec<f64> = vec![0.0; self.full.len()];
        let mut hit_counts: Vec<u16> = vec![0; self.full.len()];

        // 双层谓词感知索引预过滤
        let core_filter: Option<std::collections::HashSet<u32>> =
            if self.tiered_mode && query_terms.len() >= 2 {
                let mut core_ids = Vec::new();
                for term in &query_terms {
                    if let Some(&id) = self.vocabulary.get(term) {
                        if self.tier1_terms.contains(&id) {
                            core_ids.push(id);
                        }
                    }
                }
                if !core_ids.is_empty() {
                    Some(self.full.candidates(&core_ids).into_iter().collect())
                } else {
                    None
                }
            } else {
                None
            };

        let mut touched: Vec<u32> = Vec::new();
        for term in &query_terms {
            let Some(&term_id) = self.vocabulary.get(term) else {
                continue;
            };
            let idf = self.full.idf(term_id);
            if idf <= 0.0 {
                continue;
            }
            let boost = self.weight.boost(term);
            let occurrences = self.full.occurrences(term_id);
            let mut cursor = 0usize;
            while cursor < occurrences.len() {
                let doc = occurrences[cursor].doc;
                let start = cursor;
                while cursor < occurrences.len() && occurrences[cursor].doc == doc {
                    cursor += 1;
                }

                if let Some(filter) = &core_filter {
                    if !filter.contains(&doc) {
                        continue;
                    }
                }

                let tf_full = (cursor - start) as f64;
                let tf_eff = if boost > 0.0 {
                    let tf_title = self.title.positions_in(term_id, doc).len() as f64;
                    tf_full + boost * tf_title
                } else {
                    tf_full
                };
                let index = doc as usize;
                if scores[index] == 0.0 {
                    touched.push(doc);
                }
                hit_counts[index] += 1;
                let denom = tf_eff + K1 * (1.0 - B + B * self.full.doc_len(doc) as f64 / avg_len);
                let tf_comp = tf_eff * (K1 + 1.0) / denom + self.delta;
                scores[index] += idf * tf_comp;
            }
        }

        let mut ranked: Vec<(u32, f64, f64)> = touched
            .iter()
            .filter(|&&doc| scores[doc as usize] > 0.0)
            .map(|&doc| {
                let idx = doc as usize;
                (
                    doc,
                    scores[idx],
                    (hit_counts[idx] as f64 / num_terms).min(1.0),
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.full.doc_id(a.0).cmp(self.full.doc_id(b.0)))
        });
        ranked.truncate(k);
        Ok(ranked
            .into_iter()
            .map(|(doc, score, cov)| (self.full.doc_id(doc).to_string(), score, cov))
            .collect())
    }
}

/// 工件目录解析：`SL_FIELD_ARTIFACT` > `SL_DATA_DIR`/`SL_DATASET`/artifacts/field-v0。
fn resolve_artifact_dir() -> anyhow::Result<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("SL_FIELD_ARTIFACT") {
        return Ok(std::path::PathBuf::from(dir));
    }
    let data_dir = std::env::var("SL_DATA_DIR").unwrap_or_else(|_| "data".to_string());
    let dataset = std::env::var("SL_DATASET").map_err(|_| {
        anyhow::anyhow!(
            "field-bm25 需要因子工件：设置 SL_FIELD_ARTIFACT，或由 `sl run` 提供 SL_DATA_DIR/SL_DATASET。\n构建工件：uv run python tools/kernel/build_field.py <dataset>"
        )
    })?;
    Ok(std::path::Path::new(&data_dir)
        .join(dataset)
        .join("artifacts")
        .join("field-v0"))
}

impl Default for FieldBm25Adapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Adapter for FieldBm25Adapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        self.vocabulary.clear();

        // 词表与全文 token 一次建好：与 `bm25` 用同一个分词器、同一套 id 分配顺序。
        let mut full_tokens: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        let mut title_tokens: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        for doc in docs {
            let tokens = bm25_terms(&doc.full_text());
            let ids: Vec<u32> = tokens
                .iter()
                .map(|term| {
                    let term = self.map(term);
                    let next = u32::try_from(self.vocabulary.len()).expect("词表超过 u32::MAX");
                    *self.vocabulary.entry(term).or_insert(next)
                })
                .collect();
            // 标题 token 复用同一词表；标题是 full_text 的一部分，理论上不会 OOV，
            // 真的有就跳过——静默补 id 会让两个索引的词表分叉。
            let title_ids: Vec<u32> = bm25_terms(&doc.title)
                .iter()
                .filter_map(|term| self.vocabulary.get(&self.map(term)).copied())
                .collect();
            full_tokens.push((doc.id.clone(), ids));
            title_tokens.push((doc.id.clone(), title_ids));
        }

        self.full = Arc::new(PositionalIndex::build(full_tokens));
        self.title = Arc::new(PositionalIndex::build(title_tokens));

        self.tier1_terms.clear();
        if self.tiered_mode {
            let max_bg_df = (self.full.len() / 3).max(10) as u32;
            for &id in self.vocabulary.values() {
                let df = self.full.doc_freq(id);
                if df > 0 && df <= max_bg_df {
                    self.tier1_terms.insert(id);
                }
            }
        }
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let with_cov = self.search_with_coverage(query, k)?;
        Ok(with_cov
            .into_iter()
            .map(|(doc, score, _cov)| (doc, score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::Bm25Adapter;

    fn titled(id: &str, title: &str, text: &str) -> Doc {
        Doc {
            id: id.into(),
            title: title.into(),
            text: text.into(),
        }
    }

    fn corpus() -> Vec<Doc> {
        vec![
            titled("d1", "apple pie", "banana apple"),
            titled("d2", "banana bread", "apple cherry"),
            titled("d3", "cherry tart", "banana cherry cherry"),
        ]
    }

    /// 无因子时必须与 `bm25` 逐查询一致：这条守着「因子没接上就等于没做」。
    #[test]
    fn equivalent_to_bm25_without_factor() {
        let docs = corpus();
        let mut baseline = Bm25Adapter::new();
        baseline.index(&docs).unwrap();
        let mut shell = FieldBm25Adapter::new();
        shell.index(&docs).unwrap();

        for query in ["apple", "banana cherry", "apple cherry pie", "zzz"] {
            let expected = baseline.search(query, 10).unwrap();
            let actual = shell.search(query, 10).unwrap();
            assert_eq!(expected.len(), actual.len(), "查询 {query} 命中数不同");
            for ((bid, bscore), (fid, fscore)) in expected.iter().zip(&actual) {
                assert_eq!(bid, fid, "查询 {query} 排序不同");
                assert!(
                    (bscore - fscore).abs() < 1e-12,
                    "查询 {query} 分数不同：{bscore} vs {fscore}"
                );
            }
        }
    }

    /// 因子把标题命中放大：同样 tf、同样长度，标题命中的文档应当排到前面。
    #[test]
    fn factor_boosts_title_matches() {
        let docs = vec![
            titled("d1", "apple", "banana"),
            titled("d2", "banana", "apple"),
        ];
        let mut neutral = FieldBm25Adapter::new();
        neutral.index(&docs).unwrap();
        let flat = neutral.search("apple", 10).unwrap();
        assert!(
            (flat[0].1 - flat[1].1).abs() < 1e-12,
            "无因子时两者应同分，实际 {} vs {}",
            flat[0].1,
            flat[1].1
        );

        let mut factor = HashMap::new();
        factor.insert("apple".to_string(), 1.0);
        let mut boosted = FieldBm25Adapter::with_factor(factor, 2.0, None);
        boosted.index(&docs).unwrap();
        let hits = boosted.search("apple", 10).unwrap();
        assert_eq!(hits[0].0, "d1", "标题命中的文档应排第一");
        assert!(
            hits[0].1 > hits[1].1,
            "标题命中的分数应更高：{} vs {}",
            hits[0].1,
            hits[1].1
        );
    }

    /// 因子没登记的 token 不能改变分数：否则就是「因子顺带改了别的算法」。
    #[test]
    fn unregistered_terms_are_unaffected() {
        let docs = corpus();
        let mut factor = HashMap::new();
        factor.insert("apple".to_string(), 0.5);
        let mut shell = FieldBm25Adapter::with_factor(factor, 3.0, None);
        shell.index(&docs).unwrap();
        let mut baseline = Bm25Adapter::new();
        baseline.index(&docs).unwrap();

        let expected = baseline.search("banana cherry", 10).unwrap();
        let actual = shell.search("banana cherry", 10).unwrap();
        for ((bid, bscore), (fid, fscore)) in expected.iter().zip(&actual) {
            assert_eq!(bid, fid);
            assert!((bscore - fscore).abs() < 1e-12);
        }
    }

    /// 因子条目为空时 beta 被强制归零，避免「有超参没因子」的静默放大。
    #[test]
    fn empty_factor_forces_zero_beta() {
        let adapter = FieldBm25Adapter::with_factor(HashMap::new(), 5.0, None);
        assert_eq!(
            adapter.info().config.get("beta").and_then(|v| v.as_f64()),
            Some(0.0),
            "因子为空时 beta 必须归零，否则就是「有超参没因子」的静默放大"
        );
        assert_eq!(
            adapter.info().config.get("mode").and_then(|v| v.as_str()),
            Some("off")
        );
    }

    /// 纯 BM25F 模式：不依赖因子，标题命中一律按 beta 放大。
    #[test]
    fn uniform_mode_boosts_every_title_hit() {
        let docs = vec![
            titled("d1", "apple", "banana"),
            titled("d2", "banana", "apple"),
        ];
        let mut shell = FieldBm25Adapter::uniform(2.0);
        shell.index(&docs).unwrap();
        let hits = shell.search("apple", 10).unwrap();
        assert_eq!(hits[0].0, "d1");
        assert!(hits[0].1 > hits[1].1);
        assert_eq!(
            shell.info().config.get("mode").and_then(|v| v.as_str()),
            Some("uniform")
        );
    }

    /// 概念映射为空时必须与 `bm25` 逐查询一致：归并没接上就等于没做。
    #[test]
    fn concept_mode_with_empty_map_matches_bm25() {
        let docs = corpus();
        let mut baseline = Bm25Adapter::new();
        baseline.index(&docs).unwrap();
        let mut shell =
            FieldBm25Adapter::with_concept_and_weight(HashMap::new(), None, FieldWeight::Off);
        shell.index(&docs).unwrap();

        for query in ["apple", "banana cherry", "apple cherry pie", "zzz"] {
            let expected = baseline.search(query, 10).unwrap();
            let actual = shell.search(query, 10).unwrap();
            assert_eq!(expected.len(), actual.len(), "查询 {query} 命中数不同");
            for ((bid, bscore), (fid, fscore)) in expected.iter().zip(&actual) {
                assert_eq!(bid, fid, "查询 {query} 排序不同");
                assert!(
                    (bscore - fscore).abs() < 1e-12,
                    "查询 {query} 分数不同：{bscore} vs {fscore}"
                );
            }
        }
    }

    /// 归并后「查询用变体、文档用原形」也能命中，且同一概念的词频合并计入 tf。
    #[test]
    fn concept_merge_makes_variants_equivalent() {
        let docs = vec![titled("d1", "", "cell"), titled("d2", "", "banana")];
        let mut plain = FieldBm25Adapter::new();
        plain.index(&docs).unwrap();
        assert!(
            plain.search("cells", 10).unwrap().is_empty(),
            "不归并时 cells 不应命中任何文档"
        );

        let mut concept = HashMap::new();
        concept.insert("cells".to_string(), "cell".to_string());
        let mut merged = FieldBm25Adapter::with_concept_and_weight(concept, None, FieldWeight::Off);
        merged.index(&docs).unwrap();
        let hits = merged.search("cells", 10).unwrap();
        assert_eq!(hits.len(), 1, "归并后 cells 应命中 d1");
        assert_eq!(hits[0].0, "d1");
    }

    /// 真实工件的加载路径（`from_env_concept`）：工件不在时跳过，在时必须通过准入校验。
    ///
    /// 这条测试守的是「注册表里能构造出来」——`concept-bm25` 只经 `sl run` 走到这里，
    /// 没有这条测试，那条路径就是无人走过的死代码。
    #[test]
    fn concept_adapter_loads_real_artifact_from_env() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/concept-v0");
        if !dir.join("manifest.json").is_file() {
            eprintln!("跳过：找不到 {}", dir.display());
            return;
        }
        std::env::set_var("SL_CONCEPT_ARTIFACT", dir);
        let adapter = FieldBm25Adapter::from_env_concept().expect("加载 concept 工件");
        std::env::remove_var("SL_CONCEPT_ARTIFACT");

        assert_eq!(adapter.info().name, "concept-bm25");
        assert_eq!(adapter.info().family, "factor-shell");
        let factor = adapter
            .info()
            .config
            .get("prior_factor_concept")
            .expect("config 必须登记概念因子");
        assert_eq!(factor["kind"], "concept");
        assert!(
            factor["entries"].as_u64().unwrap_or(0) > 0,
            "因子条目数应大于 0：{factor}"
        );
    }

    /// 文档侧的形态也被归并：d1 的 `cells` 与 d2 的 `cell` 落在同一词项上。
    #[test]
    fn concept_merge_applies_to_documents_too() {
        let docs = vec![titled("d1", "", "cells cells"), titled("d2", "", "cell")];
        let mut concept = HashMap::new();
        concept.insert("cells".to_string(), "cell".to_string());
        let mut merged = FieldBm25Adapter::with_concept_and_weight(concept, None, FieldWeight::Off);
        merged.index(&docs).unwrap();
        let hits = merged.search("cell", 10).unwrap();
        assert_eq!(hits.len(), 2, "归并后两篇文档都应命中 cell");
        assert_eq!(hits[0].0, "d1", "重复两次的那篇分数应更高");
    }
}
