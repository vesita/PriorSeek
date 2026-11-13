//! 外壳适配器：把 `sl-engine` 的索引与打分器接成 `sl_core::Adapter`。
//!
//! 两个名字：
//! - `bow`：词袋端点（BM25 数学），用于**等价性门 G1** —— 必须与既有 `bm25` 适配器逐查询一致，
//!   证明外壳是忠实的；
//! - `oac`：本仓主算法「有序对齐覆盖」。
//!
//! tokenizer 由 `SL_SHELL_TOKENIZER` 选择（`word` 默认 / `qwen`）。切分不同、索引与算法完全相同 ——
//! 这正是外壳存在的意义：**换 tokenizer 只是换一个变量，不是换一套实现**。

use crate::tokenize::bm25_terms;
use crate::tokenizer_llm::LlmTokenizer;
use anyhow::Context;
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::{BowScorer, KernelGapTable, OacParams, OacScorer, PositionalIndex, WeightedTerm};
use std::collections::HashMap;
use std::sync::Arc;

/// 切分方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerChoice {
    /// 词级（`tokenize::bm25_terms`），token 经本地字典映射成 id。
    Word,
    /// LLM BPE（Qwen2.5），token id 直接用。
    Qwen,
}

impl TokenizerChoice {
    fn from_env() -> anyhow::Result<Self> {
        match std::env::var("SL_SHELL_TOKENIZER")
            .unwrap_or_else(|_| "word".to_string())
            .as_str()
        {
            "word" => Ok(Self::Word),
            "qwen" | "qwen2.5-0.5b" | "bpe" => Ok(Self::Qwen),
            other => anyhow::bail!("SL_SHELL_TOKENIZER 只接受 word / qwen，收到 {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Word => "word(bm25_terms)",
            Self::Qwen => "qwen2.5-0.5b(bpe)",
        }
    }
}

/// 把文本切成 token id。词级走本地字典；BPE 直接用 tokenizer 的 id。
enum Tokenizer {
    Word(HashMap<String, u32>),
    /// 装箱：`LlmTokenizer` 比词级字典大得多，不装箱会让整个枚举的每个值都很大。
    Qwen(Box<LlmTokenizer>),
}

impl Tokenizer {
    fn word() -> Self {
        Self::Word(HashMap::new())
    }

    fn qwen() -> anyhow::Result<Self> {
        Ok(Self::Qwen(Box::new(LlmTokenizer::from_env_or_default(
            None,
        )?)))
    }

    fn encode(&mut self, text: &str) -> Vec<u32> {
        match self {
            Self::Word(dict) => bm25_terms(text)
                .into_iter()
                .map(|term| {
                    let next = dict.len() as u32;
                    *dict.entry(term).or_insert(next)
                })
                .collect(),
            Self::Qwen(tok) => tok.encode_ids(text),
        }
    }
}

/// **先验因子**：`token → [(关联 token, 置信度)]` 的扩展表（`kind = query_expansion`）。
///
/// 由内核产出（P0 档是统计直接给的），外壳只负责消费：扩展词按 `alpha × conf` 加进查询。
#[derive(Debug, Default)]
struct ExpansionTable {
    entries: std::collections::HashMap<String, Vec<(String, f64)>>,
    /// 置信度的全局缩放（外壳的超参，不是内核参数）。
    alpha: f64,
    top_k: usize,
    source: String,
    sha256: String,
}

impl ExpansionTable {
    /// 从工件目录加载（`manifest.json` + `expansion.json`）。
    fn load(dir: &std::path::Path, alpha: f64, top_k: Option<usize>) -> anyhow::Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let payload_path = dir.join("expansion.json");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).with_context(|| {
                format!(
                    "读取扩展工件 {} 失败（先跑 tools/kernel/build_expansion.py）",
                    manifest_path.display()
                )
            })?)
            .with_context(|| format!("解析 {} 失败", manifest_path.display()))?;
        let payload = std::fs::read(&payload_path)
            .with_context(|| format!("读取 {} 失败", payload_path.display()))?;

        // 摘要校验：换因子等于换实验变量，必须能查出来。
        let expected = manifest
            .get("files")
            .and_then(|files| files.get("expansion.json"))
            .and_then(|file| file.get("sha256"))
            .and_then(|sha| sha.as_str())
            .unwrap_or("");
        if !expected.is_empty() {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&payload);
            let actual = format!("{:x}", hasher.finalize());
            if actual != expected {
                anyhow::bail!(
                    "扩展工件 {} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）",
                    payload_path.display()
                );
            }
        }

        let parsed: serde_json::Value = serde_json::from_slice(&payload)
            .with_context(|| format!("解析 {} 失败", payload_path.display()))?;
        let mut entries = std::collections::HashMap::new();
        if let Some(map) = parsed.get("entries").and_then(|value| value.as_object()) {
            for (term, rows) in map {
                let mut list = Vec::new();
                if let Some(rows) = rows.as_array() {
                    for row in rows {
                        let Some(pair) = row.as_array() else { continue };
                        let (Some(neighbor), Some(conf)) = (
                            pair.first().and_then(|value| value.as_str()),
                            pair.get(1).and_then(|value| value.as_f64()),
                        ) else {
                            continue;
                        };
                        list.push((neighbor.to_string(), conf));
                    }
                }
                entries.insert(term.clone(), list);
            }
        }
        let default_top_k = manifest
            .get("params")
            .and_then(|params| params.get("top_k"))
            .and_then(|value| value.as_u64())
            .unwrap_or(5) as usize;
        let source = manifest
            .get("source")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            entries,
            alpha,
            top_k: top_k.unwrap_or(default_top_k),
            source,
            sha256: sha256_bytes_of(&payload),
        })
    }

    /// 查一个 token 的扩展项（最多 `top_k` 条）。
    fn expansions(&self, term: &str) -> &[(String, f64)] {
        match self.entries.get(term) {
            Some(list) => &list[..list.len().min(self.top_k)],
            None => &[],
        }
    }
}

/// **先验因子**：`token → S(t)` 的显著性因子表。
#[derive(Debug, Default)]
struct SaliencyTable {
    entries: std::collections::HashMap<String, f64>,
    source: String,
    sha256: String,
}

impl SaliencyTable {
    fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let payload_path = dir.join("saliency.json");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).with_context(|| {
                format!(
                    "读取 Saliency 工件 {} 失败（先跑 tools/kernel/build_saliency.py）",
                    manifest_path.display()
                )
            })?)
            .with_context(|| format!("解析 {} 失败", manifest_path.display()))?;
        let payload = std::fs::read(&payload_path)
            .with_context(|| format!("读取 {} 失败", payload_path.display()))?;

        let expected = manifest
            .get("files")
            .and_then(|files| files.get("saliency.json"))
            .and_then(|file| file.get("sha256"))
            .and_then(|sha| sha.as_str())
            .unwrap_or("");
        if !expected.is_empty() {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&payload);
            let actual = format!("{:x}", hasher.finalize());
            if actual != expected {
                anyhow::bail!(
                    "Saliency 工件 {} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）",
                    payload_path.display()
                );
            }
        }

        let parsed: serde_json::Value = serde_json::from_slice(&payload)
            .with_context(|| format!("解析 {} 失败", payload_path.display()))?;
        let mut entries = std::collections::HashMap::new();
        if let Some(map) = parsed.get("entries").and_then(|value| value.as_object()) {
            for (term, rows) in map {
                if let Some(rows) = rows.as_array() {
                    for row in rows {
                        let Some(pair) = row.as_array() else { continue };
                        if let Some(score) = pair.get(1).and_then(|v| v.as_f64()) {
                            entries.insert(term.clone(), score);
                            break;
                        }
                    }
                }
            }
        }
        let source = manifest
            .get("source")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            entries,
            source,
            sha256: sha256_bytes_of(&payload),
        })
    }
}

fn sha256_bytes_of(payload: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}

/// 外壳的打分算法选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scorer {
    Bow,
    Oac,
}

pub struct ShellAdapter {
    info: AdapterInfo,
    tokenizer: Tokenizer,
    tokenizer_name: &'static str,
    scorer_kind: Scorer,
    index: Option<Arc<PositionalIndex>>,
    bow: Option<BowScorer>,
    oac: Option<OacScorer>,
    oac_params: OacParams,
    expansion: Option<ExpansionTable>,
    /// `None` = 总是扩展；`Some(τ)` = 只有检索置信度低于 τ（榜首优势不足）时才扩展。
    gate_margin: Option<f64>,
    /// BM25+ 下界偏置（0.0 = 纯 BM25）。
    delta: f64,
    /// 先验显著性因子 S(t)。
    saliency: Option<SaliencyTable>,
    /// 动态饱和度调节强度 lambda。
    lambda_k1: f64,
}

impl std::fmt::Debug for ShellAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellAdapter")
            .field("name", &self.info.name)
            .field("tokenizer", &self.tokenizer_name)
            .finish()
    }
}

impl ShellAdapter {
    fn build(scorer_kind: Scorer, name: &str, family: &str) -> anyhow::Result<Self> {
        let choice = TokenizerChoice::from_env()?;
        let tokenizer = match choice {
            TokenizerChoice::Word => Tokenizer::word(),
            TokenizerChoice::Qwen => Tokenizer::qwen()?,
        };
        let oac_params = oac_params_from_env()?;

        let mut config = serde_json::json!({
            "tokenizer": choice.name(),
            "index": "sl-engine::PositionalIndex（一份索引，所有算法共用）",
        });
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "algorithm".into(),
                serde_json::json!(match scorer_kind {
                    Scorer::Bow => "bow（词袋端点，BM25 数学）",
                    Scorer::Oac => "oac（有序对齐覆盖）",
                }),
            );
            if scorer_kind == Scorer::Oac {
                if let Some(oac) = oac_params.to_json().as_object() {
                    for (key, value) in oac {
                        obj.insert(key.clone(), value.clone());
                    }
                }
            }
        }

        // 先验因子（可选）：地址来自 SL_EXPANSION_ARTIFACT；给了就消费，没给就退化成纯原词检索。
        let expansion = match std::env::var("SL_EXPANSION_ARTIFACT") {
            Ok(dir) => {
                let alpha: f64 = std::env::var("SL_EXP_ALPHA")
                    .unwrap_or_else(|_| "0.3".to_string())
                    .parse()
                    .map_err(|_| anyhow::anyhow!("SL_EXP_ALPHA 不是数字"))?;
                if !alpha.is_finite() || alpha < 0.0 {
                    anyhow::bail!("SL_EXP_ALPHA 必须是非负有限数");
                }
                let top_k = match std::env::var("SL_EXP_TOPK") {
                    Ok(raw) => Some(
                        raw.parse::<usize>()
                            .map_err(|_| anyhow::anyhow!("SL_EXP_TOPK 不是正整数：{raw:?}"))?,
                    ),
                    Err(_) => None,
                };
                Some(ExpansionTable::load(
                    std::path::Path::new(&dir),
                    alpha,
                    top_k,
                )?)
            }
            Err(_) => None,
        };
        // 第一阶段过滤器的**固定规则版**：只在检索置信度低时才启用扩展。
        // 门限 = (top1 − top_{k}) / top1；越大说明榜首优势越明显、越不需要扩展。
        let gate_margin: Option<f64> = match std::env::var("SL_EXP_GATE") {
            Ok(raw) => {
                let value: f64 = raw
                    .parse()
                    .map_err(|_| anyhow::anyhow!("SL_EXP_GATE 不是数字：{raw:?}"))?;
                if !value.is_finite() || value < 0.0 {
                    anyhow::bail!("SL_EXP_GATE 必须是非负有限数：{raw:?}");
                }
                Some(value)
            }
            Err(_) => None,
        };

        // BM25+ 下界偏置（可选，默认 0.0 即传统 BM25）。
        let delta: f64 = match std::env::var("SL_BOW_DELTA") {
            Ok(raw) => {
                let value: f64 = raw
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("SL_BOW_DELTA 不是数字：{raw:?}"))?;
                if !value.is_finite() || value < 0.0 {
                    anyhow::bail!("SL_BOW_DELTA 必须是非负有限数：{raw:?}");
                }
                value
            }
            Err(_) => 0.0,
        };

        if delta > 0.0 {
            if let Some(obj) = config.as_object_mut() {
                obj.insert("delta".into(), serde_json::json!(delta));
            }
        }

        if let (Some(table), Some(obj)) = (expansion.as_ref(), config.as_object_mut()) {
            obj.insert(
                "prior_factor".into(),
                serde_json::json!({
                    "name": "query_expansion",
                    "kind": "token_confidence_list",
                    "source": table.source,
                    "sha256": table.sha256,
                    "alpha": table.alpha,
                    "top_k": table.top_k,
                    "entries": table.entries.len(),
                }),
            );
        } else if let Some(obj) = config.as_object_mut() {
            obj.insert("prior_factor".into(), serde_json::json!(null));
        }

        Ok(Self {
            info: AdapterInfo::new(
                name,
                family,
                "0.2.0",
                "外壳：crates/sl-engine（PositionalIndex + 打分器）；适配器：crates/sl-adapters/src/shell.rs",
            )
            .with_config(config),
            tokenizer,
            tokenizer_name: choice.name(),
            scorer_kind,
            index: None,
            bow: None,
            oac: None,
            oac_params,
            expansion,
            gate_margin,
            delta,
            saliency: None,
            lambda_k1: 0.0,
        })
    }

    /// `saliency-bm25`：带先验 Saliency 自适应调制的 BM25 外壳。
    pub fn saliency_bm25() -> anyhow::Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/saliency-v0");
        let dir = std::env::var("SL_SALIENCY_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 Saliency 工件目录 {}（先跑 tools/kernel/build_saliency.py {dataset}）",
                dir
            );
        }
        let delta = std::env::var("SL_BOW_DELTA")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.5);
        let lambda_k1 = std::env::var("SL_LAMBDA_K1")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0.5);

        let table = SaliencyTable::load(path)?;
        let mut adapter = Self::build(Scorer::Bow, "saliency-bm25", "shell-saliency-bm25")?;
        adapter.delta = delta;
        adapter.lambda_k1 = lambda_k1;
        if let Some(config) = adapter.info.config.as_object_mut() {
            config.insert("delta".into(), serde_json::json!(delta));
            config.insert("lambda_k1".into(), serde_json::json!(lambda_k1));
            config.insert(
                "prior_factor_saliency".into(),
                serde_json::json!({
                    "name": "token_saliency",
                    "kind": "token_confidence_list",
                    "source": table.source,
                    "sha256": table.sha256,
                    "entries": table.entries.len(),
                }),
            );
        }
        adapter.saliency = Some(table);
        Ok(adapter)
    }

    /// `bow`：词袋端点（G1 等价性门用）。
    pub fn bow() -> anyhow::Result<Self> {
        Self::build(Scorer::Bow, "bow", "shell-bow")
    }

    /// `bow-plus`：带 BM25+ 词频下界偏置 delta 的词袋端点。
    pub fn bow_plus(delta: f64) -> anyhow::Result<Self> {
        let mut adapter = Self::build(Scorer::Bow, "bow-plus", "shell-bow-plus")?;
        adapter.delta = delta;
        if let Some(config) = adapter.info.config.as_object_mut() {
            config.insert("delta".into(), serde_json::json!(delta));
        }
        Ok(adapter)
    }

    /// `oac`：有序对齐覆盖。
    pub fn oac() -> anyhow::Result<Self> {
        Self::build(Scorer::Oac, "oac", "shell-oac")
    }

    /// `gated-exp`：自适应门控扩展（Token 筛选 + 门控扩展）。
    pub fn gated_exp() -> anyhow::Result<Self> {
        let dir = std::env::var("SL_GATED_EXP_ARTIFACT")
            .or_else(|_| std::env::var("SL_EXPANSION_ARTIFACT"))
            .unwrap_or_else(|_| "data/scifact/artifacts/gated-exp-v1".to_string());
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 gated-exp 工件目录 {}（先跑 tools/kernel/build_gated_factors.py）",
                dir
            );
        }
        let alpha: f64 = std::env::var("SL_EXP_ALPHA")
            .unwrap_or_else(|_| "0.25".to_string())
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_EXP_ALPHA 不是数字"))?;
        let gate: f64 = std::env::var("SL_EXP_GATE")
            .unwrap_or_else(|_| "0.5".to_string())
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_EXP_GATE 不是数字"))?;
        let top_k: usize = std::env::var("SL_EXP_TOPK")
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_EXP_TOPK 不是正整数"))?;

        let expansion = ExpansionTable::load(path, alpha, Some(top_k))?;
        let mut adapter = Self::build(Scorer::Bow, "gated-exp", "shell-gated-exp")?;
        adapter.expansion = Some(expansion);
        adapter.gate_margin = Some(gate);

        if let Some(config) = adapter.info.config.as_object_mut() {
            if let Some(table) = adapter.expansion.as_ref() {
                config.insert(
                    "prior_factor".into(),
                    serde_json::json!({
                        "name": "gated_expansion",
                        "kind": "token_confidence_list",
                        "source": table.source,
                        "sha256": table.sha256,
                        "alpha": table.alpha,
                        "top_k": table.top_k,
                        "gate_margin": gate,
                        "entries": table.entries.len(),
                    }),
                );
            }
        }
        Ok(adapter)
    }
}

/// 从环境变量读 OAC 参数（消融用）；非法值直接报错。
fn oac_params_from_env() -> anyhow::Result<OacParams> {
    let mut params = OacParams::default();

    if let Ok(raw) = std::env::var("SL_OAC_ORDERED") {
        params.ordered = match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            other => anyhow::bail!("SL_OAC_ORDERED 只接受 true/false，收到 {other:?}"),
        };
    }
    if let Ok(raw) = std::env::var("SL_OAC_SPAN_DISCOUNT") {
        params.use_span_discount = match raw.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            other => anyhow::bail!("SL_OAC_SPAN_DISCOUNT 只接受 true/false，收到 {other:?}"),
        };
    }
    if let Ok(raw) = std::env::var("SL_OAC_WINDOW") {
        params.window = raw
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_OAC_WINDOW 不是正整数：{raw:?}"))?;
        if params.window == 0 {
            anyhow::bail!("SL_OAC_WINDOW 必须 ≥ 1");
        }
    }
    if let Ok(raw) = std::env::var("SL_OAC_LENGTH_B") {
        params.length_b = raw
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_OAC_LENGTH_B 不是数字：{raw:?}"))?;
        if !params.length_b.is_finite() || params.length_b < 0.0 {
            anyhow::bail!("SL_OAC_LENGTH_B 必须是非负有限数：{raw:?}");
        }
    }
    if let Ok(raw) = std::env::var("SL_OAC_MAX_SKIP") {
        params.max_skip = raw
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("SL_OAC_MAX_SKIP 不是非负整数：{raw:?}"))?;
    }
    // 内核 gap 权重表：逗号分隔的 5 个数（gap=1 / 2 / 3-4 / 5-8 / >8）。
    if let Ok(raw) = std::env::var("SL_OAC_GAP_WEIGHTS") {
        let parts: Vec<&str> = raw.split(',').map(|s| s.trim()).collect();
        if parts.len() != 5 {
            anyhow::bail!("SL_OAC_GAP_WEIGHTS 需要 5 个逗号分隔的数，收到 {raw:?}");
        }
        let mut weights = [1.0f64; 5];
        for (slot, part) in weights.iter_mut().zip(parts) {
            *slot = part
                .parse()
                .map_err(|_| anyhow::anyhow!("SL_OAC_GAP_WEIGHTS 里有非数字：{part:?}"))?;
            if !slot.is_finite() || *slot < 0.0 {
                anyhow::bail!("SL_OAC_GAP_WEIGHTS 必须是非负有限数，收到 {part:?}");
            }
        }
        params.kernel = KernelGapTable { weights };
    }
    Ok(params)
}

/// 检索是否"有把握"：看**头部**的分数衰减 `1 − top_k / top1`（k 取 min(10, 命中数)）。
///
/// 衰减大 ⇒ 榜首优势明显 ⇒ 有把握 ⇒ 不扩展；衰减小 ⇒ 一片混战 ⇒ 扩展可能有帮助。
/// 注意不能用「最后一名」当分母：长尾分数接近 0，比值恒为 1，门等于永远开着。
fn confident(hits: &[(String, f64)], tau: f64) -> bool {
    let Some((_, top1)) = hits.first() else {
        return false;
    };
    if *top1 <= 0.0 {
        return false;
    }
    if hits.len() < 2 {
        return true;
    }
    let head = hits.len().min(10) - 1;
    let kth = hits[head].1;
    (top1 - kth) / top1 >= tau
}

impl ShellAdapter {
    /// 只用原词打一次分（第一阶段）。
    fn encode_base(&mut self, query: &str) -> Vec<WeightedTerm> {
        match &mut self.tokenizer {
            Tokenizer::Word(dict) => bm25_terms(query)
                .into_iter()
                .map(|term| {
                    let next = dict.len() as u32;
                    let id = *dict.entry(term).or_insert(next);
                    (id, 1.0)
                })
                .collect(),
            Tokenizer::Qwen(tok) => tok
                .encode_ids(query)
                .into_iter()
                .map(|id| (id, 1.0))
                .collect(),
        }
    }

    fn score(&mut self, weighted: &[WeightedTerm], k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        match self.scorer_kind {
            Scorer::Bow => {
                let scorer = self.bow.as_ref().context("bow 未建索引（先调 index）")?;
                Ok(scorer.search(weighted, k))
            }
            Scorer::Oac => {
                let scorer = self.oac.as_mut().context("oac 未建索引（先调 index）")?;
                Ok(scorer.search(weighted, k))
            }
        }
    }

    /// 查询 → 带权词项：原词权重 1.0，扩展词权重 `alpha × conf`。
    ///
    /// 扩展表按 **token 字符串** 建（与 tokenizer 无关），所以这里必须拿到字符串；
    /// 只有词级 tokenizer 支持扩展（BPE 的 token 与扩展表的词不在同一空间，暂不支持）。
    fn encode_weighted(&mut self, query: &str) -> Vec<WeightedTerm> {
        match &mut self.tokenizer {
            Tokenizer::Word(dict) => {
                let terms = bm25_terms(query);
                let mut weighted: Vec<WeightedTerm> = Vec::with_capacity(terms.len() * 2);
                for term in &terms {
                    let next = dict.len() as u32;
                    let id = *dict.entry(term.clone()).or_insert(next);
                    weighted.push((id, 1.0));
                }
                if let Some(table) = self.expansion.as_ref() {
                    for term in &terms {
                        for (neighbor, conf) in table.expansions(term) {
                            // 扩展词必须在索引词表里才有意义；OOV 直接跳过。
                            if let Some(&id) = dict.get(neighbor) {
                                weighted.push((id, table.alpha * conf));
                            }
                        }
                    }
                }
                weighted
            }
            Tokenizer::Qwen(tok) => {
                // BPE 路径暂不支持扩展（字符串空间不同），保持纯原词行为。
                tok.encode_ids(query)
                    .into_iter()
                    .map(|id| (id, 1.0))
                    .collect()
            }
        }
    }
}

impl Adapter for ShellAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        // 切分一次、建索引一次，所有打分器共享同一份（Arc）：换算法不重建索引。
        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        for doc in docs {
            encoded.push((doc.id.clone(), self.tokenizer.encode(&doc.full_text())));
        }
        let index = Arc::new(PositionalIndex::build(encoded));
        let bow = match &self.saliency {
            Some(table) => {
                let mut map = std::collections::HashMap::new();
                if let Tokenizer::Word(dict) = &self.tokenizer {
                    for (term, &score) in &table.entries {
                        if let Some(&id) = dict.get(term) {
                            map.insert(id, score);
                        }
                    }
                }
                BowScorer::with_saliency(Arc::clone(&index), self.delta, map, self.lambda_k1)?
            }
            None => BowScorer::with_delta(Arc::clone(&index), self.delta)?,
        };
        self.bow = Some(bow);
        self.oac = Some(OacScorer::new(Arc::clone(&index), self.oac_params.clone()));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        // 第一阶段：只用原词检索一次（这一步也是"检索置信度"的来源）。
        let base = self.encode_base(query);
        let base_hits = self.score(&base, k)?;

        let expand = match (self.expansion.as_ref(), self.gate_margin) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(_), Some(tau)) => !confident(&base_hits, tau),
        };
        if !expand {
            return Ok(base_hits);
        }

        // 第二阶段：把扩展词加进查询再打一次分。
        let weighted = self.encode_weighted(query);
        if weighted.len() == base.len() {
            return Ok(base_hits);
        }
        self.score(&weighted, k)
    }
}
