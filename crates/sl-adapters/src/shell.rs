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
        match std::env::var("SL_SHELL_TOKENIZER").unwrap_or_else(|_| "word".to_string()).as_str() {
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
        Ok(Self::Qwen(Box::new(LlmTokenizer::from_env_or_default(None)?)))
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
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&manifest_path)
                .with_context(|| format!("读取扩展工件 {} 失败（先跑 tools/kernel/build_expansion.py）", manifest_path.display()))?,
        )
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
            obj.insert("algorithm".into(), serde_json::json!(match scorer_kind {
                Scorer::Bow => "bow（词袋端点，BM25 数学）",
                Scorer::Oac => "oac（有序对齐覆盖）",
            }));
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
                Some(ExpansionTable::load(std::path::Path::new(&dir), alpha, top_k)?)
            }
            Err(_) => None,
        };
        if let (Some(table), Some(obj)) = (expansion.as_ref(), config.as_object_mut()) {
            obj.insert("prior_factor".into(), serde_json::json!({
                "name": "query_expansion",
                "kind": "token_confidence_list",
                "source": table.source,
                "sha256": table.sha256,
                "alpha": table.alpha,
                "top_k": table.top_k,
                "entries": table.entries.len(),
            }));
        } else if let Some(obj) = config.as_object_mut() {
            obj.insert("prior_factor".into(), serde_json::json!(null));
        }

        Ok(Self {
            info: AdapterInfo::new(
                name,
                family,
                "0.1.0",
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
        })
    }

    /// `bow`：词袋端点（G1 等价性门用）。
    pub fn bow() -> anyhow::Result<Self> {
        Self::build(Scorer::Bow, "bow", "shell-bow")
    }

    /// `oac`：有序对齐覆盖。
    pub fn oac() -> anyhow::Result<Self> {
        Self::build(Scorer::Oac, "oac", "shell-oac")
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

impl ShellAdapter {
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
                tok.encode_ids(query).into_iter().map(|id| (id, 1.0)).collect()
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
        self.bow = Some(BowScorer::new(Arc::clone(&index)));
        self.oac = Some(OacScorer::new(Arc::clone(&index), self.oac_params.clone()));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let weighted = self.encode_weighted(query);
        match self.scorer_kind {
            Scorer::Bow => {
                let scorer = self.bow.as_ref().context("bow 未建索引（先调 index）")?;
                Ok(scorer.search(&weighted, k))
            }
            Scorer::Oac => {
                let scorer = self.oac.as_mut().context("oac 未建索引（先调 index）")?;
                Ok(scorer.search(&weighted, k))
            }
        }
    }
}
