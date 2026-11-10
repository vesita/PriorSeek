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
use sl_engine::{BowScorer, KernelGapTable, OacParams, OacScorer, PositionalIndex};
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
        let ids = self.tokenizer.encode(query);
        match self.scorer_kind {
            Scorer::Bow => {
                let scorer = self.bow.as_ref().context("bow 未建索引（先调 index）")?;
                Ok(scorer.search(&ids, k))
            }
            Scorer::Oac => {
                let scorer = self.oac.as_mut().context("oac 未建索引（先调 index）")?;
                Ok(scorer.search(&ids, k))
            }
        }
    }
}
