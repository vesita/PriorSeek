//! LLM tokenizer 的加载与封装（**不自研词表**，直接用成熟 BPE）。
//!
//! 为什么要它：现在的 `tokenize::bm25_terms` 是词级切分，`monocytes` / `monocyte` 是两个 token，
//! 实测「硬漏」查询的相关文档词级覆盖率只有 0.32，而字符 n-gram 能抬到 0.60 —— 说明词级切分丢掉了
//! 形态学信息。成熟 BPE 直接把这件事做掉，不用我们自己造子词方案。
//!
//! 双端一致性：Python（`tools/kernel/`）与 Rust（本模块）读**同一个** `tokenizer.json`，
//! `tests/llm_tokenizer_parity.rs` 断言两边切出的 token id 序列逐位相同。
//!
//! 运行时要求：纯 CPU、离线（文件已下载到 `data/vendor/tokenizers/<id>/`），
//! sha256 与同目录 `manifest.json` 比对，不符直接报错 —— 不允许「悄悄换了个 tokenizer」。

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// 默认 tokenizer：Qwen2.5 BPE（151k 词表，中英文通用）。
pub const DEFAULT_TOKENIZER_ID: &str = "qwen2.5-0.5b";

/// 一个已加载并校验过的 LLM tokenizer。
pub struct LlmTokenizer {
    inner: tokenizers::Tokenizer,
    pub id: String,
    pub sha256: String,
    pub vocab_size: usize,
    pub path: PathBuf,
}

impl std::fmt::Debug for LlmTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmTokenizer")
            .field("id", &self.id)
            .field("sha256", &self.sha256)
            .field("vocab_size", &self.vocab_size)
            .field("path", &self.path)
            .finish()
    }
}

impl LlmTokenizer {
    /// 按 `SL_LLM_TOKENIZER`（文件路径）或默认位置加载。
    pub fn from_env_or_default(data_dir: Option<&Path>) -> Result<Self> {
        if let Ok(path) = std::env::var("SL_LLM_TOKENIZER") {
            return Self::from_path(Path::new(&path));
        }
        let data_dir = match data_dir {
            Some(dir) => dir.to_path_buf(),
            None => PathBuf::from(std::env::var("SL_DATA_DIR").unwrap_or_else(|_| "data".into())),
        };
        Self::from_path(
            &data_dir
                .join("vendor")
                .join("tokenizers")
                .join(DEFAULT_TOKENIZER_ID)
                .join("tokenizer.json"),
        )
    }

    /// 从 `tokenizer.json` 加载，并校验同目录 `manifest.json` 里钉住的 sha256。
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| {
            format!(
                "读取 tokenizer 失败：{}（先跑 `cd tools/kernel && uv run python fetch_tokenizer.py {}`）",
                path.display(),
                DEFAULT_TOKENIZER_ID
            )
        })?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let sha256 = format!("{:x}", hasher.finalize());

        // 与 manifest 比对：换 tokenizer 等于换实验变量，必须显式失败而不是静默生效。
        let manifest_path = path.with_file_name("manifest.json");
        if manifest_path.is_file() {
            let manifest: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&manifest_path)
                    .with_context(|| format!("读取 {} 失败", manifest_path.display()))?,
            )
            .with_context(|| format!("解析 {} 失败", manifest_path.display()))?;
            if let Some(expected) = manifest.get("sha256").and_then(|v| v.as_str()) {
                if expected != sha256 {
                    bail!(
                        "tokenizer 文件的 sha256 与 manifest 不符（期望 {expected}，实际 {sha256}）——\
                         文件被换过；若确实要换，改 fetch_tokenizer.py 里钉住的值并重建工件"
                    );
                }
            }
        }

        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| anyhow::anyhow!("解析 tokenizer.json 失败（{}）：{err}", path.display()))?;
        let vocab_size = inner.get_vocab_size(true);
        let id = path
            .parent()
            .and_then(|dir| dir.file_name())
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| DEFAULT_TOKENIZER_ID.to_string());

        Ok(Self { inner, id, sha256, vocab_size, path: path.to_path_buf() })
    }

    /// 切分并返回 token id 序列。**不加特殊 token**（BOS/EOS 对检索是噪声）。
    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    /// 切分并返回 token 字符串（调试 / 报告用；索引与检索都用 [`Self::encode_ids`]）。
    pub fn encode_tokens(&self, text: &str) -> Vec<String> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_tokens().iter().map(|t| t.to_string()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// 进 `AdapterInfo.config` 的元信息。
    pub fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "tokenizer_id": self.id,
            "tokenizer_sha256": self.sha256,
            "tokenizer_vocab_size": self.vocab_size,
            "tokenizer_kind": "bpe（llm，非自研）",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_if_available() -> Option<LlmTokenizer> {
        LlmTokenizer::from_env_or_default(None).ok()
    }

    #[test]
    fn encodes_text_into_ids_without_special_tokens() {
        let Some(tok) = tokenizer_if_available() else {
            eprintln!("跳过：找不到 tokenizer（先跑 fetch_tokenizer.py）");
            return;
        };
        let ids = tok.encode_ids("monocytes respond to inflammation");
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|id| (*id as usize) < tok.vocab_size));
        // 同一文本必须稳定
        assert_eq!(ids, tok.encode_ids("monocytes respond to inflammation"));
    }

    #[test]
    fn subword_shares_pieces_across_morphological_variants() {
        let Some(tok) = tokenizer_if_available() else {
            eprintln!("跳过：找不到 tokenizer");
            return;
        };
        // BPE 的核心价值：形态学变体共享子词（这正是词级切分做不到的）
        let a = tok.encode_tokens("monocytes");
        let b = tok.encode_tokens("monocyte");
        let shared = a.iter().filter(|t| b.contains(t)).count();
        assert!(shared >= 1, "monocytes 与 monocyte 应至少共享一个子词：{a:?} vs {b:?}");
    }

    #[test]
    fn rejects_tokenizer_whose_sha_does_not_match_manifest() {
        let Some(tok) = tokenizer_if_available() else {
            eprintln!("跳过：找不到 tokenizer");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let copied = tmp.path().join("tokenizer.json");
        std::fs::copy(&tok.path, &copied).unwrap();
        std::fs::write(
            tmp.path().join("manifest.json"),
            serde_json::json!({"sha256": "0".repeat(64)}).to_string(),
        )
        .unwrap();
        let err = LlmTokenizer::from_path(&copied).expect_err("sha256 不符必须报错");
        assert!(format!("{err:#}").contains("sha256"));
    }
}
