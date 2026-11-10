//! Rust ↔ Python 的 LLM tokenizer 对拍。
//!
//! 为什么必须有这条：文档 token 在 Python 侧固化进工件、查询 token 在 Rust 运行时切分，
//! 两边若有一处不一致（版本、特殊 token、normalizer 设置），检索会静默错位。
//!
//! fixture 由 Python `tokenizers` 库生成（见 JSON 的 `note` 字段），本测试断言 **token id 序列逐位相同**，
//! 并额外断言 token 字符串一致（便于出错时直接看出是哪一段切分不同）。
//!
//! 重跑 fixture（改 tokenizer 或上游版本时）：
//!
//! ```bash
//! cd tools/kernel && uv run python - <<'PY'
//! from tokenizers import Tokenizer; import json, pathlib
//! tok = Tokenizer.from_file("../../data/vendor/tokenizers/qwen2.5-0.5b/tokenizer.json")
//! ...  # 见 tests/fixtures/llm_tokenizer_parity.json 的 cases 结构
//! PY
//! ```

use sl_adapters::tokenizer_llm::LlmTokenizer;
use std::path::Path;

#[derive(serde::Deserialize)]
struct FixtureCase {
    text: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    tokenizer_id: String,
    tokenizer_sha256: String,
    cases: Vec<FixtureCase>,
}

fn tokenizer_or_skip() -> Option<LlmTokenizer> {
    match LlmTokenizer::from_env_or_default(None) {
        Ok(tok) => Some(tok),
        Err(err) => {
            eprintln!("跳过对拍：找不到 tokenizer（{err:#}）");
            None
        }
    }
}

#[test]
fn rust_tokenization_matches_python_fixture_bit_for_bit() {
    let Some(tok) = tokenizer_or_skip() else { return };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llm_tokenizer_parity.json");
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("读 {} 失败：{err}", path.display())),
    )
    .expect("解析 parity fixture 失败");

    assert_eq!(tok.id, fixture.tokenizer_id, "跑测的 tokenizer 与 fixture 不是同一份");
    assert_eq!(tok.sha256, fixture.tokenizer_sha256, "tokenizer 文件摘要与 fixture 不一致");

    for case in &fixture.cases {
        let ids = tok.encode_ids(&case.text);
        assert_eq!(
            ids, case.ids,
            "文本 {:?} 的 token id 与 Python 不一致\n  rust={:?}\n  python={:?}",
            case.text, ids, case.ids
        );
        let tokens = tok.encode_tokens(&case.text);
        assert_eq!(
            tokens, case.tokens,
            "文本 {:?} 的 token 串与 Python 不一致\n  rust={:?}\n  python={:?}",
            case.text, tokens, case.tokens
        );
    }
    eprintln!("对拍通过：{} 条用例（{}）", fixture.cases.len(), fixture.tokenizer_id);
}
