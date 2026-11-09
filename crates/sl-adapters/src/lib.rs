//! 适配器注册表：`sl run --adapters a,b,c` 里的名字到实现的映射。
//!
//! 增删适配器只改这一个文件；每个适配器实现 `sl_core::Adapter`，并给出 `AdapterInfo`
//! 说明自己「是谁、从哪来、带什么参数」。
//!
//! 现有 7 个名字：
//!
//! | 名字 | 算法 | 实现位置 |
//! |---|---|---|
//! | `bm25` | 自实现 Okapi BM25（k1=1.5, b=0.75，字母数字词项 + CJK jieba 分词） | [`bm25`] |
//! | `faq-char` / `faq-bigram` / `faq-word` / `faq-hybrid` / `faq-bm25` | 中文 FAQ 五方案（移植 faq-search） | [`faq`] |
//! | `rs` | relation-search（Rust BM25 + ast-grep），子进程接入 | [`rs`] |
//! | `seq` | token 时序相关度内核（BM25 主体 + 内核工件的时序边权） | [`seq`] |
//!
//! 共享的清洗/切分点在 [`tokenize`]；中文语料相关的适用范围写在 [`faq`] 的模块文档里。

use anyhow::{bail, Result};

mod bm25;
mod faq;
mod rs;
mod seq;
pub mod tokenize;

pub use bm25::Bm25Adapter;
pub use faq::{FaqAdapter, Scheme};
pub use rs::RelationSearchAdapter;
pub use seq::SeqAdapter;

/// 当前可用的适配器名（进 `sl run --adapters` 的取值），顺序即展示顺序。
pub fn names() -> Vec<&'static str> {
    vec![
        "bm25",
        "faq-char",
        "faq-bigram",
        "faq-word",
        "faq-hybrid",
        "faq-bm25",
        "rs",
        "seq",
    ]
}

/// 按名字构造一个适配器。未知名字必须报错并列出可选名字。
///
/// `rs` 会在这里就检查 relation-search 二进制并读取它的 sha256 / 来源 commit：
/// 外部依赖缺失应该在「构造」时暴露，而不是跑完一整轮只得到空结果。
pub fn build(name: &str) -> Result<Box<dyn sl_core::Adapter>> {
    let adapter: Box<dyn sl_core::Adapter> = match name {
        "bm25" => Box::new(Bm25Adapter::new()),
        "rs" => Box::new(RelationSearchAdapter::new()?),
        "seq" => Box::new(SeqAdapter::new()?),
        other => match Scheme::from_name(other) {
            Some(scheme) => Box::new(FaqAdapter::new(scheme)),
            None => bail!("未知适配器 {other:?}；当前可用：{}", names().join(", ")),
        },
    };
    Ok(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_all_names_without_duplicates() {
        assert_eq!(
            names(),
            vec![
                "bm25",
                "faq-char",
                "faq-bigram",
                "faq-word",
                "faq-hybrid",
                "faq-bm25",
                "rs",
                "seq"
            ]
        );
        let unique: std::collections::BTreeSet<&str> = names().into_iter().collect();
        assert_eq!(unique.len(), names().len());
    }

    #[test]
    fn build_unknown_name_lists_available_names() {
        let err = match build("no-such-adapter") {
            Ok(_) => panic!("未知适配器名不应构造成功"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("未知适配器"));
        for name in names() {
            assert!(err.contains(name), "错误信息应列出 {name}：{err}");
        }
    }

    #[test]
    fn build_returns_adapter_with_matching_info_name() {
        for name in [
            "bm25",
            "faq-char",
            "faq-bigram",
            "faq-word",
            "faq-hybrid",
            "faq-bm25",
        ] {
            let adapter = build(name).expect("构造适配器");
            assert_eq!(adapter.info().name, name);
            assert!(!adapter.info().family.is_empty());
            assert!(!adapter.info().version.is_empty());
        }
    }

    #[test]
    fn faq_adapters_record_scheme_specific_config() {
        let hybrid = build("faq-hybrid").expect("构造 faq-hybrid");
        let config = &hybrid.info().config;
        assert_eq!(config["scheme"], "hybrid");
        assert_eq!(config["hybrid_weights"]["word"], 2.0);
        assert_eq!(config["hybrid_weights"]["bigram"], 1.0);
        assert_eq!(config["hybrid_weights"]["char"], 0.3);

        let bm25 = build("faq-bm25").expect("构造 faq-bm25");
        assert_eq!(bm25.info().config["bm25"]["k1"], 1.5);
        assert_eq!(bm25.info().config["bm25"]["b"], 0.75);

        let char = build("faq-char").expect("构造 faq-char");
        assert!(char.info().config.get("hybrid_weights").is_none());
    }

    #[test]
    fn rs_info_records_binary_digest_and_block_factor() {
        if std::env::var_os("RELATION_SEARCH_BIN").is_none() {
            let default = std::env::var_os("HOME")
                .map(|home| {
                    std::path::PathBuf::from(home)
                        .join("coding/my/relation-search/target/release/relation-search")
                })
                .unwrap_or_default();
            if !default.is_file() {
                eprintln!(
                    "跳过：找不到 relation-search 二进制（{}）",
                    default.display()
                );
                return;
            }
        }
        let adapter = build("rs").expect("构造 rs");
        assert_eq!(adapter.info().name, "rs");
        assert_eq!(adapter.info().family, "relation-search");
        // version = 二进制 sha256 前 12 位 + 仓库 HEAD。
        let (digest, _) = adapter
            .info()
            .version
            .split_once('+')
            .expect("version 应含 '+'");
        assert_eq!(
            digest.len(),
            12,
            "version 里的二进制摘要应为 sha256 前 12 位：{}",
            adapter.info().version
        );
        assert_eq!(adapter.info().config["block_factor"], 8);
        assert_eq!(adapter.info().config["min_block_limit"], 50);
    }
}
