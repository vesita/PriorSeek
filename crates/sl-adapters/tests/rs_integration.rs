//! `rs` 适配器的集成测试：真的去调 relation-search 子进程。
//!
//! 二进制不存在时打印跳过原因并直接返回 —— 这台机器上没构建 relation-search
//! 不应该让整个适配器测试变红（它是外部依赖，不是本仓代码）。
//!
//! 断言口径：返回的 doc id 必须全部属于语料、不能重复（块级候选已折叠到文档级）。
//! 「至少有一条命中」只对**精确 token** 的查询成立，避免因外部实现的排序细节变红。

use sl_adapters::RelationSearchAdapter;
use sl_core::{Adapter, Doc};

fn relation_search_bin() -> Option<std::path::PathBuf> {
    if let Some(value) = std::env::var_os("RELATION_SEARCH_BIN") {
        let path = std::path::PathBuf::from(value);
        return path.is_file().then_some(path);
    }
    let home = std::env::var_os("HOME")?;
    let path = std::path::PathBuf::from(home)
        .join("coding/my/relation-search/target/release/relation-search");
    path.is_file().then_some(path)
}

fn docs() -> Vec<Doc> {
    vec![
        // id 里带 `/` 与空格：验证文件名清洗（`docs/a b.rs` → `docs_a_b.rs.<sha8>.txt`）。
        Doc::new(
            "docs/a b.rs",
            "fn parse_config(path: &str) -> Config { load(path) }\nfn reload() {}",
        ),
        // 中文 id：文件名清洗 + CJK 分词两条路径都要过。
        Doc::new(
            "中文/报警阈值说明",
            "报警阈值配置说明：阈值超过六十度触发报警，报警阈值保存在配置文件里。",
        ),
        Doc::new(
            "plain",
            "logging retention days configuration: default retention is thirty days.",
        ),
    ]
}

#[test]
fn rs_index_and_search_returns_corpus_doc_ids() {
    if relation_search_bin().is_none() {
        eprintln!(
            "跳过 rs 集成测试：找不到 relation-search 二进制（RELATION_SEARCH_BIN 或默认检出路径）"
        );
        return;
    }
    let docs = docs();
    let corpus_ids: std::collections::BTreeSet<&str> =
        docs.iter().map(|doc| doc.id.as_str()).collect();

    let mut adapter = RelationSearchAdapter::new().expect("构造 rs 适配器");
    // AdapterInfo 必须能追溯外部实现：version = 二进制 sha256 前 12 位 + 仓库 HEAD。
    assert_eq!(adapter.info().name, "rs");
    assert!(
        adapter.info().version.contains('+'),
        "version 应含二进制摘要与 HEAD：{}",
        adapter.info().version
    );
    assert_eq!(adapter.info().config["block_factor"], 8);

    adapter.index(&docs).expect("建索引");

    for query in ["parse_config", "报警阈值配置", "retention days"] {
        let hits = adapter.search(query, 3).expect("检索");
        let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
        for id in &ids {
            assert!(
                corpus_ids.contains(id),
                "查询 {query:?} 返回了语料外的 doc id {id:?}"
            );
        }
        let unique: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "查询 {query:?} 返回了重复 doc id：{ids:?}（块级候选没折叠干净）"
        );
        assert!(hits.len() <= 3, "k=3 却返回了 {} 条", hits.len());
    }

    // 精确 token 的查询必须命中：否则说明索引/映射链路（临时文件名 → doc id）坏了。
    let hits = adapter.search("parse_config", 3).expect("检索");
    assert!(
        hits.iter().any(|(id, _)| id == "docs/a b.rs"),
        "精确 token 查询没有命中预期的文档，返回：{hits:?}"
    );
}
