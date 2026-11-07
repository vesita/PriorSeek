//! FAQ 五方案 parity 测试：Rust 移植 vs 真实 `faq-search` 包的 top-k 结果。
//!
//! fixture 由 `tests/fixtures/gen_faq_parity.py` 用真实 faq-search 包算出（命令写在
//! 脚本头部）；这里逐方案、逐查询断言 **doc id 序列完全一致**，分数按相对误差 1e-9
//! 对齐（同一台机器上两边都是 IEEE-754 double，理论上逐位相同；留 1e-9 只为容忍
//! `ln` 的最后一个 ulp 差异，不掩盖排序差异）。
//!
//! 同时断言 jieba-rs 与 Python jieba 的切词结果一致：排名对不上时，这条测试会先把
//! 「切词漂移」这一层单独暴露出来。
//!
//! 断言的是 id **序列**而不是集合：同分顺序就是语义的一部分（原包按 (-score, id)
//! 排，恰好覆盖了「同分按 id 升序」这条规则）。

use serde::Deserialize;
use sl_adapters::tokenize::words;
use sl_adapters::{FaqAdapter, Scheme};
use sl_core::{Adapter, Doc};
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    topk: usize,
    schemes: Vec<String>,
    corpus: Vec<CorpusItem>,
    queries: Vec<QueryItem>,
    word_tokens: WordTokens,
    results: BTreeMap<String, BTreeMap<String, Vec<Hit>>>,
}

#[derive(Deserialize)]
struct CorpusItem {
    id: i64,
    question: String,
    answer: String,
}

#[derive(Deserialize)]
struct QueryItem {
    qid: String,
    text: String,
}

#[derive(Deserialize)]
struct WordTokens {
    docs: BTreeMap<String, Vec<String>>,
    queries: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct Hit {
    id: String,
    score: f64,
}

impl CorpusItem {
    /// 与原包 `FaqDoc.text` 一致：问题 + 空格 + 答案。
    fn text(&self) -> String {
        format!("{} {}", self.question, self.answer)
    }
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/faq_parity.json"
    );
    let text =
        std::fs::read_to_string(path).expect("读 fixture 失败（用 gen_faq_parity.py 重新生成）");
    serde_json::from_str(&text).expect("解析 fixture 失败")
}

fn scheme_of(name: &str) -> Scheme {
    match name {
        "char" => Scheme::Char,
        "bigram" => Scheme::Bigram,
        "word" => Scheme::Word,
        "hybrid" => Scheme::Hybrid,
        "bm25" => Scheme::Bm25,
        other => panic!("fixture 里有未知方案 {other}"),
    }
}

fn docs_of(fixture: &Fixture) -> Vec<Doc> {
    fixture
        .corpus
        .iter()
        .map(|item| Doc::new(item.id.to_string(), item.text()))
        .collect()
}

#[test]
fn jieba_rs_segments_same_as_python_jieba() {
    let fixture = fixture();
    for item in &fixture.corpus {
        let expected = &fixture.word_tokens.docs[&item.id.to_string()];
        let actual = words(&item.text());
        assert_eq!(
            &actual, expected,
            "文档 {} 的切词与 Python jieba 不一致（jieba-rs 词典/算法漂移）",
            item.id
        );
    }
    for query in &fixture.queries {
        let expected = &fixture.word_tokens.queries[&query.qid];
        let actual = words(&query.text);
        assert_eq!(
            &actual, expected,
            "查询 {} 的切词与 Python jieba 不一致",
            query.qid
        );
    }
}

#[test]
fn faq_schemes_match_python_faq_search_topk() {
    let fixture = fixture();
    let docs = docs_of(&fixture);
    let mut checked = 0usize;
    for scheme_name in &fixture.schemes {
        let scheme = scheme_of(scheme_name);
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&docs).expect("建索引");
        for query in &fixture.queries {
            let expected = &fixture.results[&query.qid][scheme_name];
            let actual = adapter.search(&query.text, fixture.topk).expect("检索");
            let expected_ids: Vec<&str> = expected.iter().map(|hit| hit.id.as_str()).collect();
            let actual_ids: Vec<&str> = actual.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(
                actual_ids, expected_ids,
                "faq-{scheme_name} 查询 {}（{:?}）的 top-{} doc id 序列与 Python 不一致",
                query.qid, query.text, fixture.topk
            );
            for ((_, score), hit) in actual.iter().zip(expected) {
                let tolerance = 1e-9 * hit.score.abs().max(1.0);
                assert!(
                    (score - hit.score).abs() <= tolerance,
                    "faq-{scheme_name} 查询 {} 文档 {} 分数 {score} ≠ {}",
                    query.qid,
                    hit.id,
                    hit.score
                );
            }
            checked += 1;
        }
    }
    // 5 方案 × 9 查询；数量对不上说明 fixture 被改过或遍历漏了。
    assert_eq!(checked, fixture.schemes.len() * fixture.queries.len());
}

/// fixture 必须真的覆盖三处兜底路径，否则 parity 过了也说明不了移植对齐。
#[test]
fn fixture_covers_fallback_paths() {
    let fixture = fixture();
    // 单字查询：bigram 逐文档退到字符覆盖率（否则会返回空）。
    assert!(
        !fixture.results["q4"]["bigram"].is_empty(),
        "q4 应覆盖单字查询的逐文档兜底"
    );
    // 整库无该双字：bigram 整库退到 char（「速度」不存在，但「度」在语料里）。
    assert!(
        !fixture.results["q6"]["bigram"].is_empty(),
        "q6 应覆盖整库 char 兜底"
    );
    // 空查询：返回全部文档且不过 min_score。
    assert_eq!(fixture.results["q7"]["hybrid"].len(), fixture.topk);
    // 完全超纲：word 没有整库兜底 → 空结果。
    assert!(
        fixture.results["q8"]["word"].is_empty(),
        "q8 应覆盖「无兜底方案返回空」"
    );
}
