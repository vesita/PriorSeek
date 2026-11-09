//! 倒排实现 vs 朴素全扫实现的一致性测试。
//!
//! 本文件里的 `NaiveFaq` / `NaiveBm25` 是 0.1.0 源码的搬运（逐文档扫全部查询特征），
//! 只作为**打分语义的独立 oracle**：不进生产路径，不追求性能。对比的是 doc id 序列
//! 和分数的 IEEE-754 位模式（`to_bits`）—— 任何求和顺序变化都会让测试红，而不只是
//! 「差不多相等」。
//!
//! 覆盖的语义角落：普通多特征查询、单字查询（bigram / bm25 的逐文档兜底）、
//! 整库零命中的整库 char 兜底、全停用词查询（word 兜底）、空查询（返回全部 0 分文档）、
//! 同分按 doc id 排序、命中数少于 k 时不补零、重复建索引。

use sl_adapters::tokenize::{bm25_terms, features, normalize};
use sl_adapters::{Bm25Adapter, FaqAdapter, Scheme};
use sl_core::{Adapter, Doc};
use std::collections::{BTreeMap, BTreeSet};

// ---- 0.1.0 的常量（冻结在测试里：naive 是历史语义的 oracle，不跟生产常量联动） ----

const HYBRID_W_WORD: f64 = 2.0;
const HYBRID_W_BIGRAM: f64 = 1.0;
const HYBRID_W_CHAR: f64 = 0.3;
const FAQ_BM25_K1: f64 = 1.5;
const FAQ_BM25_B: f64 = 0.75;
const MIN_SCORE: f64 = 0.0;
const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;

const SCHEMES: [Scheme; 5] = [
    Scheme::Char,
    Scheme::Bigram,
    Scheme::Word,
    Scheme::Hybrid,
    Scheme::Bm25,
];

/// 语料规模远小于这里的最大 k，所以「命中数少于 k」与「结果条数」都能被 k 卡住。
const KS: [usize; 5] = [1, 2, 3, 5, 100];

// ---------------------------------------------------------------- 朴素 faq 实现

struct NaiveDocFeatures {
    char_set: BTreeSet<String>,
    char_tf: BTreeMap<String, usize>,
    char_len: usize,
    bigram_set: BTreeSet<String>,
    bigram_tf: BTreeMap<String, usize>,
    bigram_len: usize,
    word_set: BTreeSet<String>,
}

impl NaiveDocFeatures {
    fn build(text: &str) -> Self {
        let features = features(text);
        Self {
            char_tf: counts(&features.chars),
            char_len: features.chars.len(),
            char_set: features.chars.into_iter().collect(),
            bigram_tf: counts(&features.bigrams),
            bigram_len: features.bigrams.len(),
            bigram_set: features.bigrams.into_iter().collect(),
            word_set: features.words.into_iter().collect(),
        }
    }
}

struct NaiveQueryFeatures {
    char: BTreeSet<String>,
    bigram: BTreeSet<String>,
    word: BTreeSet<String>,
}

impl NaiveQueryFeatures {
    fn build(query: &str) -> Self {
        let features = features(query);
        Self {
            char: features.chars.into_iter().collect(),
            bigram: features.bigrams.into_iter().collect(),
            word: features.words.into_iter().collect(),
        }
    }
}

fn counts(features: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for feature in features {
        *map.entry(feature.clone()).or_insert(0) += 1;
    }
    map
}

fn coverage(query_features: &BTreeSet<String>, doc_features: &BTreeSet<String>) -> f64 {
    if query_features.is_empty() {
        return 0.0;
    }
    let hit = query_features
        .iter()
        .filter(|feature| doc_features.contains(*feature))
        .count();
    hit as f64 / query_features.len() as f64
}

fn idf_of(sets: &[&BTreeSet<String>]) -> BTreeMap<String, f64> {
    let mut df: BTreeMap<&String, usize> = BTreeMap::new();
    for set in sets {
        for feature in set.iter() {
            *df.entry(feature).or_insert(0) += 1;
        }
    }
    let n = sets.len() as f64;
    df.into_iter()
        .map(|(feature, count)| {
            let count = count as f64;
            (
                feature.clone(),
                (1.0 + (n - count + 0.5) / (count + 0.5)).ln(),
            )
        })
        .collect()
}

struct NaiveFaq {
    ids: Vec<String>,
    features: Vec<NaiveDocFeatures>,
    idf_char: BTreeMap<String, f64>,
    idf_bigram: BTreeMap<String, f64>,
    avg_char_len: f64,
    avg_bigram_len: f64,
}

impl NaiveFaq {
    fn build(docs: &[Doc]) -> Self {
        let features: Vec<NaiveDocFeatures> = docs
            .iter()
            .map(|doc| NaiveDocFeatures::build(&doc.full_text()))
            .collect();
        let n = features.len();
        let avg_char_len = if n == 0 {
            0.0
        } else {
            features.iter().map(|doc| doc.char_len).sum::<usize>() as f64 / n as f64
        };
        let avg_bigram_len = if n == 0 {
            0.0
        } else {
            features.iter().map(|doc| doc.bigram_len).sum::<usize>() as f64 / n as f64
        };
        let char_sets: Vec<&BTreeSet<String>> = features.iter().map(|doc| &doc.char_set).collect();
        let bigram_sets: Vec<&BTreeSet<String>> =
            features.iter().map(|doc| &doc.bigram_set).collect();
        Self {
            ids: docs.iter().map(|doc| doc.id.clone()).collect(),
            idf_char: idf_of(&char_sets),
            idf_bigram: idf_of(&bigram_sets),
            features,
            avg_char_len,
            avg_bigram_len,
        }
    }

    fn search(&self, query: &str, k: usize, scheme: Scheme) -> Vec<(String, f64)> {
        if k == 0 {
            return Vec::new();
        }
        if normalize(query).is_empty() {
            let mut all: Vec<(usize, f64)> = (0..self.features.len()).map(|i| (i, 0.0)).collect();
            all.sort_by(|a, b| self.ids[a.0].cmp(&self.ids[b.0]));
            all.truncate(k);
            return all
                .into_iter()
                .map(|(doc, score)| (self.ids[doc].clone(), score))
                .collect();
        }
        let query_features = NaiveQueryFeatures::build(query);
        let mut scored = self.score_all(&query_features, scheme);
        if scored.is_empty() && matches!(scheme, Scheme::Bigram | Scheme::Bm25) {
            scored = self.score_all(&query_features, Scheme::Char);
        }
        scored.retain(|(_, score)| *score > MIN_SCORE);
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.ids[a.0].cmp(&self.ids[b.0]))
        });
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(doc, score)| (self.ids[doc].clone(), score))
            .collect()
    }

    fn score_all(&self, query: &NaiveQueryFeatures, scheme: Scheme) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = (0..self.features.len())
            .map(|i| (i, self.score_doc(query, i, scheme)))
            .filter(|(_, score)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored
    }

    fn score_doc(&self, query: &NaiveQueryFeatures, doc_index: usize, scheme: Scheme) -> f64 {
        let doc = &self.features[doc_index];
        match scheme {
            Scheme::Char => coverage(&query.char, &doc.char_set),
            Scheme::Bigram => {
                if query.bigram.is_empty() {
                    coverage(&query.char, &doc.char_set)
                } else {
                    coverage(&query.bigram, &doc.bigram_set)
                }
            }
            Scheme::Word => {
                if query.word.is_empty() {
                    coverage(&query.char, &doc.char_set)
                } else {
                    coverage(&query.word, &doc.word_set)
                }
            }
            Scheme::Hybrid => {
                let word = coverage(&query.word, &doc.word_set);
                let bigram = coverage(&query.bigram, &doc.bigram_set);
                let char = coverage(&query.char, &doc.char_set);
                HYBRID_W_WORD * word + HYBRID_W_BIGRAM * bigram + HYBRID_W_CHAR * char
            }
            Scheme::Bm25 => {
                if query.bigram.is_empty() {
                    self.bm25_features(
                        &query.char,
                        &doc.char_tf,
                        doc.char_len,
                        self.avg_char_len,
                        &self.idf_char,
                    )
                } else {
                    self.bm25_features(
                        &query.bigram,
                        &doc.bigram_tf,
                        doc.bigram_len,
                        self.avg_bigram_len,
                        &self.idf_bigram,
                    )
                }
            }
        }
    }

    fn bm25_features(
        &self,
        query_features: &BTreeSet<String>,
        tf: &BTreeMap<String, usize>,
        doc_len: usize,
        avg_len: f64,
        idf: &BTreeMap<String, f64>,
    ) -> f64 {
        let mut total = 0.0;
        for feature in query_features {
            let Some(freq) = tf.get(feature).copied() else {
                continue;
            };
            let freq = freq as f64;
            let denom = freq
                + FAQ_BM25_K1
                    * (1.0 - FAQ_BM25_B + FAQ_BM25_B * doc_len as f64 / avg_len.max(1e-9));
            let idf = idf.get(feature).copied().unwrap_or(0.0);
            total += idf * freq * (FAQ_BM25_K1 + 1.0) / denom;
        }
        total
    }
}

// ---------------------------------------------------------------- 朴素 bm25 实现

struct NaiveBm25Doc {
    id: String,
    tf: BTreeMap<String, f64>,
    len: f64,
}

struct NaiveBm25 {
    docs: Vec<NaiveBm25Doc>,
    idf: BTreeMap<String, f64>,
    avg_len: f64,
}

impl NaiveBm25 {
    fn build(docs: &[Doc]) -> Self {
        let mut df: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_len = 0usize;
        let mut doc_terms: Vec<NaiveBm25Doc> = Vec::new();
        for doc in docs {
            let terms = bm25_terms(&doc.full_text());
            let mut tf: BTreeMap<String, f64> = BTreeMap::new();
            for term in &terms {
                *tf.entry(term.clone()).or_insert(0.0) += 1.0;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            total_len += terms.len();
            doc_terms.push(NaiveBm25Doc {
                id: doc.id.clone(),
                tf,
                len: terms.len() as f64,
            });
        }
        let n = docs.len() as f64;
        Self {
            docs: doc_terms,
            avg_len: if docs.is_empty() {
                0.0
            } else {
                total_len as f64 / n
            },
            idf: df
                .into_iter()
                .map(|(term, count)| {
                    let df = count as f64;
                    (term, (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
                })
                .collect(),
        }
    }

    fn search(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        let query_terms: BTreeSet<String> = bm25_terms(query).into_iter().collect();
        if query_terms.is_empty() || self.docs.is_empty() {
            return Vec::new();
        }
        let avg_len = self.avg_len.max(1e-9);
        let mut ranked: Vec<(String, f64)> = Vec::new();
        for doc in &self.docs {
            let mut score = 0.0;
            for term in &query_terms {
                let Some(tf) = doc.tf.get(term) else { continue };
                let Some(idf) = self.idf.get(term) else {
                    continue;
                };
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc.len / avg_len);
                score += idf * tf * (BM25_K1 + 1.0) / denom;
            }
            if score > 0.0 {
                ranked.push((doc.id.clone(), score));
            }
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
    }
}

// ---------------------------------------------------------------- 语料与断言

/// 小语料：中文为主，混入英文/数字词项；含重复内容（同分）与带 title 的文档。
fn corpus() -> Vec<Doc> {
    vec![
        Doc::new("a", "报警阈值怎么设置"),
        Doc::new("b", "日志留存天数配置"),
        Doc::new("c", "邮件服务器的端口怎么改"),
        Doc::new("d", "报警阈值和日志留存"),
        Doc::new("e", "刀片更换"),
        Doc::new("f", "加速运行"),
        Doc::new("g", "最高温度"),
        Doc::new("h", "端口端口端口"),
        Doc::new("i", "阈值"),
        Doc::new("j", "完全无关的内容"),
        Doc::new("k", "PDF 报表 version 2"),
        Doc::new("l", "report PDF PDF report"),
        // 与 a 内容完全相同 → 同分，排序只能靠 doc id。
        Doc::new("m", "报警阈值怎么设置"),
        Doc {
            id: "n".to_string(),
            title: "邮件设置".to_string(),
            text: "报表导出失败".to_string(),
        },
    ]
}

/// 查询覆盖：普通查询 / 单字查询 / 全停用词 / 整库零命中 / 空查询 / 英文与数字词项。
fn queries() -> Vec<&'static str> {
    vec![
        "报警阈值怎么改",
        "报警阈值怎么设置",
        "端口",
        "端口端口",
        "刀",
        "速度",
        "量子纠缠",
        "怎么",
        "的",
        "了吗",
        "？？",
        "",
        "阈值",
        "日志 留存",
        "加速 温度",
        "邮件服务器 端口 设置",
        "完全无关",
        "报表",
        "报",
        "z",
        "pdf report",
        "PDF Report version 2",
        "version",
    ]
}

fn assert_bit_equal(context: &str, naive: &[(String, f64)], actual: &[(String, f64)]) {
    assert_eq!(
        naive.len(),
        actual.len(),
        "{context}：返回条数不同（naive {} 条，倒排 {} 条）",
        naive.len(),
        actual.len()
    );
    for (position, ((naive_id, naive_score), (actual_id, actual_score))) in
        naive.iter().zip(actual).enumerate()
    {
        assert_eq!(
            naive_id, actual_id,
            "{context}：第 {position} 条 doc id 不同（naive {naive_id}，倒排 {actual_id}）"
        );
        assert_eq!(
            naive_score.to_bits(),
            actual_score.to_bits(),
            "{context}：文档 {naive_id} 的分数位模式不同（naive {naive_score:?}，倒排 {actual_score:?}）"
        );
    }
}

// ---------------------------------------------------------------- 测试

#[test]
fn faq_schemes_match_naive_bit_for_bit() {
    let docs = corpus();
    let naive = NaiveFaq::build(&docs);
    for scheme in SCHEMES {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&docs).expect("建索引");
        for query in queries() {
            for k in KS {
                let want = naive.search(query, k, scheme);
                let got = adapter.search(query, k).expect("检索");
                assert_bit_equal(
                    &format!("faq-{} 查询 {query:?} k={k}", scheme.name()),
                    &want,
                    &got,
                );
            }
        }
    }
}

#[test]
fn bm25_matches_naive_bit_for_bit() {
    let docs = corpus();
    let naive = NaiveBm25::build(&docs);
    let mut adapter = Bm25Adapter::new();
    adapter.index(&docs).expect("建索引");
    for query in queries() {
        for k in KS {
            let want = naive.search(query, k);
            let got = adapter.search(query, k).expect("检索");
            assert_bit_equal(&format!("bm25 查询 {query:?} k={k}"), &want, &got);
        }
    }
}

/// 空查询是**唯一**会返回 0 分文档的路径：全部文档、按 doc id 升序、截断到 k。
#[test]
fn empty_query_returns_zero_score_docs_sorted_by_doc_id() {
    let docs = corpus();
    let mut expected_ids: Vec<String> = docs.iter().map(|doc| doc.id.clone()).collect();
    expected_ids.sort();
    for scheme in SCHEMES {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&docs).expect("建索引");
        for k in [3usize, 100] {
            let hits = adapter.search("？!?", k).expect("检索");
            assert_eq!(hits.len(), k.min(expected_ids.len()));
            for ((id, score), want) in hits.iter().zip(&expected_ids) {
                assert_eq!(id, want, "faq-{} k={k}", scheme.name());
                assert_eq!(score.to_bits(), 0.0f64.to_bits());
            }
        }
    }
    // bm25 没有「空查询不过滤」的约定：词项切不出来 → 空结果。
    let mut bm25 = Bm25Adapter::new();
    bm25.index(&docs).expect("建索引");
    assert!(bm25.search("？!?", 10).expect("检索").is_empty());
}

/// 命中数少于 k 时只返回命中文档（`min_score` 滤掉 0 分文档），**不补零到 k**。
#[test]
fn fewer_hits_than_k_are_not_padded_with_zero_score_docs() {
    let docs = corpus();
    for scheme in SCHEMES {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&docs).expect("建索引");
        let hits = adapter.search("阈值", 100).expect("检索");
        assert!(
            !hits.is_empty() && hits.len() < 100,
            "faq-{}：命中数应少于 k（实际 {} 条）",
            scheme.name(),
            hits.len()
        );
        assert!(
            hits.iter().all(|(_, score)| *score > 0.0),
            "faq-{}：返回集合里不应出现 0 分文档",
            scheme.name()
        );
        assert_bit_equal(
            &format!("faq-{} 查询 \"阈值\" k=100", scheme.name()),
            &NaiveFaq::build(&docs).search("阈值", 100, scheme),
            &hits,
        );
    }
}

/// 先在大语料上搜过、再用小语料重建索引：复用缓冲不能留下跨语料的脏数据。
#[test]
fn reindexing_smaller_corpus_after_search_stays_correct() {
    let full = corpus();
    let small = full[..4].to_vec();
    for scheme in SCHEMES {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&full).expect("建索引");
        for query in queries() {
            adapter.search(query, 100).expect("检索");
        }
        adapter.index(&small).expect("重建索引");
        let naive = NaiveFaq::build(&small);
        for query in queries() {
            let want = naive.search(query, 100, scheme);
            let got = adapter.search(query, 100).expect("检索");
            assert_bit_equal(
                &format!("faq-{} 重建后查询 {query:?}", scheme.name()),
                &want,
                &got,
            );
        }
    }

    let mut bm25 = Bm25Adapter::new();
    bm25.index(&full).expect("建索引");
    for query in queries() {
        bm25.search(query, 100).expect("检索");
    }
    bm25.index(&small).expect("重建索引");
    let naive = NaiveBm25::build(&small);
    for query in queries() {
        let want = naive.search(query, 100);
        let got = bm25.search(query, 100).expect("检索");
        assert_bit_equal(&format!("bm25 重建后查询 {query:?}"), &want, &got);
    }
}

/// 空语料不能 panic（runner 的契约是「返回少于 k 条合法」）。
#[test]
fn empty_corpus_returns_empty_results() {
    let mut bm25 = Bm25Adapter::new();
    bm25.index(&[]).expect("建索引");
    assert!(bm25.search("阈值", 10).expect("检索").is_empty());
    for scheme in SCHEMES {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(&[]).expect("建索引");
        assert!(adapter.search("阈值", 10).expect("检索").is_empty());
    }
}
