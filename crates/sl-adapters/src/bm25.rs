//! `bm25` 适配器：自实现的 BM25（Okapi，k1=1.5 / b=0.75）。
//!
//! - 词项 = [`tokenize::bm25_terms`]：Unicode 字母数字连续串小写化；含 CJK 的连续串
//!   交给 jieba-rs 分词。**索引与查询共用同一个函数**，保证两侧词表同源。
//! - IDF 用 `ln(1 + (N - df + 0.5) / (df + 0.5))`（只增不减的形式，词项出现在所有
//!   文档里也不会变负分）。
//! - 查询词项去重后计分（BM25 的标准做法；重复词只表示用户强调，不改变排序语义）。
//! - 长度归一用文档词项数 / 全库平均词项数。
//!
//! 与 `faq-bm25` 的区别：那一个移植的是 FAQ 包在**双字**特征上的 BM25（中文 FAQ 语义），
//! 这一个是在真实词项上的通用 BM25，英文/代码语料上也能用。
//!
//! # 0.2.0：倒排实现
//!
//! 0.2.0 = 倒排实现，打分语义与 0.1.0 逐位一致（由 parity 测试与 scifact 基线核验）。
//! 索引期建 `词项 -> [(doc_idx, tf)]` 倒排表，查询期只遍历「查询词项命中的文档」，
//! 不再对每篇文档扫一遍查询词项。**累加顺序**仍是「按查询词项升序、逐文档累加」，
//! 因此每个文档的浮点求和顺序与 0.1.0 的逐文档全扫完全相同，分数逐位相等。

use crate::tokenize::bm25_terms;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::{BTreeSet, HashMap};

/// 词频饱和参数（越大越不饱和）。
pub const K1: f64 = 1.5;
/// 文档长度归一强度（0 = 不归一，1 = 完全归一）。
pub const B: f64 = 0.75;

/// 倒排表的一条记录：文档序号 + 该词项在这篇文档里的词频。
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u32,
    tf: u32,
}

pub struct Bm25Adapter {
    info: AdapterInfo,
    /// 语料文档 id，按建索引时的顺序；倒排表里的 `doc` 就是这个 Vec 的下标。
    ids: Vec<String>,
    /// 每篇文档的词项数（长度归一分母的一部分），与 `ids` 同序。
    doc_len: Vec<usize>,
    /// 词项字典：词项 -> 词项 id（`postings` / `idf` 的下标）。
    ///
    /// 用「字典 + 连续列表」而不是 `HashMap<String, Vec<Posting>>`：索引期每个词项
    /// 只分配一次键，查询期一次哈希拿到列表，列表在内存里连续。
    term_ids: HashMap<String, u32>,
    /// 倒排表，下标即词项 id；每条列表内按文档序号升序。
    postings: Vec<Vec<Posting>>,
    /// 词项 idf，下标与 `postings` 对齐。
    idf: Vec<f64>,
    avg_len: f64,
    /// 单查询复用的累加缓冲。约定：每次查询结束把命中文档清零，避免每查询 O(N) 清零。
    scores: Vec<f64>,
}

impl Default for Bm25Adapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Adapter {
    pub fn new() -> Self {
        let config = serde_json::json!({
            "k1": K1,
            "b": B,
            "tokenizer": "unicode-alnum-run/lowercase + jieba-rs(hmm) for CJK blocks",
            "query_terms": "dedup",
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
        });
        Self {
            info: AdapterInfo::new(
                "bm25",
                "bm25",
                "0.2.0",
                "本仓 crates/sl-adapters/src/{bm25,tokenize}.rs（自实现，非第三方检索库）",
            )
            .with_config(config),
            ids: Vec::new(),
            doc_len: Vec::new(),
            term_ids: HashMap::new(),
            postings: Vec::new(),
            idf: Vec::new(),
            avg_len: 0.0,
            scores: Vec::new(),
        }
    }
}

impl Adapter for Bm25Adapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        self.ids.clear();
        self.doc_len.clear();
        self.term_ids.clear();
        self.postings.clear();
        self.idf.clear();
        self.scores.clear();

        let mut total_len = 0usize;
        for (doc_index, doc) in docs.iter().enumerate() {
            let doc_id =
                u32::try_from(doc_index).expect("文档数超过 u32::MAX，倒排表的 doc 装不下");
            let terms = bm25_terms(&doc.full_text());
            // 先在本篇内聚词频，再逐词项追加到倒排表：每篇文档对同一个词项只产出
            // 一条 posting，且列表天然按文档序号升序（外层的 docs 顺序）。
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for term in &terms {
                *tf.entry(term.as_str()).or_insert(0) += 1;
            }
            for (term, count) in tf {
                let term_id = match self.term_ids.get(term) {
                    Some(&term_id) => term_id,
                    None => {
                        let term_id = u32::try_from(self.postings.len())
                            .expect("词项数超过 u32::MAX，倒排表的 id 装不下");
                        self.term_ids.insert(term.to_string(), term_id);
                        self.postings.push(Vec::new());
                        term_id
                    }
                };
                self.postings[term_id as usize].push(Posting {
                    doc: doc_id,
                    tf: count,
                });
            }
            total_len += terms.len();
            self.ids.push(doc.id.clone());
            self.doc_len.push(terms.len());
        }

        let n = docs.len() as f64;
        self.avg_len = if docs.is_empty() {
            0.0
        } else {
            total_len as f64 / n
        };
        self.idf = self
            .postings
            .iter()
            .map(|list| {
                let df = list.len() as f64;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .collect();
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let query_terms: BTreeSet<String> = bm25_terms(query).into_iter().collect();
        if query_terms.is_empty() || self.ids.is_empty() {
            return Ok(Vec::new());
        }
        let avg_len = self.avg_len.max(1e-9);
        if self.scores.len() < self.ids.len() {
            self.scores.resize(self.ids.len(), 0.0);
        }

        // 按查询词项升序累加：每个文档拿到的加法顺序与 0.1.0 的「逐文档、按词项升序」
        // 完全相同，这是分数逐位一致的前提。
        let scores = &mut self.scores;
        let mut touched: Vec<u32> = Vec::new();
        for term in &query_terms {
            let Some(&term_id) = self.term_ids.get(term) else {
                continue;
            };
            let idf = self.idf[term_id as usize];
            for posting in &self.postings[term_id as usize] {
                let doc = posting.doc as usize;
                let tf = f64::from(posting.tf);
                if scores[doc] == 0.0 {
                    touched.push(posting.doc);
                }
                let denom = tf + K1 * (1.0 - B + B * self.doc_len[doc] as f64 / avg_len);
                scores[doc] += idf * tf * (K1 + 1.0) / denom;
            }
        }

        let mut ranked: Vec<(u32, f64)> = touched
            .iter()
            .filter(|&&doc| scores[doc as usize] > 0.0)
            .map(|&doc| (doc, scores[doc as usize]))
            .collect();
        for &doc in &touched {
            scores[doc as usize] = 0.0;
        }
        // runner 还会再排一次；这里排好是为了「适配器自己返回的顺序」也可信。
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.ids[a.0 as usize].cmp(&self.ids[b.0 as usize]))
        });
        ranked.truncate(k);
        Ok(ranked
            .into_iter()
            .map(|(doc, score)| (self.ids[doc as usize].clone(), score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Doc> {
        vec![
            Doc::new("d1", "apple apple banana"),
            Doc::new("d2", "banana cherry"),
            Doc::new("d3", "apple cherry cherry"),
        ]
    }

    fn adapter() -> Bm25Adapter {
        let mut adapter = Bm25Adapter::new();
        adapter.index(&docs()).expect("建索引");
        adapter
    }

    /// 手算依据（k1=1.5, b=0.75, N=3, avg_len=8/3）：
    ///
    /// - 三个词项 df 都是 2 → idf = ln(1 + 1.5/2.5) = ln(1.6) = 0.47000362924573563
    /// - 长度分母 = tf + 0.375 + 1.125 * dl / (8/3)
    /// - d1(dl=3)：apple tf=2 → 0.47000362924573563 * 5 / 3.640625 = 0.6454985466；
    ///   cherry 不出现 → 0
    /// - d2(dl=2)：cherry tf=1 → 0.47000362924573563 * 2.5 / 2.21875 = 0.5295815541
    /// - d3(dl=3)：apple tf=1 → 0.47000362924573563 * 2.5 / 2.640625 = 0.4449738502；
    ///   cherry tf=2 → 0.6454985466，合计 1.0904723968
    #[test]
    fn bm25_scores_match_hand_computation() {
        let mut adapter = adapter();
        let hits = adapter.search("apple cherry", 3).expect("检索");
        assert_eq!(
            hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["d3", "d1", "d2"]
        );
        let expected = [
            ("d3", 1.090_472_396_8),
            ("d1", 0.645_498_546_6),
            ("d2", 0.529_581_554_1),
        ];
        for ((id, score), (want_id, want_score)) in hits.iter().zip(expected) {
            assert_eq!(id, want_id);
            assert!(
                (score - want_score).abs() < 1e-6,
                "{id} 分数 {score} ≠ {want_score}"
            );
        }
    }

    #[test]
    fn ties_break_by_doc_id_and_k_truncates() {
        let mut adapter = Bm25Adapter::new();
        // 两篇内容完全相同 → 同分；id 顺序与插入顺序相反，验证同分按 id 升序。
        adapter
            .index(&[Doc::new("z-doc", "alpha"), Doc::new("a-doc", "alpha")])
            .expect("建索引");
        let hits = adapter.search("alpha", 5).expect("检索");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "a-doc");
        assert_eq!(hits[1].0, "z-doc");
        assert!((hits[0].1 - hits[1].1).abs() < f64::EPSILON);

        let hits = adapter.search("alpha", 1).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a-doc");
    }

    #[test]
    fn cjk_is_segmented_by_jieba_and_out_of_corpus_terms_score_zero() {
        let mut adapter = Bm25Adapter::new();
        adapter
            .index(&[Doc::new("a", "报警阈值设置"), Doc::new("b", "日志留存天数")])
            .expect("建索引");
        let hits = adapter.search("阈值", 5).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");

        // 查询里的词项一个都不在语料里 → 空结果，而不是给所有文档 0 分。
        assert!(adapter.search("邮件服务器", 5).expect("检索").is_empty());
        // 空查询同样返回空结果。
        assert!(adapter.search("？？", 5).expect("检索").is_empty());
    }

    /// 少于 k 条命中时只返回命中文档（0.1.0 的 `score > 0` 过滤），不补零分文档。
    #[test]
    fn fewer_hits_than_k_returns_only_positive_hits() {
        let mut adapter = Bm25Adapter::new();
        adapter
            .index(&[
                Doc::new("a", "alpha"),
                Doc::new("b", "beta"),
                Doc::new("c", "gamma"),
            ])
            .expect("建索引");
        let hits = adapter.search("alpha", 10).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");
    }
}
