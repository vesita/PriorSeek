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

use crate::tokenize::bm25_terms;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::{BTreeSet, HashMap};

/// 词频饱和参数（越大越不饱和）。
pub const K1: f64 = 1.5;
/// 文档长度归一强度（0 = 不归一，1 = 完全归一）。
pub const B: f64 = 0.75;

/// 一篇文档的词项统计：只需要 tf 与词项数。
struct DocTerms {
    id: String,
    tf: HashMap<String, f64>,
    len: f64,
}

pub struct Bm25Adapter {
    info: AdapterInfo,
    docs: Vec<DocTerms>,
    idf: HashMap<String, f64>,
    avg_len: f64,
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
                env!("CARGO_PKG_VERSION"),
                "本仓 crates/sl-adapters/src/{bm25,tokenize}.rs（自实现，非第三方检索库）",
            )
            .with_config(config),
            docs: Vec::new(),
            idf: HashMap::new(),
            avg_len: 0.0,
        }
    }
}

impl Adapter for Bm25Adapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        self.docs.clear();
        self.idf.clear();

        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;
        for doc in docs {
            let terms = bm25_terms(&doc.full_text());
            let mut tf: HashMap<String, f64> = HashMap::new();
            for term in &terms {
                *tf.entry(term.clone()).or_insert(0.0) += 1.0;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            total_len += terms.len();
            self.docs.push(DocTerms {
                id: doc.id.clone(),
                tf,
                len: terms.len() as f64,
            });
        }

        let n = docs.len() as f64;
        self.avg_len = if docs.is_empty() {
            0.0
        } else {
            total_len as f64 / n
        };
        self.idf = df
            .into_iter()
            .map(|(term, count)| {
                let df = count as f64;
                (term, (1.0 + (n - df + 0.5) / (df + 0.5)).ln())
            })
            .collect();
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let query_terms: BTreeSet<String> = bm25_terms(query).into_iter().collect();
        if query_terms.is_empty() || self.docs.is_empty() {
            return Ok(Vec::new());
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
                let denom = tf + K1 * (1.0 - B + B * doc.len / avg_len);
                score += idf * tf * (K1 + 1.0) / denom;
            }
            if score > 0.0 {
                ranked.push((doc.id.clone(), score));
            }
        }
        // runner 还会再排一次；这里排好是为了「适配器自己返回的顺序」也可信。
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        Ok(ranked)
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
}
