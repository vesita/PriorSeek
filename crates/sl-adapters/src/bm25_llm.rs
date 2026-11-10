//! `bm25-llm`：把 BM25 的切分单元换成 **LLM 的 BPE tokenizer**（不自研子词）。
//!
//! 与 `bm25` 的唯一区别就是 tokenizer；打分公式、参数（k1=1.5, b=0.75）、同分排序、过滤规则完全一致。
//! 这样 `bm25` vs `bm25-llm` 就是一次**单变量**对照，能干净地回答「换 tokenizer 值不值」。
//!
//! 词项直接用 token id（`u32`），不做字符串往返——索引期与查询期用同一个 tokenizer，
//! 天然同源，不会出现「查询侧与文档侧词表不同」的错位。

use crate::tokenizer_llm::LlmTokenizer;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::HashMap;

/// BM25 词频饱和参数（与 `bm25` 一致）。
pub const K1: f64 = 1.5;
/// 文档长度归一强度（与 `bm25` 一致）。
pub const B: f64 = 0.75;

#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u32,
    tf: u32,
}

pub struct Bm25LlmAdapter {
    info: AdapterInfo,
    tokenizer: LlmTokenizer,
    ids: Vec<String>,
    doc_len: Vec<usize>,
    /// token id → 倒排表（用 Vec 直寻址，id < vocab_size）。
    postings: Vec<Vec<Posting>>,
    idf: Vec<f64>,
    avg_len: f64,
    scores: Vec<f64>,
    touched: Vec<u32>,
    marked: Vec<bool>,
}

impl std::fmt::Debug for Bm25LlmAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25LlmAdapter")
            .field("adapter", &self.info.name)
            .field("tokenizer", &self.tokenizer.id)
            .field("docs", &self.ids.len())
            .finish()
    }
}

impl Bm25LlmAdapter {
    pub fn new() -> anyhow::Result<Self> {
        let tokenizer = LlmTokenizer::from_env_or_default(None)?;
        let mut config = tokenizer.config_json();
        if let Some(obj) = config.as_object_mut() {
            obj.insert("k1".into(), serde_json::json!(K1));
            obj.insert("b".into(), serde_json::json!(B));
            obj.insert("idf".into(), serde_json::json!("ln(1 + (N - df + 0.5) / (df + 0.5))"));
            obj.insert("query_terms".into(), serde_json::json!("dedup by token id"));
            obj.insert("compared_with".into(), serde_json::json!("bm25（词级切分）"));
        }
        let info = AdapterInfo::new(
            "bm25-llm",
            "bm25",
            "0.1.0",
            "本仓 crates/sl-adapters/src/{bm25_llm,tokenizer_llm}.rs（BM25 公式同 bm25；切分用 LLM BPE tokenizer，非自研）",
        )
        .with_config(config);
        Ok(Self {
            info,
            tokenizer,
            ids: Vec::new(),
            doc_len: Vec::new(),
            postings: Vec::new(),
            idf: Vec::new(),
            avg_len: 0.0,
            scores: Vec::new(),
            touched: Vec::new(),
            marked: Vec::new(),
        })
    }
}

impl Adapter for Bm25LlmAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        let vocab = self.tokenizer.vocab_size;
        self.ids.clear();
        self.doc_len.clear();
        self.postings = vec![Vec::new(); vocab];
        self.idf = vec![0.0; vocab];
        self.scores = vec![0.0; docs.len()];
        self.marked = vec![false; docs.len()];
        self.touched.clear();

        let mut df = vec![0u32; vocab];
        let mut total_len = 0usize;

        for (doc_index, doc) in docs.iter().enumerate() {
            let tokens = self.tokenizer.encode_ids(&doc.full_text());
            total_len += tokens.len();
            self.doc_len.push(tokens.len());

            let mut tf: HashMap<u32, u32> = HashMap::new();
            for token in tokens {
                *tf.entry(token).or_insert(0) += 1;
            }
            for (token, count) in tf {
                let slot = token as usize;
                df[slot] += 1;
                self.postings[slot].push(Posting { doc: doc_index as u32, tf: count });
            }
            self.ids.push(doc.id.clone());
        }

        let n = docs.len() as f64;
        for (token, dfi) in df.iter().enumerate() {
            if *dfi > 0 {
                let df_f = *dfi as f64;
                self.idf[token] = (1.0 + (n - df_f + 0.5) / (df_f + 0.5)).ln();
            }
        }
        self.avg_len = if docs.is_empty() { 0.0 } else { total_len as f64 / docs.len() as f64 };
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        for &doc in &self.touched {
            self.scores[doc as usize] = 0.0;
            self.marked[doc as usize] = false;
        }
        self.touched.clear();
        if k == 0 || self.ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut terms: Vec<u32> = Vec::new();
        for token in self.tokenizer.encode_ids(query) {
            if !terms.contains(&token) {
                terms.push(token);
            }
        }
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let avg_len = self.avg_len.max(1e-9);
        for token in terms {
            let idf = self.idf[token as usize];
            if idf <= 0.0 {
                continue;
            }
            for posting in &self.postings[token as usize] {
                let doc = posting.doc as usize;
                let len = self.doc_len[doc] as f64;
                let tf = posting.tf as f64;
                let denom = tf + K1 * (1.0 - B + B * len / avg_len);
                if !self.marked[doc] {
                    self.marked[doc] = true;
                    self.touched.push(posting.doc);
                }
                self.scores[doc] += idf * tf * (K1 + 1.0) / denom;
            }
        }

        let mut ranked: Vec<(String, f64)> = self
            .touched
            .iter()
            .filter_map(|&doc| {
                let score = self.scores[doc as usize];
                if score > 0.0 {
                    Some((self.ids[doc as usize].clone(), score))
                } else {
                    None
                }
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_if_available() -> Option<Bm25LlmAdapter> {
        Bm25LlmAdapter::new().ok()
    }

    #[test]
    fn ranks_document_containing_query_terms_first() {
        let Some(mut adapter) = adapter_if_available() else {
            eprintln!("跳过：找不到 tokenizer（先跑 fetch_tokenizer.py）");
            return;
        };
        let docs = vec![
            Doc::new("d1", "inflammation and monocyte recruitment in tissue"),
            Doc::new("d2", "unrelated text about geology and rocks"),
            Doc::new("d3", "monocytes respond to inflammatory signals"),
        ];
        adapter.index(&docs).unwrap();
        let hits = adapter.search("monocytes inflammation", 10).unwrap();
        assert!(!hits.is_empty());
        // d2 一个词项都不命中，不应出现
        assert!(hits.iter().all(|(id, _)| id != "d2"));
    }

    #[test]
    fn empty_query_returns_nothing_and_reindex_is_clean() {
        let Some(mut adapter) = adapter_if_available() else {
            eprintln!("跳过：找不到 tokenizer");
            return;
        };
        adapter.index(&[Doc::new("d1", "alpha beta")]).unwrap();
        assert!(adapter.search("", 10).unwrap().is_empty());
        // 用更小语料重建后，旧文档不应残留
        adapter.index(&[Doc::new("d9", "alpha beta")]).unwrap();
        let hits = adapter.search("alpha", 10).unwrap();
        assert!(hits.iter().all(|(id, _)| id == "d9"));
    }
}
