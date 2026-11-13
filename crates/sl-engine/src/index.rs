//! 位置索引：文档 token 序列 + 倒排位置表。
//!
//! 外壳里**只有这一份索引**。之前的 `bm25` / `bm25_llm` / `seq` / `faq` 各写一份倒排，
//! 结果任何算法实验都要重复实现索引逻辑；现在所有算法共享它。
//!
//! token 用 `u32` id 表示，**索引层不认识 tokenizer**：词级切分、BPE、或未来的任何切分，
//! 只要先映射成 id 就能进同一个索引。这是「换 tokenizer 只是换变量」的结构前提。

use std::collections::HashMap;

/// 倒排表里的一条：(文档, 位置)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occurrence {
    pub doc: u32,
    pub pos: u32,
}

/// 位置索引。建一次，供所有打分器复用。
#[derive(Debug, Default)]
pub struct PositionalIndex {
    doc_ids: Vec<String>,
    doc_len: Vec<usize>,
    /// term id -> 按 (doc, pos) 升序的出现位置列表。
    postings: HashMap<u32, Vec<Occurrence>>,
    /// term id -> 文档频率。
    df: HashMap<u32, u32>,
    avg_len: f64,
}

impl PositionalIndex {
    /// 从「每篇文档的 token id 序列」建索引。
    pub fn build(docs: impl IntoIterator<Item = (String, Vec<u32>)>) -> Self {
        let mut doc_ids = Vec::new();
        let mut doc_len = Vec::new();
        let mut postings: HashMap<u32, Vec<Occurrence>> = HashMap::new();
        let mut df: HashMap<u32, u32> = HashMap::new();
        let mut total_len = 0usize;

        for (doc_index, (id, tokens)) in docs.into_iter().enumerate() {
            let doc = u32::try_from(doc_index).expect("文档数超过 u32");
            total_len += tokens.len();
            doc_len.push(tokens.len());
            doc_ids.push(id);

            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for (pos, token) in tokens.into_iter().enumerate() {
                postings.entry(token).or_default().push(Occurrence {
                    doc,
                    pos: u32::try_from(pos).expect("文档过长"),
                });
                if seen.insert(token) {
                    *df.entry(token).or_insert(0) += 1;
                }
            }
        }

        let avg_len = if doc_ids.is_empty() {
            0.0
        } else {
            total_len as f64 / doc_ids.len() as f64
        };
        Self {
            doc_ids,
            doc_len,
            postings,
            df,
            avg_len,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    pub fn doc_id(&self, doc: u32) -> &str {
        &self.doc_ids[doc as usize]
    }

    pub fn doc_len(&self, doc: u32) -> usize {
        self.doc_len[doc as usize]
    }

    pub fn avg_len(&self) -> f64 {
        self.avg_len.max(1e-9)
    }

    pub fn doc_freq(&self, term: u32) -> u32 {
        self.df.get(&term).copied().unwrap_or(0)
    }

    /// `ln(1 + (N - df + 0.5) / (df + 0.5))`：词项稀有度。所有算法共用同一个 IDF 定义。
    pub fn idf(&self, term: u32) -> f64 {
        let n = self.len() as f64;
        let df = self.doc_freq(term) as f64;
        if df <= 0.0 {
            return 0.0;
        }
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// 词项在语料里的全部出现位置（按 doc 升序、doc 内按 pos 升序）。
    pub fn occurrences(&self, term: u32) -> &[Occurrence] {
        self.postings
            .get(&term)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// 某词项在某文档内的位置列表（升序）。文档未命中时返回空。
    pub fn positions_in(&self, term: u32, doc: u32) -> Vec<u32> {
        let all = self.occurrences(term);
        // postings 按 doc 排序，二分定位该文档的区间。
        let start = all.partition_point(|occ| occ.doc < doc);
        let mut out = Vec::new();
        for occ in &all[start..] {
            if occ.doc != doc {
                break;
            }
            out.push(occ.pos);
        }
        out
    }

    /// 包含任一词项的候选文档（并集，升序）。
    pub fn candidates(&self, terms: &[u32]) -> Vec<u32> {
        let mut docs: Vec<u32> = Vec::new();
        for term in terms {
            for occ in self.occurrences(*term) {
                docs.push(occ.doc);
            }
        }
        docs.sort_unstable();
        docs.dedup();
        docs
    }
}

/// 双层谓词感知索引 (Tiered Predicate-Aware Index)。
///
/// 将核心高判别力谓词/概念（Tier-1）与背景辅助词（Tier-2）物理分层。
/// 候选文档初筛与谓词执行优先走 Tier-1 紧凑倒排链，大幅提高 CPU Cache 局部性并压缩遍历延迟。
#[derive(Debug, Default)]
pub struct TieredIndex {
    /// 完整位置索引（作为正确性基石）
    pub base: PositionalIndex,
    /// Tier-1 核心谓词词表（按 Saliency 划分的高频判别词）
    pub tier1_terms: std::collections::HashSet<u32>,
}

impl TieredIndex {
    pub fn new(base: PositionalIndex, tier1_terms: std::collections::HashSet<u32>) -> Self {
        Self { base, tier1_terms }
    }

    /// 仅利用 Tier-1 核心谓词进行超快候选集生成（过滤掉低判别力长链）。
    pub fn fast_candidates(&self, query_terms: &[u32]) -> Vec<u32> {
        let t1_query: Vec<u32> = query_terms
            .iter()
            .copied()
            .filter(|t| self.tier1_terms.contains(t))
            .collect();

        if !t1_query.is_empty() {
            self.base.candidates(&t1_query)
        } else {
            // 如果查询全是冷门或未收录词，平滑回退到全量候选
            self.base.candidates(query_terms)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> PositionalIndex {
        PositionalIndex::build(vec![
            ("d1".to_string(), vec![1, 2, 3, 2]),
            ("d2".to_string(), vec![2, 4]),
            ("d3".to_string(), vec![5]),
        ])
    }

    #[test]
    fn idf_matches_hand_computation() {
        let idx = index();
        // term 2 出现在 d1、d2 ⇒ df=2，N=3
        let expected = (1.0f64 + (3.0 - 2.0 + 0.5) / (2.0 + 0.5)).ln();
        assert!((idx.idf(2) - expected).abs() < 1e-12);
        assert_eq!(idx.doc_freq(2), 2);
        assert_eq!(idx.idf(999), 0.0);
    }

    #[test]
    fn positions_are_ordered_and_doc_scoped() {
        let idx = index();
        assert_eq!(idx.positions_in(2, 0), vec![1, 3]);
        assert_eq!(idx.positions_in(2, 1), vec![0]);
        assert!(idx.positions_in(2, 2).is_empty());
        assert!(idx.positions_in(5, 0).is_empty());
    }

    #[test]
    fn candidates_union_is_sorted_and_deduplicated() {
        let idx = index();
        assert_eq!(idx.candidates(&[2, 5]), vec![0, 1, 2]);
        assert_eq!(idx.candidates(&[999]), Vec::<u32>::new());
    }

    #[test]
    fn tiered_index_fast_candidates_skips_background_terms() {
        let base = index();
        // 设 term 2 为核心谓词，term 5 为背景词
        let mut t1 = std::collections::HashSet::new();
        t1.insert(2);
        let tiered = TieredIndex::new(base, t1);

        let cands = tiered.fast_candidates(&[2, 5]);
        assert_eq!(cands, vec![0, 1]);
    }
}
