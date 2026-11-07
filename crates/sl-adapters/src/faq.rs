//! `faq-*` 适配器：移植 `~/coding/hub/faq-search` 的 5 种打分方案。
//!
//! 五个适配器共用一个索引与一套打分器，差别只在 `scheme`：
//! `faq-char` / `faq-bigram` / `faq-word` / `faq-hybrid` / `faq-bm25`。
//!
//! # 逐条对齐的语义（原实现在 scorer.py / index.py）
//!
//! 1. **覆盖率型打分是 OR，不是 AND**：`命中特征数 / 查询特征数`。文档多出来的内容
//!    不扣分，绝不会因为「多了一个查询里没出现的字」整条归零。
//! 2. **hybrid 三档权重**：词 2.0 / 双字 1.0 / 单字 0.3，三项覆盖率加权求和。
//! 3. **faq-bm25 跑在相邻双字特征上**：`idf = ln(1 + (N - df + 0.5) / (df + 0.5))`，
//!    `k1 = 1.5`，`b = 0.75`，用双字的 tf 与双字长度做长度归一。
//! 4. **三处兜底**（缺一不可，否则单字查询或超纲查询会返回空）：
//!    - `faq-bigram`：查询切不出双字（单字查询）→ 该文档退到**单字覆盖率**；
//!    - `faq-word`：查询分词后为空（整条都是停用词）→ 该文档退到单字覆盖率；
//!    - `faq-bm25`：查询切不出双字 → 退到**单字 BM25**（char 的 tf / idf / 平均长度）；
//!    - `faq-bigram` / `faq-bm25`：切得出特征但**整库零命中** → 整库退到 `char` 方案
//!      重算。必须整库重算：逐文档退化会把双字分与单字分混在同一张榜上排（量纲错误）。
//! 5. `min_score` 默认 0，只保留正分（零分 = 一个特征都没命中，不必进排序）。
//! 6. 空查询 / 纯标点查询退化为「不过滤」：返回全部文档（受 k 限制），分数 0，
//!    且**不过 min_score 门槛**（原包在这里提前返回）。
//!
//! 原包的 `SearchHit.fallback` 只是「这次结果来自退化路径」的可观测标记，不影响
//! 排名；本适配器不把它带出来（`Adapter` 只吐 `(doc_id, score)`），排名语义不变。
//!
//! # 与原包的实现差异（不影响结果）
//!
//! 原包对每篇文档都重新构造一次查询特征集合；这里把查询特征在**一个查询内只构造一次**
//! （`QueryFeatures`），文档侧特征在索引期算好。语义完全一致，但把 hybrid 在 5000 篇
//! 语料上的每查询开销从「每个特征 × 每篇文档一次集合构造」降到一个查询三次 ——
//! 延迟对比表里的数该反映算法本身，不该反映这种分配浪费。
//!
//! # 适用范围
//!
//! 原包是**中文 FAQ** 场景（问题 + 答案短文本、字符/双字/分词特征）。移到英文语料上
//! 会退化：`char` / `bigram` 变成字母级重合率，几乎任何文档都有分，区分度极低；
//! `word` / `hybrid` 的词特征依赖 jieba 对英文串的切法。英文语料应看 `bm25` / `rs`。
//! 这里保留原包语义是为了让「同一种算法在两类语料上的表现差异」本身成为可比的一行。

use crate::tokenize;
use anyhow::Context;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::{BTreeMap, BTreeSet};

/// hybrid 的权重：词命中权重最高（语义单元最准），双字兜底跨词匹配，单字最低。
pub const HYBRID_W_WORD: f64 = 2.0;
pub const HYBRID_W_BIGRAM: f64 = 1.0;
pub const HYBRID_W_CHAR: f64 = 0.3;

/// faq-bm25 的参数（与原包 scorer.py 一致）。
pub const FAQ_BM25_K1: f64 = 1.5;
pub const FAQ_BM25_B: f64 = 0.75;

/// 相关性门槛，默认 0 = 只滤掉零分（原包 `min_score` 的默认值）。
pub const MIN_SCORE: f64 = 0.0;

/// 5 种打分方案。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    Char,
    Bigram,
    Word,
    Hybrid,
    Bm25,
}

impl Scheme {
    /// 注册表里的名字 → 方案。
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "faq-char" => Some(Self::Char),
            "faq-bigram" => Some(Self::Bigram),
            "faq-word" => Some(Self::Word),
            "faq-hybrid" => Some(Self::Hybrid),
            "faq-bm25" => Some(Self::Bm25),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Bigram => "bigram",
            Self::Word => "word",
            Self::Hybrid => "hybrid",
            Self::Bm25 => "bm25",
        }
    }

    /// 整库零命中时是否退到 `char` 方案重算（原包 `FALLBACK_SCHEMES`）。
    /// `word` / `hybrid` 不在其中：分词命中是它们各自的核心语义。
    fn has_batch_char_fallback(self) -> bool {
        matches!(self, Self::Bigram | Self::Bm25)
    }
}

/// 一篇文档的预计算特征。索引只算一次，查询时反复使用。
struct DocFeatures {
    char_set: BTreeSet<String>,
    char_tf: BTreeMap<String, usize>,
    char_len: usize,
    bigram_set: BTreeSet<String>,
    bigram_tf: BTreeMap<String, usize>,
    bigram_len: usize,
    word_set: BTreeSet<String>,
}

impl DocFeatures {
    fn build(text: &str) -> Self {
        let features = tokenize::features(text);
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

/// 查询侧特征集合：一个查询只构造一次，逐文档打分时复用。
struct QueryFeatures {
    char: BTreeSet<String>,
    bigram: BTreeSet<String>,
    word: BTreeSet<String>,
}

impl QueryFeatures {
    fn build(query: &str) -> Self {
        let features = tokenize::features(query);
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

/// 覆盖率 = 命中的查询特征数 / 查询特征总数（OR 语义：文档多出来的内容不扣分）。
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

/// FAQ 索引：文档特征 + 全库统计量（IDF、平均长度）。
///
/// 统计量必须在**全库**上算，所以索引与文档集合绑在一起，而不是无状态的函数集合。
struct FaqIndex {
    ids: Vec<String>,
    features: Vec<DocFeatures>,
    idf_char: BTreeMap<String, f64>,
    idf_bigram: BTreeMap<String, f64>,
    avg_char_len: f64,
    avg_bigram_len: f64,
}

impl FaqIndex {
    fn build(docs: &[Doc]) -> Self {
        let features: Vec<DocFeatures> = docs
            .iter()
            .map(|doc| DocFeatures::build(&doc.full_text()))
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

    /// 对一个查询的全部文档打分，只保留正分，按分数降序、文档序号升序返回。
    ///
    /// 不在这里按 k 截断：截断前的同分排序要用 doc id，那是适配器层的事。
    fn score_all(&self, query: &QueryFeatures, scheme: Scheme) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = (0..self.features.len())
            .map(|i| (i, self.score_doc(query, i, scheme)))
            .filter(|(_, score)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored
    }

    /// 检索：返回 `(文档序号, 分数)`，已按「分数降序、doc id 升序」排好并截断到 k。
    ///
    /// 空查询 / 纯标点查询退化为「不过滤」：全部文档得 0 分，**且不过 min_score 门槛**
    /// （原包在这里提前返回，否则会被门槛滤成空列表，违反「空查询不返回空」的约定）。
    fn search(&self, query: &str, k: usize, scheme: Scheme) -> Vec<(usize, f64)> {
        if k == 0 {
            return Vec::new();
        }
        if tokenize::normalize(query).is_empty() {
            let mut all: Vec<(usize, f64)> = (0..self.features.len()).map(|i| (i, 0.0)).collect();
            // 原包按语料顺序给前 topk 条；适配器统一按 doc id 排（runner 对同分也是这么排的）。
            all.sort_by(|a, b| self.ids[a.0].cmp(&self.ids[b.0]));
            all.truncate(k);
            return all;
        }

        let query_features = QueryFeatures::build(query);
        let mut scored = self.score_all(&query_features, scheme);
        if scored.is_empty() && scheme.has_batch_char_fallback() {
            // 整库零命中 → 整库退到 char 特征重算（见模块文档第 4 条）。
            scored = self.score_all(&query_features, Scheme::Char);
        }
        scored.retain(|(_, score)| *score > MIN_SCORE);
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.ids[a.0].cmp(&self.ids[b.0]))
        });
        scored.truncate(k);
        scored
    }

    fn score_doc(&self, query: &QueryFeatures, doc_index: usize, scheme: Scheme) -> f64 {
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

    /// 在给定特征空间上算 BM25 分（只对文档里出现过的特征累加）。
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

pub struct FaqAdapter {
    info: AdapterInfo,
    scheme: Scheme,
    index: Option<FaqIndex>,
}

impl FaqAdapter {
    pub fn new(scheme: Scheme) -> Self {
        let name = format!("faq-{}", scheme.name());
        let mut config = serde_json::json!({
            "scheme": scheme.name(),
            "min_score": MIN_SCORE,
            "tokenizer": "normalize(P/Z/C/S 丢弃 + lower) + jieba-rs(hmm) + 停用词表",
            "fallback": "单字查询→逐文档 char 覆盖率；word 分词为空→逐文档 char；bigram/bm25 整库零命中→整库 char 重算",
        });
        if scheme == Scheme::Hybrid {
            config["hybrid_weights"] = serde_json::json!({
                "word": HYBRID_W_WORD,
                "bigram": HYBRID_W_BIGRAM,
                "char": HYBRID_W_CHAR
            });
        }
        if scheme == Scheme::Bm25 {
            config["bm25"] =
                serde_json::json!({ "k1": FAQ_BM25_K1, "b": FAQ_BM25_B, "features": "相邻双字" });
        }
        Self {
            info: AdapterInfo::new(
                name,
                "faq-scheme",
                "0.1.0",
                "~/coding/hub/faq-search 的 {normalize,tokenize,scorer,index}.py 的 Rust 移植（本仓 crates/sl-adapters/src/faq.rs）",
            )
            .with_config(config),
            scheme,
            index: None,
        }
    }
}

impl Adapter for FaqAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        self.index = Some(FaqIndex::build(docs));
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let index = self
            .index
            .as_ref()
            .context("faq 适配器还没建索引（先调 index）")?;
        let hits = index.search(query, k, self.scheme);
        Ok(hits
            .into_iter()
            .map(|(doc_index, score)| (index.ids[doc_index].clone(), score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Doc> {
        vec![
            Doc::new("a", "报警阈值怎么设置"),
            Doc::new("b", "日志留存天数配置"),
            Doc::new("c", "邮件服务器的端口怎么改"),
        ]
    }

    fn index() -> FaqIndex {
        FaqIndex::build(&docs())
    }

    #[test]
    fn coverage_is_or_semantics_not_and() {
        // 查询比文档多出「改」等字，文档仍然拿正分（AND 型实现会整条归零）。
        let idx = index();
        let query = QueryFeatures::build("报警阈值怎么改");
        let score = idx.score_doc(&query, 0, Scheme::Char);
        assert!(score > 0.0);
        let hits = idx.search("报警阈值怎么改", 3, Scheme::Char);
        // a（报警阈值怎么设置）与 c（邮件服务器的端口怎么改）有重合，b 一个字符都不重合。
        assert_eq!(hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 2]);
        assert!((hits[0].1 - 6.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn hybrid_weights_word_above_bigram_above_char() {
        // 「端口」是一个词：在文档 c 上词、双字、单字三项都命中，验证加权公式本身。
        let idx = index();
        let query = QueryFeatures::build("端口");
        let doc = &idx.features[2];
        let word = coverage(&query.word, &doc.word_set);
        let bigram = coverage(&query.bigram, &doc.bigram_set);
        let char = coverage(&query.char, &doc.char_set);
        assert!(word > 0.0 && bigram > 0.0 && char > 0.0);
        let hybrid = idx.score_doc(&query, 2, Scheme::Hybrid);
        let expected = HYBRID_W_WORD * word + HYBRID_W_BIGRAM * bigram + HYBRID_W_CHAR * char;
        assert!((hybrid - expected).abs() < 1e-12, "{hybrid} ≠ {expected}");
    }

    #[test]
    fn single_char_query_falls_back_to_char_coverage() {
        let idx = FaqIndex::build(&[Doc::new("a", "更换刀片"), Doc::new("b", "校验阈值")]);
        // bigram 切不出双字 → 逐文档退到单字覆盖率，而不是空结果。
        let hits = idx.search("刀", 5, Scheme::Bigram);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
        assert!((hits[0].1 - 1.0).abs() < 1e-12);

        // bm25 同理（退到单字 BM25）。
        let hits = idx.search("刀", 5, Scheme::Bm25);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    #[test]
    fn batch_fallback_when_no_bigram_hit_in_corpus() {
        // 查询「速度」的双字在整库都不存在，但「速」「度」两字分别出现在两篇文档里：
        // 整库退到 char 方案后两条都该命中（逐文档退化会得到空/量纲混排）。
        let idx = FaqIndex::build(&[Doc::new("a", "加速运行"), Doc::new("b", "最高温度")]);
        assert!(idx.search("速度", 5, Scheme::Bigram).len() == 2);
        assert!(idx.search("速度", 5, Scheme::Bm25).len() == 2);
    }

    #[test]
    fn empty_query_returns_all_docs() {
        let idx = index();
        let hits = idx.search("？？", 2, Scheme::Hybrid);
        assert_eq!(hits, vec![(0, 0.0), (1, 0.0)]);
    }

    #[test]
    fn zero_hit_without_fallback_scheme_returns_empty() {
        // word 方案没有整库兜底：查询词一个都不在语料里 → 空结果。
        let idx = index();
        assert!(idx.search("量子纠缠", 5, Scheme::Word).is_empty());
    }

    #[test]
    fn ties_break_by_doc_id() {
        let mut adapter = FaqAdapter::new(Scheme::Char);
        adapter
            .index(&[Doc::new("z", "阈值"), Doc::new("a", "阈值")])
            .expect("建索引");
        let hits = adapter.search("阈", 5).expect("检索");
        assert_eq!(
            hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["a", "z"]
        );
    }
}
