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
//! # 0.2.0：倒排实现
//!
//! 0.2.0 = 倒排实现，打分语义与 0.1.0 逐位一致（由 parity 测试与 scifact 基线核验）。
//! 索引期给三种特征各建一张倒排表：`特征 -> [(文档序号, tf)]`（单字 / 相邻双字带 tf，
//! 分词只记文档集合）。查询期只对**至少命中一个查询特征**的文档累加分数，不再对每篇
//! 文档扫一遍查询特征。三条做法保证结果与逐文档全扫逐位一致：
//!
//! - 覆盖率用**整数命中数**除以查询特征数：与 0.1.0 的集合交集计数同值，除法只做一次。
//! - hybrid 的加权求和顺序仍是「词 → 双字 → 单字」：每个文档的分数按
//!   `2.0*word + 1.0*bigram + 0.3*char` 的左结合顺序累加，加法顺序不变。
//! - `faq-bm25` 的累加顺序仍是「按查询特征升序、逐文档累加」，与 0.1.0 相同。
//!
//! 零分文档的位置与 0.1.0 相同：普通查询里它们被 `min_score` 门槛滤掉（所以命中数
//! 少于 k 时返回条数可以少于 k，**不补零**）；空查询路径照旧返回全部文档、按 doc id
//! 升序截断到 k（这条路径本来就不门槛过滤）。
//!
//! # 与 0.1.0 的实现差异（不影响结果）
//!
//! 0.1.0 对每篇文档都重新遍历一次查询特征集合；这里改成遍历倒排表，并复用一张单查询
//! 累加缓冲（查询结束只还原命中的位置）。语义不变，延迟对比表里的数该反映算法本身，
//! 不该反映这种逐文档扫描的浪费。
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
use std::collections::{BTreeSet, HashMap, HashSet};

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

/// 倒排表的一条记录：文档序号 + 该特征在这篇文档里的出现次数（tf）。
///
/// 覆盖率型打分只用 `doc`；`faq-bm25` 用 `tf`。分词特征只按集合计数，`tf` 恒为 1。
#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u32,
    tf: u32,
}

/// 一个特征空间的倒排表：特征字典（`特征 -> id`）+ 按 id 的倒排列表。
///
/// 用「字典 + 连续列表」而不是 `HashMap<String, Vec<Posting>>`：索引期每个特征只分配
/// 一次键，查询期一次哈希拿到列表，列表在内存里连续。每条列表内按文档序号升序。
#[derive(Default)]
struct InvertedIndex {
    ids: HashMap<String, u32>,
    lists: Vec<Vec<Posting>>,
}

impl InvertedIndex {
    fn insert(&mut self, feature: &str, posting: Posting) {
        let feature_id = match self.ids.get(feature) {
            Some(&feature_id) => feature_id,
            None => {
                let feature_id = u32::try_from(self.lists.len())
                    .expect("特征数超过 u32::MAX，倒排表的 id 装不下");
                self.ids.insert(feature.to_string(), feature_id);
                self.lists.push(Vec::new());
                feature_id
            }
        };
        self.lists[feature_id as usize].push(posting);
    }

    /// 查询特征对应的「特征 id + 倒排列表」；该特征在语料里没出现过时返回 `None`。
    fn postings(&self, feature: &str) -> Option<(u32, &[Posting])> {
        let &feature_id = self.ids.get(feature)?;
        Some((feature_id, self.lists[feature_id as usize].as_slice()))
    }

    /// 全库统计出的 idf，下标与特征 id 对齐。
    fn idf(&self, docs: usize) -> Vec<f64> {
        let n = docs as f64;
        self.lists
            .iter()
            .map(|list| {
                let df = list.len() as f64;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .collect()
    }
}

/// 查询侧特征集合：一个查询只构造一次，逐命中文档打分时复用。
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

/// 单查询复用的累加缓冲，避免每次查询都分配 / 清零 O(文档数) 的数组。
///
/// 约定：`FaqIndex::search` 返回前把用过的位置全部还原（counts 清零、scores / hit
/// 复位、touched / hits 清空），下一次查询直接复用。`prepare` 只增不减，缩小语料后
/// 多出来的尾巴不会被访问。
#[derive(Default)]
struct Scratch {
    /// 当前特征空间里每个文档命中的查询特征数（整数计数，逐位确定）。
    counts: Vec<u32>,
    /// 当前特征空间被命中的文档（文档序号升序）。
    touched: Vec<u32>,
    /// 候选文档的累加分数。
    scores: Vec<f64>,
    /// 文档是否已进入候选集合。
    hit: Vec<bool>,
    /// 候选集合，首次进入候选集时追加。
    hits: Vec<u32>,
}

impl Scratch {
    fn prepare(&mut self, docs: usize) {
        if self.counts.len() < docs {
            self.counts.resize(docs, 0);
        }
        if self.scores.len() < docs {
            self.scores.resize(docs, 0.0);
        }
        if self.hit.len() < docs {
            self.hit.resize(docs, false);
        }
    }
}

/// FAQ 索引：三张倒排表 + 全库统计量（IDF、平均长度）。
///
/// 统计量必须在**全库**上算，所以索引与文档集合绑在一起，而不是无状态的函数集合。
struct FaqIndex {
    ids: Vec<String>,
    char_index: InvertedIndex,
    bigram_index: InvertedIndex,
    word_index: InvertedIndex,
    /// 每篇文档的单字 / 双字特征数（`faq-bm25` 的长度归一），与 `ids` 同序。
    char_len: Vec<u32>,
    bigram_len: Vec<u32>,
    idf_char: Vec<f64>,
    idf_bigram: Vec<f64>,
    avg_char_len: f64,
    avg_bigram_len: f64,
}

impl FaqIndex {
    fn build(docs: &[Doc]) -> Self {
        let mut index = Self {
            ids: Vec::with_capacity(docs.len()),
            char_index: InvertedIndex::default(),
            bigram_index: InvertedIndex::default(),
            word_index: InvertedIndex::default(),
            char_len: Vec::with_capacity(docs.len()),
            bigram_len: Vec::with_capacity(docs.len()),
            idf_char: Vec::new(),
            idf_bigram: Vec::new(),
            avg_char_len: 0.0,
            avg_bigram_len: 0.0,
        };
        let mut total_char_len = 0usize;
        let mut total_bigram_len = 0usize;
        for (doc_pos, doc) in docs.iter().enumerate() {
            let doc_id = u32::try_from(doc_pos).expect("文档数超过 u32::MAX，倒排表的 doc 装不下");
            let features = tokenize::features(&doc.full_text());

            // 单字：tf 要给 faq-bm25 的单字兜底路径用。
            let mut char_tf: HashMap<&str, u32> = HashMap::new();
            for ch in &features.chars {
                *char_tf.entry(ch.as_str()).or_insert(0) += 1;
            }
            for (feature, tf) in char_tf {
                index
                    .char_index
                    .insert(feature, Posting { doc: doc_id, tf });
            }

            // 相邻双字：同样记 tf。
            let mut bigram_tf: HashMap<&str, u32> = HashMap::new();
            for bigram in &features.bigrams {
                *bigram_tf.entry(bigram.as_str()).or_insert(0) += 1;
            }
            for (feature, tf) in bigram_tf {
                index
                    .bigram_index
                    .insert(feature, Posting { doc: doc_id, tf });
            }

            // 分词：覆盖率只看「文档里有没有这个词」，重复词只记一次（0.1.0 用 BTreeSet）。
            let words: HashSet<&str> = features.words.iter().map(String::as_str).collect();
            for word in words {
                index
                    .word_index
                    .insert(word, Posting { doc: doc_id, tf: 1 });
            }

            total_char_len += features.chars.len();
            total_bigram_len += features.bigrams.len();
            index
                .char_len
                .push(u32::try_from(features.chars.len()).expect("单篇文档字数超过 u32::MAX"));
            index
                .bigram_len
                .push(u32::try_from(features.bigrams.len()).expect("单篇文档双字数超过 u32::MAX"));
            index.ids.push(doc.id.clone());
        }

        let n = docs.len();
        if n > 0 {
            index.avg_char_len = total_char_len as f64 / n as f64;
            index.avg_bigram_len = total_bigram_len as f64 / n as f64;
        }
        index.idf_char = index.char_index.idf(n);
        index.idf_bigram = index.bigram_index.idf(n);
        index
    }

    /// 检索：返回 `(文档序号, 分数)`，已按「分数降序、doc id 升序」排好并截断到 k。
    ///
    /// 空查询 / 纯标点查询退化为「不过滤」：全部文档得 0 分，**且不过 min_score 门槛**
    /// （原包在这里提前返回，否则会被门槛滤成空列表，违反「空查询不返回空」的约定）。
    fn search(
        &self,
        query: &str,
        k: usize,
        scheme: Scheme,
        scratch: &mut Scratch,
    ) -> Vec<(u32, f64)> {
        if k == 0 {
            return Vec::new();
        }
        if tokenize::normalize(query).is_empty() {
            let mut all: Vec<(u32, f64)> =
                (0..self.ids.len() as u32).map(|doc| (doc, 0.0)).collect();
            // 原包按语料顺序给前 topk 条；适配器统一按 doc id 排（runner 对同分也是这么排的）。
            all.sort_by(|a, b| self.ids[a.0 as usize].cmp(&self.ids[b.0 as usize]));
            all.truncate(k);
            return all;
        }

        scratch.prepare(self.ids.len());
        let query_features = QueryFeatures::build(query);
        let mut scored = self.score(&query_features, scheme, scratch);
        if scored.is_empty() && scheme.has_batch_char_fallback() {
            // 整库零命中 → 整库退到 char 特征重算（见模块文档第 4 条）。
            scored = self.score(&query_features, Scheme::Char, scratch);
        }
        scored.retain(|(_, score)| *score > MIN_SCORE);
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.ids[a.0 as usize].cmp(&self.ids[b.0 as usize]))
        });
        scored.truncate(k);
        scored
    }

    /// 只在「至少命中一个查询特征」的文档上累加分数；未命中的文档保持 0 分。
    fn score(
        &self,
        query: &QueryFeatures,
        scheme: Scheme,
        scratch: &mut Scratch,
    ) -> Vec<(u32, f64)> {
        match scheme {
            Scheme::Char => coverage_scores(&self.char_index, &query.char, scratch),
            Scheme::Bigram => {
                if query.bigram.is_empty() {
                    coverage_scores(&self.char_index, &query.char, scratch)
                } else {
                    coverage_scores(&self.bigram_index, &query.bigram, scratch)
                }
            }
            Scheme::Word => {
                if query.word.is_empty() {
                    coverage_scores(&self.char_index, &query.char, scratch)
                } else {
                    coverage_scores(&self.word_index, &query.word, scratch)
                }
            }
            Scheme::Hybrid => {
                // 三项按「词 → 双字 → 单字」累加，与 0.1.0 的
                // `2.0*word + 1.0*bigram + 0.3*char` 左结合求值顺序一致。
                add_coverage(&self.word_index, &query.word, HYBRID_W_WORD, scratch);
                add_coverage(&self.bigram_index, &query.bigram, HYBRID_W_BIGRAM, scratch);
                add_coverage(&self.char_index, &query.char, HYBRID_W_CHAR, scratch);
                collect_scores(scratch)
            }
            Scheme::Bm25 => {
                if query.bigram.is_empty() {
                    bm25_scores(
                        &self.char_index,
                        &self.idf_char,
                        &self.char_len,
                        self.avg_char_len,
                        &query.char,
                        scratch,
                    )
                } else {
                    bm25_scores(
                        &self.bigram_index,
                        &self.idf_bigram,
                        &self.bigram_len,
                        self.avg_bigram_len,
                        &query.bigram,
                        scratch,
                    )
                }
            }
        }
    }
}

/// 覆盖率型打分的一趟：统计每个文档命中的查询特征数（整数，逐位确定）。
fn count_hits(index: &InvertedIndex, query: &BTreeSet<String>, scratch: &mut Scratch) {
    for feature in query {
        let Some((_, list)) = index.postings(feature) else {
            continue;
        };
        for posting in list {
            let doc = posting.doc as usize;
            if scratch.counts[doc] == 0 {
                scratch.touched.push(posting.doc);
            }
            scratch.counts[doc] += 1;
        }
    }
}

/// 覆盖率方案的单空间打分：命中数 / 查询特征数，只保留正分。
fn coverage_scores(
    index: &InvertedIndex,
    query: &BTreeSet<String>,
    scratch: &mut Scratch,
) -> Vec<(u32, f64)> {
    count_hits(index, query, scratch);
    let total = query.len() as f64;
    let mut scored = Vec::with_capacity(scratch.touched.len());
    // 逆序弹出只是为了让「清 counts」与「取 touched」共用同一个可变借用。
    while let Some(doc) = scratch.touched.pop() {
        let coverage = scratch.counts[doc as usize] as f64 / total;
        if coverage > 0.0 {
            scored.push((doc, coverage));
        }
        scratch.counts[doc as usize] = 0;
    }
    scored
}

/// 把当前空间的覆盖率按权重累加进候选集。
///
/// 首次进入候选集的文档直接赋值（等价于 `0.0 + 该项`），已有分数的文档做加法 ——
/// 与 0.1.0 的三项加权求和左结合顺序一致，浮点结果逐位相同。
fn add_coverage(
    index: &InvertedIndex,
    query: &BTreeSet<String>,
    weight: f64,
    scratch: &mut Scratch,
) {
    count_hits(index, query, scratch);
    if query.is_empty() {
        // 空查询集对每个文档都贡献 0.0；counts / touched 也必为空，没有要清理的。
        return;
    }
    let total = query.len() as f64;
    while let Some(doc) = scratch.touched.pop() {
        let doc_pos = doc as usize;
        let coverage = scratch.counts[doc_pos] as f64 / total;
        if scratch.hit[doc_pos] {
            scratch.scores[doc_pos] += weight * coverage;
        } else {
            scratch.hit[doc_pos] = true;
            scratch.hits.push(doc);
            scratch.scores[doc_pos] = weight * coverage;
        }
        scratch.counts[doc_pos] = 0;
    }
}

/// 单字 / 相邻双字特征上的 BM25 一趟：按查询特征升序、逐文档累加。
///
/// 累加顺序（外层查询特征升序、内层文档升序）让每个文档拿到的加法顺序与 0.1.0 的
/// 「逐文档、按查询特征升序」完全相同，分数逐位相等。
fn bm25_scores(
    index: &InvertedIndex,
    idf: &[f64],
    doc_len: &[u32],
    avg_len: f64,
    query: &BTreeSet<String>,
    scratch: &mut Scratch,
) -> Vec<(u32, f64)> {
    let avg_len = avg_len.max(1e-9);
    for feature in query {
        let Some((feature_id, list)) = index.postings(feature) else {
            continue;
        };
        let weight = idf[feature_id as usize];
        for posting in list {
            let doc = posting.doc as usize;
            let tf = f64::from(posting.tf);
            let denom =
                tf + FAQ_BM25_K1 * (1.0 - FAQ_BM25_B + FAQ_BM25_B * doc_len[doc] as f64 / avg_len);
            let contribution = weight * tf * (FAQ_BM25_K1 + 1.0) / denom;
            if !scratch.hit[doc] {
                scratch.hit[doc] = true;
                scratch.hits.push(posting.doc);
            }
            scratch.scores[doc] += contribution;
        }
    }
    collect_scores(scratch)
}

/// 收集候选集里的正分文档，并把 `scores` / `hit` / `hits` 复位。
fn collect_scores(scratch: &mut Scratch) -> Vec<(u32, f64)> {
    let mut scored = Vec::with_capacity(scratch.hits.len());
    while let Some(doc) = scratch.hits.pop() {
        let doc_pos = doc as usize;
        let score = scratch.scores[doc_pos];
        if score > 0.0 {
            scored.push((doc, score));
        }
        scratch.scores[doc_pos] = 0.0;
        scratch.hit[doc_pos] = false;
    }
    scored
}

pub struct FaqAdapter {
    info: AdapterInfo,
    scheme: Scheme,
    index: Option<FaqIndex>,
    scratch: Scratch,
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
                "0.2.0",
                "~/coding/hub/faq-search 的 {normalize,tokenize,scorer,index}.py 的 Rust 移植（本仓 crates/sl-adapters/src/faq.rs）",
            )
            .with_config(config),
            scheme,
            index: None,
            scratch: Scratch::default(),
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
        let Self {
            index,
            scheme,
            scratch,
            ..
        } = self;
        let index = index
            .as_ref()
            .context("faq 适配器还没建索引（先调 index）")?;
        let hits = index.search(query, k, *scheme, scratch);
        Ok(hits
            .into_iter()
            .map(|(doc_pos, score)| (index.ids[doc_pos as usize].clone(), score))
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

    fn indexed(scheme: Scheme, docs: &[Doc]) -> FaqAdapter {
        let mut adapter = FaqAdapter::new(scheme);
        adapter.index(docs).expect("建索引");
        adapter
    }

    #[test]
    fn coverage_is_or_semantics_not_and() {
        // 查询比文档多出「改」等字，文档仍然拿正分（AND 型实现会整条归零）。
        let mut adapter = indexed(Scheme::Char, &docs());
        let hits = adapter.search("报警阈值怎么改", 3).expect("检索");
        // a（报警阈值怎么设置）与 c（邮件服务器的端口怎么改）有重合，b 一个字符都不重合。
        assert_eq!(
            hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
        assert!((hits[0].1 - 6.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn hybrid_weights_word_above_bigram_above_char() {
        // 「端口」是一个词：在文档 c 上词、双字、单字三项都命中，验证加权公式本身。
        let mut adapter = indexed(Scheme::Hybrid, &docs());
        let hits = adapter.search("端口", 5).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "c");
        let expected = HYBRID_W_WORD * 1.0 + HYBRID_W_BIGRAM * 1.0 + HYBRID_W_CHAR * 1.0;
        assert!(
            (hits[0].1 - expected).abs() < 1e-12,
            "{} ≠ {expected}",
            hits[0].1
        );
    }

    #[test]
    fn single_char_query_falls_back_to_char_coverage() {
        let docs = [Doc::new("a", "更换刀片"), Doc::new("b", "校验阈值")];
        // bigram 切不出双字 → 逐文档退到单字覆盖率，而不是空结果。
        let mut bigram = indexed(Scheme::Bigram, &docs);
        let hits = bigram.search("刀", 5).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");
        assert!((hits[0].1 - 1.0).abs() < 1e-12);

        // bm25 同理（退到单字 BM25）。
        let mut bm25 = indexed(Scheme::Bm25, &docs);
        let hits = bm25.search("刀", 5).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");
    }

    #[test]
    fn batch_fallback_when_no_bigram_hit_in_corpus() {
        // 查询「速度」的双字在整库都不存在，但「速」「度」两字分别出现在两篇文档里：
        // 整库退到 char 方案后两条都该命中（逐文档退化会得到空/量纲混排）。
        let docs = [Doc::new("a", "加速运行"), Doc::new("b", "最高温度")];
        let mut bigram = indexed(Scheme::Bigram, &docs);
        assert_eq!(bigram.search("速度", 5).expect("检索").len(), 2);
        let mut bm25 = indexed(Scheme::Bm25, &docs);
        assert_eq!(bm25.search("速度", 5).expect("检索").len(), 2);
    }

    #[test]
    fn empty_query_returns_all_docs() {
        let mut adapter = indexed(Scheme::Hybrid, &docs());
        let hits = adapter.search("？？", 2).expect("检索");
        // 空查询是唯一会返回 0 分文档的路径：按 doc id 升序截断到 k。
        assert_eq!(hits, vec![("a".to_string(), 0.0), ("b".to_string(), 0.0)]);
    }

    #[test]
    fn zero_hit_without_fallback_scheme_returns_empty() {
        // word 方案没有整库兜底：查询词一个都不在语料里 → 空结果。
        let mut adapter = indexed(Scheme::Word, &docs());
        assert!(adapter.search("量子纠缠", 5).expect("检索").is_empty());
    }

    /// 命中数少于 k 时只返回命中文档：0 分文档被 `min_score` 门槛滤掉，不补零。
    #[test]
    fn fewer_hits_than_k_returns_only_positive_hits() {
        let mut adapter = indexed(Scheme::Char, &docs());
        let hits = adapter.search("阈值", 10).expect("检索");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "a");
    }

    #[test]
    fn ties_break_by_doc_id() {
        let mut adapter = indexed(
            Scheme::Char,
            &[Doc::new("z", "阈值"), Doc::new("a", "阈值")],
        );
        let hits = adapter.search("阈", 5).expect("检索");
        assert_eq!(
            hits.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["a", "z"]
        );
    }
}
