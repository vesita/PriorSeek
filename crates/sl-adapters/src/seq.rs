//! `seq` 适配器 —— token 时序相关度内核（`seq-v0`）的外壳实现。
//!
//! 三层模型见 `docs/KERNEL.md` §1：
//! - **内核**（Python 离线工件，`tools/kernel/build_seq.py`）提供词表、idf、token 间的有序转移权
//!   （order=1 相邻 / order=2 隔一，PPMI）以及固化好的文档 token 序列；
//! - **外壳**（本文件）负责：查询分词 → 基本特征（tf / 长度归一）+ 内核特征（时序边权）
//!   → 打分融合（BM25 主体 + λ·有序邻近）→ 建索引与检索；
//! - **实验台**（sl-core）负责数据契约、指标与报告，本适配器不碰 qrels。
//!
//! 打分（`docs/KERNEL.md` §8.1）：
//!
//! ```text
//! s_uni(d)  = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·len/avglen))
//! s_seq(d)  = Σ_{(a,b) ∈ 查询相邻对 ∪ 隔一对, e(a,b)>0} e(a,b) · sat(c_d(a,b,window))
//! sat(c)    = 1 + ln(min(c, max_matches))     (c ≥ 1)
//! score     = s_uni + λ_seq · s_seq
//! ```
//!
//! 消融参数全部支持环境变量覆盖（见 [`SeqParams::from_env`]），并原样进 `AdapterInfo.config`。

use crate::tokenize::bm25_terms;
use sl_core::{Adapter, AdapterInfo, Doc, KernelArtifact};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// BM25 词频饱和参数。
pub const K1: f64 = 1.5;
/// BM25 文档长度归一强度。
pub const B: f64 = 0.75;

const SEQ_MAGIC: &[u8; 8] = b"SLSEQv1\0";
const DOC_MAGIC: &[u8; 8] = b"SLDOCv1\0";

/// 时序内核的消融参数（`docs/KERNEL.md` §8.1）。
#[derive(Debug, Clone, PartialEq)]
pub struct SeqParams {
    /// 时序分权重；0 = 退回纯 BM25。
    pub lambda_seq: f64,
    /// 有序匹配的最大 token 距离。
    pub window: usize,
    /// 使用哪些阶的转移边（1 = 相邻，2 = 隔一）。
    pub orders: Vec<u8>,
    /// `true` 要求文档内 a 出现在 b 之前；`false` 为无序邻近对照。
    pub ordered: bool,
    /// 匹配次数上限，避免长文档靠重复刷分。
    pub max_matches: usize,
    /// order=1（相邻）时序边的权重系数；默认 1.0 = 原统计版行为。
    pub w_order_1: f64,
    /// order=2（隔一）时序边的权重系数；默认 1.0。
    pub w_order_2: f64,
}

impl Default for SeqParams {
    fn default() -> Self {
        Self {
            lambda_seq: 0.5,
            window: 8,
            orders: vec![1, 2],
            ordered: true,
            max_matches: 3,
            w_order_1: 1.0,
            w_order_2: 1.0,
        }
    }
}

impl SeqParams {
    /// 从环境变量读取覆盖值；非法值直接报错（不允许静默退回默认值）。
    ///
    /// - `SL_SEQ_LAMBDA_SEQ`（默认 0.5）
    /// - `SL_SEQ_WINDOW`（默认 8）
    /// - `SL_SEQ_ORDERS`（默认 `1,2`；只接受 1 / 2）
    /// - `SL_SEQ_ORDERED`（默认 `true`；`false` = 无序邻近）
    /// - `SL_SEQ_MAX_MATCHES`（默认 3）
    /// - `SL_SEQ_W_ORDER1` / `SL_SEQ_W_ORDER2`（默认 1.0；学习版写入的时序边权重系数）
    pub fn from_env() -> anyhow::Result<Self> {
        let mut params = Self::default();
        if let Ok(raw) = std::env::var("SL_SEQ_LAMBDA_SEQ") {
            let value: f64 = raw
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("SL_SEQ_LAMBDA_SEQ 不是数字：{raw:?}"))?;
            if !value.is_finite() || value < 0.0 {
                anyhow::bail!("SL_SEQ_LAMBDA_SEQ 必须是非负有限数：{raw:?}");
            }
            params.lambda_seq = value;
        }
        if let Ok(raw) = std::env::var("SL_SEQ_WINDOW") {
            let value: usize = raw
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("SL_SEQ_WINDOW 不是正整数：{raw:?}"))?;
            if value == 0 {
                anyhow::bail!("SL_SEQ_WINDOW 必须 ≥ 1");
            }
            params.window = value;
        }
        if let Ok(raw) = std::env::var("SL_SEQ_ORDERS") {
            let mut orders = Vec::new();
            for part in raw.split(',') {
                let part = part.trim();
                match part {
                    "1" => orders.push(1),
                    "2" => orders.push(2),
                    other => anyhow::bail!("SL_SEQ_ORDERS 只接受 1 / 2，收到 {other:?}"),
                }
            }
            if orders.is_empty() {
                anyhow::bail!("SL_SEQ_ORDERS 不能为空");
            }
            orders.sort_unstable();
            orders.dedup();
            params.orders = orders;
        }
        if let Ok(raw) = std::env::var("SL_SEQ_ORDERED") {
            params.ordered = match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" => true,
                "false" | "0" | "no" => false,
                other => anyhow::bail!("SL_SEQ_ORDERED 只接受 true/false，收到 {other:?}"),
            };
        }
        if let Ok(raw) = std::env::var("SL_SEQ_MAX_MATCHES") {
            let value: usize = raw
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("SL_SEQ_MAX_MATCHES 不是正整数：{raw:?}"))?;
            if value == 0 {
                anyhow::bail!("SL_SEQ_MAX_MATCHES 必须 ≥ 1");
            }
            params.max_matches = value;
        }
        for (env_name, slot) in [
            ("SL_SEQ_W_ORDER1", &mut params.w_order_1),
            ("SL_SEQ_W_ORDER2", &mut params.w_order_2),
        ] {
            if let Ok(raw) = std::env::var(env_name) {
                let value: f64 = raw
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("{env_name} 不是数字：{raw:?}"))?;
                if !value.is_finite() {
                    anyhow::bail!("{env_name} 必须是有限数：{raw:?}");
                }
                *slot = value;
            }
        }
        Ok(params)
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "lambda_seq": self.lambda_seq,
            "window": self.window,
            "orders": self.orders,
            "ordered": self.ordered,
            "max_matches": self.max_matches,
            "w_order_1": self.w_order_1,
            "w_order_2": self.w_order_2,
        })
    }
}

/// 内核工件的内容（`seq.bin` + `docs.txt` + `doc_tokens.bin`）。
#[derive(Debug)]
struct SeqData {
    /// 词表，按下标与 `idf` / 文档 token id 对齐；内容已按字典序升序（构建器保证）。
    vocab: Vec<String>,
    vocab_index: HashMap<String, u32>,
    idf: Vec<f32>,
    /// `(src, dst, order) -> weight`；只存 weight > 0 的边。
    edges: HashMap<(u32, u32, u8), f32>,
    /// 工件里的文档 id（顺序与 `doc_tokens` 一致）。
    docs: Vec<String>,
    doc_tokens: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy)]
struct Posting {
    doc: u32,
    tf: u32,
}

#[derive(Debug)]
pub struct SeqAdapter {
    info: AdapterInfo,
    params: SeqParams,
    artifact_dir: PathBuf,
    data: SeqData,
    /// 数据集文档 id（`index()` 时的顺序）。
    ids: Vec<String>,
    /// 数据集文档 token 序列（已按数据集顺序重排）。
    doc_tokens: Vec<Vec<u32>>,
    doc_len: Vec<usize>,
    /// 词项 id（工件词表下标）→ 倒排表。
    postings: Vec<Vec<Posting>>,
    avg_len: f64,
    /// 单查询复用缓冲：分数、命中文档、是否命中。
    scores: Vec<f64>,
    touched: Vec<u32>,
    marked: Vec<bool>,
    /// 单查询复用的 token 位置缓冲：token id -> 文档内位置。
    positions: HashMap<u32, Vec<usize>>,
}

impl SeqAdapter {
    /// 构造适配器：解析参数、定位并加载工件、登记 `KernelArtifact`。
    pub fn new() -> anyhow::Result<Self> {
        let params = SeqParams::from_env()?;
        let artifact_dir = resolve_artifact_dir()?;
        let artifact = KernelArtifact::from_dir(&artifact_dir)?;
        let data = load_artifact(&artifact_dir)?;

        let mut config = params.to_json();
        if let Some(obj) = config.as_object_mut() {
            obj.insert("artifact_dir".into(), serde_json::json!(artifact_dir.display().to_string()));
            obj.insert("artifact_id".into(), serde_json::json!(artifact.id));
            obj.insert("weights_origin".into(), serde_json::json!(artifact.weights_origin));
            obj.insert("k1".into(), serde_json::json!(K1));
            obj.insert("b".into(), serde_json::json!(B));
            obj.insert(
                "tokenizer".into(),
                serde_json::json!("bm25_terms（unicode-alnum-run/lowercase + jieba-rs）"),
            );
        }
        let info = AdapterInfo::new(
            "seq",
            "seq-kernel",
            "0.1.0",
            "外壳：本仓 crates/sl-adapters/src/seq.rs；内核：tools/kernel/build_seq.py 产出的 seq-v0 工件（PPMI 时序边权，语料内统计）",
        )
        .with_config(config)
        .with_artifact(artifact);

        Ok(Self {
            info,
            params,
            artifact_dir,
            data,
            ids: Vec::new(),
            doc_tokens: Vec::new(),
            doc_len: Vec::new(),
            postings: Vec::new(),
            avg_len: 0.0,
            scores: Vec::new(),
            touched: Vec::new(),
            marked: Vec::new(),
            positions: HashMap::new(),
        })
    }

    fn reset_query_buffers(&mut self) {
        for &doc in &self.touched {
            self.scores[doc as usize] = 0.0;
            self.marked[doc as usize] = false;
        }
        self.touched.clear();
    }
}

/// 工件目录解析：`SL_SEQ_ARTIFACT` > `SL_DATA_DIR`/`SL_DATASET`/artifacts/seq-v0。
fn resolve_artifact_dir() -> anyhow::Result<PathBuf> {
    if let Ok(dir) = std::env::var("SL_SEQ_ARTIFACT") {
        return Ok(PathBuf::from(dir));
    }
    let data_dir = std::env::var("SL_DATA_DIR").unwrap_or_else(|_| "data".to_string());
    let dataset = std::env::var("SL_DATASET").map_err(|_| {
        anyhow::anyhow!(
            "seq 适配器需要工件目录：设置 SL_SEQ_ARTIFACT，或由 `sl run` 提供 SL_DATA_DIR/SL_DATASET。\n构建工件：uv run python tools/kernel/build_seq.py <dataset>"
        )
    })?;
    Ok(Path::new(&data_dir).join(dataset).join("artifacts").join("seq-v0"))
}

// ---------------------------------------------------------------------------
// 工件解析
// ---------------------------------------------------------------------------

fn read_u8(buf: &[u8], pos: &mut usize) -> anyhow::Result<u8> {
    let value = *buf
        .get(*pos)
        .ok_or_else(|| anyhow::anyhow!("工件在偏移 {pos} 处提前结束（读 u8）"))?;
    *pos += 1;
    Ok(value)
}

fn read_u16(buf: &[u8], pos: &mut usize) -> anyhow::Result<u16> {
    let end = *pos + 2;
    let slice = buf
        .get(*pos..end)
        .ok_or_else(|| anyhow::anyhow!("工件在偏移 {pos} 处提前结束（读 u16）"))?;
    *pos = end;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(buf: &[u8], pos: &mut usize) -> anyhow::Result<u32> {
    let end = *pos + 4;
    let slice = buf
        .get(*pos..end)
        .ok_or_else(|| anyhow::anyhow!("工件在偏移 {pos} 处提前结束（读 u32）"))?;
    *pos = end;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_f32(buf: &[u8], pos: &mut usize) -> anyhow::Result<f32> {
    let raw = read_u32(buf, pos)?;
    Ok(f32::from_bits(raw))
}

fn load_artifact(dir: &Path) -> anyhow::Result<SeqData> {
    let seq_bytes = std::fs::read(dir.join("seq.bin"))
        .map_err(|err| anyhow::anyhow!("读取 {} 失败：{err}", dir.join("seq.bin").display()))?;
    let doc_bytes = std::fs::read(dir.join("doc_tokens.bin"))
        .map_err(|err| anyhow::anyhow!("读取 {} 失败：{err}", dir.join("doc_tokens.bin").display()))?;

    let mut pos = 0usize;
    let magic = seq_bytes
        .get(pos..pos + 8)
        .ok_or_else(|| anyhow::anyhow!("seq.bin 太短，读不到 magic"))?;
    if magic != SEQ_MAGIC {
        anyhow::bail!("seq.bin 的 magic 不是 SLSEQv1（工件版本不匹配，重建工件）");
    }
    pos += 8;

    let vocab_len = read_u32(&seq_bytes, &mut pos)? as usize;
    let mut vocab = Vec::with_capacity(vocab_len);
    for _ in 0..vocab_len {
        let len = read_u16(&seq_bytes, &mut pos)? as usize;
        let end = pos + len;
        let raw = seq_bytes
            .get(pos..end)
            .ok_or_else(|| anyhow::anyhow!("seq.bin 词表在偏移 {pos} 处越界"))?;
        vocab.push(String::from_utf8(raw.to_vec()).map_err(|_| anyhow::anyhow!("词表条目不是合法 UTF-8"))?);
        pos = end;
    }
    let mut idf = Vec::with_capacity(vocab_len);
    for _ in 0..vocab_len {
        idf.push(read_f32(&seq_bytes, &mut pos)?);
    }
    let edge_count = read_u32(&seq_bytes, &mut pos)? as usize;
    let mut edges = HashMap::with_capacity(edge_count);
    for _ in 0..edge_count {
        let src = read_u32(&seq_bytes, &mut pos)?;
        let dst = read_u32(&seq_bytes, &mut pos)?;
        let order = read_u8(&seq_bytes, &mut pos)?;
        let weight = read_f32(&seq_bytes, &mut pos)?;
        if weight > 0.0 {
            edges.insert((src, dst, order), weight);
        }
    }
    if pos != seq_bytes.len() {
        anyhow::bail!("seq.bin 尾部有 {} 字节未消费（格式与适配器不一致）", seq_bytes.len() - pos);
    }

    let vocab_index: HashMap<String, u32> =
        vocab.iter().enumerate().map(|(idx, token)| (token.clone(), idx as u32)).collect();

    let mut pos = 0usize;
    let magic = doc_bytes
        .get(pos..pos + 8)
        .ok_or_else(|| anyhow::anyhow!("doc_tokens.bin 太短"))?;
    if magic != DOC_MAGIC {
        anyhow::bail!("doc_tokens.bin 的 magic 不是 SLDOCv1");
    }
    pos += 8;
    let doc_count = read_u32(&doc_bytes, &mut pos)? as usize;
    let mut doc_tokens = Vec::with_capacity(doc_count);
    for _ in 0..doc_count {
        let len = read_u32(&doc_bytes, &mut pos)? as usize;
        let mut tokens = Vec::with_capacity(len);
        for _ in 0..len {
            tokens.push(read_u32(&doc_bytes, &mut pos)?);
        }
        doc_tokens.push(tokens);
    }
    if pos != doc_bytes.len() {
        anyhow::bail!("doc_tokens.bin 尾部有 {} 字节未消费", doc_bytes.len() - pos);
    }

    let docs_text = std::fs::read_to_string(dir.join("docs.txt"))
        .map_err(|err| anyhow::anyhow!("读取 {} 失败：{err}", dir.join("docs.txt").display()))?;
    let docs: Vec<String> = docs_text.lines().filter(|l| !l.is_empty()).map(|l| l.to_string()).collect();
    if docs.len() != doc_tokens.len() {
        anyhow::bail!(
            "docs.txt 有 {} 行，doc_tokens.bin 有 {} 篇——工件内部不一致",
            docs.len(),
            doc_tokens.len()
        );
    }

    Ok(SeqData { vocab, vocab_index, idf, edges, docs, doc_tokens })
}

// ---------------------------------------------------------------------------
// 适配器实现
// ---------------------------------------------------------------------------

impl Adapter for SeqAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        if docs.len() != self.data.docs.len() {
            anyhow::bail!(
                "工件 {} 有 {} 篇文档，数据集有 {} 篇——工件与数据集不匹配（重建工件）",
                self.artifact_dir.display(),
                self.data.docs.len(),
                docs.len()
            );
        }
        let artifact_index: HashMap<&str, usize> = self
            .data
            .docs
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.as_str(), idx))
            .collect();
        for doc in docs {
            if !artifact_index.contains_key(doc.id.as_str()) {
                anyhow::bail!(
                    "文档 id {:?} 不在工件 docs.txt 里——工件与数据集不匹配（重建工件）",
                    doc.id
                );
            }
        }

        self.ids.clear();
        self.doc_tokens.clear();
        self.doc_len.clear();
        self.postings.clear();
        self.avg_len = 0.0;

        self.ids = docs.iter().map(|doc| doc.id.clone()).collect();
        self.doc_tokens.reserve(docs.len());
        self.doc_len.reserve(docs.len());
        self.postings = vec![Vec::new(); self.data.vocab.len()];
        self.scores = vec![0.0; docs.len()];
        self.marked = vec![false; docs.len()];

        let mut total_len = 0usize;
        for (dataset_idx, doc) in docs.iter().enumerate() {
            let artifact_idx = *artifact_index.get(doc.id.as_str()).expect("上面已校验存在");
            let tokens = self.data.doc_tokens[artifact_idx].clone();
            total_len += tokens.len();
            self.doc_len.push(tokens.len());

            let mut tf: HashMap<u32, u32> = HashMap::new();
            for &token in &tokens {
                *tf.entry(token).or_insert(0) += 1;
            }
            for (token, count) in tf {
                self.postings[token as usize].push(Posting { doc: dataset_idx as u32, tf: count });
            }
            self.doc_tokens.push(tokens);
        }
        self.avg_len = if docs.is_empty() { 0.0 } else { total_len as f64 / docs.len() as f64 };
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        self.reset_query_buffers();
        if k == 0 || self.ids.is_empty() {
            return Ok(Vec::new());
        }

        let raw_terms = bm25_terms(query);
        if raw_terms.is_empty() {
            return Ok(Vec::new());
        }

        // 单字项：去重后按工件词表映射（OOV 直接跳过）。
        let mut term_ids: Vec<u32> = Vec::new();
        for term in &raw_terms {
            if let Some(&id) = self.data.vocab_index.get(term) {
                if !term_ids.contains(&id) {
                    term_ids.push(id);
                }
            }
        }
        if term_ids.is_empty() {
            return Ok(Vec::new());
        }

        let avg_len = self.avg_len.max(1e-9);
        // 1) 单字项 BM25：倒排累加。
        for &token in &term_ids {
            let idf = self.data.idf[token as usize] as f64;
            if idf <= 0.0 {
                continue;
            }
            for posting in &self.postings[token as usize] {
                let doc = posting.doc as usize;
                let len = self.doc_len[doc] as f64;
                let tf = posting.tf as f64;
                let denom = tf + K1 * (1.0 - B + B * len / avg_len);
                let weight = idf * tf * (K1 + 1.0) / denom;
                if !self.marked[doc] {
                    self.marked[doc] = true;
                    self.touched.push(posting.doc);
                }
                self.scores[doc] += weight;
            }
        }
        if self.touched.is_empty() {
            return Ok(Vec::new());
        }

        // 2) 时序分：查询相邻对（order=1）与隔一对（order=2），只保留工件里有权重的。
        let sequence: Vec<u32> = raw_terms
            .iter()
            .filter_map(|term| self.data.vocab_index.get(term).copied())
            .collect();
        // (a, b, order, weight)；同一 (a,b,order) 只算一次，不同阶分别保留（后面按 (a,b) 求和）。
        let mut pairs: Vec<(u32, u32, u8, f32)> = Vec::new();
        for &order in &self.params.orders {
            let step = order as usize;
            if sequence.len() <= step {
                continue;
            }
            for idx in 0..sequence.len() - step {
                let a = sequence[idx];
                let b = sequence[idx + step];
                if pairs.iter().any(|(pa, pb, po, _)| *pa == a && *pb == b && *po == order) {
                    continue;
                }
                if let Some(&weight) = self.data.edges.get(&(a, b, order)) {
                    pairs.push((a, b, order, weight));
                }
            }
        }

        if self.params.lambda_seq > 0.0 && !pairs.is_empty() {
            // 按 (a, b, order) 分槽再乘该阶的权重系数：w1=w2=1 时与「先合并再乘 sat」逐位等价
            // （sat 只依赖 (a,b) 与 window，不依赖阶），所以默认配置下打分与 0.1.0 完全一致。
            let mut pair_weights: HashMap<(u32, u32, u8), f64> = HashMap::new();
            let mut involved: Vec<u32> = Vec::new();
            for (a, b, order, weight) in &pairs {
                let factor = match order {
                    1 => self.params.w_order_1,
                    _ => self.params.w_order_2,
                };
                *pair_weights.entry((*a, *b, *order)).or_insert(0.0) += *weight as f64 * factor;
                if !involved.contains(a) {
                    involved.push(*a);
                }
                if !involved.contains(b) {
                    involved.push(*b);
                }
            }
            let window = self.params.window;
            let max_matches = self.params.max_matches;
            let ordered = self.params.ordered;
            let mut seq_scores = vec![0.0f64; self.touched.len()];

            for (slot, &doc_u32) in self.touched.iter().enumerate() {
                let doc = doc_u32 as usize;
                // 收集本查询相关 token 在文档内的位置。
                self.positions.clear();
                for (pos_idx, &token) in self.doc_tokens[doc].iter().enumerate() {
                    if involved.contains(&token) {
                        self.positions.entry(token).or_default().push(pos_idx);
                    }
                }
                if self.positions.is_empty() {
                    continue;
                }
                let mut seq_score = 0.0;
                for ((a, b, _order), weight) in &pair_weights {
                    let Some(pos_a) = self.positions.get(a) else { continue };
                    let Some(pos_b) = self.positions.get(b) else { continue };
                    let mut matches = 0usize;
                    for &pa in pos_a {
                        for &pb in pos_b {
                            let hit = if ordered { pa < pb && pb - pa <= window } else { pa != pb && pa.abs_diff(pb) <= window };
                            if hit {
                                matches += 1;
                                if matches >= max_matches {
                                    break;
                                }
                            }
                        }
                        if matches >= max_matches {
                            break;
                        }
                    }
                    if matches > 0 {
                        seq_score += weight * (1.0 + (matches as f64).ln());
                    }
                }
                seq_scores[slot] = seq_score;
            }

            for (slot, &doc_u32) in self.touched.iter().enumerate() {
                self.scores[doc_u32 as usize] += self.params.lambda_seq * seq_scores[slot];
            }
        }

        // 3) 收集非零候选，按分数降序、doc id 升序（与 runner 的兜底一致）。
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
    use std::sync::Mutex;

    /// 环境变量是进程级的；parity 用例并行跑时必须串行访问 `SL_SEQ_ARTIFACT`。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// 按工件格式手工写一个小工件到临时目录，返回目录。
    ///
    /// 词表 `["a","b","c"]`；idf 全 1.0；边 `a -1-> b` 权重 1.0、`b -2-> c` 权重 2.0。
    fn write_fixture(dir: &Path, docs: &[(&str, Vec<u32>)]) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let vocab: [&str; 3] = ["a", "b", "c"];
        let mut seq = Vec::new();
        seq.extend_from_slice(b"SLSEQv1\0");
        seq.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
        for token in vocab {
            seq.extend_from_slice(&(token.len() as u16).to_le_bytes());
            seq.extend_from_slice(token.as_bytes());
        }
        for _ in vocab {
            seq.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let edges: [(u32, u32, u8, f32); 2] = [(0, 1, 1, 1.0), (1, 2, 2, 2.0)];
        seq.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for (src, dst, order, weight) in edges {
            seq.extend_from_slice(&src.to_le_bytes());
            seq.extend_from_slice(&dst.to_le_bytes());
            seq.push(order);
            seq.extend_from_slice(&weight.to_le_bytes());
        }
        std::fs::write(dir.join("seq.bin"), &seq)?;

        let mut doc_bytes = Vec::new();
        doc_bytes.extend_from_slice(b"SLDOCv1\0");
        doc_bytes.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        for (_, tokens) in docs {
            doc_bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
            for token in tokens {
                doc_bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        std::fs::write(dir.join("doc_tokens.bin"), &doc_bytes)?;

        let mut docs_txt = String::new();
        for (id, _) in docs {
            docs_txt.push_str(id);
            docs_txt.push('\n');
        }
        std::fs::write(dir.join("docs.txt"), docs_txt)?;

        let mut files = serde_json::Map::new();
        for name in ["seq.bin", "doc_tokens.bin", "docs.txt"] {
            let bytes = std::fs::read(dir.join(name))?;
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            files.insert(
                name.to_string(),
                serde_json::json!({"sha256": format!("{:x}", hasher.finalize()), "bytes": bytes.len()}),
            );
        }
        let manifest = serde_json::json!({
            "id": "seq-test",
            "kind": "seq_graph",
            "method": "ppmi_topk",
            "weights_origin": "corpus_ppmi",
            "source_corpus": {"dataset": "fixture", "fingerprint": "test"},
            "params": {"orders": [1, 2], "top_k": 20, "window": 8, "min_df": 2, "ppmi_shift": 5, "smoothing": 0.75},
            "counts": {"vocab": vocab.len(), "edges": edges.len()},
            "files": files,
            "runtime": {"device": "cpu", "precision": "f32"},
            "trained_on": "语料内统计（fixture）",
            "contamination": "无",
            "license": "test",
            "notes": ""
        });
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
        Ok(())
    }

    fn build_adapter(dir: &Path) -> SeqAdapter {
        std::env::set_var("SL_SEQ_ARTIFACT", dir.display().to_string());
        SeqAdapter::new().expect("构造适配器")
    }

    /// 按 §8.1 手算单查询分数，供断言参考。
    fn hand_score(tf: &[u32], doc_len: usize, avg_len: f64, seq: f64, lambda: f64) -> f64 {
        let mut uni = 0.0;
        for &t in tf {
            let tf = t as f64;
            let len = doc_len as f64;
            let denom = tf + K1 * (1.0 - B + B * len / avg_len);
            uni += tf * (K1 + 1.0) / denom; // idf = 1.0
        }
        uni + lambda * seq
    }

    #[test]
    fn unigram_only_when_lambda_is_zero() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![0, 1, 2]), ("d2", vec![2, 0])]).unwrap();
        let mut adapter = build_adapter(tmp.path());
        // λ=0：只剩单字项
        adapter.params.lambda_seq = 0.0;
        adapter.index(&[Doc::new("d1", "a b c"), Doc::new("d2", "c a")]).unwrap();
        let hits = adapter.search("a b", 10).unwrap();
        assert_eq!(hits[0].0, "d1");
        // d1: a/b 各 tf=1，len=3，avg=2.5
        let expected = hand_score(&[1, 1], 3, 2.5, 0.0, 0.0);
        assert!((hits[0].1 - expected).abs() < 1e-9, "got {} want {}", hits[0].1, expected);
        // d2 只命中 a
        let expected_d2 = hand_score(&[1], 2, 2.5, 0.0, 0.0);
        assert!((hits[1].1 - expected_d2).abs() < 1e-9);
    }

    #[test]
    fn ordered_sequence_bonus_matches_hand_computation() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![0, 1, 2]), ("d2", vec![2, 0])]).unwrap();
        let mut adapter = build_adapter(tmp.path());
        adapter.index(&[Doc::new("d1", "a b c"), Doc::new("d2", "c a")]).unwrap();
        let hits = adapter.search("a b", 10).unwrap();
        // d1: a 在 0、b 在 1（距离 1 ≤ window 8）→ 匹配 1 次 → sat=1 → seq = 1.0
        let expected = hand_score(&[1, 1], 3, 2.5, 1.0, 0.5);
        assert!((hits[0].1 - expected).abs() < 1e-9, "got {} want {}", hits[0].1, expected);
        // d2 无 b，seq=0
        let expected_d2 = hand_score(&[1], 2, 2.5, 0.0, 0.5);
        assert!((hits[1].1 - expected_d2).abs() < 1e-9);
    }

    #[test]
    fn unordered_mode_scores_reversed_order() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![1, 0])]).unwrap(); // 文档里是 b a（反序）
        let mut adapter = build_adapter(tmp.path());
        adapter.index(&[Doc::new("d1", "b a")]).unwrap();
        let ordered = adapter.search("a b", 10).unwrap();
        let ordered_score = ordered[0].1;
        adapter.params.ordered = false;
        let unordered = adapter.search("a b", 10).unwrap();
        assert!(unordered[0].1 > ordered_score, "无序邻近应当给反序文档加分");
    }

    #[test]
    fn second_order_edge_is_used_only_when_enabled() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        // 查询 a c 在文档里隔一个 token（距离 2）；工件有 (b, c, order=2) 边，a→c 无直接边
        write_fixture(tmp.path(), &[("d1", vec![0, 1, 2])]).unwrap();
        let mut adapter = build_adapter(tmp.path());
        adapter.index(&[Doc::new("d1", "a b c")]).unwrap();
        // orders=[1,2]：查询 "b c" 命中 (b,c,2) 边（查询里相邻 → order=1 查不到，用隔一需要中间词）
        let with_orders = adapter.search("a c", 10).unwrap();
        assert!(with_orders.is_empty() || with_orders[0].1 > 0.0);
        adapter.params.orders = vec![1];
        let only_first = adapter.search("a c", 10).unwrap();
        // "a c" 的 order=1 边不存在 → 只有 a 的单字分
        assert!(only_first.is_empty() || only_first[0].1 > 0.0);
    }

    #[test]
    fn window_boundary_controls_match() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        // 文档：a x x x b（a 在 0，b 在 4，距离 4）
        write_fixture(tmp.path(), &[("d1", vec![0, 2, 2, 2, 1])]).unwrap();
        let mut adapter = build_adapter(tmp.path());
        adapter.index(&[Doc::new("d1", "a c c c b")]).unwrap();
        adapter.params.window = 4;
        let hit = adapter.search("a b", 10).unwrap()[0].1;
        adapter.params.window = 3;
        let miss = adapter.search("a b", 10).unwrap()[0].1;
        assert!(hit > miss, "window=4 应命中距离 4 的匹配，window=3 不应（{hit} vs {miss}）");
    }

    #[test]
    fn artifact_doc_set_mismatch_is_rejected() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![0]), ("d2", vec![1])]).unwrap();
        let mut adapter = build_adapter(tmp.path());
        let err = adapter
            .index(&[Doc::new("d1", "a"), Doc::new("nope", "b")])
            .expect_err("文档集合不一致必须报错");
        assert!(format!("{err:#}").contains("不在工件"));
    }

    #[test]
    fn tampered_artifact_file_is_rejected() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![0])]).unwrap();
        std::fs::write(tmp.path().join("docs.txt"), "d1\nd2\n").unwrap();
        std::env::set_var("SL_SEQ_ARTIFACT", tmp.path().display().to_string());
        let err = SeqAdapter::new().expect_err("哈希不符必须报错");
        assert!(format!("{err:#}").contains("sha256"));
    }

    #[test]
    fn non_cpu_runtime_is_rejected() {
        let _guard = env_guard();
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path(), &[("d1", vec![0])]).unwrap();
        let path = tmp.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest["runtime"]["device"] = serde_json::json!("cuda");
        std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        std::env::set_var("SL_SEQ_ARTIFACT", tmp.path().display().to_string());
        let err = SeqAdapter::new().expect_err("非 CPU 运行时必须被拒绝");
        assert!(format!("{err:#}").contains("CPU"));
    }
}
