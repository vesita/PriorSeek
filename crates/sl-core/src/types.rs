//! 数据契约：数据集产出 [`Dataset`]，适配器只接受 [`Doc`] / [`Query`]。
//!
//! 这里的类型一旦改动，所有数据集构建脚本与所有适配器都要同步 —— 它就是「对比标准」的
//! 结构层。语义层（怎么评、怎么比）见仓库根 `docs/STANDARD.md`。

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// 语料里的一篇文档。`title` 可以为空（纯文本语料）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    pub text: String,
}

impl Doc {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self { id: id.into(), title: String::new(), text: text.into() }
    }

    /// 参与检索的全文。`title` 非空时拼在前面 —— 与 BEIR 的通行做法一致。
    pub fn full_text(&self) -> String {
        if self.title.is_empty() {
            self.text.clone()
        } else {
            format!("{}\n{}", self.title, self.text)
        }
    }
}

/// 一条查询。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
}

/// 相关性判定：`qid -> docid -> rel`。`rel = 0` 表示判定为不相关（可以出现，也可以省略）。
pub type Qrels = BTreeMap<String, BTreeMap<String, u32>>;

/// 各文件的内容摘要，由数据集构建器写入，用来判「这份规范数据还是不是原来那份」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub sha256: String,
    pub bytes: u64,
}

/// 数据集规模。`scored_queries` = qrels 里至少有一个正相关文档的查询数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub docs: usize,
    pub queries: usize,
    pub qrels: usize,
    pub scored_queries: usize,
}

/// 数据集清单：来源、构建参数、规模、文件摘要。
///
/// **跨机器 / 跨时间的可比性全靠它**：两个 run 的 `manifest_sha256` 不同就不是同一道题。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    /// 原始数据来源（URL、仓库、论文），一行一个。
    pub source: Vec<String>,
    /// 谁按什么参数构建的（人类可读，含子集口径时写死 pool 大小与 seed）。
    pub built_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub notes: String,
    pub counts: Counts,
    pub files: BTreeMap<String, FileDigest>,
}

impl Manifest {
    /// 清单整体摘要：对「参与比对的四个文件摘要」做 SHA-256。
    /// 不含 `built_by` / `notes`，因为它们不影响题目本身。
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update([0]);
        hasher.update(self.version.as_bytes());
        hasher.update([0]);
        for (file, digest) in &self.files {
            hasher.update(file.as_bytes());
            hasher.update([0]);
            hasher.update(digest.sha256.as_bytes());
            hasher.update([0]);
        }
        hasher.update(self.counts.docs.to_le_bytes());
        hasher.update(self.counts.queries.to_le_bytes());
        hasher.update(self.counts.qrels.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// 一份可以直接拿来跑的规范数据集。
#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    pub qrels: Qrels,
    pub manifest: Manifest,
}

impl Dataset {
    pub fn doc_ids(&self) -> HashSet<&str> {
        self.docs.iter().map(|d| d.id.as_str()).collect()
    }

    /// 该查询的正相关文档（`rel > 0`），按 `rel` 降序、doc id 升序给稳定顺序。
    pub fn relevant(&self, qid: &str) -> Vec<(&str, u32)> {
        let Some(map) = self.qrels.get(qid) else { return Vec::new() };
        let mut rows: Vec<(&str, u32)> = map
            .iter()
            .filter(|(_, rel)| **rel > 0)
            .map(|(doc, rel)| (doc.as_str(), *rel))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        rows
    }

    /// 参与计分的查询：qrels 里至少有一条正相关判定。
    pub fn scored_query_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .qrels
            .iter()
            .filter(|(_, map)| map.values().any(|rel| *rel > 0))
            .map(|(qid, _)| qid.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// 契约自检：qrels 指向的文档必须都在语料里，查询必须都在查询表里。
    /// 返回 (被丢弃的 qrels 条数, 被丢弃的查询条数, 详情)。被丢弃的条目会在加载时剔除。
    pub fn prune_inconsistent(&mut self) -> PruneReport {
        let doc_ids: HashSet<String> = self.docs.iter().map(|d| d.id.clone()).collect();
        let query_ids: BTreeSet<String> = self.queries.iter().map(|q| q.id.clone()).collect();

        let mut missing_docs: Vec<(String, String)> = Vec::new();
        let mut missing_queries: Vec<String> = Vec::new();

        let qrels = std::mem::take(&mut self.qrels);
        let mut kept: Qrels = BTreeMap::new();
        for (qid, map) in qrels {
            if !query_ids.contains(&qid) {
                missing_queries.push(qid);
                continue;
            }
            let mut kept_map = BTreeMap::new();
            for (docid, rel) in map {
                if doc_ids.contains(&docid) {
                    kept_map.insert(docid, rel);
                } else {
                    missing_docs.push((qid.clone(), docid));
                }
            }
            if !kept_map.is_empty() {
                kept.insert(qid, kept_map);
            }
        }
        self.qrels = kept;

        let scored = self.scored_query_ids().len();
        let mut report = PruneReport {
            dropped_qrels: missing_docs.len(),
            dropped_queries: missing_queries.len(),
            missing_docs,
            missing_queries,
            ..Default::default()
        };
        report.scored_queries = scored;
        report
    }

    /// 从 manifest 与当前内容重算规模，返回 (与 manifest 一致的 Counts, 是否一致)。
    pub fn recount(&self) -> (Counts, bool) {
        let qrels: usize = self.qrels.values().map(|m| m.len()).sum();
        let counts = Counts {
            docs: self.docs.len(),
            queries: self.queries.len(),
            qrels,
            scored_queries: self.scored_query_ids().len(),
        };
        let same = counts == self.manifest.counts;
        (counts, same)
    }
}

/// [`Dataset::prune_inconsistent`] 的结果。
#[derive(Debug, Clone, Default)]
pub struct PruneReport {
    pub dropped_qrels: usize,
    pub dropped_queries: usize,
    pub scored_queries: usize,
    pub missing_docs: Vec<(String, String)>,
    pub missing_queries: Vec<String>,
}

impl PruneReport {
    pub fn is_clean(&self) -> bool {
        self.dropped_qrels == 0 && self.dropped_queries == 0
    }
}
