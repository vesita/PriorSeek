//! 适配器契约：一个搜索算法只要实现 [`Adapter`]，就能进对比表。
//!
//! 契约的三条硬要求（`run` 会强制）：
//! 1. `search` 返回的是**语料文档 id**，不是内部块 id / 文件路径 / 行号；
//! 2. 分数只用于排序，跨适配器不可比（不做归一、不跨算法比分数）；
//! 3. `index` 只看 `docs`，`search` 只看 query 文本 —— 适配器拿不到 qrels，不能用答案调参。

use crate::types::Doc;
use serde::{Deserialize, Serialize};

/// 适配器元信息，会原样进运行记录。定义「和谁在比」靠它。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterInfo {
    /// 对比表里的行名，仓库内唯一（如 `faq-hybrid`、`rs`、`bm25`）。
    pub name: String,
    /// 算法族（如 `bm25` / `faq-scheme` / `relation-search`）。
    pub family: String,
    /// 实现版本或来源版本；外部实现写它的 commit / 二进制摘要。
    pub version: String,
    /// 来源说明：本仓文件、外部仓库路径、上游项目与版本。
    pub source: String,
    /// 参与排序的全部显式参数（k1/b、权重、分词器、是否重排……），原样记录。
    #[serde(default)]
    pub config: serde_json::Value,
}

impl AdapterInfo {
    pub fn new(
        name: impl Into<String>,
        family: impl Into<String>,
        version: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            family: family.into(),
            version: version.into(),
            source: source.into(),
            config: serde_json::Value::Null,
        }
    }

    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }
}

/// 一个可参与对比的检索算法。
///
/// 生命周期：`index(docs)` 一次 → `search(query, k)` 多次。实现可以缓存索引。
pub trait Adapter {
    fn info(&self) -> &AdapterInfo;

    /// 建索引。允许做任何预处理，但只能看 `docs`。
    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()>;

    /// 检索：返回 `(doc_id, score)`，按 score 降序（同一分由 runner 用 doc id 兜底排序）。
    /// `k` 是**期望返回条数上限**，返回少于 k 条是合法的（表示算法认为没有更多结果）。
    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>>;
}
