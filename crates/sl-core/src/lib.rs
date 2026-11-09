//! search-lab 核心契约。
//!
//! 这个 crate 是「对比标准」在代码上的落点，只有它定义的东西是跨数据集 / 跨适配器共享的：
//!
//! - [`types`]：`Doc` / `Query` / `Qrels` / `Dataset` / `Manifest` —— 数据契约；
//! - [`canonical`]：规范三元组目录的读写（corpus.jsonl / queries.jsonl / qrels.tsv / manifest.json）；
//! - [`metrics`]：指标口径（Recall@k / MRR@10 / nDCG@10 / MAP@100）；
//! - [`adapter`]：适配器 trait 与适配器元信息；
//! - [`run`]：跑一轮对比并把结果写成运行记录；
//! - [`report`]：把运行记录渲染成 Markdown 表。
//!
//! 口径与纪律写在仓库根 `docs/STANDARD.md`，代码只是它的强制点。

pub mod adapter;
pub mod canonical;
pub mod metrics;
pub mod report;
pub mod run;
pub mod types;

pub use adapter::{Adapter, AdapterInfo};
pub use metrics::{evaluate_rankings, Metrics, PerQuery};
pub use run::{run_dataset, AdapterOutcome, EnvInfo, RunRecord};
pub use types::{Counts, Dataset, Doc, FileDigest, KernelArtifact, Manifest, Qrels, Query};
