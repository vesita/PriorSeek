//! search-lab 的**检索外壳**：一份索引 + 可替换的打分算法。
//!
//! 设计背景（`docs/ALGORITHM.md`）：之前每个想法都写成一个独立适配器，
//! 各自复制一遍倒排、BM25 数学与排序规则，导致"加一个想法 = 打一个补丁"。
//! 这里把外壳抽成两层：
//!
//! ```text
//! PositionalIndex   位置索引（唯一一份）：doc token 序列 + 倒排位置表 + df/len/idf
//! 打分器（算法）     BowScorer（词袋端点）/ OacScorer（有序对齐覆盖，本仓主算法）
//! ```
//!
//! **索引层不认识 tokenizer**：token 以 `u32` id 传入，所以词级切分与 BPE 共用同一份外壳代码，
//! 换 tokenizer 只是换变量。
//!
//! 内核接口只有一处：[`KernelGapTable`] —— 内核只回答「间隔为 g 的相邻匹配值多少分」。

pub mod bow;
pub mod dense;
pub mod flow_matcher;
pub mod index;
pub mod logic_executor;
pub mod micro_net;
pub mod oac;

pub use bow::{BowScorer, WeightedTerm};
pub use dense::{rrf_fuse, DenseIndex};
pub use flow_matcher::{CostGraph, FlowMatcher};
pub use index::{Occurrence, PositionalIndex, TieredIndex};
pub use logic_executor::{ExecutionExplanation, LogicExecutor, QueryConstraint};
pub use micro_net::MicroNet;
pub use oac::{KernelGapTable, OacParams, OacScorer};
