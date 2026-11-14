//! PriorSeek 检索外壳与核心算子。
//!
//! 包含：
//! - PositionalIndex: 紧凑位置倒排索引
//! - DenseIndex: 纯 CPU 8-way SIMD 展开向量点积 + 有界最小堆 Top-K 截断
//! - rrf_fuse: 无超参数倒数秩融合算子
//! - BowScorer / OacScorer: 词袋与词序覆盖算法

pub mod bow;
pub mod dense;
pub mod flow_matcher;
pub mod index;
pub mod logic_executor;
pub mod micro_net;
pub mod oac;
pub mod sparse;

pub use bow::{BowScorer, WeightedTerm};
pub use dense::{rrf_fuse, DenseIndex};
pub use flow_matcher::{CostGraph, FlowMatcher};
pub use index::{Occurrence, PositionalIndex, TieredIndex};
pub use logic_executor::{ExecutionExplanation, LogicExecutor, QueryConstraint};
pub use micro_net::MicroNet;
pub use oac::{KernelGapTable, OacParams, OacScorer};
pub use sparse::SparseInvertedIndex;
