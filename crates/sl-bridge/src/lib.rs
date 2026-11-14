//! PriorSeek 算法衔接桥 (PriorSeek Algorithm Bridge)
//!
//! 解决外部应用调用 PriorSeek 检索内核的三大核心痛点：
//! 1. **数据模型解耦**：彻底摆脱对 `(doc_id, score)` 简陋元组的绑架，支持任意结构体 `T` 的泛型多路融合；
//! 2. **无超参 RRF 融合管道**：提供纯秩次倒数融合，彻底消除跨模型、跨路由（如 BM25 vs 向量 vs AST）分数量纲不可比的顽疾；
//! 3. **空间/区间重叠共振 (Overlap Synergy)**：对代码块、文档 span 提供就地重叠合并与共振加权置顶，让多路命中的高置信目标必然置顶。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::HashMap;
use std::hash::Hash;

/// 默认 RRF 稳定常数（在 PriorSeek 大量消融实验中验证的最优经验常数）
pub const DEFAULT_RRF_K: f64 = 60.0;

/// 底层快速标量名次融合：输入若干路 `[(doc_id, score)]`，返回 RRF 排序后的 `[(doc_id, rrf_score)]`。
pub fn fuse_ranked(routes: &[&[(String, f64)]], k: usize, rrf_k: f64) -> Vec<(String, f64)> {
    sl_engine::dense::rrf_fuse_many(routes, k, rrf_k)
}

/// 单个路由的候选输入
#[derive(Debug, Clone)]
pub struct RankedRoute<T> {
    pub name: String,
    pub weight: f64,
    pub items: Vec<T>,
}

/// 融合输出项：完整保留原始结构体 `T`，并注入 RRF 融合元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedItem<T> {
    pub item: T,
    pub rrf_score: f64,
    /// 来源归因：记录该候选命中了哪些路由，以及在对应路由中的 0-based 名次 `(route_name, rank)`
    pub provenance: Vec<(String, usize)>,
}

impl<T> FusedItem<T> {
    /// 判断该候选是否由两条及以上独立路由共同命中（触发了双路/多路物理共振）
    pub fn is_synergistic(&self) -> bool {
        self.provenance.len() > 1
    }
}

/// 泛型多路 RRF 融合流水线 (Generic RRF Pipeline)
pub struct RrfPipeline<T> {
    rrf_k: f64,
    routes: Vec<RankedRoute<T>>,
}

impl<T> Default for RrfPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RrfPipeline<T> {
    pub fn new() -> Self {
        Self {
            rrf_k: DEFAULT_RRF_K,
            routes: Vec::new(),
        }
    }

    /// 设置 RRF 平滑常数（默认 60.0）
    pub fn with_k(mut self, k: f64) -> Self {
        self.rrf_k = k.max(1.0);
        self
    }

    /// 添加一条平等权重的检索路由结果（有序列表，下标即排名）
    pub fn add_route(mut self, name: impl Into<String>, items: Vec<T>) -> Self {
        self.routes.push(RankedRoute {
            name: name.into(),
            weight: 1.0,
            items,
        });
        self
    }

    /// 添加一条带先验权重的检索路由结果（若内核学到了路由级置信度先验）
    pub fn add_weighted_route(
        mut self,
        name: impl Into<String>,
        weight: f64,
        items: Vec<T>,
    ) -> Self {
        self.routes.push(RankedRoute {
            name: name.into(),
            weight: weight.max(0.0),
            items,
        });
        self
    }

    /// 精确主键融合：根据 `key_fn` 返回的主键做精确对齐与名次累加
    pub fn fuse_exact<K: Hash + Eq>(self, key_fn: impl Fn(&T) -> K) -> Vec<FusedItem<T>> {
        struct Entry<T> {
            item: T,
            score: f64,
            provenance: Vec<(String, usize)>,
        }

        let mut map: HashMap<K, Entry<T>> = HashMap::new();

        for route in self.routes {
            let weight = route.weight;
            if weight <= 0.0 {
                continue;
            }
            for (rank, item) in route.items.into_iter().enumerate() {
                let key = key_fn(&item);
                let score_contrib = weight / (self.rrf_k + (rank as f64) + 1.0);

                match map.entry(key) {
                    std::collections::hash_map::Entry::Occupied(mut occ) => {
                        let e = occ.get_mut();
                        e.score += score_contrib;
                        e.provenance.push((route.name.clone(), rank));
                    }
                    std::collections::hash_map::Entry::Vacant(vac) => {
                        vac.insert(Entry {
                            item,
                            score: score_contrib,
                            provenance: vec![(route.name.clone(), rank)],
                        });
                    }
                }
            }
        }

        let mut results: Vec<FusedItem<T>> = map
            .into_values()
            .map(|e| FusedItem {
                item: e.item,
                rrf_score: e.score,
                provenance: e.provenance,
            })
            .collect();

        // 统一按 RRF 分数降序，同分保持稳定
        results.sort_by(|a, b| b.rrf_score.total_cmp(&a.rrf_score));
        results
    }

    /// 空间/区间重叠共振融合（专门针对代码块、文档 span、实体覆盖的杀手级接口）：
    /// 当不同路由命中的候选在分区键（如文件路径）相同且区间重叠时，自动合并并叠加双路共振分数。
    pub fn fuse_overlapping<K: Hash + Eq>(
        self,
        group_key_fn: impl Fn(&T) -> K,
        overlap_fn: impl Fn(&T, &T) -> bool,
    ) -> Vec<FusedItem<T>> {
        self.fuse_overlapping_with_merge(group_key_fn, overlap_fn, |_, _| {})
    }

    /// 空间/区间重叠共振融合（带领域状态合并）：
    /// 当两路候选发生空间重叠共振时，叠加 RRF 得分、记录归因并调用 `merge_fn(&mut existing, &incoming)`
    /// 将后入路由的丰富语法或元数据（如 AST 关系、作用域、元变量等）注入到合并项中。
    pub fn fuse_overlapping_with_merge<K: Hash + Eq>(
        self,
        group_key_fn: impl Fn(&T) -> K,
        overlap_fn: impl Fn(&T, &T) -> bool,
        merge_fn: impl Fn(&mut T, &T),
    ) -> Vec<FusedItem<T>> {
        let mut groups: HashMap<K, Vec<FusedItem<T>>> = HashMap::new();

        for route in self.routes {
            let weight = route.weight;
            if weight <= 0.0 {
                continue;
            }
            for (rank, item) in route.items.into_iter().enumerate() {
                let gkey = group_key_fn(&item);
                let score_contrib = weight / (self.rrf_k + (rank as f64) + 1.0);
                let group_items = groups.entry(gkey).or_default();

                let mut matched = false;
                for existing in group_items.iter_mut() {
                    if overlap_fn(&existing.item, &item) {
                        // 发生区间重叠共振！叠加得分、合并属性并记录归因
                        existing.rrf_score += score_contrib;
                        existing.provenance.push((route.name.clone(), rank));
                        merge_fn(&mut existing.item, &item);
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    group_items.push(FusedItem {
                        item,
                        rrf_score: score_contrib,
                        provenance: vec![(route.name.clone(), rank)],
                    });
                }
            }
        }

        let mut all_results: Vec<FusedItem<T>> = groups.into_values().flatten().collect();
        all_results.sort_by(|a, b| b.rrf_score.total_cmp(&a.rrf_score));
        all_results
    }
}

/// 通用检索适配器桥 (Adapter Bridge)
///
/// 将任意多个实现了 `sl_core::Adapter` 的独立路由封装为一个统一的 `Adapter`，
/// 供 PriorSeek runner 或评测基准透明调用。
pub struct AdapterBridge {
    info: AdapterInfo,
    routes: Vec<(Box<dyn Adapter>, f64)>,
    rrf_k: f64,
    candidate_depth: usize,
}

impl AdapterBridge {
    pub fn new(
        name: impl Into<String>,
        family: impl Into<String>,
        source: impl Into<String>,
        routes: Vec<Box<dyn Adapter>>,
    ) -> Self {
        let route_tuples: Vec<(Box<dyn Adapter>, f64)> =
            routes.into_iter().map(|r| (r, 1.0)).collect();
        Self::with_weighted_routes(name, family, source, route_tuples)
    }

    pub fn with_weighted_routes(
        name: impl Into<String>,
        family: impl Into<String>,
        source: impl Into<String>,
        routes: Vec<(Box<dyn Adapter>, f64)>,
    ) -> Self {
        let route_names: Vec<String> = routes
            .iter()
            .map(|(route, w)| format!("{}(w={:.2})", route.info().name, w))
            .collect();
        let info = AdapterInfo::new(name, family, "0.2.0", source).with_config(serde_json::json!({
            "fusion": "priorseek_rrf_pipeline",
            "rrf_k": DEFAULT_RRF_K,
            "routes": route_names,
        }));
        Self {
            info,
            routes,
            rrf_k: DEFAULT_RRF_K,
            candidate_depth: 60,
        }
    }

    pub fn with_candidate_depth(mut self, depth: usize) -> Self {
        self.candidate_depth = depth.max(1);
        self
    }

    pub fn with_k(mut self, k: f64) -> Self {
        self.rrf_k = k.max(1.0);
        self
    }
}

impl Adapter for AdapterBridge {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        for (route, _) in &mut self.routes {
            route.index(docs)?;
        }
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if k == 0 || self.routes.is_empty() {
            return Ok(Vec::new());
        }
        let depth = k.max(self.candidate_depth);

        let mut pipeline = RrfPipeline::new().with_k(self.rrf_k);
        for (route, weight) in &mut self.routes {
            let res = route.search(query, depth)?;
            pipeline = pipeline.add_weighted_route(route.info().name.clone(), *weight, res);
        }

        // 使用精确 ID 对齐输出
        let fused = pipeline.fuse_exact(|(doc_id, _)| doc_id.clone());
        let top_k: Vec<(String, f64)> = fused
            .into_iter()
            .take(k)
            .map(|fi| (fi.item.0, fi.rrf_score))
            .collect();
        Ok(top_k)
    }
}

/// 兼容别名（确保旧调用点平滑过渡）
pub type RrfBridge = AdapterBridge;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CodeCandidate {
        path: String,
        start_line: u64,
        end_line: u64,
        code: String,
    }

    #[test]
    fn test_pipeline_exact_fusion() {
        let bm25_hits = vec![
            ("doc_a".to_string(), 10.5),
            ("doc_b".to_string(), 8.2),
            ("doc_c".to_string(), 3.1),
        ];
        let dense_hits = vec![
            ("doc_b".to_string(), 0.95),
            ("doc_d".to_string(), 0.88),
            ("doc_a".to_string(), 0.40),
        ];

        let fused = RrfPipeline::new()
            .with_k(60.0)
            .add_route("bm25", bm25_hits)
            .add_route("dense", dense_hits)
            .fuse_exact(|(id, _)| id.clone());

        assert_eq!(fused.len(), 4);
        // doc_b 在两路均高分（rank 1 + rank 0），应产生共振夺冠
        assert_eq!(fused[0].item.0, "doc_b");
        assert!(fused[0].is_synergistic());
        assert_eq!(fused[0].provenance.len(), 2);
    }

    #[test]
    fn test_pipeline_overlapping_synergy() {
        // 模拟外部检索（如 relation-search）：词法路与 AST 结构路命中重叠代码块
        let bm25_results = vec![
            CodeCandidate {
                path: "src/auth.rs".into(),
                start_line: 10,
                end_line: 30,
                code: "fn login() { ... }".into(),
            },
            CodeCandidate {
                path: "src/db.rs".into(),
                start_line: 1,
                end_line: 50,
                code: "fn query() { ... }".into(),
            },
        ];

        let ast_results = vec![
            CodeCandidate {
                path: "src/auth.rs".into(),
                start_line: 12, // 区间重叠
                end_line: 28,
                code: "fn login()".into(),
            },
            CodeCandidate {
                path: "src/util.rs".into(),
                start_line: 5,
                end_line: 15,
                code: "fn helper()".into(),
            },
        ];

        let fused = RrfPipeline::new()
            .with_k(60.0)
            .add_route("bm25", bm25_results)
            .add_route("ast", ast_results)
            .fuse_overlapping(
                |c| c.path.clone(),
                |a, b| a.start_line <= b.end_line && b.start_line <= a.end_line,
            );

        assert_eq!(fused.len(), 3);
        // src/auth.rs 产生了双路区间重叠共振，必须排在第一位
        assert_eq!(fused[0].item.path, "src/auth.rs");
        assert!(fused[0].is_synergistic());
        assert_eq!(fused[0].provenance.len(), 2);
    }
}
