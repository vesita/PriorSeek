//! 算法桥接：**稳定的“多路检索路由 → 统一 Adapter”接口**。
//!
//! PriorSeek 的各个适配器原本各自处理具体数据；这个 crate 只负责把若干个
//! 独立路由（BM25、稠密、概念、图、记忆……）包成同一个 `sl_core::Adapter`
//! 并用无参 RRF 融合。路由本身不感知 benchmark 数据集，也不把原始分数量纲
//! 带进融合层——这样新增方向只需要实现一个路由 Adapter，不用复制融合算法。
//!
//! 结构上分两层：
//! - [`fuse_ranked`]：纯函数，任意路数的 `(doc_id, score)` 名次融合；
//! - [`RrfBridge`]：实现 `Adapter`，负责 index/search 编排与候选深度。

use anyhow::Result;
use sl_core::{Adapter, AdapterInfo, Doc};

/// 任意路数的 RRF 融合。每路只在内部去重，按名次计票，不比较分数。
pub fn fuse_ranked(routes: &[&[(String, f64)]], k: usize, rrf_k: f64) -> Vec<(String, f64)> {
    sl_engine::dense::rrf_fuse_many(routes, k, rrf_k)
}

/// 多路 RRF 桥：把一组 `Adapter` 当作独立路由。
///
/// 路由数量不设上限；`candidate_depth` 控制每路先取多少候选再融合，
/// 对小语料/大语料都保持一致的名次语义。
pub struct RrfBridge {
    info: AdapterInfo,
    routes: Vec<Box<dyn Adapter>>,
    rrf_k: f64,
    candidate_depth: usize,
}

impl RrfBridge {
    pub fn new(
        name: impl Into<String>,
        family: impl Into<String>,
        source: impl Into<String>,
        routes: Vec<Box<dyn Adapter>>,
    ) -> Self {
        let route_names: Vec<String> = routes
            .iter()
            .map(|route| route.info().name.clone())
            .collect();
        let info = AdapterInfo::new(name, family, "0.2.0", source).with_config(serde_json::json!({
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": 60.0,
            "hyperparameters": "none (rank reciprocity only)",
            "routes": route_names,
        }));
        Self {
            info,
            routes,
            rrf_k: 60.0,
            candidate_depth: 60,
        }
    }

    pub fn with_candidate_depth(mut self, depth: usize) -> Self {
        self.candidate_depth = depth.max(1);
        self
    }

    pub fn route_names(&self) -> Vec<&str> {
        self.routes
            .iter()
            .map(|route| route.info().name.as_str())
            .collect()
    }
}

impl Adapter for RrfBridge {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        for route in &mut self.routes {
            route.index(docs)?;
        }
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let depth = k.max(self.candidate_depth);
        let mut lists: Vec<Vec<(String, f64)>> = Vec::with_capacity(self.routes.len());
        for route in &mut self.routes {
            lists.push(route.search(query, depth)?);
        }
        let refs: Vec<&[(String, f64)]> = lists.iter().map(Vec::as_slice).collect();
        Ok(fuse_ranked(&refs, k, self.rrf_k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sl_core::{Adapter, AdapterInfo, Doc};

    struct FixedRoute {
        info: AdapterInfo,
        hits: Vec<(String, f64)>,
    }

    impl FixedRoute {
        fn new(name: &str, hits: &[(&str, f64)]) -> Self {
            Self {
                info: AdapterInfo::new(name, "test-route", "0", "test"),
                hits: hits
                    .iter()
                    .map(|(id, score)| ((*id).to_string(), *score))
                    .collect(),
            }
        }
    }

    impl Adapter for FixedRoute {
        fn info(&self) -> &AdapterInfo {
            &self.info
        }

        fn index(&mut self, _docs: &[Doc]) -> Result<()> {
            Ok(())
        }

        fn search(&mut self, _query: &str, k: usize) -> Result<Vec<(String, f64)>> {
            Ok(self.hits.iter().take(k).cloned().collect())
        }
    }

    #[test]
    fn bridge_fuses_more_than_two_routes() {
        let mut bridge = RrfBridge::new(
            "test-bridge",
            "rrf",
            "test",
            vec![
                Box::new(FixedRoute::new("a", &[("d1", 10.0), ("d2", 5.0)])),
                Box::new(FixedRoute::new("b", &[("d2", 9.0), ("d1", 8.0)])),
                Box::new(FixedRoute::new("c", &[("d3", 7.0), ("d1", 6.0)])),
            ],
        );
        let hits = bridge.search("q", 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, "d1");
    }

    #[test]
    fn fuse_ranked_keeps_rank_reciprocity() {
        let a = [("d1".to_string(), 100.0), ("d2".to_string(), 1.0)];
        let b = [("d2".to_string(), 100.0), ("d1".to_string(), 1.0)];
        let fused = fuse_ranked(&[&a, &b], 2, 60.0);
        assert_eq!(fused.len(), 2);
        assert!(fused.iter().all(|(_, score)| *score > 0.0));
    }
}
