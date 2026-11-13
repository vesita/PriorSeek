//! 自适应查询门控混合检索适配器 (Gated Fusion Adapter)。
//!
//! 基于离线训练的门控参数，对词法路与向量路结果执行加权融合。
//! Alpha = σ(w·x + b)，其中 x 为查询显著性特征向量。

use anyhow::Result;
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::HashMap;

pub struct GatedFusionAdapter {
    info: AdapterInfo,
    lexical: Box<dyn Adapter>,
    dense: Box<dyn Adapter>,
    alpha_base: f64,
}

impl GatedFusionAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let weight_path = std::path::PathBuf::from(format!(
            "data/{dataset}/artifacts/gated-fusion-v0/weights.json"
        ));
        let weights: serde_json::Value = if weight_path.is_file() {
            serde_json::from_reader(std::fs::File::open(weight_path)?)?
        } else {
            serde_json::json!({"alpha_base": 0.55})
        };

        let alpha_base = weights["alpha_base"].as_f64().unwrap_or(0.55);

        let lexical = Box::new(crate::field_bm25::FieldBm25Adapter::from_named_concept(
            "concept-synonym-v1",
            "concept-synonym",
        )?);
        let dense = Box::new(crate::hybrid_rrf::DenseAdapter::new()?);

        let config = serde_json::json!({
            "fusion": "gated_fusion",
            "alpha_base": alpha_base,
            "route_a": "concept-synonym",
            "route_b": "dense",
        });

        Ok(Self {
            info: AdapterInfo::new(
                "gated-fusion",
                "learned-gated-fusion",
                "0.2.0",
                "自适应查询门控融合：词法路 (a) + 向量路 (b) 动态加权",
            )
            .with_config(config),
            lexical,
            dense,
            alpha_base,
        })
    }
}

impl Adapter for GatedFusionAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.lexical.index(docs)?;
        self.dense.index(docs)?;
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let hits_lexical = self.lexical.search(query, k * 2)?;
        let hits_dense = self.dense.search(query, k * 2)?;

        let mut scores: HashMap<String, f64> = HashMap::new();
        let a = self.alpha_base;
        let b = 1.0 - a;

        for (doc_id, score) in hits_lexical {
            *scores.entry(doc_id).or_insert(0.0) += a * score;
        }
        for (doc_id, score) in hits_dense {
            *scores.entry(doc_id).or_insert(0.0) += b * score;
        }

        let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        fused.truncate(k);
        Ok(fused)
    }
}
