//! 无参数化逻辑执行适配器 (Parameterless Logic Execution Adapter)。
//!
//! 外壳代码零硬编码超参数（无 k1, 无 b, 无 magic weights）。
//! 由内核产出的 logic-graph-v0 工件提供先验显著性与因果拓扑约束，
//! 外壳纯粹作为物理逻辑约束谓词执行引擎运行。

use crate::tokenize::bm25_terms;
use anyhow::{Context, Result};
use sl_core::{Adapter, AdapterInfo, Doc};
use sl_engine::bow::BowScorer;
use sl_engine::index::PositionalIndex;
use sl_engine::logic_executor::{LogicExecutor, QueryConstraint};
use std::collections::HashMap;
use std::sync::Arc;

pub struct LogicExecutionAdapter {
    info: AdapterInfo,
    index: Option<Arc<PositionalIndex>>,
    bow: Option<BowScorer>,
    vocabulary: HashMap<String, u32>,
    /// 词项间因果流约束：src_word -> [(dst_word, strength)]
    flow_constraints: HashMap<String, Vec<(String, f32)>>,
    /// 词项显著性先验
    saliency_map: HashMap<String, f32>,
}

impl LogicExecutionAdapter {
    pub fn new() -> Result<Self> {
        let dataset = std::env::var("SL_DATASET").unwrap_or_else(|_| "scifact".to_string());
        let default_dir = format!("data/{dataset}/artifacts/logic-graph-v0");
        let dir = std::env::var("SL_LOGIC_GRAPH_ARTIFACT").unwrap_or(default_dir);
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            anyhow::bail!(
                "找不到 logic-graph 工件目录 {}（先跑 tools/kernel/build_logic_graph.py {dataset}）",
                dir
            );
        }

        let (flow_constraints, sha256) = Self::load_logic_graph(path)?;

        // 读取 saliency 先验
        let saliency_dir = format!("data/{dataset}/artifacts/saliency-v0/saliency.json");
        let mut saliency_map = HashMap::new();
        if let Ok(bytes) = std::fs::read(&saliency_dir) {
            if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(entries) = parsed.get("entries").and_then(|v| v.as_object()) {
                    for (term, rows) in entries {
                        if let Some(rows) = rows.as_array() {
                            for row in rows {
                                if let Some(pair) = row.as_array() {
                                    if let Some(score) = pair.get(1).and_then(|v| v.as_f64()) {
                                        saliency_map.insert(term.clone(), score as f32);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let config = serde_json::json!({
            "shell_type": "parameterless_predicate_runtime",
            "hyperparameters": "none (pure physics & logic constraint)",
            "prior_factor": {
                "name": "logic_constraint_graph",
                "kind": "token_confidence_list",
                "sha256": sha256,
                "source_terms": flow_constraints.len(),
            }
        });

        Ok(Self {
            info: AdapterInfo::new(
                "logic-exec",
                "parameterless-logic-shell",
                "0.2.0",
                "外壳：crates/sl-adapters/src/logic_exec.rs（无超参谓词执行器）；内核：tools/kernel/build_logic_graph.py",
            )
            .with_config(config),
            index: None,
            bow: None,
            vocabulary: HashMap::new(),
            flow_constraints,
            saliency_map,
        })
    }

    fn load_logic_graph(
        dir: &std::path::Path,
    ) -> Result<(HashMap<String, Vec<(String, f32)>>, String)> {
        let artifact = sl_core::KernelArtifact::from_dir(dir)?;
        let payload_path = dir.join("logic_graph.json");
        let bytes = std::fs::read(&payload_path)
            .with_context(|| format!("读取 {} 失败", payload_path.display()))?;

        let expected = artifact
            .files
            .get("logic_graph.json")
            .ok_or_else(|| anyhow::anyhow!("工件缺少 logic_graph.json 哈希登记"))?;
        let actual = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };
        if actual != *expected {
            anyhow::bail!(
                "{} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）",
                payload_path.display()
            );
        }

        let parsed: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("解析 {} 失败", payload_path.display()))?;
        let mut map = HashMap::new();
        if let Some(entries) = parsed.get("entries").and_then(|v| v.as_object()) {
            for (term, rows) in entries {
                let mut list = Vec::new();
                if let Some(rows) = rows.as_array() {
                    for row in rows {
                        let Some(pair) = row.as_array() else { continue };
                        let (Some(target), Some(strength)) = (
                            pair.first().and_then(|v| v.as_str()),
                            pair.get(1).and_then(|v| v.as_f64()),
                        ) else {
                            continue;
                        };
                        list.push((target.to_string(), strength as f32));
                    }
                }
                map.insert(term.clone(), list);
            }
        }
        Ok((map, actual))
    }
}

impl Adapter for LogicExecutionAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> Result<()> {
        self.vocabulary.clear();

        let mut encoded: Vec<(String, Vec<u32>)> = Vec::with_capacity(docs.len());
        for doc in docs {
            let terms = bm25_terms(&doc.full_text());
            let ids: Vec<u32> = terms
                .into_iter()
                .map(|term| {
                    let next = self.vocabulary.len() as u32;
                    *self.vocabulary.entry(term).or_insert(next)
                })
                .collect();
            encoded.push((doc.id.clone(), ids));
        }

        let index = Arc::new(PositionalIndex::build(encoded));
        self.bow = Some(BowScorer::new(Arc::clone(&index)));
        self.index = Some(index);
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let bow = self.bow.as_ref().context("未建索引（先调 index）")?;
        let index = self.index.as_ref().context("未建索引")?;

        let terms = bm25_terms(query);
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let term_ids: Vec<u32> = terms
            .iter()
            .filter_map(|t| self.vocabulary.get(t).copied())
            .collect();

        if term_ids.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 编译当前查询的逻辑约束描述符 (QueryConstraint)
        let mut anchors = Vec::with_capacity(terms.len());
        for term in &terms {
            if let Some(&id) = self.vocabulary.get(term) {
                let idf = index.idf(id) as f32;
                let saliency = self.saliency_map.get(term).copied().unwrap_or(0.5);
                anchors.push((id, saliency, idf.max(0.1)));
            }
        }

        // 编译词对间的因果流拓扑
        let mut causal_flows = Vec::new();
        for i in 0..terms.len() {
            for j in (i + 1)..terms.len() {
                let (t1, t2) = (&terms[i], &terms[j]);
                if let (Some(&id1), Some(&id2)) = (self.vocabulary.get(t1), self.vocabulary.get(t2))
                {
                    if let Some(targets) = self.flow_constraints.get(t1) {
                        for (dst, strength) in targets {
                            if dst == t2 {
                                causal_flows.push((id1, id2, *strength));
                            }
                        }
                    }
                }
            }
        }

        let constraint = QueryConstraint {
            anchors,
            causal_flows,
            saturation_scale: 2.0,
            gate_softness: 1.0,
        };

        // 2. 粗排候选池（倒排快速获取 top-60 候选）
        let weighted: Vec<(u32, f64)> = term_ids.iter().map(|&id| (id, 1.0)).collect();
        let candidate_hits = bow.search(&weighted, k.max(60));
        if candidate_hits.is_empty() {
            return Ok(Vec::new());
        }

        // 3. 逻辑执行器：纯谓词物理验证打分（零超参数）
        let avg_len = index.avg_len();
        let mut scored: Vec<(String, f64)> = Vec::with_capacity(candidate_hits.len());

        for (doc_id, _) in candidate_hits {
            let doc_idx = match (0..index.len() as u32).find(|&d| index.doc_id(d) == doc_id) {
                Some(idx) => idx,
                None => continue,
            };

            let doc_len = index.doc_len(doc_idx);
            let mut doc_tf: HashMap<u32, u32> = HashMap::new();
            for &(term_id, _, _) in &constraint.anchors {
                let count = index.positions_in(term_id, doc_idx).len() as u32;
                if count > 0 {
                    doc_tf.insert(term_id, count);
                }
            }

            let (final_score, _) = LogicExecutor::execute(
                &constraint,
                &doc_tf,
                |t| index.positions_in(t, doc_idx),
                doc_len,
                avg_len,
            );

            if final_score > 0.0 {
                scored.push((doc_id, final_score));
            }
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logic_execution_adapter_indexes_and_executes_predicates() {
        let mut adapter = LogicExecutionAdapter {
            info: AdapterInfo::new("test", "test", "0.2.0", "test"),
            index: None,
            bow: None,
            vocabulary: HashMap::new(),
            flow_constraints: HashMap::new(),
            saliency_map: HashMap::new(),
        };

        let docs = vec![Doc {
            id: "d1".to_string(),
            title: "Logic Test".to_string(),
            text: "Testing parameterless logic runtime.".to_string(),
        }];

        adapter.index(&docs).unwrap();
        let hits = adapter.search("parameterless runtime", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "d1");
        assert!(hits[0].1 > 0.0);
    }
}
