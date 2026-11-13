//! 跑一轮对比：建索引 → 逐查询检索 → 归一排序 → 算指标 → 组装运行记录。
//!
//! 这一层是「公平性」的强制点：
//! - 同一轮里所有适配器看到**同一份** `Dataset`、同一个 `k`、同样的查询顺序；
//! - 适配器返回的 doc id 必须属于语料，重复 id 去重、未知 id 记账（不静默吞掉）；
//! - 索引耗时与查询延迟分开记，查询延迟按进程内单查询墙钟时间统计（含适配器自身开销）。

use crate::adapter::Adapter;
use crate::metrics::{evaluate_rankings, Metrics, PerQuery};
use crate::types::Dataset;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// 一个适配器在一轮里的全部产出。
pub struct AdapterOutcome {
    pub info: crate::adapter::AdapterInfo,
    pub index_ms: f64,
    pub search_ms_total: f64,
    pub latency_ms: Vec<f64>,
    pub metrics: Metrics,
    pub per_query: Vec<PerQuery>,
    pub duplicate_ids: usize,
    pub unknown_ids: usize,
}

impl AdapterOutcome {
    pub fn latency_p50(&self) -> f64 {
        percentile(&self.latency_ms, 0.50)
    }

    pub fn latency_p95(&self) -> f64 {
        percentile(&self.latency_ms, 0.95)
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// 跑一轮：所有适配器在同一数据集上各建一次索引、检索全部查询。
///
/// `warmup` 条查询会先跑一遍但不计入延迟（其余指标仍用完整一轮的结果）。
pub fn run_dataset(
    dataset: &Dataset,
    adapters: &mut [Box<dyn Adapter>],
    k: usize,
    warmup: usize,
) -> Result<Vec<AdapterOutcome>> {
    if k == 0 {
        bail!("k 必须大于 0");
    }
    let corpus: HashSet<&str> = dataset.docs.iter().map(|d| d.id.as_str()).collect();
    let mut outcomes = Vec::with_capacity(adapters.len());

    for adapter in adapters.iter_mut() {
        let info = adapter.info().clone();
        eprintln!(
            "[sl] {} ({}) 建索引：{} 篇文档",
            info.name,
            info.family,
            dataset.docs.len()
        );
        let started = Instant::now();
        adapter
            .index(&dataset.docs)
            .map_err(|err| anyhow::anyhow!("适配器 {} 建索引失败：{err}", info.name))?;
        let index_ms = started.elapsed().as_secs_f64() * 1000.0;

        // 预热：结果丢弃，不影响指标，只让 JIT/缓存/文件句柄进入稳定态。
        for query in dataset.queries.iter().take(warmup) {
            adapter
                .search(&query.text, k)
                .map_err(|err| anyhow::anyhow!("适配器 {} 检索失败（预热）：{err}", info.name))?;
        }

        let mut rankings: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut latency_ms = Vec::with_capacity(dataset.queries.len());
        let mut search_ms_total = 0.0;
        let mut duplicate_ids = 0usize;
        let mut unknown_ids = 0usize;

        for query in &dataset.queries {
            let started = Instant::now();
            let candidates = adapter.search(&query.text, k).map_err(|err| {
                anyhow::anyhow!("适配器 {} 检索失败（查询 {}）：{err}", info.name, query.id)
            })?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            search_ms_total += elapsed;
            latency_ms.push(elapsed);

            let (ranking, dup, unknown) = normalize_candidates(candidates, k, &corpus);
            duplicate_ids += dup;
            unknown_ids += unknown;
            rankings.insert(query.id.clone(), ranking);
        }

        let (metrics, per_query) = evaluate_rankings(&rankings, &dataset.qrels, k);
        eprintln!(
            "[sl] {} 完成：nDCG@{}={:.4} MRR@{}={:.4} recall@10={:.4}（未知 doc id {}，重复 {}）",
            info.name,
            metrics.ndcg_at,
            metrics.ndcg,
            metrics.mrr_at,
            metrics.mrr,
            metrics.recall.get(&10).copied().unwrap_or(0.0),
            unknown_ids,
            duplicate_ids
        );

        outcomes.push(AdapterOutcome {
            info,
            index_ms,
            search_ms_total,
            latency_ms,
            metrics,
            per_query,
            duplicate_ids,
            unknown_ids,
        });
    }
    Ok(outcomes)
}

/// 把适配器返回的候选归一成排序列表：
/// 按分数降序、同分按 doc id 升序；去重（保留首次出现）；丢掉不在语料里的 id 并记账；截断到 k。
pub fn normalize_candidates(
    mut candidates: Vec<(String, f64)>,
    k: usize,
    corpus: &HashSet<&str>,
) -> (Vec<String>, usize, usize) {
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut seen: HashSet<String> = HashSet::new();
    let mut ranking = Vec::with_capacity(k.min(candidates.len()));
    let mut duplicates = 0usize;
    let mut unknown = 0usize;
    for (doc_id, _score) in candidates {
        if !corpus.contains(doc_id.as_str()) {
            unknown += 1;
            continue;
        }
        if !seen.insert(doc_id.clone()) {
            duplicates += 1;
            continue;
        }
        if ranking.len() < k {
            ranking.push(doc_id);
        }
    }
    (ranking, duplicates, unknown)
}

/// 运行记录里的一项（summary 级；逐查询明细写到 `per_query/<adapter>.jsonl`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterResult {
    pub info: crate::adapter::AdapterInfo,
    pub index_ms: f64,
    pub search_ms_total: f64,
    pub latency_ms_p50: f64,
    pub latency_ms_p95: f64,
    pub metrics: Metrics,
    pub duplicate_ids: usize,
    pub unknown_ids: usize,
}

impl From<&AdapterOutcome> for AdapterResult {
    fn from(outcome: &AdapterOutcome) -> Self {
        Self {
            info: outcome.info.clone(),
            index_ms: outcome.index_ms,
            search_ms_total: outcome.search_ms_total,
            latency_ms_p50: outcome.latency_p50(),
            latency_ms_p95: outcome.latency_p95(),
            metrics: outcome.metrics.clone(),
            duplicate_ids: outcome.duplicate_ids,
            unknown_ids: outcome.unknown_ids,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub name: String,
    pub version: String,
    /// manifest 指纹：不同指纹 = 不是同一道题。
    pub fingerprint: String,
    pub docs: usize,
    pub queries: usize,
    pub scored_queries: usize,
}

impl From<&Dataset> for DatasetSummary {
    fn from(dataset: &Dataset) -> Self {
        Self {
            name: dataset.name.clone(),
            version: dataset.manifest.version.clone(),
            fingerprint: dataset.manifest.fingerprint(),
            docs: dataset.docs.len(),
            queries: dataset.queries.len(),
            scored_queries: dataset.scored_query_ids().len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    pub git_commit: Option<String>,
    pub rustc: String,
    pub host: String,
    pub os: String,
    pub started_at: String,
    /// 机器上是否有 GPU（只记录，不参与打分；内核准入要求运行时 CPU）。
    #[serde(default)]
    pub gpu: Option<String>,
}

impl EnvInfo {
    pub fn collect() -> Self {
        Self {
            git_commit: command_output("git", &["rev-parse", "HEAD"]),
            rustc: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
            host: std::fs::read_to_string("/etc/hostname")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "unknown".into()),
            os: os_pretty_name(),
            started_at: chrono::Local::now().to_rfc3339(),
            gpu: command_output("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]),
        }
    }
}

fn command_output(cmd: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text.lines().next().unwrap_or_default().to_string())
    }
}

fn os_pretty_name() -> String {
    if let Ok(text) = std::fs::read_to_string("/etc/os-release") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("PRETTY_NAME=") {
                return rest.trim_matches('"').to_string();
            }
        }
    }
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub dataset: DatasetSummary,
    pub k: usize,
    pub warmup: usize,
    pub adapters: Vec<AdapterResult>,
    pub env: EnvInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sorts_by_score_then_id_and_dedupes() {
        let corpus: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let candidates = vec![
            ("b".to_string(), 2.0),
            ("a".to_string(), 2.0),
            ("ghost".to_string(), 9.0),
            ("b".to_string(), 1.0),
            ("c".to_string(), 0.5),
        ];
        let (ranking, dup, unknown) = normalize_candidates(candidates, 3, &corpus);
        assert_eq!(
            ranking,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(dup, 1);
        assert_eq!(unknown, 1);
    }

    #[test]
    fn normalize_truncates_to_k() {
        let corpus: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let candidates: Vec<(String, f64)> =
            vec![("a".into(), 3.0), ("b".into(), 2.0), ("c".into(), 1.0)];
        let (ranking, _, _) = normalize_candidates(candidates, 2, &corpus);
        assert_eq!(ranking, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn percentile_handles_edges() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[5.0], 0.95), 5.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), 3.0);
    }
}
