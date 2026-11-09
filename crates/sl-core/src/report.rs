//! 把运行记录渲染成 Markdown 报告。
//!
//! 报告只做一件事：在**同一数据集指纹 + 同一 k** 内给出排序表，并在表头写清这个前提。
//! 不跨数据集合并、不把不同任务族的分数放在一张表里排名。

use crate::run::RunRecord;
use crate::run::AdapterResult;

pub fn render_markdown(record: &RunRecord) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {} 检索对比\n\n", record.dataset.name));
    out.push_str(&format!(
        "- 数据集指纹：`{}`\n- 规模：{} 文档 / {} 查询 / {} 条计分查询\n- 召回深度 k = {}，预热 {} 条查询\n- 运行：{}（{} / {} / {}）\n",
        record.dataset.fingerprint,
        record.dataset.docs,
        record.dataset.queries,
        record.dataset.scored_queries,
        record.k,
        record.warmup,
        record.env.started_at,
        record.env.os,
        record.env.rustc,
        record.env.host,
    ));
    if let Some(commit) = &record.env.git_commit {
        out.push_str(&format!("- harness commit：`{commit}`\n"));
    }
    out.push_str(&format!("- 运行 id：`{}`\n\n", record.run_id));

    out.push_str("> **可比范围**：只有同一数据集指纹、同一 k 的同一张表内可以横向比较；\n");
    out.push_str("> 不同数据集 / 不同任务族的绝对分数不构成「谁更强」的结论。\n\n");

    let mut ordered: Vec<&AdapterResult> = record.adapters.iter().collect();
    ordered.sort_by(|a, b| b.metrics.ndcg.total_cmp(&a.metrics.ndcg));

    let ks: Vec<usize> = record
        .adapters
        .first()
        .map(|a| a.metrics.recall.keys().copied().collect())
        .unwrap_or_default();
    let sample = ordered.first();

    out.push_str("| # | 适配器 | 族 |");
    for k in &ks {
        out.push_str(&format!(" R@{k} |"));
    }
    out.push_str(" MRR | nDCG | MAP | 索引 ms | 查询 p50 ms | 查询 p95 ms |\n");
    out.push_str("|---:|---|---|");
    for _ in &ks {
        out.push_str("---:|");
    }
    out.push_str("---:|---:|---:|---:|---:|---:|\n");

    for (idx, adapter) in ordered.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} |",
            idx + 1,
            adapter.info.name,
            adapter.info.family
        ));
        for k in &ks {
            out.push_str(&format!(
                " {:.4} |",
                adapter.metrics.recall.get(k).copied().unwrap_or(f64::NAN)
            ));
        }
        out.push_str(&format!(
            " {:.4} | {:.4} | {:.4} | {:.1} | {:.3} | {:.3} |\n",
            adapter.metrics.mrr,
            adapter.metrics.ndcg,
            adapter.metrics.map,
            adapter.index_ms,
            adapter.latency_ms_p50,
            adapter.latency_ms_p95,
        ));
    }

    if let Some(sample) = sample {
        out.push_str(&format!(
            "\nMRR 窗口 = @{}，nDCG 窗口 = @{}，MAP 窗口 = @{}；计分查询 {} 条，跳过 {} 条（qrels 里没有正相关）。\n",
            sample.metrics.mrr_at,
            sample.metrics.ndcg_at,
            sample.metrics.map_at,
            sample.metrics.queries_scored,
            sample.metrics.queries_skipped,
        ));
    }

    let noisy: Vec<&AdapterResult> =
        record.adapters.iter().filter(|a| a.unknown_ids > 0 || a.duplicate_ids > 0).collect();
    if !noisy.is_empty() {
        out.push_str("\n**契约告警**：以下适配器返回过不属于语料的 doc id 或重复 id（已去重/丢弃并计入指标，但说明实现有 bug）：\n");
        for adapter in noisy {
            out.push_str(&format!(
                "- `{}`：未知 id {} 条，重复 id {} 条\n",
                adapter.info.name, adapter.unknown_ids, adapter.duplicate_ids
            ));
        }
    }

    out.push_str("\n## 适配器来源\n\n");
    for adapter in &ordered {
        out.push_str(&format!(
            "- `{}`（{}，{}）：{}\n",
            adapter.info.name, adapter.info.family, adapter.info.version, adapter.info.source
        ));
        if !adapter.info.config.is_null() {
            out.push_str(&format!(
                "  - 参数：`{}`\n",
                serde_json::to_string(&adapter.info.config).unwrap_or_default()
            ));
        }
        for artifact in &adapter.info.artifacts {
            out.push_str(&format!(
                "  - 工件：`{}`（{} / {}；manifest {}…；运行时 {}；训练来源：{}）\n",
                artifact.id,
                artifact.kind,
                artifact.weights_origin,
                &artifact.manifest_sha256[..12.min(artifact.manifest_sha256.len())],
                artifact.runtime_device,
                artifact.trained_on,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::AdapterInfo;
    use crate::metrics::Metrics;
    use crate::run::{DatasetSummary, EnvInfo};

    fn fake_adapter(name: &str, ndcg: f64) -> AdapterResult {
        let mut metrics = Metrics { mrr_at: 10, ndcg_at: 10, map_at: 100, ..Default::default() };
        metrics.recall.insert(1, 0.5);
        metrics.recall.insert(10, 0.9);
        metrics.ndcg = ndcg;
        metrics.mrr = 0.7;
        metrics.map = 0.6;
        metrics.queries_scored = 3;
        metrics.queries_skipped = 1;
        AdapterResult {
            info: AdapterInfo::new(name, "test", "0", "unit"),
            index_ms: 1.0,
            search_ms_total: 2.0,
            latency_ms_p50: 0.1,
            latency_ms_p95: 0.2,
            metrics,
            duplicate_ids: 0,
            unknown_ids: 0,
        }
    }

    fn fake_record() -> RunRecord {
        RunRecord {
            run_id: "r1".into(),
            dataset: DatasetSummary {
                name: "fixture".into(),
                version: "1".into(),
                fingerprint: "f".repeat(64),
                docs: 10,
                queries: 5,
                scored_queries: 3,
            },
            k: 100,
            warmup: 2,
            adapters: vec![fake_adapter("low", 0.1), fake_adapter("high", 0.9)],
            env: EnvInfo {
                git_commit: Some("abc".into()),
                rustc: "rustc test".into(),
                host: "host".into(),
                os: "os".into(),
                started_at: "2026-01-01T00:00:00+08:00".into(),
                gpu: None,
            },
        }
    }

    #[test]
    fn markdown_ranks_by_ndcg_and_states_comparability() {
        let md = render_markdown(&fake_record());
        assert!(md.contains("可比范围"));
        let high = md.find("| high |").expect("high 行存在");
        let low = md.find("| low |").expect("low 行存在");
        assert!(high < low, "nDCG 高的适配器要排在前面");
        assert!(md.contains("R@1"));
    }
}
