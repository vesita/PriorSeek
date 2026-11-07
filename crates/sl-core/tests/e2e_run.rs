//! 端到端：一份 3 文档的小数据集 + 两个行为已知的假适配器，走完
//! `run_dataset → 指标 → 运行记录 → 报告`，钉住「同轮同题、排序归一、指标口径」的联动。

use sl_core::adapter::{Adapter, AdapterInfo};
use sl_core::canonical::{load_dataset, write_dataset};
use sl_core::run::run_dataset;
use sl_core::types::{Counts, Dataset, Doc, Manifest, Qrels, Query};
use sl_core::report::render_markdown;
use std::collections::BTreeMap;

/// 假适配器：按查询里是否包含文档关键词返回固定顺序，行为完全可预测。
struct KeywordAdapter {
    info: AdapterInfo,
    /// (query 关键词, 有序 doc id)
    answers: Vec<(&'static str, Vec<&'static str>)>,
    hits: std::cell::Cell<usize>,
}

impl KeywordAdapter {
    fn new(name: &str, answers: Vec<(&'static str, Vec<&'static str>)>) -> Self {
        Self {
            info: AdapterInfo::new(name, "test", "0", "tests/e2e_run.rs"),
            answers,
            hits: std::cell::Cell::new(0),
        }
    }
}

impl Adapter for KeywordAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, _docs: &[Doc]) -> anyhow::Result<()> {
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        self.hits.set(self.hits.get() + 1);
        let docs = self
            .answers
            .iter()
            .find(|(key, _)| query.contains(key))
            .map(|(_, docs)| docs.clone())
            .unwrap_or_default();
        Ok(docs
            .iter()
            .take(k)
            .enumerate()
            .map(|(idx, id)| ((*id).to_string(), 10.0 - idx as f64))
            .collect())
    }
}

fn tiny_dataset() -> Dataset {
    let docs = vec![
        Doc { id: "d1".into(), title: String::new(), text: "参数修改后需要重启".into() },
        Doc { id: "d2".into(), title: String::new(), text: "报警和预警的区别".into() },
        Doc { id: "d3".into(), title: String::new(), text: "刀具磨损补偿".into() },
    ];
    let queries = vec![
        Query { id: "q1".into(), text: "参数重启".into() },
        Query { id: "q2".into(), text: "报警预警".into() },
    ];
    let mut qrels: Qrels = BTreeMap::new();
    qrels.entry("q1".into()).or_default().insert("d1".into(), 1);
    qrels.entry("q2".into()).or_default().insert("d2".into(), 1);
    Dataset {
        name: "tiny".into(),
        docs,
        queries,
        qrels,
        manifest: Manifest {
            name: "tiny".into(),
            version: "test".into(),
            source: vec!["synthetic".into()],
            built_by: "tests/e2e_run.rs".into(),
            seed: None,
            notes: "端到端夹具".into(),
            counts: Counts::default(),
            files: BTreeMap::new(),
        },
    }
}

#[test]
fn run_produces_comparable_metrics_and_report() {
    let tmp = tempfile::tempdir().unwrap();
    let dataset = tiny_dataset();
    write_dataset(&dataset, tmp.path(), dataset.manifest.clone()).unwrap();
    let loaded = load_dataset(tmp.path()).unwrap();
    loaded.verify_strict().unwrap();

    let mut good = KeywordAdapter::new(
        "good",
        vec![("参数", vec!["d1", "d3"]), ("报警", vec!["d2"])],
    );
    let mut bad = KeywordAdapter::new(
        "bad",
        vec![("参数", vec!["d3", "d1"]), ("报警", vec!["d1"])],
    );

    let outcomes = run_dataset(
        &loaded.dataset,
        &mut [Box::new(good) as Box<dyn Adapter>, Box::new(bad)],
        10,
        1,
    )
    .unwrap();
    assert_eq!(outcomes.len(), 2);

    let good = &outcomes[0];
    assert_eq!(good.info.name, "good");
    assert_eq!(good.metrics.queries_scored, 2);
    assert!((good.metrics.mrr - 1.0).abs() < 1e-9);
    assert!((good.metrics.map - 1.0).abs() < 1e-9);
    assert_eq!(good.duplicate_ids, 0);
    assert_eq!(good.unknown_ids, 0);

    let bad = &outcomes[1];
    // q1 排到第二（首个相关在 rank 2 → MRR 0.5，recall@1=0、recall@3=1）；
    // q2 完全没命中（全 0）→ 两条查询宏平均：MRR 0.25、recall@3 0.5。
    assert!((bad.metrics.mrr - 0.25).abs() < 1e-9);
    assert_eq!(bad.metrics.recall[&1], 0.0);
    assert_eq!(bad.metrics.recall[&3], 0.5);

    let record = sl_core::run::RunRecord {
        run_id: "e2e".into(),
        dataset: (&loaded.dataset).into(),
        k: 10,
        warmup: 1,
        adapters: outcomes.iter().map(sl_core::run::AdapterResult::from).collect(),
        env: sl_core::run::EnvInfo {
            git_commit: None,
            rustc: "test".into(),
            host: "localhost".into(),
            os: "test".into(),
            started_at: "2026-01-01T00:00:00+00:00".into(),
        },
    };
    let md = render_markdown(&record);
    assert!(md.contains("| good |"));
    assert!(md.contains("| bad |"));
    let good_pos = md.find("| good |").unwrap();
    let bad_pos = md.find("| bad |").unwrap();
    assert!(good_pos < bad_pos, "好适配器应排在前面");

    // 运行记录要能序列化回读（run.json 的可复现性）
    let json = serde_json::to_string(&record).unwrap();
    let back: sl_core::run::RunRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.dataset.fingerprint, record.dataset.fingerprint);
    assert_eq!(back.adapters.len(), 2);
}

#[test]
fn warmup_does_not_change_metrics_only_latency() {
    let tmp = tempfile::tempdir().unwrap();
    let dataset = tiny_dataset();
    write_dataset(&dataset, tmp.path(), dataset.manifest.clone()).unwrap();
    let loaded = load_dataset(tmp.path()).unwrap();

    let mut without = KeywordAdapter::new("a", vec![("参数", vec!["d1"]), ("报警", vec!["d2"])]);
    let outcome_without = run_dataset(&loaded.dataset, &mut [Box::new(without)], 10, 0).unwrap();

    let mut with = KeywordAdapter::new("a", vec![("参数", vec!["d1"]), ("报警", vec!["d2"])]);
    let outcome_with = run_dataset(&loaded.dataset, &mut [Box::new(with)], 10, 2).unwrap();

    assert_eq!(outcome_without[0].metrics, outcome_with[0].metrics);
    assert_eq!(outcome_without[0].latency_ms.len(), 2);
    assert_eq!(outcome_with[0].latency_ms.len(), 2);
}
