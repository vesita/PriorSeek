//! `sl` —— search-lab 的命令行入口。
//!
//! 三件事：校验规范数据集、跑一轮适配器对比、把运行记录渲染成报告。
//! 数据集构建在 Python 侧（`tools/datasets/`），见 `docs/DATASETS.md`。

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use sl_core::{canonical, report, run};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "sl",
    version,
    about = "search-lab：搜索算法对比实验台（规范数据集 → 适配器 → 指标 → 报告）",
    long_about = "同一份规范数据集、同一套指标口径下跑多个搜索算法，产出可复现的对比报告。\n口径与纪律见 docs/STANDARD.md。"
)]
struct Cli {
    /// 规范数据集根目录（每个数据集一个子目录，含 corpus/queries/qrels/manifest）
    #[arg(long, global = true, default_value = "data")]
    data_dir: PathBuf,

    /// 运行结果根目录
    #[arg(long, global = true, default_value = "results")]
    results_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 数据集：列出 / 校验 / 查看清单
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    /// 适配器：列出当前可用的算法
    Adapters,
    /// 跑一轮对比（同一数据集、同一 k、同一查询顺序）
    Run {
        /// 数据集名（`data/<name>/`）
        #[arg(long)]
        dataset: String,
        /// 逗号分隔的适配器名，如 `bm25,faq-hybrid,rs`
        #[arg(long, value_delimiter = ',')]
        adapters: Vec<String>,
        /// 召回深度
        #[arg(long, default_value_t = 100)]
        k: usize,
        /// 预热查询条数（不计入延迟，但仍计入指标）
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// 结果目录名追加的标签
        #[arg(long)]
        label: Option<String>,
        /// 跳过「规范数据与 manifest 摘要一致」的强校验（只用于调试）
        #[arg(long)]
        allow_dirty: bool,
    },
    /// 重新渲染某次运行目录里的 report.md
    Report {
        /// 运行目录（含 run.json）
        run_dir: PathBuf,
    },
    /// 单查询调试：构造指定适配器，打印一条查询的 top-k（用于与参考实现对拍）
    Probe {
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        adapter: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// 把语料索引 / 查询打印成 JSON
        #[arg(long)]
        json: bool,
    },
    /// 字段启发评测：train pool 内切 heldout，比较 bm25 与 field-bm25 各臂
    FieldEval {
        #[arg(long, default_value = "scifact")]
        dataset: String,
        /// 召回深度
        #[arg(long, default_value_t = 100)]
        k: usize,
        /// 逐查询指标与配对区间的 JSON 输出
        #[arg(long, default_value = "results/field-eval.json")]
        output: PathBuf,
    },
    /// 概念启发评测：内核学到的 `token → 概念` 因子，与字段加权正交组合
    ConceptEval {
        #[arg(long, default_value = "scifact")]
        dataset: String,
        /// 召回深度
        #[arg(long, default_value_t = 100)]
        k: usize,
        /// 逐查询指标与配对区间的 JSON 输出
        #[arg(long, default_value = "results/concept-eval.json")]
        output: PathBuf,
        /// 在哪个划分上评测：heldout（默认，判决用）或 outer（n 更大，Δ_min 更小）
        /// heldout（默认，判决）/ outer（方向复核）/ test（预注册后才能用一次）
        #[arg(long, value_parser = ["heldout", "outer", "test"], default_value = "heldout")]
        split: String,
        /// 把逐查询 nDCG 落盘（任意两臂之间的配对对比要用它）
        #[arg(long)]
        dump_per_query: bool,
        /// 概念因子工件 id（默认 concept-v0；对比不同候选空间时用 concept-v1）
        #[arg(long, default_value = "concept-v0")]
        artifact: String,
    },
    /// 内核工件：复核某次运行记录的工件哈希（换机器 / 长期归档后确认工件没被换掉）
    Artifact {
        #[command(subcommand)]
        command: ArtifactCommand,
    },
}

#[derive(Subcommand)]
enum ArtifactCommand {
    /// 按 run.json 里登记的工件目录重新算哈希并比对
    Verify {
        /// 运行目录（含 run.json）
        run_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum DatasetCommand {
    /// 列出 data 目录下所有规范数据集
    List,
    /// 校验一个数据集的 manifest 摘要与规模（严格，不一致即失败）
    Verify { name: String },
    /// 打印一个数据集的 manifest
    Show { name: String },
}

fn main() {
    if let Err(err) = real_main() {
        eprintln!("[sl] 错误：{err:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Dataset { command } => match command {
            DatasetCommand::List => list_datasets(&cli.data_dir),
            DatasetCommand::Verify { name } => verify_dataset(&cli.data_dir, &name),
            DatasetCommand::Show { name } => show_dataset(&cli.data_dir, &name),
        },
        Command::Adapters => {
            let names = sl_adapters::names();
            if names.is_empty() {
                println!("（还没有注册任何适配器）");
            } else {
                for name in names {
                    println!("{name}");
                }
            }
            Ok(())
        }
        Command::Run {
            dataset,
            adapters,
            k,
            warmup,
            label,
            allow_dirty,
        } => run_one(
            &cli.data_dir,
            &cli.results_dir,
            &dataset,
            &adapters,
            k,
            warmup,
            label,
            allow_dirty,
        ),
        Command::Report { run_dir } => rerender(&run_dir),
        Command::Probe {
            dataset,
            adapter,
            query,
            k,
            json,
        } => probe(&cli.data_dir, &dataset, &adapter, &query, k, json),
        Command::FieldEval { dataset, k, output } => run_field_eval(
            &cli.data_dir,
            &dataset,
            k,
            &output,
            EvalSplit::Heldout,
            false,
        ),
        Command::ConceptEval {
            dataset,
            k,
            output,
            split,
            dump_per_query,
            artifact,
        } => run_concept_eval(
            &cli.data_dir,
            &dataset,
            k,
            &output,
            EvalSplit::from_arg(&split),
            dump_per_query,
            &artifact,
        ),
        Command::Artifact { command } => match command {
            ArtifactCommand::Verify { run_dir } => verify_artifacts(&run_dir),
        },
    }
}

fn probe(
    data_dir: &Path,
    dataset_name: &str,
    adapter_name: &str,
    query: &str,
    k: usize,
    json: bool,
) -> Result<()> {
    let loaded = canonical::load_dataset(&data_dir.join(dataset_name))?;
    if !loaded.pruned.is_clean() {
        eprintln!("[sl] 警告：qrels 有悬挂条目，已剔除");
    }
    std::env::set_var("SL_DATA_DIR", data_dir);
    std::env::set_var("SL_DATASET", dataset_name);
    let dataset = loaded.dataset;
    let mut adapter = sl_adapters::build(adapter_name)?;
    adapter.index(&dataset.docs)?;
    let hits = adapter.search(query, k)?;
    let info = adapter.info();
    if json {
        let payload = serde_json::json!({
            "dataset": dataset.name,
            "adapter": info.name,
            "query": query,
            "hits": hits.iter().map(|(id, score)| serde_json::json!({"id": id, "score": score})).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        for (rank, (id, score)) in hits.iter().enumerate() {
            println!("{}\t{}\t{:.6}", rank + 1, id, score);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_artifacts(run_dir: &Path) -> Result<()> {
    let path = run_dir.join("run.json");
    let text = fs::read_to_string(&path).with_context(|| format!("读 {} 失败", path.display()))?;
    let record: run::RunRecord =
        serde_json::from_str(&text).with_context(|| format!("解析 {} 失败", path.display()))?;

    let mut checked = 0usize;
    for adapter in &record.adapters {
        for artifact in &adapter.info.artifacts {
            let dir = Path::new(&artifact.dir);
            let current = sl_core::KernelArtifact::from_dir(dir).with_context(|| {
                format!(
                    "复核适配器 {} 的工件 {} 失败",
                    adapter.info.name, artifact.id
                )
            })?;
            if current.manifest_sha256 != artifact.manifest_sha256 {
                bail!(
                    "适配器 {} 的工件 {} 与运行记录不一致：manifest 哈希 {} ≠ 记录里的 {}",
                    adapter.info.name,
                    artifact.id,
                    &current.manifest_sha256[..12],
                    &artifact.manifest_sha256[..12]
                );
            }
            if current.files != artifact.files {
                bail!(
                    "适配器 {} 的工件 {} 内容文件哈希与运行记录不一致",
                    adapter.info.name,
                    artifact.id
                );
            }
            checked += 1;
            println!(
                "OK {} → {}（{}，{}…，运行时 {}）",
                adapter.info.name,
                artifact.id,
                artifact.kind,
                &artifact.manifest_sha256[..12],
                artifact.runtime_device
            );
        }
    }
    if checked == 0 {
        println!("（{} 里没有登记内核工件）", path.display());
    } else {
        println!("已复核 {checked} 个工件");
    }
    Ok(())
}

fn list_datasets(data_dir: &Path) -> Result<()> {
    if !data_dir.exists() {
        println!(
            "{} 不存在：还没有构建任何数据集（见 docs/DATASETS.md）",
            data_dir.display()
        );
        return Ok(());
    }
    let mut found = 0usize;
    let mut entries: Vec<PathBuf> = fs::read_dir(data_dir)
        .with_context(|| format!("读 {} 失败", data_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join(canonical::MANIFEST_FILE).exists())
        .collect();
    entries.sort();
    println!("| 数据集 | 版本 | 文档 | 查询 | 计分查询 | 指纹 |");
    println!("|---|---|---:|---:|---:|---|");
    for dir in entries {
        // 目录里可能有别的工程的 manifest.json（比如缓存目录），读不懂就跳过并说明，不整体失败。
        let loaded = match canonical::load_dataset(&dir) {
            Ok(loaded) => loaded,
            Err(err) => {
                eprintln!(
                    "[sl] 跳过 {}：不是 search-lab 规范数据集（{err:#}）",
                    dir.display()
                );
                continue;
            }
        };
        let m = &loaded.dataset.manifest;
        found += 1;
        println!(
            "| {} | {} | {} | {} | {} | {}… |",
            m.name,
            m.version,
            m.counts.docs,
            m.counts.queries,
            m.counts.scored_queries,
            &m.fingerprint()[..12]
        );
    }
    if found == 0 {
        println!("（{} 下没有规范数据集）", data_dir.display());
    }
    Ok(())
}

fn verify_dataset(data_dir: &Path, name: &str) -> Result<()> {
    let dir = data_dir.join(name);
    let loaded = canonical::load_dataset(&dir).with_context(|| {
        format!("加载数据集 {name} 失败；先构建：`uv run tools/datasets/build.py {name}`")
    })?;
    loaded.verify_strict()?;
    let m = &loaded.dataset.manifest;
    println!(
        "OK {} v{}：{} 文档 / {} 查询 / {} 计分查询，指纹 {}",
        m.name,
        m.version,
        m.counts.docs,
        m.counts.queries,
        m.counts.scored_queries,
        m.fingerprint()
    );
    Ok(())
}

fn show_dataset(data_dir: &Path, name: &str) -> Result<()> {
    let path = data_dir.join(name).join(canonical::MANIFEST_FILE);
    let text = fs::read_to_string(&path).with_context(|| format!("读 {} 失败", path.display()))?;
    println!("{text}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    data_dir: &Path,
    results_dir: &Path,
    dataset_name: &str,
    adapter_names: &[String],
    k: usize,
    warmup: usize,
    label: Option<String>,
    allow_dirty: bool,
) -> Result<()> {
    if adapter_names.is_empty() {
        bail!("至少要给一个 --adapters（可用：{}）", format_names());
    }
    if k == 0 {
        bail!("--k 必须大于 0");
    }

    let dir = data_dir.join(dataset_name);
    let loaded = canonical::load_dataset(&dir).with_context(|| {
        format!("加载数据集 {dataset_name} 失败；先构建：`uv run tools/datasets/build.py {dataset_name}`")
    })?;
    if !allow_dirty {
        loaded.verify_strict().with_context(|| {
            format!(
                "数据集 {dataset_name} 未通过校验（调试可用 --allow-dirty 绕过，但结果不可入库）"
            )
        })?;
    }
    if !loaded.pruned.is_clean() {
        eprintln!(
            "[sl] 警告：qrels 有 {} 条指向不存在的文档、{} 个查询不存在，已剔除",
            loaded.pruned.dropped_qrels, loaded.pruned.dropped_queries
        );
    }
    let dataset = loaded.dataset;

    // 把数据集上下文交给适配器：内核适配器（如 `seq`）按它定位 `data/<dataset>/artifacts/<id>/`。
    // 显式设置比让适配器猜路径可靠；单独的 `SL_SEQ_ARTIFACT` 仍可覆盖具体工件目录。
    std::env::set_var("SL_DATA_DIR", data_dir);
    std::env::set_var("SL_DATASET", dataset_name);

    let mut adapters: Vec<Box<dyn sl_core::Adapter>> = Vec::new();
    for name in adapter_names {
        adapters.push(sl_adapters::build(name)?);
    }

    eprintln!(
        "[sl] 数据集 {}（{} 文档 / {} 查询 / {} 计分），k={}，适配器 {} 个",
        dataset.name,
        dataset.docs.len(),
        dataset.queries.len(),
        dataset.scored_query_ids().len(),
        k,
        adapters.len()
    );

    let started_at = chrono::Local::now();
    let outcomes = run::run_dataset(&dataset, &mut adapters, k, warmup)?;

    let run_id = format!("{}-{}", dataset.name, started_at.format("%Y%m%d-%H%M%S"));
    let run_id = match &label {
        Some(label) => format!("{run_id}-{label}"),
        None => run_id,
    };
    let out_dir = results_dir.join(&run_id);
    fs::create_dir_all(out_dir.join("per_query"))
        .with_context(|| format!("创建 {} 失败", out_dir.display()))?;

    let record = run::RunRecord {
        run_id: run_id.clone(),
        dataset: (&dataset).into(),
        k,
        warmup,
        adapters: outcomes.iter().map(run::AdapterResult::from).collect(),
        env: run::EnvInfo::collect(),
    };
    let run_json = serde_json::to_string_pretty(&record)?;
    fs::write(out_dir.join("run.json"), run_json + "\n")?;

    for outcome in &outcomes {
        let path = out_dir
            .join("per_query")
            .join(format!("{}.jsonl", outcome.info.name));
        let file =
            fs::File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        for row in &outcome.per_query {
            serde_json::to_writer(&mut writer, row)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }

    let markdown = report::render_markdown(&record);
    fs::write(out_dir.join("report.md"), &markdown)?;
    println!("{markdown}");
    eprintln!("[sl] 运行记录已写入 {}", out_dir.display());
    Ok(())
}

fn rerender(run_dir: &Path) -> Result<()> {
    let path = run_dir.join("run.json");
    let text = fs::read_to_string(&path).with_context(|| format!("读 {} 失败", path.display()))?;
    let record: run::RunRecord =
        serde_json::from_str(&text).with_context(|| format!("解析 {} 失败", path.display()))?;
    let markdown = report::render_markdown(&record);
    fs::write(run_dir.join("report.md"), &markdown)?;
    println!("{markdown}");
    Ok(())
}

fn format_names() -> String {
    let names = sl_adapters::names();
    if names.is_empty() {
        "（还没有注册任何适配器）".to_string()
    } else {
        names.join(", ")
    }
}

/// train/heldout 划分：FNV-1a(qid, seed=20261110) % 5 == 0 进 heldout。
fn is_heldout_qid(qid: &str) -> bool {
    const SEED: u64 = 20261110;
    let mut hash = SEED ^ 0xcbf29ce484222325;
    for byte in qid.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash.is_multiple_of(5)
}

/// 只读语料，并校验它与数据集 manifest 的摘要一致。
///
/// 评测**不能**走 `load_dataset`：它会一并读 test qrels，而这条实验路径不碰 test。
fn load_corpus_only(dir: &Path) -> Result<Vec<sl_core::Doc>> {
    let corpus_path = dir.join("corpus.jsonl");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)
            .context("解析数据集 manifest 失败")?;
    let expected = manifest["files"]["corpus.jsonl"]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("manifest 缺 corpus.jsonl 摘要"))?;
    let bytes = std::fs::read(&corpus_path)
        .with_context(|| format!("读取 {} 失败", corpus_path.display()))?;
    let actual = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    };
    if actual != expected {
        bail!(
            "{} 的 sha256 与 manifest 不符（期望 {expected}，实际 {actual}）",
            corpus_path.display()
        );
    }
    let mut docs = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        docs.push(serde_json::from_slice(line).context("解析 corpus.jsonl 行失败")?);
    }
    Ok(docs)
}

/// 配对 bootstrap：返回 (delta, se, ci_low, ci_high)。
fn paired_bootstrap<F>(
    base: &[sl_core::PerQuery],
    arm: &[sl_core::PerQuery],
    metric: F,
) -> (f64, f64, f64, f64)
where
    F: Fn(&sl_core::PerQuery) -> f64,
{
    let by_qid: std::collections::HashMap<&str, &sl_core::PerQuery> =
        arm.iter().map(|row| (row.qid.as_str(), row)).collect();
    let deltas: Vec<f64> = base
        .iter()
        .filter_map(|row| {
            by_qid
                .get(row.qid.as_str())
                .map(|other| metric(other) - metric(row))
        })
        .collect();
    if deltas.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = deltas.len() as f64;
    let delta = deltas.iter().sum::<f64>() / n;
    let se = if deltas.len() < 2 {
        0.0
    } else {
        (deltas.iter().map(|x| (x - delta).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() / n.sqrt()
    };
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut samples = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sum = 0.0;
        for _ in 0..deltas.len() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            sum += deltas[(state >> 33) as usize % deltas.len()];
        }
        samples.push(sum / n);
    }
    samples.sort_by(f64::total_cmp);
    (delta, se, samples[50], samples[1949])
}

/// 评测划分。
///
/// `Heldout` / `Outer` 都在 train split 内部（判决 / 方向复核）；
/// `Test` 只用于**预注册过的一次性迁移检验**，会读 test qrels——见 `docs/PREREG-concept-transfer.md`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalSplit {
    Heldout,
    Outer,
    Test,
}

impl EvalSplit {
    fn from_arg(value: &str) -> Self {
        match value {
            "outer" => Self::Outer,
            "test" => Self::Test,
            _ => Self::Heldout,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Heldout => "heldout",
            Self::Outer => "outer_train",
            Self::Test => "test",
        }
    }
}

/// 评测驱动：给定臂清单，跑配对评测并落盘。
///
/// `field-eval` 与 `concept-eval` 只差在「臂怎么构造」——split、配对 bootstrap、落盘格式
/// 只在这里实现一份，否则两条路径的口径会各自漂移。
/// 一次因子评测的全部输入。
struct SweepRequest<'a> {
    dir: &'a Path,
    dataset_name: &'a str,
    k: usize,
    output: &'a Path,
    factor_record: serde_json::Value,
    split: EvalSplit,
    dump_per_query: bool,
    arms: Vec<(String, Box<dyn sl_core::Adapter>)>,
}

fn run_arm_sweep(req: SweepRequest<'_>) -> Result<()> {
    let SweepRequest {
        dir,
        dataset_name,
        k,
        output,
        factor_record,
        split,
        dump_per_query,
        arms,
    } = req;
    // Test 划分走完整数据集（含 test qrels），只该在预注册之后用一次。
    let (eval_queries, eval_qrels, docs, train_queries, eval_count_label) = match split {
        EvalSplit::Test => {
            let loaded = canonical::load_dataset(dir)?;
            let ds = loaded.dataset;
            let scored = ds.scored_query_ids().len();
            (ds.queries, ds.qrels, ds.docs, 0usize, scored)
        }
        EvalSplit::Heldout | EvalSplit::Outer => {
            let on_outer = split == EvalSplit::Outer;
            let eval_is: fn(&str) -> bool = if on_outer {
                |qid| !is_heldout_qid(qid)
            } else {
                is_heldout_qid
            };
            let train_is: fn(&str) -> bool = if on_outer {
                is_heldout_qid
            } else {
                |qid| !is_heldout_qid(qid)
            };
            let train = canonical::load_train_split_for(dir, train_is)?
                .ok_or_else(|| anyhow::anyhow!("{dataset_name} 没有 train split"))?;
            let eval = canonical::load_train_split_for(dir, eval_is)?
                .ok_or_else(|| anyhow::anyhow!("{dataset_name} 没有评测划分"))?;
            let eval_count = eval.queries.len();
            (
                eval.queries,
                eval.qrels,
                load_corpus_only(dir)?,
                train.queries.len(),
                eval_count,
            )
        }
    };

    let qrels = &eval_qrels;
    let mut summaries = serde_json::Map::new();
    let mut per_query_rows: std::collections::BTreeMap<String, Vec<sl_core::PerQuery>> =
        std::collections::BTreeMap::new();

    for (name, adapter) in arms {
        let mut adapter = adapter;
        adapter.index(&docs)?;
        let mut rankings = std::collections::BTreeMap::new();
        let mut elapsed = Vec::new();
        for query in &eval_queries {
            let started = std::time::Instant::now();
            let hits = adapter.search(&query.text, k)?;
            elapsed.push(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
            rankings.insert(
                query.id.clone(),
                hits.into_iter()
                    .map(|(doc, _)| doc)
                    .collect::<Vec<String>>(),
            );
        }
        let (metrics, per_query) = sl_core::metrics::evaluate_rankings(&rankings, qrels, k);
        elapsed.sort_unstable();
        let p50 = elapsed.get(elapsed.len() / 2).copied().unwrap_or(0);
        let p95 = elapsed
            .get((elapsed.len() as f64 * 0.95) as usize)
            .copied()
            .unwrap_or(0);
        summaries.insert(
            name.clone(),
            serde_json::json!({
                "recall@100": metrics.recall.get(&100).copied().unwrap_or(0.0),
                "ndcg@10": metrics.ndcg,
                "hit@10": metrics.hit.get(&10).copied().unwrap_or(0.0),
                "hit@100": metrics.hit.get(&100).copied().unwrap_or(0.0),
                "p50_us": p50,
                "p95_us": p95,
                "scored_queries": metrics.queries_scored,
            }),
        );
        per_query_rows.insert(name, per_query);
    }

    let baseline = per_query_rows.get("bm25").context("缺 bm25 基线")?.clone();
    let mut paired = serde_json::Map::new();
    for (name, rows) in &per_query_rows {
        if name == "bm25" {
            continue;
        }
        let (d_ndcg, se_ndcg, lo_ndcg, hi_ndcg) = paired_bootstrap(&baseline, rows, |row| row.ndcg);
        let (d_rec, se_rec, lo_rec, hi_rec) = paired_bootstrap(&baseline, rows, |row| {
            row.recall.get(&100).copied().unwrap_or(0.0)
        });
        // hit@10 = 首个相关文档排名 ≤ 10（等价于「top-10 里至少有一条相关」）。
        let (d_hit, se_hit, lo_hit, hi_hit) =
            paired_bootstrap(&baseline, rows, |row| match row.first_rel_rank {
                Some(rank) if rank <= 10 => 1.0,
                _ => 0.0,
            });
        paired.insert(
            name.clone(),
            serde_json::json!({
                "ndcg@10": {"delta": d_ndcg, "se": se_ndcg, "ci95": [lo_ndcg, hi_ndcg],
                             "delta_min_2se": 2.0 * se_ndcg},
                "recall@100": {"delta": d_rec, "se": se_rec, "ci95": [lo_rec, hi_rec],
                                "delta_min_2se": 2.0 * se_rec},
                "hit@10": {"delta": d_hit, "se": se_hit, "ci95": [lo_hit, hi_hit],
                            "delta_min_2se": 2.0 * se_hit},
            }),
        );
    }

    // 逐查询 nDCG 可选落盘：报告里只比「每臂 vs bm25」，但任意两臂之间的对比
    // （例如 concept+β vs uniform+β）需要原始配对值才能算置信区间。
    let per_query_dump: serde_json::Value = if dump_per_query {
        serde_json::Value::Object(
            per_query_rows
                .iter()
                .map(|(name, rows)| {
                    (
                        name.clone(),
                        serde_json::Value::Object(
                            rows.iter()
                                .map(|row| (row.qid.clone(), serde_json::json!(row.ndcg)))
                                .collect(),
                        ),
                    )
                })
                .collect(),
        )
    } else {
        serde_json::Value::Null
    };

    let report = serde_json::json!({
        "dataset": dataset_name,
        "k": k,
        "test_read": false,
        "split": {
            "evaluated_on": split.label(),
            "evaluated_queries": eval_count_label,
            "train_queries": train_queries,
        },
        "factors": factor_record,
        "arms": summaries,
        "paired_vs_bm25": paired,
        "per_query_ndcg": per_query_dump,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("创建 {} 失败", parent.display()))?;
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("写 {} 失败", output.display()))?;

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// 字段启发评测：`uniform`（纯 BM25F，不依赖因子）与 `factor`（逐 token 置信度）对照。
fn run_field_eval(
    data_dir: &Path,
    dataset_name: &str,
    k: usize,
    output: &Path,
    split: EvalSplit,
    _dump_per_query: bool,
) -> Result<()> {
    let dir = data_dir.join(dataset_name);
    let (factor, factor_sha) = sl_adapters::FieldBm25Adapter::load_factor(&dir)?;

    let mut arms: Vec<(String, Box<dyn sl_core::Adapter>)> =
        vec![("bm25".to_string(), sl_adapters::build("bm25")?)];
    for beta in [1.0_f64, 2.0, 5.0] {
        arms.push((
            format!("uniform_beta{beta}"),
            Box::new(sl_adapters::FieldBm25Adapter::uniform(beta)),
        ));
    }
    for beta in [1.0_f64, 2.0, 5.0] {
        arms.push((
            format!("factor_beta{beta}"),
            Box::new(sl_adapters::FieldBm25Adapter::with_factor(
                factor.clone(),
                beta,
                Some(factor_sha.clone()),
            )),
        ));
    }
    // 另一套 conf 定义（条件比例，量级 1e-1）：检验「量纲」是不是 per-token 因子失效的原因。
    if let Ok((cond_factor, cond_sha)) =
        sl_adapters::FieldBm25Adapter::load_factor_named(&dir, "field-v1")
    {
        for beta in [1.0_f64, 2.0, 5.0] {
            arms.push((
                format!("cond_beta{beta}"),
                Box::new(sl_adapters::FieldBm25Adapter::with_factor(
                    cond_factor.clone(),
                    beta,
                    Some(cond_sha.clone()),
                )),
            ));
        }
    }

    run_arm_sweep(SweepRequest {
        dir: &dir,
        dataset_name,
        k,
        output,
        factor_record: serde_json::json!({
            "id": "field-v0",
            "sha256": factor_sha,
            "tokens": factor.len(),
        }),
        split,
        dump_per_query: false,
        arms,
    })
}

/// 概念启发评测：内核学到的 `token → 规范形式` 因子，与字段加权正交组合。
///
/// 四条臂的差别只有两个开关（概念归并 / 字段加权），其余变量全同。
fn run_concept_eval(
    data_dir: &Path,
    dataset_name: &str,
    k: usize,
    output: &Path,
    split: EvalSplit,
    dump_per_query: bool,
    artifact_id: &str,
) -> Result<()> {
    let dir = data_dir.join(dataset_name);
    let (concept, concept_sha) =
        sl_adapters::FieldBm25Adapter::load_concept_named(&dir, artifact_id)?;
    let merged = concept.len();

    let make = |beta: f64| -> Box<dyn sl_core::Adapter> {
        let weight = if beta > 0.0 {
            sl_adapters::FieldWeight::Uniform(beta)
        } else {
            sl_adapters::FieldWeight::Off
        };
        Box::new(sl_adapters::FieldBm25Adapter::with_concept_and_weight(
            concept.clone(),
            Some(concept_sha.clone()),
            weight,
        ))
    };

    let mut arms: Vec<(String, Box<dyn sl_core::Adapter>)> =
        vec![("bm25".to_string(), sl_adapters::build("bm25")?)];
    for beta in [1.0_f64, 2.0] {
        arms.push((
            format!("uniform_beta{beta}"),
            Box::new(sl_adapters::FieldBm25Adapter::uniform(beta)),
        ));
    }
    arms.push(("concept".to_string(), make(0.0)));
    for beta in [1.0_f64, 2.0, 5.0] {
        arms.push((format!("concept_beta{beta}"), make(beta)));
    }

    run_arm_sweep(SweepRequest {
        dir: &dir,
        dataset_name,
        k,
        output,
        factor_record: serde_json::json!({
            "id": artifact_id,
            "kind": "concept",
            "sha256": concept_sha,
            "merged_tokens": merged,
        }),
        split,
        dump_per_query,
        arms,
    })
}
