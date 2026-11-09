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
        Command::Run { dataset, adapters, k, warmup, label, allow_dirty } => {
            run_one(&cli.data_dir, &cli.results_dir, &dataset, &adapters, k, warmup, label, allow_dirty)
        }
        Command::Report { run_dir } => rerender(&run_dir),
        Command::Probe { dataset, adapter, query, k, json } => {
            probe(&cli.data_dir, &dataset, &adapter, &query, k, json)
        }
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

fn verify_artifacts(run_dir: &Path) -> Result<()> {
    let path = run_dir.join("run.json");
    let text = fs::read_to_string(&path).with_context(|| format!("读 {} 失败", path.display()))?;
    let record: run::RunRecord =
        serde_json::from_str(&text).with_context(|| format!("解析 {} 失败", path.display()))?;

    let mut checked = 0usize;
    for adapter in &record.adapters {
        for artifact in &adapter.info.artifacts {
            let dir = Path::new(&artifact.dir);
            let current = sl_core::KernelArtifact::from_dir(dir)
                .with_context(|| format!("复核适配器 {} 的工件 {} 失败", adapter.info.name, artifact.id))?;
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
                bail!("适配器 {} 的工件 {} 内容文件哈希与运行记录不一致", adapter.info.name, artifact.id);
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
        println!("{} 不存在：还没有构建任何数据集（见 docs/DATASETS.md）", data_dir.display());
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
                eprintln!("[sl] 跳过 {}：不是 search-lab 规范数据集（{err:#}）", dir.display());
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
        format!(
            "加载数据集 {name} 失败；先构建：`uv run tools/datasets/build.py {name}`"
        )
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
            format!("数据集 {dataset_name} 未通过校验（调试可用 --allow-dirty 绕过，但结果不可入库）")
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

    let run_id = format!(
        "{}-{}",
        dataset.name,
        started_at.format("%Y%m%d-%H%M%S")
    );
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
        let path = out_dir.join("per_query").join(format!("{}.jsonl", outcome.info.name));
        let file = fs::File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
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
    let record: run::RunRecord = serde_json::from_str(&text)
        .with_context(|| format!("解析 {} 失败", path.display()))?;
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
