//! `rs` 适配器：relation-search（Rust BM25 + ast-grep 代码检索）的子进程接入。
//!
//! 二进制解析顺序：环境变量 `RELATION_SEARCH_BIN` → `$HOME/coding/my/relation-search/target/release/relation-search`。
//! 不存在时构造阶段直接报错，不给「跑起来才发现 0 结果」的机会。
//!
//! # 块级 → 文档级
//!
//! **rs 返回的是块级候选**：relation-search 的索引单位是「代码块」，同一文件可能有多块
//! 同时命中。适配器契约要求返回**语料文档 id**，所以这里必须按 path 折叠到文档级
//! （同一 path 取最高分）；不折叠的话同一个 doc id 会重复出现，被 runner 记成
//! `duplicate_ids`。为了折叠后还能凑满 k 条文档级结果，子进程按
//! `k * BLOCK_FACTOR`（下限 [`MIN_BLOCK_LIMIT`]）取块级候选，再折叠截断。
//!
//! # 数据集怎么喂给它
//!
//! `index` 把每篇 `Doc` 写成临时目录下的一个 `<安全文件名>.txt`，再对该目录建索引；
//! 语料 doc id 可能含 `/`、空格、中文等，无法直接当文件名，因此文件名 = 清洗后的 id
//! （只留 `[A-Za-z0-9._-]`，其余替换为 `_`）+ `sha256(id)` 前 8 位（防清洗碰撞）。
//! 临时目录随适配器对象销毁，`search` 期间一直存在。
//!
//! 检索用 `--no-kind-weight --summary`：数据集是通用文本不是代码仓，不该吃
//! 「代码优先」的文件类型权重；`--summary` 省掉 snippet 生成，扫描更快。

use anyhow::{bail, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sl_core::{Adapter, AdapterInfo, Doc};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 块级候选的放大倍数：一个文件平均被切成多块，先多取再折叠。
pub const BLOCK_FACTOR: usize = 8;
/// 块级候选的下限：k 很小时也要给「折叠掉重复 path」留余量。
pub const MIN_BLOCK_LIMIT: usize = 50;

/// 文件名清洗后允许保留的字符集；其余一律替换成 `_`。
fn is_safe_file_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')
}

/// 语料 doc id → 临时目录里的文件名。
///
/// 清洗后主体最多保留 100 字节（文件名长度上限 255），末尾固定追加
/// `sha256(id)` 前 8 位十六进制，保证不同 doc id 不会静默落到同一个文件上。
pub fn safe_file_name(doc_id: &str) -> String {
    let mut stem: String = doc_id
        .chars()
        .map(|ch| if is_safe_file_char(ch) { ch } else { '_' })
        .collect();
    stem.truncate(100);
    if stem.is_empty() {
        stem.push_str("doc");
    }
    let digest = hex_sha256(doc_id.as_bytes());
    format!("{stem}.{}.txt", &digest[..8])
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// 解析二进制路径：环境变量优先，其次默认检出目录下的 release 构建。
fn resolve_bin() -> Result<PathBuf> {
    if let Some(value) = std::env::var_os("RELATION_SEARCH_BIN") {
        let path = PathBuf::from(value);
        if !path.is_file() {
            bail!(
                "RELATION_SEARCH_BIN 指向的 relation-search 二进制不存在：{}；\
                 构建：cargo build --release --manifest-path ~/coding/my/relation-search/Cargo.toml",
                path.display()
            );
        }
        return Ok(path);
    }
    let home =
        std::env::var_os("HOME").context("读环境变量 HOME 失败，无法定位 relation-search")?;
    let path = PathBuf::from(home).join("coding/my/relation-search/target/release/relation-search");
    if !path.is_file() {
        // `cargo build --release -C <dir>` 里的 `-C` 在 stable 上仍是 unstable，
        // 所以给 --manifest-path 这种真能跑通的写法。
        bail!(
            "找不到 relation-search 二进制：{}；\
             构建：cargo build --release --manifest-path ~/coding/my/relation-search/Cargo.toml，\
             或用环境变量 RELATION_SEARCH_BIN 指向已构建的二进制",
            path.display()
        );
    }
    Ok(path)
}

/// 二进制的可追溯标识：sha256 前 12 位 + relation-search 仓库 HEAD。
fn describe_bin(bin: &Path) -> (String, String) {
    let digest = std::fs::read(bin)
        .map(|bytes| hex_sha256(&bytes))
        .unwrap_or_default();
    let short: String = digest.chars().take(12).collect();
    let short = if short.is_empty() {
        "unknown".to_string()
    } else {
        short
    };
    let repo = bin
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let head = command_stdout(
        "git",
        &[
            "-C",
            &repo.to_string_lossy(),
            "rev-parse",
            "--short",
            "HEAD",
        ],
    )
    .unwrap_or_else(|| "unknown".to_string());
    (short, head)
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
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

/// 候选路径 → 语料 doc id。relation-search 给的是相对 root 的路径；本适配器的语料
/// 全部平铺在临时目录根下，因此先按整串匹配，再退到文件名匹配。
fn map_path<'a>(path: &str, doc_ids: &'a HashMap<String, String>) -> Option<&'a String> {
    if let Some(doc_id) = doc_ids.get(path) {
        return Some(doc_id);
    }
    let file_name = Path::new(path).file_name().and_then(|name| name.to_str())?;
    doc_ids.get(file_name)
}

/// 把块级候选折叠到文档级：同一 doc id 取最高分，按分数降序、doc id 升序，截断到 k。
pub fn fold_candidates(
    path_scores: &[(String, f64)],
    doc_ids: &HashMap<String, String>,
    k: usize,
) -> Result<Vec<(String, f64)>> {
    let mut best: BTreeMap<&str, f64> = BTreeMap::new();
    for (path, score) in path_scores {
        let doc_id = map_path(path, doc_ids)
            .ok_or_else(|| anyhow::anyhow!("候选路径 {path:?} 映射不回任何语料文档 id"))?;
        best.entry(doc_id.as_str())
            .and_modify(|current| {
                if *score > *current {
                    *current = *score;
                }
            })
            .or_insert(*score);
    }
    let mut folded: Vec<(String, f64)> = best
        .into_iter()
        .map(|(doc_id, score)| (doc_id.to_string(), score))
        .collect();
    folded.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    folded.truncate(k);
    Ok(folded)
}

pub struct RelationSearchAdapter {
    info: AdapterInfo,
    bin: PathBuf,
    /// 临时语料目录（`index` 时创建，随对象销毁）。
    workdir: Option<tempfile::TempDir>,
    /// 临时文件名 → 语料 doc id。
    doc_ids: HashMap<String, String>,
}

impl RelationSearchAdapter {
    pub fn new() -> Result<Self> {
        let bin = resolve_bin()?;
        let (digest, head) = describe_bin(&bin);
        let config = serde_json::json!({
            "block_factor": BLOCK_FACTOR,
            "min_block_limit": MIN_BLOCK_LIMIT,
            "block_level_candidates": true,
            "fold": "同一 path 取最高分，折叠到文档级",
            "search_flags": ["--no-kind-weight", "--summary"],
            "corpus_layout": "<临时目录>/<清洗后 id>.<sha256(id)[:8]>.txt",
        });
        Ok(Self {
            info: AdapterInfo::new(
                "rs",
                "relation-search",
                format!("{digest}+{head}"),
                format!(
                    "{}（Rust BM25 + 代码感知分词 + 内嵌 ast-grep 结构召回；子进程接入，非本仓代码）",
                    bin.display()
                ),
            )
            .with_config(config),
            bin,
            workdir: None,
            doc_ids: HashMap::new(),
        })
    }

    fn run(&self, args: &[&str]) -> Result<std::process::Output> {
        let output = Command::new(&self.bin)
            .args(args)
            .output()
            .with_context(|| format!("执行 {} 失败", self.bin.display()))?;
        if !output.status.success() {
            bail!(
                "relation-search {} 失败（退出码 {:?}）：{}",
                args.first().copied().unwrap_or(""),
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(output)
    }
}

impl Adapter for RelationSearchAdapter {
    fn info(&self) -> &AdapterInfo {
        &self.info
    }

    fn index(&mut self, docs: &[Doc]) -> anyhow::Result<()> {
        let dir = tempfile::Builder::new()
            .prefix("sl-rs-")
            .tempdir()
            .context("创建 rs 适配器的临时语料目录失败")?;
        let root = dir.path().to_path_buf();
        let mut doc_ids: HashMap<String, String> = HashMap::new();
        for doc in docs {
            let file_name = safe_file_name(&doc.id);
            if doc_ids.contains_key(&file_name) {
                // 文件名已带 sha256(id) 前 8 位，撞上说明有更异常的问题，宁可报错。
                bail!(
                    "rs 适配器：文档 id {:?} 与已有文档清洗到同一个文件名 {file_name}",
                    doc.id
                );
            }
            std::fs::write(root.join(&file_name), doc.full_text())
                .with_context(|| format!("写临时语料文件 {file_name} 失败"))?;
            doc_ids.insert(file_name, doc.id.clone());
        }

        self.workdir = Some(dir);
        self.doc_ids = doc_ids;

        let index_dir = root.join(".idx");
        let output = self.run(&[
            "index",
            "--root",
            &root.to_string_lossy(),
            "--index-dir",
            &index_dir.to_string_lossy(),
            "--rebuild",
        ])?;
        let parsed: Value = serde_json::from_slice(&output.stdout)
            .context("解析 relation-search index 输出失败（stdout 不是 JSON）")?;
        if parsed.get("ok").and_then(Value::as_bool) != Some(true) {
            bail!("relation-search index 没有报告 ok=true：{}", parsed);
        }
        Ok(())
    }

    fn search(&mut self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let workdir = self
            .workdir
            .as_ref()
            .context("rs 适配器还没建索引（先调 index）")?
            .path()
            .to_path_buf();
        let index_dir = workdir.join(".idx");
        // rs 返回块级候选：多取 BLOCK_FACTOR 倍，给「同一文件多块命中」留折叠余量。
        let block_limit = k.saturating_mul(BLOCK_FACTOR).max(MIN_BLOCK_LIMIT);
        let output = self.run(&[
            "search",
            "--root",
            &workdir.to_string_lossy(),
            "--index-dir",
            &index_dir.to_string_lossy(),
            "--query",
            query,
            "--limit",
            &block_limit.to_string(),
            "--no-kind-weight",
            "--summary",
        ])?;
        let parsed: Value = serde_json::from_slice(&output.stdout)
            .context("解析 relation-search search 输出失败（stdout 不是 JSON）")?;
        let candidates = parsed
            .get("candidates")
            .and_then(Value::as_array)
            .context("relation-search search 输出缺少 candidates 数组")?;
        let mut path_scores: Vec<(String, f64)> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let path = candidate
                .get("path")
                .and_then(Value::as_str)
                .context("候选缺少 path 字段")?;
            let score = candidate
                .get("score")
                .and_then(Value::as_f64)
                .context("候选缺少 score 字段")?;
            path_scores.push((path.to_string(), score));
        }
        fold_candidates(&path_scores, &self.doc_ids, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_file_name_cleans_and_disambiguates() {
        let a = safe_file_name("docs/a b.pdf");
        assert!(
            a.chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')),
            "{a}"
        );
        assert!(a.ends_with(".txt"));
        // 只差一个字符的两个 id 清洗后主体相同，但摘要不同 → 文件名不同。
        let b = safe_file_name("docs/a_b.pdf");
        assert_ne!(a, b);
        // 空 id 也有可用文件名。
        assert!(safe_file_name("").ends_with(".txt"));
    }

    #[test]
    fn fold_takes_max_score_per_doc_and_orders_by_score_then_id() {
        let doc_ids: HashMap<String, String> = [
            ("a.11111111.txt".to_string(), "a".to_string()),
            ("b.22222222.txt".to_string(), "b".to_string()),
            ("c.33333333.txt".to_string(), "c".to_string()),
        ]
        .into_iter()
        .collect();
        // a 有两块命中：取最高分 2.0（若误把块级分累加会得到 3.0 并抢到第一）。
        let path_scores = vec![
            ("a.11111111.txt".to_string(), 1.0),
            ("b.22222222.txt".to_string(), 5.0),
            ("a.11111111.txt".to_string(), 2.0),
            ("c.33333333.txt".to_string(), 5.0),
        ];
        let folded = fold_candidates(&path_scores, &doc_ids, 2).expect("折叠");
        // b 与 c 同分 5.0 → 按 doc id 升序，取前 2。
        assert_eq!(folded, vec![("b".to_string(), 5.0), ("c".to_string(), 5.0)]);

        let folded = fold_candidates(&path_scores, &doc_ids, 3).expect("折叠");
        assert_eq!(
            folded,
            vec![
                ("b".to_string(), 5.0),
                ("c".to_string(), 5.0),
                ("a".to_string(), 2.0)
            ]
        );
    }

    #[test]
    fn fold_rejects_paths_that_do_not_map_back() {
        let doc_ids: HashMap<String, String> = [("a.11111111.txt".to_string(), "a".to_string())]
            .into_iter()
            .collect();
        let err = fold_candidates(&[("ghost.txt".to_string(), 1.0)], &doc_ids, 5).unwrap_err();
        assert!(err.to_string().contains("映射不回"));
    }
}
