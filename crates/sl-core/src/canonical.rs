//! 规范数据目录的读写。
//!
//! 一个数据集在磁盘上长这样（`data/<name>/`）：
//!
//! ```text
//! corpus.jsonl   每行 {"id": "...", "title": "...", "text": "..."}
//! queries.jsonl  每行 {"id": "...", "text": "..."}
//! qrels.tsv      表头 qid\tdocid\trel（制表符分隔）
//! manifest.json  来源、构建参数、规模与三个文件的 sha256
//! ```
//!
//! 只有这里的格式是「规范格式」；原始数据（BEIR zip、HF parquet、业务库）都由
//! `tools/datasets/` 里的 Python 构建器转成它。适配器永远看不到原始格式。

use crate::types::{Counts, Dataset, Doc, FileDigest, Manifest, PruneReport, Qrels, Query};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

pub const CORPUS_FILE: &str = "corpus.jsonl";
pub const QUERIES_FILE: &str = "queries.jsonl";
pub const QRELS_FILE: &str = "qrels.tsv";
pub const MANIFEST_FILE: &str = "manifest.json";

/// 规范数据的三个内容文件（manifest 里记录它们各自的摘要）。
pub const CANONICAL_FILES: [&str; 3] = [CORPUS_FILE, QUERIES_FILE, QRELS_FILE];

pub fn file_digest(path: &Path) -> Result<FileDigest> {
    let bytes = fs::read(path).with_context(|| format!("读取 {} 失败", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(FileDigest { sha256: format!("{:x}", hasher.finalize()), bytes: bytes.len() as u64 })
}

/// 把一份内存里的 `Dataset` 落成规范目录，并写 manifest（摘要与规模按实际内容重算）。
///
/// `manifest` 里的 `counts` / `files` 会被覆盖成真实值，其余字段原样保留 —— 构建器只负责
/// 声明「这份数据从哪来、按什么口径裁的」。
pub fn write_dataset(dataset: &Dataset, dir: &Path, mut manifest: Manifest) -> Result<Manifest> {
    fs::create_dir_all(dir).with_context(|| format!("创建目录 {} 失败", dir.display()))?;

    write_jsonl(dir.join(CORPUS_FILE), dataset.docs.iter())?;
    write_jsonl(dir.join(QUERIES_FILE), dataset.queries.iter())?;
    write_qrels(dir.join(QRELS_FILE), &dataset.qrels)?;

    let mut files = BTreeMap::new();
    for name in CANONICAL_FILES {
        files.insert(name.to_string(), file_digest(&dir.join(name))?);
    }

    manifest.name = dataset.name.clone();
    let qrels: usize = dataset.qrels.values().map(|m| m.len()).sum();
    manifest.counts = Counts {
        docs: dataset.docs.len(),
        queries: dataset.queries.len(),
        qrels,
        scored_queries: dataset.scored_query_ids().len(),
    };
    manifest.files = files;

    let path = dir.join(MANIFEST_FILE);
    let text = serde_json::to_string_pretty(&manifest)?;
    fs::write(&path, text + "\n").with_context(|| format!("写 {} 失败", path.display()))?;
    Ok(manifest)
}

fn write_jsonl<T: serde::Serialize, I: IntoIterator<Item = T>>(path: PathBuf, rows: I) -> Result<()> {
    let file = File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
    let mut out = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut out, &row)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn write_qrels(path: PathBuf, qrels: &Qrels) -> Result<()> {
    let file = File::create(&path).with_context(|| format!("创建 {} 失败", path.display()))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "qid\tdocid\trel")?;
    for (qid, map) in qrels {
        for (docid, rel) in map {
            writeln!(out, "{qid}\t{docid}\t{rel}")?;
        }
    }
    out.flush()?;
    Ok(())
}

/// 读取规范目录，并做契约清理（丢掉指向不存在文档 / 不存在查询的 qrels）。
///
/// 清理数在 [`Loaded::pruned`] 里，调用方必须把它打出来 —— 静默丢数据会让两次 run
/// 看起来可比而实际不可比。
pub fn load_dataset(dir: &Path) -> Result<Loaded> {
    let manifest_path = dir.join(MANIFEST_FILE);
    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(&manifest_path)
            .with_context(|| format!("读 {} 失败（先跑 `sl dataset build`）", manifest_path.display()))?,
    )
    .with_context(|| format!("解析 {} 失败", manifest_path.display()))?;

    let docs: Vec<Doc> = read_jsonl(&dir.join(CORPUS_FILE))?;
    let queries: Vec<Query> = read_jsonl(&dir.join(QUERIES_FILE))?;
    let qrels = read_qrels(&dir.join(QRELS_FILE))?;

    let mut dataset = Dataset { name: manifest.name.clone(), docs, queries, qrels, manifest };
    let pruned = dataset.prune_inconsistent();
    Ok(Loaded { dataset, pruned, dir: dir.to_path_buf() })
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("打开 {} 失败", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line)
            .with_context(|| format!("{} 第 {} 行不是合法 JSONL", path.display(), lineno + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn read_qrels(path: &Path) -> Result<Qrels> {
    let file = File::open(path).with_context(|| format!("打开 {} 失败", path.display()))?;
    let reader = BufReader::new(file);
    let mut qrels: Qrels = BTreeMap::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let (Some(qid), Some(docid), Some(rel)) = (parts.next(), parts.next(), parts.next()) else {
            bail!("{} 第 {} 行不是 qid\\tdocid\\trel：{line}", path.display(), lineno + 1);
        };
        if lineno == 0 && qid == "qid" {
            continue;
        }
        let rel: u32 = rel
            .trim()
            .parse()
            .with_context(|| format!("{} 第 {} 行 rel 不是整数：{rel}", path.display(), lineno + 1))?;
        qrels.entry(qid.to_string()).or_default().insert(docid.to_string(), rel);
    }
    Ok(qrels)
}

/// `load_dataset` 的结果：数据集本体 + 契约清理报告 + 目录。
pub struct Loaded {
    pub dataset: Dataset,
    pub pruned: PruneReport,
    pub dir: PathBuf,
}

impl Loaded {
    /// 磁盘内容与 manifest 摘要不一致的文件名（空 = 一致）。
    pub fn digest_mismatches(&self) -> Result<Vec<String>> {
        let mut mismatched = Vec::new();
        for (name, expected) in &self.dataset.manifest.files {
            let actual = file_digest(&self.dir.join(name))?;
            if actual != *expected {
                mismatched.push(name.clone());
            }
        }
        Ok(mismatched)
    }

    /// 严格自检：摘要一致 + 没有 qrels 被丢弃 + 实际规模与 manifest 一致。
    /// 任一条不成立就报错，报错信息直接给出修法。
    pub fn verify_strict(&self) -> Result<()> {
        let mismatched = self.digest_mismatches()?;
        if !mismatched.is_empty() {
            bail!(
                "数据集 {} 的文件与 manifest 摘要不一致（{}）——规范数据被改过；重新跑 `uv run tools/datasets/build.py {}` 重建",
                self.dataset.name,
                mismatched.join(", "),
                self.dataset.name
            );
        }
        if !self.pruned.is_clean() {
            bail!(
                "数据集 {} 有 qrels 指向不存在的文档/查询（丢 {} 条 qrels、{} 个查询）——构建器有 bug，重建",
                self.dataset.name,
                self.pruned.dropped_qrels,
                self.pruned.dropped_queries
            );
        }
        let (counts, same) = self.dataset.recount();
        if !same {
            bail!(
                "数据集 {} 的实际规模与 manifest 不一致：实际 docs={} queries={} qrels={} scored={}",
                self.dataset.name,
                counts.docs,
                counts.queries,
                counts.qrels,
                counts.scored_queries
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset() -> Dataset {
        let docs = vec![
            Doc { id: "d1".into(), title: String::new(), text: "alpha beta".into() },
            Doc { id: "d2".into(), title: "标题".into(), text: "gamma delta".into() },
        ];
        let queries = vec![Query { id: "q1".into(), text: "alpha".into() }];
        let mut qrels: Qrels = BTreeMap::new();
        qrels.entry("q1".into()).or_default().insert("d1".into(), 1);
        Dataset {
            name: "fixture".into(),
            docs,
            queries,
            qrels,
            manifest: Manifest {
                name: "fixture".into(),
                version: "1".into(),
                source: vec!["fixture".into()],
                built_by: "unittest".into(),
                seed: None,
                notes: String::new(),
                counts: Counts::default(),
                files: BTreeMap::new(),
            },
        }
    }

    #[test]
    fn roundtrip_preserves_content_and_digests() {
        let tmp = tempfile::tempdir().unwrap();
        let dataset = sample_dataset();
        let written = write_dataset(&dataset, tmp.path(), dataset.manifest.clone()).unwrap();
        assert_eq!(written.counts.docs, 2);
        assert_eq!(written.counts.queries, 1);
        assert_eq!(written.counts.qrels, 1);
        assert_eq!(written.counts.scored_queries, 1);

        let loaded = load_dataset(tmp.path()).unwrap();
        assert!(loaded.pruned.is_clean());
        loaded.verify_strict().unwrap();
        assert_eq!(loaded.dataset.docs.len(), 2);
        assert_eq!(loaded.dataset.docs[1].title, "标题");
        assert_eq!(loaded.dataset.relevant("q1"), vec![("d1", 1)]);
        assert_eq!(loaded.dataset.manifest.fingerprint().len(), 64);
    }

    #[test]
    fn qrels_pointing_at_missing_docs_are_pruned_and_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let mut dataset = sample_dataset();
        dataset.qrels.entry("q1".into()).or_default().insert("nope".into(), 1);
        dataset.qrels.entry("q-missing".into()).or_default().insert("d1".into(), 1);
        write_dataset(&dataset, tmp.path(), dataset.manifest.clone()).unwrap();

        let loaded = load_dataset(tmp.path()).unwrap();
        assert_eq!(loaded.pruned.dropped_qrels, 1);
        assert_eq!(loaded.pruned.dropped_queries, 1);
        assert_eq!(loaded.dataset.relevant("q1"), vec![("d1", 1)]);
        assert!(loaded.verify_strict().is_err());
    }

    #[test]
    fn tampered_file_is_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let dataset = sample_dataset();
        write_dataset(&dataset, tmp.path(), dataset.manifest.clone()).unwrap();
        fs::write(tmp.path().join(CORPUS_FILE), "{\"id\":\"d1\",\"text\":\"tampered\"}\n").unwrap();

        let loaded = load_dataset(tmp.path()).unwrap();
        assert_eq!(loaded.digest_mismatches().unwrap(), vec![CORPUS_FILE.to_string()]);
        assert!(loaded.verify_strict().is_err());
    }

    #[test]
    fn fingerprint_ignores_prose_fields() {
        let mut a = sample_dataset().manifest;
        a.files.insert("corpus.jsonl".into(), FileDigest { sha256: "x".into(), bytes: 1 });
        a.counts.docs = 2;
        let mut b = a.clone();
        b.built_by = "another".into();
        b.notes = "different".into();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }
}
