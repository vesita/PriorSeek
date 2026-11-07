"""规范数据集目录的写出与自检。

格式的唯一权威是 Rust 侧的 `crates/sl-core/src/canonical.rs`，这里逐字对齐：

- `corpus.jsonl`：每行 `{"id","title","text"}`；`title` 为空时省略该字段（与 Rust
  `Doc` 的 `skip_serializing_if = "String::is_empty"` 一致），按 doc id 升序；
- `queries.jsonl`：每行 `{"id","text"}`，按 query id 升序；
- `qrels.tsv`：首行 `qid\tdocid\trel`，此后每行三列、制表符分隔，按 (qid, docid) 升序；
- `manifest.json`：字段与 Rust `Manifest` 一致；`files` 只记录上面三个内容文件的
  sha256（小写十六进制）与字节数；`counts.scored_queries` 是 qrels 中至少有一个
  rel>0 的查询数。

构建器只负责把原始数据变成内存里的 docs / queries / qrels；排序、摘要、manifest 与
自检全部收口在 `write_dataset()`，这样四个数据集共用同一份落盘逻辑，相同输入必然得到
相同字节。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CORPUS_FILE = "corpus.jsonl"
QUERIES_FILE = "queries.jsonl"
QRELS_FILE = "qrels.tsv"
MANIFEST_FILE = "manifest.json"
CANONICAL_FILES = (CORPUS_FILE, QUERIES_FILE, QRELS_FILE)
QRELS_HEADER = "qid\tdocid\trel"


class BuildError(RuntimeError):
    """构建失败：CLI 只打印一句话，不打堆栈；测试可以直接断言。"""


def doc_row(doc_id: str, text: str, title: str = "") -> dict:
    """内存里的一篇文档。空 title 不写进 JSON，与 Rust 侧序列化一致。"""
    row = {"id": doc_id, "title": title, "text": text}
    if not title:
        del row["title"]
    return row


def query_row(query_id: str, text: str) -> dict:
    return {"id": query_id, "text": text}


def sha256_file(path: Path) -> str:
    """文件字节的 SHA-256（小写十六进制）。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_digest(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def manifest_fingerprint(manifest: Mapping) -> str:
    """复刻 Rust `Manifest::fingerprint()`，用于和 `sl-cli dataset verify` 的输出对照。

    Rust 侧按 64 位 usize 的小端字节写入 counts；这里同样按 8 字节小端编码。
    """
    hasher = hashlib.sha256()
    hasher.update(manifest["name"].encode())
    hasher.update(b"\x00")
    hasher.update(manifest["version"].encode())
    hasher.update(b"\x00")
    for file, digest in sorted(manifest["files"].items()):
        hasher.update(file.encode())
        hasher.update(b"\x00")
        hasher.update(digest["sha256"].encode())
        hasher.update(b"\x00")
    for key in ("docs", "queries", "qrels"):
        hasher.update(int(manifest["counts"][key]).to_bytes(8, "little"))
    return hasher.hexdigest()


def write_dataset(
    out_dir: Path,
    *,
    name: str,
    version: str,
    source: Sequence[str],
    built_by: str,
    notes: str,
    docs: Iterable[Mapping],
    queries: Iterable[Mapping],
    qrels: Iterable[tuple[str, str, int]],
    seed: int | None = None,
) -> dict:
    """把一份数据集落成规范目录（四个文件），返回写入的 manifest。

    `out_dir` 必须是一个空目录（构建器先写临时目录，自检通过后再整体挪到 `data/<name>`）；
    已存在的输出文件会被覆盖，这是给测试写临时目录用的。
    """
    if not name:
        raise BuildError("name 不能为空")
    if not version:
        raise BuildError(f"数据集 {name} 缺 version")
    if not source:
        raise BuildError(f"数据集 {name} 缺 source")

    doc_rows = [_check_doc(row, name) for row in docs]
    query_rows = [_check_query(row, name) for row in queries]
    doc_rows.sort(key=lambda row: row["id"])
    query_rows.sort(key=lambda row: row["id"])
    _check_unique([row["id"] for row in doc_rows], f"数据集 {name} 的 corpus id")
    _check_unique([row["id"] for row in query_rows], f"数据集 {name} 的 query id")

    doc_ids = {row["id"] for row in doc_rows}
    query_ids = {row["id"] for row in query_rows}
    qrel_rows = _normalize_qrels(qrels, doc_ids, query_ids, name)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / CORPUS_FILE, doc_rows)
    _write_jsonl(out_dir / QUERIES_FILE, query_rows)
    _write_qrels(out_dir / QRELS_FILE, qrel_rows)

    counts = {
        "docs": len(doc_rows),
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
        "scored_queries": len({qid for qid, _docid, rel in qrel_rows if rel > 0}),
    }
    files = {canonical_name: file_digest(out_dir / canonical_name) for canonical_name in sorted(CANONICAL_FILES)}

    manifest: dict = {
        "name": name,
        "version": version,
        "source": list(source),
        "built_by": built_by,
    }
    if seed is not None:
        manifest["seed"] = seed
    manifest["notes"] = notes
    manifest["counts"] = counts
    manifest["files"] = files

    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (out_dir / MANIFEST_FILE).write_text(text, encoding="utf-8", newline="\n")

    # 落盘后按 Rust 的角度重新读一遍，任何不一致都在构建当场炸掉，而不是等到 verify。
    actual = verify_dataset(out_dir)
    if actual != manifest:
        raise BuildError(f"数据集 {name} 落盘后的 manifest 与内存中的不一致（构建器 bug）")
    return manifest


def verify_dataset(out_dir: Path) -> dict:
    """按 Rust `load_dataset` + `verify_strict` 的口径自检一个规范目录，返回 manifest。

    检查项：manifest 可解析、files 覆盖三个内容文件且摘要/字节数一致、三个文件格式
    合法（JSONL 每行可解析、id 升序无重复、qrels 表头正确且按 (qid,docid) 升序）、
    qrels 不指向不存在的文档/查询、counts 与实际内容一致。
    """
    manifest_path = out_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise BuildError(f"{out_dir} 下没有 {MANIFEST_FILE}，不能当规范数据集使用")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildError(f"解析 {manifest_path} 失败：{exc}") from exc

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BuildError(f"{manifest_path} 的 files 不是对象")
    missing = [name for name in CANONICAL_FILES if name not in files]
    extra = [name for name in files if name not in CANONICAL_FILES]
    if missing or extra:
        raise BuildError(f"{manifest_path} 的 files 覆盖不对：缺 {missing}，多 {extra}")
    for name in CANONICAL_FILES:
        expected = files[name]
        if not isinstance(expected, dict) or "sha256" not in expected or "bytes" not in expected:
            raise BuildError(f"{manifest_path} 的 files.{name} 缺 sha256/bytes")
        actual = file_digest(out_dir / name)
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise BuildError(
                f"数据集 {manifest.get('name')} 的 {name} 与 manifest 摘要不一致"
                f"（manifest: {expected}，实际: {actual}）——规范数据被改过或构建器有 bug"
            )

    docs = read_jsonl(out_dir / CORPUS_FILE)
    queries = read_jsonl(out_dir / QUERIES_FILE)
    qrels = read_qrels(out_dir / QRELS_FILE)
    _check_sorted([row["id"] for row in docs], f"{out_dir / CORPUS_FILE} 的 doc id")
    _check_sorted([row["id"] for row in queries], f"{out_dir / QUERIES_FILE} 的 query id")
    _check_sorted([(qid, docid) for qid, docid, _rel in qrels], f"{out_dir / QRELS_FILE} 的 (qid, docid)")

    doc_ids = {row["id"] for row in docs}
    query_ids = {row["id"] for row in queries}
    for qid, docid, _rel in qrels:
        if qid not in query_ids:
            raise BuildError(f"{out_dir} 的 qrels 指向不存在的查询 {qid}（Rust 侧会丢弃并让 verify 失败）")
        if docid not in doc_ids:
            raise BuildError(f"{out_dir} 的 qrels 指向不存在的文档 {docid}（Rust 侧会丢弃并让 verify 失败）")

    counts = {
        "docs": len(docs),
        "queries": len(queries),
        "qrels": len(qrels),
        "scored_queries": len({qid for qid, _docid, rel in qrels if rel > 0}),
    }
    if manifest.get("counts") != counts:
        raise BuildError(f"{out_dir} 的 manifest.counts={manifest.get('counts')} 与实际规模 {counts} 不一致")
    return manifest


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL；空行视为错误（规范数据永远不会产生空行）。"""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="\n") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                raise BuildError(f"{path} 第 {lineno} 行是空行")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"{path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise BuildError(f"{path} 第 {lineno} 行不是 JSON 对象")
            rows.append(row)
    return rows


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    """读规范 qrels.tsv：首行必须是 qid/docid/rel 表头，rel 必须是整数。"""
    with open(path, "r", encoding="utf-8", newline="\n") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0] != QRELS_HEADER:
        raise BuildError(f"{path} 首行不是表头 {QRELS_HEADER!r}")
    rows: list[tuple[str, str, int]] = []
    for lineno, line in enumerate(lines[1:], 2):
        if not line.strip():
            raise BuildError(f"{path} 第 {lineno} 行是空行")
        parts = line.split("\t")
        if len(parts) != 3:
            raise BuildError(f"{path} 第 {lineno} 行不是三列：{line!r}")
        qid, docid, rel_text = parts
        try:
            rel = int(rel_text)
        except ValueError as exc:
            raise BuildError(f"{path} 第 {lineno} 行 rel 不是整数：{rel_text!r}") from exc
        if rel < 0:
            raise BuildError(f"{path} 第 {lineno} 行 rel 为负数：{rel}")
        rows.append((qid, docid, rel))
    return rows


def _check_doc(row: Mapping, name: str) -> dict:
    if not isinstance(row, dict):
        raise BuildError(f"数据集 {name} 的 corpus 行不是对象：{row!r}")
    doc_id = row.get("id")
    if not isinstance(doc_id, str) or not doc_id:
        raise BuildError(f"数据集 {name} 的 corpus 行缺 id：{row!r}")
    text = row.get("text")
    title = row.get("title", "")
    if not isinstance(text, str):
        raise BuildError(f"数据集 {name} 的文档 {doc_id} 缺 text（或不是字符串）")
    if title is None:
        title = ""
    if not isinstance(title, str):
        raise BuildError(f"数据集 {name} 的文档 {doc_id} 的 title 不是字符串")
    return doc_row(doc_id, text, title)


def _check_query(row: Mapping, name: str) -> dict:
    if not isinstance(row, dict):
        raise BuildError(f"数据集 {name} 的 queries 行不是对象：{row!r}")
    query_id = row.get("id")
    if not isinstance(query_id, str) or not query_id:
        raise BuildError(f"数据集 {name} 的 queries 行缺 id：{row!r}")
    text = row.get("text")
    if not isinstance(text, str):
        raise BuildError(f"数据集 {name} 的查询 {query_id} 缺 text（或不是字符串）")
    return query_row(query_id, text)


def _normalize_qrels(
    qrels: Iterable[tuple[str, str, int]],
    doc_ids: set[str],
    query_ids: set[str],
    name: str,
) -> list[tuple[str, str, int]]:
    merged: dict[tuple[str, str], int] = {}
    for item in qrels:
        try:
            qid, docid, rel = item
        except (TypeError, ValueError) as exc:
            raise BuildError(f"数据集 {name} 的 qrels 行不是 (qid, docid, rel)：{item!r}") from exc
        if not isinstance(qid, str) or not qid:
            raise BuildError(f"数据集 {name} 的 qrels 行 qid 非法：{item!r}")
        if not isinstance(docid, str) or not docid:
            raise BuildError(f"数据集 {name} 的 qrels 行 docid 非法：{item!r}")
        if not isinstance(rel, int) or isinstance(rel, bool) or rel < 0:
            raise BuildError(f"数据集 {name} 的 qrels 行 rel 必须是 >=0 的整数：{item!r}")
        if qid not in query_ids:
            raise BuildError(f"数据集 {name} 的 qrels 指向不存在的查询 {qid}（应先过滤）")
        if docid not in doc_ids:
            raise BuildError(f"数据集 {name} 的 qrels 指向不存在的文档 {docid}（应先过滤）")
        key = (qid, docid)
        if key in merged:
            raise BuildError(f"数据集 {name} 的 qrels 有重复配对 ({qid}, {docid})（应先聚合）")
        merged[key] = rel
    return sorted(
        ((qid, docid, rel) for (qid, docid), rel in merged.items()),
        key=lambda row: (row[0], row[1]),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_qrels(path: Path, rows: Iterable[tuple[str, str, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(QRELS_HEADER + "\n")
        for qid, docid, rel in rows:
            handle.write(f"{qid}\t{docid}\t{rel}\n")


def _check_unique(ids: list[str], label: str) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise BuildError(f"{label} 重复：{item}")
        seen.add(item)


def _check_sorted(keys: list, label: str) -> None:
    if keys != sorted(keys):
        raise BuildError(f"{label} 没有按升序排列（确定性要求）")
