"""规范目录写出 / 自检的离线测试（不碰网络）。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import canonical


def _dataset():
    """固定的小数据集：输入顺序故意乱掉，由 write_dataset 负责排序。"""
    docs = [
        canonical.doc_row("d2", "beta", ""),
        canonical.doc_row("d1", "alpha", "标题"),
        canonical.doc_row("d10", "gamma", ""),
    ]
    queries = [
        canonical.query_row("q10", "beta"),
        canonical.query_row("q1", "alpha"),
    ]
    qrels = [("q10", "d2", 2), ("q1", "d2", 0), ("q1", "d1", 1)]
    return docs, queries, qrels


def _write(out_dir: Path, *, docs=None, queries=None, qrels=None, seed=42):
    default_docs, default_queries, default_qrels = _dataset()
    return canonical.write_dataset(
        out_dir,
        name="unit",
        version="v1",
        source=("fixture://unit",),
        built_by="pytest",
        seed=seed,
        notes="单元测试用数据集",
        docs=default_docs if docs is None else docs,
        queries=default_queries if queries is None else queries,
        qrels=default_qrels if qrels is None else qrels,
    )


def test_write_dataset_manifest_counts_and_digests(tmp_path):
    manifest = _write(tmp_path / "unit")

    assert manifest["name"] == "unit"
    assert manifest["seed"] == 42
    assert manifest["counts"] == {"docs": 3, "queries": 2, "qrels": 3, "scored_queries": 2}
    assert list(manifest["files"]) == ["corpus.jsonl", "qrels.tsv", "queries.jsonl"]
    for name, digest in manifest["files"].items():
        payload = (tmp_path / "unit" / name).read_bytes()
        assert digest["sha256"] == hashlib.sha256(payload).hexdigest()
        assert digest["bytes"] == len(payload)

    # 落盘内容按 id 升序；空 title 不写字段（与 Rust 序列化一致）。
    corpus_lines = (tmp_path / "unit" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in corpus_lines] == ["d1", "d10", "d2"]
    assert json.loads(corpus_lines[0]) == {"id": "d1", "title": "标题", "text": "alpha"}
    assert "title" not in json.loads(corpus_lines[1])

    query_lines = (tmp_path / "unit" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in query_lines] == ["q1", "q10"]

    qrels_text = (tmp_path / "unit" / "qrels.tsv").read_text(encoding="utf-8")
    assert qrels_text == "qid\tdocid\trel\nq1\td1\t1\nq1\td2\t0\nq10\td2\t2\n"

    # 自检能重算出同一份 manifest，且文件以换行结尾。
    assert canonical.verify_dataset(tmp_path / "unit") == manifest
    assert (tmp_path / "unit" / "manifest.json").read_text(encoding="utf-8").endswith("\n")
    for name in canonical.CANONICAL_FILES:
        assert (tmp_path / "unit" / name).read_text(encoding="utf-8").endswith("\n")


def test_absent_seed_is_omitted_from_manifest(tmp_path):
    manifest = _write(tmp_path / "unit", seed=None)
    assert "seed" not in manifest


def test_output_bytes_are_deterministic_and_order_independent(tmp_path):
    docs, queries, qrels = _dataset()
    first = _write(tmp_path / "a", docs=docs, queries=queries, qrels=qrels)
    second = _write(
        tmp_path / "b",
        docs=list(reversed(docs)),
        queries=list(reversed(queries)),
        qrels=list(reversed(qrels)),
    )
    assert first == second
    for name in (*canonical.CANONICAL_FILES, canonical.MANIFEST_FILE):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_verify_detects_tampered_corpus(tmp_path):
    _write(tmp_path / "unit")
    corpus_path = tmp_path / "unit" / "corpus.jsonl"
    corpus_path.write_bytes(corpus_path.read_bytes().replace(b"alpha", b"omega"))
    with pytest.raises(canonical.BuildError, match="摘要不一致"):
        canonical.verify_dataset(tmp_path / "unit")


def test_verify_detects_wrong_counts(tmp_path):
    _write(tmp_path / "unit")
    manifest_path = tmp_path / "unit" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["docs"] = 999
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(canonical.BuildError, match="counts"):
        canonical.verify_dataset(tmp_path / "unit")


def test_write_rejects_dangling_qrels(tmp_path):
    docs, queries, _qrels = _dataset()
    with pytest.raises(canonical.BuildError, match="不存在的文档"):
        _write(tmp_path / "unit", docs=docs, queries=queries, qrels=[("q1", "nope", 1)])
    with pytest.raises(canonical.BuildError, match="不存在的查询"):
        _write(tmp_path / "unit", docs=docs, queries=queries, qrels=[("q9", "d1", 1)])


def test_write_rejects_duplicate_pairs_and_ids(tmp_path):
    docs, queries, _qrels = _dataset()
    with pytest.raises(canonical.BuildError, match="重复配对"):
        _write(tmp_path / "unit", docs=docs, queries=queries, qrels=[("q1", "d1", 1), ("q1", "d1", 2)])
    duplicated = docs + [canonical.doc_row("d1", "dup", "")]
    with pytest.raises(canonical.BuildError, match="重复"):
        _write(tmp_path / "unit", docs=duplicated, queries=queries, qrels=[("q1", "d1", 1)])


def test_manifest_fingerprint_matches_rust_recipe():
    manifest = {
        "name": "unit",
        "version": "v1",
        "built_by": "pytest",
        "notes": "prose",
        "counts": {"docs": 3, "queries": 2, "qrels": 3, "scored_queries": 2},
        "files": {
            "corpus.jsonl": {"sha256": "aa", "bytes": 1},
            "qrels.tsv": {"sha256": "bb", "bytes": 2},
            "queries.jsonl": {"sha256": "cc", "bytes": 3},
        },
    }
    # 独立复刻 Rust Manifest::fingerprint()：name/version/文件摘要按 BTreeMap 升序，
    # counts 用 64 位小端；built_by / notes 不参与。
    hasher = hashlib.sha256()
    hasher.update(b"unit")
    hasher.update(b"\x00")
    hasher.update(b"v1")
    hasher.update(b"\x00")
    for name in ("corpus.jsonl", "qrels.tsv", "queries.jsonl"):
        hasher.update(name.encode())
        hasher.update(b"\x00")
        hasher.update(manifest["files"][name]["sha256"].encode())
        hasher.update(b"\x00")
    for count in (3, 2, 3):
        hasher.update(count.to_bytes(8, "little"))
    assert canonical.manifest_fingerprint(manifest) == hasher.hexdigest()

    prose_changed = copy.deepcopy(manifest)
    prose_changed["built_by"] = "someone-else"
    prose_changed["notes"] = "另一段说明"
    assert canonical.manifest_fingerprint(prose_changed) == canonical.manifest_fingerprint(manifest)

    counts_changed = copy.deepcopy(manifest)
    counts_changed["counts"]["docs"] = 4
    assert canonical.manifest_fingerprint(counts_changed) != canonical.manifest_fingerprint(manifest)


def test_read_jsonl_and_qrels_are_strict(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text('{"id": "d1", "text": "ok"}\n\n', encoding="utf-8")
    with pytest.raises(canonical.BuildError, match="空行"):
        canonical.read_jsonl(corpus_path)

    qrels_path = tmp_path / "qrels.tsv"
    qrels_path.write_text("docid\tqid\trel\n", encoding="utf-8")
    with pytest.raises(canonical.BuildError, match="表头"):
        canonical.read_qrels(qrels_path)

    qrels_path.write_text("qid\tdocid\trel\nq1\td1\tx\n", encoding="utf-8")
    with pytest.raises(canonical.BuildError, match="rel 不是整数"):
        canonical.read_qrels(qrels_path)
