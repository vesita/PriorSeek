"""构建流水线的离线测试：qrels 解析、子集选择、manifest 计数与 sha256、字节确定性。

所有测试都只用 `tests/fixtures/` 与本机临时目录，不发网络请求。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import build
import canonical
from registry import SPECS, DatasetSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_qrels_tsv_handles_beir_header_crlf_and_blank_lines():
    text = "query-id\tcorpus-id\tscore\r\nq1\td1\t1\r\n\r\nq2\td2\t0\r\n"
    assert build.parse_qrels_tsv(text, label="fixture") == [("q1", "d1", 1), ("q2", "d2", 0)]


def test_parse_qrels_tsv_rejects_malformed_rows():
    with pytest.raises(canonical.BuildError, match="三列"):
        build.parse_qrels_tsv("q1\td1\n", label="fixture")
    with pytest.raises(canonical.BuildError, match="rel 不是整数"):
        build.parse_qrels_tsv("q1\td1\tx\n", label="fixture")
    with pytest.raises(canonical.BuildError, match="负数"):
        build.parse_qrels_tsv("q1\td1\t-1\n", label="fixture")


def _beir_built() -> build.Built:
    base = FIXTURES / "beir_sample"
    corpus = _read_jsonl(base / "corpus.jsonl")
    queries = _read_jsonl(base / "queries.jsonl")
    qrels = build.parse_qrels_tsv((base / "qrels" / "test.tsv").read_text(encoding="utf-8"), label="test.tsv")
    return build.convert_beir(corpus, queries, qrels)


def test_convert_beir_drops_dangling_rows_and_reports():
    built = _beir_built()
    docs = {row["id"]: row for row in built.docs}
    assert set(docs) == {"d1", "d2", "d3", "d4"}
    assert docs["d1"]["title"] == "Alpha"
    assert "title" not in docs["d2"]  # 空 title 省略字段
    assert {row["id"] for row in built.queries} == {"q1", "q2", "q3", "q4"}
    assert sorted(built.qrels) == [("q1", "d1", 1), ("q1", "d2", 1), ("q2", "d3", 1), ("q3", "d1", 1)]
    assert built.dropped_qrels == 1  # missing-doc 那一行
    assert built.stats["missing_query_qrels"] == 1  # qX 不在 queries 里
    assert built.stats["duplicate_qrels"] == 1  # q3/d1 出现两次
    joined = " ".join(built.warnings)
    assert "不存在文档" in joined and "不在 queries" in joined and "重复配对" in joined
    # train.tsv 里的 (q1, d3) 不应出现（只读 test.tsv）
    assert ("q1", "d3", 1) not in built.qrels


def test_build_beir_reads_zip_member_paths(tmp_path):
    base = FIXTURES / "beir_sample"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with zipfile.ZipFile(cache_dir / "sample.zip", "w") as archive:
        for member in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "qrels/train.tsv"):
            archive.write(base / member, f"sample/{member}")
    spec = DatasetSpec(
        name="sample",
        version="v-fixture",
        kind="beir",
        source=("fixture://sample.zip",),
        archive="sample.zip",
        archive_url="http://unused.invalid/sample.zip",
    )
    built = build.build_beir(spec, cache_dir)
    assert len(built.docs) == 4
    assert len(built.qrels) == 4


def _qbqtc_rows() -> list[dict]:
    return json.loads((FIXTURES / "qbqtc_rows.json").read_text(encoding="utf-8"))


def test_text_id_is_stable_16_hex():
    assert build.text_id("同一个文本") == build.text_id("同一个文本")
    assert build.text_id("a") != build.text_id("b")
    value = build.text_id("任意文本")
    assert len(value) == 16
    assert all(char in "0123456789abcdef" for char in value)
    assert value == hashlib.sha256("任意文本".encode("utf-8")).hexdigest()[:16]


def test_convert_qbqtc_constructs_retrieval_task():
    built = build.convert_qbqtc(_qbqtc_rows())
    apple_q = build.text_id("苹果手机多少钱")
    rust_q = build.text_id("如何学习Rust")
    apple_doc = build.text_id("苹果手机价格")
    rust_doc = build.text_id("Rust 入门教程")
    android_doc = build.text_id("安卓手机价格")

    assert {row["id"] for row in built.docs} == {apple_doc, rust_doc, android_doc}
    docs = {row["id"]: row for row in built.docs}
    assert docs[apple_doc]["title"] == "苹果手机价格"
    assert docs[apple_doc]["text"] == ""  # title-only 语料
    assert {row["id"] for row in built.queries} == {apple_q, rust_q}
    assert sorted(built.qrels) == sorted([(apple_q, apple_doc, 2), (rust_q, rust_doc, 2)])
    assert built.stats == {
        "rows": 7,
        "unique_docs": 3,
        "unique_queries": 2,
        "scored_pairs": 2,
        "skipped_empty": 2,
    }
    joined = " ".join(built.warnings)
    assert "空文本" in joined and "重复配对" in joined


def test_cli_builds_qbqtc_offline_and_skips_existing(tmp_path, monkeypatch, capsys):
    rows = _qbqtc_rows()
    calls = {"fetch": 0}

    def fake_fetch(spec, cache_dir):
        calls["fetch"] += 1
        return rows

    monkeypatch.setattr(build, "fetch_qbqtc_rows", fake_fetch)
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"

    assert build.main(["qbqtc", "--data-dir", str(data_dir), "--cache-dir", str(cache_dir)]) == 0
    assert calls["fetch"] == 1
    out_dir = data_dir / "qbqtc"
    manifest = canonical.verify_dataset(out_dir)
    assert manifest["counts"] == {"docs": 3, "queries": 2, "qrels": 2, "scored_queries": 2}
    assert manifest["notes"] == SPECS["qbqtc"].notes
    assert "不是标准 IR 基准" in manifest["notes"]
    assert manifest["version"] == SPECS["qbqtc"].version
    assert sorted(path.name for path in data_dir.iterdir()) == ["qbqtc"]  # 没有残留临时目录
    capsys.readouterr()

    # 第二次运行：输出已存在且自检通过 → 跳过，不再拉数据
    assert build.main(["qbqtc", "--data-dir", str(data_dir), "--cache-dir", str(cache_dir)]) == 0
    assert "跳过 qbqtc" in capsys.readouterr().out
    assert calls["fetch"] == 1

    # --force 重建后四个文件字节完全一致（确定性）
    names = (*canonical.CANONICAL_FILES, canonical.MANIFEST_FILE)
    before = {name: (out_dir / name).read_bytes() for name in names}
    assert build.main(["qbqtc", "--data-dir", str(data_dir), "--cache-dir", str(cache_dir), "--force"]) == 0
    after = {name: (out_dir / name).read_bytes() for name in names}
    assert before == after
    assert calls["fetch"] == 2


def test_cli_rejects_unknown_dataset():
    with pytest.raises(SystemExit) as excinfo:
        build.main(["nope"])
    assert excinfo.value.code == 2


def _write_nq_fixture(tmp_path: Path, *, gz: bool) -> tuple[Path, Path, Path]:
    base = FIXTURES / "nq_sample"
    corpus_path = tmp_path / ("corpus.jsonl.gz" if gz else "corpus.jsonl")
    payload = (base / "corpus.jsonl").read_bytes()
    if gz:
        with gzip.open(corpus_path, "wb") as handle:
            handle.write(payload)
    else:
        corpus_path.write_bytes(payload)
    queries_path = tmp_path / "queries.jsonl"
    shutil.copyfile(base / "queries.jsonl", queries_path)
    qrels_path = tmp_path / "test.tsv"
    shutil.copyfile(base / "qrels" / "test.tsv", qrels_path)
    return corpus_path, queries_path, qrels_path


def _expected_pool(seed: int, pool_size: int = 4) -> set[str]:
    """测试侧独立复刻抽样规则：正例全保留，其余按 sha256(f"{id}:{seed}") 升序补足。"""
    positives = {"d3", "d5"}
    candidates = [f"d{i}" for i in range(8) if f"d{i}" not in positives]
    candidates.sort(key=lambda doc_id: int(hashlib.sha256(f"{doc_id}:{seed}".encode()).hexdigest(), 16))
    return positives | set(candidates[: pool_size - len(positives)])


@pytest.mark.parametrize("gz", [False, True])
def test_build_nq_pool_selection_is_deterministic(tmp_path, gz):
    corpus_path, queries_path, qrels_path = _write_nq_fixture(tmp_path, gz=gz)
    built = build.build_nq_pool(corpus_path, queries_path, qrels_path, pool_size=4, seed=7)

    assert {row["id"] for row in built.docs} == _expected_pool(7)
    assert built.stats["total_corpus_docs"] == 8
    assert built.stats["pool_docs"] == 4
    assert built.stats["pool_positives"] == 2
    assert built.stats["pool_fill"] == 2
    # 只有「至少一个正例文档在池内」的查询被保留：q2 的正例文档不存在，q3 的 d7 没被抽中
    assert {row["id"] for row in built.queries} == {"q0", "q1"}
    assert sorted(built.qrels) == [("q0", "d3", 1), ("q1", "d5", 1)]
    assert built.dropped_qrels == 1  # q2 的正例文档缺失-doc 不在 corpus 里
    joined = " ".join(built.warnings)
    assert "不在 corpus 里" in joined and "丢弃 1 行" in joined

    other = build.build_nq_pool(corpus_path, queries_path, qrels_path, pool_size=4, seed=8)
    assert {row["id"] for row in other.docs} == _expected_pool(8)
    assert {row["id"] for row in other.docs} != {row["id"] for row in built.docs}


def test_build_nq_pool_rejects_insufficient_corpus(tmp_path):
    corpus_path, queries_path, qrels_path = _write_nq_fixture(tmp_path, gz=False)
    with pytest.raises(canonical.BuildError, match="不够 pool_size"):
        build.build_nq_pool(corpus_path, queries_path, qrels_path, pool_size=9, seed=7)


def test_nq_pool_output_writes_verifiable_canonical_dir(tmp_path):
    corpus_path, queries_path, qrels_path = _write_nq_fixture(tmp_path, gz=False)
    built = build.build_nq_pool(corpus_path, queries_path, qrels_path, pool_size=4, seed=7)
    manifest = canonical.write_dataset(
        tmp_path / "nq-sub",
        name="nq-sub",
        version="v-fixture",
        source=("fixture://nq",),
        built_by="pytest",
        seed=7,
        notes="fixture",
        docs=built.docs,
        queries=built.queries,
        qrels=built.qrels,
    )
    assert manifest["counts"] == {"docs": 4, "queries": 2, "qrels": 2, "scored_queries": 2}
    assert canonical.verify_dataset(tmp_path / "nq-sub") == manifest
    assert manifest["files"]["corpus.jsonl"]["sha256"] == hashlib.sha256(
        (tmp_path / "nq-sub" / "corpus.jsonl").read_bytes()
    ).hexdigest()


class _FakeResponse:
    def __init__(self, status_code, *, payload=None, headers=None):
        self.status_code = status_code
        self.reason = "fake"
        self.headers = headers or {}
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise build.requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


def test_http_json_retries_429_then_succeeds(monkeypatch):
    responses = [
        _FakeResponse(429, headers={"retry-after": "0"}),
        _FakeResponse(200, payload={"ok": True}),
    ]
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(build.requests, "get", fake_get)
    monkeypatch.setattr(build.time, "sleep", lambda seconds: None)
    assert build._http_json("http://example.invalid/rows") == {"ok": True}
    assert len(calls) == 2


def test_fetch_qbqtc_rows_uses_page_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_http_json(url):
        calls["n"] += 1
        return {
            "num_rows_total": 5000,
            "rows": [
                {"row": {"sentence1": f"问题 {offset}", "sentence2": f"文档 {offset}", "score": 1}}
                for offset in range(100)
            ],
        }

    monkeypatch.setattr(build, "_http_json", fake_http_json)
    monkeypatch.setattr(build.time, "sleep", lambda seconds: None)
    spec = SPECS["qbqtc"]

    rows = build.fetch_qbqtc_rows(spec, tmp_path)
    assert len(rows) == spec.num_rows == 5000
    assert calls["n"] == 50
    assert rows[0] == {"sentence1": "问题 0", "sentence2": "文档 0", "score": 1}

    # 第二次全部命中页缓存（模拟上次构建被 429 打断后重跑的场景）
    again = build.fetch_qbqtc_rows(spec, tmp_path)
    assert again == rows
    assert calls["n"] == 50


def test_registry_covers_required_datasets_and_nq_subset_is_declared():
    assert set(SPECS) == {"scifact", "arguana", "nq-sub", "qbqtc"}
    for spec in SPECS.values():
        assert spec.source and spec.notes
    nq = SPECS["nq-sub"]
    assert nq.kind == "nq-sub"
    assert nq.pool_size == 50000
    assert nq.seed == 20261107
    assert "pool_size=50000" in nq.notes
    assert "不能与全量 NQ" in nq.notes
