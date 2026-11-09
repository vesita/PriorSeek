"""pytest 公共配置与 fixture：把 tools/kernel 加进 import path，并提供小数据集 / 小工件。"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_seq  # noqa: E402  （路径注入后才能 import）


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_dataset(
    root: Path,
    name: str = "tiny",
    docs: list[dict] | None = None,
    *,
    version: str = "v1",
) -> Path:
    """在 `root/<name>/` 写一份最小规范数据集（corpus + 空 queries/qrels + manifest）。

    `build_seq` 只读 corpus.jsonl，但会校验它的 sha256 与 manifest 一致，所以这里按
    与 Rust `Manifest::fingerprint()` 相同的口径手写 manifest。
    """
    if docs is None:
        docs = [
                {"id": "d1", "title": "标题一", "text": "alpha beta gamma alpha beta"},
            {"id": "d2", "text": "alpha beta delta"},
            {"id": "d3", "text": "beta gamma alpha"},
            {"id": "d4", "text": "alpha beta gamma"},
        ]
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    corpus_payload = "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs).encode()
    corpus_path = dataset_dir / "corpus.jsonl"
    corpus_path.write_bytes(corpus_payload)
    queries_payload = b""
    qrels_payload = "qid\tdocid\trel\n".encode()
    (dataset_dir / "queries.jsonl").write_bytes(queries_payload)
    (dataset_dir / "qrels.tsv").write_bytes(qrels_payload)

    manifest = {
        "name": name,
        "version": version,
        "source": ["fixture://tiny"],
        "built_by": "pytest",
        "notes": "单元测试用小数据集",
        "counts": {"docs": len(docs), "queries": 0, "qrels": 0, "scored_queries": 0},
        "files": {
            "corpus.jsonl": {"sha256": _sha256(corpus_payload), "bytes": len(corpus_payload)},
            "qrels.tsv": {"sha256": _sha256(qrels_payload), "bytes": len(qrels_payload)},
            "queries.jsonl": {"sha256": _sha256(queries_payload), "bytes": len(queries_payload)},
        },
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


def write_artifact(
    root: Path,
    *,
    vocab: list[str],
    idf: list[float],
    edges: list[tuple[int, int, int, float]],
    docs: list[tuple[str, list[int]]],
    manifest_extra: dict | None = None,
    write_manifest: bool = True,
) -> Path:
    """按 §8.2 手写一个 seq 工件目录（供 ref_score 手算用例），返回目录。"""
    artifact_dir = root / "seq-v0"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seq = bytearray(b"SLSEQv1\0")
    seq += struct.pack("<I", len(vocab))
    for token in vocab:
        payload = token.encode()
        seq += struct.pack("<H", len(payload)) + payload
    seq += struct.pack(f"<{len(idf)}f", *idf)
    seq += struct.pack("<I", len(edges))
    for src, dst, order, weight in edges:
        seq += struct.pack("<IIBf", src, dst, order, weight)
    (artifact_dir / "seq.bin").write_bytes(bytes(seq))

    doc_bytes = bytearray(b"SLDOCv1\0")
    doc_bytes += struct.pack("<I", len(docs))
    for _, tokens in docs:
        doc_bytes += struct.pack("<I", len(tokens))
        if tokens:
            doc_bytes += struct.pack(f"<{len(tokens)}I", *tokens)
    (artifact_dir / "doc_tokens.bin").write_bytes(bytes(doc_bytes))
    (artifact_dir / "docs.txt").write_text(
        "".join(f"{doc_id}\n" for doc_id, _ in docs), encoding="utf-8"
    )

    if write_manifest:
        files = {}
        for name in ("doc_tokens.bin", "docs.txt", "seq.bin"):
            payload = (artifact_dir / name).read_bytes()
            files[name] = {"sha256": _sha256(payload), "bytes": len(payload)}
        manifest = {
            "id": "seq-v0",
            "kind": "seq_graph",
            "method": "ppmi_topk",
            "weights_origin": "corpus_ppmi",
            "source_corpus": {"dataset": "fixture", "fingerprint": "f" * 64, "manifest_sha256": "f" * 64},
            "params": {
                "orders": [1, 2],
                "top_k": 20,
                "window": 8,
                "min_df": 2,
                "ppmi_shift": 5,
                "smoothing": 0.75,
            },
            "counts": {"vocab": len(vocab), "edges": len(edges)},
            "files": files,
            "runtime": {"device": "cpu", "precision": "f32"},
            "trained_on": "语料内统计（fixture）",
            "contamination": "无",
            "license": "test",
            "notes": "",
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return artifact_dir


def built_artifact(data_dir: Path, name: str = "tiny") -> Path:
    """跑一次真实构建并返回工件目录。"""
    summary = build_seq.build(name, data_dir=data_dir, force=True)
    return summary.out_dir
