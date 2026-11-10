#!/usr/bin/env python3
"""把 BEIR 的 `qrels/train.tsv` 接成规范 train split（`data/<name>/` 下的两个可选文件）。

只处理有 train qrels 的数据集（当前只有 scifact：809 条训练查询）；其余数据集没有 train split，
本脚本会明确报「上游没有 train qrels」而不是造一个假的。

产出：
    data/<name>/train_queries.jsonl   每行 {"id","text"}（与 queries.jsonl 同 schema）
    data/<name>/train_qrels.tsv       首行 qid\\tdocid\\trel
    data/<name>/manifest.json         追加 train_files 摘要（**不进 fingerprint**，测试题没变）

`train_qrels.tsv` 只给训练工具读；`sl_core::Adapter` 接口里没有它，结构上无法被适配器拿去调参。

用法（在 tools/datasets/ 下）：
    uv run python build_train_split.py scifact
    uv run python build_train_split.py scifact --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

TRAIN_QUERIES = "train_queries.jsonl"
TRAIN_QRELS = "train_qrels.tsv"

# 只有这些数据集的上游带 train qrels（见 docs/DATASETS.md）。
TRAIN_SOURCES: dict[str, tuple[str, str]] = {
    # dataset -> (zip 文件名, zip 内 train qrels 成员)
    "scifact": ("scifact.zip", "scifact/qrels/train.tsv"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries[row["id"]] = row["text"]
    return queries


def read_train_qrels(zip_path: Path, member: str) -> dict[str, dict[str, int]]:
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member).decode("utf-8")
    qrels: dict[str, dict[str, int]] = {}
    for line in raw.splitlines():
        if not line.strip() or line.startswith("query-id"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"{member} 行格式不对：{line!r}")
        qid, docid, rel = parts
        qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def build(name: str, *, data_dir: Path, cache_dir: Path, force: bool = False) -> int:
    if name not in TRAIN_SOURCES:
        print(
            f"[跳过] {name}：上游没有 train qrels（只有 test）。"
            f"当前有 train split 的数据集：{', '.join(sorted(TRAIN_SOURCES))}",
            file=sys.stderr,
        )
        return 1
    dataset_dir = data_dir / name
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"找不到 {manifest_path}（先跑 build.py {name}）")

    queries_path = dataset_dir / "queries.jsonl"
    corpus_ids = {
        json.loads(line)["id"]
        for line in (dataset_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_queries = load_queries(queries_path)

    zip_name, member = TRAIN_SOURCES[name]
    zip_path = cache_dir / zip_name
    if not zip_path.is_file():
        raise SystemExit(f"找不到 {zip_path}（先跑 build.py {name} 下载缓存）")

    qrels_raw = read_train_qrels(zip_path, member)
    missing_queries = [qid for qid in qrels_raw if qid not in test_queries]
    missing_docs = sum(
        1 for qid, docs in qrels_raw.items() for doc in docs if doc not in corpus_ids
    )
    if missing_queries:
        raise SystemExit(
            f"{len(missing_queries)} 个训练查询不在 queries.jsonl 里（上游版本不一致？）"
        )
    if missing_docs:
        raise SystemExit(f"{missing_docs} 条训练 qrels 指向不存在的文档（上游版本不一致？）")

    train_qids = sorted(q for q in qrels_raw if any(rel > 0 for rel in qrels_raw[q].values()))
    if not train_qids:
        raise SystemExit("上游 train qrels 里没有任何正例")

    out_queries = dataset_dir / TRAIN_QUERIES
    out_qrels = dataset_dir / TRAIN_QRELS
    if out_queries.exists() and out_qrels.exists() and not force:
        print(f"[跳过] {name}：train split 已存在（--force 重建）", file=sys.stderr)
    else:
        with out_queries.open("w", encoding="utf-8") as fh:
            for qid in train_qids:
                fh.write(json.dumps({"id": qid, "text": test_queries[qid]}, ensure_ascii=False) + "\n")
        with out_qrels.open("w", encoding="utf-8") as fh:
            fh.write("qid\tdocid\trel\n")
            for qid in train_qids:
                for docid in sorted(qrels_raw[qid]):
                    fh.write(f"{qid}\t{docid}\t{qrels_raw[qid][docid]}\n")

    # manifest 里追加 train_files（不动 files / counts，因此 fingerprint 不变）
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_files"] = {
        name_file: {"sha256": sha256_file(dataset_dir / name_file), "bytes": (dataset_dir / name_file).stat().st_size}
        for name_file in (TRAIN_QUERIES, TRAIN_QRELS)
    }
    manifest.setdefault("notes", "")
    note = "train split 来自上游 train qrels，仅训练工具读取；不参与 manifest 指纹。"
    if note not in manifest["notes"]:
        manifest["notes"] = (manifest["notes"] + " " + note).strip()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    positive = sum(1 for q in qrels_raw for rel in qrels_raw[q].values() if rel > 0)
    print(
        f"OK {name}: train 查询 {len(train_qids)}（正例 {positive}）→ {out_queries.name} / {out_qrels.name}\n"
        f"   sha256 {TRAIN_QUERIES}={manifest['train_files'][TRAIN_QUERIES]['sha256'][:16]} "
        f"{TRAIN_QRELS}={manifest['train_files'][TRAIN_QRELS]['sha256'][:16]}",
        file=sys.stderr,
    )
    print(f"写回 {manifest_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把上游 train qrels 接成规范 train split")
    parser.add_argument("dataset", help="数据集名（当前只有 scifact 有 train qrels）")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--force", action="store_true", help="覆盖已有 train split")
    args = parser.parse_args(argv)
    return build(args.dataset, data_dir=args.data_dir, cache_dir=args.cache_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
