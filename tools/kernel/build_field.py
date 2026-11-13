#!/usr/bin/env python3
"""从语料的分字段统计产出**先验因子**：`token → 它在哪些字段出现过 + 出现率`。

因子表达的是证据，不是用法：这里只回答「这个 token 在标题里出现得多不多、在正文里出现得
多不多」，至于标题命中该加多少分，由外壳的算法超参决定（`docs/KERNEL.md` §4）。

与 `exp-ppmi-v0` 同构：同样是 `{"entries": {token: [[target, conf], ...]}}`，
`shape` 同样是 `[n_query_tokens, k]`，只是 target 从「关联词」换成「字段名」。
外壳因此只需一条查表路径，换 kind 不用换读取代码。

产物（`data/<dataset>/artifacts/field-v0/`）：

    manifest.json   id / kind / source / params / counts / files(sha256) / runtime / prior_factor
    fields.json     {"entries": {token: [["title", conf], ["text", conf]]}}

用法（在 tools/kernel/ 下）：

    uv run python build_field.py scifact
    uv run python build_field.py scifact --min-df 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizer import bm25_terms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARTIFACT_ID = "field-v0"
# 两种置信度定义：
#   df_ratio     -> df_field / N（文档比例，被 idf 基本覆盖，量级 1e-4）
#   conditional  -> df_field / Σ df（该 token 出现时落在该字段的条件比例，量级 1e-1）
CONF_MODES = ("df_ratio", "conditional")
# 字段顺序固定：同一份语料构建两遍必须字节一致，排序不能依赖 dict 插入顺序。
FIELDS = ("title", "text")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_corpus(dataset_dir: Path) -> tuple[dict, list[dict]]:
    """读语料并校验它没被手改。"""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"找不到数据集清单 {manifest_path}（先用 tools/datasets/build.py 构建）")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料 {corpus_path}")
    expected = (manifest.get("files") or {}).get("corpus.jsonl") or {}
    expected_sha = expected.get("sha256")
    if expected_sha:
        actual = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise SystemExit(
                f"{corpus_path} 的 sha256 与数据集 manifest 不符（期望 {expected_sha}，实际 {actual}）"
                " —— 语料被改过，先重建数据集"
            )

    docs = []
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    if not docs:
        raise SystemExit(f"{corpus_path} 是空的")
    return manifest, docs


def count_field_df(docs: list[dict], min_df: int) -> dict[str, dict[str, int]]:
    """`token → {字段: 出现文档数}`。同一文档内重复出现只算一次。"""
    counts: dict[str, dict[str, int]] = {}
    for doc in docs:
        for field in FIELDS:
            terms = set(bm25_terms(doc.get(field, "")))
            for term in terms:
                slot = counts.setdefault(term, {})
                slot[field] = slot.get(field, 0) + 1
    # 只在极少数文档里出现过的 token 不登记：它们对检索几乎没有可迁移的信号。
    return {
        token: per_field
        for token, per_field in counts.items()
        if sum(per_field.values()) >= min_df
    }


def build(
    dataset: str,
    *,
    data_dir: Path,
    min_df: int,
    conf_mode: str = "df_ratio",
    artifact_id: str = ARTIFACT_ID,
) -> int:
    dataset_dir = data_dir / dataset
    manifest_ds, docs = load_corpus(dataset_dir)
    total_docs = len(docs)

    counts = count_field_df(docs, min_df)
    if not counts:
        raise SystemExit("没有任何 token 满足 min_df（阈值太严？）")

    entries: dict[str, list[list]] = {}
    total_pairs = 0
    for token in sorted(counts):
        per_field = counts[token]
        total = sum(per_field.values())
        if conf_mode == "conditional":
            denom = total
        else:
            denom = total_docs
        ranked = [
            [field, round(per_field[field] / denom, 6)]
            for field in FIELDS
            if per_field.get(field)
        ]
        if not ranked:
            continue
        entries[token] = ranked
        total_pairs += len(ranked)

    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    out_dir = dataset_dir / "artifacts" / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fields.json").write_bytes(payload)

    manifest = {
        "id": artifact_id,
        "kind": "field_evidence",
        "weights_origin": "corpus_stats",
        "source": f"data/{dataset}/corpus.jsonl（按 {list(FIELDS)} 分字段统计）",
        "prior_factor": {
            "name": "token_field",
            "kind": "field",
            "shape": ["n_query_tokens", len(FIELDS)],
            "dtype": "f32",
            "count": total_pairs,
        },
        "network": {
            "arch": "none（统计直接产出，零参数基线）",
            "params": 0,
            "checkpoint_sha256": None,
        },
        "params": {"min_df": min_df, "fields": list(FIELDS), "conf_mode": conf_mode},
        "counts": {"tokens": len(entries), "pairs": total_pairs, "docs": total_docs},
        "files": {"fields.json": {"sha256": sha256_bytes(payload), "bytes": len(payload)}},
        "runtime": {"device": "cpu", "precision": "f32"},
        "trained_on": "语料内统计（分字段文档频率，无监督，未使用 qrels）",
        "contamination": "无预训练模型参与，无外部语料",
        "license": "由本仓从数据集语料构建",
        "notes": f"conf_mode={conf_mode}：df_ratio = df/N；conditional = df/Σdf。纯事实，不含用法。",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"OK {dataset}: {len(entries)} 个 token / {total_pairs} 条字段证据 → {out_dir}\n"
        f"   fields.json {len(payload)} B sha256={manifest['files']['fields.json']['sha256'][:16]}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 token 字段证据先验因子")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--conf-mode", choices=CONF_MODES, default="df_ratio")
    parser.add_argument("--artifact-id", default=ARTIFACT_ID)
    args = parser.parse_args(argv)
    return build(
        args.dataset,
        data_dir=args.data_dir,
        min_df=args.min_df,
        conf_mode=args.conf_mode,
        artifact_id=args.artifact_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
