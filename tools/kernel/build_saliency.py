#!/usr/bin/env python3
"""Token 先验显著性因子构建器 (saliency-v0)。

内核输出先验因子：token -> [["saliency", S(t)]]，其中 S(t) in [0.0, 1.0]。
外壳据此动态调制 BM25 的词频饱和度 k1 与下界偏置 delta。
严格无监督/仅用语料统计与 train split，不使用任何测试集 qrels。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "kernel"))
from tokenizer import bm25_terms  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACT_ID = "saliency-v0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compute_saliency(corpus_docs: list[dict]) -> dict[str, float]:
    n_docs = len(corpus_docs)
    df: dict[str, int] = {}
    tf_sum: dict[str, int] = {}
    tf_peak: dict[str, int] = {}

    for doc in corpus_docs:
        full = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        terms = bm25_terms(full)
        counts: dict[str, int] = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            df[t] = df.get(t, 0) + 1
            tf_sum[t] = tf_sum.get(t, 0) + c
            if c > tf_peak.get(t, 0):
                tf_peak[t] = c

    saliency: dict[str, float] = {}
    for t, doc_freq in df.items():
        if doc_freq <= 1:
            continue
        idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        concentration = tf_peak[t] / max(tf_sum[t], 1)
        # S(t) = 归一化到 (0, 1] 范围的判别力得分
        # 极高频泛化词（如 the/and/of 或论文常见词 study/results）得到较小的得分
        # 领域核心专有名词得到较高的得分
        raw = idf * (0.5 + concentration)
        norm = 1.0 / (1.0 + math.exp(-0.4 * (raw - 3.0)))
        saliency[t] = round(norm, 4)

    return saliency


def build(dataset: str, *, data_dir: Path) -> int:
    dataset_dir = data_dir / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料文件 {corpus_path}")

    docs = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    saliency_map = compute_saliency(docs)

    # 规范数据封装：token -> [["saliency", score]]
    entries = {
        term: [["saliency", score]]
        for term, score in saliency_map.items()
    }

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha256 = sha256_bytes(payload)
    (out_dir / "saliency.json").write_bytes(payload)

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "token_saliency",
        "weights_origin": "corpus_statistics",
        "source": f"data/{dataset}/corpus.jsonl",
        "prior_factor": {
            "name": "token_saliency",
            "kind": "token_confidence_list",
            "shape": ["vocab_size", 1],
            "dtype": "f32",
            "count": len(entries),
        },
        "network": {
            "arch": "none（无监督语料统计先验）",
            "params": 0,
            "checkpoint_sha256": None,
        },
        "counts": {
            "tokens": len(entries),
        },
        "files": {
            "saliency.json": {
                "sha256": payload_sha256,
                "bytes": len(payload),
            }
        },
        "runtime": {
            "device": "cpu",
            "precision": "f32",
        },
        "trained_on": "语料内统计（无监督，未使用任何 qrels）",
        "contamination": "无预训练模型参与，无外部语料污染",
        "license": "由本仓从数据集语料构建",
        "notes": "产出每个 Token 的先验判别力置信度 S(t) in [0, 1]，外壳用于自适应调节饱和与下界偏置。",
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OK {dataset}: {len(entries)} 个 Token 的 Saliency 先验因子已生成 → {out_dir}")
    print(f"   sha256: {payload_sha256[:16]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    args = parser.parse_args()
    raise SystemExit(build(args.dataset, data_dir=DATA_DIR))
