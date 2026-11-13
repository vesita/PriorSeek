#!/usr/bin/env python3
"""语义流匹配传输代价工件构建器 (flow-cost-v0)。

内核输出概念间的语义搬运代价矩阵 C(i, j) in [0.0, 1.0]。
数值越小，语义搬运代价越低（相同概念代价为 0.0，无关概念代价为 1.0）。
外壳据此计算查询意图流与文档论证流之间的最优传输（Wasserstein）成本。
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
ARTIFACT_ID = "flow-cost-v0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_flow_costs(dataset: str, *, data_dir: Path) -> int:
    dataset_dir = data_dir / dataset
    seq_dir = dataset_dir / "artifacts" / "seq-v0"
    if not (seq_dir / "manifest.json").is_file():
        raise SystemExit(f"找不到 {seq_dir}（先跑 build_seq.py {dataset}）")

    from build_seq import read_artifact
    seq = read_artifact(seq_dir)
    vocab = seq.vocab
    edges = seq.edges

    # 聚合每个词的最佳关联出边
    best_edges: dict[int, dict[int, float]] = {}
    for (src, dst, order), weight in edges.items():
        if weight <= 0.0:
            continue
        slot = best_edges.setdefault(src, {})
        if weight > slot.get(dst, 0.0):
            slot[dst] = weight

    # 构建距离代价条目：token -> [[neighbor, distance_cost]]
    entries: dict[str, list[list]] = {}
    total_pairs = 0

    for src, neighbors in best_edges.items():
        src_term = vocab[src]
        max_w = max(neighbors.values())
        if max_w <= 0.0:
            continue

        # 相似度 weight / max_w in (0, 1] 转化为距离代价 cost = 1.0 - similarity * 0.85
        # 即使最强相关词，代价也在 [0.15, 0.90] 之间，同词自身代价为 0.0
        ranked = sorted(
            ((dst, w) for dst, w in neighbors.items() if (w / max_w) >= 0.25),
            key=lambda row: (-row[1], row[0]),
        )[:10]

        if not ranked:
            continue

        entries[src_term] = [
            [vocab[dst], round(max(0.10, 1.0 - (w / max_w) * 0.85), 4)]
            for dst, w in ranked
        ]
        total_pairs += len(ranked)

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha256 = sha256_bytes(payload)
    (out_dir / "cost_graph.json").write_bytes(payload)

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "semantic_flow_cost",
        "weights_origin": "corpus_semantic_graph",
        "source": f"data/{dataset}/artifacts/seq-v0",
        "prior_factor": {
            "name": "semantic_transport_cost",
            "kind": "token_confidence_list",
            "shape": ["vocab_size", 10],
            "dtype": "f32",
            "count": total_pairs,
        },
        "network": {
            "arch": "optimal_transport_metric_projection",
            "params": 0,
            "checkpoint_sha256": None,
        },
        "counts": {
            "source_tokens": len(entries),
            "pairs": total_pairs,
        },
        "files": {
            "cost_graph.json": {
                "sha256": payload_sha256,
                "bytes": len(payload),
            }
        },
        "runtime": {
            "device": "cpu",
            "precision": "f32",
        },
        "trained_on": "语料内拓扑关联图谱（无监督，未使用任何 qrels）",
        "contamination": "无外部预训练污染",
        "license": "由本仓从数据集语料构建",
        "notes": "产出概念间语义流搬运代价 C(i, j) in [0, 1]，供最优传输外壳计算 Wasserstein 流匹配成本。",
    }

    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")

    print(f"OK {dataset}: {len(entries)} 个词项的语义流传输代价图已生成（共 {total_pairs} 条边）")
    print(f"   工件目录: {out_dir}")
    print(f"   cost_graph.json sha256: {payload_sha256[:16]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    args = parser.parse_args()
    raise SystemExit(build_flow_costs(args.dataset, data_dir=DATA_DIR))
