#!/usr/bin/env python3
"""从语料统计产出**先验因子**：`token → 关联 token + 置信度`（P0 档，零参数基线）。

这是内核与外壳的边界产物（`docs/KERNEL.md` §0）：内核（此处是统计，不是网络）产出
`{(token, confidence)}`，外壳把这些词和原词一起按权重送进检索。

为什么先做 P0：它**不需要训练**就能回答「token 扩展这个内核形态有没有信号」。
如果 P0 在「硬漏子集」上一点提升都没有，再谈 P1（用网络学置信度）就没意义。

扩展候选来源：`seq-v0` 工件里的 PPMI 边（order=1/2 的有序共现）。
置信度：按每个源 token 的最大出边权重归一 ⇒ `conf ∈ (0, 1]`。

产物（`data/<dataset>/artifacts/exp-ppmi-v0/`）：

    manifest.json     id / kind / source / params / counts / files(sha256) / runtime
    expansion.json    {"entries": {token: [[neighbor, conf], ...]}}

用法（在 tools/kernel/ 下）：

    uv run python build_expansion.py scifact
    uv run python build_expansion.py scifact --top-k 5 --min-weight 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from build_seq import read_artifact

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ARTIFACT_ID = "exp-ppmi-v0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build(
    dataset: str,
    *,
    data_dir: Path,
    top_k: int,
    min_weight: float,
    orders: tuple[int, ...],
) -> int:
    dataset_dir = data_dir / dataset
    seq_dir = dataset_dir / "artifacts" / "seq-v0"
    if not (seq_dir / "manifest.json").is_file():
        raise SystemExit(f"找不到 {seq_dir}（先跑 build_seq.py {dataset}）")

    seq = read_artifact(seq_dir)
    vocab = seq.vocab
    edges = seq.edges  # {(src, dst, order): weight}

    # 逐步聚合：同一 (src, dst) 在不同 order 上取最大权重（不做相加，避免重复计功）。
    best: dict[int, dict[int, float]] = {}
    for (src, dst, order), weight in edges.items():
        if order not in orders or weight <= 0.0:
            continue
        slot = best.setdefault(src, {})
        if weight > slot.get(dst, 0.0):
            slot[dst] = weight

    entries: dict[str, list[list]] = {}
    kept_terms = 0
    total_pairs = 0
    for src, neighbors in best.items():
        max_w = max(neighbors.values())
        if max_w <= 0.0:
            continue
        ranked = sorted(
            ((dst, w) for dst, w in neighbors.items() if w >= min_weight),
            key=lambda row: (-row[1], row[0]),
        )[:top_k]
        if not ranked:
            continue
        entries[vocab[src]] = [[vocab[dst], round(w / max_w, 6)] for dst, w in ranked]
        kept_terms += 1
        total_pairs += len(ranked)

    if not entries:
        raise SystemExit("没有任何扩展对生成（min_weight/top_k 太严？）")

    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expansion.json").write_bytes(payload)

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "query_expansion",
        "source": f"data/{dataset}/artifacts/seq-v0（PPMI 边，orders={list(orders)}）",
        "prior_factor": {
            "name": "query_expansion",
            "kind": "token_confidence_list",
            "shape": ["n_query_tokens", top_k],
            "dtype": "f32",
            "count": total_pairs,
        },
        "network": {"arch": "none（P0：统计直接产出，零参数基线）", "params": 0, "checkpoint_sha256": None},
        "params": {"top_k": top_k, "min_weight": min_weight, "orders": list(orders)},
        "counts": {"source_tokens": kept_terms, "pairs": total_pairs},
        "files": {"expansion.json": {"sha256": sha256_bytes(payload), "bytes": len(payload)}},
        "runtime": {"device": "cpu", "precision": "f32"},
        "trained_on": "语料内统计（PPMI，无监督，未使用 qrels）",
        "contamination": "无预训练模型参与，无外部语料",
        "license": "由本仓从数据集语料构建",
        "notes": "置信度 = 该源 token 的最大出边权重归一；外壳用 alpha 全局缩放后作为扩展词权重。",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"OK {dataset}: {kept_terms} 个源 token / {total_pairs} 条扩展对 → {out_dir}\n"
        f"   expansion.json {len(payload)} B sha256={manifest['files']['expansion.json']['sha256'][:16]}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 token 扩展先验因子（P0 统计基线）")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-weight", type=float, default=0.5)
    parser.add_argument("--orders", default="1,2")
    args = parser.parse_args(argv)
    orders = tuple(int(x) for x in args.orders.split(",") if x.strip())
    if not orders or any(o not in (1, 2) for o in orders):
        raise SystemExit(f"--orders 只接受 1 / 2 的组合，收到 {orders}")
    return build(
        args.dataset,
        data_dir=args.data_dir,
        top_k=args.top_k,
        min_weight=args.min_weight,
        orders=orders,
    )


if __name__ == "__main__":
    raise SystemExit(main())
