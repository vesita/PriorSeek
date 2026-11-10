#!/usr/bin/env python3
"""在 scifact train split 上，**直接以「搜索缺失率」为目标**优化 `seq` 内核的 2 个权重。

为什么不用梯度
--------------
缺失率是阶跃函数：微调权重几乎从不改变「这条查询命中/未命中」，梯度处处为 0
（上一轮 pairwise logistic 已经踩过饱和的坑，见 `docs/PREREG-M2.md` §6.2）。
参数量只有 2，所以**不需要代理损失**：直接对目标指标做**精确网格搜索**（粗网格 + 局部细化），
优化的就是部署时真正要看的那个量，彻底绕开「代理 ≠ 指标」。

目标（`--objective`）
--------------------
- `hit`（默认）：最大化 train 上的**查询命中率** `hit@k`，等价于最小化搜索缺失率 `1 − hit@k`；
  并列时用 train nDCG@10 打平。
- `ndcg`：最大化 train nDCG@10（对照口径）。

打分与正式部署完全同一口径（`ref_score` 分量）：

    score(q,d) = s_uni(d) + a1·s1(d) + a2·s2(d)

训练器**只读** train 文件，从不打开测试集 `qrels.tsv`（`tests/test_train_seq_weights.py` 有路径探针）。

用法（在 tools/kernel/ 下）：

    uv run python optimize_seq_hit.py scifact --k 10
    uv run python optimize_seq_hit.py scifact --k 10 --out ../../data/scifact/artifacts/seq-v0/learned-hit10.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from train_seq_weights import (
    compute_components,
    load_train_split,
    sha256_file,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def ndcg_values(
    per_query: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], a1: float, a2: float, k: int = 10
) -> float:
    """在已缓存的每查询数组上算 nDCG@k（二值相关）。"""
    total = 0.0
    for uni, s1, s2, rel in per_query:
        scores = uni + a1 * s1 + a2 * s2
        order = np.argsort(-scores, kind="stable")[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, idx in enumerate(order) if rel[idx])
        n_rel = int(rel.sum())
        ideal = min(n_rel, k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        total += dcg / idcg if idcg else 0.0
    return total / len(per_query) if per_query else 0.0


def hit_rates(
    per_query: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grid: list[tuple[float, float]],
    k: int,
) -> np.ndarray:
    """对每个网格点算 hit@k（查询命中率）。全部向量化，不逐网格点重算排名。"""
    combined = np.zeros(len(grid), dtype=np.float64)
    for uni, s1, s2, rel in per_query:
        g = np.asarray(grid, dtype=np.float64)  # (G, 2)
        # scores[i, j] = 文档 i 在第 j 个网格点下的分数
        scores = uni[:, None] + s1[:, None] * g[:, 0][None, :] + s2[:, None] * g[:, 1][None, :]
        kk = min(k, scores.shape[0])
        top = np.argpartition(-scores, kk - 1, axis=0)[:kk, :]  # (kk, G)
        combined += rel[top].any(axis=0)
    return combined / len(per_query)


def make_grid(a1_values, a2_values) -> list[tuple[float, float]]:
    return [(a1, a2) for a1 in a1_values for a2 in a2_values]


def search(
    per_query: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    objective: str,
    k: int,
    coarse_step: float,
    fine_step: float,
    max_a: float,
) -> dict:
    """粗网格 → 局部细化。返回最优参数与网格统计。"""
    started = time.monotonic()

    def score_grid(grid: list[tuple[float, float]]) -> tuple[float, float, tuple[float, float]]:
        hits = hit_rates(per_query, grid, k)
        if objective == "hit":
            best_score = hits.max()
            # 并列时用 nDCG 打平（只在前若干名上算，省时间）
            candidates = [i for i, v in enumerate(hits) if v >= best_score - 1e-12]
            best = max(candidates, key=lambda i: ndcg_values(per_query, *grid[i], k=10))
        else:
            ndcgs = np.array([ndcg_values(per_query, *g, k=10) for g in grid])
            best = int(ndcgs.argmax())
            best_score = float(ndcgs[best])
            hits = hits  # 只用于报告
        return float(best_score), float(hits[best]), grid[best]

    coarse = make_grid(np.arange(0.0, max_a + 1e-9, coarse_step), np.arange(0.0, max_a + 1e-9, coarse_step))
    best_score, best_hit, (a1, a2) = score_grid(coarse)
    coarse_seconds = time.monotonic() - started

    fine = make_grid(
        np.clip(np.arange(a1 - 3 * coarse_step, a1 + 3 * coarse_step + 1e-9, fine_step), 0.0, None),
        np.clip(np.arange(a2 - 3 * coarse_step, a2 + 3 * coarse_step + 1e-9, fine_step), 0.0, None),
    )
    fine_started = time.monotonic()
    best_score_f, best_hit_f, (a1_f, a2_f) = score_grid(fine)
    fine_seconds = time.monotonic() - started - coarse_seconds

    if best_score_f > best_score + 1e-12:
        best_score, best_hit, (a1, a2) = best_score_f, best_hit_f, (a1_f, a2_f)

    return {
        "a1": float(a1),
        "a2": float(a2),
        "objective": objective,
        "hit_k": k,
        "best_hit": float(best_hit),
        "best_ndcg10": float(ndcg_values(per_query, a1, a2, k=10)),
        "grid_points_coarse": len(coarse),
        "grid_points_fine": len(fine),
        "coarse_seconds": coarse_seconds,
        "fine_seconds": fine_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="以搜索缺失率为目标，精确网格搜索 seq 的 2 个权重")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--objective", choices=("hit", "ndcg"), default="hit")
    parser.add_argument("--k", type=int, default=10, help="hit@k 里的 k（默认 10 = 用户看到的第一屏）")
    parser.add_argument("--coarse-step", type=float, default=0.1)
    parser.add_argument("--fine-step", type=float, default=0.02)
    parser.add_argument("--max-a", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset_dir = args.data_dir / args.dataset
    artifact_dir = args.artifact or (dataset_dir / "artifacts" / "seq-v0")
    if not (artifact_dir / "manifest.json").is_file():
        raise SystemExit(f"找不到工件 {artifact_dir}（先跑 build_seq.py {args.dataset}）")

    queries, positives = load_train_split(dataset_dir)  # 只读 train（N3）
    components, comp_seconds = compute_components(artifact_dir, queries)

    # 预先摊平成 numpy 数组（每查询：uni / s1 / s2 / 相关掩码）
    per_query = []
    for qid, pos_docs in positives.items():
        if qid not in components:
            continue
        uni_d, s1_d, s2_d = components[qid]
        docs = list(uni_d.keys())
        if not docs:
            continue
        rel = np.array([d in set(pos_docs) for d in docs], dtype=bool)
        if not rel.any():
            continue
        per_query.append(
            (
                np.array([uni_d[d] for d in docs], dtype=np.float64),
                np.array([s1_d.get(d, 0.0) for d in docs], dtype=np.float64),
                np.array([s2_d.get(d, 0.0) for d in docs], dtype=np.float64),
                rel,
            )
        )
    print(f"train: {len(per_query)} 条计分查询（分量计算 {comp_seconds:.1f}s）", file=sys.stderr)

    baseline_hit = float(hit_rates(per_query, [(0.0, 0.0)], args.k)[0])
    baseline_ndcg = ndcg_values(per_query, 0.0, 0.0)
    print(f"基线（a=0，等价 bm25 打分）：hit@{args.k} = {baseline_hit:.4f}  nDCG@10 = {baseline_ndcg:.4f}", file=sys.stderr)

    best = search(
        per_query,
        objective=args.objective,
        k=args.k,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        max_a=args.max_a,
    )

    result = {
        "dataset": args.dataset,
        "kernel": "seq-v0",
        "objective": args.objective,
        "learned": {"a1": best["a1"], "a2": best["a2"]},
        "mapped_to_adapter": {"lambda_seq": 1.0, "w_order_1": best["a1"], "w_order_2": best["a2"]},
        "baseline": {"hit": baseline_hit, "ndcg10": baseline_ndcg},
        "best": {"hit": best["best_hit"], "ndcg10": best["best_ndcg10"]},
        "delta_hit": best["best_hit"] - baseline_hit,
        "delta_ndcg10": best["best_ndcg10"] - baseline_ndcg,
        "grid": {
            "coarse_points": best["grid_points_coarse"],
            "fine_points": best["grid_points_fine"],
            "coarse_seconds": best["coarse_seconds"],
            "fine_seconds": best["fine_seconds"],
        },
        "artifact_manifest_sha256": sha256_file(artifact_dir / "manifest.json"),
        "train_queries_sha256": sha256_file(dataset_dir / "train_queries.jsonl"),
        "train_qrels_sha256": sha256_file(dataset_dir / "train_qrels.tsv"),
        "note": "目标 = train 上的搜索缺失率（1 - hit@k）；只读 train split，测试集 qrels.tsv 从未被打开。",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"写入 {args.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
