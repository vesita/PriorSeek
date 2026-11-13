#!/usr/bin/env python3
"""`concept` 因子的等价对照：真实归并的增益，能不能被「等结构的随机归并」复现？

动机：`docs/RESULTS.md` §18 里 `concept-v1` 是六个优化旋钮中唯一越过噪声带的
（heldout Δ nDCG@10 = +0.02487）。但 §19 的词性实验暴露出一个方法论缺口——
**任何在 df 结构上做 K 分箱并配 K 个自由度的操作，都能在 146 条查询上凑出 +0.01~+0.02**。
`concept` 从未做过同类对照，所以那个 +0.02487 里有多少来自「词形归并」、
多少来自「把 8506 个 token 按某种方式分组」本身，是未知的。

对照保留真实归并的**全部结构特征**，只打乱「谁和谁同组」：

  - 被归并的 token 集合 `S`（canon ∪ 被归并者）不变；
  - 组大小分布不变；
  - 规范形式仍取组内 df 最大者（与真实规则一致）；
  - 只有「哪个 token 落到哪个组」是随机的。

两个零假设：
  - `shuffle`：把 `S` 随机重新分组（多重 seeds ⇒ 得到零分布，可看真实值落在第几分位）；
  - `df_adjacent`：按 df 排序后取连续块分组——比随机更苛刻，因为 df 相邻的词未必是词形变体。

只读 corpus.jsonl + train split；不读 test qrels。

用法（在 tools/kernel/ 下）：
    uv run python null_concept.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_concept as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "data/scifact/artifacts/concept-v1/concepts.json"


def groups_of(mapping: dict[str, str]) -> dict[str, list[str]]:
    """从 `token → 规范形式` 还原分组（含规范形式自身）。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for token, canon in mapping.items():
        groups[canon].append(token)
    for canon in list(groups):
        groups[canon].append(canon)
    return dict(groups)


def mapping_from_groups(groups: list[list[str]], df: dict[str, int]) -> dict[str, str]:
    """组内 df 最大者为规范形式；并列时取字典序小者（与 `train_concept.make_mapping` 一致）。"""
    out: dict[str, str] = {}
    for members in groups:
        if len(members) < 2:
            continue
        canon = max(members, key=lambda t: (df.get(t, 0), [-ord(c) for c in t]))
        for m in members:
            if m != canon:
                out[m] = canon
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)

    docs, queries, qrels = T.load()
    corpus = T.Corpus(docs)
    ids = corpus.ids
    df = corpus.df
    outer = [q for q in sorted(qrels) if not T.is_heldout(q)]
    held = [q for q in sorted(qrels) if T.is_heldout(q)]

    real = json.loads(ARTIFACT.read_text(encoding="utf-8"))["entries"]
    real_mapping = {t: v[0][0] for t, v in real.items()}
    real_groups = groups_of(real_mapping)
    sizes = sorted((len(g) for g in real_groups.values()), reverse=True)
    members_all = [t for g in real_groups.values() for t in g]
    print(
        f"真实归并：{len(real_mapping)} 个 token → {len(real_groups)} 组，"
        f"覆盖 {len(members_all)} 个 token，组大小 {min(sizes)}..{max(sizes)}",
        file=sys.stderr,
    )

    base_ix = T.build_index(corpus, {})
    base_outer = T.evaluate(base_ix, ids, outer, queries, qrels)
    base_held = T.evaluate(base_ix, ids, held, queries, qrels)

    def delta_vs_bm25(mapping):
        ix = T.build_index(corpus, mapping)
        out = {}
        for name, qids, base in (("outer", outer, base_outer), ("heldout", held, base_held)):
            scores = T.evaluate(ix, ids, qids, queries, qrels)
            d = np.array([scores[q] - base[q] for q in qids])
            m, se, _, _ = T.bootstrap(d)
            out[name] = (m, 2 * se)
        return out

    real_delta = delta_vs_bm25(real_mapping)
    print(
        f"\n真实归并（concept-v1）：outer Δ={real_delta['outer'][0]:+.5f}  "
        f"heldout Δ={real_delta['heldout'][0]:+.5f}（2SE {real_delta['heldout'][1]:.5f}）",
        file=sys.stderr,
    )

    # --- 零假设 1：随机重新分组（保留组大小分布与被归并 token 集合）---
    rng = np.random.default_rng(20261113)
    shuffle_outer, shuffle_held = [], []
    pool = sorted(members_all)
    for seed in range(args.seeds):
        perm = rng.permutation(len(pool))
        shuffled = [pool[i] for i in perm]
        groups, cursor = [], 0
        for size in sizes:
            groups.append(shuffled[cursor : cursor + size])
            cursor += size
        mapping = mapping_from_groups(groups, df)
        d = delta_vs_bm25(mapping)
        shuffle_outer.append(d["outer"][0])
        shuffle_held.append(d["heldout"][0])
        print(f"  seed {seed:2d}: outer {d['outer'][0]:+.5f}  heldout {d['heldout'][0]:+.5f}", file=sys.stderr)

    # --- 零假设 2：按 df 相邻分组（同样的组大小序列）---
    ordered = sorted(pool, key=lambda t: (-df.get(t, 0), t))
    groups, cursor = [], 0
    for size in sizes:
        groups.append(ordered[cursor : cursor + size])
        cursor += size
    df_adj_delta = delta_vs_bm25(mapping_from_groups(groups, df))

    def report(label, values, real):
        arr = np.array(values)
        pct = float((arr < real).mean() * 100)
        print(
            f"  {label:14s} 均值 {arr.mean():+.5f}  SE {arr.std(ddof=1)/np.sqrt(len(arr)):.5f}  "
            f"区间 [{arr.min():+.5f}, {arr.max():+.5f}]  真实值分位 {pct:.0f}%"
        )

    print(f"\n{'零假设分布（各 %d 次）' % args.seeds}")
    report("outer", shuffle_outer, real_delta["outer"][0])
    report("heldout", shuffle_held, real_delta["heldout"][0])
    print(
        f"  df_adjacent    outer {df_adj_delta['outer'][0]:+.5f}  "
        f"heldout {df_adj_delta['heldout'][0]:+.5f}"
    )

    # 判定：真实值是否显著高于随机归并的零分布
    held_arr = np.array(shuffle_held)
    m, se, lo, hi = T.bootstrap_full(held_arr) if hasattr(T, "bootstrap_full") else (
        held_arr.mean(),
        held_arr.std(ddof=1) / np.sqrt(len(held_arr)),
        float(np.percentile(held_arr, 2.5)),
        float(np.percentile(held_arr, 97.5)),
    )
    print(
        f"\n随机归并 heldout 零分布：均值 {m:+.5f}，SE {se:.5f}，"
        f"95% [{lo:+.5f}, {hi:+.5f}]"
    )
    diff = real_delta["heldout"][0] - m
    print(
        f"真实 − 零分布均值 = {diff:+.5f}"
        f"  ⇒ {'真实值超出零分布' if real_delta['heldout'][0] > held_arr.max() else '真实值落在零分布内'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
