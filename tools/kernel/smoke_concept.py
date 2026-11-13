#!/usr/bin/env python3
"""干预非零性冒烟：概念归并到底改了多少查询的 top-10？

`docs/REFERENCE_NOTES.md` §2 的 R29：一个变体如果没真的改变被测量的量（这里是排序），
它的「效果」读数就不作数。本脚本回答一个问题——**9 成查询的排序真的变了，还是只有个别几条？**
`docs/RESULTS.md` §15.3b 的表格由它产出。

只读 corpus.jsonl + train split；不读 qrels（冒烟不需要相关性判定）。

用法（在 tools/kernel/ 下）：
    uv run python smoke_concept.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_concept as kernel  # noqa: E402


def main() -> int:
    docs, queries, qrels = kernel.load()
    corpus = kernel.Corpus(docs)

    learned = json.loads(kernel.WEIGHTS_PATH.read_text(encoding="utf-8"))
    scale = np.array(learned["feature_scale"], dtype=np.float64)
    cands = corpus.candidates()
    X = np.stack([corpus.pair_features(t, u) for t, u in cands])
    X = X / np.where(scale > 0.0, scale, 1.0)
    mapping = kernel.make_mapping(
        cands,
        X,
        np.array(learned["weights"], dtype=np.float64),
        float(learned["intercept"]),
        corpus.df,
    )

    plain = kernel.build_index(corpus, {})
    merged = kernel.build_index(corpus, mapping)

    for label, keep in (("heldout", kernel.is_heldout), ("outer", lambda q: not kernel.is_heldout(q))):
        qids = [q for q in sorted(qrels) if keep(q)]
        same = 0
        members_changed = 0
        for qid in qids:
            before = kernel.rank(plain, corpus.ids, queries[qid], 10)
            after = kernel.rank(merged, corpus.ids, queries[qid], 10)
            if before == after:
                same += 1
            elif set(before) != set(after):
                members_changed += 1
        n = len(qids)
        changed = n - same
        print(
            f"{label:8s} n={n:4d}  top-10 完全不变 {same:4d} ({same / n:.1%})  "
            f"顺序或成员变化 {changed:4d} ({changed / n:.1%})  "
            f"成员集合变化 {members_changed:4d} ({members_changed / n:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
