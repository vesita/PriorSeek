#!/usr/bin/env python3
"""词性的一种**结构性**用法：只对「构成短语」的查询相邻对给相邻性加分。

为什么换用法：`docs/RESULTS.md` §19 已证伪「每类一个乘子」（`w(t)=Σ_k p(k|t)·λ_k`）——
在 df 分层置换对照下为零。但词性还有一类未被检验的用法：**结构匹配**。

假设：`seq` / `oac` 的失败（§8 / §12）是因为它们对**所有**查询相邻对都给奖励，
其中大量是「名词 + 介词」这种跨短语的噪声对。若只用词性筛出**构成短语**的对
（如 ADJ+NOUN、NOUN+NOUN），相邻性奖励也许能变成正信号。

四条臂（唯一差别是「选哪些查询对」；γ 在 outer-train 上扫，heldout 只判决）：

  - `bm25`：基线；
  - `pos_phrase`：只对词性模式属于短语集合的对加分（待测）；
  - `all_pairs`：对所有内容词相邻对加分（复现 §8 的 seq 族，作正对照）；
  - `random_subset`：从查询的相邻对里**随机抽同样数量**的对——控制「加了多少个奖励」
    这个混杂因素，只看「选得对不对」。

只读 corpus.jsonl + train split；不读 test qrels。

用法（在 tools/kernel/ 下）：
    uv run python pos_phrase.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_concept as T  # noqa: E402
from diagnose_pos import UPOS, build_lexicon, CACHE  # noqa: E402

K1, B = T.K1, T.B
WINDOW = 5
CONTENT = {"ADJ", "ADV", "NOUN", "PROPN", "VERB", "NUM"}
# 「构成短语」的模式：修饰语 + 中心语，或并列的中心语
PHRASE = {
    ("ADJ", "NOUN"), ("ADJ", "PROPN"), ("ADJ", "ADJ"),
    ("NOUN", "NOUN"), ("PROPN", "NOUN"), ("NOUN", "PROPN"),
    ("VERB", "NOUN"), ("ADV", "ADJ"), ("ADV", "VERB"), ("NUM", "NOUN"),
}


def positional_index(tokens_per_doc):
    """token → {doc: [positions]}。"""
    out: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, toks in enumerate(tokens_per_doc):
        for p, tok in enumerate(toks):
            out[tok][i].append(p)
    return out


def min_gap(pa: list[int], pb: list[int]) -> int:
    """两个升序位置列表的最小距离。"""
    i = j = 0
    best = 10**9
    while i < len(pa) and j < len(pb):
        best = min(best, abs(pa[i] - pb[j]))
        if best == 0:
            return 0
        if pa[i] < pb[j]:
            i += 1
        else:
            j += 1
    return best


def main() -> int:
    docs, queries, qrels = T.load()
    corpus = T.Corpus(docs)
    ids = corpus.ids
    outer = [q for q in sorted(qrels) if not T.is_heldout(q)]
    held = [q for q in sorted(qrels) if T.is_heldout(q)]

    craft = sorted((CACHE / "CRAFT/structural-annotation/dependency/conllu").glob("*.conllu"))
    ewt = [CACHE / "UD_English-EWT" / f"en_ewt-ud-{s}.conllu" for s in ("train",)]
    if not craft or not all(p.is_file() for p in ewt):
        raise SystemExit("缺少词性语料，先跑 fetch_grammar_corpora.py")
    dist, _ = build_lexicon(craft + ewt)
    tag = {t: UPOS[int(np.argmax(v))] for t, v in dist.items()}
    print(f"词性词表 {len(tag)} 个词形", file=sys.stderr)

    pos_idx = positional_index(corpus.doc_tokens)
    # 基线 BM25 分（与 Rust bm25 同口径）
    base_scores: dict[str, dict[int, float]] = {}
    query_pairs: dict[str, list[tuple[str, str]]] = {}
    for qid in outer + held:
        terms = T.bm25_terms(queries[qid])
        uniq = sorted(set(terms))
        scores: dict[int, float] = defaultdict(float)
        for t in uniq:
            plist = corpus.postings.get(t)
            if not plist:
                continue
            idf = math.log(1.0 + (corpus.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for d, tfv in corpus.tf[t].items():
                den = tfv + K1 * (1.0 - B + B * corpus.doc_len[d] / max(corpus.avg_len, 1e-9))
                scores[d] += idf * tfv * (K1 + 1.0) / den
        base_scores[qid] = dict(scores)
        # 内容词序列上的相邻对（跳过功能词）
        content = [t for t in terms if tag.get(t) in CONTENT]
        query_pairs[qid] = list(zip(content, content[1:]))

    def pair_docs(a: str, b: str):
        """同时含 a、b 且最小距离 ≤ WINDOW 的文档。"""
        da, db = pos_idx.get(a), pos_idx.get(b)
        if not da or not db:
            return {}
        out = {}
        for d, pa in da.items():
            pb = db.get(d)
            if pb is None:
                continue
            g = min_gap(pa, pb)
            if g <= WINDOW:
                out[d] = 1.0 / (1.0 + g)  # 越近越强
        return out

    def select(qid: str, mode: str, rng=None) -> list[tuple[str, str]]:
        pairs = query_pairs.get(qid, [])
        if mode == "pos_phrase":
            return [p for p in pairs if (tag.get(p[0]), tag.get(p[1])) in PHRASE]
        if mode == "all_pairs":
            return pairs
        if mode == "random_subset":
            k = len([p for p in pairs if (tag.get(p[0]), tag.get(p[1])) in PHRASE])
            if k == 0 or not pairs:
                return []
            idx = rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)
            return [pairs[i] for i in sorted(idx)]
        raise ValueError(mode)

    def bonus_map(qid: str, pairs) -> dict[int, float]:
        acc: dict[int, float] = defaultdict(float)
        for a, b in pairs:
            if a == b:
                continue
            for d, v in pair_docs(a, b).items():
                acc[d] += v
        return dict(acc)

    def ndcg_with(qid: str, bonus: dict[int, float], gamma: float) -> float:
        scores = dict(base_scores[qid])
        if gamma:
            for d, v in bonus.items():
                scores[d] = scores.get(d, 0.0) + gamma * v
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], ids[kv[0]]))[:100]
        return T.ndcg_at_10([ids[d] for d, _ in ranked], qrels[qid])

    # 预计算每臂每查询的 bonus
    rng = np.random.default_rng(20261113)
    arms = ("pos_phrase", "all_pairs", "random_subset")
    bonus: dict[str, dict[str, dict[int, float]]] = {a: {} for a in arms}
    for qid in outer + held:
        for mode in arms:
            bonus[mode][qid] = bonus_map(qid, select(qid, mode, rng))

    fired = np.mean([len(select(q, "pos_phrase")) for q in outer])
    fired_all = np.mean([len(select(q, "all_pairs")) for q in outer])
    print(f"平均每查询：短语对 {fired:.2f} 个 / 全部相邻对 {fired_all:.2f} 个", file=sys.stderr)

    base_outer = {q: ndcg_with(q, {}, 0.0) for q in outer}
    base_held = {q: ndcg_with(q, {}, 0.0) for q in held}
    base_o, base_h = float(np.mean(list(base_outer.values()))), float(np.mean(list(base_held.values())))
    print(f"baseline outer={base_o:.5f} heldout={base_h:.5f}", file=sys.stderr)
    if abs(base_h - 0.66018) > 1e-5:
        raise SystemExit(f"口径漂了：{base_h:.5f} ≠ 0.66018")

    gammas = (0.2, 0.5, 1.0, 2.0)
    best = {}
    for mode in arms:
        trials = []
        for g in gammas:
            m = float(np.mean([ndcg_with(q, bonus[mode][q], g) for q in outer]))
            trials.append((m, g))
        trials.sort(reverse=True)
        best[mode] = trials[0]
        print(f"  {mode:14s} outer 最优 γ={trials[0][1]} → {trials[0][0]:.5f}"
              f"（全部：{', '.join(f'g{g}={m:.4f}' for m, g in trials)}）", file=sys.stderr)

    print(f"\n{'arm':16s} {'γ':>5s} {'outer':>8s} {'heldout':>8s} {'Δheld':>9s} {'2SE':>8s}  判定")
    print(f"{'bm25':16s} {'—':>5s} {base_o:8.5f} {base_h:8.5f} {'—':>9s} {'—':>8s}")
    scores_held = {}
    for mode in arms:
        _, g = best[mode]
        held_scores = np.array([ndcg_with(q, bonus[mode][q], g) for q in held])
        scores_held[mode] = held_scores
        d = held_scores - np.array([base_held[q] for q in held])
        m, se, _, _ = T.bootstrap(d)
        flag = "越过 2SE" if abs(m) > 2 * se else "噪声带内"
        print(f"{mode:16s} {g:5.1f} {best[mode][0]:8.5f} {held_scores.mean():8.5f} "
              f"{m:+9.5f} {2*se:8.5f}  {flag}")

    # 核心对照：选得对不对（pos_phrase vs 同数量的随机对）
    d = scores_held["pos_phrase"] - scores_held["random_subset"]
    m, se, lo, hi = T.bootstrap(d)
    print(f"\npos_phrase − random_subset：Δ={m:+.5f}  2SE={2*se:.5f}  [{lo:+.5f}, {hi:+.5f}]"
          f"  {'越过 2SE' if abs(m) > 2 * se else '噪声带内'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
