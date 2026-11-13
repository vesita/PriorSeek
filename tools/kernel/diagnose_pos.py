#!/usr/bin/env python3
"""语法（词性）先验因子的实测。

判据继承 `docs/PREREG-grammar.md` §6（决策规则 #0–#4），其中 #0「零假设对照不得复现增益」
在上一版失败后被**收紧**为 `df_strat_perm`：不是纯随机打乱，而是在 log df 的每一层内部置换，
保留每类的规模与 df 结构。收紧的动机与结果记在 `docs/RESULTS.md` §19。

与 `diagnose_grammar.py`（无监督聚类版，已作废）的区别：
类别**不是**聚类出来的，而是来自 gold 词性标注（CRAFT 生物医学 CC BY 3.0 + UD English-EWT
CC BY-SA 4.0），因此类别本身就是语言学定义的 UPOS（ADJ / ADP / VERB / ...），不是任意分箱。

因子形态：`token → [p(ADJ), p(ADP), ...]`（17 个 UPOS 的分布）。
外壳用法：`w(t) = Σ_k p(k|t)·λ_k`；λ 全 1 时逐位退化为 bm25（等价性门）。

三条臂（唯一差别是「词 → 类」的对应表）：
  - `grammar_pos`   ：真实的 UPOS 分布（待测）；
  - `df_bucket`     ：log df 等频 17 桶（参数量相同的频率对照）；
  - `df_strat_perm` ：**在 log df 的每一层内部置换分布行**——保留每类规模与 df 结构，
    只打乱「哪个词归哪类」。这是比「纯随机」严格的零假设：上一版实验里纯随机能完整
    复现增益，正因为它破坏了 df 结构。

只读 corpus.jsonl + train split；不读 test qrels。
内置基线断言：heldout 上 bm25 的 nDCG@10 必须等于 0.66018（±1e-5），否则拒绝出读数。

用法（在 tools/kernel/ 下）：
    uv run python diagnose_pos.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import bm25_terms  # noqa: E402
from diagnose_grammar import (  # noqa: E402
    BASELINE_HELDOUT,
    bootstrap,
    build_index,
    contributions,
    is_heldout,
    load,
    ndcg_at_10,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "tools" / "kernel" / ".cache" / "grammar"
SEED = 20261112
K = 100
UPOS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)


def read_conllu(path: Path):
    """产出 `(归一后 token, UPOS)`；标点与切多片的词形跳过（与 bm25 词项空间对齐）。"""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        form, upos = cols[1], cols[3]
        if upos not in UPOS or upos in ("PUNCT", "SYM"):
            continue
        pieces = bm25_terms(form)
        if len(pieces) == 1:
            yield pieces[0], upos


def build_lexicon(paths: list[Path]) -> tuple[dict[str, np.ndarray], int]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        for tok, tag in read_conllu(path):
            counts[tok][tag] += 1
    dist: dict[str, np.ndarray] = {}
    for tok, c in counts.items():
        v = np.zeros(len(UPOS))
        total = sum(c.values())
        for tag, n in c.items():
            v[UPOS.index(tag)] = n / total
        dist[tok] = v
    return dist, len(paths)


def main() -> int:
    docs, queries, qrels = load()
    ids = [d["id"] for d in docs]
    outer = [q for q in sorted(qrels) if not is_heldout(q)]
    held = [q for q in sorted(qrels) if is_heldout(q)]

    craft = sorted((CACHE / "CRAFT/structural-annotation/dependency/conllu").glob("*.conllu"))
    ewt = [CACHE / "UD_English-EWT" / f"en_ewt-ud-{s}.conllu" for s in ("train",)]
    if not craft or not all(p.is_file() for p in ewt):
        raise SystemExit(f"缺少词性语料：CRAFT {len(craft)} 个文件，EWT 存在={[p.is_file() for p in ewt]}")
    dist, n_files = build_lexicon(craft + ewt)
    print(f"词性语料：{n_files} 个 CoNLL-U 文件 → {len(dist)} 个词形", file=sys.stderr)

    tokens_per_doc = []
    for d in docs:
        full = (d.get("title", "") + "\n" + d.get("text", "")) if d.get("title") else d.get("text", "")
        tokens_per_doc.append(bm25_terms(full))
    postings, doc_len, avg, idf = build_index(tokens_per_doc)
    vocab = sorted(postings)
    covered = [t for t in vocab if t in dist]
    total_df = sum(len(postings[t]) for t in vocab)
    cov_df = sum(len(postings[t]) for t in covered)
    print(
        f"scifact 词表 {len(vocab)}：词性覆盖 {len(covered)} 词形 "
        f"（{len(covered)/len(vocab):.1%}），按 df 加权 {cov_df/total_df:.1%}",
        file=sys.stderr,
    )

    n_docs = len(ids)
    id_rank = np.argsort(np.argsort(np.array(ids)))
    tok_pos = {t: i for i, t in enumerate(covered)}

    def matrix_of(table: list[np.ndarray]) -> np.ndarray:
        return np.stack(table) if table else np.zeros((0, len(UPOS)))

    M_pos = matrix_of([dist[t] for t in covered])

    # df_bucket：log df 等频 17 桶
    order = np.argsort([math.log1p(len(postings[t])) for t in covered])
    M_bucket = np.zeros((len(covered), len(UPOS)))
    for b, chunk in enumerate(np.array_split(order, len(UPOS))):
        M_bucket[chunk, b] = 1.0

    # df_strat_perm：40 个 df 层内置换分布行
    rng = np.random.default_rng(SEED)
    M_perm = M_pos.copy()
    for chunk in np.array_split(order, 40):
        if len(chunk) > 1:
            M_perm[chunk] = M_pos[chunk[rng.permutation(len(chunk))]]

    # 预计算每条查询的 (docs, vals, spans)，避免每次拟合重算
    zero_cls = {t: 0 for t in vocab}

    def prep(qids):
        rows = []
        for qid in qids:
            terms = sorted(set(bm25_terms(queries[qid])))
            packed = contributions(postings, doc_len, avg, idf, terms, zero_cls)
            if packed is None:
                rows.append((qid, None, None))
                continue
            docs_idx, vals, _ = packed
            spans, cursor = [], 0
            for t in terms:
                plist = postings.get(t)
                if not plist:
                    continue
                spans.append((cursor, cursor + len(plist), t))
                cursor += len(plist)
            rows.append((qid, (docs_idx, vals, spans), qrels[qid]))
        return rows

    outer_rows = prep(outer)
    held_rows = prep(held)

    def rank(rows, matrix, lam):
        """返回 qid → top-k doc 下标列表。"""
        out = {}
        for qid, data, _ in rows:
            if data is None:
                out[qid] = []
                continue
            docs_idx, vals, spans = data
            w = np.ones(len(vals))
            for lo, hi, t in spans:
                j = tok_pos.get(t)
                if j is not None:
                    w[lo:hi] = float(matrix[j] @ lam)
            total = np.bincount(docs_idx, weights=vals * w, minlength=n_docs)
            cand_n = min(500, n_docs)
            cand = np.argpartition(-total, cand_n - 1)[:cand_n]
            o = np.lexsort((id_rank[cand], -total[cand]))
            out[qid] = cand[o][:K]
        return out

    def ndcg(rows, matrix, lam):
        return np.array(
            [ndcg_at_10([ids[d] for d in ranked], rel)
             for (qid, _, rel), ranked in zip(rows, rank(rows, matrix, lam).values())]
        )

    def fit(matrix, rounds=2):
        lam = np.ones(len(UPOS))
        best = float(ndcg(outer_rows, matrix, lam).mean())
        grid = (0.5, 0.75, 1.0, 1.5, 2.0)
        for _ in range(rounds):
            for j in range(len(UPOS)):
                for g in grid:
                    trial = lam.copy()
                    trial[j] = g
                    s = float(ndcg(outer_rows, matrix, trial).mean())
                    if s > best:
                        best, lam = s, trial
            print(f"    best_outer={best:.5f}", file=sys.stderr)
        return lam, best

    ones = np.ones(len(UPOS))
    base_held = ndcg(held_rows, M_pos, ones)
    base_outer = float(ndcg(outer_rows, M_pos, ones).mean())
    base_ndcg = float(base_held.mean())
    print(f"\nbaseline heldout nDCG@10={base_ndcg:.5f}（应为 {BASELINE_HELDOUT}）", file=sys.stderr)
    if abs(base_ndcg - BASELINE_HELDOUT) > 1e-5:
        raise SystemExit(f"口径漂了：{base_ndcg:.5f} ≠ {BASELINE_HELDOUT}，拒绝出读数")
    print("口径对照通过", file=sys.stderr)

    print(f"\n{'arm':16s} {'outer':>8s} {'held':>8s} {'Δheld':>9s} {'2SE':>8s} {'95% CI':>24s}  判定")
    print(f"{'bm25':16s} {base_outer:8.5f} {base_ndcg:8.5f} {'—':>9s} {'—':>8s}")

    results = {}
    for name, matrix in (("grammar_pos", M_pos), ("df_bucket", M_bucket), ("df_strat_perm", M_perm)):
        print(f"\n--- 拟合 {name} λ（{len(UPOS)} 个标量）---", file=sys.stderr)
        lam, _ = fit(matrix)
        held_scores = ndcg(held_rows, matrix, lam)
        outer_scores = float(ndcg(outer_rows, matrix, lam).mean())
        delta = held_scores - base_held
        m, se, lo, hi = bootstrap(delta)
        results[name] = (lam, held_scores, m, 2 * se)
        flag = "越过 2SE" if abs(m) > 2 * se else "噪声带内"
        print(f"{name:16s} {outer_scores:8.5f} {held_scores.mean():8.5f} "
              f"{m:+9.5f} {2*se:8.5f}  [{lo:+.5f}, {hi:+.5f}]  {flag}")

    # 核心检验：语法内容 vs 同等 df 结构的置换
    d = results["grammar_pos"][1] - results["df_strat_perm"][1]
    m, se, lo, hi = bootstrap(d)
    print(f"\ngrammar_pos − df_strat_perm：Δ={m:+.5f}  2SE={2*se:.5f}  [{lo:+.5f}, {hi:+.5f}]"
          f"  {'越过 2SE' if abs(m) > 2 * se else '噪声带内'}")

    lam = results["grammar_pos"][0]
    print("\n学到的 λ（按 UPOS）与类别规模：")
    argmax = np.argmax(M_pos, axis=1) if len(covered) else np.array([], dtype=int)
    for i, tag in enumerate(UPOS):
        n = int((argmax == i).sum())
        print(f"  {tag:6s} λ={lam[i]:4.2f}  词形数={n:5d}")

    # 干预非零性冒烟（R29）
    top_default = rank(held_rows, M_pos, ones)
    top_learned = rank(held_rows, M_pos, lam)
    changed = sum(1 for qid in top_default if top_default[qid].tolist() != top_learned[qid].tolist())
    print(f"\n干预冒烟：heldout top-10 变化 {changed}/{len(held)} = {changed/len(held):.1%}（门槛 10%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
