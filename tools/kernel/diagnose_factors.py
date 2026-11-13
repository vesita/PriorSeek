#!/usr/bin/env python3
"""离线诊断：判断某个候选「词项加权」因子族值不值得进 Rust 实现。

**结论用途**：`docs/RESULTS.md` §15.2 的三条「已证伪方向」就是本脚本的产出。留着它是为了
以后不必重做同一批否定实验；新候选因子先进这里，过了噪声带再进外壳。

口径锁定：本脚本自己实现了 BM25，因此**必须**与权威 harness 对得上。脚本内置断言——
heldout 上 `bm25` 的 nDCG@10 必须等于 `0.66018`（±1e-5），对不上就说明口径漂了，
此时脚本的读数一律不可用（`ml-experiment-attribution` §2.2：测量函数先过对照输入）。

只读 corpus.jsonl + train_queries.jsonl + train_qrels.tsv，不读 qrels.tsv（test）。

用法（在 tools/kernel/ 下）：
    uv run python diagnose_factors.py                 # heldout 146（与权威口径对齐）
    DIAG_SPLIT=outer uv run python diagnose_factors.py  # outer-train 663（n 大，Δ_min 小）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import bm25_terms  # noqa: E402

DATA = ROOT / "data/scifact"
K1, B = 1.5, 0.75


def is_heldout(qid: str) -> bool:
    h = 20261110 ^ 0xCBF29CE484222325
    for byte in qid.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % 5 == 0


def load():
    docs = []
    with (DATA / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    queries = {}
    with (DATA / "train_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["id"]] = q["text"]
    qrels = {}
    with (DATA / "train_qrels.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, did, rel = parts
            if int(rel) > 0 and qid in queries:
                qrels.setdefault(qid, {})[did] = int(rel)
    return docs, queries, qrels


class Index:
    def __init__(self, docs):
        self.ids = [d["id"] for d in docs]
        self.n = len(docs)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.tpostings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len = []
        total = 0
        for i, d in enumerate(docs):
            full = (d.get("title", "") + "\n" + d.get("text", "")) if d.get("title") else d.get("text", "")
            terms = bm25_terms(full)
            total += len(terms)
            self.doc_len.append(len(terms))
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.postings.setdefault(t, []).append((i, c))
            ttf: dict[str, int] = {}
            for t in bm25_terms(d.get("title", "")):
                ttf[t] = ttf.get(t, 0) + 1
            for t, c in ttf.items():
                self.tpostings.setdefault(t, []).append((i, c))
        self.avg_len = total / self.n if self.n else 0.0
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.tdf = {t: len(p) for t, p in self.tpostings.items()}
        self.idf = {t: math.log(1.0 + (self.n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        self.tidf = {t: math.log(1.0 + (self.n - df + 0.5) / (df + 0.5)) for t, df in self.tdf.items()}
        self.idx = {d: i for i, d in enumerate(self.ids)}

    def terms(self, query: str):
        return sorted(set(bm25_terms(query)))

    def bm25_component(self, tf: float, dl: int) -> float:
        denom = tf + K1 * (1.0 - B + B * dl / max(self.avg_len, 1e-9))
        return tf * (K1 + 1.0) / denom

    def rank(self, query: str, k: int, scorer) -> list[str]:
        scores: dict[int, float] = {}
        for t in self.terms(query):
            if t in self.postings:
                scorer(self, t, scores)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))[:k]
        return [self.ids[i] for i, _ in ranked]


def s_baseline(ix: Index, t: str, scores: dict) -> None:
    idf = ix.idf[t]
    for doc, tf in ix.postings[t]:
        scores[doc] = scores.get(doc, 0.0) + idf * ix.bm25_component(tf, ix.doc_len[doc])


def s_uniform(ix: Index, t: str, scores: dict, beta: float = 1.0) -> None:
    idf = ix.idf[t]
    tmap = dict(ix.tpostings.get(t, []))
    for doc, tf in ix.postings[t]:
        eff = tf + beta * tmap.get(doc, 0)
        scores[doc] = scores.get(doc, 0.0) + idf * ix.bm25_component(eff, ix.doc_len[doc])


def s_field_idf(ix: Index, t: str, scores: dict) -> None:
    """标题命中用标题 idf，正文命中用全文 idf（标准 BM25F 只用单一 idf）。"""
    tmap = dict(ix.tpostings.get(t, []))
    full = dict(ix.postings[t])
    for doc, tf in full.items():
        ttf = tmap.get(doc, 0)
        body = max(tf - ttf, 0)
        dl = ix.doc_len[doc]
        score = 0.0
        if ttf:
            score += ix.tidf[t] * ix.bm25_component(ttf, dl)
        if body:
            score += ix.idf[t] * ix.bm25_component(body, dl)
        if score:
            scores[doc] = scores.get(doc, 0.0) + score


def make_lift_scorer(gamma: float):
    def scorer(ix: Index, t: str, scores: dict) -> None:
        base = ix.idf[t]
        lift = (ix.tidf.get(t, 0.0) - base) / base if base > 0 else 0.0
        w = max(0.0, 1.0 + gamma * lift)
        for doc, tf in ix.postings[t]:
            scores[doc] = scores.get(doc, 0.0) + w * base * ix.bm25_component(tf, ix.doc_len[doc])
    return scorer


def make_saliency_scorer(gamma: float):
    """token → 词频分布集中度：max_tf/sum_tf 越高越「文档专属」。"""
    cache: dict[str, float] = {}

    def score_of(ix: Index, t: str) -> float:
        if t in cache:
            return cache[t]
        plist = ix.postings[t]
        total = sum(c for _, c in plist)
        peak = max((c for _, c in plist), default=0)
        v = peak / total if total else 0.0
        cache[t] = v
        return v

    def scorer(ix: Index, t: str, scores: dict) -> None:
        base = ix.idf[t]
        w = max(0.0, 1.0 + gamma * (score_of(ix, t) - 0.5) * 2.0)
        for doc, tf in ix.postings[t]:
            scores[doc] = scores.get(doc, 0.0) + w * base * ix.bm25_component(tf, ix.doc_len[doc])

    return scorer


# ---------- 指标 ----------


def ndcg_at_10(ranking: list[str], rel: dict[str, int]) -> float:
    dcg = 0.0
    for i, did in enumerate(ranking[:10]):
        r = rel.get(did, 0)
        if r:
            dcg += (2**r - 1) / math.log2(i + 2)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def hit_at_10(ranking: list[str], rel: dict[str, int]) -> float:
    return 1.0 if any(d in rel for d in ranking[:10]) else 0.0


def bootstrap(deltas: np.ndarray, rounds: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(20261111)
    n = len(deltas)
    means = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(rounds)])
    return deltas.mean(), deltas.std(ddof=1) / math.sqrt(n), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    docs, queries, qrels = load()
    split_args = __import__("os").environ.get("DIAG_SPLIT", "heldout")
    split = split_args
    if split == "heldout":
        outer = [q for q in sorted(qrels) if is_heldout(q)]
    else:
        outer = [q for q in sorted(qrels) if not is_heldout(q)]
    print(f"docs={len(docs)} split={split} queries={len(outer)}", file=sys.stderr)

    ix = Index(docs)

    arms = {
        "baseline": (ix, s_baseline),
        "uniform_b1": (ix, lambda i, t, s: s_uniform(i, t, s, 1.0)),
        "uniform_b2": (ix, lambda i, t, s: s_uniform(i, t, s, 2.0)),
        "field_idf": (ix, s_field_idf),
        "lift_g1": (ix, make_lift_scorer(1.0)),
        "lift_g3": (ix, make_lift_scorer(3.0)),
        "lift_g10": (ix, make_lift_scorer(10.0)),
        "saliency_g1": (ix, make_saliency_scorer(1.0)),
        "saliency_g3": (ix, make_saliency_scorer(3.0)),
    }

    per_query: dict[str, dict[str, tuple[float, float]]] = {}
    for name, (index, scorer) in arms.items():
        rows = {}
        for qid in outer:
            ranking = index.rank(queries[qid], 100, scorer)
            rel = qrels[qid]
            rows[qid] = (ndcg_at_10(ranking, rel), hit_at_10(ranking, rel))
        per_query[name] = rows
        nd = np.mean([v[0] for v in rows.values()])
        ht = np.mean([v[1] for v in rows.values()])
        print(f"{name:20s} nDCG@10={nd:.5f} hit@10={ht:.4f}", file=sys.stderr)

    if "heldout" in split_args:
        baseline_ndcg = float(np.mean([v[0] for v in per_query["baseline"].values()]))
        expected = 0.66018
        if abs(baseline_ndcg - expected) > 1e-5:
            raise SystemExit(
                f"口径漂了：heldout baseline nDCG@10 = {baseline_ndcg:.5f}，"
                f"应为 {expected:.5f}。先修本脚本，再用它的读数。"
            )
        print(f"口径对照通过：baseline nDCG@10 = {baseline_ndcg:.5f}", file=sys.stderr)

    base = per_query["baseline"]
    print("\n== paired vs baseline ==")
    print(f"{'arm':20s} {'dNDCG':>9s} {'SE':>8s} {'2SE':>8s} {'95% CI':>22s} {'dHit':>8s}")
    for name in arms:
        if name == "baseline":
            continue
        d = np.array([per_query[name][q][0] - base[q][0] for q in outer])
        dh = np.array([per_query[name][q][1] - base[q][1] for q in outer])
        m, se, lo, hi = bootstrap(d)
        print(
            f"{name:20s} {m:+9.5f} {se:8.5f} {2*se:8.5f}  [{lo:+.5f}, {hi:+.5f}] {dh.mean():+8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
