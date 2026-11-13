#!/usr/bin/env python3
"""内核 v2：**学习**概念归并谓词，产出 `token → concept` 先验因子。

与 P0 统计基线的区别：没有任何手写后缀规则。这里只有两件事是固定的——

  ① **候选对空间**（结构性的，与语言无关）：`t` 是 `u` 的前缀且 `len(t) >= 3`；
  ② **特征表**（全部来自语料统计，不看 qrels）。

「哪些候选对真的该合并」由**学到的权重** `w` 决定：`merge(t,u) = σ(w·x(t,u) + b) >= 0.5`。
参数量 = 特征数 + 1（截距），在 `AGENTS.md` 定的 ≤10k 预算内。

训练目标直接是 harness 指标：在 **outer-train** 上最大化 nDCG@10（坐标搜索）。
**heldout 不参与训练**，只用于最后判决；test qrels 全程不读。

用法：
    uv run python train_concept.py --passes 2          # 训练 + heldout 判决
    uv run python train_concept.py --emit artifact     # 只会用已存的权重产工件
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "kernel"))
from tokenizer import bm25_terms  # noqa: E402

DATA = ROOT / "data" / "scifact"
K1, B = 1.5, 0.75
WEIGHTS_PATH = ROOT / "data" / "scifact" / "artifacts" / "concept-v0" / "weights.json"

FEATURES = (
    "p_t_given_u",   # u 出现时 t 出现的比例（形态变体应接近 1）
    "p_u_given_t",   # t 出现时 u 出现的比例
    "jaccard",       # posting 集合的 Jaccard
    "log_df_ratio",  # log(df_t) - log(df_u)
    "len_ratio",     # len(t) / len(u)
    "extra_chars",   # len(u) - len(t)，只由长度决定
    "tf_cosine",     # 两词 tf 向量（以文档为轴）的余弦
    "ppmi",          # 文档内共现 PPMI
)


# ---------- 数据 ----------


def is_heldout(qid: str) -> bool:
    h = 20261110 ^ 0xCBF29CE484222325
    for byte in qid.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % 5 == 0


def load_corpus(dataset_dir: Path = DATA):
    """只读语料。跨数据集构建工件时目标语料可能没有 train split。"""
    return [json.loads(l) for l in (dataset_dir / "corpus.jsonl").open(encoding="utf-8") if l.strip()]


def load(dataset_dir: Path = DATA):
    docs = load_corpus(dataset_dir)
    queries: dict[str, str] = {}
    with (dataset_dir / "train_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with (dataset_dir / "train_qrels.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, rel = parts
                if int(rel) > 0 and qid in queries:
                    qrels.setdefault(qid, {})[did] = int(rel)
    return docs, queries, qrels


# ---------- 语料统计（一次性，与 qrels 无关） ----------


class Corpus:
    def __init__(self, docs):
        self.ids = [d["id"] for d in docs]
        self.n = len(docs)
        self.doc_tokens: list[list[str]] = []
        self.title_tokens: list[list[str]] = []
        for d in docs:
            full = (d.get("title", "") + "\n" + d.get("text", "")) if d.get("title") else d.get("text", "")
            self.doc_tokens.append(bm25_terms(full))
            self.title_tokens.append(bm25_terms(d.get("title", "")))
        self.postings: dict[str, set[int]] = {}
        self.tf: dict[str, dict[int, int]] = {}
        for i, terms in enumerate(self.doc_tokens):
            local: dict[str, int] = {}
            for t in terms:
                local[t] = local.get(t, 0) + 1
            for t, c in local.items():
                self.postings.setdefault(t, set()).add(i)
                self.tf.setdefault(t, {})[i] = c
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.doc_len = [len(x) for x in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / self.n

    def candidates(
        self,
        rule: str = "lcp",
        min_len: int = 4,
        max_extra: int = 8,
        lcp_ratio: float = 0.7,
        bucket: int = 4,
        max_bucket: int = 4000,
    ):
        """候选对 `(短, 长)`：两者**共享前缀**至少覆盖较长词的 `lcp_ratio`。

        为什么不用「严格前缀」：那会漏掉 `antibody`/`antibodies` 这类词干发生变化的复数
        （scifact 上 228 对 y→ies 只覆盖到 6 对），又会放进 `the`/`therapies` 这种
        「短词是长词前缀但毫无形态关系」的对。共享前缀比例同时修掉两边。

        `lcp_ratio` 与 `min_len` 是**候选空间**的结构参数，不是判定规则：
        哪些候选对真的合并，仍由内核学到的权重决定。

        `rule="prefix"` 是严格前缀版（`t` 是 `u` 的前缀），保留它是为了让已发布的
        `concept-v0` 工件能被逐字节复现——两个工件之间唯一的差别就是这条候选规则。
        """
        if rule == "prefix":
            out = []
            for u in self.postings:
                lu = len(u)
                if lu <= min_len:
                    continue
                for i in range(min_len, lu):
                    if lu - i > max_extra:
                        continue
                    t = u[:i]
                    if t in self.postings:
                        out.append((t, u))
            return out
        if rule != "lcp":
            raise ValueError(f"未知候选规则 {rule!r}，只接受 prefix / lcp")
        by_prefix: dict[str, list[str]] = {}
        for token in self.postings:
            if len(token) >= min_len:
                by_prefix.setdefault(token[:bucket], []).append(token)

        out = []
        for members in by_prefix.values():
            if len(members) < 2 or len(members) > max_bucket:
                continue
            members.sort()
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    la, lb = len(a), len(b)
                    if abs(la - lb) > max_extra:
                        continue
                    shared = 0
                    for ca, cb in zip(a, b):
                        if ca != cb:
                            break
                        shared += 1
                    if shared >= lcp_ratio * max(la, lb):
                        out.append((a, b) if la <= lb else (b, a))
        return out

    def pair_features(self, t: str, u: str) -> np.ndarray:
        pt, pu = self.postings[t], self.postings[u]
        dt, du = len(pt), len(pu)
        inter = len(pt & pu)
        # 高频词互信息不可靠：df 越大，PPMI 越趋近 0，天然把高频对压下去。
        tt, tu = self.tf[t], self.tf[u]
        dot = sum(c * tu[d] for d, c in tt.items() if d in tu)
        nt = math.sqrt(sum(c * c for c in tt.values())) or 1.0
        nu = math.sqrt(sum(c * c for c in tu.values())) or 1.0
        # PPMI：把「共现」当独立事件，N 用文档数
        pmi = 0.0
        if inter:
            pmi = math.log((inter / self.n) / ((dt / self.n) * (du / self.n) + 1e-12) + 1e-12)
        return np.array(
            [
                inter / dt if dt else 0.0,
                inter / du if du else 0.0,
                inter / (dt + du - inter) if (dt + du - inter) else 0.0,
                math.log(dt + 1.0) - math.log(du + 1.0),
                len(t) / len(u),
                float(len(u) - len(t)),
                dot / (nt * nu),
                max(0.0, pmi),
            ],
            dtype=np.float64,
        )


# ---------- 打分（与 Rust bm25 同口径） ----------


def build_index(corpus: Corpus, mapping: dict[str, str]):
    """应用概念映射后重建倒排表。映射只改 token 身份，不改分词。"""
    postings: dict[str, list[tuple[int, int]]] = {}
    tpostings: dict[str, list[tuple[int, int]]] = {}
    doc_len = []
    total = 0
    for i in range(corpus.n):
        terms = [mapping.get(t, t) for t in corpus.doc_tokens[i]]
        total += len(terms)
        doc_len.append(len(terms))
        local: dict[str, int] = {}
        for t in terms:
            local[t] = local.get(t, 0) + 1
        for t, c in local.items():
            postings.setdefault(t, []).append((i, c))
        tlocal: dict[str, int] = {}
        for t in [mapping.get(x, x) for x in corpus.title_tokens[i]]:
            tlocal[t] = tlocal.get(t, 0) + 1
        for t, c in tlocal.items():
            tpostings.setdefault(t, []).append((i, c))
    n = corpus.n
    idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}
    return {
        "postings": postings,
        "tpostings": tpostings,
        "idf": idf,
        "doc_len": doc_len,
        "avg_len": total / n if n else 0.0,
        "mapping": mapping,
    }


def rank(ix, ids, query: str, k: int, uniform_beta: float = 0.0) -> list[str]:
    scores: dict[int, float] = {}
    terms = sorted({ix["mapping"].get(t, t) for t in bm25_terms(query)})
    for t in terms:
        plist = ix["postings"].get(t)
        if not plist:
            continue
        idf = ix["idf"][t]
        tmap = dict(ix["tpostings"].get(t, [])) if uniform_beta else {}
        for doc, tf in plist:
            eff = tf + uniform_beta * tmap.get(doc, 0)
            denom = eff + K1 * (1.0 - B + B * ix["doc_len"][doc] / max(ix["avg_len"], 1e-9))
            scores[doc] = scores.get(doc, 0.0) + idf * eff * (K1 + 1.0) / denom
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], ids[kv[0]]))[:k]
    return [ids[i] for i, _ in ordered]


def ndcg_at_10(ranking, rel) -> float:
    dcg = sum((2 ** rel[d] - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:10]) if d in rel)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(ix, ids, qids, queries, qrels, uniform_beta: float = 0.0) -> dict[str, float]:
    return {qid: ndcg_at_10(rank(ix, ids, queries[qid], 100, uniform_beta), qrels[qid]) for qid in qids}


def bootstrap(deltas, rounds=2000):
    rng = np.random.default_rng(20261111)
    n = len(deltas)
    means = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(rounds)])
    return deltas.mean(), deltas.std(ddof=1) / math.sqrt(n), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- 学习 ----------


def make_mapping(cands, X, w, b, df) -> dict[str, str]:
    """σ(w·x+b) >= 0.5 ⇔ w·x+b >= 0；并查集取组内 df 最大者为规范形式。"""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    z = X @ w + b
    for idx in np.nonzero(z >= 0.0)[0]:
        t, u = cands[idx]
        rt, ru = find(t), find(u)
        if rt != ru:
            parent[ru] = rt
    groups: dict[str, list[str]] = {}
    for tok in parent:
        groups.setdefault(find(tok), []).append(tok)
    mapping: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canon = max(members, key=lambda t: (df.get(t, 0), [-ord(c) for c in t]))
        for m in members:
            if m != canon:
                mapping[m] = canon
    return mapping


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-extra", type=int, default=8)
    ap.add_argument("--lcp-ratio", type=float, default=0.7)
    ap.add_argument("--cand-rule", choices=("lcp", "prefix"), default="lcp")
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    args = ap.parse_args(argv)

    docs, queries, qrels = load()
    outer = [q for q in sorted(qrels) if not is_heldout(q)]
    held = [q for q in sorted(qrels) if is_heldout(q)]
    print(f"docs={len(docs)} outer_train={len(outer)} heldout={len(held)}", file=sys.stderr)

    corpus = Corpus(docs)
    cands = corpus.candidates(
        rule=args.cand_rule,
        min_len=args.min_len,
        max_extra=args.max_extra,
        lcp_ratio=args.lcp_ratio,
    )
    print(f"vocab={len(corpus.df)} candidate_pairs={len(cands)}", file=sys.stderr)
    X = np.stack([corpus.pair_features(t, u) for t, u in cands])

    base_ix = build_index(corpus, {})
    base_outer = evaluate(base_ix, corpus.ids, outer, queries, qrels)
    base_ndcg = float(np.mean(list(base_outer.values())))
    print(f"baseline outer nDCG@10 = {base_ndcg:.5f}", file=sys.stderr)

    # 权重初值：特征归一化到可比尺度后，从「只信 p_t_given_u」出发（不是手写规则，是初始化）。
    scale = np.abs(X).max(axis=0)
    Xn = X / np.where(scale > 0, scale, 1.0)
    w = np.zeros(len(FEATURES))
    w[0] = 1.0
    b = -0.5
    best = (-1.0, w.copy(), b)

    grid = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    for p in range(args.passes):
        for j in range(len(w) + 1):
            for g in grid:
                w_try, b_try = w.copy(), b
                if j < len(w):
                    w_try[j] = g
                else:
                    b_try = g * 0.5
                mapping = make_mapping(cands, Xn, w_try, b_try, corpus.df)
                ix = build_index(corpus, mapping)
                score = float(np.mean(list(evaluate(ix, corpus.ids, outer, queries, qrels).values())))
                if score > best[0]:
                    best = (score, w_try, b_try)
            w, b = best[1].copy(), best[2]
            print(f"pass{p} dim{j} -> best_outer={best[0]:.5f} merged={len(make_mapping(cands, Xn, w, b, corpus.df))}", file=sys.stderr)

    score, w, b = best
    mapping = make_mapping(cands, Xn, w, b, corpus.df)
    print(f"\nlearned: outer nDCG@10 {base_ndcg:.5f} -> {score:.5f} (merged {len(mapping)} tokens)", file=sys.stderr)
    for name, wj in zip(FEATURES, w):
        print(f"  {name:14s} {wj:+.3f}", file=sys.stderr)
    print(f"  {'intercept':14s} {b:+.3f}", file=sys.stderr)

    ix = build_index(corpus, mapping)
    held_scores = evaluate(ix, corpus.ids, held, queries, qrels)
    d = np.array([held_scores[q] - base_outer.get(q, ndcg_at_10(rank(base_ix, corpus.ids, queries[q], 100), qrels[q])) for q in held])
    m, se, lo, hi = bootstrap(d)
    print(f"\nheldout nDCG@10 delta {m:+.5f}  2SE {2*se:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
          f"{'  * exceeds noise band' if abs(m) > 2 * se else '  (within noise band)'}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "features": list(FEATURES),
                "weights": w.tolist(),
                "intercept": b,
                "feature_scale": scale.tolist(),
                "train_split": "outer_train (n=%d)" % len(outer),
                "trained_on": "scifact train_qrels.tsv 的 outer-train 划分（heldout 不参与）",
                "objective": "nDCG@10（坐标搜索，非代理损失）",
                "candidate_rule": {
                    "rule": args.cand_rule,
                    "min_len": args.min_len,
                    "max_extra": args.max_extra,
                    "lcp_ratio": args.lcp_ratio,
                },
                "params": len(FEATURES) + 1,
                "merged_tokens": len(mapping),
                "outer_ndcg": score,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
