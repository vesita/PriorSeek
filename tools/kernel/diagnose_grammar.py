#!/usr/bin/env python3
"""【读数已作废】语法先验因子 v1：无监督结构聚类 + 每类一个 λ。

**不要引用本脚本的输出。** 它用 K=16 无监督聚类产出类别，读数为
`grammar +0.02516` / `random +0.02516`——随机标签逐位复现了增益，按
`docs/PREREG-grammar.md` §6 决策规则 #0 判定为**测量无效**（被测量的是「任意 16 分箱上的
重加权」，不是语法类）。原因与修法见 `docs/RESULTS.md` §19.4。

保留它的两个理由：① 记录这次作废，防止重蹈；② `build_index` / `contributions` /
`bootstrap` / `ndcg_at_10` 被 `diagnose_pos.py`（有效版本）复用。

有效的版本：`diagnose_pos.py`（gold 词性标注 + df 分层置换对照）。

原始意图（作废前）：语法（结构）先验因子的离线诊断 + 外壳 λ 校准，按 `docs/PREREG-grammar.md` 执行。

做三件事：
  1. Stage A：从语料原始文本抽 12 维**结构特征**，K=16 无监督聚类，得 `p(k|t)`（不读 qrels）；
  2. 质量前置检查：类别与 df 桶的归一化互信息（PREREG 决策规则 #1）；
  3. Stage B：只在 outer-train 上拟合外壳 λ（每类一个标量），heldout 判决；
     并跑 `df-bucket-λ`（参数量相同的频率对照）与 `random-λ`（测量自检）。

口径与 Rust `bm25` 一致：idf = ln(1+(N-df+0.5)/(df+0.5))，k1=1.5，b=0.75，查询词项去重。
脚本内置基线断言：heldout 上 `bm25` 的 nDCG@10 必须等于 0.66018（±1e-5），否则拒绝出读数。

用法（在 tools/kernel/ 下）：
    uv run python diagnose_grammar.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import bm25_terms  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "scifact"
K1, B = 1.5, 0.75
N_CLASSES = 16
SEED = 20261112
BASELINE_HELDOUT = 0.66018

FEATURE_NAMES = (
    "log_df",
    "cap_ratio",
    "sentence_initial_ratio",
    "title_ratio",
    "mean_rel_pos",
    "left_neighbor_ttr",
    "right_neighbor_ttr",
    "mean_log_df_left",
    "mean_log_df_right",
    "has_digit_ratio",
    "all_alpha_ratio",
    "mean_len",
)


def is_heldout(qid: str) -> bool:
    h = 20261110 ^ 0xCBF29CE484222325
    for byte in qid.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % 5 == 0


def load():
    docs = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8") if l.strip()]
    queries: dict[str, str] = {}
    with (DATA / "train_queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with (DATA / "train_qrels.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, rel = parts
                if int(rel) > 0 and qid in queries:
                    qrels.setdefault(qid, {})[did] = int(rel)
    return docs, queries, qrels


def raw_blocks(text: str):
    """与 bm25_terms 同一次扫描，但保留原始形态（大小写 / 数字 / 是否纯字母）。"""
    out = []
    start = None
    for i, ch in enumerate(text):
        if ch.isalnum():
            if start is None:
                start = i
        elif start is not None:
            out.append((text[start:i], i))
            start = None
    if start is not None:
        out.append((text[start:], start))
    return out


def structural_features(docs):
    """返回 (tokens_per_doc, features: {token: np.array(12)}, df)。"""
    tokens_per_doc: list[list[str]] = []
    attrs: list[list[tuple]] = []  # (cap, sent_init, has_digit, all_alpha, len)
    title_df: dict[str, int] = defaultdict(int)

    for doc in docs:
        full = (doc.get("title", "") + "\n" + doc.get("text", "")) if doc.get("title") else doc.get("text", "")
        terms = bm25_terms(full)
        blocks = raw_blocks(full)
        if len(terms) != len(blocks):
            raise SystemExit(f"分词块数不一致（{len(terms)} vs {len(blocks)}），无法对齐原始形态")
        tokens_per_doc.append(terms)

        local_attrs = []
        prev_boundary = True
        for raw, pos in blocks:
            cap = raw[0].isupper()
            before = full[:pos].rstrip()
            sent_init = prev_boundary or (bool(before) and before[-1] in ".?!")
            local_attrs.append(
                (cap, sent_init, any(c.isdigit() for c in raw), raw.isalpha(), len(raw))
            )
            prev_boundary = False
        attrs.append(local_attrs)

        for t in set(bm25_terms(doc.get("title", ""))):
            title_df[t] += 1

    df: dict[str, int] = defaultdict(int)
    for terms in tokens_per_doc:
        for t in set(terms):
            df[t] += 1

    # 邻居统计
    left_types: dict[str, set] = defaultdict(set)
    right_types: dict[str, set] = defaultdict(set)
    left_count: dict[str, int] = defaultdict(int)
    right_count: dict[str, int] = defaultdict(int)
    left_df_sum: dict[str, float] = defaultdict(float)
    right_df_sum: dict[str, float] = defaultdict(float)

    for terms in tokens_per_doc:
        for i, t in enumerate(terms):
            if i > 0:
                lt = terms[i - 1]
                left_types[t].add(lt)
                left_count[t] += 1
                left_df_sum[t] += math.log(df.get(lt, 1) + 1.0)
            if i + 1 < len(terms):
                rt = terms[i + 1]
                right_types[t].add(rt)
                right_count[t] += 1
                right_df_sum[t] += math.log(df.get(rt, 1) + 1.0)

    # 按 token 汇总原始形态与位置
    cap_n: dict[str, int] = defaultdict(int)
    sent_n: dict[str, int] = defaultdict(int)
    digit_n: dict[str, int] = defaultdict(int)
    alpha_n: dict[str, int] = defaultdict(int)
    len_sum: dict[str, float] = defaultdict(float)
    pos_sum: dict[str, float] = defaultdict(float)
    occ_n: dict[str, int] = defaultdict(int)

    for terms, local_attrs in zip(tokens_per_doc, attrs):
        n = max(len(terms), 1)
        for i, (t, a) in enumerate(zip(terms, local_attrs)):
            cap, sent, digit, alpha, ln = a
            occ_n[t] += 1
            cap_n[t] += cap
            sent_n[t] += sent
            digit_n[t] += digit
            alpha_n[t] += alpha
            len_sum[t] += ln
            pos_sum[t] += i / n

    feats: dict[str, np.ndarray] = {}
    for t, n in occ_n.items():
        d = df[t]
        feats[t] = np.array(
            [
                math.log(d + 1.0),
                cap_n[t] / n,
                sent_n[t] / n,
                title_df.get(t, 0) / d if d else 0.0,
                pos_sum[t] / n,
                len(left_types[t]) / max(left_count[t], 1),
                len(right_types[t]) / max(right_count[t], 1),
                left_df_sum[t] / max(left_count[t], 1),
                right_df_sum[t] / max(right_count[t], 1),
                digit_n[t] / n,
                alpha_n[t] / n,
                len_sum[t] / n,
            ],
            dtype=np.float64,
        )
    return tokens_per_doc, feats, df


def kmeans(X: np.ndarray, k: int, seed: int, restarts: int = 20, iters: int = 100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        # k-means++（增量维护「到最近质心的距离」，避免每步重算全量距离矩阵）
        C = np.empty((k, X.shape[1]))
        C[0] = X[rng.integers(len(X))]
        d2 = ((X - C[0]) ** 2).sum(1)
        for j in range(1, k):
            total = d2.sum()
            p = d2 / total if total > 0 else np.ones(len(X)) / len(X)
            C[j] = X[rng.choice(len(X), p=p)]
            d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
        for _ in range(iters):
            D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = D.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C, atol=1e-8):
                C = newC
                break
            C = newC
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        inertia = D.min(1).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, D.min(1))
    return best[1], best[2]


def build_index(tokens_per_doc):
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    doc_len = []
    total = 0
    for i, terms in enumerate(tokens_per_doc):
        total += len(terms)
        doc_len.append(len(terms))
        local: dict[str, int] = defaultdict(int)
        for t in terms:
            local[t] += 1
        for t, c in local.items():
            postings[t].append((i, float(c)))
    n = len(tokens_per_doc)
    avg = total / n if n else 0.0
    idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}
    return postings, doc_len, avg, idf


def contributions(postings, doc_len, avg, idf, tokens, class_index):
    """把一条查询压成三个等长数组：文档下标 / BM25 贡献 / 该词项的类别下标。

    这样「给定 λ 求分数」只需一次 `bincount`（向量化），而不是按词项循环。
    """
    docs_l, vals_l, cls_l = [], [], []
    for t in tokens:
        plist = postings.get(t)
        if not plist:
            continue
        w = idf[t]
        k = class_index[t]
        for d, tf in plist:
            docs_l.append(d)
            vals_l.append(w * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len[d] / max(avg, 1e-9))))
            cls_l.append(k)
    if not docs_l:
        return None
    return (
        np.array(docs_l, dtype=np.int64),
        np.array(vals_l, dtype=np.float64),
        np.array(cls_l, dtype=np.int64),
    )


def ndcg_at_10(ranking, rel) -> float:
    dcg = sum((2 ** rel[d] - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:10]) if d in rel)
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def bootstrap(deltas, rounds=2000):
    rng = np.random.default_rng(20261111)
    n = len(deltas)
    means = np.array([deltas[rng.integers(0, n, n)].mean() for _ in range(rounds)])
    return deltas.mean(), deltas.std(ddof=1) / math.sqrt(n), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    docs, queries, qrels = load()
    ids = [d["id"] for d in docs]
    outer = [q for q in sorted(qrels) if not is_heldout(q)]
    held = [q for q in sorted(qrels) if is_heldout(q)]
    print(f"docs={len(docs)} outer_train={len(outer)} heldout={len(held)}", file=sys.stderr)

    tokens_per_doc, feats, df = structural_features(docs)
    vocab = sorted(df)
    print(f"vocab={len(vocab)}", file=sys.stderr)

    X = np.stack([feats[t] for t in vocab])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    C, dmin = kmeans(Xz, N_CLASSES, SEED)
    tau = float(np.median(dmin**2)) or 1.0
    D2 = ((Xz[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    logits = -D2 / tau
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(1, keepdims=True)
    hard = P.argmax(1)
    class_of = {t: int(hard[i]) for i, t in enumerate(vocab)}
    soft_of = {t: P[i] for i, t in enumerate(vocab)}
    print(f"k-means inertia={dmin.sum():.1f} tau={tau:.3f}", file=sys.stderr)

    # ---- 类别质量：成员样例 ----
    print("\n=== 16 个结构类（按 df 排序的成员）===")
    for k in range(N_CLASSES):
        members = [t for t in vocab if class_of[t] == k]
        members.sort(key=lambda t: -df[t])
        print(f"  C{k:2d} n={len(members):5d}: {', '.join(members[:15])}")

    # ---- PREREG 决策规则 #1：类别 vs df 桶 的归一化互信息 ----
    log_df = np.log1p(np.array([df[t] for t in vocab], dtype=float))
    order = np.argsort(log_df)
    bucket = np.empty(len(vocab), dtype=int)
    for b, chunk in enumerate(np.array_split(order, N_CLASSES)):
        bucket[chunk] = b
    joint = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(len(vocab)):
        joint[class_of[vocab[i]], bucket[i]] += 1
    joint /= joint.sum()
    pc, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pc[:, None] * pb[None, :])[nz])).sum())
    hc = float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    nmi = mi / ((hc + hb) / 2) if (hc + hb) else 0.0
    print(f"\n类别 vs df 桶：MI={mi:.4f}  H(class)={hc:.4f}  H(bucket)={hb:.4f}  NMI={nmi:.4f}")
    print(f"  决策规则 #1（NMI ≥ 0.5 ⇒ 频率代理）: {'命中 ⇒ 停止' if nmi >= 0.5 else '未命中 ⇒ 继续'}")

    # ---- Stage B：拟合 λ（全程向量化：每条查询一次 bincount）----
    postings, doc_len, avg, idf = build_index(tokens_per_doc)
    n_docs = len(ids)
    # 同分按 doc id 升序（与 runner / Rust bm25 一致）：预计算 id 的字典序名次
    id_rank = np.argsort(np.argsort(np.array(ids)))
    print(f"postings={len(postings)}", file=sys.stderr)

    df_bucket = {t: int(bucket[i]) for i, t in enumerate(vocab)}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_CLASSES)
    random_class = {t: int(perm[class_of[t]]) for t in vocab}
    identity_class = {t: 0 for t in vocab}  # 全部归一类 ⇒ λ 全 1 时就是 bm25

    def prep(qids, class_index):
        rows = []
        for qid in qids:
            terms = sorted(set(bm25_terms(queries[qid])))
            packed = contributions(postings, doc_len, avg, idf, terms, class_index)
            rows.append((qid, packed, qrels[qid]))
        return rows

    def topk(packed, lam, k=100):
        if packed is None:
            return []
        docs, vals, cls = packed
        total = np.bincount(docs, weights=vals * lam[cls], minlength=n_docs)
        cand_n = min(max(500, k), n_docs)
        cand = np.argpartition(-total, cand_n - 1)[:cand_n]
        order = np.lexsort((id_rank[cand], -total[cand]))
        return cand[order][:k]

    def ndcg_map(rows, lam):
        return np.array(
            [ndcg_at_10([ids[d] for d in topk(packed, lam)], rel) for _, packed, rel in rows]
        )

    def fit_lambda(rows, rounds=2):
        lam = np.ones(N_CLASSES)
        best = float(ndcg_map(rows, lam).mean())
        grid = (0.5, 0.75, 1.0, 1.5, 2.0)
        for _ in range(rounds):
            for j in range(N_CLASSES):
                for g in grid:
                    trial = lam.copy()
                    trial[j] = g
                    score = float(ndcg_map(rows, trial).mean())
                    if score > best:
                        best, lam = score, trial
            print(f"    best_outer={best:.5f} lam={np.round(lam, 2).tolist()}", file=sys.stderr)
        return lam, best

    arms = {
        "grammar": class_of,
        "df_bucket": df_bucket,
        "random": random_class,
    }

    base_rows = prep(held, identity_class)
    base_outer_rows = prep(outer, identity_class)
    base_held = ndcg_map(base_rows, np.ones(N_CLASSES))
    base_outer = float(ndcg_map(base_outer_rows, np.ones(N_CLASSES)).mean())
    base_ndcg = float(base_held.mean())
    print(f"\nbaseline heldout nDCG@10={base_ndcg:.5f}（应为 {BASELINE_HELDOUT}）", file=sys.stderr)
    if abs(base_ndcg - BASELINE_HELDOUT) > 1e-5:
        raise SystemExit(f"口径漂了：{base_ndcg:.5f} ≠ {BASELINE_HELDOUT}，拒绝出读数")
    print("口径对照通过", file=sys.stderr)

    print(f"\n{'arm':12s} {'outer':>8s} {'held':>8s} {'Δheld':>9s} {'2SE':>8s} {'95% CI':>24s}  判定")
    print(f"{'bm25':12s} {base_outer:8.5f} {base_ndcg:8.5f} {'—':>9s} {'—':>8s}")
    held_rows_by_arm = {}
    for name, assign in arms.items():
        print(f"\n--- 拟合 {name} λ（{N_CLASSES} 个标量）---", file=sys.stderr)
        held_rows = prep(held, assign)
        outer_rows = prep(outer, assign)
        lam, best = fit_lambda(outer_rows)
        held_scores = ndcg_map(held_rows, lam)
        outer_scores = ndcg_map(outer_rows, lam)
        d = held_scores - base_held
        m, se, lo, hi = bootstrap(d)
        held_rows_by_arm[name] = (lam, held_rows, held_scores)
        flag = "越过 2SE" if abs(m) > 2 * se else "噪声带内"
        print(
            f"{name:12s} {outer_scores.mean():8.5f} {held_scores.mean():8.5f} "
            f"{m:+9.5f} {2*se:8.5f}  [{lo:+.5f}, {hi:+.5f}]  {flag}"
        )

    # PREREG 决策规则 #4：grammar 与 df_bucket 的直接配对对比（参数量相同）
    lam_g, held_g, scores_g = held_rows_by_arm["grammar"]
    lm_d, held_d, scores_d = held_rows_by_arm["df_bucket"]
    d = scores_g - scores_d
    m, se, lo, hi = bootstrap(d)
    print(
        f"\ngrammar − df_bucket：Δ={m:+.5f}  2SE={2*se:.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
        f"  {'越过 2SE' if abs(m) > 2 * se else '噪声带内'}"
    )

    # PREREG §7：干预非零性冒烟（R29）
    changed = 0
    ones = np.ones(N_CLASSES)
    for (_, packed, _), (_, _, _) in zip(held_g, held_g):
        if list(topk(packed, ones, 10)) != list(topk(packed, lam_g, 10)):
            changed += 1
    print(f"干预冒烟：heldout 上 top-10 变化 {changed}/{len(held_g)} = {changed/len(held_g):.1%}（门槛 10%）")

    # PREREG §7：λ 与「功能词程度」是否一致（供人工判断类别是否真像语法类）
    print("\n各类的 λ 与该类平均 log df / 大小写率：")
    for k in range(N_CLASSES):
        members = [t for t in vocab if class_of[t] == k]
        if not members:
            continue
        idxs = [vocab.index(t) for t in members]
        print(
            f"  C{k:2d} λ={lam_g[k]:4.2f}  n={len(members):5d}  "
            f"mean_log_df={np.log1p(np.array([df[t] for t in members], dtype=float)).mean():5.2f}  "
            f"mean_cap={np.array([feats[t][1] for t in members]).mean():4.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
