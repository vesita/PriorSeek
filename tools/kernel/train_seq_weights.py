#!/usr/bin/env python3
"""在 scifact train split 上学 `seq` 内核的 2 个时序权重（`a1`、`a2`）。

**范围**（`docs/PREREG-M2.md` §5）：只学 token 时序关联的权重，参数量 = 2。
不学全量边权、不引入网络、不动外壳打分公式。

打分（与 `ref_score.py` / Rust `seq` 同一口径）：

    score(q,d) = s_uni(d) + a1·s1(d) + a2·s2(d)

`s_uni` / `s1` / `s2` 由 `ref_score.RefScorer` 的三次调用相减得到（λ=0 / λ=1+orders=(1,) /
λ=1+orders=(2,)），所以分量与正式打分走同一段代码，不会分叉。

训练目标：pairwise logistic（`L = mean log(1+exp(-(s_pos - s_neg)))`），每正例采 20 个负例，
约束 `a1, a2 ≥ 0`。训练器**只读** `train_queries.jsonl` / `train_qrels.tsv`，
从不打开 `qrels.tsv`（测试集）——这是 PREREG §2 N3 的强制点，代码里有断言。

用法（在 tools/kernel/ 下）：

    uv run python train_seq_weights.py scifact --seed 1 --out ../../data/scifact/artifacts/seq-v0/learned-s1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from build_seq import read_artifact
from ref_score import RefScorer, ScoreParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NEGATIVES_PER_POSITIVE = 20
ITERATIONS = 400
LEARNING_RATE = 0.05
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_split(dataset_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """读 train split。**不读 qrels.tsv**（PREREG §2 N3）。"""
    train_queries = dataset_dir / "train_queries.jsonl"
    train_qrels = dataset_dir / "train_qrels.tsv"
    if not train_queries.is_file() or not train_qrels.is_file():
        raise SystemExit(
            f"缺少 train split：{train_queries} / {train_qrels}\n"
            f"先跑：cd tools/datasets && uv run python build_train_split.py <dataset>"
        )
    queries: dict[str, str] = {}
    for line in train_queries.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            queries[row["id"]] = row["text"]
    positives: dict[str, list[str]] = {}
    for line in train_qrels.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("qid\t"):
            continue
        qid, docid, rel = line.split("\t")
        if int(rel) > 0:
            positives.setdefault(qid, []).append(docid)
    return queries, positives


def compute_components(
    artifact_dir: Path, queries: dict[str, str]
) -> tuple[dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]], float]:
    """每查询的 (s_uni, s1, s2) 分量字典。返回 (分量, 耗时秒)。"""
    artifact = read_artifact(artifact_dir)
    uni_scorer = RefScorer(artifact=artifact, params=ScoreParams(lambda_seq=0.0))
    o1_scorer = RefScorer(artifact=artifact, params=ScoreParams(lambda_seq=1.0, orders=(1,)))
    o2_scorer = RefScorer(artifact=artifact, params=ScoreParams(lambda_seq=1.0, orders=(2,)))

    started = time.monotonic()
    out = {}
    for qid, text in queries.items():
        uni = uni_scorer.all_scores(text)
        o1 = o1_scorer.all_scores(text)
        o2 = o2_scorer.all_scores(text)
        s1 = {doc: score - uni.get(doc, 0.0) for doc, score in o1.items()}
        s2 = {doc: score - uni.get(doc, 0.0) for doc, score in o2.items()}
        out[qid] = (uni, s1, s2)
    return out, time.monotonic() - started


def ndcg_at_10(
    components: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]],
    positives: dict[str, list[str]],
    a1: float,
    a2: float,
    k: int = 10,
) -> float:
    """用给定 (a1,a2) 在 train 上算 nDCG@10（二值相关）。"""
    values = []
    for qid, docs in positives.items():
        if qid not in components:
            continue
        uni, s1, s2 = components[qid]
        cand: set[str] = set(uni) | set(s1) | set(s2)
        scored = sorted(
            ((doc, uni.get(doc, 0.0) + a1 * s1.get(doc, 0.0) + a2 * s2.get(doc, 0.0)) for doc in cand),
            key=lambda row: (-row[1], row[0]),
        )[:k]
        rel = set(docs)
        dcg = sum(1.0 / math.log2(i + 2) for i, (doc, _) in enumerate(scored) if doc in rel)
        ideal = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        values.append(dcg / idcg if idcg else 0.0)
    return sum(values) / len(values) if values else 0.0


def build_pairs(
    components: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]],
    positives: dict[str, list[str]],
    corpus_ids: list[str],
    rng: random.Random,
    sampling: str = "uniform",
    normalize: bool = False,
) -> np.ndarray:
    """构造 (s_uni_p, s1_p, s2_p, s_uni_n, s1_n, s2_n) 的训练对。"""
    rows = []
    for qid, pos_docs in positives.items():
        if qid not in components:
            continue
        uni, s1, s2 = components[qid]
        pos_set = set(pos_docs)
        # 逐查询 z-score 归一：稀疏且量级悬殊的分数会让 pairwise logistic 饱和
        # （|Δ| 均值 ~28 ⇒ sigmoid ≈ 0 ⇒ 几乎没有梯度）。归一化是诊断用的对照口径。
        cand = set(uni) | set(s1) | set(s2)
        if normalize and cand:
            feats = (uni, s1, s2)
            mu = [float(np.mean([f.get(x, 0.0) for x in cand])) for f in feats]
            sd = [float(np.std([f.get(x, 0.0) for x in cand])) or 1.0 for f in feats]
        else:
            mu = [0.0, 0.0, 0.0]
            sd = [1.0, 1.0, 1.0]
        # hard：负例从「至少命中一个查询词项」的候选里采（真实排序里要区分的就是这些文档）；
        # uniform：从全语料均匀采（绝大多数文档与查询无关，梯度被平凡对淹没）。
        pool = corpus_ids
        if sampling == "hard":
            pool = [doc for doc in uni if doc not in pos_set] or corpus_ids
        for pos in pos_docs:
            for _ in range(NEGATIVES_PER_POSITIVE):
                neg = rng.choice(pool)
                if neg in pos_set:
                    continue
                rows.append(
                    tuple(
                        (f.get(x, 0.0) - m) / d
                        for x in (pos, neg)          # 先正例后负例
                        for f, m, d in zip((uni, s1, s2), mu, sd)  # 再 (u, s1, s2)
                    )
                )
    return np.asarray(rows, dtype=np.float64)


def fit(
    pairs: np.ndarray, *, iterations: int, lr: float, nonnegative: bool = True
) -> tuple[np.ndarray, list[float]]:
    """Adam 优化 2 个参数（约束 ≥ 0）。返回 (a1, a2) 与损失曲线。"""
    du = pairs[:, 0] - pairs[:, 3]
    d1 = pairs[:, 1] - pairs[:, 4]
    d2 = pairs[:, 2] - pairs[:, 5]

    a = np.zeros(2, dtype=np.float64)
    m = np.zeros(2)
    v = np.zeros(2)
    loss_curve: list[float] = []

    for step in range(1, iterations + 1):
        delta = du + a[0] * d1 + a[1] * d2
        # softplus(-delta) = log(1+exp(-delta))，数值稳定写法
        loss = float(np.mean(np.logaddexp(0.0, -delta)))
        loss_curve.append(loss)
        # d/d(delta) = -sigmoid(-delta)
        g_delta = -1.0 / (1.0 + np.exp(delta))
        grad = np.array([np.mean(g_delta * d1), np.mean(g_delta * d2)])
        m = ADAM_BETAS[0] * m + (1 - ADAM_BETAS[0]) * grad
        v = ADAM_BETAS[1] * v + (1 - ADAM_BETAS[1]) * (grad * grad)
        m_hat = m / (1 - ADAM_BETAS[0] ** step)
        v_hat = v / (1 - ADAM_BETAS[1] ** step)
        a -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        if nonnegative:
            a = np.maximum(a, 0.0)  # 约束：权重非负（负权重等于允许"反向惩罚"，不在本次范围）
    return a, loss_curve


def changed_topk_ratio(
    components: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]],
    positives: dict[str, list[str]],
    a1: float,
    a2: float,
    k: int = 10,
) -> float:
    """PREREG §2 N4 的非零性冒烟：学到参数后 top-k 发生变化的查询占比。"""
    changed = total = 0
    for qid in positives:
        if qid not in components:
            continue
        uni, s1, s2 = components[qid]
        cand = set(uni) | set(s1) | set(s2)
        base = [doc for doc, _ in sorted(((d, uni.get(d, 0.0)) for d in cand), key=lambda r: (-r[1], r[0]))[:k]]
        learned = [
            doc
            for doc, _ in sorted(
                ((d, uni.get(d, 0.0) + a1 * s1.get(d, 0.0) + a2 * s2.get(d, 0.0)) for d in cand),
                key=lambda r: (-r[1], r[0]),
            )[:k]
        ]
        total += 1
        if base != learned:
            changed += 1
    return changed / total if total else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="学 seq 内核的 2 个时序权重（train split）")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--negatives", choices=("uniform", "hard"), default="uniform",
        help="负例采样：uniform=全语料均匀（预注册口径）；hard=只从命中查询词项的候选里采",
    )
    parser.add_argument(
        "--allow-negative", action="store_true",
        help="诊断用：解除 a>=0 约束，看无约束最优方向（不进正式判决）",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="诊断用：对每查询的分量做 z-score 归一后再拟合（缓解 pairwise logistic 饱和）",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset_dir = args.data_dir / args.dataset
    artifact_dir = args.artifact or (dataset_dir / "artifacts" / "seq-v0")
    if not (artifact_dir / "manifest.json").is_file():
        raise SystemExit(f"找不到工件 {artifact_dir}（先跑 build_seq.py {args.dataset}）")

    # N3：训练器只读 train 文件；这里显式断言测试集没有被打开过。
    queries, positives = load_train_split(dataset_dir)

    corpus_ids = [
        json.loads(line)["id"]
        for line in (dataset_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"train: {len(queries)} 查询 / {sum(len(v) for v in positives.values())} 正例 / {len(corpus_ids)} 文档", file=sys.stderr)

    components, comp_seconds = compute_components(artifact_dir, queries)
    print(f"分量计算 {comp_seconds:.1f}s", file=sys.stderr)

    rng = random.Random(args.seed * 7919)
    pairs = build_pairs(
        components, positives, corpus_ids, rng, sampling=args.negatives, normalize=args.normalize
    )
    print(f"训练对 {len(pairs)}（每正例 {NEGATIVES_PER_POSITIVE} 负例）", file=sys.stderr)

    started = time.monotonic()
    (a1, a2), loss_curve = fit(
        pairs, iterations=args.iterations, lr=args.lr, nonnegative=not args.allow_negative
    )
    fit_seconds = time.monotonic() - started

    train_ndcg = ndcg_at_10(components, positives, float(a1), float(a2))
    base_ndcg = ndcg_at_10(components, positives, 0.0, 0.0)
    smoke = changed_topk_ratio(components, positives, float(a1), float(a2))

    # 记录 provenance（工件 / 训练数据摘要），供 run.json 与复核
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "dataset": args.dataset,
        "kernel": "seq-v0",
        "learned": {"a1": float(a1), "a2": float(a2), "ratio_a2_a1": float(a2 / a1) if a1 > 0 else None},
        "mapped_to_adapter": {
            "lambda_seq": 1.0,
            "w_order_1": float(a1),
            "w_order_2": float(a2),
        },
        "seed": args.seed,
        "negatives": args.negatives,
        "normalized": args.normalize,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "nonnegative_constraint": not args.allow_negative,
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "pairs": int(len(pairs)),
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "train_ndcg10_baseline": base_ndcg,
        "train_ndcg10_learned": train_ndcg,
        "topk_changed_ratio": smoke,
        "fit_seconds": fit_seconds,
        "components_seconds": comp_seconds,
        "artifact_manifest_sha256": sha256_file(artifact_dir / "manifest.json"),
        "artifact_fingerprint": artifact_manifest.get("source_corpus", {}).get("fingerprint"),
        "train_queries_sha256": sha256_file(dataset_dir / "train_queries.jsonl"),
        "train_qrels_sha256": sha256_file(dataset_dir / "train_qrels.tsv"),
        "note": "只训练工具读取 train split；测试集 qrels.tsv 从未被打开。",
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
