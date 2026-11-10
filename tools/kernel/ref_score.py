#!/usr/bin/env python3
"""seq 内核 v0 的 Python 参考打分器（按 `docs/KERNEL.md` §8.1，与 Rust `seq` 适配器同口径）。

用途：给 Rust 适配器对拍。输入工件目录 + 查询，输出 top-k 的 doc id 与分数（JSON 到 stdout）。

用法（在 `tools/kernel/` 下）：

    uv run python ref_score.py scifact --limit 20 --k 10
    uv run python ref_score.py scifact --qids 3,5 --k 5 --lambda-seq 0
    uv run python ref_score.py scifact --query "the role of protein" --k 10
    uv run python ref_score.py scifact --orders 1 --unordered --window 4

打分（与 Rust `crates/sl-adapters/src/seq.rs` 逐条对齐）：

    tokenize(q) -> [t1..tn]（tokenizer.bm25_terms）
    s_uni(d) = Σ_{t ∈ 去重查询词, idf>0} idf(t) · tf·(k1+1) / (tf + k1·(1−b+b·len/avg_len))
    P        = 阶 ∈ orders 的查询位置对 (q_i, q_{i+step})，按 (a,b,order) 去重
    e(a,b)   = Σ_{order ∈ orders} 工件边 (a,b,order) 的权重（查不到为 0；零权边从不落盘）
    s_seq(d) = Σ_{e(a,b)>0} e(a,b) · (1 + ln(min(c_d(a,b,window), max_matches)))
    score    = s_uni(d) + lambda_seq · s_seq(d)

`c_d` 的匹配计数与 Rust 完全一致：ordered=true 要求 a 的位置 < b 的位置且距离 ≤ window；
ordered=false 时按无序邻近（|pa−pb| ≤ window 且位置不同），a==b 时有序/无序都按
「不同出现位置对」计。候选文档 = 至少含一个查询词项的文档（= Rust 的 touched），
时序分只在候选文档上算。输出只保留分数 > 0 的文档，按分数降序、doc id 升序。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from build_seq import DEFAULT_DATA_DIR, ARTIFACT_ID, Artifact, BuildError, read_artifact
from tokenizer import bm25_terms

# BM25 词频饱和 / 长度归一参数，与 Rust `seq.rs` 的 K1 / B 一致。
K1 = 1.5
B = 0.75


@dataclass(frozen=True)
class ScoreParams:
    """消融参数（与 Rust `SeqParams` 默认值一致，全部进输出 JSON 便于复核）。"""

    lambda_seq: float = 0.5
    window: int = 8
    orders: tuple[int, ...] = (1, 2)
    ordered: bool = True
    max_matches: int = 3
    w_order1: float = 1.0
    w_order2: float = 1.0

    def to_json(self) -> dict:
        return {
            "lambda_seq": self.lambda_seq,
            "window": self.window,
            "orders": list(self.orders),
            "ordered": self.ordered,
            "max_matches": self.max_matches,
            "w_order1": self.w_order1,
            "w_order2": self.w_order2,
            "k1": K1,
            "b": B,
        }


def _intersect(sorted_a: array, sorted_b: array) -> list[int]:
    """两个升序 int 数组的交集（文档序号升序）。"""
    result: list[int] = []
    i = j = 0
    len_a, len_b = len(sorted_a), len(sorted_b)
    while i < len_a and j < len_b:
        a, b = sorted_a[i], sorted_b[j]
        if a == b:
            result.append(a)
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return result


@dataclass
class RefScorer:
    """参考打分器：加载一次工件，多次查询。"""

    artifact: Artifact
    params: ScoreParams = field(default_factory=ScoreParams)
    _term_docs: list[array | None] = field(init=False, repr=False)
    _term_tfs: list[array | None] = field(init=False, repr=False)
    _doc_len: list[int] = field(init=False, repr=False)
    _avg_len: float = field(init=False, repr=False)
    _vocab_index: dict[str, int] = field(init=False, repr=False)
    _edges: dict[tuple[int, int, int], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        artifact = self.artifact
        vocab = artifact.vocab
        self._vocab_index = {token: idx for idx, token in enumerate(vocab)}
        self._doc_len = [len(tokens) for tokens in artifact.doc_tokens]
        total = sum(self._doc_len)
        # 与 Rust 一致：avg_len 兜底 1e-9，避免空语料除零。
        self._avg_len = max(total / len(self._doc_len), 1e-9) if self._doc_len else 1e-9
        self._edges = artifact.edges

        # 倒排表：term id -> 文档序号（升序）+ 词频。惰性创建，避免几十万个空 list。
        self._term_docs = [None] * len(vocab)
        self._term_tfs = [None] * len(vocab)
        for doc_idx, tokens in enumerate(artifact.doc_tokens):
            tf: dict[int, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                docs = self._term_docs[token]
                if docs is None:
                    docs = array("I")
                    self._term_docs[token] = docs
                    self._term_tfs[token] = array("I")
                docs.append(doc_idx)
                self._term_tfs[token].append(count)

    # ------------------------------------------------------------------ 查询

    def all_scores(self, query: str) -> dict[str, float]:
        """返回**全部**非零分文档的 `{doc_id: score}`（不截断、不排序）。

        训练器用它取「单字项 / 各阶时序项」的分量：分别用 λ=0、λ=1+orders=(1,)、λ=1+orders=(2,)
        三次调用相减即得 (s_uni, s1, s2)，因此分量与正式打分走**同一段代码**，不会分叉。
        """
        return {doc_id: score for doc_id, score in self._raw_scores(query).items()}

    def _raw_scores(self, query: str) -> dict[str, float]:
        """内部：未截断的文档分数字典（键是 doc id）。"""
        return dict(self._score_docs(query))

    def _score_docs(self, query: str) -> dict[str, float]:
        """按 §8.1 计算 `{doc_id: score}`，不排序不截断。"""
        if not self.artifact.doc_tokens:
            return {}
        raw_terms = bm25_terms(query)
        if not raw_terms:
            return {}
        term_ids: list[int] = []
        seen: set[int] = set()
        for term in raw_terms:
            token = self._vocab_index.get(term)
            if token is not None and token not in seen:
                seen.add(token)
                term_ids.append(token)
        if not term_ids:
            return {}

        scores: dict[int, float] = {}
        for token in term_ids:
            idf = self.artifact.idf[token]
            if idf <= 0.0:
                continue
            docs = self._term_docs[token]
            if docs is None:
                continue
            tfs = self._term_tfs[token]
            assert tfs is not None
            for doc_idx, tf in zip(docs, tfs):
                doc_len = float(self._doc_len[doc_idx])
                denom = tf + K1 * (1.0 - B + B * doc_len / self._avg_len)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (K1 + 1.0) / denom

        if self.params.lambda_seq > 0.0:
            seq_scores = self._sequence_scores(raw_terms)
            for doc_idx, bonus in seq_scores.items():
                scores[doc_idx] = scores.get(doc_idx, 0.0) + self.params.lambda_seq * bonus

        return {
            self.artifact.doc_ids[doc_idx]: score
            for doc_idx, score in scores.items()
            if score > 0.0
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """返回 [(doc_id, score)]，按分数降序、doc id 升序，最多 k 条。"""
        if k <= 0 or not self.artifact.doc_tokens:
            return []
        raw_terms = bm25_terms(query)
        if not raw_terms:
            return []

        # 单字项：按查询出现顺序去重（与 Rust 的 term_ids 构造一致），OOV 跳过。
        term_ids: list[int] = []
        seen: set[int] = set()
        for term in raw_terms:
            token = self._vocab_index.get(term)
            if token is not None and token not in seen:
                seen.add(token)
                term_ids.append(token)
        if not term_ids:
            return []

        scores: dict[int, float] = {}
        for token in term_ids:
            idf = self.artifact.idf[token]
            if idf <= 0.0:
                continue
            docs = self._term_docs[token]
            if docs is None:
                continue
            tfs = self._term_tfs[token]
            assert tfs is not None
            for doc_idx, tf in zip(docs, tfs):
                doc_len = float(self._doc_len[doc_idx])
                denom = tf + K1 * (1.0 - B + B * doc_len / self._avg_len)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (K1 + 1.0) / denom

        if self.params.lambda_seq > 0.0:
            seq_scores = self._sequence_scores(raw_terms)
            for doc_idx, bonus in seq_scores.items():
                # 候选集合与上一步相同（含时序对的文档必然含查询词项），这里仍做兜底。
                scores[doc_idx] = scores.get(doc_idx, 0.0) + self.params.lambda_seq * bonus

        ranking = [(self.artifact.doc_ids[doc_idx], score) for doc_idx, score in scores.items() if score > 0.0]
        ranking.sort(key=lambda row: (-row[1], row[0]))
        return ranking[:k]

    def _sequence_scores(self, raw_terms: list[str]) -> dict[int, float]:
        """每篇候选文档的 s_seq（未乘 lambda_seq）。"""
        # 查询 token 序列：保留重复与 OOV 过滤（与 Rust sequence 一致）。
        sequence = [token for token in (self._vocab_index.get(term) for term in raw_terms) if token is not None]

        # 位置对 (a,b,order) 去重；按阶分别乘以该阶的权重系数（w=1 时与旧版逐位等价：
        # sat(c) 只依赖 (a,b) 与 window，不依赖阶）。
        pair_weights: dict[tuple[int, int, int], float] = {}
        seen_triples: set[tuple[int, int, int]] = set()
        involved: set[int] = set()
        for order in self.params.orders:
            step = int(order)
            if len(sequence) <= step:
                continue
            for idx in range(len(sequence) - step):
                a, b = sequence[idx], sequence[idx + step]
                triple = (a, b, order)
                if triple in seen_triples:
                    continue
                seen_triples.add(triple)
                weight = self._edges.get(triple)
                if not weight:
                    continue  # 查不到或零权边（Rust 侧只加载 weight>0 的边）
                factor = self.params.w_order1 if order == 1 else self.params.w_order2
                pair_weights[(a, b, order)] = pair_weights.get((a, b, order), 0.0) + weight * factor
                involved.add(a)
                involved.add(b)
        if not pair_weights:
            return {}

        # 候选文档：每个 (a,b) 的倒排交集。
        pair_docs: list[tuple[tuple[int, int], list[int]]] = []
        candidates: set[int] = set()
        for pair in pair_weights:
            docs_a = self._term_docs[pair[0]]
            docs_b = self._term_docs[pair[1]]
            if docs_a is None or docs_b is None:
                continue
            common = _intersect(docs_a, docs_b)
            if common:
                pair_docs.append((pair, common))
                candidates.update(common)
        if not candidates:
            return {}

        window = self.params.window
        max_matches = self.params.max_matches
        ordered = self.params.ordered
        result: dict[int, float] = {}
        for doc_idx in sorted(candidates):
            positions: dict[int, list[int]] = {}
            for pos, token in enumerate(self.artifact.doc_tokens[doc_idx]):
                if token in involved:
                    positions.setdefault(token, []).append(pos)
            if not positions:
                continue
            seq_score = 0.0
            for (a, b, _order), weight in pair_weights.items():
                pos_a = positions.get(a)
                pos_b = positions.get(b)
                if not pos_a or not pos_b:
                    continue
                matches = 0
                if ordered:
                    # a 在 b 之前，距离 ≤ window；pos_b 升序，用二分数区间。
                    for pa in pos_a:
                        matches += bisect_right(pos_b, pa + window) - bisect_right(pos_b, pa)
                        if matches >= max_matches:
                            break
                elif a != b:
                    for pa in pos_a:
                        matches += bisect_right(pos_b, pa + window) - bisect_left(pos_b, pa - window)
                        if matches >= max_matches:
                            break
                else:
                    # a == b：与 Rust 的双重循环一致 —— 每个位置数窗口内的其它位置，
                    # 每对会被两端各数一次（Rust 的 pa != pb 不会排除自己所在的这一对
                    # 在对方视角下的重复计数）。
                    for pa in pos_a:
                        matches += bisect_right(pos_b, pa + window) - bisect_left(pos_b, pa - window) - 1
                        if matches >= max_matches:
                            break
                capped = min(matches, max_matches)
                if capped > 0:
                    seq_score += weight * (1.0 + math.log(capped))
            result[doc_idx] = seq_score
        return result


# --------------------------------------------------------------------------- CLI


def load_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ref_score.py",
        description="seq 内核 v0 参考打分：输出 top-k（doc id + 分数）JSON 到 stdout",
    )
    parser.add_argument("dataset", help="数据集名（默认从 data/<name>/ 定位工件与查询）")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=f"工件目录（默认 <data-dir>/<dataset>/artifacts/{ARTIFACT_ID}）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"规范数据根目录（默认 {DEFAULT_DATA_DIR}）"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="查询文件 queries.jsonl（默认 <data-dir>/<dataset>/queries.jsonl）",
    )
    parser.add_argument("--qids", default="", help="逗号分隔的查询 id（默认全部 / 与 --limit 组合）")
    parser.add_argument("--limit", type=int, default=0, help="只跑查询文件里的前 N 条（0 = 全部）")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="直接给查询文本（可重复；给了它就不用 --queries/--qids）",
    )
    parser.add_argument("--k", type=int, default=10, help="返回 top-k（默认 10）")
    parser.add_argument("--lambda-seq", type=float, default=0.5, help="时序分权重（0 = 纯 BM25，默认 0.5）")
    parser.add_argument("--window", type=int, default=8, help="有序匹配最大 token 距离（默认 8）")
    parser.add_argument("--orders", default="1,2", help="使用的边阶，逗号分隔（默认 1,2）")
    parser.add_argument(
        "--unordered", action="store_true", help="关闭顺序要求，改为无序邻近（同窗口）对照"
    )
    parser.add_argument("--max-matches", type=int, default=3, help="匹配次数上限（默认 3）")
    parser.add_argument("--w-order1", type=float, default=1.0, help="order=1 边权系数（默认 1.0）")
    parser.add_argument("--w-order2", type=float, default=1.0, help="order=2 边权系数（默认 1.0）")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    artifact_dir = (args.artifact or data_dir / args.dataset / "artifacts" / ARTIFACT_ID).resolve()
    queries_path = (args.queries or data_dir / args.dataset / "queries.jsonl").resolve()

    try:
        orders = tuple(int(part) for part in args.orders.split(",") if part.strip())
    except ValueError:
        print(f"--orders 不是整数列表：{args.orders!r}", file=sys.stderr)
        return 2
    if not orders or any(order not in (1, 2) for order in orders):
        print(f"--orders 只接受 1 / 2 的组合，收到 {orders}", file=sys.stderr)
        return 2

    params = ScoreParams(
        lambda_seq=args.lambda_seq,
        window=args.window,
        orders=orders,
        ordered=not args.unordered,
        max_matches=args.max_matches,
        w_order1=args.w_order1,
        w_order2=args.w_order2,
    )

    started = time.monotonic()
    try:
        artifact = read_artifact(artifact_dir)
    except BuildError as exc:
        print(f"加载工件失败：{exc}", file=sys.stderr)
        return 1
    load_seconds = time.monotonic() - started

    if args.query:
        queries = [{"id": f"adhoc-{index}", "text": text} for index, text in enumerate(args.query, 1)]
    else:
        try:
            queries = load_queries(queries_path)
        except OSError as exc:
            print(f"读查询文件失败：{exc}", file=sys.stderr)
            return 1
        if args.qids:
            wanted = [qid.strip() for qid in args.qids.split(",") if qid.strip()]
            by_id = {row["id"]: row for row in queries}
            missing = [qid for qid in wanted if qid not in by_id]
            if missing:
                print(f"查询文件里没有这些 qid：{missing[:5]}", file=sys.stderr)
                return 1
            queries = [by_id[qid] for qid in wanted]
        elif args.limit > 0:
            queries = queries[: args.limit]

    started = time.monotonic()
    scorer = RefScorer(artifact, params)
    rows = []
    for row in queries:
        hits = scorer.search(row["text"], args.k)
        rows.append(
            {
                "qid": row["id"],
                "query": row["text"],
                "top": [{"doc_id": doc_id, "score": score} for doc_id, score in hits],
            }
        )
    score_seconds = time.monotonic() - started

    output = {
        "dataset": args.dataset,
        "artifact_dir": str(artifact_dir),
        "artifact_fingerprint": artifact.manifest.get("source_corpus", {}).get("fingerprint"),
        "params": params.to_json(),
        "queries": rows,
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    print(
        f"加载工件 {load_seconds * 1000:.0f}ms，打分 {len(rows)} 条查询 {score_seconds * 1000:.0f}ms"
        f"（{score_seconds / max(len(rows), 1) * 1000:.2f}ms/查询）",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
