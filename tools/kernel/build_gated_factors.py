#!/usr/bin/env python3
"""自适应门控扩展与 Token 筛选内核工件构建器 (gated-exp-v1)。

严格遵循 search-lab 的架构契约：
1. 内核只产出先验因子（Token Saliency 与语义关联候选及转移置信度），不包含 alpha / top_k 等外壳用法超参。
2. 离线产出二进制 / JSON 工件，在 CPU 上以只读稀疏方式被检索外壳消费。
3. 不读取 test split 的 qrels，保证无污染。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "kernel"))
from tokenizer import bm25_terms  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACT_ID = "gated-exp-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compute_token_saliency(corpus_docs: list[dict]) -> dict[str, float]:
    """
    计算 Token 的先验判别力（Saliency）：
    结合文档频率 (IDF) 与集中度 (Peak/Sum ratio)。
    高频且分散在所有文档的词 (泛化词/虚词) 赋予低 Saliency；
    特定领域专有名词赋予高 Saliency。
    """
    n_docs = len(corpus_docs)
    df: dict[str, int] = {}
    tf_sum: dict[str, int] = {}
    tf_peak: dict[str, int] = {}

    for doc in corpus_docs:
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        terms = bm25_terms(full_text)
        doc_counts: dict[str, int] = {}
        for t in terms:
            doc_counts[t] = doc_counts.get(t, 0) + 1
        for t, c in doc_counts.items():
            df[t] = df.get(t, 0) + 1
            tf_sum[t] = tf_sum.get(t, 0) + c
            if c > tf_peak.get(t, 0):
                tf_peak[t] = c

    saliency: dict[str, float] = {}
    for t, doc_freq in df.items():
        if doc_freq <= 1:
            continue
        idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        concentration = tf_peak[t] / max(tf_sum[t], 1)
        # Saliency 归一化到 (0, 1]
        score = 1.0 / (1.0 + math.exp(-0.5 * (idf * concentration - 2.0)))
        saliency[t] = round(score, 4)

    return saliency


def build(
    dataset: str,
    *,
    data_dir: Path,
    top_k: int = 3,
    min_saliency: float = 0.3,
) -> int:
    dataset_dir = data_dir / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料文件 {corpus_path}")

    docs = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    # 1. 计算 Token Saliency 因子
    saliency_map = compute_token_saliency(docs)

    # 2. 从已有的 seq-v0 PPMI / 语义关联中提取候选，只对高 Saliency 词保留扩散
    seq_dir = dataset_dir / "artifacts" / "seq-v0"
    entries: dict[str, list[list]] = {}
    total_pairs = 0

    if (seq_dir / "manifest.json").is_file():
        from build_seq import read_artifact
        seq = read_artifact(seq_dir)
        vocab = seq.vocab
        edges = seq.edges

        best_edges: dict[int, dict[int, float]] = {}
        for (src, dst, order), weight in edges.items():
            if weight <= 0.0:
                continue
            slot = best_edges.setdefault(src, {})
            if weight > slot.get(dst, 0.0):
                slot[dst] = weight

        for src, neighbors in best_edges.items():
            src_term = vocab[src]
            src_saliency = saliency_map.get(src_term, 0.0)
            # 只有具有足够判别力的核心词才允许扩展，杜绝泛词扩散导致查询漂移
            if src_saliency < min_saliency:
                continue

            max_w = max(neighbors.values())
            if max_w <= 0.0:
                continue
            ranked = sorted(
                ((dst, w) for dst, w in neighbors.items()),
                key=lambda row: (-row[1], row[0]),
            )[:top_k]

            entries[src_term] = [
                [vocab[dst], round((w / max_w) * src_saliency, 4)]
                for dst, w in ranked
            ]
            total_pairs += len(ranked)

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写入 expansion.json（与现有 TokenConfidenceList 工件同构）
    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha256 = sha256_bytes(payload)
    (out_dir / "expansion.json").write_bytes(payload)

    # 写入 manifest.json，严格契约
    manifest = {
        "id": ARTIFACT_ID,
        "kind": "query_expansion",
        "weights_origin": "corpus_saliency_ppmi",
        "source": f"data/{dataset}/corpus.jsonl + seq-v0",
        "prior_factor": {
            "name": "gated_expansion",
            "kind": "token_confidence_list",
            "shape": ["n_query_tokens", top_k],
            "dtype": "f32",
            "count": total_pairs,
        },
        "network": {
            "arch": "saliency_gated_projection",
            "params": 0,
            "checkpoint_sha256": None,
        },
        "counts": {
            "source_tokens": len(entries),
            "pairs": total_pairs,
        },
        "files": {
            "expansion.json": {
                "sha256": payload_sha256,
                "bytes": len(payload),
            }
        },
        "runtime": {
            "device": "cpu",
            "precision": "f32",
        },
        "trained_on": "语料内统计与 Saliency 门控（无监督，未使用 qrels）",
        "contamination": "无外部预训练污染",
        "license": "由本仓从数据集语料构建",
        "notes": "Token 置信度结合了源词的 Saliency 因子与出边转移强度，供外壳在低置信度下自适应触发。",
    }

    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")

    print(f"OK: 工件已生成至 {out_dir}")
    print(f"    源词数: {len(entries)}, 扩展对数: {total_pairs}, sha256: {payload_sha256[:16]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    args = parser.parse_args()
    raise SystemExit(build(args.dataset, data_dir=DATA_DIR))
