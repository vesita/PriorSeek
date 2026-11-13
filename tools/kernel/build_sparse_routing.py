#!/usr/bin/env python3
"""神经编译稀疏索引构建器 (sparse-route-v1)。

将文档级语义扩展项 (Document Expansion) 离线烘焙为工件，
供 Rust 外壳在构建 PositionalIndex 时直接编译进倒排表。
线上检索 100% 纯 CPU，零实时神经网络计算。
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
ARTIFACT_ID = "sparse-route-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_sparse_routing(
    dataset: str,
    *,
    data_dir: Path,
    top_k_per_doc: int = 16,
    min_confidence: float = 0.25,
) -> int:
    dataset_dir = data_dir / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料文件 {corpus_path}")

    # 读取语料
    docs = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    # 读取已有关联先验（seq-v0 PPMI 边）
    seq_dir = dataset_dir / "artifacts" / "seq-v0"
    if not (seq_dir / "manifest.json").is_file():
        raise SystemExit(f"找不到 {seq_dir}（先跑 build_seq.py {dataset}）")

    from build_seq import read_artifact
    seq = read_artifact(seq_dir)
    vocab = seq.vocab
    vocab_inv = {word: idx for idx, word in enumerate(vocab)}
    edges = seq.edges

    # 聚合每个词的最佳出边
    best_edges: dict[int, dict[int, float]] = {}
    for (src, dst, order), weight in edges.items():
        if weight <= 0.0:
            continue
        slot = best_edges.setdefault(src, {})
        if weight > slot.get(dst, 0.0):
            slot[dst] = weight

    # 读取 Saliency 先验因子以对核心词进行加权
    saliency_path = dataset_dir / "artifacts" / "saliency-v0" / "saliency.json"
    saliency_map = {}
    if saliency_path.is_file():
        saliency_data = json.loads(saliency_path.read_text(encoding="utf-8"))
        for term, rows in saliency_data.get("entries", {}).items():
            if rows and len(rows[0]) > 1:
                saliency_map[term] = float(rows[0][1])

    # 为每篇文档生成 Top-K 个高置信度语义扩展词
    entries: dict[str, list[list]] = {}
    total_pairs = 0

    for doc in docs:
        doc_id = doc["id"]
        full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        doc_tokens = set(bm25_terms(full_text))

        # 候选语义扩展池：neighbor -> cumulative_score
        candidate_scores: dict[str, float] = {}

        for token in doc_tokens:
            token_saliency = saliency_map.get(token, 0.5)
            # 只有具有一定判别力的词才触发语义扩散
            if token_saliency < 0.2:
                continue

            src_idx = vocab_inv.get(token)
            if src_idx is None or src_idx not in best_edges:
                continue

            neighbors = best_edges[src_idx]
            max_w = max(neighbors.values())
            if max_w <= 0.0:
                continue

            for dst_idx, w in neighbors.items():
                neighbor_word = vocab[dst_idx]
                # 排除文档内已有的词，只扩展新语义词项
                if neighbor_word in doc_tokens:
                    continue
                # 置信度 = 出边强度 * 源词 Saliency
                norm_w = (w / max_w) * token_saliency
                if norm_w >= min_confidence:
                    candidate_scores[neighbor_word] = candidate_scores.get(neighbor_word, 0.0) + norm_w

        if not candidate_scores:
            continue

        # 按得分排序，截取 Top-K
        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: (-x[1], x[0]),
        )[:top_k_per_doc]

        # 归一化到 (0, 1]
        max_c = ranked_candidates[0][1] if ranked_candidates else 1.0
        doc_entries = [
            [word, round(score / max_c, 4)]
            for word, score in ranked_candidates
        ]
        entries[doc_id] = doc_entries
        total_pairs += len(doc_entries)

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha256 = sha256_bytes(payload)
    (out_dir / "doc_expansions.json").write_bytes(payload)

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "doc_expansion",
        "weights_origin": "corpus_semantic_routing",
        "source": f"data/{dataset}/corpus.jsonl + seq-v0 + saliency-v0",
        "prior_factor": {
            "name": "doc_expansion",
            "kind": "token_confidence_list",
            "shape": ["n_docs", top_k_per_doc],
            "dtype": "f32",
            "count": total_pairs,
        },
        "network": {
            "arch": "saliency_routed_graph_projection",
            "params": 0,
            "checkpoint_sha256": None,
        },
        "counts": {
            "docs": len(entries),
            "pairs": total_pairs,
        },
        "files": {
            "doc_expansions.json": {
                "sha256": payload_sha256,
                "bytes": len(payload),
            }
        },
        "runtime": {
            "device": "cpu",
            "precision": "f32",
        },
        "trained_on": "语料内语义关联图谱与 Saliency 门控（无监督，未使用任何 qrels）",
        "contamination": "无外部预训练污染",
        "license": "由本仓从数据集语料构建",
        "notes": "产出每篇文档的语义路由扩展词项与置信度权重，用于离线编译进倒排索引。",
    }

    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")

    print(f"OK {dataset}: {len(entries)} 篇文档已生成语义路由扩展（共 {total_pairs} 条扩展项）")
    print(f"   工件目录: {out_dir}")
    print(f"   doc_expansions.json sha256: {payload_sha256[:16]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-conf", type=float, default=0.25)
    args = parser.parse_args()
    raise SystemExit(
        build_sparse_routing(
            args.dataset,
            data_dir=DATA_DIR,
            top_k_per_doc=args.top_k,
            min_confidence=args.min_conf,
        )
    )
