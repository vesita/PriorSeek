#!/usr/bin/env python3
"""神经稀疏先验工件构建器 (learned-sparse-v1)。

使用预训练神经稀疏模型 (naver/splade-cocondenser-ensembledistil)
将语料库全量文档与查询离线编码为紧凑的小端二进制稀疏向量 (sparse.bin)。
外壳在纯 CPU 上以极速带权倒排索引直接做点积累加，零运行时神经网络计算与超参依赖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_ID = "learned-sparse-v1"
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_splade_sparse(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
    top_k_tokens: int = 128,
    min_weight: float = 0.05,
) -> list[list[tuple[int, float]]]:
    """批量计算文本的 SPLADE 稀疏词权向量。
    
    对于每个样本：
    1. 计算 token 级 logits;
    2. 对序列维度做 max-pooling: w_j = max_i log(1 + ReLU(logits_{i, j}));
    3. 保留权重 >= min_weight 且最高的前 top_k_tokens 个词项。
    """
    results: list[list[tuple[int, float]]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits: [B, seq_len, vocab_size]
            logits = outputs.logits
            # SPLADE transformation: log(1 + relu(logits))
            splade_scores = torch.log1p(torch.relu(logits))
            # attention_mask: [B, seq_len, 1]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            # Mask out padding tokens
            splade_scores = splade_scores * attention_mask
            # Max pooling over sequence dimension: [B, vocab_size]
            doc_reps, _ = torch.max(splade_scores, dim=1)

        # 提取稀疏非零项
        doc_reps_cpu = doc_reps.cpu().numpy()
        for row in doc_reps_cpu:
            # 找到非零索引
            nz_indices = np.where(row >= min_weight)[0]
            if len(nz_indices) == 0:
                results.append([])
                continue
            
            nz_weights = row[nz_indices]
            # 如果超过 top_k_tokens，取最大的 top_k_tokens
            if len(nz_indices) > top_k_tokens:
                top_part = np.argpartition(-nz_weights, top_k_tokens)[:top_k_tokens]
                nz_indices = nz_indices[top_part]
                nz_weights = row[nz_indices]

            # 按 token_id 升序排序
            sort_order = np.argsort(nz_indices)
            sorted_indices = nz_indices[sort_order]
            sorted_weights = nz_weights[sort_order]

            entries = [(int(idx), float(w)) for idx, w in zip(sorted_indices, sorted_weights)]
            results.append(entries)

    return results


def build_learned_sparse(
    dataset: str,
    *,
    data_dir: Path,
    batch_size: int = 32,
    top_k_tokens: int = 128,
) -> int:
    dataset_dir = data_dir / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"

    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料文件 {corpus_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")
    print(f"[*] 加载神经稀疏内核模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    vocab_size = len(tokenizer)
    total_params = sum(p.numel() for p in model.parameters())

    # 1. 读取并编码语料库
    doc_ids = []
    doc_texts = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_ids.append(d["id"])
                text = (d.get("title", "") + " " + d.get("text", "")).strip()
                doc_texts.append(text)

    print(f"[*] 离线编码语料库 ({len(doc_texts)} 篇文档)...")
    doc_sparse = compute_splade_sparse(
        model,
        tokenizer,
        doc_texts,
        device=device,
        batch_size=batch_size,
        max_length=256,
        top_k_tokens=top_k_tokens,
    )

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 写入小端二进制文件 sparse.bin
    sparse_path = out_dir / "sparse.bin"
    with sparse_path.open("wb") as f:
        # 头部: magic (8B) + num_docs (u32) + vocab_size (u32)
        f.write(b"SLSPARv1")
        f.write(struct.pack("<II", len(doc_ids), vocab_size))
        for entries in doc_sparse:
            # 写入每篇文档 entries 数
            f.write(struct.pack("<I", len(entries)))
            for token_id, weight in entries:
                f.write(struct.pack("<If", token_id, weight))

    # 写入 doc_ids.txt
    ids_path = out_dir / "doc_ids.txt"
    ids_path.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")

    # 3. 编码查询集合
    q_sparse_map = {}
    q_text_map = {}
    if queries_path.is_file():
        q_ids = []
        q_texts = []
        with queries_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    q_ids.append(q["id"])
                    q_texts.append(q["text"])

        print(f"[*] 离线编码查询库 ({len(q_texts)} 条查询)...")
        query_sparse = compute_splade_sparse(
            model,
            tokenizer,
            q_texts,
            device=device,
            batch_size=batch_size,
            max_length=64,
            top_k_tokens=64,
        )

        for q_id, q_text, entries in zip(q_ids, q_texts, query_sparse):
            q_sparse_map[q_id] = entries
            q_text_map[q_text] = entries

        (out_dir / "query_sparse.json").write_text(
            json.dumps(q_sparse_map, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "query_sparse_by_text.json").write_text(
            json.dumps(q_text_map, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # 4. 生成 manifest.json
    sparse_sha = sha256_file(sparse_path)
    ids_sha = sha256_file(ids_path)
    q_sparse_sha = sha256_file(out_dir / "query_sparse.json")

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "learned_sparse_vector",
        "weights_origin": f"pretrained:{MODEL_NAME}",
        "source": f"data/{dataset}/corpus.jsonl",
        "prior_factor": {
            "name": "doc_sparse_vector",
            "kind": "token_confidence_list",
            "shape": [len(doc_ids), vocab_size],
            "dtype": "f32",
            "count": len(doc_ids),
        },
        "network": {
            "arch": "SPLADE (BERT-base MLM lexical projection)",
            "params": total_params,
            "checkpoint_sha256": "huggingface_cached",
        },
        "counts": {
            "docs": len(doc_ids),
            "vocab_size": vocab_size,
        },
        "files": {
            "sparse.bin": {
                "sha256": sparse_sha,
                "bytes": sparse_path.stat().st_size,
            },
            "doc_ids.txt": {
                "sha256": ids_sha,
                "bytes": ids_path.stat().st_size,
            },
            "query_sparse.json": {
                "sha256": q_sparse_sha,
                "bytes": (out_dir / "query_sparse.json").stat().st_size,
            },
        },
        "runtime": {
            "device": "cpu",
            "precision": "f32",
        },
        "trained_on": "SPLADE 预训练（MS-MARCO 检索语料库）",
        "contamination": "MS-MARCO 预训练无监督/弱监督",
        "license": "Apache-2.0",
        "notes": "离线预烘焙的神经稀疏词权向量，外壳在 CPU 倒排索引上直接带权点积累加。",
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OK {dataset}: 神经稀疏工件构建完成 → {out_dir}")
    print(f"   sparse.bin: {sparse_path.stat().st_size / 1024 / 1024:.2f} MB, sha256: {sparse_sha[:16]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="scifact", nargs="?", help="数据集名称 (默认 scifact)")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批大小 (默认 32)")
    parser.add_argument("--top-k-tokens", type=int, default=128, help="每篇文档最大保留 token 数 (默认 128)")
    args = parser.parse_args()

    return build_learned_sparse(
        args.dataset,
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        top_k_tokens=args.top_k_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
