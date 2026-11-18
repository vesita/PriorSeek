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
DEFAULT_ARTIFACT_ID = "learned-sparse-v1"
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


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
    batch_size: int = 8,
    max_length: int = 256,
    top_k_tokens: int = 128,
    min_weight: float = 0.05,
    doc_transform: str = "none",
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
            if doc_transform == "sqrt":
                row = np.sqrt(row)
            elif doc_transform == "log1p":
                row = np.log1p(row)
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
    model_name: str = DEFAULT_MODEL,
    artifact_id: str = DEFAULT_ARTIFACT_ID,
    batch_size: int = 8,
    top_k_tokens: int = 128,
    max_length: int = 256,
    doc_limit: int = 0,
    device_kind: str = "auto",
    title_repeat: int = 1,
    title_weight: float = 0.0,
    doc_transform: str = "none",
) -> int:
    dataset_dir = data_dir / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"

    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料文件 {corpus_path}")

    device = torch.device(
        "cpu" if device_kind == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] 运行设备: {device}")
    print(f"[*] 加载神经稀疏内核模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    vocab_size = len(tokenizer)
    total_params = sum(p.numel() for p in model.parameters())

    # 1. 读取并编码语料库
    doc_ids = []
    doc_texts = []
    doc_titles = []
    doc_bodies = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_ids.append(d["id"])
                doc_titles.append(d.get("title", "").strip())
                doc_bodies.append(d.get("text", "").strip())
                text = (d.get("title", "") + " " + d.get("text", "")).strip()
                doc_texts.append(text)

    if doc_limit > 0:
        doc_ids = doc_ids[:doc_limit]
        doc_texts = doc_texts[:doc_limit]
    print(f"[*] 离线编码语料库 ({len(doc_texts)} 篇文档, max_length={max_length})...")
    if title_weight > 0.0:
        # 分字段编码：标题与正文各自编码，文档向量 = 正文 + λ·标题。
        # 注意：SPLADE 用 max-pool，单纯重复标题文本**不会**改变权重（已实测无差异），
        # 必须分开编码再叠加才等价于字段加权。
        titles = doc_titles
        bodies = doc_bodies
        body_sparse = compute_splade_sparse(
            model, tokenizer, bodies, device=device, batch_size=batch_size,
            max_length=max_length, top_k_tokens=top_k_tokens,
        )
        title_sparse = compute_splade_sparse(
            model, tokenizer, titles, device=device, batch_size=batch_size,
            max_length=64, top_k_tokens=top_k_tokens,
        )
        doc_sparse = []
        for body, title in zip(body_sparse, title_sparse):
            merged = {}
            for token, weight in body:
                merged[token] = merged.get(token, 0.0) + weight
            for token, weight in title:
                merged[token] = merged.get(token, 0.0) + title_weight * weight
            items = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k_tokens]
            doc_sparse.append(sorted(items))
    else:
        doc_sparse = compute_splade_sparse(
            model,
            tokenizer,
            doc_texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            top_k_tokens=top_k_tokens,
            doc_transform=doc_transform,
        )

    out_dir = dataset_dir / "artifacts" / artifact_id
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
        "id": artifact_id,
        "kind": "learned_sparse_vector",
        "weights_origin": f"pretrained:{model_name}",
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
    parser.add_argument("--max-length", type=int, default=256, help="文档截断长度 (默认 256)")
    parser.add_argument("--doc-limit", type=int, default=0, help="只编码前 N 篇（0 = 全部，仅用于计时）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="产因子模型（默认 SPLADE）")
    parser.add_argument("--artifact-id", default=DEFAULT_ARTIFACT_ID, help="工件目录名")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="推理设备")
    parser.add_argument("--title-repeat", type=int, default=1, help="标题重复次数（1 = 不加权）")
    parser.add_argument("--title-weight", type=float, default=0.0, help="标题字段权重 λ（>0 时分开编码再叠加）")
    parser.add_argument(
        "--doc-transform",
        default="none",
        choices=["none", "sqrt", "log1p"],
        help="文档侧权重变换（决定高权重词项的压制程度）",
    )
    args = parser.parse_args()

    return build_learned_sparse(
        args.dataset,
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        top_k_tokens=args.top_k_tokens,
        max_length=args.max_length,
        doc_limit=args.doc_limit,
        model_name=args.model,
        artifact_id=args.artifact_id,
        device_kind=args.device,
        title_repeat=args.title_repeat,
        title_weight=args.title_weight,
        doc_transform=args.doc_transform,
    )


if __name__ == "__main__":
    sys.exit(main())
