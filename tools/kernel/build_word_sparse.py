#!/usr/bin/env python3
"""词级神经稀疏工件构建器 (word-sparse-v1)。

motivation：`bilingual-sparse-v1` 消费的是 SPLADE 的 **WordPiece 子词**词表（30k），
与外壳的整词分词（`bm25_terms`）不在同一身份空间，无法与概念因子、字段因子同表相加。

本构建器把同一次离线推理得到的稀疏权重，按"子词 → 整词"聚合到**词级**词表：

- 子词 `##ing` 归到 `ing` 所在词；
- 一词多片（如 `anti` + `##bodies`）取**最大**权重（保守：不放大）；
- 纯符号、特殊 token、数字串丢弃；
- 产出与 `learned-sparse-v1` 相同的工件布局，外壳可直接复用同一个适配器。

用法：`uv run --project tools/kernel python tools/kernel/build_word_sparse.py scifact`
（需要带 torch 的环境；本机用 docseek 的 venv。）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_ID = "word-sparse-v1"
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

# 只保留"整词可检索"的形态：字母（可含数字）串，长度 ≥ 2。
WORD_RE = re.compile(r"^[a-z][a-z0-9]*$")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_piece(piece: str) -> str | None:
    """子词 → 整词：去掉 `##` 前缀、小写、只留字母数字串。"""
    word = piece.lower()
    if word.startswith("##"):
        word = word[2:]
    word = word.strip()
    if len(word) < 2:
        return None
    if not WORD_RE.match(word):
        return None
    return word


def splade_activations(
    model, tokenizer, texts: list[str], *, device, batch_size: int, max_length: int
) -> list[list[tuple[int, float]]]:
    """返回每篇文本的 (子词 id, 权重) 列表（权重 ≥ 0.05）。"""
    out: list[list[tuple[int, float]]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.log1p(torch.relu(outputs.logits))
            mask = inputs["attention_mask"].unsqueeze(-1)
            scores = scores * mask
            reps, _ = torch.max(scores, dim=1)
        del inputs, outputs, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for row in reps.cpu().numpy():
            indices = np.where(row >= 0.05)[0]
            out.append([(int(i), float(row[i])) for i in indices])
    return out


def to_word_level(
    activations: list[tuple[int, float]], tokenizer
) -> dict[str, float]:
    """子词激活 → 词级权重（同词取最大）。"""
    words: dict[str, float] = {}
    for token_id, weight in activations:
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(piece, str):
            continue
        word = normalize_piece(piece)
        if word is None:
            continue
        if weight > words.get(word, 0.0):
            words[word] = weight
    return words


def build(dataset: str, *, batch_size: int = 16) -> int:
    dataset_dir = DATA_DIR / dataset
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    if not corpus_path.is_file():
        raise SystemExit(f"找不到语料 {corpus_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 设备 {device}，模型 {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    params = sum(p.numel() for p in model.parameters())

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_ids.append(doc["id"])
            doc_texts.append((doc.get("title", "") + " " + doc.get("text", "")).strip())

    print(f"[*] 文档 {len(doc_texts)} 篇，子词激活 → 词级聚合")
    doc_acts = splade_activations(
        model, tokenizer, doc_texts, device=device, batch_size=batch_size, max_length=256
    )
    doc_words = [to_word_level(acts, tokenizer) for acts in doc_acts]

    # 词级词表：只收文档侧出现过的词（查询里 OOV 的词直接跳过）。
    vocab = sorted({word for words in doc_words for word in words})
    vocab_index = {word: index for index, word in enumerate(vocab)}
    print(f"[*] 词级词表 {len(vocab)}（子词词表 {len(tokenizer)}）")

    out_dir = dataset_dir / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = out_dir / "sparse.bin"
    with sparse_path.open("wb") as f:
        f.write(b"SLSPARv1")
        f.write(struct.pack("<II", len(doc_ids), len(vocab)))
        for words in doc_words:
            entries = sorted(
                (vocab_index[word], weight) for word, weight in words.items()
            )
            f.write(struct.pack("<I", len(entries)))
            for word_id, weight in entries:
                f.write(struct.pack("<If", word_id, weight))

    ids_path = out_dir / "doc_ids.txt"
    ids_path.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")

    q_map: dict[str, list[list]] = {}
    q_text_map: dict[str, list[list]] = {}
    if queries_path.is_file():
        q_ids: list[str] = []
        q_texts: list[str] = []
        with queries_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                q_ids.append(q["id"])
                q_texts.append(q["text"])
        print(f"[*] 查询 {len(q_texts)} 条")
        q_acts = splade_activations(
            model, tokenizer, q_texts, device=device, batch_size=batch_size, max_length=64
        )
        for qid, text, acts in zip(q_ids, q_texts, q_acts):
            words = to_word_level(acts, tokenizer)
            entries = sorted(
                ([vocab_index[w], weight] for w, weight in words.items() if w in vocab_index),
                key=lambda pair: pair[0],
            )
            entries = [[int(w), float(weight)] for w, weight in entries]
            q_map[qid] = entries
            q_text_map[text] = entries

        (out_dir / "query_sparse.json").write_text(
            json.dumps(q_map, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out_dir / "query_sparse_by_text.json").write_text(
            json.dumps(q_text_map, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    manifest = {
        "id": ARTIFACT_ID,
        "kind": "learned_sparse_vector",
        "weights_origin": f"pretrained:{MODEL_NAME}（子词 → 整词聚合，max）",
        "source": f"data/{dataset}/corpus.jsonl",
        "prior_factor": {
            "name": "doc_word_sparse",
            "kind": "token_confidence_list",
            "shape": [len(doc_ids), len(vocab)],
            "dtype": "f32",
            "count": len(doc_ids),
        },
        "network": {
            "arch": "SPLADE (BERT-base MLM)，离线一次推理后按词聚合",
            "params": params,
            "checkpoint_sha256": "huggingface_cached",
        },
        "counts": {"docs": len(doc_ids), "vocab_size": len(vocab)},
        "files": {
            "sparse.bin": {
                "sha256": sha256_file(sparse_path),
                "bytes": sparse_path.stat().st_size,
            },
            "doc_ids.txt": {
                "sha256": sha256_file(ids_path),
                "bytes": ids_path.stat().st_size,
            },
            "query_sparse.json": {
                "sha256": sha256_file(out_dir / "query_sparse.json"),
                "bytes": (out_dir / "query_sparse.json").stat().st_size,
            },
        },
        "runtime": {"device": "cpu", "precision": "f32"},
        "trained_on": "SPLADE 预训练（MS-MARCO），本工件只做词表对齐",
        "contamination": "与 learned-sparse-v1 同源（MS-MARCO 弱监督）",
        "license": "Apache-2.0",
        "notes": "与外壳整词分词同一身份空间的稀疏词权；用于把最强因子接进主线公式。",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OK → {out_dir}（sparse.bin {sparse_path.stat().st_size/1048576:.2f} MB）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    return build(args.dataset, batch_size=args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
