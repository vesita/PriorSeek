#!/usr/bin/env python3
"""把记忆事实烤成 dsh-pod 记忆召回可直接消费的「文档侧稀疏词权」工件。

dsh-pod 的服务端没有推理环境，所以稀疏因子只能以**离线工件**的形式接入：
本脚本用检索训练过的稀疏模型给每条事实打分，再把权重投影到 dsh-pod 自己的分词空间
（jieba 词 + 小写字母数字串），产出 `{"<fact-id>": [["<token>", 1.23], ...]}`。

为什么要"投影"：模型的词表是子词（WordPiece），而 dsh-pod 的分词是整词/整串。
两边对不齐时，直接存模型词表会在服务端全部查不到。这里按字符串匹配把子词权重
汇总到整词上（`##ing` 这类续接片段并入前一个词），是一种**有损但有据可查**的近似。

用法：
    python tools/kernel/build_memory_sparse.py facts.jsonl -o memory-sparse.json
    # facts.jsonl 每行 {"id": "...", "text": "..."}；dsh-pod 侧用
    # DSH_POD_SPARSE_PRIOR=/path/to/memory-sparse.json 打开影子路径。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
CJK = re.compile(r"[\u4e00-\u9fff]")


def memory_tokens(text: str) -> list[str]:
    """与 dsh-pod `retrieval::tokens` 对齐的粗分词：CJK 连续串按字，非 CJK 字母数字串整体小写。

    注意：dsh-pod 对 CJK 用 jieba 切词，这里不引 jieba（工具侧保持零额外依赖），
    因此按「字」切；投影时子词权重会落到单字上，服务端整词命中时取不到。
    需要整词粒度时用 `--jieba`（装了 jieba 才生效）。
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if CJK.match(ch):
            flush()
            out.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush()
    flush()
    return out


def tokenize_words(text: str) -> list[str]:
    """jieba 整词切分（`--jieba` 时使用），非 CJK 串保持整体。"""
    import jieba  # 延迟导入：不装 jieba 时仍可用按字模式

    out: list[str] = []
    for word in jieba.cut(text):
        word = word.strip()
        if not word:
            continue
        if CJK.match(word):
            out.append(word)
        else:
            out.extend(re.findall(r"[a-z0-9]+", word.lower()))
    return out


def build_vocab_index(tokenizer) -> dict[str, list[int]]:
    """词表字符串 -> 词表 id 列表（同一字符串可能对应多个 id）。"""
    vocab: dict[str, list[int]] = {}
    for piece, idx in tokenizer.get_vocab().items():
        vocab.setdefault(piece.lower(), []).append(idx)
    return vocab


def project(weights: torch.Tensor, pieces: list[str], words: list[str]) -> dict[str, float]:
    """把子词权重投影到整词：整词精确匹配优先，其次前缀/续接片段累加。"""
    scores: dict[str, float] = {}
    piece_weights = {p.lower().lstrip("▁##"): float(w) for p, w in zip(pieces, weights) if w > 0}
    for word in words:
        key = word.lower()
        value = piece_weights.get(key)
        if value is None:
            # 续接片段（##xxx / ▁xxx）与前缀匹配
            parts = [w for p, w in piece_weights.items() if p and (p == key or key.startswith(p))]
            value = sum(parts) if parts else 0.0
        if value > 0.0:
            scores[key] = max(scores.get(key, 0.0), value)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("facts", help="事实 JSONL：每行 {\"id\", \"text\"}")
    parser.add_argument("-o", "--out", default="memory-sparse.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-terms", type=int, default=128, help="每条事实最多保留多少词")
    parser.add_argument("--jieba", action="store_true", help="用 jieba 整词切分（需安装 jieba）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    facts = []
    with Path(args.facts).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                facts.append((str(row["id"]), str(row.get("text", ""))))
    if not facts:
        print("没有事实可处理", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()
    split = tokenize_words if args.jieba else memory_tokens

    artifact: dict[str, list[list[object]]] = {}
    for start in range(0, len(facts), args.batch_size):
        batch = facts[start : start + args.batch_size]
        enc = tokenizer(
            [text for _, text in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            scores = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            rep = scores.max(dim=1).values.cpu()
        for (fact_id, text), row in zip(batch, rep):
            top = row.topk(min(args.top_terms, row.numel()))
            ids = top.indices.tolist()
            values = top.values.tolist()
            pieces = [tokenizer.convert_ids_to_tokens(i) for i in ids]
            projected = project(
                torch.tensor(values), pieces, split(text)
            )
            terms = sorted(projected.items(), key=lambda kv: -kv[1])[: args.top_terms]
            artifact[fact_id] = [[token, round(weight, 6)] for token, weight in terms]
        if start % (args.batch_size * 20) == 0:
            print(f"  {start}/{len(facts)}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    covered = sum(1 for terms in artifact.values() if terms)
    print(f"OK {len(artifact)} 条事实（{covered} 条有词权）→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
