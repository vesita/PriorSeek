#!/usr/bin/env python3
"""把一个检索训练过的稀疏教师（SPLADE）蒸馏成同构的小模型（单一语种内核）。

思路（借鉴 SPLADE / DistilSPLADE 的做法）：

- 教师：`naver/splade-cocondenser-ensembledistil`（冻结），对学生输入产出
  `log1p(relu(logits))` 的词表稀疏向量作为**软目标**；
- 学生：同架构（MLM + 同一套 log1p(relu) + max-pool），小骨干（层数/隐藏维可配），
  词表与教师一致，所以目标可以直接对齐；
- 损失：对整张词表做 MSE（教师的零权重区域也参与，等价于把"不该激活的词"压下去）；
- 数据：语料文档 + train split 查询（**不使用 test 查询**，避免泄漏）。

产出是一个 `AutoModelForMaskedLM` 目录，可直接交给 `build_learned_sparse.py --model <dir>` 建工件。

用法（本机用带 torch 的 venv）：
    python tools/kernel/train_sparse_student.py scifact --steps 1500 --out /tmp/student-scifact
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TEACHER = "naver/splade-cocondenser-ensembledistil"
STUDENT = "google/bert_uncased_L-4_H-256_A-4"


def sparse_vector(model, tokenizer, texts, *, device, batch_size, max_length):
    """教师/学生共用的稀疏化：log1p(relu(logits))，按 attention mask 做 max-pool。"""
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            scores = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            rep, _ = scores.max(dim=1)
        out.append(rep.cpu())
    return torch.cat(out, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--student", default=STUDENT)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="/tmp/student-sparse")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = DATA_DIR / args.dataset
    docs: list[str] = []
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs.append((d.get("title", "") + " " + d.get("text", "")).strip())
    queries: list[str] = []
    train_queries = dataset_dir / "train_queries.jsonl"
    if train_queries.is_file():
        with train_queries.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line)["text"])
    print(f"[*] 文档 {len(docs)} / train 查询 {len(queries)}（test 查询不参与训练）")

    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForMaskedLM.from_pretrained(TEACHER).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("[*] 预先计算教师软目标 …")
    doc_targets = sparse_vector(
        teacher, tok, docs, device=device, batch_size=2, max_length=args.max_length
    )
    query_targets = (
        sparse_vector(
            teacher, tok, queries, device=device, batch_size=16, max_length=64
        )
        if queries
        else None
    )

    student = AutoModelForMaskedLM.from_pretrained(args.student).to(device)
    student.resize_token_embeddings(len(tok))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student.train()

    texts = docs + queries
    targets = (
        torch.cat([doc_targets, query_targets], dim=0)
        if query_targets is not None
        else doc_targets
    )
    n = len(texts)
    print(f"[*] 训练样本 {n}，步数 {args.steps}，学生 {args.student}")

    step = 0
    while step < args.steps:
        order = list(range(n))
        random.shuffle(order)
        for start in range(0, n, args.batch_size):
            if step >= args.steps:
                break
            idx = order[start : start + args.batch_size]
            batch_texts = [texts[i] for i in idx]
            enc = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            logits = student(**enc).logits
            scores = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            pred, _ = scores.max(dim=1)
            gold = targets[idx].to(device)
            # 直接 MSE 会塌到"全零"（教师向量大部分是 0，全零的 MSE 已经很小）。
            # 改成归一化后的 KL：把概率质量压到教师的活跃词上，才学得到"该激活哪些词"。
            eps = 1e-9
            pred_p = pred / (pred.sum(dim=1, keepdim=True) + eps)
            gold_p = gold / (gold.sum(dim=1, keepdim=True) + eps)
            loss = F.kl_div((pred_p + eps).log(), gold_p, reduction="batchmean")
            if step % 100 == 0:
                mass = pred.sum().item() / max(pred.shape[0], 1)
                active = (pred > 0.05).sum().item() / max(pred.shape[0], 1)
                print(f"    诊断: 每样本质量和 {mass:.1f}，活跃词数 {active:.0f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"  step {step:5d}  mse {loss.item():.5f}")
            step += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"OK 学生模型已保存 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
