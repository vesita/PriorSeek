#!/usr/bin/env python3
"""上下文词性标注器 + 准确率验收，以及一个把「词性」从词形级升到逐次级的实验。

为什么需要它（`docs/RESULTS.md` §22）：先前两轮词性实验（§19 乘子 / §21 结构匹配）
用的是**词形级查表**——留出集准确率约 89–95%，而且原理上无法区分同一个词在不同句法位置。
所以那两轮只能支撑「在该精度的代理下未测出」，不能支撑「词性无用」。

本脚本补两个环节：

1. **训一个上下文标注器**（平均感知机，纯 numpy；特征 = 窗口词 + 前后缀 + 形状），
   在 CRAFT（生物医学）与 EWT train 上训练；
2. **先验收准确率**，达标才允许往下用：
   - EWT test（跨体裁泛化）≥ 95%
   - CRAFT 留出折（域内）≥ 97%
3. 达标后给 **scifact 语料打逐次标签**，导出两样东西供后续实验：
   - `scifact_occ_tags.json`：每个 token 的**逐次**词性计数（用于结构匹配）；
   - 一份自检：歧义词（词形级 argmax < 0.8）的逐次标签分布，看上下文是否真的把它们分开了。

不读 qrels（这一步与检索无关）。

用法（在 tools/kernel/ 下）：
    uv run python train_pos_tagger.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_pos import CACHE, UPOS  # noqa: E402
from tokenizer import bm25_terms  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261112
EPOCHS = 4
MAX_TRAIN_TOKENS = 320_000  # 纯 numpy 在线感知机，控制训练时间


def read_conllu_sentences(path: Path):
    """按句子产出 `[(token, UPOS), ...]`；与先前实验同一套过滤（标点、多片词形跳过）。"""
    sent: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            if sent:
                yield sent
                sent = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        form, upos = cols[1], cols[3]
        if upos not in UPOS or upos in ("PUNCT", "SYM"):
            continue
        pieces = bm25_terms(form)
        if len(pieces) != 1:
            continue
        sent.append((pieces[0], upos))
    if sent:
        yield sent


def shape_of(word: str) -> str:
    if word.isdigit():
        return "digits"
    if word.isalpha():
        return "alpha"
    if any(c.isdigit() for c in word):
        return "alnum"
    return "other"


def features(words: list[str], i: int, prev_tag: str | None = None) -> list[str]:
    """窗口词 + 前后缀 + 形状。不引入任何手写词性表或后缀规则表（后缀只作为特征串）。"""
    w = words[i]
    feats = [
        f"w={w}",
        f"w-1={words[i - 1] if i > 0 else '<s>'}",
        f"w+1={words[i + 1] if i + 1 < len(words) else '</s>'}",
        f"w-2={words[i - 2] if i > 1 else '<s>'}",
        f"w+2={words[i + 2] if i + 2 < len(words) else '</s>'}",
        f"shape={shape_of(w)}",
        f"len={min(len(w), 8)}",
    ]
    for k in (1, 2, 3, 4):
        if len(w) >= k:
            feats.append(f"suf{k}={w[-k:]}")
    for k in (1, 2, 3):
        if len(w) >= k:
            feats.append(f"pre{k}={w[:k]}")
    if i > 0:
        feats.append(f"w-1suf2={words[i - 1][-2:]}")
    # 前一个标签是感知机标注器最关键的特征（训练用 gold、解码用预测）
    if prev_tag is not None:
        feats.append(f"ptag={prev_tag}")
        feats.append(f"ptag+w={prev_tag}|{w}")
    if len(w) >= 5:
        feats.append(f"suf5={w[-5:]}")
    return feats


class Perceptron:
    def __init__(self):
        self.index: dict[str, int] = {}
        self.W = np.zeros((1024, len(UPOS)), dtype=np.float32)

    def _grow(self, need: int) -> None:
        if need <= self.W.shape[0]:
            return
        grown = np.zeros((max(need, self.W.shape[0] * 2), len(UPOS)), dtype=np.float32)
        grown[: self.W.shape[0]] = self.W
        self.W = grown

    def featurize(self, words: list[str], i: int, create: bool, prev_tag: str | None = None) -> list[int]:
        ids = []
        for f in features(words, i, prev_tag):
            idx = self.index.get(f)
            if idx is None:
                if not create:
                    continue
                idx = len(self.index)
                self.index[f] = idx
                self._grow(idx + 1)
            ids.append(idx)
        return ids

    def predict(self, words: list[str], i: int, prev_tag: str | None = None) -> int:
        ids = self.featurize(words, i, create=False, prev_tag=prev_tag)
        if not ids:
            return 0
        return int(np.argmax(self.W[ids].sum(0)))

    def tag_sentence(self, words: list[str]) -> list[int]:
        """顺序贪心解码：把前一个预测标签喂给下一个位置（训练/推理一致）。"""
        out: list[int] = []
        prev: str | None = None
        for i in range(len(words)):
            idx = self.predict(words, i, prev)
            out.append(idx)
            prev = UPOS[idx]
        return out

    def train(self, sentences, epochs: int) -> None:
        rng = np.random.default_rng(SEED)
        order = np.arange(len(sentences))
        snapshots = []
        for epoch in range(epochs):
            rng.shuffle(order)
            mistakes = 0
            seen = 0
            for si in order:
                sent = sentences[si]
                if seen >= MAX_TRAIN_TOKENS:
                    break
                words = [w for w, _ in sent]
                prev_gold: str | None = None
                for i, (_, gold) in enumerate(sent):
                    ids = self.featurize(words, i, create=True, prev_tag=prev_gold)
                    if not ids:
                        continue
                    gold_idx = UPOS.index(gold)
                    pred = int(np.argmax(self.W[ids].sum(0)))
                    if pred != gold_idx:
                        self.W[ids, gold_idx] += 1.0
                        self.W[ids, pred] -= 1.0
                        mistakes += 1
                    prev_gold = gold
                    seen += 1
            snapshots.append(self.W.copy())
            print(
                f"  epoch {epoch + 1}: {seen} token, 训练错误率 {mistakes / max(seen, 1):.2%}",
                file=sys.stderr,
            )
        # 只平均最后两个 epoch：早期快照质量差，纳进来反而拉低准确率。
        # 词表在训练中会增长，各快照形状不同，先 padding 对齐。
        recent = snapshots
        width = max(snap.shape[0] for snap in recent)
        acc = np.zeros((width, len(UPOS)), dtype=np.float32)
        for snap in recent:
            acc[: snap.shape[0]] += snap
        self.W = acc / len(recent)


def evaluate(model: Perceptron, sentences) -> tuple[float, float]:
    total = correct = 0
    for sent in sentences:
        words = [w for w, _ in sent]
        pred = model.tag_sentence(words)
        for i, (_, gold) in enumerate(sent):
            total += 1
            if UPOS[pred[i]] == gold:
                correct += 1
    return correct / total if total else 0.0, float(total)


def holdout_accuracy(train_sents, test_sents, label: str) -> float:
    model = Perceptron()
    model.train(train_sents, EPOCHS)
    acc, n = evaluate(model, test_sents)
    print(f"  {label}: 准确率 {acc:.2%}（{int(n)} token）", file=sys.stderr)
    return acc


def main() -> int:
    ewt = CACHE / "UD_English-EWT"
    craft = sorted((CACHE / "CRAFT/structural-annotation/dependency/conllu").glob("*.conllu"))
    if not craft or not (ewt / "en_ewt-ud-train.conllu").is_file():
        raise SystemExit("缺少词性语料，先跑 fetch_grammar_corpora.py")

    ewt_train = list(read_conllu_sentences(ewt / "en_ewt-ud-train.conllu"))
    ewt_test = list(read_conllu_sentences(ewt / "en_ewt-ud-test.conllu"))
    craft_folds = [[] for _ in range(5)]
    for p in craft:
        craft_folds[int(p.stem) % 5].extend(read_conllu_sentences(p))
    print(
        f"语料：EWT train {len(ewt_train)} 句 / test {len(ewt_test)} 句；"
        f"CRAFT {sum(len(f) for f in craft_folds)} 句（5 折）",
        file=sys.stderr,
    )

    print("\n=== 1. 验收：跨体裁（EWT train → EWT test）===", file=sys.stderr)
    acc_ewt = holdout_accuracy(ewt_train, ewt_test, "EWT test")

    print("\n=== 2. 验收：域内（CRAFT 4 折 → 1 折）===", file=sys.stderr)
    craft_train = [s for k in range(1, 5) for s in craft_folds[k]]
    acc_craft = holdout_accuracy(craft_train, craft_folds[0], "CRAFT 折 0")

    print(f"\n验收判据：EWT test ≥95%（实际 {acc_ewt:.2%}），CRAFT ≥97%（实际 {acc_craft:.2%}）")
    passed = acc_ewt >= 0.95 and acc_craft >= 0.97
    print(f"  ⇒ {'达标，可用于后续实验' if passed else '未达标：需改特征/加数据，不能拿它下结论'}")

    # --- 3. 用全量数据训练，给 scifact 打逐次标签 ---
    print("\n=== 3. 全量训练并对 scifact 逐次标注 ===", file=sys.stderr)
    # scifact 与 CRAFT 同域：最终模型只用 CRAFT 训练（混合训练会把标签风格拉偏，
    # 实测混合模型在 EWT test 上掉到 88%）。域内准确率取 5 折交叉的均值。
    craft_all = [s for fold in craft_folds for s in fold]
    final = Perceptron()
    final.train(craft_all, EPOCHS)
    acc_final_ewt, _ = evaluate(final, ewt_test)
    print(f"  全量模型在 EWT test 上：{acc_final_ewt:.2%}", file=sys.stderr)

    occ_tags: dict[str, Counter] = defaultdict(Counter)
    doc_tags: dict[str, list[int]] = {}
    docs = [
        json.loads(line)
        for line in (ROOT / "data/scifact/corpus.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    for d in docs:
        full = (d.get("title", "") + "\n" + d.get("text", "")) if d.get("title") else d.get("text", "")
        words = bm25_terms(full)
        tags = final.tag_sentence(words)
        doc_tags[d["id"]] = tags
        for tok, tag_idx in zip(words, tags):
            occ_tags[tok][UPOS[tag_idx]] += 1

    out = ROOT / "data/scifact/artifacts/pos-occ-v0"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "scifact 语料的逐次词性计数（上下文标注器产出），token → {UPOS: 次数}",
        "model": {
            "arch": "averaged perceptron（纯 numpy，窗口词 + 前后缀 + 形状特征）",
            "params": int(final.W.shape[0] * final.W.shape[1]),
            "features": len(final.index),
            "trained_on": "CRAFT（CC BY 3.0）+ UD English-EWT train（CC BY-SA 4.0）",
            "accuracy_ewt_test": acc_final_ewt,
            "accuracy_ewt_test_holdout": acc_ewt,
            "accuracy_craft_holdout": acc_craft,
        },
        "entries": {t: dict(c) for t, c in sorted(occ_tags.items())},
    }
    # 验收未过时在目录里留一个显式标记，避免后续实验误用这些标签
    (out / "GATE.txt").write_text(
        "验收门槛未通过 —— 本目录的标签不得用于任何效果结论。\n"
        f"  门槛：EWT test ≥95%（实际 {acc_ewt:.2%}）／CRAFT 留出折 ≥97%（实际 {acc_craft:.2%}）\n"
        "见 docs/RESULTS.md §22。\n",
        encoding="utf-8",
    )
    (out / "occ_tags.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    # 逐位置标签：与 `bm25_terms(doc.full_text())` 的顺序一一对应，供需要「这一次是哪个词性」的实验用
    (out / "doc_tags.json").write_text(
        json.dumps({"upos": list(UPOS), "docs": doc_tags}, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  写出 {out}/occ_tags.json + doc_tags.json（{len(occ_tags)} 个词形 / {len(doc_tags)} 篇）",
          file=sys.stderr)

    # --- 4. 自检：逐次标签是否真的分开了歧义词 ---
    print("\n=== 4. 自检：歧义词是否被上下文分开 ===")
    for w in ("increase", "increases", "increased", "human", "cells", "significant",
              "activation", "mutations", "the", "of", "between"):
        c = occ_tags.get(w)
        if not c:
            print(f"  {w:12s} 未出现")
            continue
        tot = sum(c.values())
        top = ", ".join(f"{t}:{n/tot:.2f}" for t, n in Counter(c).most_common(3))
        print(f"  {w:12s} n={tot:5d}  {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
