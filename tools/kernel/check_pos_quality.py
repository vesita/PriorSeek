#!/usr/bin/env python3
"""先回答一个前置问题：我们用的词性标签本身准不准？

`docs/RESULTS.md` §19 / §21 直接拿「词形 → argmax P(UPOS)」当因子去测排名，
但**从没验证过这张表本身对不对**。如果它只有 85% 准，那零结果的含义是
「在这个精度的代理下没测出」，而不是「词性无用」。本脚本补这一步。

三件事：

1. **留出评估**：用 EWT train 建表 → 在 EWT dev / test 上测准确率；
   CRAFT 做 5 折交叉（按文件名确定性分折），测生物医学域内的准确率。
2. **歧义统计**：词形级最高概率的分布——多少词形是真的有歧义（不是干脆的单一词性）。
3. **在 scifact 上的可用性**：按置信度分桶，看 scifact 的 **token 次**有多少落在
   「高置信」「歧义」「未覆盖」里。这一步决定因子在目标语料上到底有多少可用信息。

注意：这里评估的是**词形查表**（类型级贪心），不是上下文标注器。
上下文标注器的准确率通常显著更高（同等语料上多 5–10 个点），本脚本能给出的只是
「我们实际用过的那个代理」的上界。

用法（在 tools/kernel/ 下）：
    uv run python check_pos_quality.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_pos import CACHE, UPOS, read_conllu  # noqa: E402


def build_table(paths: list[Path]) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        for tok, tag in read_conllu(path):
            table[tok][tag] += 1
    return table


def argmax_tag(counter: Counter) -> str:
    return max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]


def evaluate(table: dict[str, Counter], paths: list[Path], fallback: str | None):
    """返回 (覆盖 token 数, 总 token 数, 正确数, 覆盖内正确数)。"""
    covered = correct_cov = total = correct_fb = 0
    for path in paths:
        for tok, gold in read_conllu(path):
            total += 1
            pred_cov = argmax_tag(table[tok]) if tok in table else None
            pred = pred_cov if pred_cov is not None else fallback
            if pred_cov is not None:
                covered += 1
                if pred_cov == gold:
                    correct_cov += 1
            if pred is not None and pred == gold:
                correct_fb += 1
    return covered, total, correct_fb, correct_cov


def ambiguity(table: dict[str, Counter]) -> tuple[np.ndarray, np.ndarray]:
    types = list(table)
    top = np.array([max(c.values()) / sum(c.values()) for c in table.values()])
    return top, np.array([sum(c.values()) for c in table.values()])


def main() -> int:
    ewt = CACHE / "UD_English-EWT"
    craft = sorted((CACHE / "CRAFT/structural-annotation/dependency/conllu").glob("*.conllu"))
    if not craft or not (ewt / "en_ewt-ud-train.conllu").is_file():
        raise SystemExit("缺少词性语料，先跑 fetch_grammar_corpora.py")

    train, dev, test = (ewt / f"en_ewt-ud-{s}.conllu" for s in ("train", "dev", "test"))

    print("=== 1. 留出评估：EWT train 建表，在 dev / test 上测 ===")
    table_ewt = build_table([train])
    print(f"  表大小 {len(table_ewt)} 个词形")
    for name, paths in (("EWT dev", [dev]), ("EWT test", [test])):
        cov, tot, cor_fb, cor_cov = evaluate(table_ewt, paths, None)
        print(
            f"  {name:9s} 覆盖 {cov/tot:6.1%}  覆盖内准确率 {cor_cov/cov:6.2%}  "
            f"（{cor_cov}/{cov}）"
        )

    print("\n=== 2. CRAFT 5 折交叉（生物医学域内，按文件名确定性分折）===")
    folds: list[list[Path]] = [[] for _ in range(5)]
    for p in craft:
        folds[int(p.stem) % 5].append(p)
    accs, covs = [], []
    for k in range(5):
        held = folds[k]
        rest = [p for j, fs in enumerate(folds) if j != k for p in fs]
        table = build_table(rest)
        cov, tot, _, cor_cov = evaluate(table, held, None)
        accs.append(cor_cov / cov if cov else 0.0)
        covs.append(cov / tot)
    print(
        f"  5 折：覆盖 {np.mean(covs):6.1%}（范围 {min(covs):.1%}–{max(covs):.1%}）  "
        f"覆盖内准确率 {np.mean(accs):6.2%}（范围 {min(accs):.2%}–{max(accs):.2%}）"
    )

    print("\n=== 3. 歧义统计（词形级最高概率）===")
    top, freq = ambiguity(table_ewt)
    for thr in (0.5, 0.7, 0.9, 0.99):
        share_types = float((top < thr).mean())
        share_tokens = float(freq[top < thr].sum() / freq.sum())
        print(f"  最高概率 < {thr:<4} 的词形占比 {share_types:6.1%}，按出现次数 {share_tokens:6.1%}")

    print("\n=== 4. 在 scifact 上的可用性（决定因子有多少信息）===")
    # 与实测一致：CRAFT + EWT train 合并建表（实测用的就是这两份）
    table_all = build_table(craft + [train])
    dist: dict[str, np.ndarray] = {}
    for tok, c in table_all.items():
        v = np.zeros(len(UPOS))
        total_c = sum(c.values())
        for tag, n in c.items():
            v[UPOS.index(tag)] = n / total_c
        dist[tok] = v

    conf: dict[str, float] = {t: float(v.max()) for t, v in dist.items()}
    scif_occ: Counter = Counter()
    scif_types: Counter = Counter()
    for line in (Path(__file__).resolve().parents[2] / "data/scifact/corpus.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        full = (d.get("title", "") + "\n" + d.get("text", "")) if d.get("title") else d.get("text", "")
        from tokenizer import bm25_terms  # 局部导入，避免脚本顶部依赖顺序

        toks = bm25_terms(full)
        scif_occ.update(toks)
        scif_types.update(set(toks))

    tot_occ = sum(scif_occ.values())
    buckets = {
        "高置信 (≥0.95)": lambda c: c >= 0.95,
        "较置信 (0.80–0.95)": lambda c: 0.80 <= c < 0.95,
        "歧义 (<0.80)": lambda c: c < 0.80,
    }
    assigned = sum(v for t, v in scif_occ.items() if t in conf)
    print(f"  scifact token 次总数 {tot_occ}，被表覆盖 {assigned}（{assigned/tot_occ:.1%}）")
    for label, pred in buckets.items():
        occ = sum(v for t, v in scif_occ.items() if t in conf and pred(conf[t]))
        typ = sum(1 for t in scif_types if t in conf and pred(conf[t]))
        print(f"  {label:20s} token 次 {occ:8d}（{occ/tot_occ:6.1%}）  词形 {typ:6d}")
    uncovered = tot_occ - assigned
    print(f"  {'未覆盖':20s} token 次 {uncovered:8d}（{uncovered/tot_occ:6.1%}）")

    print("\n=== 5. 结论判据 ===")
    cov_test, tot_test, _, cor_test = evaluate(table_ewt, [test], None)
    acc = cor_test / cov_test
    occ_conf = sum(v for t, v in scif_occ.items() if t in conf and conf[t] >= 0.95) / tot_occ
    print(f"  词形查表在 EWT test 上：覆盖 {cov_test/tot_test:.1%}，准确率 {acc:.2%}")
    print(f"  scifact 上高置信覆盖：{occ_conf:.1%} 的 token 次")
    if acc >= 0.90 and occ_conf >= 0.60:
        print("  ⇒ 代理质量足够：§19/§21 的零结果可以读成「词性（在该用法下）没有可测增益」")
    else:
        print("  ⇒ 代理质量不足：§19/§21 的零结果只能读成「在该精度下未测出」，不能上升为「词性无用」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
