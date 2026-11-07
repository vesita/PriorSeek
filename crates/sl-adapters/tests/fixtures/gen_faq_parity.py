#!/usr/bin/env python3
"""生成 FAQ 五方案 parity fixture —— 必须用**真实 faq-search 包**算，不是手写期望值。

用法（在仓库根目录执行）::

    uv run --project /home/vesita/coding/hub/faq-search python \
        crates/sl-adapters/tests/fixtures/gen_faq_parity.py

输出 `faq_parity.json`，Rust 侧 `tests/faq_parity.rs` 逐方案断言 top-k 的 doc id 序列一致。

fixture 里的 `word_tokens` 是 Python 侧 jieba 的切词结果：Rust 用 jieba-rs，两边词表
同源但不是同一个实现，把查询/文档的切词结果也记下来，切词一漂移就能在测试里直接看到
差在哪条文本上，而不是只看到「排名对不上」。

语料 id 用 "1".."8" 这种**单位数**字符串：原包内部把 id 转 int，按整数 id 排同分；
Rust 侧按字符串 id 排同分。单位数下两种排序等价，fixture 因此不依赖「语料插入顺序」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from faq_search import FaqIndex, SCHEMES, words

# 中文 FAQ 语料：覆盖「问题 + 答案」、停用词、单字查询、整库无命中、代码/格式词。
CORPUS = [
    {
        "id": 1,
        "module_kind": "faq",
        "question": "报警阈值怎么设置",
        "answer": "在参数页修改阈值后保存，报警阈值立即生效。",
    },
    {
        "id": 2,
        "module_kind": "faq",
        "question": "日志留存天数在哪里配置",
        "answer": "在系统设置的日志页面配置留存天数，默认保留三十天。",
    },
    {
        "id": 3,
        "module_kind": "faq",
        "question": "修改参数后需要重启系统吗",
        "answer": "修改大部分参数不需要重启，只有端口变更需要重启服务。",
    },
    {
        "id": 4,
        "module_kind": "faq",
        "question": "邮件服务器端口是多少",
        "answer": "邮件服务器默认端口为二十五，加密端口为四百六十五。",
    },
    {
        "id": 5,
        "module_kind": "faq",
        "question": "刀片磨损了怎么办",
        "answer": "刀片磨损超过零点五毫米时更换新刀片，不要继续使用。",
    },
    {
        "id": 6,
        "module_kind": "faq",
        "question": "报表导出支持哪些格式",
        "answer": "报表支持导出为表格和文档两种格式，导出后可以打印。",
    },
    {
        "id": 7,
        "module_kind": "faq",
        "question": "用户权限如何分配",
        "answer": "在用户管理页面给账号分配角色，角色决定可访问的菜单。",
    },
    {
        "id": 8,
        "module_kind": "faq",
        "question": "温度过高会报警吗",
        "answer": "设备温度超过六十度时触发高温报警，需要检查散热风扇。",
    },
]

# 每条查询针对一个语义点：
# q1 停用词 + 改（word 方案里「怎么」会被滤掉）
# q2 多词、跨问题与答案
# q3 短词（「重启」两字）
# q4 单字查询 → bigram/bm25 必须逐文档退到字符特征
# q5 词命中（「端口」在词典里是词）
# q6 整库无该双字 → bigram/bm25 必须整库退到 char（「速度」不在语料里，但「度」在文档 8）
# q7 纯标点 → 空查询退化路径（返回全部文档，不过 min_score）
# q8 完全超纲 → 各方案的空结果行为
# q9 组合：邮件服务器的加密端口
QUERIES = [
    ("q1", "报警阈值怎么改"),
    ("q2", "日志留存天数"),
    ("q3", "重启"),
    ("q4", "刀"),
    ("q5", "端口"),
    ("q6", "速度"),
    ("q7", "？？"),
    ("q8", "量子纠缠计算"),
    ("q9", "邮件服务器的加密端口"),
]

TOPK = 5


def doc_text(item: dict) -> str:
    """与原包 `FaqDoc.text` 一致：问题 + 空格 + 答案。"""
    return f"{item['question']} {item['answer']}"


def main() -> int:
    index = FaqIndex.from_items(CORPUS)
    results: dict[str, dict[str, list[dict]]] = {}
    for qid, text in QUERIES:
        per_scheme: dict[str, list[dict]] = {}
        for scheme in SCHEMES:
            hits = index.search(text, topk=TOPK, scheme=scheme)
            per_scheme[scheme] = [{"id": str(hit.id), "score": hit.score} for hit in hits]
        results[qid] = per_scheme

    fixture = {
        "generated_by": (
            "uv run --project ~/coding/hub/faq-search python "
            "crates/sl-adapters/tests/fixtures/gen_faq_parity.py"
        ),
        "note": (
            "由真实 faq-search 包算出；word_tokens 是 Python jieba 切词结果，"
            "用于定位 jieba-rs 与 jieba 的切词漂移"
        ),
        "topk": TOPK,
        "schemes": list(SCHEMES),
        "corpus": CORPUS,
        "queries": [{"qid": qid, "text": text} for qid, text in QUERIES],
        "word_tokens": {
            "docs": {str(item["id"]): words(doc_text(item)) for item in CORPUS},
            "queries": {qid: words(text) for qid, text in QUERIES},
        },
        "results": results,
    }
    out = Path(__file__).with_name("faq_parity.json")
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {out}（{len(CORPUS)} 篇文档 / {len(QUERIES)} 条查询 / {len(SCHEMES)} 个方案）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
