"""分词器测试：与 Rust `bm25_terms` 的口径对照（不联网、不依赖数据集）。"""

from __future__ import annotations

import tokenizer


def test_ascii_runs_are_lowercased_and_split_on_punctuation():
    # 与 Rust `tokenize.rs` 的测试同一组期望：PDF Report_v2 → pdf / report / v2。
    assert tokenizer.bm25_terms("PDF Report_v2") == ["pdf", "report", "v2"]


def test_no_stopword_filtering():
    # bm25_terms 不做停用词过滤（FAQ 方案的停用词只属于 faq.rs）。
    assert tokenizer.bm25_terms("怎么设置") == ["怎么", "设置"]


def test_cjk_block_is_segmented_and_round_trips():
    terms = tokenizer.bm25_terms("报警阈值")
    assert terms  # jieba 切出来的内容词
    assert "".join(terms) == "报警阈值"


def test_mixed_block_goes_through_jieba():
    terms = tokenizer.bm25_terms("PDF报表")
    assert terms
    assert "".join(terms) == "pdf报表"


def test_empty_and_punctuation_only():
    assert tokenizer.bm25_terms("") == []
    assert tokenizer.bm25_terms("？？！…---") == []


def test_alnum_blocks_keep_digits_together():
    assert tokenizer.bm25_terms("abc123 中文456") == ["abc123", "中文", "456"]


def test_cjk_detection_matches_known_ranges():
    assert tokenizer.contains_cjk("汉字")
    assert tokenizer.contains_cjk("abc中文")
    assert not tokenizer.contains_cjk("hello world")
    assert tokenizer.is_cjk("の")
    assert not tokenizer.is_cjk("a")


def test_terms_are_deterministic():
    text = "The quick brown fox 跳过了 lazy dog 的阈值设置"
    assert tokenizer.bm25_terms(text) == tokenizer.bm25_terms(text)
