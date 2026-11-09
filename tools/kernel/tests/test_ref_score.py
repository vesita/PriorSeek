"""参考打分器测试：手算用例（与 Rust `seq.rs` 的 scoring 口径逐步对齐）。"""

from __future__ import annotations

import math

import build_seq
import pytest
import ref_score
from conftest import built_artifact, make_dataset, write_artifact

K1 = 1.5
B = 0.75


def _tf_norm(tf: float, doc_len: float, avg_len: float) -> float:
    """手算 §8.1 的 BM25 饱和 + 长度归一（idf 单独乘）。"""
    return tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len / avg_len))


def _scorer(artifact_dir, **params) -> ref_score.RefScorer:
    artifact = build_seq.read_artifact(artifact_dir)
    return ref_score.RefScorer(artifact, ref_score.ScoreParams(**params))


def test_hand_computed_score(tmp_path):
    # 词表 [alpha, beta, gamma]；idf 手写 [1.0, 2.0, 1.0]；
    # 边 alpha -1-> beta 权重 2.0、beta -1-> alpha 权重 1.0。
    # 文档 d1=[alpha,beta,alpha]（len 3）、d2=[beta]（len 1）、d3=[]（len 0）；
    # avg_len = (3+1+0)/3 = 4/3。
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha", "beta", "gamma"],
        idf=[1.0, 2.0, 1.0],
        edges=[(0, 1, 1, 2.0), (1, 0, 1, 1.0)],
        docs=[("d1", [0, 1, 0]), ("d2", [1]), ("d3", [])],
    )
    scorer = _scorer(artifact_dir)
    hits = scorer.search("alpha beta", 10)
    assert [doc_id for doc_id, _score in hits] == ["d1", "d2"]

    avg_len = 4.0 / 3.0
    expected_d1 = 1.0 * _tf_norm(2, 3, avg_len) + 2.0 * _tf_norm(1, 3, avg_len)
    # 时序：查询对 (alpha,beta)，d1 里 alpha 在 0/2、beta 在 1 → 匹配 1 次（sat=1）。
    expected_d1 += 0.5 * 2.0 * 1.0
    expected_d2 = 2.0 * _tf_norm(1, 1, avg_len)
    assert hits[0][1] == pytest.approx(expected_d1, rel=1e-12)
    assert hits[1][1] == pytest.approx(expected_d2, rel=1e-12)

    # lambda_seq=0：退回纯 BM25。
    plain = _scorer(artifact_dir, lambda_seq=0.0).search("alpha beta", 10)
    assert plain[0][1] == pytest.approx(expected_d1 - 0.5 * 2.0, rel=1e-12)


def test_ordered_requires_a_before_b(tmp_path):
    # d1 = [alpha, beta, alpha]；查询 "beta alpha"：
    # ordered：beta 在 1，alpha 在其后有 2 → 匹配 1 次，e=1.0 → s_seq=1.0。
    # unordered：|1-0|、|1-2| 都在窗口内 → 匹配 2 次 → s_seq=1+ln2。
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha", "beta"],
        idf=[1.0, 1.0],
        edges=[(1, 0, 1, 1.0)],
        docs=[("d1", [0, 1, 0])],
    )
    ordered = _scorer(artifact_dir).search("beta alpha", 10)
    unordered = _scorer(artifact_dir, ordered=False).search("beta alpha", 10)
    assert ordered[0][1] == pytest.approx(1.0 * _tf_norm(2, 3, 3) + 1.0 * _tf_norm(1, 3, 3) + 0.5 * 1.0)
    assert unordered[0][1] == pytest.approx(
        1.0 * _tf_norm(2, 3, 3) + 1.0 * _tf_norm(1, 3, 3) + 0.5 * (1.0 + math.log(2.0))
    )


def test_window_and_max_matches(tmp_path):
    # d1 = [alpha, alpha, alpha, beta]：alpha→beta 的有序匹配数随 window 变化；
    # 默认 window=8 → 3 次（被 max_matches=3 截断），window=1 → 1 次。
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha", "beta"],
        idf=[1.0, 1.0],
        edges=[(0, 1, 1, 1.0)],
        docs=[("d1", [0, 0, 0, 1])],
    )
    uni = _tf_norm(3, 4, 4) + _tf_norm(1, 4, 4)
    wide = _scorer(artifact_dir).search("alpha beta", 10)
    assert wide[0][1] == pytest.approx(uni + 0.5 * (1.0 + math.log(3.0)))
    narrow = _scorer(artifact_dir, window=1).search("alpha beta", 10)
    assert narrow[0][1] == pytest.approx(uni + 0.5 * (1.0 + math.log(1.0)))
    capped = _scorer(artifact_dir, max_matches=1).search("alpha beta", 10)
    assert capped[0][1] == pytest.approx(uni + 0.5 * (1.0 + math.log(1.0)))


def test_orders_sum_weights_for_same_pair(tmp_path):
    # 同一 (alpha,beta) 同时有 order=1（1.0）与 order=2（2.0）的边。
    # 查询 "alpha beta beta alpha beta"（q=[a,b,b,a,b]）里 (a,b) 既出现在相邻位置，
    # 也出现在隔一位置 → e(a,b) = 1.0 + 2.0 = 3.0；只开 orders=(1,) 时 e = 1.0。
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha", "beta"],
        idf=[1.0, 1.0],
        edges=[(0, 1, 1, 1.0), (0, 1, 2, 2.0)],
        docs=[("d1", [0, 1])],
    )
    uni = _tf_norm(1, 2, 2) + _tf_norm(1, 2, 2)
    hits = _scorer(artifact_dir).search("alpha beta beta alpha beta", 10)
    assert hits[0] == ("d1", pytest.approx(uni + 0.5 * 3.0 * 1.0))

    only_first_order = _scorer(artifact_dir, orders=(1,)).search("alpha beta beta alpha beta", 10)
    assert only_first_order[0] == ("d1", pytest.approx(uni + 0.5 * 1.0 * 1.0))

    # 只开 order=2：查询里 (a,b) 的隔一对来自位置 0/2 与 2/4 → 权重 2.0。
    only_second_order = _scorer(artifact_dir, orders=(2,)).search("alpha beta beta alpha beta", 10)
    assert only_second_order[0] == ("d1", pytest.approx(uni + 0.5 * 2.0 * 1.0))


def test_zero_weight_edges_are_ignored(tmp_path):
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha", "beta"],
        idf=[1.0, 1.0],
        edges=[(0, 1, 1, 0.0)],
        docs=[("d1", [0, 1])],
    )
    hits = _scorer(artifact_dir).search("alpha beta", 10)
    assert hits[0][1] == pytest.approx(_tf_norm(1, 2, 2) * 2)


def test_top_k_and_tie_break_by_doc_id(tmp_path):
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha"],
        idf=[1.0],
        edges=[],
        docs=[("d2", [0]), ("d1", [0]), ("d3", [0])],
    )
    hits = _scorer(artifact_dir).search("alpha", 2)
    assert [doc_id for doc_id, _score in hits] == ["d1", "d2"]  # 同分按 doc id 升序


def test_oov_query_and_empty_query(tmp_path):
    artifact_dir = write_artifact(
        tmp_path,
        vocab=["alpha"],
        idf=[1.0],
        edges=[],
        docs=[("d1", [0])],
    )
    assert _scorer(artifact_dir).search("zzz", 10) == []
    assert _scorer(artifact_dir).search("", 10) == []


def test_built_artifact_scores_a_real_query(tmp_path):
    # 端到端：构建 tiny 数据集，再用参考打分器检索；只断言分数为正、排序稳定。
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)
    scorer = _scorer(out)
    hits = scorer.search("alpha beta", 4)
    assert hits and all(score > 0 for _doc_id, score in hits)
    assert [score for _doc_id, score in hits] == sorted(
        (score for _doc_id, score in hits), reverse=True
    )
    # 重复查询结果一致。
    assert scorer.search("alpha beta", 4) == hits
