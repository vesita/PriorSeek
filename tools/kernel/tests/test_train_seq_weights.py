"""训练器（`train_seq_weights.py`）的离线测试。

覆盖两件事：
1. **N3 训练/测试隔离**：训练器绝不打开测试集 `qrels.tsv`（预注册 §2 N3 的代码级强制点）；
2. 拟合器在「时序分量能分开正负例」的合成数据上确实学会正权重（防止优化器写错导致永远输出 0）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_seq_weights as trainer  # noqa: E402


def test_fit_recovers_positive_weight_on_separable_synthetic_data():
    """合成数据：正例的 s1 恒高、负例恒低 ⇒ 无约束最优 a1 应显著为正。"""
    rng = np.random.default_rng(0)
    n = 2000
    du = rng.normal(0.0, 0.1, n)  # 单字项无信息
    d1 = np.full(n, 1.0)  # order-1 时序恒正
    d2 = rng.normal(0.0, 0.1, n)
    pairs = np.column_stack([du, d1, d2, np.zeros(n), np.zeros(n), np.zeros(n)])
    a, loss = trainer.fit(pairs, iterations=200, lr=0.05)
    assert a[0] > 1.0, f"应学到显著的 a1，实测 {a[0]:.4f}"
    assert loss[-1] < loss[0]
    assert a[1] == pytest.approx(0.0, abs=1e-9)


def test_fit_keeps_nonnegative_constraint():
    """若梯度把权重推向负值，非负约束应把它压在 0（预注册口径 a1,a2 ≥ 0）。"""
    rng = np.random.default_rng(1)
    n = 2000
    d1 = np.full(n, -1.0)  # 正例的 s1 更小 ⇒ 增大 a1 只会更差
    pairs = np.column_stack([np.zeros(n), d1, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    a, _ = trainer.fit(pairs, iterations=100, lr=0.05)
    assert np.all(a >= 0.0)


def test_trainer_never_opens_test_qrels(tmp_path, monkeypatch):
    """N3：整个训练过程不得打开 `qrels.tsv`（测试集），只允许读 train split。"""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "tiny"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "corpus.jsonl").write_text(
        "".join(json.dumps({"id": f"d{i}", "text": f"word{i}"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    (dataset_dir / "train_queries.jsonl").write_text(
        json.dumps({"id": "q1", "text": "word0 word1"}) + "\n", encoding="utf-8"
    )
    (dataset_dir / "train_qrels.tsv").write_text("qid\tdocid\trel\nq1\td0\t1\n", encoding="utf-8")

    opened: list[str] = []
    real_read_text = Path.read_text
    real_open = Path.open

    def spy_read_text(self, *args, **kwargs):  # noqa: ANN001
        opened.append(str(self))
        return real_read_text(self, *args, **kwargs)

    def spy_open(self, *args, **kwargs):  # noqa: ANN001
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)
    monkeypatch.setattr(Path, "open", spy_open)

    queries, positives = trainer.load_train_split(dataset_dir)
    assert queries == {"q1": "word0 word1"}
    assert positives == {"q1": ["d0"]}
    assert opened, "读取记录为空说明探针没生效"
    # 注意：训练文件叫 `train_qrels.tsv`，所以必须比 basename 全等，不能用 endswith。
    assert not any(Path(p).name == "qrels.tsv" for p in opened), f"训练器打开了测试集 qrels：{opened}"
