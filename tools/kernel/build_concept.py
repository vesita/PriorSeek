#!/usr/bin/env python3
"""把**学到的**概念归并谓词固化成先验因子工件：`token → 规范形式`。

内核输出（`FactorKind::Concept`）：`{"entries": {token: [[canonical, 1.0]]}}`。
形态与 `exp-ppmi-v0` / `field-v0` 同构（`token → [[目标, 置信度]]`），外壳只需一条查表路径。

权重来自 `train_concept.py`（在 outer-train 上优化 nDCG@10 得到，`weights.json`）。
**本脚本不做任何判定**：它只把已学到的 `w`、`b` 应用到候选对上，产出映射。
候选对空间（`t` 是 `u` 的前缀）是结构约束，不含任何语言相关的后缀规则。

用法（在 tools/kernel/ 下）：
    uv run python build_concept.py scifact
    uv run python build_concept.py scifact --artifact-id concept-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_concept as kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_ID = "concept-v0"
PAYLOAD = "concepts.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build(
    dataset: str,
    *,
    data_dir: Path,
    artifact_id: str,
    weights_path: Path | None,
) -> int:
    dataset_dir = data_dir / dataset
    weights_file = weights_path or dataset_dir / "artifacts" / artifact_id / "weights.json"
    if not weights_file.is_file():
        raise SystemExit(f"找不到学到的权重 {weights_file}（先跑 train_concept.py）")
    learned = json.loads(weights_file.read_text(encoding="utf-8"))

    docs = kernel.load_corpus(dataset_dir)
    corpus = kernel.Corpus(docs)
    rule = learned.get("candidate_rule") or {}
    # 候选规则必须显式记录：缺了就停车，不猜默认值（猜错会让工件与权重对不上）。
    if not rule.get("rule"):
        raise SystemExit(
            f"{weights_file} 没有记录 candidate_rule.rule —— 用 train_concept.py 重新产一份"
        )
    cands = corpus.candidates(
        rule=rule["rule"],
        min_len=rule.get("min_len", 4),
        max_extra=rule.get("max_extra", 8),
        lcp_ratio=rule.get("lcp_ratio", 0.7),
    )
    scale = np.array(learned["feature_scale"], dtype=np.float64)
    X = np.stack([corpus.pair_features(t, u) for t, u in cands])
    X = X / np.where(scale > 0.0, scale, 1.0)
    mapping = kernel.make_mapping(
        cands,
        X,
        np.array(learned["weights"], dtype=np.float64),
        float(learned["intercept"]),
        corpus.df,
    )
    if not mapping:
        raise SystemExit("学到的谓词没有合并任何候选对——空因子不应产出工件")

    entries = {token: [[canonical, 1.0]] for token, canonical in sorted(mapping.items())}
    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    weights_bytes = weights_file.read_bytes()

    out_dir = dataset_dir / "artifacts" / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PAYLOAD).write_bytes(payload)
    (out_dir / "weights.json").write_bytes(weights_bytes)

    manifest = {
        "id": artifact_id,
        "kind": "concept_map",
        "weights_origin": "learned_weights",
        "source": f"data/{dataset}/corpus.jsonl（候选对与特征全部来自语料统计）",
        "prior_factor": {
            "name": "token_concept",
            "kind": "concept",
            "shape": ["n_vocab_tokens", 1],
            "dtype": "f32",
            "count": len(entries),
        },
        "network": {
            "arch": "feature-weighted merge predicate：merge = σ(w·x + b) ≥ 0.5，"
                    f"{len(learned['features'])} 个特征 + 截距",
            "params": len(learned["features"]) + 1,
            "checkpoint_sha256": sha256_bytes(weights_bytes),
            "features": learned["features"],
            "weights": learned["weights"],
            "intercept": learned["intercept"],
        },
        "params": {
            "candidate_rule": learned.get("candidate_rule"),
            "objective": learned["objective"],
        },
        "counts": {
            "vocab": len(corpus.df),
            "candidate_pairs": len(cands),
            "merged_tokens": len(entries),
            "concepts": len(set(mapping.values())),
        },
        "files": {
            PAYLOAD: {"sha256": sha256_bytes(payload), "bytes": len(payload)},
            "weights.json": {"sha256": sha256_bytes(weights_bytes), "bytes": len(weights_bytes)},
        },
        "runtime": {"device": "cpu", "precision": "f32"},
        "trained_on": learned["trained_on"],
        "contamination": "无预训练模型参与，无外部语料",
        "license": "由本仓从数据集语料构建",
        "notes": "因子只给事实（哪个 token 归到哪个规范形式）；怎么用（是否硬归并、是否加权）由外壳决定。",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"OK {dataset}: {len(entries)} 个 token 归并到概念 → {out_dir}\n"
        f"   {PAYLOAD} {len(payload)} B sha256={manifest['files'][PAYLOAD]['sha256'][:16]}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="产出概念归并先验因子工件（权重来自内核学习）")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-id", default=ARTIFACT_ID)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)
    return build(
        args.dataset,
        data_dir=args.data_dir,
        artifact_id=args.artifact_id,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    raise SystemExit(main())
