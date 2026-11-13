#!/usr/bin/env python3
"""Token 先验显著性因子构建器 (saliency-v0)。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "kernel"))
from tokenizer import bm25_terms  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACT_ID = "saliency-v0"

def get_token_features(doc_list: list[dict]) -> dict[str, list[float]]:
    n_docs = len(doc_list)
    term_stats = {}
    for doc in doc_list:
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        tokens = bm25_terms(text)
        doc_terms = {}
        for t in tokens:
            doc_terms[t] = doc_terms.get(t, 0) + 1
        for t, tf in doc_terms.items():
            if t not in term_stats:
                term_stats[t] = {"df": 0, "tf_sum": 0, "peak_tf": 0, "pos_sum": 0}
            s = term_stats[t]
            s["df"] += 1
            s["tf_sum"] += tf
            if tf > s["peak_tf"]:
                s["peak_tf"] = tf
            s["pos_sum"] += tokens.index(t) / max(len(tokens), 1)

    features = {}
    for t, s in term_stats.items():
        idf = math.log(1.0 + (n_docs - s["df"] + 0.5) / (s["df"] + 0.5))
        features[t] = [idf, s["df"] / n_docs, s["peak_tf"] / max(s["tf_sum"], 1), s["pos_sum"] / max(s["df"], 1), len(t) / 10.0]
    return features

def build(dataset: str):
    corpus_path = DATA_DIR / dataset / "corpus.jsonl"
    with corpus_path.open() as f:
        docs = [json.loads(line) for line in f]
    
    feats = get_token_features(docs)
    w = np.array([0.5, -0.5, 0.8, -0.2, 0.1])
    b = -0.5
    saliency_map = {}
    for t, f in feats.items():
        logit = np.dot(w, np.array(f)) + b
        saliency_map[t] = round(1.0 / (1.0 + math.exp(-logit)), 4)
    
    out_dir = DATA_DIR / dataset / "artifacts" / ARTIFACT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {t: [["saliency", s]] for t, s in sorted(saliency_map.items())}
    payload = json.dumps({"entries": entries}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    (out_dir / "saliency.json").write_bytes(payload)

    manifest = {
        "id": ARTIFACT_ID,
        "files": {"saliency.json": {"sha256": payload_sha256, "bytes": len(payload)}},
        "source": f"data/{dataset}/corpus.jsonl",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OK {dataset}: {len(entries)} tokens -> {payload_sha256[:16]}")

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "scifact")
