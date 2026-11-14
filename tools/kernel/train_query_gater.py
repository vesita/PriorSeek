import json
import math
import hashlib
import os
import re
from pathlib import Path

def tokenize(text: str):
    return [w.lower() for w in re.findall(r'[A-Za-z0-9_\-]+', text) if len(w) > 1]

def build_query_gater(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / dataset
    
    print("[*] 统计语料库词频与 IDF...")
    doc_freq = {}
    total_docs = 0
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            total_docs += 1
            doc = json.loads(line)
            tokens = set(tokenize(doc.get("title", "") + " " + doc.get("text", "")))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

    def idf(token: str):
        df = doc_freq.get(token, 0)
        return math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

    print(f"[*] 全量文档数: {total_docs}, 唯一词项数: {len(doc_freq)}")
    
    queries = {}
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["id"]] = q["text"]

    print(f"[*] 针对 {len(queries)} 条查询计算自适应先验路由权重...")
    query_weights = {}

    for qid, text in queries.items():
        tokens = tokenize(text)
        if not tokens:
            query_weights[qid] = [1.0, 1.0, 0.5]
            continue

        idfs = [idf(t) for t in tokens]
        max_idf = max(idfs)
        mean_idf = sum(idfs) / len(idfs)
        q_len = len(tokens)

        # 核心先验理论公式（由内核特征驱动）：
        # 1. 概念词法路 (w_concept)：max_idf 越大（专有名词越强），词法置信度越高
        w_concept = 1.0 + 0.1 * max(0.0, max_idf - 6.0) - 0.15 * max(0.0, 4.0 - mean_idf)
        
        # 2. 稠密向量路 (w_dense)：mean_idf 越低（抽象句、修饰语多），连续语义流越关键
        w_dense = 1.0 + 0.2 * max(0.0, 4.5 - mean_idf) - 0.1 * max(0.0, max_idf - 7.5)
        
        # 3. 显著性门控路 (w_saliency)：长查询且含有高低反差词时显著性最重要
        idf_spread = max_idf - min(idfs)
        w_saliency = 0.4 + 0.1 * min(3.0, idf_spread) + (0.1 if q_len >= 8 else 0.0)

        # 保持在安全合理的区间内
        w_concept = round(max(0.4, min(1.8, w_concept)), 4)
        w_dense = round(max(0.4, min(1.8, w_dense)), 4)
        w_saliency = round(max(0.2, min(1.2, w_saliency)), 4)

        query_weights[qid] = [w_concept, w_dense, w_saliency]

    # 输出工件
    out_dir = data_dir / "artifacts" / "query-gater-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.json"

    export_data = {
        "description": "PriorSeek Query-Aware Adaptive RRF Route Weights",
        "routes": ["concept-synonym", "dense-v0", "saliency-bm25"],
        "default": [1.0, 1.0, 0.5],
        "queries": query_weights,
    }

    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    # 计算 sha256 并生成 manifest.json
    hasher = hashlib.sha256()
    with open(weights_path, "rb") as f:
        hasher.update(f.read())
    file_sha = hasher.hexdigest()

    manifest = {
        "name": "query-gater-v1",
        "files": {
            "weights.json": file_sha
        },
        "routes": ["concept-synonym", "dense-v0", "saliency-bm25"]
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 工件生成成功: {weights_path} (sha256: {file_sha[:16]}...)")

if __name__ == "__main__":
    build_query_gater("scifact")
