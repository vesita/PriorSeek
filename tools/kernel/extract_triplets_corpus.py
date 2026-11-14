import json
import hashlib
import re
from pathlib import Path

REL_MAP = {
    # 促进/上调
    "increase": "PROMOTES",
    "increases": "PROMOTES",
    "increased": "PROMOTES",
    "increasing": "PROMOTES",
    "promote": "PROMOTES",
    "promotes": "PROMOTES",
    "promoted": "PROMOTES",
    "promoting": "PROMOTES",
    "activate": "PROMOTES",
    "activates": "PROMOTES",
    "activated": "PROMOTES",
    "activating": "PROMOTES",
    "activation": "PROMOTES",
    "upregulate": "PROMOTES",
    "upregulates": "PROMOTES",
    "upregulated": "PROMOTES",
    "upregulation": "PROMOTES",
    "enhance": "PROMOTES",
    "enhances": "PROMOTES",
    "enhanced": "PROMOTES",
    "stimulate": "PROMOTES",
    "stimulates": "PROMOTES",
    "induce": "PROMOTES",
    "induces": "PROMOTES",
    "induced": "PROMOTES",
    "induction": "PROMOTES",
    "cause": "PROMOTES",
    "causes": "PROMOTES",
    "improve": "PROMOTES",
    "improves": "PROMOTES",

    # 抑制/下调
    "decrease": "INHIBITS",
    "decreases": "INHIBITS",
    "decreased": "INHIBITS",
    "decreasing": "INHIBITS",
    "inhibit": "INHIBITS",
    "inhibits": "INHIBITS",
    "inhibited": "INHIBITS",
    "inhibiting": "INHIBITS",
    "inhibition": "INHIBITS",
    "suppress": "INHIBITS",
    "suppresses": "INHIBITS",
    "suppressed": "INHIBITS",
    "downregulate": "INHIBITS",
    "downregulates": "INHIBITS",
    "downregulated": "INHIBITS",
    "reduce": "INHIBITS",
    "reduces": "INHIBITS",
    "reduced": "INHIBITS",
    "reduction": "INHIBITS",
    "block": "INHIBITS",
    "blocks": "INHIBITS",
    "blocked": "INHIBITS",
    "prevent": "INHIBITS",
    "prevents": "INHIBITS",
    "prevented": "INHIBITS",
    "attenuate": "INHIBITS",
    "attenuates": "INHIBITS",
    "impair": "INHIBITS",
    "impairs": "INHIBITS",
    "deplete": "INHIBITS",
    "depletes": "INHIBITS",
    "loss": "INHIBITS",

    # 相互作用/关联
    "bind": "ASSOCIATES",
    "binds": "ASSOCIATES",
    "interact": "ASSOCIATES",
    "interacts": "ASSOCIATES",
    "associate": "ASSOCIATES",
    "associates": "ASSOCIATES",
    "correlate": "ASSOCIATES",
    "correlates": "ASSOCIATES",
}

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must",
    "can", "could", "of", "and", "or", "nor", "that", "this", "these",
    "those", "which", "who", "whom", "whose", "it", "its", "they", "their",
    "not", "no", "we", "our", "here", "study", "results", "data", "found"
}

def extract_clause_triplets(sentence: str):
    words = [w.lower() for w in re.findall(r'[A-Za-z0-9_\-]+', sentence)]
    if len(words) < 3:
        return []

    triplets = []
    # 查找谓词所在位置
    for idx, w in enumerate(words):
        if w in REL_MAP:
            rel = REL_MAP[w]
            # 谓词左侧提取核心主格实体
            left_words = [t for t in words[:idx] if t not in STOPWORDS and len(t) > 1]
            # 谓词右侧提取核心宾格实体
            right_words = [t for t in words[idx+1:] if t not in STOPWORDS and len(t) > 1]

            if left_words and right_words:
                subj = left_words[-1] # 最贴近谓词的核心主实体
                obj = right_words[0]  # 最贴近谓词的核心宾实体
                if subj != obj:
                    triplets.append([subj, rel, obj])
    return triplets

def build_triplet_artifacts(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / dataset
    out_dir = data_dir / "artifacts" / "triplets-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] 正在从语料库中提取实体-关系因果三元组拓扑...")
    corpus_triplets = {}
    total_triplets = 0

    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["id"]
            text = (doc.get("title", "") + ". " + doc.get("text", "")).replace("\n", " ")
            # 按句子切分
            sentences = re.split(r'[.!?]+', text)
            doc_t = []
            for s in sentences:
                doc_t.extend(extract_clause_triplets(s))
            if doc_t:
                corpus_triplets[doc_id] = doc_t
                total_triplets += len(doc_t)

    print(f"[*] 全量语料库提取完成：覆盖 {len(corpus_triplets)}/5183 篇文档，累计生成 {total_triplets} 条因果三元组！")

    print("[*] 正在从查询集中提取命题三元组...")
    query_triplets = {}
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            qid = q["id"]
            text = q["text"]
            t_list = extract_clause_triplets(text)
            if t_list:
                query_triplets[qid] = t_list[0] # 取主命题

    print(f"[*] 查询集命题提取完成：成功结构化 {len(query_triplets)}/1109 条查询命题！")

    c_file = out_dir / "corpus_triplets.json"
    q_file = out_dir / "query_triplets.json"

    with open(c_file, "w", encoding="utf-8") as f:
        json.dump(corpus_triplets, f, ensure_ascii=False)

    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(query_triplets, f, indent=2, ensure_ascii=False)

    manifest = {
        "name": "triplets-v1",
        "description": "Normalized (Subject, Relation, Object) causal topology triplets",
        "num_doc_covered": len(corpus_triplets),
        "total_triplets": total_triplets,
        "files": {
            "corpus_triplets.json": hashlib.sha256(open(c_file, "rb").read()).hexdigest(),
            "query_triplets.json": hashlib.sha256(open(q_file, "rb").read()).hexdigest(),
        }
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 因果三元组拓扑工件已导出至 {out_dir}")

    # 打印前 3 条展示
    print("\n=== 三元组提取案例展示 ===")
    for qid in list(query_triplets.keys())[:3]:
        print(f"QID {qid}: {query_triplets[qid]}")

if __name__ == "__main__":
    build_triplet_artifacts("scifact")
