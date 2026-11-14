import json
import hashlib
import re
from pathlib import Path

# 因果谓词逆转映射表（正向 ↔ 负向互换）
POLARITY_INVERSIONS = {
    # 动词原形与第三人称单数
    "increase": "decrease",
    "increases": "decreases",
    "increased": "decreased",
    "increasing": "decreasing",
    "decrease": "increase",
    "decreases": "increases",
    "decreased": "increased",
    "decreasing": "increasing",

    "promote": "inhibit",
    "promotes": "inhibits",
    "promoted": "inhibited",
    "promoting": "inhibiting",
    "inhibit": "promote",
    "inhibits": "promotes",
    "inhibited": "promoted",
    "inhibiting": "promoting",

    "activate": "suppress",
    "activates": "suppresses",
    "activated": "suppressed",
    "activating": "suppressing",
    "suppress": "activate",
    "suppresses": "activates",
    "suppressed": "activated",
    "suppressing": "activating",

    "upregulate": "downregulate",
    "upregulates": "downregulates",
    "upregulated": "downregulated",
    "upregulation": "downregulation",
    "downregulate": "upregulate",
    "downregulates": "upregulates",
    "downregulated": "upregulated",
    "downregulation": "upregulation",

    "enhance": "impair",
    "enhances": "impairs",
    "enhanced": "impaired",
    "enhancing": "impairing",
    "impair": "enhance",
    "impairs": "enhances",
    "impaired": "enhanced",
    "impairing": "enhancing",

    "induce": "prevent",
    "induces": "prevents",
    "induced": "prevented",
    "inducing": "preventing",
    "prevent": "induce",
    "prevents": "induces",
    "prevented": "induced",
    "preventing": "inducing",

    "stimulate": "block",
    "stimulates": "blocks",
    "stimulated": "blocked",
    "block": "stimulate",
    "blocks": "stimulates",
    "blocked": "stimulated",
}

def invert_causal_statement(text: str):
    words = re.findall(r'\b[A-Za-z]+\b|\S', text)
    inverted = []
    flipped = False

    for w in words:
        lower = w.lower()
        if not flipped and lower in POLARITY_INVERSIONS:
            inv = POLARITY_INVERSIONS[lower]
            # 保持大小写形态
            if w.isupper():
                inv = inv.upper()
            elif w[0].isupper():
                inv = inv.capitalize()
            inverted.append(inv)
            flipped = True
        else:
            inverted.append(w)

    if flipped:
        # 重构文本，保留标点紧跟
        res = ""
        for i, token in enumerate(inverted):
            if i > 0 and re.match(r'^[A-Za-z0-9]', token):
                res += " "
            res += token
        return res
    return None

def build_counterfactual_benchmark(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / dataset
    out_dir = data_dir / "artifacts" / "counterfactual-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 qrels
    qrels = {}
    with open(data_dir / "qrels.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0] != "qid":
                qrels.setdefault(parts[0], []).append(parts[1])

    # 读取 queries
    queries = {}
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["id"]] = q["text"]

    probes = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue
        inv_text = invert_causal_statement(text)
        if inv_text and inv_text != text:
            probes.append({
                "qid": qid,
                "factual_query": text,
                "counterfactual_query": inv_text,
                "gold_docs": qrels[qid]
            })

    print(f"[*] 从 {len(queries)} 条查询中成功提取 {len(probes)} 对严格反事实因果对立探针！")

    out_file = out_dir / "probes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)

    # 计算 sha256
    file_sha = hashlib.sha256(open(out_file, "rb").read()).hexdigest()
    manifest = {
        "name": "counterfactual-v1",
        "description": "Counterfactual causal evaluation benchmark pairs",
        "num_probes": len(probes),
        "files": {
            "probes.json": file_sha
        }
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 反事实因果评测基准工件已生成: {out_file} (sha256: {file_sha[:16]}...)")

    # 打印前 3 个经典反事实样本展示
    print("\n=== 反事实因果探针样本展示 ===")
    for p in probes[:3]:
        print(f"QID {p['qid']}:")
        print(f"  事实原句:   {p['factual_query']}")
        print(f"  反事实逆转: {p['counterfactual_query']}")
        print(f"  真值文献:   {p['gold_docs']}")

if __name__ == "__main__":
    build_counterfactual_benchmark("scifact")
