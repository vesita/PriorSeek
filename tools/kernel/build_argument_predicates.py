import json
import hashlib
import re
from pathlib import Path

# 因果谓词词典
PREDICATES = {
    "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing",
    "promote", "promotes", "promoted", "promoting",
    "inhibit", "inhibits", "inhibited", "inhibiting", "inhibition",
    "activate", "activates", "activated", "activating", "activation",
    "suppress", "suppresses", "suppressed", "suppressing", "suppression",
    "upregulate", "upregulates", "upregulated", "upregulation",
    "downregulate", "downregulates", "downregulated", "downregulation",
    "enhance", "enhances", "enhanced", "enhancing",
    "reduce", "reduces", "reduced", "reducing", "reduction",
    "stimulate", "stimulates", "stimulated", "stimulation",
    "block", "blocks", "blocked", "blocking",
    "cause", "causes", "caused", "causing",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "induce", "induces", "induced", "inducing", "induction",
    "attenuate", "attenuates", "attenuated", "attenuating",
    "improve", "improves", "improved", "improving",
    "impair", "impairs", "impaired", "impairment",
    "accelerate", "accelerates", "accelerated",
    "lack", "lacks", "lacked", "lacking",
    "deplete", "depletes", "depleted", "depletion"
}

# 常见功能词/虚词停用词
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must",
    "can", "could", "of", "and", "or", "nor", "that", "this", "these",
    "those", "which", "who", "whom", "whose", "it", "its", "they", "their"
}

def extract_arguments(text: str):
    words = [w.lower() for w in re.findall(r'[A-Za-z0-9_\-]+', text)]
    # 论元核心：过滤掉因果谓词与功能虚词，保留名词、基因、突变与实体概念
    args = [w for w in words if w not in PREDICATES and w not in STOPWORDS and len(w) > 1]
    return args

def build_decoupled_prior(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / dataset
    out_dir = data_dir / "artifacts" / "decoupled-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] 读取查询并解耦提取核心实体论元 (Argument Cores)...")
    query_args = {}
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            args = extract_arguments(q["text"])
            query_args[q["id"]] = args

    print(f"[*] 已为 {len(query_args)} 条查询完成论元解耦抽取")

    # 导出到 JSON
    q_file = out_dir / "query_arguments.json"
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(query_args, f, indent=2, ensure_ascii=False)

    # 复制复用 predicate-v1 的因果极性
    src_pred = data_dir / "artifacts" / "predicate-v1" / "doc_polarities.json"
    dst_pred = out_dir / "doc_polarities.json"
    import shutil
    shutil.copy(src_pred, dst_pred)

    manifest = {
        "name": "decoupled-v1",
        "description": "Decoupled entity arguments and causal predicate polarities",
        "files": {
            "query_arguments.json": hashlib.sha256(open(q_file, "rb").read()).hexdigest(),
            "doc_polarities.json": hashlib.sha256(open(dst_pred, "rb").read()).hexdigest(),
        }
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 结构解耦先验工件已生成至 {out_dir}")

if __name__ == "__main__":
    build_decoupled_prior("scifact")
