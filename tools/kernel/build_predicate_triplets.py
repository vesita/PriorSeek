import json
import hashlib
import re
from pathlib import Path

POS_WORDS = {
    "increase", "increases", "increased", "increasing",
    "promote", "promotes", "promoted", "promoting",
    "activate", "activates", "activated", "activating", "activation",
    "upregulate", "upregulates", "upregulated", "upregulation",
    "enhance", "enhances", "enhanced", "enhancing",
    "stimulate", "stimulates", "stimulated", "stimulation",
    "cause", "causes", "caused", "causing",
    "induce", "induces", "induced", "inducing", "induction",
    "improve", "improves", "improved", "improving",
    "accelerate", "accelerates", "accelerated",
    "foster", "fosters", "fostered",
    "trigger", "triggers", "triggered"
}

NEG_WORDS = {
    "decrease", "decreases", "decreased", "decreasing",
    "inhibit", "inhibits", "inhibited", "inhibiting", "inhibition",
    "suppress", "suppresses", "suppressed", "suppressing", "suppression",
    "downregulate", "downregulates", "downregulated", "downregulation",
    "reduce", "reduces", "reduced", "reducing", "reduction",
    "block", "blocks", "blocked", "blocking",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "attenuate", "attenuates", "attenuated", "attenuating",
    "impair", "impairs", "impaired", "impairment",
    "loss", "deplete", "depletes", "depleted", "depletion",
    "lack", "lacks", "lacked", "lacking",
    "diminish", "diminishes", "diminished"
}

def tokenize_words(text: str):
    return [w.lower() for w in re.findall(r'[A-Za-z]+', text)]

def extract_polarity(tokens):
    pos_count = sum(1 for t in tokens if t in POS_WORDS)
    neg_count = sum(1 for t in tokens if t in NEG_WORDS)
    if pos_count > neg_count and pos_count >= 1:
        return 1
    elif neg_count > pos_count and neg_count >= 1:
        return -1
    else:
        return 0

def build_predicate_prior(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data" / dataset
    out_dir = data_dir / "artifacts" / "predicate-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] 正在扫描全量语料库文档的因果极性拓扑...")
    doc_polarities = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["id"]
            # 标题权重更高
            title_tokens = tokenize_words(doc.get("title", ""))
            text_tokens = tokenize_words(doc.get("text", ""))
            
            title_pol = extract_polarity(title_tokens)
            text_pol = extract_polarity(text_tokens)
            
            # 若标题有明确极性则以标题为先，否则看正文
            doc_pol = title_pol if title_pol != 0 else text_pol
            doc_polarities[doc_id] = doc_pol

    print(f"[*] 已分析 {len(doc_polarities)} 篇文档极性 (正向: {sum(1 for v in doc_polarities.values() if v==1)}, 负向: {sum(1 for v in doc_polarities.values() if v==-1)}, 中性: {sum(1 for v in doc_polarities.values() if v==0)})")

    print("[*] 正在分析查询集的因果极性...")
    query_polarities = {}
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            tokens = tokenize_words(q["text"])
            query_polarities[q["id"]] = extract_polarity(tokens)

    print(f"[*] 已分析 {len(query_polarities)} 条查询极性")

    doc_file = out_dir / "doc_polarities.json"
    query_file = out_dir / "query_polarities.json"

    with open(doc_file, "w", encoding="utf-8") as f:
        json.dump(doc_polarities, f)

    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(query_polarities, f)

    # 计算 sha256 并生成 manifest
    manifest = {
        "name": "predicate-v1",
        "description": "Causal predicate polarity priors (+1 positive, -1 negative, 0 neutral)",
        "files": {
            "doc_polarities.json": hashlib.sha256(open(doc_file, "rb").read()).hexdigest(),
            "query_polarities.json": hashlib.sha256(open(query_file, "rb").read()).hexdigest(),
        }
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 谓词因果先验工件已导出至 {out_dir}")

if __name__ == "__main__":
    build_predicate_prior("scifact")
