#!/usr/bin/env python3
"""seq 内核 v0 工件构建器：从规范数据集 `data/<name>/` 产出 `artifacts/seq-v0/`。

用法（在 `tools/kernel/` 下）：

    uv run python build_seq.py scifact
    uv run python build_seq.py nq-sub --force
    uv run python build_seq.py scifact --data-dir /tmp/data

产物四个文件（格式的唯一权威是 `docs/KERNEL.md` §8.2）：

- `manifest.json`  元数据 + 内容文件摘要 + 数据集指纹（`KernelArtifact::from_dir` 会逐项校验）；
- `seq.bin`        词表（字典序）+ idf(f32) + 边表（src,dst,order,weight，按 (src,dst,order) 升序）；
- `docs.txt`       文档 id 列表，顺序与 `doc_tokens.bin` 一致（= corpus.jsonl 行序）；
- `doc_tokens.bin` 每篇文档的 token id 序列，由本模块的 Python 分词器切好固化。

口径（与 `docs/KERNEL.md` §8 一致，未写死的点见 README）：

- 只用 `corpus.jsonl`（不看 qrels）；文档全文 = title + "\\n" + text（同 Rust `Doc::full_text`）；
- 词表 = 文档 token 去重；`idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`；
- 边 = 文档内 order=1（相邻）与 order=2（隔一）的有序共现，逐阶独立算 PPMI：
  `pmi = ln(c^0.75 * N_pair / (c_src^0.75 * c_dst^0.75))`，`weight = max(0, pmi - ln(5))`；
  只保留 `c >= min_df(2)` 的边，每个 src 跨两阶合并保留 weight 最大的 top-20；
  零权边不落盘（对打分贡献恒为 0），边的 (src,dst,order) 唯一；
- 相同数据 + 相同参数重复构建，四个文件字节完全相同（没有时间戳、没有集合迭代顺序泄漏）。

读写工件的函数（`read_artifact`）同时供 `ref_score.py` 与测试使用，格式只在这里实现一次。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tokenizer import bm25_terms

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

ARTIFACT_ID = "seq-v0"
ARTIFACT_KIND = "seq_graph"
ARTIFACT_METHOD = "ppmi_topk"
WEIGHTS_ORIGIN = "corpus_ppmi"

MAGIC_SEQ = b"SLSEQv1\0"
MAGIC_DOC = b"SLDOCv1\0"
SEQ_HEADER = len(MAGIC_SEQ) + 4
DOC_HEADER = len(MAGIC_DOC) + 4
EDGE_STRUCT = struct.Struct("<IIBf")
EDGE_BYTES = EDGE_STRUCT.size  # 13，小端无对齐填充

# 构建参数（与 docs/KERNEL.md §8.2 的 manifest.params 一致）。
ORDERS: tuple[int, ...] = (1, 2)
TOP_K = 20
WINDOW = 8
MIN_DF = 2
PPMI_SHIFT = 5.0
SMOOTHING = 0.75

PROGRESS_EVERY = 10000
TOKEN_LEN_LIMIT = 0xFFFF  # 词表里单个 token 的 UTF-8 字节数上限（u16 len）


class BuildError(RuntimeError):
    """构建失败：CLI 只打印一句话，不打堆栈；测试可以直接断言。"""


# --------------------------------------------------------------------------- 摘要与指纹


def sha256_file(path: Path) -> str:
    """文件字节的 SHA-256（小写十六进制）。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_digest(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def dataset_manifest_fingerprint(manifest: dict) -> str:
    """复刻 Rust `Manifest::fingerprint()`（与 `tools/datasets/canonical.py` 同口径）。

    工件 `source_corpus.fingerprint` 存这个值：它绑定「同一道题」，而不是 manifest
    文件本身被重新排版后的字节。counts 按 64 位 usize 小端写入，files 按文件名排序。
    """
    hasher = hashlib.sha256()
    hasher.update(manifest["name"].encode())
    hasher.update(b"\x00")
    hasher.update(manifest["version"].encode())
    hasher.update(b"\x00")
    for file, digest in sorted(manifest["files"].items()):
        hasher.update(file.encode())
        hasher.update(b"\x00")
        hasher.update(digest["sha256"].encode())
        hasher.update(b"\x00")
    for key in ("docs", "queries", "qrels"):
        hasher.update(int(manifest["counts"][key]).to_bytes(8, "little"))
    return hasher.hexdigest()


# --------------------------------------------------------------------------- 读数据集


@dataclass(frozen=True)
class SourceDoc:
    """一篇规范文档（只保留构建需要的字段）。"""

    id: str
    full_text: str


def load_dataset(data_dir: Path, name: str) -> tuple[dict, list[SourceDoc], Path]:
    """读 `data/<name>/manifest.json` 与 `corpus.jsonl`，并校验语料没被手改。"""
    dataset_dir = data_dir / name
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BuildError(f"找不到数据集清单 {manifest_path}（先用 tools/datasets/build.py 构建）")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        raise BuildError(f"找不到语料 {corpus_path}")
    expected = (manifest.get("files") or {}).get("corpus.jsonl") or {}
    expected_sha = expected.get("sha256")
    if expected_sha:
        actual = sha256_file(corpus_path)
        if actual != expected_sha:
            raise BuildError(
                f"{corpus_path} 的 sha256 与数据集 manifest 不符（期望 {expected_sha}，实际 {actual}）"
                " —— 语料被改过，先重建数据集"
            )

    docs: list[SourceDoc] = []
    seen: set[str] = set()
    with open(corpus_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = row["id"]
            if doc_id in seen:
                raise BuildError(f"{corpus_path} 第 {lineno} 行重复 doc id {doc_id!r}")
            seen.add(doc_id)
            title = row.get("title") or ""
            text = row["text"]
            # 与 Rust Doc::full_text 一致：title 非空时拼在前面，否则只用 text。
            full_text = f"{title}\n{text}" if title else text
            docs.append(SourceDoc(id=doc_id, full_text=full_text))
    if not docs:
        raise BuildError(f"{corpus_path} 没有文档")
    return manifest, docs, dataset_dir


# --------------------------------------------------------------------------- 分词与统计


def tokenize_docs(docs: list[SourceDoc]) -> list[list[str]]:
    """逐篇切 token；进度走 stderr（stdout 只留最终摘要）。"""
    tokenized: list[list[str]] = []
    started = time.monotonic()
    for index, doc in enumerate(docs, 1):
        tokenized.append(bm25_terms(doc.full_text))
        if index % PROGRESS_EVERY == 0:
            print(f"  分词 {index}/{len(docs)}（{time.monotonic() - started:.1f}s）", file=sys.stderr)
    return tokenized


def build_vocab(tokenized: list[list[str]]) -> tuple[list[str], dict[str, int], list[list[int]], list[float]]:
    """词表（字典序）、token→id、文档 token id 序列、idf。"""
    vocab = sorted({token for tokens in tokenized for token in tokens})
    term_id = {token: index for index, token in enumerate(vocab)}
    for token in vocab:
        byte_len = len(token.encode("utf-8"))
        if byte_len > TOKEN_LEN_LIMIT:
            raise BuildError(f"token {token[:40]!r} 的 UTF-8 长度 {byte_len} 超出 u16 上限")

    doc_token_ids: list[list[int]] = []
    df = [0] * len(vocab)
    for tokens in tokenized:
        ids = [term_id[token] for token in tokens]
        doc_token_ids.append(ids)
        for tid in set(ids):
            df[tid] += 1

    n_docs = len(doc_token_ids)
    idf = [math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5)) for count in df]
    return vocab, term_id, doc_token_ids, idf


def _top_k_per_src(best: dict[int, list[tuple[float, int, int]]], top_k: int) -> dict[int, list[tuple[float, int, int]]]:
    """每个 src 只留 (weight, dst, order) 最大的 top_k 条（同分按 dst、order 升序）。"""
    trimmed: dict[int, list[tuple[float, int, int]]] = {}
    for src, rows in best.items():
        rows.sort(key=lambda row: (-row[0], row[1], row[2]))
        trimmed[src] = rows[:top_k]
    return trimmed


def count_edges(
    doc_token_ids: list[list[int]],
    vocab_size: int,
    *,
    orders: tuple[int, ...] = ORDERS,
    min_df: int = MIN_DF,
    top_k: int = TOP_K,
    ppmi_shift: float = PPMI_SHIFT,
    smoothing: float = SMOOTHING,
) -> list[tuple[int, int, int, float]]:
    """统计有序共现边（PPMI + min_df + 每 src top_k），返回按 (src,dst,order) 升序的边。

    对每个 order 独立统计：`c(a,b)` 是该阶有序共现计数，`N_pair` 是该阶总对数，
    `c(a) = Σ_b c(a,b)`、`c(b) = Σ_a c(a,b)`；`pmi = ln(c^s · N_pair / (c(a)^s · c(b)^s))`，
    `weight = max(0, pmi - ln(ppmi_shift))`。按阶分别取 top_k 再合并仍是全局 top_k：
    某个阶里排不进前 top_k 的边，在合并集里也不可能进前 top_k。
    """
    shift_log = math.log(ppmi_shift)
    merged: dict[int, list[tuple[float, int, int]]] = {}
    for order in orders:
        counts: dict[int, int] = {}
        out_count: dict[int, int] = {}
        in_count: dict[int, int] = {}
        total_pairs = 0
        started = time.monotonic()
        for tokens in doc_token_ids:
            limit = len(tokens) - order
            for pos in range(limit):
                src = tokens[pos]
                dst = tokens[pos + order]
                key = src * vocab_size + dst
                counts[key] = counts.get(key, 0) + 1
                out_count[src] = out_count.get(src, 0) + 1
                in_count[dst] = in_count.get(dst, 0) + 1
                total_pairs += 1
        if total_pairs == 0:
            print(f"  order={order}: 语料里没有该阶共现，跳过", file=sys.stderr)
            continue

        best: dict[int, list[tuple[float, int, int]]] = {}
        for key, count in counts.items():
            if count < min_df:
                continue
            src, dst = divmod(key, vocab_size)
            pmi = math.log(
                (count**smoothing) * total_pairs
                / ((out_count[src] ** smoothing) * (in_count[dst] ** smoothing))
            )
            weight = pmi - shift_log
            if weight <= 0.0:
                continue  # 零权边对打分贡献恒为 0，不落盘
            best.setdefault(src, []).append((weight, dst, order))
        del counts, out_count, in_count

        trimmed = _top_k_per_src(best, top_k)
        for src, rows in trimmed.items():
            merged.setdefault(src, []).extend(rows)
        print(
            f"  order={order}: 对数 {total_pairs}，c>={min_df} 且 weight>0 的候选 {sum(len(v) for v in best.values())}，"
            f"每 src top{top_k} 后保留 {sum(len(v) for v in trimmed.values())}（{time.monotonic() - started:.1f}s）",
            file=sys.stderr,
        )

    final = _top_k_per_src(merged, top_k)
    edges: list[tuple[int, int, int, float]] = []
    for src, rows in final.items():
        for weight, dst, order in rows:
            edges.append((src, dst, order, weight))
    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))
    return edges


# --------------------------------------------------------------------------- 二进制编码


def encode_seq_bin(vocab: list[str], idf: list[float], edges: list[tuple[int, int, int, float]]) -> bytes:
    """`seq.bin`：magic + u32 n + n×(u16 len + UTF-8) + n×f32 idf + u32 m + m×(u32,u32,u8,f32)。"""
    if len(vocab) != len(idf):
        raise BuildError(f"词表长度 {len(vocab)} 与 idf 长度 {len(idf)} 不一致（构建器 bug）")
    buffer = bytearray()
    buffer += MAGIC_SEQ
    buffer += struct.pack("<I", len(vocab))
    for token in vocab:
        payload = token.encode("utf-8")
        buffer += struct.pack("<H", len(payload))
        buffer += payload
    buffer += struct.pack(f"<{len(idf)}f", *idf)
    buffer += struct.pack("<I", len(edges))
    start = len(buffer)
    buffer.extend(b"\x00" * (EDGE_BYTES * len(edges)))
    offset = start
    for src, dst, order, weight in edges:
        EDGE_STRUCT.pack_into(buffer, offset, src, dst, order, weight)
        offset += EDGE_BYTES
    return bytes(buffer)


def encode_doc_tokens(doc_token_ids: list[list[int]]) -> bytes:
    """`doc_tokens.bin`：magic + u32 d + 每篇 (u32 len + len×u32 token id)。"""
    buffer = bytearray()
    buffer += MAGIC_DOC
    buffer += struct.pack("<I", len(doc_token_ids))
    for ids in doc_token_ids:
        buffer += struct.pack("<I", len(ids))
        if ids:
            buffer += struct.pack(f"<{len(ids)}I", *ids)
    return bytes(buffer)


# --------------------------------------------------------------------------- 读工件（ref_score / 测试共用）


@dataclass(frozen=True)
class Artifact:
    """已解析的 seq 工件：词表、idf、边表、文档 token 序列。"""

    dir: Path
    manifest: dict
    vocab: list[str]
    idf: list[float]
    edges: dict[tuple[int, int, int], float]
    doc_ids: list[str]
    doc_tokens: list[list[int]]


def read_seq_bin(path: Path) -> tuple[list[str], list[float], dict[tuple[int, int, int], float]]:
    """解析 `seq.bin`（严格校验长度、字典序、边序，坏文件直接报错）。"""
    payload = path.read_bytes()
    if len(payload) < SEQ_HEADER or payload[: len(MAGIC_SEQ)] != MAGIC_SEQ:
        raise BuildError(f"{path} 不是 seq.bin（magic 不符）")
    offset = len(MAGIC_SEQ)
    (vocab_size,) = struct.unpack_from("<I", payload, offset)
    offset += 4

    vocab: list[str] = []
    for _ in range(vocab_size):
        if offset + 2 > len(payload):
            raise BuildError(f"{path} 截断：词表声明 {vocab_size} 条")
        (byte_len,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        byte_len_end = offset + byte_len
        if byte_len_end > len(payload):
            raise BuildError(f"{path} 截断：token 字节数越界")
        vocab.append(payload[offset:byte_len_end].decode("utf-8"))
        offset = byte_len_end
    if vocab != sorted(vocab):
        raise BuildError(f"{path} 的词表不是字典序")

    idf_bytes = 4 * vocab_size
    if offset + idf_bytes + 4 > len(payload):
        raise BuildError(f"{path} 截断：idf 或边数缺失")
    idf = list(struct.unpack_from(f"<{vocab_size}f", payload, offset))
    offset += idf_bytes
    (edge_count,) = struct.unpack_from("<I", payload, offset)
    offset += 4

    expected_len = offset + EDGE_BYTES * edge_count
    if expected_len != len(payload):
        raise BuildError(f"{path} 长度不符：边数 {edge_count} 期望 {expected_len} 字节，实际 {len(payload)}")
    edges: dict[tuple[int, int, int], float] = {}
    previous: tuple[int, int, int] | None = None
    for _ in range(edge_count):
        src, dst, order, weight = EDGE_STRUCT.unpack_from(payload, offset)
        offset += EDGE_BYTES
        if src >= vocab_size or dst >= vocab_size:
            raise BuildError(f"{path} 的边 ({src},{dst}) 超出词表 {vocab_size}")
        if order not in (1, 2):
            raise BuildError(f"{path} 的边 order={order} 非法（只允许 1 / 2）")
        key = (src, dst, order)
        if previous is not None and key <= previous:
            raise BuildError(f"{path} 的边不是按 (src,dst,order) 严格升序：{key} 在 {previous} 之后")
        previous = key
        edges[key] = weight
    return vocab, idf, edges


def read_doc_tokens_file(artifact_dir: Path) -> tuple[list[str], list[list[int]]]:
    """读 `docs.txt` 与 `doc_tokens.bin`，校验两者一一对应。"""
    docs_path = artifact_dir / "docs.txt"
    tokens_path = artifact_dir / "doc_tokens.bin"
    doc_ids = [line for line in docs_path.read_text(encoding="utf-8").splitlines() if line != ""]
    if len(doc_ids) != len(set(doc_ids)):
        raise BuildError(f"{docs_path} 里有重复 doc id")

    payload = tokens_path.read_bytes()
    if len(payload) < DOC_HEADER or payload[: len(MAGIC_DOC)] != MAGIC_DOC:
        raise BuildError(f"{tokens_path} 不是 doc_tokens.bin（magic 不符）")
    offset = len(MAGIC_DOC)
    (doc_count,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    if doc_count != len(doc_ids):
        raise BuildError(f"{tokens_path} 有 {doc_count} 篇，{docs_path} 有 {len(doc_ids)} 行，不一致")

    doc_tokens: list[list[int]] = []
    for _ in range(doc_count):
        if offset + 4 > len(payload):
            raise BuildError(f"{tokens_path} 截断：第 {len(doc_tokens)} 篇长度缺失")
        (length,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        end = offset + 4 * length
        if end > len(payload):
            raise BuildError(f"{tokens_path} 截断：第 {len(doc_tokens)} 篇 token 越界")
        doc_tokens.append(list(struct.unpack_from(f"<{length}I", payload, offset)))
        offset = end
    if offset != len(payload):
        raise BuildError(f"{tokens_path} 尾部多出 {len(payload) - offset} 字节")
    return doc_ids, doc_tokens


def read_artifact(artifact_dir: Path) -> Artifact:
    """读并校验一个 seq 工件目录（manifest 字段、文件摘要、二进制结构）。"""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BuildError(f"找不到工件清单 {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    for key in ("id", "kind", "method", "weights_origin", "trained_on", "contamination", "runtime", "source_corpus"):
        if key not in manifest:
            raise BuildError(f"{manifest_path} 缺字段 {key!r}")

    files = manifest.get("files") or {}
    for file, digest in files.items():
        path = artifact_dir / file
        if not path.is_file():
            raise BuildError(f"工件清单登记了 {file}，但 {path} 不存在")
        if sha256_file(path) != digest.get("sha256"):
            raise BuildError(f"{path} 的 sha256 与 manifest 不符 —— 工件被改过")

    vocab, idf, edges = read_seq_bin(artifact_dir / "seq.bin")
    doc_ids, doc_tokens = read_doc_tokens_file(artifact_dir)
    for index, ids in enumerate(doc_tokens):
        for tid in ids:
            if tid >= len(vocab):
                raise BuildError(f"第 {index} 篇文档的 token id {tid} 超出词表 {len(vocab)}")
    return Artifact(
        dir=artifact_dir,
        manifest=manifest,
        vocab=vocab,
        idf=idf,
        edges=edges,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
    )


# --------------------------------------------------------------------------- 落盘


def write_artifact(
    staging: Path,
    *,
    dataset_name: str,
    dataset_manifest: dict,
    dataset_manifest_sha256: str,
    vocab: list[str],
    idf: list[float],
    edges: list[tuple[int, int, int, float]],
    doc_ids: list[str],
    doc_token_ids: list[list[int]],
    notes: str = "",
) -> dict:
    """把四个文件写进 staging 目录并返回 manifest（写完即按 manifest 自检）。"""
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "seq.bin").write_bytes(encode_seq_bin(vocab, idf, edges))
    (staging / "docs.txt").write_text("".join(f"{doc_id}\n" for doc_id in doc_ids), encoding="utf-8")
    (staging / "doc_tokens.bin").write_bytes(encode_doc_tokens(doc_token_ids))

    files = {
        name: file_digest(staging / name) for name in ("doc_tokens.bin", "docs.txt", "seq.bin")
    }
    manifest = {
        "id": ARTIFACT_ID,
        "kind": ARTIFACT_KIND,
        "method": ARTIFACT_METHOD,
        "weights_origin": WEIGHTS_ORIGIN,
        "source_corpus": {
            "dataset": dataset_name,
            "fingerprint": dataset_manifest_fingerprint(dataset_manifest),
            "manifest_sha256": dataset_manifest_sha256,
        },
        "params": {
            "orders": list(ORDERS),
            "top_k": TOP_K,
            "window": WINDOW,
            "min_df": MIN_DF,
            "ppmi_shift": PPMI_SHIFT,
            "smoothing": SMOOTHING,
        },
        "counts": {"vocab": len(vocab), "edges": len(edges)},
        "files": files,
        "runtime": {"device": "cpu", "precision": "f32"},
        "trained_on": "语料内统计（无监督，未使用 qrels）",
        "contamination": "无预训练模型参与，无外部语料",
        "license": "由本仓从数据集语料构建",
        "notes": notes,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verified = read_artifact(staging)
    if verified.manifest["counts"] != manifest["counts"]:
        raise BuildError(f"工件写出后自检不一致（构建器 bug）：{staging}")
    if (verified.manifest.get("files") or {}) != files:
        raise BuildError(f"工件写出后文件摘要不一致（构建器 bug）：{staging}")
    return manifest


def _swap_into_place(staging: Path, out_dir: Path) -> None:
    """目录级替换：先把旧目录改名备份，再把 staging 改名就位，最后删备份。"""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if out_dir.exists():
        backup = out_dir.with_name(out_dir.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        out_dir.rename(backup)
    staging.rename(out_dir)
    if backup is not None:
        shutil.rmtree(backup)


# --------------------------------------------------------------------------- 构建入口


@dataclass
class BuildSummary:
    dataset: str
    out_dir: Path
    skipped: bool
    docs: int
    vocab: int
    edges: int
    edges_order1: int
    edges_order2: int
    files: dict[str, dict]
    elapsed: float
    fingerprint: str


def build(name: str, *, data_dir: Path, force: bool = False) -> BuildSummary:
    dataset_manifest, docs, dataset_dir = load_dataset(data_dir, name)
    if not docs:
        raise BuildError(f"数据集 {name} 没有文档（corpus.jsonl 为空），不能构建内核工件")
    out_dir = data_dir / name / "artifacts" / ARTIFACT_ID
    if out_dir.exists() and not force:
        previous = read_artifact(out_dir)
        return BuildSummary(
            dataset=name,
            out_dir=out_dir,
            skipped=True,
            docs=len(previous.doc_ids),
            vocab=len(previous.vocab),
            edges=len(previous.edges),
            edges_order1=sum(1 for key in previous.edges if key[2] == 1),
            edges_order2=sum(1 for key in previous.edges if key[2] == 2),
            files=previous.manifest.get("files") or {},
            elapsed=0.0,
            fingerprint=previous.manifest["source_corpus"]["fingerprint"],
        )

    started = time.monotonic()
    print(f"构建 {name}：{len(docs)} 篇文档 → {out_dir}", file=sys.stderr)
    tokenized = tokenize_docs(docs)
    if not any(tokens for tokens in tokenized):
        raise BuildError(f"数据集 {name} 的文档切不出任何 token（分词口径或语料有问题）")
    vocab, _term_id, doc_token_ids, idf = build_vocab(tokenized)
    print(f"  词表 {len(vocab)}，文档 token 总数 {sum(len(ids) for ids in doc_token_ids)}", file=sys.stderr)
    edges = count_edges(doc_token_ids, len(vocab))
    print(f"  边 {len(edges)}（order1={sum(1 for e in edges if e[2] == 1)}，order2={sum(1 for e in edges if e[2] == 2)}）", file=sys.stderr)

    notes = (
        "文档 token 由 tools/kernel/tokenizer.py（与 Rust bm25_terms 等价的 Python 实现）切好固化；"
        "边为文档内 order=1/2 有序共现的 PPMI（逐阶独立统计 marginals），c>=2 后每 src 保留 top-20，"
        "零权边不落盘（对打分贡献恒为 0）。"
    )
    artifacts_dir = out_dir.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{ARTIFACT_ID}.tmp-", dir=artifacts_dir) as tmp:
        staging = Path(tmp) / ARTIFACT_ID
        manifest = write_artifact(
            staging,
            dataset_name=name,
            dataset_manifest=dataset_manifest,
            dataset_manifest_sha256=sha256_file(dataset_dir / "manifest.json"),
            vocab=vocab,
            idf=idf,
            edges=edges,
            doc_ids=[doc.id for doc in docs],
            doc_token_ids=doc_token_ids,
            notes=notes,
        )
        _swap_into_place(staging, out_dir)

    return BuildSummary(
        dataset=name,
        out_dir=out_dir,
        skipped=False,
        docs=len(docs),
        vocab=len(vocab),
        edges=len(edges),
        edges_order1=sum(1 for edge in edges if edge[2] == 1),
        edges_order2=sum(1 for edge in edges if edge[2] == 2),
        files=manifest["files"],
        elapsed=time.monotonic() - started,
        fingerprint=manifest["source_corpus"]["fingerprint"],
    )


def _print_summary(summary: BuildSummary) -> None:
    if summary.skipped:
        print(
            f"跳过 {summary.dataset}：{summary.out_dir} 已存在（docs={summary.docs} "
            f"vocab={summary.vocab} edges={summary.edges}）；--force 重建"
        )
        return
    print(f"构建完成 {summary.dataset} → {summary.out_dir}")
    print(
        f"  docs={summary.docs} vocab={summary.vocab} edges={summary.edges}"
        f"（order1={summary.edges_order1} order2={summary.edges_order2}）"
    )
    total_bytes = 0
    for name in ("manifest.json", "seq.bin", "docs.txt", "doc_tokens.bin"):
        digest = summary.files.get(name)
        if digest is not None:
            size = digest["bytes"]
            total_bytes += size
            print(f"  {name:<15} {size:>12,} B  sha256={digest['sha256']}")
    total_bytes += (summary.out_dir / "manifest.json").stat().st_size
    print(f"  合计 {total_bytes:,} B，manifest 指纹 {summary.fingerprint}，耗时 {summary.elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_seq.py",
        description="构建 seq 内核 v0 工件（data/<name>/artifacts/seq-v0/）",
    )
    parser.add_argument("dataset", help="数据集名（data/ 下的目录名，如 scifact）")
    parser.add_argument("--force", action="store_true", help="丢弃已有工件并重建")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"规范数据根目录（默认 {DEFAULT_DATA_DIR}）",
    )
    args = parser.parse_args(argv)

    try:
        summary = build(args.dataset, data_dir=args.data_dir.resolve(), force=args.force)
    except BuildError as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
