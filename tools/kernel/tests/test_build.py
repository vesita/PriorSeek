"""工件构建测试：二进制格式、manifest 字段、构建确定性、失败路径（全部离线）。"""

from __future__ import annotations

import hashlib
import json
import math
import struct

import pytest

import build_seq
from conftest import built_artifact, make_dataset


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- 二进制格式


def test_build_writes_four_files_and_matching_manifest(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    summary = build_seq.build("tiny", data_dir=data_dir, force=True)
    out = summary.out_dir

    assert sorted(path.name for path in out.iterdir()) == [
        "doc_tokens.bin",
        "docs.txt",
        "manifest.json",
        "seq.bin",
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "seq-v0"
    assert manifest["kind"] == "seq_graph"
    assert manifest["method"] == "ppmi_topk"
    assert manifest["weights_origin"] == "corpus_ppmi"
    assert manifest["runtime"] == {"device": "cpu", "precision": "f32"}
    assert manifest["trained_on"] == "语料内统计（无监督，未使用 qrels）"
    assert "无预训练模型" in manifest["contamination"]
    assert manifest["license"]
    assert manifest["params"] == {
        "orders": [1, 2],
        "top_k": 20,
        "window": 8,
        "min_df": 2,
        "ppmi_shift": 5,
        "smoothing": 0.75,
    }
    assert manifest["counts"] == {"vocab": summary.vocab, "edges": summary.edges}

    # files 摘要 = 实际内容（`KernelArtifact::from_dir` 会逐项重算并比对）。
    assert set(manifest["files"]) == {"seq.bin", "docs.txt", "doc_tokens.bin"}
    for name, digest in manifest["files"].items():
        assert digest["sha256"] == _digest(out / name)
        assert digest["bytes"] == (out / name).stat().st_size

    # 数据集指纹 = Rust Manifest::fingerprint() 的口径。
    dataset_manifest = json.loads((data_dir / "tiny" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_corpus"]["dataset"] == "tiny"
    assert manifest["source_corpus"]["fingerprint"] == build_seq.dataset_manifest_fingerprint(
        dataset_manifest
    )
    assert len(manifest["source_corpus"]["fingerprint"]) == 64


def test_seq_bin_layout_matches_spec(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)
    payload = (out / "seq.bin").read_bytes()

    assert payload[:8] == b"SLSEQv1\0"
    (vocab_size,) = struct.unpack_from("<I", payload, 8)
    offset = 12
    vocab = []
    for _ in range(vocab_size):
        (length,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        vocab.append(payload[offset : offset + length].decode("utf-8"))
        offset += length
    assert vocab == sorted(vocab)  # 字典序，便于二分查找
    assert vocab == sorted({term for term in vocab})
    idf = struct.unpack_from(f"<{vocab_size}f", payload, offset)
    offset += 4 * vocab_size
    (edge_count,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    assert offset + 13 * edge_count == len(payload)

    previous = None
    saw_order = set()
    for _ in range(edge_count):
        src, dst, order, weight = struct.unpack_from("<IIBf", payload, offset)
        offset += 13
        key = (src, dst, order)
        assert src < vocab_size and dst < vocab_size
        assert order in (1, 2)
        assert weight > 0.0
        assert previous is None or key > previous  # 严格按 (src,dst,order) 升序
        previous = key
        saw_order.add(order)
    assert saw_order and saw_order <= {1, 2}
    assert len(idf) == vocab_size


def test_doc_tokens_and_docs_match_corpus_order_and_tokenizer(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)

    doc_ids, doc_tokens = build_seq.read_doc_tokens_file(out)
    corpus = [
        json.loads(line)
        for line in (data_dir / "tiny" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert doc_ids == [row["id"] for row in corpus]

    vocab, _idf, _edges = build_seq.read_seq_bin(out / "seq.bin")
    expected_tokens = []
    for row in corpus:
        title = row.get("title", "")
        full_text = f"{title}\n{row['text']}" if title else row["text"]
        expected_tokens.append([vocab.index(term) for term in build_seq.bm25_terms(full_text)])
    assert doc_tokens == expected_tokens


def test_idf_matches_formula(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)
    vocab, idf, _edges = build_seq.read_seq_bin(out / "seq.bin")
    _doc_ids, doc_tokens = build_seq.read_doc_tokens_file(out)

    # 手算 df：每篇文档的去重 token。
    df: dict[int, int] = {}
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    n_docs = len(doc_tokens)
    for token_id, count in df.items():
        expected = math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
        assert idf[token_id] == pytest.approx(expected, rel=1e-6)
    assert "alpha" in vocab and "beta" in vocab
    # 所有文档都出现的词 df = N，idf 仍为正（只增不减形式）。
    assert idf[vocab.index("alpha")] > 0.0


# --------------------------------------------------------------------------- 统计口径


def test_count_edges_hand_computed():
    # 词表 3 个 token；两篇文档：[0,1,2] 与 [0,1]。
    # order=1 的有序对：(0,1) 出现 2 次，(1,2) 出现 1 次；N_pair=3。
    # c(0)=2、c(1)=2（作为 dst 的边际）… 用 smoothing=1、shift=1 手算：
    # pmi(0,1) = ln(2 * 3 / (2 * 2)) = ln(1.5)。
    edges = build_seq.count_edges(
        [[0, 1, 2], [0, 1]],
        vocab_size=3,
        orders=(1,),
        min_df=2,
        top_k=20,
        ppmi_shift=1.0,
        smoothing=1.0,
    )
    assert edges == [(0, 1, 1, pytest.approx(math.log(1.5)))]


def test_count_edges_applies_min_df_and_top_k():
    # src=0 指向 1..5，每个 (0,dst) 都出现 2 次；min_df=2 都保留，top_k=2 只留权重最大的。
    # 让 dst=5 的边际最小 ⇒ 权重最大。
    docs = [[0, 1, 0, 1], [0, 2, 0, 2], [0, 3, 0, 3], [0, 4, 0, 4], [0, 5, 0, 5], [0, 5]]
    edges = build_seq.count_edges(
        docs,
        vocab_size=6,
        orders=(1,),
        min_df=2,
        top_k=2,
        ppmi_shift=1.0,
        smoothing=1.0,
    )
    from_zero = [edge for edge in edges if edge[0] == 0]
    assert len(from_zero) == 2  # src=0 有 5 条候选边，top_k=2 截断
    assert all(order == 1 for _src, _dst, order, _weight in from_zero)
    assert all(weight > 0 for _src, _dst, _order, weight in from_zero)
    # 边表整体按 (src,dst,order) 升序。
    assert edges == sorted(edges, key=lambda edge: (edge[0], edge[1], edge[2]))


# --------------------------------------------------------------------------- 确定性


def test_rebuild_is_byte_identical(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    first = build_seq.build("tiny", data_dir=data_dir, force=True)
    payloads = {path.name: path.read_bytes() for path in first.out_dir.iterdir()}
    manifest_first = payloads["manifest.json"]

    second = build_seq.build("tiny", data_dir=data_dir, force=True)
    assert second.skipped is False
    assert sorted(path.name for path in second.out_dir.iterdir()) == sorted(payloads)
    for name, payload in payloads.items():
        assert (second.out_dir / name).read_bytes() == payload, f"{name} 两次构建字节不一致"
    assert (second.out_dir / "manifest.json").read_bytes() == manifest_first


def test_second_build_skips_without_force(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    first = build_seq.build("tiny", data_dir=data_dir, force=True)
    before = {path.name: path.read_bytes() for path in first.out_dir.iterdir()}
    second = build_seq.build("tiny", data_dir=data_dir, force=False)
    assert second.skipped is True
    assert second.docs == first.docs and second.vocab == first.vocab and second.edges == first.edges
    for name, payload in before.items():
        assert (second.out_dir / name).read_bytes() == payload


# --------------------------------------------------------------------------- 失败路径


def test_rejects_tampered_corpus(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    corpus = data_dir / "tiny" / "corpus.jsonl"
    corpus.write_bytes(corpus.read_bytes() + b"\n")
    with pytest.raises(build_seq.BuildError, match="sha256"):
        build_seq.build("tiny", data_dir=data_dir, force=True)


def test_rejects_missing_dataset_manifest(tmp_path):
    with pytest.raises(build_seq.BuildError, match="数据集清单"):
        build_seq.build("nope", data_dir=tmp_path / "data", force=True)


def test_read_artifact_rejects_tampered_file(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)
    seq_path = out / "seq.bin"
    payload = bytearray(seq_path.read_bytes())
    payload[-1] ^= 0xFF
    seq_path.write_bytes(bytes(payload))
    with pytest.raises(build_seq.BuildError, match="sha256"):
        build_seq.read_artifact(out)


def test_read_artifact_rejects_docs_mismatch(tmp_path):
    data_dir = make_dataset(tmp_path / "data")
    out = built_artifact(data_dir)
    # 改 docs.txt 之后需要同步 manifest 摘要，才能走到「docs 与 doc_tokens 数量不一致」的检查。
    (out / "docs.txt").write_text("only-one\n", encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    payload = (out / "docs.txt").read_bytes()
    manifest["files"]["docs.txt"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(build_seq.BuildError, match="不一致"):
        build_seq.read_artifact(out)


def test_empty_corpus_is_rejected(tmp_path):
    data_dir = make_dataset(tmp_path / "data", docs=[])
    with pytest.raises(build_seq.BuildError):
        build_seq.build("tiny", data_dir=data_dir, force=True)
