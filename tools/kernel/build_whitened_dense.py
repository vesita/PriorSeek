import struct
import numpy as np
import json
import hashlib
from pathlib import Path

def build_whitened_dense(dataset: str = "scifact"):
    root = Path(__file__).resolve().parent.parent.parent
    src_dir = root / "data" / dataset / "artifacts" / "dense-v0"
    out_dir = root / "data" / dataset / "artifacts" / "dense-whitened-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] 读取 dense-v0 文档向量...")
    with open(src_dir / "dense.bin", "rb") as f:
        header = f.read(16)
        magic, num_docs, dim = struct.unpack("<8sII", header)
        assert magic == b"SLDENSv1"
        data = f.read()
        doc_vectors = np.frombuffer(data, dtype=np.float32).reshape(num_docs, dim).copy()

    print(f"[*] 文档向量形状: {doc_vectors.shape}")

    print("[*] 读取 query_embeddings.bin...")
    with open(src_dir / "query_embeddings.bin", "rb") as f:
        header = f.read(16)
        magic, num_queries, dim = struct.unpack("<8sII", header)
        assert magic == b"SLQEMBv1"
        data = f.read()
        query_vectors = np.frombuffer(data, dtype=np.float32).reshape(num_queries, dim).copy()

    print(f"[*] 查询向量形状: {query_vectors.shape}")

    print("[*] 计算语义流形各向异性均值 mu 与协方差矩阵 Sigma...")
    mu = np.mean(doc_vectors, axis=0, keepdims=True)
    centered_docs = doc_vectors - mu

    # 计算协方差矩阵 (384, 384)
    cov = np.cov(centered_docs, rowvar=False)

    print("[*] 奇异值分解 (SVD) 与对角白化核构建...")
    u, s, vh = np.linalg.svd(cov)
    # 正则化逆平方根，避免微小特征值数值发散
    eps = 1e-4
    inv_s = 1.0 / np.sqrt(s + eps)
    # 白化矩阵: W = u * diag(inv_s)
    W = np.dot(u, np.diag(inv_s))

    print("[*] 对文档与查询向量施加流形白化投影并做 L2 归一化...")
    white_docs = np.dot(centered_docs, W)
    white_docs = white_docs / np.linalg.norm(white_docs, axis=1, keepdims=True)

    centered_queries = query_vectors - mu
    white_queries = np.dot(centered_queries, W)
    white_queries = white_queries / np.linalg.norm(white_queries, axis=1, keepdims=True)

    # 导出 white_docs 到 dense.bin
    out_dense_bin = out_dir / "dense.bin"
    with open(out_dense_bin, "wb") as f:
        f.write(struct.pack("<8sII", b"SLDENSv1", num_docs, dim))
        f.write(white_docs.astype(np.float32).tobytes())

    # 导出 white_queries 到 query_embeddings.bin
    out_query_bin = out_dir / "query_embeddings.bin"
    with open(out_query_bin, "wb") as f:
        f.write(struct.pack("<8sII", b"SLQEMBv1", num_queries, dim))
        f.write(white_queries.astype(np.float32).tobytes())

    # 复制其他辅助元数据
    import shutil
    shutil.copy(src_dir / "doc_ids.txt", out_dir / "doc_ids.txt")
    shutil.copy(src_dir / "query_ids.txt", out_dir / "query_ids.txt")
    shutil.copy(src_dir / "query_texts.json", out_dir / "query_texts.json")

    # 生成带 sha256 的 manifest
    manifest = {
        "name": "dense-whitened-v1",
        "files": {
            "dense.bin": hashlib.sha256(open(out_dense_bin, "rb").read()).hexdigest(),
            "query_embeddings.bin": hashlib.sha256(open(out_query_bin, "rb").read()).hexdigest(),
        },
        "description": "384-dim SVD Anisotropy-Whitened Normalized Continuous Vectors"
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] 白化工件构建完成！保存在 {out_dir}")

if __name__ == "__main__":
    build_whitened_dense("scifact")
