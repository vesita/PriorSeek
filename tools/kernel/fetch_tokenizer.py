#!/usr/bin/env python3
"""下载并固化一个 LLM tokenizer（不自己造词表）。

用途：`sl-adapters` 的 `bm25-llm` 适配器在运行时用同一个 `tokenizer.json` 做查询侧切分；
文档侧切分也在 Python 侧用同一份文件，因此**双端必须逐 token 一致**（有对拍测试钉住）。

产物（都不进 git，可整目录删除后重下）：

    data/vendor/tokenizers/<id>/tokenizer.json
    data/vendor/tokenizers/<id>/manifest.json     # id / 来源 URL / sha256 / bytes / 词表大小

用法（在 tools/kernel/ 下）：

    uv run python fetch_tokenizer.py qwen2.5-0.5b
    uv run python fetch_tokenizer.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# 只放「成熟、公开、单文件可离线加载」的 tokenizer；sha256 一旦固定就不要改（改了等于换实验变量）。
TOKENIZERS: dict[str, dict] = {
    "qwen2.5-0.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json",
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "note": "Qwen2.5 BPE（151k 词表，中英文通用），用于替换自研词级切分",
        "kind": "bpe",
    },
    "bert-base-uncased": {
        "url": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json",
        "sha256": "",
        "note": "BERT WordPiece（30k 词表，小写，经典 IR 基线）",
        "kind": "wordpiece",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(tid: str, *, data_dir: Path, force: bool = False) -> int:
    if tid not in TOKENIZERS:
        print(f"未知 tokenizer {tid!r}；可选：{', '.join(TOKENIZERS)}", file=sys.stderr)
        return 2
    spec = TOKENIZERS[tid]
    out_dir = data_dir / "vendor" / "tokenizers" / tid
    target = out_dir / "tokenizer.json"
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = sha256_file(target)
        if actual == manifest.get("sha256"):
            print(f"[跳过] {tid} 已存在，sha256={actual[:16]}（--force 重下）", file=sys.stderr)
            return 0
        print(f"[重新下载] {tid} 的 sha256 与 manifest 不符", file=sys.stderr)

    print(f"下载 {spec['url']}", file=sys.stderr)
    with urllib.request.urlopen(spec["url"], timeout=180) as resp:
        payload = resp.read()
    target.write_bytes(payload)

    digest = sha256_file(target)
    pinned = spec["sha256"]
    if pinned and digest != pinned:
        target.unlink()
        raise SystemExit(
            f"{tid} 的 sha256 与钉住的值不符（期望 {pinned}，实际 {digest}）——上游文件被换过，拒绝固化"
        )
    if not pinned:
        print(
            f"[提示] {tid} 没有钉死的 sha256，本次实测值 = {digest}\n"
            f"       如果决定采用，请把该值填进 TOKENIZERS[\"{tid}\"][\"sha256\"]",
            file=sys.stderr,
        )

    # 记录词表规模（用 tokenizers 库读，保证与运行时同一份实现）
    vocab_size = None
    try:
        from tokenizers import Tokenizer  # type: ignore

        vocab_size = Tokenizer.from_file(str(target)).get_vocab_size(with_added_tokens=True)
    except Exception as exc:  # pragma: no cover - 仅用于元数据
        print(f"[提示] 读词表规模失败（{exc}），manifest 里记 null", file=sys.stderr)

    manifest = {
        "id": tid,
        "kind": spec["kind"],
        "url": spec["url"],
        "sha256": digest,
        "bytes": len(payload),
        "vocab_size": vocab_size,
        "note": spec["note"],
        "license": "随上游模型仓库（见 HF 模型卡）",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK {tid}: {len(payload)} B, sha256={digest[:16]}, vocab={vocab_size}", file=sys.stderr)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载并固化 LLM tokenizer")
    parser.add_argument("tokenizer", nargs="?", help="tokenizer id，如 qwen2.5-0.5b")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="列出可用的 tokenizer")
    args = parser.parse_args(argv)
    if args.list or not args.tokenizer:
        for tid, spec in TOKENIZERS.items():
            pinned = spec["sha256"][:16] or "（未钉住）"
            print(f"{tid:20s} {spec['kind']:10s} {pinned}  {spec['note']}")
        return 0
    return fetch(args.tokenizer, data_dir=args.data_dir, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
