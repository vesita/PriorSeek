#!/usr/bin/env python3
"""抓取词性（语法）先验因子所需的 gold 标注语料，并校验来源。

两个来源，都是**公开发布、有明确许可、且从未接触过 scifact** 的 gold 词性标注：

| 语料 | 领域 | 许可 | 为什么用它 |
|---|---|---|---|
| CRAFT（`lhunter-lab/CRAFT`） | PubMed Central 开放获取的**生物医学**全文 97 篇 | CC BY 3.0 | 与 scifact 同域；scifact 词表里生医专有词只有它覆盖 |
| UD English-EWT | 通用网络文本 | CC BY-SA 4.0 | 补常见词覆盖；UPOS 17 类齐全 |

**为什么必须锁 commit**：GitHub 的 `archive/<branch>.tar.gz` 字节不稳定（重新打包会变），
所以两条 URL 都钉死到 commit，而不是校验 tarball 的 sha256。

产物（`tools/kernel/.cache/grammar/`，不进 git）：

    CRAFT/structural-annotation/dependency/conllu/*.conllu   97 个
    UD_English-EWT/en_ewt-ud-{train,dev,test}.conllu
    provenance.json     4 项：每个来源的 url / commit / 许可 / 文件数

用法（在 tools/kernel/ 下）：
    uv run python fetch_grammar_corpora.py            # 已存在则跳过
    uv run python fetch_grammar_corpora.py --force    # 重新下载
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache" / "grammar"

SOURCES = (
    {
        "name": "CRAFT",
        "repo": "lhunter-lab/CRAFT",
        "commit": "67ddf409fac787a9765b65dc16338ca237d7ab38",
        "license": "CC BY 3.0",
        "domain": "biomedical (PubMed Central OA, 97 articles)",
        # 只取句法标注（含 UPOS/XPOS）与许可文件：整个仓库 240MB，全文 nxml 用不上
        # 相对归档根的路径（归档根是 `<repo>-<commit>`，不写死）
        "members": [
            "structural-annotation/dependency/",
            "LICENSE.txt",
            "README.md",
        ],
    },
    {
        "name": "UD_English-EWT",
        "repo": "UniversalDependencies/UD_English-EWT",
        "commit": "491be069d36e378f67c29f183699eea88308ea42",
        "license": "CC BY-SA 4.0",
        "domain": "general English web text",
        "members": [
            "en_ewt-ud-train.conllu",
            "en_ewt-ud-dev.conllu",
            "en_ewt-ud-test.conllu",
            "LICENSE.txt",
        ],
    },
)


def download(url: str, dest: Path, attempts: int = 8) -> None:
    """带断点续传的下载：大文件（CRAFT 240MB）在这个网络上会中途断流。"""
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                resume = have > 0 and resp.status == 206
                with dest.open("ab" if resume else "wb") as fh:
                    shutil.copyfileobj(resp, fh, 1 << 20)
            return
        except Exception as err:  # noqa: BLE001 —— 网络异常种类多，统一重试
            print(f"    第 {attempt}/{attempts} 次中断（已有 {have} B）：{err}", file=sys.stderr)
    raise SystemExit(f"下载失败（{attempts} 次）：{url}")


def fetch(source: dict, force: bool) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    out_dir = CACHE / source["name"]
    if out_dir.exists() and not force:
        print(f"  跳过 {source['name']}（已存在 {out_dir}/）", file=sys.stderr)
        return {"name": source["name"], "commit": source["commit"], "license": source["license"],
                "domain": source["domain"], "repo": source["repo"], "skipped": True}

    url = f"https://github.com/{source['repo']}/archive/{source['commit']}.tar.gz"
    tarball = CACHE / f"{source['name']}-{source['commit'][:8]}.tar.gz"
    print(f"  下载 {source['name']}: {url}", file=sys.stderr)
    download(url, tarball)
    print(f"  {tarball.name} {tarball.stat().st_size} B", file=sys.stderr)

    with tarfile.open(tarball) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extraction_filter = tarfile.data_filter
        # 归档根是 `<repo>-<commit>`：剥掉第一层，落到稳定的 `CACHE/<name>/`
        stripped = []
        found = 0
        for m in tf.getmembers():
            rel = m.name.split("/", 1)
            if len(rel) != 2:
                continue
            inner = rel[1]
            if not any(inner == p.rstrip("/") or inner.startswith(p) for p in source["members"]):
                continue
            m.name = inner
            stripped.append(m)
            if inner.endswith(".conllu"):
                found += 1
        if not stripped:
            raise SystemExit(f"{source['name']} 里没有匹配的成员：{source['members']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        tf.extractall(out_dir, members=stripped)
    tarball.unlink()

    return {
        "name": source["name"],
        "repo": source["repo"],
        "commit": source["commit"],
        "license": source["license"],
        "domain": source["domain"],
        "url": url,
        "skipped": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抓取 gold 词性标注语料（CRAFT + UD English-EWT）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args(argv)

    rows = [fetch(s, args.force) for s in SOURCES]

    conllu = sorted((CACHE / "CRAFT/structural-annotation/dependency/conllu").glob("*.conllu"))
    ewt = sorted((CACHE / "UD_English-EWT").glob("en_ewt-ud-*.conllu"))
    if len(conllu) != 97:
        raise SystemExit(f"CRAFT 应有 97 个 conllu，实际 {len(conllu)}")
    if len(ewt) != 3:
        raise SystemExit(f"EWT 应有 3 个 conllu，实际 {len(ewt)}")

    provenance = {"sources": rows, "counts": {"craft_conllu": len(conllu), "ewt_conllu": len(ewt)}}
    (CACHE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"OK：CRAFT {len(conllu)} 篇 / EWT {len(ewt)} 个划分 → {CACHE}\n"
        f"   来源与许可记在 provenance.json",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
