#!/usr/bin/env python3
"""search-lab 规范数据集构建器：下载 / 解压 / 裁剪 / 转换。

用法（在 `tools/datasets/` 下）：

    uv run python build.py scifact
    uv run python build.py nq-sub --force
    uv run python build.py qbqtc --cache-dir /data/cache

产物写到仓库根 `data/<name>/`（corpus.jsonl / queries.jsonl / qrels.tsv / manifest.json），
格式与自检在 `canonical.py`，口径（来源 URL、版本、pool_size、seed）在 `registry.py`。

约定：
- 原始数据缓存在 `tools/datasets/.cache/`（不进 git）；有 sha256 固定值的缓存每次使用前都会校验；
- 已存在的输出默认跳过并自检，`--force` 重建（下载缓存仍复用，只重做转换与落盘）；
- 进度与诊断走 stderr，最终摘要走 stdout；失败时退出码 1。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

import canonical
from registry import SPECS, DatasetSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"

QBQTC_PAGE_LENGTH = 100
DOWNLOAD_CHUNK = 1 << 20
PROGRESS_STEP = 64 << 20


@dataclass
class Built:
    """转换结果：还没排序、还没算摘要的「内存数据集」。"""

    docs: list[dict]
    queries: list[dict]
    qrels: list[tuple[str, str, int]]
    dropped_qrels: int = 0
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- 文本解析


def parse_qrels_tsv(text: str, *, label: str) -> list[tuple[str, str, int]]:
    """解析 BEIR 风格 qrels（`query-id\\tcorpus-id\\tscore`），跳过表头/空行。

    这是**源数据**解析；规范 qrels.tsv 的写出与读取在 `canonical.py`。
    """
    rows: list[tuple[str, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise canonical.BuildError(f"{label} 第 {lineno} 行不是三列：{line!r}")
        qid, docid, rel_text = parts
        if qid in ("qid", "query-id") and docid in ("docid", "corpus-id"):
            continue  # 表头
        try:
            rel = int(rel_text.strip())
        except ValueError as exc:
            raise canonical.BuildError(f"{label} 第 {lineno} 行 rel 不是整数：{rel_text!r}") from exc
        if rel < 0:
            raise canonical.BuildError(f"{label} 第 {lineno} 行 rel 为负数：{rel}")
        rows.append((qid, docid, rel))
    return rows


def _iter_jsonl_handle(handle, *, label: str):
    """逐行解析 JSONL（文件句柄或 StringIO 的迭代器）。"""
    for lineno, raw in enumerate(handle, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise canonical.BuildError(f"{label} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        if not isinstance(row, dict):
            raise canonical.BuildError(f"{label} 第 {lineno} 行不是 JSON 对象")
        yield row


def _iter_jsonl_text(text: str, *, label: str):
    return _iter_jsonl_handle(text.splitlines(), label=label)


def _source_id(row: dict, *, label: str) -> str:
    """BEIR 原生 corpus/queries 用 `_id`；也接受 `id`（本地 fixture 更直观）。"""
    value = row.get("_id", row.get("id"))
    if not isinstance(value, str) or not value:
        raise canonical.BuildError(f"{label} 的行缺 _id/id：{row!r}")
    return value


def _source_text(row: dict, *, label: str) -> str:
    value = row.get("text")
    if not isinstance(value, str):
        raise canonical.BuildError(f"{label} 的行缺 text：{row!r}")
    return value


def _source_title(row: dict) -> str:
    value = row.get("title")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise canonical.BuildError(f"title 不是字符串：{value!r}")
    return value


def _merge_qrels(
    rows: list[tuple[str, str, int]],
    *,
    merge_duplicates: bool,
) -> tuple[list[tuple[str, str, int]], int]:
    """合并同一 (qid, docid) 的重复行。

    BEIR 源数据按约定不该有重复；qbqtc 的配对可以重复出现，取最大 score 更符合
    「相关程度」的语义。返回 (去重后的行, 重复行数)。
    """
    merged: dict[tuple[str, str], int] = {}
    duplicates = 0
    for qid, docid, rel in rows:
        key = (qid, docid)
        if key in merged:
            duplicates += 1
            if not merge_duplicates and merged[key] != rel:
                raise canonical.BuildError(f"qrels 里 ({qid}, {docid}) 重复且 rel 冲突：{merged[key]} vs {rel}")
            merged[key] = max(merged[key], rel)
        else:
            merged[key] = rel
    return [(qid, docid, rel) for (qid, docid), rel in merged.items()], duplicates


# --------------------------------------------------------------------------- 下载与缓存


def _print_progress(done: int, total: int) -> None:
    if total:
        print(
            f"[下载] {done / (1 << 20):.0f}/{total / (1 << 20):.0f} MiB（{done * 100 // total}%）",
            file=sys.stderr,
        )
    else:
        print(f"[下载] {done / (1 << 20):.0f} MiB", file=sys.stderr)


def _download(url: str, dest: Path, *, expected_sha256: str | None) -> None:
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    print(f"[下载] {url}", file=sys.stderr)
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                done = 0
                next_report = PROGRESS_STEP
                with open(part, "wb") as handle:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            _print_progress(done, total)
                            next_report += PROGRESS_STEP
                _print_progress(done, total)
            break
        except requests.RequestException as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < 3:
                print(f"[下载] 第 {attempt} 次失败：{exc}；重试", file=sys.stderr)
                time.sleep(2 * attempt)
    else:
        raise canonical.BuildError(f"下载 {url} 失败：{last_error}")

    actual = canonical.sha256_file(part)
    if expected_sha256 and actual != expected_sha256:
        part.unlink(missing_ok=True)
        raise canonical.BuildError(
            f"下载 {url} 的 sha256 与 registry 固定值不一致（期望 {expected_sha256}，实际 {actual}）；"
            f"上游可能更新了数据。确认口径后更新 registry.py 的 archive_sha256，不要直接绕过校验"
        )
    part.replace(dest)
    print(f"[下载] 完成 {dest}（{dest.stat().st_size / (1 << 20):.1f} MiB）", file=sys.stderr)


def ensure_archive(spec: DatasetSpec, cache_dir: Path) -> Path:
    """下载缓存命中就用缓存（有固定 sha256 时先校验），否则下载。"""
    assert spec.archive is not None and spec.archive_url is not None
    dest = cache_dir / spec.archive
    if dest.is_file():
        if spec.archive_sha256:
            actual = canonical.sha256_file(dest)
            if actual != spec.archive_sha256:
                raise canonical.BuildError(
                    f"缓存 {dest} 的 sha256 与 registry 不一致（期望 {spec.archive_sha256}，实际 {actual}）；"
                    f"删除该缓存文件后重试"
                )
        print(f"[缓存] 复用 {dest}", file=sys.stderr)
        return dest
    _download(spec.archive_url, dest, expected_sha256=spec.archive_sha256)
    return dest


def _retry_wait(response: requests.Response, attempt: int) -> float:
    """429/5xx 的等待时间：优先用 Retry-After，否则指数退避（上限 60s）。"""
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass
    return min(60.0, 4.0 * 2 ** (attempt - 1))


def _http_json(url: str) -> dict:
    """GET 一个 JSON 对象；对 429/5xx 与网络错误重试（公共 API，避免被限流打断构建）。"""
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            response = requests.get(url, timeout=(30, 120))
            if response.status_code == 429 or response.status_code >= 500:
                last_error = requests.HTTPError(f"HTTP {response.status_code} {response.reason}", response=response)
                wait = _retry_wait(response, attempt)
                print(f"[请求] {url} 返回 {response.status_code}；{wait:.0f}s 后重试", file=sys.stderr)
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise canonical.BuildError(f"{url} 返回的不是 JSON 对象")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 5:
                wait = 2.0 * attempt
                print(f"[请求] 第 {attempt} 次失败：{exc}；{wait:.0f}s 后重试", file=sys.stderr)
                time.sleep(wait)
    raise canonical.BuildError(f"请求 {url} 失败（重试 5 次）：{last_error}")


def _find_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    """BEIR zip 里文件带一层数据集目录（如 `scifact/corpus.jsonl`），按后缀定位。"""
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and (name == member or name.endswith("/" + member))
    ]
    if not matches:
        raise canonical.BuildError(f"{archive.filename} 里找不到 {member}（有：{archive.namelist()[:5]}…）")
    if len(matches) > 1:
        raise canonical.BuildError(f"{archive.filename} 里 {member} 有多个候选：{matches}")
    return matches[0]


def _read_zip_text(archive: zipfile.ZipFile, member: str) -> str:
    name = _find_zip_member(archive, member)
    return archive.read(name).decode("utf-8")


# --------------------------------------------------------------------------- BEIR 转换


def convert_beir(corpus, queries, qrels) -> Built:
    """把 BEIR 的 corpus/queries/qrels 转成规范三元组。

    上游偶有 qrels 指向不存在的文档、或指向 queries.jsonl 里没有的查询；这类行会让
    Rust 侧 `verify_strict` 失败，所以在这里显式丢弃并记账（不静默）。
    """
    docs: dict[str, dict] = {}
    for label, row in (("corpus", row) for row in corpus):
        doc_id = _source_id(row, label=label)
        if doc_id in docs:
            raise canonical.BuildError(f"corpus 出现重复 id：{doc_id}")
        docs[doc_id] = canonical.doc_row(doc_id, _source_text(row, label=label), _source_title(row))

    queries_by_id: dict[str, dict] = {}
    for row in queries:
        query_id = _source_id(row, label="queries")
        if query_id in queries_by_id:
            raise canonical.BuildError(f"queries 出现重复 id：{query_id}")
        queries_by_id[query_id] = canonical.query_row(query_id, _source_text(row, label="queries"))

    merged, duplicate_rows = _merge_qrels(qrels, merge_duplicates=False)
    kept: list[tuple[str, str, int]] = []
    dropped_doc_rows = 0
    missing_query_ids: set[str] = set()
    for qid, docid, rel in merged:
        if qid not in queries_by_id:
            missing_query_ids.add(qid)
            continue
        if docid not in docs:
            dropped_doc_rows += 1
            continue
        kept.append((qid, docid, rel))

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"qrels 有 {duplicate_rows} 行重复配对，已合并（同配对取最大 rel）")
    if dropped_doc_rows:
        warnings.append(f"丢弃 {dropped_doc_rows} 条指向不存在文档的 qrels 行（上游数据不一致）")
    if missing_query_ids:
        warnings.append(f"丢弃 {len(missing_query_ids)} 个查询的 qrels（查询不在 queries.jsonl 里）")

    return Built(
        docs=list(docs.values()),
        queries=list(queries_by_id.values()),
        qrels=kept,
        dropped_qrels=dropped_doc_rows,
        warnings=warnings,
        stats={"duplicate_qrels": duplicate_rows, "missing_query_qrels": len(missing_query_ids)},
    )


def build_beir(spec: DatasetSpec, cache_dir: Path) -> Built:
    archive_path = ensure_archive(spec, cache_dir)
    with zipfile.ZipFile(archive_path) as archive:
        corpus = list(_iter_jsonl_text(_read_zip_text(archive, spec.corpus_member), label=f"{spec.name}/corpus"))
        queries = list(_iter_jsonl_text(_read_zip_text(archive, spec.queries_member), label=f"{spec.name}/queries"))
        qrels = parse_qrels_tsv(_read_zip_text(archive, spec.qrels_member), label=f"{spec.name}/qrels")
    return convert_beir(corpus, queries, qrels)


# --------------------------------------------------------------------------- NQ 子集


def _open_jsonl(path: Path):
    """打开 JSONL 源文件：`.gz` 走 gzip，其余按纯文本（BEIR NQ 镜像两种都见过）。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _pool_hash(doc_id: str, seed: int) -> int:
    """抽样键：sha256(f"{doc_id}:{seed}")，按它升序取文档。"""
    digest = hashlib.sha256(f"{doc_id}:{seed}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def build_nq_pool(
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    *,
    pool_size: int,
    seed: int,
) -> Built:
    """构造 NQ 声明式子集（两遍扫描语料，不在内存里存全量 NQ）。

    1. 第一遍：记下 qrels 里所有 rel>0 的文档全文；其余文档按 `_pool_hash` 维护大小为
       `pool_size` 的最小堆（并列时按 doc id 升序，保证与输入顺序无关）。
    2. 取 `pool_size - 正例文档数` 个最小哈希的文档作为补充；第二遍扫描补全文。
    3. 查询只保留「至少一个正例文档在池内」的；qrels 过滤到池内文档。

    正例文档超过 `pool_size` 时池子会比 `pool_size` 大（正例全保留优先）。
    """
    if pool_size <= 0:
        raise canonical.BuildError(f"pool_size 必须是正整数：{pool_size}")
    qrels_rows = parse_qrels_tsv(qrels_path.read_text(encoding="utf-8"), label=str(qrels_path))
    qrels_rows, duplicate_rows = _merge_qrels(qrels_rows, merge_duplicates=False)
    positive_docs = {docid for _qid, docid, rel in qrels_rows if rel > 0}

    positives: dict[str, dict] = {}
    heap: list[tuple[int, str]] = []  # (-hash, doc_id)，堆顶是「当前最差」的候选
    total_docs = 0
    with _open_jsonl(corpus_path) as handle:
        for row in _iter_jsonl_handle(handle, label=str(corpus_path)):
            total_docs += 1
            doc_id = _source_id(row, label=str(corpus_path))
            if doc_id in positive_docs:
                if doc_id in positives:
                    raise canonical.BuildError(f"corpus 出现重复 id：{doc_id}")
                positives[doc_id] = canonical.doc_row(
                    doc_id, _source_text(row, label=str(corpus_path)), _source_title(row)
                )
                continue
            negated = -_pool_hash(doc_id, seed)
            if len(heap) < pool_size:
                heapq.heappush(heap, (negated, doc_id))
            else:
                worst_hash, worst_id = -heap[0][0], heap[0][1]
                candidate_hash = -negated
                if candidate_hash < worst_hash or (candidate_hash == worst_hash and doc_id < worst_id):
                    heapq.heapreplace(heap, (negated, doc_id))

    if total_docs < pool_size:
        raise canonical.BuildError(
            f"{corpus_path} 只有 {total_docs} 篇文档，不够 pool_size={pool_size}（数据不完整？）"
        )
    fill_count = max(0, pool_size - len(positives))
    if fill_count > len(heap):
        raise canonical.BuildError(
            f"正例文档 {len(positives)} 篇 + 候选 {len(heap)} 篇不够 pool_size={pool_size}"
        )
    # 堆里保留的是「最小的 pool_size 个哈希」，按 (哈希, doc id) 升序取前 fill_count 个。
    chosen = sorted(heap, key=lambda item: (-item[0], item[1]))[:fill_count]
    fill_ids = {doc_id for _negated, doc_id in chosen}

    pool: dict[str, dict] = dict(positives)
    if fill_ids:
        with _open_jsonl(corpus_path) as handle:
            for row in _iter_jsonl_handle(handle, label=str(corpus_path)):
                doc_id = _source_id(row, label=str(corpus_path))
                if doc_id in fill_ids and doc_id not in pool:
                    pool[doc_id] = canonical.doc_row(
                        doc_id, _source_text(row, label=str(corpus_path)), _source_title(row)
                    )
                    if len(pool) == len(positives) + fill_count:
                        break

    queries_by_id: dict[str, dict] = {}
    with open(queries_path, "r", encoding="utf-8") as handle:
        for row in _iter_jsonl_handle(handle, label=str(queries_path)):
            query_id = _source_id(row, label=str(queries_path))
            if query_id in queries_by_id:
                raise canonical.BuildError(f"queries 出现重复 id：{query_id}")
            queries_by_id[query_id] = canonical.query_row(query_id, _source_text(row, label=str(queries_path)))

    positive_in_pool = {qid for qid, docid, rel in qrels_rows if rel > 0 and docid in pool}
    kept_queries = [row for qid, row in queries_by_id.items() if qid in positive_in_pool]
    kept_qrels = [
        (qid, docid, rel)
        for qid, docid, rel in qrels_rows
        if qid in positive_in_pool and docid in pool
    ]

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"qrels 有 {duplicate_rows} 行重复配对，已合并（同配对取最大 rel）")
    missing_positives = positive_docs - set(positives)
    if missing_positives:
        warnings.append(f"{len(missing_positives)} 个 rel>0 的文档不在 corpus 里，无法保留")
    if rel_rows_dropped := len(qrels_rows) - len(kept_qrels):
        warnings.append(f"qrels 丢弃 {rel_rows_dropped} 行（查询没有正例在池内，或文档不在池内）")

    return Built(
        docs=list(pool.values()),
        queries=kept_queries,
        qrels=kept_qrels,
        dropped_qrels=len(qrels_rows) - len(kept_qrels),
        warnings=warnings,
        stats={
            "total_corpus_docs": total_docs,
            "pool_docs": len(pool),
            "pool_positives": len(positives),
            "pool_fill": fill_count,
            "queries_total": len(queries_by_id),
            "queries_in_pool": len(kept_queries),
        },
    )


def build_nq(spec: DatasetSpec, cache_dir: Path) -> Built:
    archive_path = ensure_archive(spec, cache_dir)
    extracted = cache_dir / f"{spec.name}-extracted"
    targets = {
        spec.corpus_member: extracted / spec.corpus_member,
        spec.queries_member: extracted / spec.queries_member,
        spec.qrels_member: extracted / spec.qrels_member,
    }
    if not all(path.is_file() for path in targets.values()):
        print(f"[解压] {archive_path} → {extracted}", file=sys.stderr)
        with zipfile.ZipFile(archive_path) as archive:
            for member, path in targets.items():
                name = _find_zip_member(archive, member)
                path.parent.mkdir(parents=True, exist_ok=True)
                part = path.with_name(path.name + ".part")
                with archive.open(name) as source, open(part, "wb") as sink:
                    shutil.copyfileobj(source, sink, length=DOWNLOAD_CHUNK)
                part.replace(path)
    assert spec.pool_size is not None and spec.seed is not None
    return build_nq_pool(
        targets[spec.corpus_member],
        targets[spec.queries_member],
        targets[spec.qrels_member],
        pool_size=spec.pool_size,
        seed=spec.seed,
    )


# --------------------------------------------------------------------------- QBQTC 转换


def text_id(text: str) -> str:
    """稳定可复现的 id：sha256(文本) 前 16 个十六进制字符（同文本必同 id）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_qbqtc_row(item: dict, *, url: str) -> dict:
    row = item.get("row") if isinstance(item, dict) else None
    if not isinstance(row, dict):
        raise canonical.BuildError(f"{url} 的 rows 项没有 row 对象：{item!r}")
    return {
        "sentence1": row.get("sentence1"),
        "sentence2": row.get("sentence2"),
        "score": row.get("score"),
    }


def fetch_qbqtc_rows(spec: DatasetSpec, cache_dir: Path) -> list[dict]:
    """从 HF datasets-server rows API 按 length=100 分页拉前 `num_rows` 行。

    页级缓存：每页存成 `pages/<offset>.json`，构建失败（比如被 429 打断）后重跑会从断点
    续拉，不需要从第一页重来；页缓存内容就是 `sentence1`/`sentence2`/`score` 三个字段。
    """
    assert spec.rows_url is not None
    pages_dir = cache_dir / spec.name / "pages"
    rows: list[dict] = []
    offset = 0
    fetched_pages = 0
    while offset < spec.num_rows:
        page_path = pages_dir / f"{offset:05d}.json"
        if page_path.is_file():
            page_rows = json.loads(page_path.read_text(encoding="utf-8"))
            if not isinstance(page_rows, list):
                raise canonical.BuildError(f"页缓存 {page_path} 不是 JSON 数组；删除后重试")
        else:
            url = spec.rows_url.format(offset=offset, length=QBQTC_PAGE_LENGTH)
            payload = _http_json(url)
            raw_page = payload.get("rows") or []
            page_rows = [_normalize_qbqtc_row(item, url=url) for item in raw_page]
            if not page_rows:
                raise canonical.BuildError(
                    f"HF rows API 在 offset={offset} 返回空页（num_rows_total={payload.get('num_rows_total')!r}），"
                    f"拉不到声明的 {spec.num_rows} 行"
                )
            pages_dir.mkdir(parents=True, exist_ok=True)
            part = page_path.with_name(page_path.name + ".part")
            part.write_text(json.dumps(page_rows, ensure_ascii=False), encoding="utf-8")
            part.replace(page_path)
            fetched_pages += 1
            time.sleep(0.2)  # 公共 API：页间留间隔，避免触发限流
        rows.extend(page_rows)
        offset += len(page_rows)
        print(f"[拉取] {min(offset, spec.num_rows)}/{spec.num_rows} 行", file=sys.stderr)
    if fetched_pages == 0:
        print(f"[缓存] 复用 {pages_dir}（{len(rows)} 行）", file=sys.stderr)
    return rows[: spec.num_rows]


def convert_qbqtc(rows: list[dict]) -> Built:
    """把 query-title 配对改造成检索任务：sentence2 当 title，sentence1 当 query。"""
    docs: dict[str, dict] = {}
    queries: dict[str, dict] = {}
    scores: dict[tuple[str, str], int] = {}
    skipped_empty = 0
    duplicate_pairs = 0
    for row in rows:
        sentence1 = row.get("sentence1")
        sentence2 = row.get("sentence2")
        score = row.get("score")
        if not isinstance(sentence1, str) or not isinstance(sentence2, str):
            raise canonical.BuildError(f"qbqtc 行缺 sentence1/sentence2：{row!r}")
        if not isinstance(score, int) or isinstance(score, bool) or score < 0:
            raise canonical.BuildError(f"qbqtc 行 score 不是非负整数：{row!r}")
        if not sentence1.strip() or not sentence2.strip():
            skipped_empty += 1
            continue
        query_id = text_id(sentence1)
        doc_id = text_id(sentence2)
        queries[query_id] = canonical.query_row(query_id, sentence1)
        docs[doc_id] = canonical.doc_row(doc_id, "", sentence2)  # title-only 语料
        if score >= 1:
            key = (query_id, doc_id)
            if key in scores:
                duplicate_pairs += 1
            scores[key] = max(scores.get(key, 0), score)

    warnings: list[str] = []
    if skipped_empty:
        warnings.append(f"跳过 {skipped_empty} 行空文本（sentence1/sentence2 为空）")
    if duplicate_pairs:
        warnings.append(f"qrels 有 {duplicate_pairs} 个重复配对，已合并（同配对取最大 score）")

    return Built(
        docs=list(docs.values()),
        queries=list(queries.values()),
        qrels=[(qid, docid, rel) for (qid, docid), rel in scores.items()],
        warnings=warnings,
        stats={
            "rows": len(rows),
            "unique_docs": len(docs),
            "unique_queries": len(queries),
            "scored_pairs": len(scores),
            "skipped_empty": skipped_empty,
        },
    )


def build_qbqtc(spec: DatasetSpec, cache_dir: Path) -> Built:
    return convert_qbqtc(fetch_qbqtc_rows(spec, cache_dir))


# --------------------------------------------------------------------------- 构建入口


def _built_by(spec: DatasetSpec) -> str:
    text = f"tools/datasets/build.py（uv run python build.py {spec.name}）"
    if spec.kind == "nq-sub":
        text += f"；pool_size={spec.pool_size}，seed={spec.seed}，rel>0 的文档全保留"
    elif spec.kind == "qbqtc":
        text += f"；HF rows API 分页 length={QBQTC_PAGE_LENGTH}，取前 {spec.num_rows} 行"
    return text


def build_payload(spec: DatasetSpec, cache_dir: Path) -> Built:
    if spec.kind == "beir":
        return build_beir(spec, cache_dir)
    if spec.kind == "nq-sub":
        return build_nq(spec, cache_dir)
    if spec.kind == "qbqtc":
        return build_qbqtc(spec, cache_dir)
    raise canonical.BuildError(f"未知数据集类型：{spec.kind}")


def _swap_into_place(staging: Path, target: Path, data_dir: Path, name: str) -> None:
    """把自检通过的临时目录整体换成 data/<name>（只允许动 data/<name> 这一处）。"""
    if target.parent != data_dir or target.name != name:
        raise canonical.BuildError(f"拒绝把数据写到 {target}：只允许 {data_dir / name}")
    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)


def build_one(spec: DatasetSpec, *, data_dir: Path, cache_dir: Path, force: bool) -> tuple[Path, dict, bool]:
    """构建一个数据集，返回 (输出目录, manifest, 是否跳过)。"""
    out_dir = data_dir / spec.name
    if (out_dir / canonical.MANIFEST_FILE).is_file() and not force:
        manifest = canonical.verify_dataset(out_dir)  # 已存在也要自检：坏了就别当成可用
        return out_dir, manifest, True

    built = build_payload(spec, cache_dir)
    for warning in built.warnings:
        print(f"[注意] {warning}", file=sys.stderr)
    for key, value in built.stats.items():
        print(f"[统计] {key}={value}", file=sys.stderr)

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{spec.name}.tmp-", dir=data_dir) as tmp:
        staging = Path(tmp) / spec.name
        manifest = canonical.write_dataset(
            staging,
            name=spec.name,
            version=spec.version,
            source=spec.source,
            built_by=_built_by(spec),
            seed=spec.seed,
            notes=spec.notes,
            docs=built.docs,
            queries=built.queries,
            qrels=built.qrels,
        )
        _swap_into_place(staging, out_dir, data_dir, spec.name)

    verified = canonical.verify_dataset(out_dir)
    if verified != manifest:
        raise canonical.BuildError(f"数据集 {spec.name} 写入后再自检不一致（构建器 bug）")
    return out_dir, verified, False


def _summary(manifest: dict, out_dir: Path, elapsed: float) -> None:
    counts = manifest["counts"]
    print(f"构建完成 {manifest['name']} v{manifest['version']} → {out_dir}")
    print(
        f"  docs={counts['docs']} queries={counts['queries']} "
        f"qrels={counts['qrels']} scored_queries={counts['scored_queries']}"
    )
    print(f"  manifest 指纹 {canonical.manifest_fingerprint(manifest)}")
    print(f"  耗时 {elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="构建 search-lab 规范数据集（下载 / 解压 / 裁剪 / 转换 + manifest）",
    )
    parser.add_argument("name", choices=sorted(SPECS), help="数据集名")
    parser.add_argument("--force", action="store_true", help="丢弃已有输出并重建（下载缓存仍复用）")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"原始数据缓存目录（默认 {DEFAULT_CACHE_DIR}）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"规范数据输出根目录（默认 {DEFAULT_DATA_DIR}）",
    )
    args = parser.parse_args(argv)

    spec = SPECS[args.name]
    started = time.monotonic()
    try:
        out_dir, manifest, skipped = build_one(
            spec, data_dir=args.data_dir.resolve(), cache_dir=args.cache_dir.resolve(), force=args.force
        )
    except canonical.BuildError as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    if skipped:
        counts = manifest["counts"]
        print(
            f"跳过 {spec.name}：{out_dir} 已存在且自检通过"
            f"（docs={counts['docs']} queries={counts['queries']} qrels={counts['qrels']}）；"
            f"--force 重建"
        )
        return 0
    _summary(manifest, out_dir, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
