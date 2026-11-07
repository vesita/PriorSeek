"""数据集规格：来源、版本、子集口径与 seed。

manifest 里「这份数据从哪来、按什么口径裁的」全部来自这里。改动任何字段（尤其
`archive_sha256`、`pool_size`、`seed`）都会改变产物与 manifest 指纹，等于换了一道题，
所以这里的值是口径的一部分，不是可以随手调的下载参数。
"""

from __future__ import annotations

from dataclasses import dataclass

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
QBQTC_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=C-MTEB%2FQBQTC&config=default&split=test&offset={offset}&length={length}"
)


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的构建规格。

    `kind` 决定用哪条构建流水线：`beir`（原样转换 BEIR zip）、`nq-sub`（按池子裁剪）、
    `qbqtc`（HF rows API 拉配对数据改造成检索任务）。
    """

    name: str
    version: str
    kind: str
    source: tuple[str, ...]
    notes: str = ""
    seed: int | None = None

    # beir / nq-sub：原始 zip 与 zip 内成员（BEIR zip 里带一层以数据集命名的目录）
    archive: str | None = None
    archive_url: str | None = None
    archive_sha256: str | None = None
    corpus_member: str = "corpus.jsonl"
    queries_member: str = "queries.jsonl"
    qrels_member: str = "qrels/test.tsv"

    # nq-sub：池子大小（保留的文档数）
    pool_size: int | None = None

    # qbqtc：HF rows API
    rows_url: str | None = None
    num_rows: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("beir", "nq-sub", "qbqtc"):
            raise ValueError(f"未知 kind：{self.kind}")
        if not self.source:
            raise ValueError(f"数据集 {self.name} 缺 source")
        if self.kind in ("beir", "nq-sub"):
            if not self.archive or not self.archive_url:
                raise ValueError(f"数据集 {self.name} 缺 archive/archive_url")
            if self.kind == "nq-sub" and (self.pool_size is None or self.seed is None):
                raise ValueError(f"数据集 {self.name} 的 nq-sub 必须声明 pool_size 与 seed")
        if self.kind == "qbqtc" and (not self.rows_url or self.num_rows <= 0):
            raise ValueError(f"数据集 {self.name} 的 qbqtc 必须声明 rows_url 与 num_rows")


def beir_spec(name: str, *, version: str, archive_sha256: str, notes: str) -> DatasetSpec:
    """按 UKP BEIR 镜像的固定布局造一个 BEIR 数据集规格。"""
    url = f"{BEIR_BASE}/{name}.zip"
    return DatasetSpec(
        name=name,
        version=version,
        kind="beir",
        source=(url,),
        notes=notes,
        archive=f"{name}.zip",
        archive_url=url,
        archive_sha256=archive_sha256,
    )


SPECS: dict[str, DatasetSpec] = {
    "scifact": beir_spec(
        "scifact",
        version="beir-scifact",
        archive_sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        notes="BEIR scifact 官方测试集：corpus/queries 全量，qrels 只取 qrels/test.tsv（train.tsv 不参与）。",
    ),
    "arguana": beir_spec(
        "arguana",
        version="beir-arguana",
        archive_sha256="cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
        notes=(
            "BEIR arguana 官方测试集：corpus/queries 全量，qrels 只取 qrels/test.tsv。"
            "上游 qrels 有少量行指向 corpus 里不存在的文档，构建时会被丢弃（条数见构建日志）。"
        ),
    ),
    "nq-sub": DatasetSpec(
        name="nq-sub",
        version="beir-nq-sub-pool50000-seed20261107",
        kind="nq-sub",
        source=(f"{BEIR_BASE}/nq.zip",),
        notes=(
            "从 BEIR nq.zip 的测试集构造的子集，不是全量 NQ："
            "qrels/test.tsv 里所有 rel>0 的文档全保留；其余文档按 "
            "sha256(f'{doc_id}:20261107') 升序补足 pool_size=50000；"
            "查询只保留「至少一个相关文档在池内」的，qrels 只保留池内文档。"
            "因此绝对 recall 等指标不能与全量 NQ 的结果比较，只能在同一份子集上横向比。"
        ),
        seed=20261107,
        archive="nq.zip",
        archive_url=f"{BEIR_BASE}/nq.zip",
        archive_sha256="2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752",
        # UKP 镜像里的 NQ 语料实际是未压缩的 corpus.jsonl（1.5 GB），不是文档里说的
        # corpus.jsonl.gz；build_nq_pool 两种后缀都支持，这里写实际成员名。
        corpus_member="corpus.jsonl",
        pool_size=50000,
    ),
    "qbqtc": DatasetSpec(
        name="qbqtc",
        version="hf-cmteb-qbqtc-test-5000",
        kind="qbqtc",
        source=(
            "https://datasets-server.huggingface.co/rows?dataset=C-MTEB%2FQBQTC&config=default&split=test",
            "https://huggingface.co/datasets/C-MTEB/QBQTC",
        ),
        notes=(
            "原本是 C-MTEB/QBQTC 的 query-title 配对数据集（test split 取前 5000 行），"
            "这里被构造成检索任务：corpus=所有不重复 sentence2（作为 title）、queries=所有不重复 sentence1、"
            "qrels=score>=1 的配对且 rel=score；id 为 sha256(文本)[:16]。"
            "每条 query 通常只有一个正例，绝对指标不是标准 IR 基准，只适合同口径横向比较。"
        ),
        rows_url=QBQTC_ROWS_URL,
        num_rows=5000,
    ),
}
