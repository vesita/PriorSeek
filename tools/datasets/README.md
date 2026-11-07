# 数据集构建器（tools/datasets）

把公开数据下载 / 解压 / 裁剪 / 转换成 Rust 侧消费的规范目录 `data/<name>/`。
格式与 `manifest.json` 的唯一权威是 `crates/sl-core/src/canonical.rs`；本目录只负责
「数据从哪来、按什么口径裁」，产物必须能通过：

```bash
cargo run -q -p sl-cli -- dataset verify <name>
```

## 快速开始

```bash
cd tools/datasets
uv run pytest -q                 # 纯离线测试（fixtures，不联网）
uv run python build.py scifact   # 依次可建：scifact / arguana / qbqtc / nq-sub
uv run python build.py nq-sub --force   # 已存在则默认跳过；--force 重建
cd ../.. && cargo run -q -p sl-cli -- dataset verify scifact
```

原始数据缓存在 `tools/datasets/.cache/`（不进 git，可整目录删除）；`--cache-dir` 可换位置。
`--data-dir` 可换输出根目录（默认仓库根 `data/`）。

## 数据集一览（数字来自实际构建）

| 名称 | 来源 | 文档 | 查询 | qrels 行 | 计分查询 | 首次构建* |
|---|---|---:|---:|---:|---:|---:|
| `scifact` | BEIR `scifact.zip` | 5183 | 1109 | 339 | 300 | 0.2s |
| `arguana` | BEIR `arguana.zip` | 8674 | 1406 | 1401 | 1401 | 0.3s |
| `qbqtc` | HF `C-MTEB/QBQTC` test rows API | 4889 | 4924 | 3785 | 3749 | 120.7s |
| `nq-sub` | BEIR `nq.zip`（全量 2,681,468 篇） | 50000 | 3452 | 4201 | 3452 | 20.0s |

\* 指转换 + 落盘耗时（缓存已就绪时）；不含首次网络下载。`nq-sub` 缓存就绪后重建约 14.8s。

产物磁盘占用：`scifact` 8.1 MB、`arguana` 11.3 MB、`qbqtc` 1.0 MB、`nq-sub` 27.6 MB。
`nq.zip` 下载缓存 498 MB，解压出的 NQ 语料 1.5 GB；删掉缓存后重建需要重新下载。

## 各数据集口径与已知限制

### scifact

- 口径：BEIR 官方测试集，`corpus.jsonl` / `queries.jsonl` 全量转换，qrels **只取 `qrels/test.tsv`**
  （`train.tsv` 不参与）。
- 限制：`queries.jsonl` 保留 BEIR 的 1109 条查询（含 train/test 全部），但 qrels 只覆盖
  test 的 300 个计分查询；指标只在这 300 个查询上做宏平均，其余查询不参与。qrels 为二值相关。

### arguana

- 口径：BEIR 官方测试集，同样只取 `qrels/test.tsv`。
- 限制：上游 qrels 共 1406 行，其中 **5 行指向 corpus 里不存在的文档**，构建时丢弃并打
  `[注意]` 到 stderr（否则 Rust `verify_strict` 会判失败）；最终 1401 行。qrels 为二值相关。

### qbqtc

- 口径：用 HF datasets-server rows API 按 `length=100` 分页拉 test split 前 5000 行
  （`sentence1` / `sentence2` / `score ∈ {0,1,2}`），改造成检索任务：
  corpus = 所有不重复 `sentence2`（作为 `title`，`text` 为空）；queries = 所有不重复 `sentence1`；
  qrels = `score >= 1` 的配对且 `rel = score`（同一配对重复出现取最大 score）。
  id = `sha256(文本)[:16]`，同文本必同 id。
- 限制：**原本是 query-title 配对数据集，被构造成检索任务；每条 query 通常只有一个正例，
  绝对指标不是标准 IR 基准**，只适合同口径横向比较。实际 5000 行里有 6 个重复配对（已合并、
  stderr 可见）；4924 个查询中 3749 个有正例，其余会被指标按「无正例查询」跳过。
  `score=0` 的配对不进 qrels（只让对应 `sentence2` 进 corpus）。16 位十六进制 id 前缀
  碰撞概率极低但不为零；命中会直接报 duplicate id 而不是静默合并。

### nq-sub（声明式子集，**不是全量 NQ**）

- 口径：`qrels/test.tsv` 里所有 `rel>0` 的文档全保留；其余文档按
  `sha256(f"{doc_id}:{seed}")` 升序补足 `pool_size=50000`（`seed=20261107`）；查询只保留
  「至少一个相关文档在池内」的，qrels 只留池内文档。实际：全量 2,681,468 篇 → 池内
  4201 篇正例 + 45799 篇抽样；3452 个查询全部保留（正例全保，本次没有正例文档缺失）。
- 限制：**池外文档完全没有参与，绝对 recall / MAP / nDCG 不能与全量 NQ 的结果比较**，
  只能在同一份子集（相同 seed 与版本）上横向比。NQ test qrels 只标正例，未标注文档按
  「不相关」参与指标计算（与 BEIR 口径一致）。`manifest.notes` 写死了 pool_size、seed 与
  这条不可比声明。
- 镜像说明：UKP 镜像里的 NQ 语料实际是未压缩的 `corpus.jsonl`（1.5 GB），不是常说的
  `corpus.jsonl.gz`；`build_nq_pool` 两种后缀都支持，`registry.py` 写的是实际成员名。

## 输出与确定性

- `corpus.jsonl` 按 doc id 升序、`queries.jsonl` 按 query id 升序、`qrels.tsv` 按
  `(qid, docid)` 升序；空 `title` 不写该字段（与 Rust `Doc` 的序列化一致）。
- `manifest.json` 的 `counts` / `files`（三个内容文件的 sha256 + 字节数）由
  `canonical.write_dataset()` 按实际落盘内容重算，写完立刻按 Rust 的 `verify_strict`
  口径自检（摘要、排序、qrels 不悬空、计数一致），任何不一致当场失败。
- 相同输入 + 相同参数重建，四个文件**字节完全相同**；`manifest` 里没有时间戳。
  `cargo ... dataset verify` 打印的指纹由 Rust 侧重算，应与构建日志里的 `manifest 指纹` 一致。
- `data/` 不可手改：改一个字节 verify 就会报摘要不一致。修数据 = 改 `registry.py` /
  转换逻辑后重建。

## 目录

| 文件 | 作用 |
|---|---|
| `registry.py` | 数据集规格：来源 URL、版本、zip sha256、子集口径（pool_size / seed）、notes |
| `build.py` | 下载（带缓存与重试）/ 解压 / 裁剪 / 转换 + CLI |
| `canonical.py` | 规范格式写出、manifest、自检、指纹（与 canonical.rs 对齐） |
| `tests/` | 纯离线 pytest（fixtures 覆盖 qrels 解析、子集确定性、计数与 sha256、字节确定性） |

`registry.py` 里的 `archive_sha256` 是口径的一部分：上游数据变了会**报错而不是静默换题**，
确认变动后更新固定值、重建并同步本文与 `notes`。`qbqtc` 的 rows API 没有固定哈希，靠
页缓存 + 版本标识（`hf-cmteb-qbqtc-test-5000`）复现。
