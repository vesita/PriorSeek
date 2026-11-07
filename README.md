# search-lab

搜索算法的**可复现对比实验台**：固定数据集、固定指标口径，让不同检索算法在同一道题上跑，
产出可横比的效果与成本表。

- 对比口径（唯一事实源）：[`docs/STANDARD.md`](docs/STANDARD.md)
- 数据集口径与构建：[`docs/DATASETS.md`](docs/DATASETS.md)
- 适配器清单与来源：[`docs/ADAPTERS.md`](docs/ADAPTERS.md)

## 它解决什么问题

「A 算法比 B 算法好」这句话只有在**同一语料、同一查询、同一相关性判定、同一 k、同一指标口径**下才成立。
本仓把这些前提做成结构：

- 数据集先转成规范三元组（corpus / queries / qrels）+ 带 sha256 的 manifest，`sl run` 拒绝跑「被改过」的数据；
- 适配器只实现 `index()` / `search()`，**拿不到 qrels**，结构上无法用答案调参；
- 指标口径写死在代码里并有手算 golden 测试，报告只允许在同一数据集指纹内排名。

## 快速开始

```bash
# 1) 构建规范数据集（Python 侧，首次会下载公开数据）
cd tools/datasets
uv sync
uv run python build.py scifact
uv run python build.py arguana
uv run python build.py qbqtc
uv run python build.py nq-sub        # 约 498MB 下载，构建出 50k 文档池

# 2) 校验（不一致会直接失败）
cd ../..
cargo run -q -p sl-cli -- dataset list
cargo run -q -p sl-cli -- dataset verify scifact

# 3) 跑一轮对比
cargo run -q -p sl-cli -- run --dataset scifact --adapters bm25,faq-hybrid,rs --k 100

# 4) 结果在 results/<run-id>/：run.json + per_query/*.jsonl + report.md
```

## 目录

```text
crates/
  sl-core/      数据契约、规范格式 IO、指标口径、适配器 trait、运行记录、报告渲染
  sl-adapters/  BM25 / FAQ 五方案 / relation-search 的适配器实现与注册表
  sl-cli/       命令行：dataset list|verify|show、adapters、run、report
tools/datasets/ Python 数据集构建器（下载、解压、裁剪、写 manifest）
docs/           对比标准、数据集口径、适配器来源
data/           规范数据集（不进 git）
results/        运行记录（不进 git）
```

## 当前状态（phase 1）

| 数据集 | 族 | 规模 | 状态 |
|---|---|---|---|
| scifact | ranking | 5,183 文档 / 300 查询 | 待构建校验 |
| arguana | ranking | 8,674 文档 / 1,406 查询 | 待构建校验 |
| nq-sub | ranking | 50,000 文档池（seed 固定） | 待构建校验 |
| qbqtc | ranking | 5,000 title 语料（中文配对转检索） | 待构建校验 |

适配器：`bm25`、`faq-char`/`faq-bigram`/`faq-word`/`faq-hybrid`/`faq-bm25`（移植自
`~/coding/hub/faq-search`）、`rs`（`~/coding/my/relation-search` 的 Rust BM25+ast-grep 代码检索）。

phase 2/3 的范围与「为什么 LongBench / RULER / Natural Questions 不能混在一张表里」见
[`docs/STANDARD.md`](docs/STANDARD.md) §2。
