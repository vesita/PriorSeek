# search-lab

搜索算法的**可复现对比实验台**：固定数据集、固定指标口径，让不同检索算法在同一道题上跑，
产出可横比的效果与成本表。

- 对比口径（唯一事实源）：[`docs/STANDARD.md`](docs/STANDARD.md)
- 内核设计（认知 token / token 关联 / 搜索权重，GPU 训练 + CPU 运行）：[`docs/KERNEL.md`](docs/KERNEL.md)
- 参考项目笔记（借 nanoSeek / flowme 的实验纪律）：[`docs/REFERENCE_NOTES.md`](docs/REFERENCE_NOTES.md)
- 数据集口径与构建：[`docs/DATASETS.md`](docs/DATASETS.md)
- 适配器清单与来源：[`docs/ADAPTERS.md`](docs/ADAPTERS.md)

## 方向：外壳明确，内核深度学习

外壳（规范数据、指标口径、runner、报告、公平性）固定不变；内核（token 化、关联结构、搜索权重）
可以换成深度学习实现——但只能体现在**离线工件**里，运行时必须非 GPU 可跑。
内核设计、工件契约与里程碑见 [`docs/KERNEL.md`](docs/KERNEL.md)。

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
uv run python build_train_split.py scifact   # 可选：训练划分（内核训练工具用）

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
| scifact | ranking | 5,183 文档 / 1,109 查询（300 计分） | 已评测 |
| arguana | ranking | 8,674 文档 / 1,406 查询（1,401 计分） | 已评测 |
| qbqtc | ranking | 4,889 title / 4,924 查询（3,749 计分，中文） | 已评测（中文方案在这里胜出） |
| nq-sub | ranking | 50,000 文档池 / 3,452 查询（seed 固定） | 已评测 |

数据来源地址、许可、子集口径与重建后的指纹基准：见 [`docs/DATASETS.md`](docs/DATASETS.md)。
已有结果：见 [`docs/RESULTS.md`](docs/RESULTS.md)（四个数据集各一轮，单次运行）。
一句话读法：**英文上 `rs` 与 `bm25` 并列前二；中文上 hub 的 `faq-bm25` / `faq-hybrid` 反超 `rs`**——
没有一个实现全场景通吃，所以本仓不把不同数据集的分数平均成一个总分。

适配器：`bm25`、`faq-char`/`faq-bigram`/`faq-word`/`faq-hybrid`/`faq-bm25`（移植自
`~/coding/hub/faq-search`）、`rs`（`~/coding/my/relation-search` 的 Rust BM25+ast-grep 代码检索）、
`seq`（内核第一版：token 化 + token 时序相关度，消费 `tools/kernel/` 产出的 `seq-v0` 工件；
结果与消融见 [`docs/RESULTS.md`](docs/RESULTS.md) §8–§10）。

phase 2/3 的范围与「为什么 LongBench / RULER / Natural Questions 不能混在一张表里」见
[`docs/STANDARD.md`](docs/STANDARD.md) §2。
