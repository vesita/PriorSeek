# search-lab 对比标准（v0.1）

> 本文件是 search-lab 里**唯一**的对比口径事实源。代码只是它的强制点；任何结论如果和这里不一致，
> 以这里为准，并当场改代码而不是改结论。

## 0. 这个仓库在研究什么

**一个可复现的检索算法对比实验台**：固定语料、固定查询、固定相关性判定、固定指标，
让不同搜索算法在**同一道题**上跑一遍，产出可横比的效果与成本数字。

不做的事：
- 不训练 / 不微调模型（phase 1 只评测现成算法；需要训练的能力单独排期）；
- 不把「检索质量」和「问答质量」混成一个分（见 §2 的任务族划分）；
- 不用测试集调参（见 §6）。

## 1. 一条硬规则：可比性有三重前提

两个数字可以放在同一张表里比较，当且仅当：

1. **同一任务族**（§2）；
2. **同一数据集指纹**：`manifest.json` 的 `fingerprint` 完全相同（同一语料池、同一 qrels、同一子集口径）；
3. **同一召回深度 k 与同一套指标口径**（§5）。

任一条不满足，就只能作为「不同题目的参考」，不能排名。报告（`report.md`）与 CLI 都按这个前提写，
跨数据集不会合并成一张总榜。

## 2. 任务族（不同族不排名，只登记口径）

| 族 | 输入 | 检索算法输出 | 主指标 | 现状 |
|---|---|---|---|---|
| **ranking**（检索排序） | 固定语料 + 查询 + qrels | top-k 文档 id | Recall@k / MRR@10 / nDCG@10 / MAP@100 | phase 1（本仓已实现） |
| **long_context_retrieval**（长上下文定位） | 一段超长上下文 + 问题/针 | 证据位置或片段 | evidence recall@k、随上下文长度的留存曲线 | phase 2 |
| **end_to_end_qa**（端到端问答） | 语料 + 问题 | 答案文本 | 需要 reader 模型 + 判分器（F1 / 判分模型） | phase 3 |

用户点名的三个数据集分别落在这三族里，**不能互相替代**：

- **Natural Questions（NQ）**：本仓 phase 1 用它做 **ranking** 族，且是**子集** `nq-sub`
  （50k 文档池 + 固定 seed，口径写在它的 `manifest.json` 的 `notes` 里）。
  子集上的 recall 绝对值**不能**和全量 NQ / DPR 论文数字比；同一子集指纹内横比才成立。
  NQ 的 open 设定（检索 + 阅读器出答案）属于 **end_to_end_qa**，phase 3 才做。
- **LongBench**：主体是长上下文理解任务（问答、摘要、代码等），评的是「给模型一段长上下文后答得对不对」，
  属于 **end_to_end_qa**。它不产出「候选文档排序」，所以纯检索算法**无法**在 LongBench 上直接得分。
  phase 2 只取其检索类子任务进 **long_context_retrieval** 族，指标单独定义。
- **RULER**：合成式长上下文评测（在长文本里定位/聚合信息，看有效上下文长度），属于
  **long_context_retrieval**。它的指标是「针/证据被找到的比例」与长度曲线，和 ranking 族的
  Recall@k 语义不同（后者分母是人工标注的相关文档集合）。

> 结论：**LongBench / RULER 的数字和 ranking 族的 Recall@10 不是一回事**，本仓不会把它们并表。
> 要对比「同一个检索器在长上下文里行不行」，phase 2 会用同一套 `Adapter` 接口
> 在 long_context_retrieval 族上重跑去产生**同族**数字。

## 3. 数据契约

### 3.1 规范目录

每个数据集是 `data/<name>/` 下的四个文件：

```text
corpus.jsonl   每行 {"id","title","text"}（title 可空）
queries.jsonl  每行 {"id","text"}
qrels.tsv      首行 qid\tdocid\trel，其余每行三列（制表符），rel 为整数，>0 视为相关
manifest.json  来源 / 构建参数 / 规模 / 三个内容文件的 sha256
```

构建器在 Python 侧（`tools/datasets/`），**只允许**它写 `data/`。原始格式（BEIR zip、HF parquet、
业务库）不进入评测层：适配器只看到规范格式。

### 3.2 manifest 是「同一道题」的凭证

- `fingerprint` 由「数据集名 + 版本 + 三个文件的 sha256 + 规模」哈希而来，与 `notes`/`built_by` 无关
  ——换机器、改说明文字不影响指纹，动一个字节的内容就换指纹。
- 子集必须在 `notes` 里写死口径（pool 大小、seed、过滤规则），构建必须**确定性**：
  同输入 + 同参数 → 三个内容文件字节完全相同。
- `sl dataset verify <name>` 强制：摘要一致 + 没有 qrels 指向不存在的文档/查询 + 实际规模与 manifest 一致。
  `sl run` 默认走这套强校验；`--allow-dirty` 只能用于调试，其结果**不得进报告或提交**。

### 3.3 来源与许可

每个数据集的上游来源写在 `manifest.json` 的 `source` 里（URL / 仓库 / 论文入口）。
许可以上游为准：把规范数据分发给别人之前，先核对上游许可与引用要求。
本仓不分发原始数据，`data/` 不进 git。

## 4. 适配器契约

一个搜索算法实现 `sl_core::Adapter`：

```rust
fn index(&mut self, docs: &[Doc]) -> Result<()>;
fn search(&mut self, query: &str, k: usize) -> Result<Vec<(String, f64)>>;
fn info(&self) -> &AdapterInfo;
```

硬要求：

1. `search` 返回的是**语料文档 id**，不是内部块 id / 文件路径 / 行号。块级系统（如 rs）必须在适配器里
   折叠到文档级，折叠规则写进 `AdapterInfo.config`。
2. 适配器**拿不到 qrels**（trait 里没有这个参数）——结构上杜绝用答案调参。
3. 分数只用于排序，跨适配器不可比：不做归一、不报告「分数更高所以更好」。
4. 同分由 runner 用 doc id 升序兜底，保证同一实现重复跑结果稳定。
5. 返回了语料外的 id 或重复 id：runner 会去重/丢弃并**计数告警**（写进报告），不静默吞掉。
6. 外部实现（如 relation-search）用子进程接入时，`AdapterInfo` 必须记录二进制 sha256 与来源仓库 commit。

## 5. 指标口径

设一条查询的相关文档集合为 `R`（qrels 里 `rel > 0` 的文档），系统排序为 `L`（已去重、截断到 k）：

| 指标 | 定义 |
|---|---|
| Recall@k | `|R ∩ L[:k]| / |R|`（分母是全部相关文档，不受 k 截断） |
| MRR@10 | 第一个相关文档排名的倒数；排名 > 10 记 0 |
| nDCG@10 | 增益 `2^rel - 1`，折损 `log2(i+2)`；IDCG 取该查询全部相关文档按 rel 降序的前 k 个 |
| MAP@100 | `AP@k = (1/|R|) Σ_{i≤k, L[i] 相关} P@i`，再对查询取平均 |

平均口径与边界：

- 全部指标都是**宏平均**：每条查询等权，先算单查询再平均；
- 只对「qrels 里至少有一个正相关文档」的查询计分；其余查询进 `queries_skipped`，不拉平均；
- qrels 里有、但本轮没有结果的查询按**空排序**计 0（不静默丢出分母）；
- 排序内的重复 id 先按「保留首次出现」去重；不在 qrels 里的文档视为不相关；
- 默认召回深度 `k = 100`（报告里 k 小于某指标窗口时，窗口自动截断并在报告注明）。

成本口径（与效果分开记，不合成单一分数）：

- **索引耗时**：`index()` 一次调用的墙钟毫秒；
- **查询延迟**：进程内单查询墙钟毫秒的 p50 / p95，预热查询不计入（预热条数写进运行记录）；
- 外部系统用子进程时，进程启动/通信开销是它的真实系统属性，记录并在报告里注明，不做「公平性补偿」。

## 6. 公平性规则（硬）

1. 同一轮里所有适配器看**同一份** `Dataset`（同一份语料/查询/qrels）、同一个 `k`、同一查询顺序。
2. 适配器不接触 qrels（结构保证，见 §4.2）。
3. 参数必须预注册：所有影响排序的参数进 `AdapterInfo.config` 并写进运行记录；不允许「跑完再悄悄调参」。
4. 失败不静默：适配器报错即中止本轮并指出是哪个适配器，不允许把异常变成空结果。
5. 外部实现的版本必须可追溯：二进制 sha256 / 仓库 commit / 上游版本；实现变了就换 `version`，
   旧结果不覆盖、不覆盖写。
6. 数据文件被改动即视为新题：`sl run` 会因 manifest 摘要不一致而拒绝运行。

## 7. 运行记录与报告

一轮运行的产物（`results/<run-id>/`）：

```text
run.json                 数据集指纹、k、warmup、适配器元信息、指标、成本、环境
per_query/<adapter>.jsonl 每条查询的 first_rel_rank / ndcg / ap / recall / hits
report.md                人读的对比表（按 nDCG 排序，附可比范围与来源）
```

`report.md` 的表头固定写清：数据集指纹、规模、k、warmup、运行时间、harness commit、机器与 OS，
并声明「只有同一指纹 + 同一 k 的同一张表内可比」。

## 8. 阶段路线

| 阶段 | 内容 | 进入条件 |
|---|---|---|
| phase 1（当前） | ranking 族：scifact / arguana / nq-sub / qbqtc；适配器：`bm25` / `faq-`×5 / `rs` | —— |
| phase 2 | BEIR 全集、C-MTEB 中文检索（T2Retrieval / DuRetrieval）、long_context_retrieval 族（RULER / LongBench 检索子任务） | 新增任务族必须先在本文件登记口径与指标 |
| phase 3 | end_to_end_qa（NQ open / LongBench QA），dense + RRF + cross-encoder 重排适配器 | 引进 reader 模型与判分器，成本口径扩展 |

## 9. 结论纪律

- 单次运行只是**观测**，不是「算法好坏」的结论；写结论必须标注证据强度：
  **单次 / 多 seed 重复 / 多数据集**。
- 「没测出差异」不等于「没有差异」：样本量小、指标方差大时只能写「本轮未见差异」。
- 跨数据集只能说「相对位次是否一致」，不能把不同数据集的绝对分数平均成一个总分。

## 10. 强制点清单

| 规则 | 强制点 |
|---|---|
| 数据契约与「同一道题」 | `crates/sl-core/src/canonical.rs`（读写 + 摘要）、`sl dataset verify` |
| 指标口径 | `crates/sl-core/src/metrics.rs`（手算 golden 单测） |
| 排序稳定性、重复/未知 id | `crates/sl-core/src/run.rs::normalize_candidates` + 单测 |
| 适配器不碰 qrels | `sl_core::Adapter` trait 签名 |
| 同轮同数据集/同 k | `run_dataset` 一次性遍历全部适配器 |
| 报告只同指纹内排名 | `crates/sl-core/src/report.rs`（表头声明 + 单测） |
| 数据文件不被手改 | manifest sha256 + `sl run` 默认强校验 |
| 参数可追溯 | `AdapterInfo.config` 进 `run.json` |
