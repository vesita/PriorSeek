# 适配器清单

> 每个适配器的「是谁、从哪来、带什么参数」都写进运行记录（`run.json` 的 `adapters[].info`）。
> 这里只汇总来源与可信度证据；契约见 [`docs/STANDARD.md`](STANDARD.md) §4。

## 注册表

`cargo run -q -p sl-cli -- adapters` 输出（顺序即展示顺序）：

```text
bm25  faq-char  faq-bigram  faq-word  faq-hybrid  faq-bm25  rs
```

| 名字 | 族 | 实现位置 | 来源 |
|---|---|---|---|
| `bm25` | `bm25` | `crates/sl-adapters/src/{bm25,tokenize}.rs` | 本仓自实现（k1=1.5，b=0.75；字母数字词项 + 含 CJK 块 jieba-rs 分词；倒排检索） |
| `faq-char` | `faq-scheme` | `crates/sl-adapters/src/faq.rs` | `~/coding/hub/faq-search` 五方案的 Rust 移植（字符覆盖率；倒排检索） |
| `faq-bigram` | `faq-scheme` | 同上 | 相邻双字覆盖率 + 单字兜底 |
| `faq-word` | `faq-scheme` | 同上 | jieba 词覆盖率 |
| `faq-hybrid` | `faq-scheme` | 同上 | 词 2x + 双字 1x + 单字 0.3x |
| `faq-bm25` | `faq-scheme` | 同上 | 双字 BM25（IDF + 长度归一；倒排检索） |
| `rs` | `relation-search` | `crates/sl-adapters/src/rs.rs` | `https://github.com/vesita/relation-search`（本机检出 `~/coding/my/relation-search`；Rust BM25 + 代码感知分词 + 内嵌 ast-grep），**子进程**接入 |

## 移植保真度（faq-*）

`crates/sl-adapters/tests/faq_parity.rs` 用真实的 faq-search Python 包生成 fixture
（`tests/fixtures/faq_parity.json`，生成脚本 `gen_faq_parity.py` 头部有重跑命令），断言：

- 5 个方案 × 9 条查询的 **doc id 排序序列与 Python 完全一致**（分数按 1e-9 相对误差）；
- jieba-rs 切词与 Python jieba 切词一致；
- 三条兜底路径（单字查询→char、整库无 bigram 命中→整库 char、word 分词为空→char）确实被覆盖。

上游 `~/coding/hub/faq-search` 在本机**没有 git 远端**，所以移植结果与 `tests/fixtures/faq_parity.json`
随本仓提交：换机器不装那个 Python 包也能跑 parity 测试（重跑 fixture 生成脚本时才需要它）。

**与原包的差异**（写在 `faq.rs` 模块文档里）：查询侧特征只构造一次（原包每篇文档都重建查询特征），
语义与指标逐位不变，只是更快。faq 方案最初为中文 FAQ 设计，在英文语料上退化是预期行为，
不该读成「算法更差」——见 [`docs/RESULTS.md`](RESULTS.md) 的说明。


## 0.2.0：倒排实现（性能，语义不变）

0.1.0 的 `bm25` / `faq-*` 都是「每条查询扫一遍全部文档」。在 50,000 文档的 `nq-sub` 上，
`faq-bm25` 单查询 p50 是 **274ms**，跑满 3,452 条查询要小时级，这个实验台就没法扩到 BEIR 全集。
0.2.0 把检索改成倒排：

- `bm25`：词项字典 + 连续倒排表（`doc_idx, tf`），查询只累加命中文档；文档长度归一 / IDF / 同分排序 / `score > 0` 过滤全部不变。
- `faq-*`：单字 / 双字（带 tf）/ 分词三张倒排表，覆盖率用整数命中数；hybrid 的 `2.0*w + 1.0*b + 0.3*c` 左结合顺序不变；
  三处兜底（单字查询→char、word 分词为空→char、bigram/bm25 整库零命中→整库 char）逐条保留。
- **累加顺序被刻意保持**（外层按查询特征升序、内层文档），因此浮点分数逐位一致，不只是「接近」。

**它确实没改变任何指标**（这是「实现版本」与「算法版本」的分界）：

- `tests/faq_parity.rs` 的 fixture 与断言**一字节未改**仍全过；
- 新增 `tests/inverted_parity.rs`：文件内保留 0.1.0 的朴素全扫实现，对 5 方案 × 23 条查询 × 5 种 k
  断言 doc id 序列与分数 **IEEE-754 位模式**完全一致；
- scifact 基线：7 个适配器 × 300 计分查询 = **2,100 条 per-query 记录逐字段零差异**；
- 四个数据集（scifact / arguana / qbqtc / nq-sub）优化前后的 nDCG / MRR / recall / MAP 全部相同。

**代价变化**（同机、同一批查询、相邻时刻的配对测量）：

| 适配器 | scifact p50 前→后 | nq-sub（50k）p50 前→后 | 索引 ms 前→后 |
|---|---:|---:|---:|
| `bm25` | 4.93 → **0.21** ms | 35.3 → **2.8** ms | 279 → 176 |
| `faq-char` | 6.75 → **0.55** ms | — | 2,392 → 841 |
| `faq-bigram` | 30.98 → **0.71** ms | — | 2,251 → 782 |
| `faq-word` | 0.15 → **0.01** ms | — | 2,149 → 784 |
| `faq-hybrid` | 33.72 → **0.84** ms | — | 2,162 → 879 |
| `faq-bm25` | 49.34 → **0.70** ms | 274.3 → **4.0** ms | 2,088 → 2,733（50k 文档） |

（`rs` 未改，p50 前后不变，作为「机器没有变快」的对照行。）

> 版本规则：`AdapterInfo.version` 记的是**实现版本**。只改性能不改打分时它也升位（0.1.0 → 0.2.0），
> 因为「同分但不同实现」的延迟不能混进同一张成本表；对拍证据（上面三条）保证旧结果在效果列仍然可比。

## rs（relation-search）的接入方式

- 二进制路径：环境变量 `RELATION_SEARCH_BIN`，默认 `~/coding/my/relation-search/target/release/relation-search`；
  缺失时**构造即报错**，不会跑出一轮空结果。
- 索引：把每篇 Doc 写成临时目录下的 `.txt`，文件名为「清洗后的 doc id + sha256(id) 前 8 位」防碰撞，
  再调 `relation-search index --root <tmp> --index-dir <tmp>/.idx --rebuild`。
- 检索：`relation-search search ... --limit <k×8> --no-kind-weight --summary`；rs 返回**块级**候选，
  适配器把同一文件（= 同一文档）的块折叠为一个文档并取最高分，按分数降序、同分 doc id 升序取前 k。
- 记账：`AdapterInfo.version` = 二进制 sha256 前 12 位 + relation-search 仓库 HEAD；`config` 里记录
  `block_factor` / `min_block_limit` / 使用的 flags。子进程开销是它的真实系统属性，不做补偿。
- 集成测试：`crates/sl-adapters/tests/rs_integration.rs`（二进制缺失时打印跳过）。

## 不在本表里的候选（phase 3）

dense 检索、RRF 融合、cross-encoder 重排（`~/coding/my/docseek` 那套 hybrid + rerank）与
端到端 RAG 评测，需要 embedding 模型 / reader 模型，按 [`docs/STANDARD.md`](STANDARD.md) §8 排期。
