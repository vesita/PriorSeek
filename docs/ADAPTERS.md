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
| `bm25` | `bm25` | `crates/sl-adapters/src/{bm25,tokenize}.rs` | 本仓自实现（k1=1.5，b=0.75；字母数字词项 + 含 CJK 块 jieba-rs 分词） |
| `faq-char` | `faq-scheme` | `crates/sl-adapters/src/faq.rs` | `~/coding/hub/faq-search` 五方案的 Rust 移植（字符覆盖率） |
| `faq-bigram` | `faq-scheme` | 同上 | 相邻双字覆盖率 + 单字兜底 |
| `faq-word` | `faq-scheme` | 同上 | jieba 词覆盖率 |
| `faq-hybrid` | `faq-scheme` | 同上 | 词 2x + 双字 1x + 单字 0.3x |
| `faq-bm25` | `faq-scheme` | 同上 | 双字 BM25（IDF + 长度归一） |
| `rs` | `relation-search` | `crates/sl-adapters/src/rs.rs` | `~/coding/my/relation-search`（Rust BM25 + 代码感知分词 + 内嵌 ast-grep），**子进程**接入 |

## 移植保真度（faq-*）

`crates/sl-adapters/tests/faq_parity.rs` 用真实的 faq-search Python 包生成 fixture
（`tests/fixtures/faq_parity.json`，生成脚本 `gen_faq_parity.py` 头部有重跑命令），断言：

- 5 个方案 × 9 条查询的 **doc id 排序序列与 Python 完全一致**（分数按 1e-9 相对误差）；
- jieba-rs 切词与 Python jieba 切词一致；
- 三条兜底路径（单字查询→char、整库无 bigram 命中→整库 char、word 分词为空→char）确实被覆盖。

**与原包的差异**（写在 `faq.rs` 模块文档里）：查询侧特征只构造一次（原包每篇文档都重建查询特征），
语义与指标逐位不变，只是更快。faq 方案最初为中文 FAQ 设计，在英文语料上退化是预期行为，
不该读成「算法更差」——见 [`docs/RESULTS.md`](RESULTS.md) 的说明。

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
