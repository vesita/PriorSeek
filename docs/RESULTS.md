# 结果记录

> 每次运行的完整产物在 `results/<run-id>/`（`run.json` + `per_query/*.jsonl` + `report.md`，不进 git）。
> 这里只留**可引用的结论摘要**。结论纪律见 [`docs/STANDARD.md`](STANDARD.md) §9：
> 单次运行只是观测，不能写成「谁更强」。

## phase 1 · scifact（已跑）

- 运行 id：`scifact-20261107-183659-phase1`
- 数据集指纹：`7bf0074d1628773ed1371699c1ae77c0dec1a55b6bd452b208ff6c3c7c33528c`
  （5183 文档 / 1109 查询 / 300 条计分查询）
- 命令：`./target/release/sl run --dataset scifact --adapters bm25,faq-char,faq-bigram,faq-word,faq-hybrid,faq-bm25,rs --k 100 --warmup 3`
- 证据强度：**单次运行**；延迟列只作量级参考（跑的时候机器上还有别的构建/评测）。

| # | 适配器 | 族 | R@1 | R@10 | R@100 | MRR@10 | nDCG@10 | MAP@100 | 索引 ms | p50 ms | p95 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `rs` | relation-search | 0.5375 | 0.8026 | 0.9026 | 0.6449 | 0.6785 | 0.6393 | 135.9 | 9.25 | 12.05 |
| 2 | `bm25` | bm25 | 0.5248 | 0.7849 | 0.8892 | 0.6328 | 0.6647 | 0.6260 | 206.3 | 4.93 | 7.84 |
| 3 | `faq-bm25` | faq-scheme | 0.2644 | 0.4747 | 0.6748 | 0.3479 | 0.3725 | 0.3440 | 2068.2 | 49.34 | 70.38 |
| 4 | `faq-hybrid` | faq-scheme | 0.0967 | 0.2261 | 0.3967 | 0.1430 | 0.1606 | 0.1448 | 2063.7 | 33.72 | 44.03 |
| 5 | `faq-bigram` | faq-scheme | 0.0933 | 0.2294 | 0.4011 | 0.1409 | 0.1597 | 0.1425 | 2070.5 | 30.78 | 40.40 |
| 6 | `faq-word` | faq-scheme | 0.0000 | 0.0067 | 0.0067 | 0.0022 | 0.0033 | 0.0022 | 2065.7 | 0.15 | 0.18 |
| 7 | `faq-char` | faq-scheme | 0.0000 | 0.0067 | 0.0433 | 0.0010 | 0.0022 | 0.0021 | 2144.4 | 6.75 | 7.83 |

怎么读这张表（避免误读）：

- **`rs` 与 `bm25` 在这一集上几乎并列**（nDCG 0.6785 vs 0.6647），且都是「正常的英文检索」水平。
  `rs` 略高的可解释部分：tantivy BM25 + 完整标识符相邻短语加分；两者分词与打分参数不同，不是同一算法的实现差异。
- **`faq-*` 五个方案在英文 SciFact 上大幅落后是预期**：它们是为中文 FAQ 设计的字 / 双字 / 词覆盖率打分，
  在英文语料上退化成「按字符重叠」的粗糙信号。这**不构成**「这些算法更差」的结论；
  要看它们在中语境下的表现，应看 `qbqtc`（phase 1 待跑）。
- `faq-bm25` 明显好于 `faq-bigram`：双字 BM25 的 IDF 与长度归一在英文上也还有区分度；
  `faq-word` 几乎归零是设计使然（英文整词在「字」层面的词表里命中不了）。
- 成本差异同样重要：`rs` 索引 136ms / 单查询 ~9ms（含子进程启动），`bm25` 索引 206ms / 单查询 ~5ms，
  `faq-*` 索引 ~2s 且单查询 30–50ms —— 说明这几族在「无倒排、逐文档打分」上的固有成本。

## phase 1 · 其余数据集（未跑）

`arguana`、`qbqtc`、`nq-sub` 的数据已构建并通过校验（指纹见 [`docs/DATASETS.md`](DATASETS.md) §4），
但本轮评测尚未完成（`arguana` 跑到一半被中止）。下一台机器上按下面命令补齐，每个数据集一行：

```bash
cargo build --release -p sl-cli
for d in arguana qbqtc nq-sub; do
  ./target/release/sl run --dataset $d --adapters bm25,faq-char,faq-bigram,faq-word,faq-hybrid,faq-bm25,rs --k 100 --warmup 3 --label phase1
done
```

补齐后把每个数据集的表追加到本文件，并保留「同一数据集指纹 + 同一 k」前提。
