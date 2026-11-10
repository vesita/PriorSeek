# 数据集登记

> 换机器 / 隔一段时间重建数据时**只看这张表**。每个数据集的上游地址、许可、口径、构建方式都写在
> 这里；构建器的实现与逐条限制见 [`tools/datasets/README.md`](../tools/datasets/README.md)，
> 规范格式与 manifest 语义见 [`docs/STANDARD.md`](STANDARD.md) §3。
>
> `data/` 不进 git：新机器 clone 下来是空的，按下面「重建」一节跑一遍即可，
> 重建后 manifest 指纹必须与 §4 记录的一致（否则说明上游数据或构建器变了）。

## 1. phase 1 已接入的四个数据集

| 名称 | 上游地址 | 版本标识（manifest.version） | 许可 | 口径一句话 |
|---|---|---|---|---|
| `scifact` | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip` | `beir-scifact` | CC BY-SA 4.0（按 HF `BeIR/scifact` 卡片） | BEIR 官方测试集；corpus/queries 全量，qrels 只取 `qrels/test.tsv` |
| `arguana` | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip` | `beir-arguana` | CC BY-SA 4.0（按 HF `BeIR/arguana` 卡片） | 同上；上游 qrels 有 5 行指向不存在文档，构建时丢弃 |
| `nq-sub` | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip` | `beir-nq-sub-pool50000-seed20261107` | CC BY-SA 4.0（按 HF `BeIR/nq` 卡片） | **声明式子集**：50,000 文档池（正例全保留 + `sha256(doc_id:seed)` 抽样补足），seed=20261107 |
| `qbqtc` | HF rows API：`https://datasets-server.huggingface.co/rows?dataset=C-MTEB%2FQBQTC&config=default&split=test`；数据集页 `https://huggingface.co/datasets/C-MTEB/QBQTC`；原始仓 `https://github.com/cluebenchmark/QBQTC` | `hf-cmteb-qbqtc-test-5000` | **上游未声明许可**（GitHub 仓无 LICENSE 文件、HF 卡片未标 license）⇒ 仅本地研究用，不得再分发 | 原本是 query-title 配对集；这里构造为检索任务：corpus=全部不重复 title，qrels=score≥1 的配对 |

### 下载完整性（zip 的 SHA-256，registry.py 里同值）

| 压缩包 | SHA-256 |
|---|---|
| `scifact.zip` | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` |
| `arguana.zip` | `cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b` |
| `nq.zip` | `2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752` |

QBQTC 通过 HF rows API 按页拉取（`length=100`，共 50 页），页级缓存落在 `tools/datasets/.cache/`；
接口没有整包哈希，可复现性由「固定 split + 固定前 5000 行 + 段 id = `sha256(文本)[:16]`」保证。

### 为什么 NQ 用子集

全量 BEIR nq 是 2,681,468 篇文档（镜像里是 1.5 GB 的 `corpus.jsonl`，不是常说的 `.gz`）。
在单机实验台上跑全量会让每轮评测的时间和内存成本失控，所以 phase 1 用固定 seed 的 50k 池：
**同一份池内可比，绝对指标不能与全量 NQ / DPR 论文数字比较**。要跑全量只是改 `pool_size` 后重建，
但会换 manifest 指纹 = 换一道题，旧结果不能混用。

### 相关性语义备注

- `scifact` / `arguana` / `nq-sub`：qrels 为二值（rel=1）；NQ test qrels 只标正例，未标注文档按不相关
  （与 BEIR 口径一致）。
- `qbqtc`：rel ∈ {1,2}（来自上游 score），nDCG 因此是分级相关的；每条 query 通常只有一个正例。

## 2. phase 2 已声明、尚未接入（先登记地址，避免以后找不到）

| 数据集 | 地址 | 许可 | 进哪个任务族 |
|---|---|---|---|
| BEIR 全集（18 集） | `https://github.com/UKPLab/beir`；镜像 `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip` | 逐集不同（见各集 HF 卡片） | ranking |
| C-MTEB 中文检索（T2Retrieval / DuRetrieval / MMarcoRetrieval 等） | `https://huggingface.co/C-MTEB` | 逐集不同 | ranking |
| NQ 全量 | 同 §1 的 `nq.zip`（`pool_size` 不裁剪即可） | CC BY-SA 4.0 | ranking |
| LongBench | `https://github.com/THUDM/LongBench` | MIT | **end_to_end_qa**（需要 reader + 判分）；只取其检索类子任务进 long_context_retrieval |
| RULER | `https://github.com/NVIDIA/RULER` | Apache-2.0 | **long_context_retrieval** |

> LongBench / RULER 与 ranking 族的指标**不可互换**，原因见 [`docs/STANDARD.md`](STANDARD.md) §2。

## 2.5 训练划分（train split，可选）

`scifact` 额外提供训练划分，供内核训练工具使用；**只有 scifact 有**（其余三个数据集上游只有 test qrels）。

| 文件 | 内容 | 谁读 |
|---|---|---|
| `train_queries.jsonl` | 809 条训练查询（与 `queries.jsonl` 同 schema） | 训练工具（`tools/kernel/train_seq_weights.py`） |
| `train_qrels.tsv` | 919 条正例判定 | 训练工具；**适配器读不到**（`sl_core::Adapter` 接口里没有它） |

构建（在 `tools/datasets/` 下，缓存已就绪时秒级）：

```bash
uv run python build_train_split.py scifact
```

两条契约要点：

1. **不进 manifest 指纹**：训练数据换一版、测试题没变，两次运行在 test 上仍然可比。摘要记在
   `manifest.json` 的 `train_files`（独立字段），`sl_core::load_train_split` 加载时逐一比对，
   不符直接报错——「训练用了哪份数据」可审计，但不会让旧结果失效。
2. **训练/测试隔离是结构性的**：训练器代码只打开 train 文件，`tools/kernel/tests/test_train_seq_weights.py`
   里有一条测试用路径探针断言 `qrels.tsv` 从未被打开。

## 3. 重建（新机器 / 清空缓存后）

```bash
git clone <forgejo>/vesita/search-lab.git && cd search-lab
# Rust 侧
cargo test --workspace
cargo run -q -p sl-cli -- adapters          # 应列出 7 个适配器
# 数据侧（Python + uv）
cd tools/datasets
uv sync
uv run python build.py scifact
uv run python build.py arguana
uv run python build.py qbqtc
uv run python build.py nq-sub               # 首次要下 498MB 的 nq.zip
cd ../..
for d in scifact arguana qbqtc nq-sub; do cargo run -q -p sl-cli -- dataset verify $d; done
# 可选：scifact 的训练划分（内核训练工具用）
cd tools/datasets && uv run python build_train_split.py scifact && cd ../..
```

`data/` 与 `tools/datasets/.cache/` 都不进 git。缓存可整目录删除，重建时重新下载。

## 4. 可复现性基准（重建后必须一致）

下表是 2026-11-07 在本机构建时 `sl dataset verify` 打出的指纹。**换机器重建后应逐字一致**；
不一致先查上游文件哈希（§1 表）与构建器版本，不要直接拿新数据跑结论。

| 数据集 | 文档 | 查询 | 计分查询 | manifest 指纹 |
|---|---:|---:|---:|---|
| `scifact` | 5183 | 1109 | 300 | `7bf0074d1628773ed1371699c1ae77c0dec1a55b6bd452b208ff6c3c7c33528c` |
| `arguana` | 8674 | 1406 | 1401 | `d2834b800729db4a38bacdb31b2f0521fe144a99b8bfe09564f037aae42872b9` |
| `qbqtc` | 4889 | 4924 | 3749 | `40c321d76e9cfec3e9f56cefae4ba55643085cc13fb165bcd7ea5777ab1234a5` |
| `nq-sub` | 50000 | 3452 | 3452 | `13432a7bd7c0a2f5b3dde258e45a332b2bf2caee9f5bc119fced37ab967b850d` |

指纹算法：见 `crates/sl-core/src/types.rs::Manifest::fingerprint`（数据集名 + 版本 + 三个文件的 sha256 + 规模；
`notes` / `built_by` 不参与）。

## 5. 构建耗时与磁盘（本机实测，2026-11-07）

| 数据集 | 首次下载 | 转换耗时 | 规范数据占用 |
|---|---|---|---|
| `scifact` | 2.8 MB | 0.2s | 8.1 MB |
| `arguana` | 3.8 MB | 0.3s | 11.3 MB |
| `qbqtc` | 5000 行 API（约 120s，受 HF 限流影响） | 0.1s（缓存就绪） | 1.0 MB |
| `nq-sub` | 498 MB | 20.0s（缓存就绪 14.8s） | 27.6 MB |

额外缓存：`nq.zip` 498 MB + 解压出的 NQ 语料 1.5 GB（都在 `tools/datasets/.cache/`，可删）。
