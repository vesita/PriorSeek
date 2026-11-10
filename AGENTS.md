# AGENTS.md —— search-lab 工作纪律

## 这个仓库是什么

一个检索算法的对比实验台：Rust 侧（`crates/`）定义数据契约、指标口径、适配器接口并跑对比；
Python 侧（`tools/datasets/`）把公开数据转成规范三元组。

## 三条不可违反的规则

1. **口径只有一个事实源**：`docs/STANDARD.md`。改指标、改可比性条件、加任务族，先改它，再改代码和测试。
2. **`data/` 不可手改**：规范数据由 `tools/datasets/build.py` 生成，带 sha256 manifest。
   手改一个字节，`sl dataset verify` 与 `sl run` 都会拒绝。修数据 = 改构建器 + 重建。
3. **不跨任务族排名**：ranking / long_context_retrieval / end_to_end_qa 的分数不进同一张表。
   报告与结论都按这个前提写。

## 术语（不许混用）

- **内核网络**：产出检索参数的模型，有**参数量**（可训练权重个数）。
- **先验因子**：内核网络的**输出**，外壳检索时消费；有**因子规格**（形状/dtype/个数）。
- **外壳**：用先验因子执行检索的算法，只有算法超参，**不是**模型参数。
- 规则：说"内核要小"时指的是**参数量**；说"因子变多"时指的是**输出**。两者不得混写，
  工件里也必须分列（`prior_factor` / `network`）。详见 [`docs/KERNEL.md`](docs/KERNEL.md) §0。

## 改代码时的约束

- 适配器（`crates/sl-adapters/`）只实现 `sl_core::Adapter`：`index(docs)` + `search(query, k)`。
  **不要**给它加 qrels、金标、数据集名字之类的参数 —— 结构上杜绝用答案调参。
- runner 的排序归一（分数降序、同分 doc id 升序、去重、未知 id 记账）在
  `crates/sl-core/src/run.rs`，改动必须带单测。
- 指标改动必须同时更新 `metrics.rs` 的手算 golden 测试与 `docs/STANDARD.md` §5。
- 外部实现（如 relation-search）用子进程接入时，`AdapterInfo` 必须记录二进制 sha256 + 来源 commit。

## 参考项目纪律（借自 nanoSeek / flowme，已改写到我们的场景）

详见 [`docs/REFERENCE_NOTES.md`](docs/REFERENCE_NOTES.md)。可执行的四条：

1. **预注册**：开新对照实验前先写 `docs/PREREG-*.md`（问题 / 两臂 / 唯一变量 / 判据 / 预测值），
   test 集只在最后看一次；超参用 train 内部交叉验证选。
2. **不下空结论**：没测出差异只写「该口径下未测出差异」，并报 `Δ_min = 2×SE`；单 seed 只报「超过噪声带」。
3. **消融 = retrain-without**（去掉特征重训），不是推理时把权重设 0。
4. **地板行必报**（随机 / `bm25` / `faq-bm25`），训练算力与 CPU 运行延迟分列；学习版参数预算先定 **≤10k**
   （第一步 3–5 个标量）——nanoSeek 的 75M 规模不适用于我们 809 条训练查询的数据量。

## 常用命令

```bash
cargo test --workspace                    # 单测（含指标 golden、排序归一、报告渲染）
cargo run -q -p sl-cli -- adapters        # 列出适配器
cargo run -q -p sl-cli -- dataset list
cargo run -q -p sl-cli -- dataset verify scifact
cargo run -q -p sl-cli -- run --dataset scifact --adapters bm25,faq-hybrid,rs --k 100
cargo run -q -p sl-cli -- report results/<run-id>
cd tools/datasets && uv run pytest -q     # 数据集构建器的离线测试
```

## 写域

| 路径 | 内容 |
|---|---|
| `crates/sl-core/`、`crates/sl-cli/`、`docs/`、根配置 | 契约、指标、CLI、文档 |
| `crates/sl-adapters/` | 各算法适配器与注册表 |
| `tools/datasets/` | 数据下载/转换/裁剪与 manifest |

`data/`、`results/`、`target/`、`tools/datasets/.cache/` 不进 git。
改完记得跑 `cargo test --workspace`；不要对全 workspace 跑 `cargo fmt` 覆盖别人的改动。
