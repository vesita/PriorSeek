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

## 改代码时的约束

- 适配器（`crates/sl-adapters/`）只实现 `sl_core::Adapter`：`index(docs)` + `search(query, k)`。
  **不要**给它加 qrels、金标、数据集名字之类的参数 —— 结构上杜绝用答案调参。
- runner 的排序归一（分数降序、同分 doc id 升序、去重、未知 id 记账）在
  `crates/sl-core/src/run.rs`，改动必须带单测。
- 指标改动必须同时更新 `metrics.rs` 的手算 golden 测试与 `docs/STANDARD.md` §5。
- 外部实现（如 relation-search）用子进程接入时，`AdapterInfo` 必须记录二进制 sha256 + 来源 commit。

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
