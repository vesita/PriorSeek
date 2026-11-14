# 先验搜索 (PriorSeek)

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache-blue.svg)](LICENSE-MIT)
[![ModelScope Artifacts](https://img.shields.io/badge/ModelScope-Artifacts%20Hub-orange.svg)](https://www.modelscope.cn/models/Reasone/priorseek)

**先验搜索 (PriorSeek)** 是一个高性能、严谨可复现的检索算法实验台与内核实现。项目采用**“离线深度学习先验内核 + 纯 Rust 零超参执行外壳”**的物理分离架构：
- **离线内核 (Neural Kernel)**：使用深度网络（如 MiniLM 等）抽取语义表征、领域概念拓扑与微网络参数，编译为带 SHA256 校验的不可篡改二进制先验工件；
- **在线外壳 (Rust Shell)**：纯 Rust 开发，完全无 Python/GPU 依赖，通过 8-way SIMD 展开点积、有界最小堆快速截断与双路无超参 RRF 融合，在 CPU 上实现 **p50 < 6ms** 的极致低延迟检索。
- **权重工件解耦**：代码仓库保持极度轻量纯净，所有二进制权重工件统一在 [魔搭社区 (ModelScope)](https://www.modelscope.cn/models/Reasone/priorseek) 托管与分发。

---

## 核心基准对账 (SciFact 5,183 篇科学文献 / 300 条真实测试查询)

> 所有指标均在同一评测口径、同一测试集上严格对拍，结构上无法利用金标答案调参（唯一口径源：[`docs/STANDARD.md`](docs/STANDARD.md)）。

| 算法 / 适配器 | 内核架构与特征 | nDCG@10 | hit@10 | MRR@10 | hit@1 | CPU p50 查询延迟 |
|---|---|---:|---:|---:|---:|---:|
| **BM25 (基线)** | 传统稀疏词法倒排 | 0.6542 | 78.67% | 0.6328 | 54.33% | 1.67 ms |
| **Concept-Synonym** | 离线概念与医学同义硬对齐外壳 | 0.6788 | 80.33% | 0.6470 | 55.67% | 2.61 ms |
| **Dense** | 384 维连续语义向量 + 8-way 展开点积 | 0.6451 | 79.33% | 0.6047 | 50.33% | 3.01 ms |
| **Hybrid-RRF** | 双路无超参倒数秩融合 RRF(60) | 0.7012 | 85.00% | 0.6634 | 56.00% | 0.95 ms |
| **PriorSeek (Tri-Hybrid-RRF)** | **三路先验协同 (概念同义 + 稠密向量 + 显著性门控)** | **0.7057** | **84.67%** | **0.6706** | **57.33%** | **1.54 ms** |

---

## 核心设计哲学

1. **外壳零超参**：放弃脆弱的经验分数加权（如 0.3 * BM25 + 0.7 * Dense），改用稳健的纯秩次 RRF(60)，彻底避免不同量纲分数漂移引发的性能退化。
2. **纯 CPU 极致加速**：纯 Rust 8-way 指令级展开连续内积计算 + 预分配有界最小堆（Bounded Min-Heap），替代全局排序，将向量检索延迟从 12.7ms 暴降至 **3.01ms**（提速 4.2 倍）。
3. **确定性防篡改契约**：每个先验工件（`dense.bin`、`concepts.json` 等）均携带带 sha256 签名的 `manifest.json`，确保实验结果 100% 可复现。

---

## 快速上手

### 1. 编译纯 Rust 检索外壳

```bash
git clone https://github.com/vesita/PriorSeek.git
cd PriorSeek
cargo build --release
```

### 2. 获取预编译先验权重工件（魔搭平台）

所有官方构建的先验权重工件已托管在魔搭社区：

```bash
# 从魔搭拉取预编译先验工件至数据目录
git clone https://www.modelscope.cn/Reasone/priorseek.git /tmp/priorseek-artifacts
mkdir -p data/scifact/artifacts
cp -r /tmp/priorseek-artifacts/* data/scifact/artifacts/
```

### 3. 单查询极速探针调试

```bash
# 使用最优的双路 Hybrid-RRF 执行真实单条科学查询探针
cargo run --release --package sl-cli -- probe \
  --dataset scifact \
  --adapter hybrid-rrf \
  --query "Pyridostatin stabilizes the G - quadruplex in the telomeric region." \
  --k 10
```

### 4. 运行全量对比评测

```bash
cargo run --release --package sl-cli -- run \
  --dataset scifact \
  --adapters bm25,dense,concept-synonym,hybrid-rrf \
  --k 100
```

---

## 许可证 (License)

本项目采用 **MIT OR Apache-2.0** 双重开源许可协议。详见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。
