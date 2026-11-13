# PriorSeek

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache-blue.svg)](LICENSE-MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Reasone/PriorSeek/test.yml)](https://github.com/Reasone/PriorSeek/actions)

PriorSeek 是一个高性能、工业级的检索算法对比实验台。它通过**离线神经网络先验内核**与**纯 Rust 零超参执行外壳**的解耦架构，解决了传统深度检索模型在线推理耗时长、资源占用高的痛点。

## 核心架构：PriorSeek 模式

- **离线内核 (Offline Kernel)**: 利用先进预训练模型抽取高质量语义因子，并将其编译为轻量化、不可篡改的二进制工件。
- **高性能外壳 (High-Performance Shell)**: 纯 Rust 开发，深度优化 8-way SIMD 算子，支持在 CPU 上实现极低延迟检索。

## 核心指标 (SciFact 基准)

| 方案 | nDCG@10 | hit@10 | p50 延迟 (CPU) |
|---|---:|---:|---:|
| BM25 (基线) | 0.6542 | 78.67% | 1.67 ms |
| **PriorSeek (Hybrid-RRF)** | **0.7012** | **85.00%** | **5.62 ms** |

## 特色
- **零超参原则**：检索外壳不包含任何可调超参，所有先验权重均通过离线内核生成，确保实验结果的可复现性。
- **极致性能**：通过 SIMD 算子与有界堆截断，实现 p50 < 6ms 的纯 CPU 查询响应。
- **工业级契约**：严格遵循 `docs/STANDARD.md` 指标口径，禁止通过相关性标注调参。

## 快速上手

```bash
# 构建实验环境
cargo build --release

# 下载先验因子权重 (来自魔搭 Hub)
sl artifact sync --model Reasone/PriorSeek

# 在 SciFact 数据集上跑一轮对比
sl run --dataset scifact --adapters bm25,hybrid-rrf
```

## 协议
本项目基于 **MIT** 或 **Apache-2.0** 双许可协议。
