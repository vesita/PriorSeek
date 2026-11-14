# 预注册：谓词论元因果相干流形共振算子 (structural-resonance)

- 日期：2026-11-14
- 目标：解决传统稠密内积与 BM25 无法辨析因果方向（如 A promotes B vs A inhibits B 余弦高度重叠、词项全同）的盲目性问题，引入离线结构解耦的因果谓词拓扑先验，从本质上防范过拟合与反事实混淆。

## 1. 核心理论假设与物理机制

科学命题的本质是三元事实：(Subject, Predicate, Object)。
目前系统在纯实体匹配上已经达到极高水准（hit@1 = 62.00%），但在谓词因果方向上缺乏结构先验：
- 极性 +1 (Positive Induction): increase, promote, activate, upregulate, cause, stimulate
- 极性 -1 (Negative Suppression): decrease, inhibit, suppress, downregulate, reduce, block, prevent
- 极性 0 (Neutral Association): bind, interact, associate, correlate

### 结构因果相干算子 (Structural Coherence)
令 Polarity(q) 为查询主谓语的因果极性，Polarity(d, (S, O)) 为文档在论及主宾实体时的因果极性：
- 同向共振 (Coherent): Polarity(q) * Polarity(d) > 0，相干能量激发；
- 矛盾对立 (Counterfactual Deficit): Polarity(q) * Polarity(d) < 0，能量阻尼相消。

在对数流形势能空间中计算：
LogScore(d, q) = BaseResonance(d, q) + delta_struct * PolarityMatch(d, q)

## 2. 实验对照臂

- **地板行**：`bm25` (0.6647)
- **基准臂 A**：`hybrid-rrf` (0.7002)
- **基准臂 B**：`resonance-conjunctive` (0.7283)
- **当前 SOTA**：`resonance-whitened` (0.7291)
- **实验臂**：`structural-resonance` (谓词论元结构相干共振)

## 3. 判据与反事实核验

- **主指标判据**：
  在 SciFact 300 条计分测试集上，nDCG@10 须进一步提升（目标冲刺 0.7300+），hit@1 维持在 ≥ 61.50%。
- **反事实辨析度核验（防过拟合检验）**：
  针对具有明显极性对立的查询对（例如 Query 1020: "increase survival..." vs Query 1021: "reduce survival..."），核验两者的排位是否能够基于因果谓词正确分离，而非输出完全雷同的候选。
