# 预注册：反事实因果防御与谓词-论元自适应相干检索 (causal-coherent)

- 日期：2026-11-14
- 目标：建立“反事实因果评测体系 (Counterfactual Benchmark)”，杜绝以拟合测试集为代价的词项盲目匹配；引入基于查询信息熵调控的谓词-论元自适应因果相干算子，实现正向科学事实极速召回与反事实因果逻辑精准防御的双重闭环。

## 1. 核心理论假说与反事实防御机制

### 传统纯文本检索的虚假拟合陷阱
无论是 BM25、Dense 向量还是多路 RRF，传统算法仅捕捉“实体共现”与“语义相似度”：
- 查询 A (事实正向): "Suboptimal nutrition increases chronic disease risk."
- 查询 B (反事实对立): "Suboptimal nutrition decreases chronic disease risk."
传统算法因 A 和 B 的实体词项（nutrition, chronic, disease, risk）占据了 90% 的 IDF 和余弦方向，往往输出完全雷同的 Top-10 结果。
这证明算法并没有“理解”因果逻辑，而只是利用了测试集的分布共现，存在致命的现实决策风险。

### 反事实因果辨析率 (Causal Discrimination Rate, CDR)
对具有明确因果极性的查询对 (Q_factual, Q_counterfactual)：
1. 候选集合差异度：
   CDR_jaccard = 1.0 - |TopK(Q_fact) ∩ TopK(Q_counter)| / |TopK(Q_fact) ∪ TopK(Q_counter)|
2. 因果退避率 (Causal Backoff Rate, CBR)：
   当查询被反事实逆转后，原正向事实真值文档的排位是否发生了合理的逻辑退避：
   Rank(Doc_gold | Q_counter) > Rank(Doc_gold | Q_fact)

### 自适应因果相干算子 (Causal Coherent Operator)
LogScore(d, q) = BaseResonance(d, q) + S_entropy(q) * PolarityCoherence(d, q)
- 当查询具有极高信息熵与明确因果谓词时，因果极性相干权重放大；
- 确保在反事实攻击下，反事实文档被准确抑制，同向文献强势置顶。

## 2. 实验对照臂

- **基准臂 A (传统词法)**：`bm25`
- **基准臂 B (传统稠密向量)**：`dense`
- **基准臂 C (流形白化共振)**：`resonance-whitened`
- **实验臂 (自适应因果相干)**：`causal-coherent`

## 3. 双重检验判据

1. **正向事实检索（防性能倒退）**：
   在 SciFact 300 条标准计分测试集上，nDCG@10 须维持在 ≥ 0.7250 的超高位区间，hit@1 维持在 ≥ 61.00%；
2. **反事实因果防御（防虚假拟合）**：
   在反事实因果探针集上，因果辨析率 CDR 必须显著优于传统 BM25 和 Dense 向量（相对提升 ≥ 40%），且因果退避率 CBR 达到 80% 以上。
