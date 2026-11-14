# 预注册：三元组因果拓扑检索引擎 (triplet-causal)

- 日期：2026-11-14
- 核心哲学：**从统计词频转向逻辑演绎**。彻底终结基于词法共现的检索模式，通过将科学文本重构为 (Subject, Relation, Object) 三元组索引，在底层逻辑层面杜绝反事实逻辑误召回。

## 1. 核心理论模型：因果图检索 (Graph-Path Retrieval)

科学知识不是词的集合，而是命题的连接。我们将检索任务重定义为：在知识拓扑图中寻找匹配查询意图的路径。

### 三元组索引布局 (Triplet Inverted Index)
- **索引单元**：`TripletIndex { S_map: Map<Entity, Vec<Relation>>, R_map: Map<Relation, Vec<DocID>> }`
- **查询映射**：查询被弱深度学习解析为 `(S_q, R_q, O_q)`。
- **检索算子**：
  Score(d, q) = IsMatch(S_q, S_d) * IsMatch(R_q, R_d) * IsMatch(O_q, O_d)

### 反事实防御本质
在索引结构上，`(A, promotes, B)` 和 `(A, inhibits, B)` 是两条完全不同的路径。反事实查询在检索阶段**由于谓词 Relation 节点的不匹配，其召回路径直接断开**，导致反事实召回率为 0%，CDR 辨析度达到逻辑极限。

## 2. 弱深度学习内核：Micro-Relation Extractor

- **模型设计**：一个微型嵌入式标注器，仅负责抽取实体 (Span) 与语义谓词 (Predicate Classification)。
- **物理约束**：谓词集合定义为规范化的科学因果动词集（增加、减少、导致、抑制等），消除文档间的术语异构。

## 3. 核心判据

1. **反事实逻辑辨析 (Logic-CDR)**：
   反事实查询对下的真值文档召回率为 **0%**，对于因果链路的辨析度达到 **100% 逻辑完备**；
2. **知识检索精度**：
   在 SciFact 领域实体链路验证集上，精确匹配 (Exact Triplet Match) 的准确率 ≥ 70%；
3. **计算延迟**：
   单查询的图节点哈希查询耗时 ≤ 0.5ms，实现真正的科学逻辑实时演绎。
