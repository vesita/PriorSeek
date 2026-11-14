# 预注册：论元与谓词结构解耦共振算子 (decoupled-resonance)

- 日期：2026-11-14
- 目标：彻底打破“整句混合倒排”导致的词频掩盖因果致命病态，在检索的第一阶段将查询解耦为“实体论元核 (Arguments Core)”与“因果谓词头 (Predicate Head)”，实行双轨相干共振，兼顾超高正向检索精度与高强度反事实因果防御。

## 1. 核心理论模型与物理机制

在科学文献中：
- 实体论元（名词实体）：决定“讨论的是什么主题”（如 Rapamycin, triacylglycerols, fruit flies）；
- 关系谓词（动词关系）：决定“表达的是何种因果趋势”（如 increases, decreases, promotes, blocks）。

### 双轨解耦流形共振架构 (Two-Track Decoupled Resonance)
1. 论元倒排锚定轨 (Argument Inverted Track)：
   过滤掉动词与修饰词，仅对核心名词实体与领域概念词提取倒排候选，确保实体锁定 0 漂移；
2. 谓词相位相干轨 (Predicate Phase Coherence Track)：
   在线提取查询谓词极性 Pol(q)，匹配文档因果极性先验 Pol(d)；
   - 同向因果共振：Phase(q, d) = +0.25 (当 Pol(q) * Pol(d) > 0)
   - 反事实相位相消：Phase(q, d) = -0.75 (当 Pol(q) * Pol(d) < 0，强力阻尼相位抵消！)
   - 中性：Phase(q, d) = 0.0
3. 连续流形相干势能：
   LogScore(d, q) = gamma * <q_dense, d_dense> + ln(1.0 + phi_arg(d, q)) + beta * ln(0.2 + 0.8 * Cover_arg(d, q)) + Phase(q, d)

## 2. 实验对照臂

- **基准臂 A (传统无因果流形共振)**：`resonance-whitened` (nDCG@10 = 0.7291，但 CBR 仅 24.24%)
- **基准臂 B (末端阻尼结构共振)**：`causal-coherent` (nDCG@10 = 0.7149, CBR = 25.00%)
- **实验臂 (首阶段双轨结构解耦)**：`decoupled-resonance`

## 3. 双向严苛判据

1. **正向事实基准 (SciFact 300 计分集)**：
   nDCG@10 须维持在 ≥ 0.7250 极高位区间，hit@1 ≥ 61.00%；
2. **反事实盲测防御 (83 对因果对立探针)**：
   因果退避率 CBR 必须突破 50% 大关（相比传统 resonance 的 24.24% 实现翻倍级防御暴增！），同时因果辨析率 CDR ≥ 60%。
