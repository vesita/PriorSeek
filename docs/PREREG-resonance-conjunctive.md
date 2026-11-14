# 预注册：合取命题覆盖度相干流形共振算子 (resonance-conjunctive)

- 日期：2026-11-14
- 目标：解决单高频专有名词（如单一稀有基因名）在加性词法势能中形成垄断、导致缺乏关键谓词的文档反超完整命题真值文档的缺陷（单词垄断病态）。

## 1. 核心理论模型

在科学事实与逻辑验证中，查询往往是一个合取命题：
P = A AND B AND C
传统加性词法势能 phi_lex = sum_t score(t) 允许 A 词频极大时掩盖 B 与 C 的完全缺失。

在先验流形共振能量公式中引入无超参合取命题覆盖度对数相干因子：
CoverRatio(d, q) = |{t in q_content | tf_eff(t, d) > 0}| / max(1, |q_content|)
对数势能空间算子更新为：
LogScore(d, q) = gamma * <q_dense, d_dense> + ln(1.0 + phi_lex(d, q)) + beta * ln(0.2 + 0.8 * CoverRatio(d, q))

- 当且仅当命题完整覆盖（CoverRatio = 1.0）时，惩罚归零；
- 当仅出现单一单词而漏掉其他核心意图词项时（如 CoverRatio = 0.25），产生天然的对数势能抑制；
- 物理预期：解救那 26%（78 条）排在第 2~4 名、被单一高频词文档压制的真值文献。

## 2. 实验对照臂

- **地板行**：`bm25` (0.6647)
- **对照臂 A**：`hybrid-rrf` (0.7002)
- **对照臂 B**：`tri-hybrid-rrf` (0.7057)
- **当前 SOTA**：`resonance` (0.7279, hit@1 = 60.00%)
- **实验臂**：`resonance-conjunctive` (合取覆盖相干流形共振)

## 3. 判据与预测

- **正向判据**：
  在 SciFact 300 条计分测试集上，`resonance-conjunctive` 的 nDCG@10 应进一步突破当前 0.7279 的 SOTA，hit@1 突破 60.00%（冲击 61%+）。
- **性能判据**：
  覆盖度统计在倒排链取交集或查询词项遍历时完成，纯 CPU 增加开销 < 0.1ms，总 p50 维持在 1.2ms 以内。
