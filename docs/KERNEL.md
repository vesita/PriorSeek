# 内核设计：认知 token + token 关联 + 搜索权重

> 方向定调：**外壳是明确的，内核是深度学习的**。
> 外壳（数据契约 / 指标 / runner / 报告 / 公平性）由 [`docs/STANDARD.md`](STANDARD.md) 固定，
> 换内核不改变任何口径；内核负责「把查询和文档变成 token、建立关联、产出**先验因子**」，
> 训练可以用 GPU，**运行时必须能在无 GPU 环境跑**。

## 0. 术语：内核网络 / 先验因子 / 外壳（三者不许混用）

**先验因子（prior factor）**是内核与外壳之间的**唯一边界**：内核网络**产出**它，外壳**消费**它。
把"模型自己有多少参数"和"模型产出多少东西"分开，是这套架构能被正确评估的前提。

| 术语 | 定义 | 例子 | 计量单位 |
|---|---|---|---|
| **内核网络** | 读文本、产出先验因子的神经网络（或可训练模块） | 5 参数的低秩词项权重网络；24k 参数的投影头 | **参数量**（可训练权重的个数） |
| **先验因子** | 内核网络的**输出**，外壳检索时直接消费 | 每个 token 的权重 `w(t)`；gap 权重表（5 个数）；token 向量 | **因子规格**（形状 / dtype / 个数），与参数量无关 |
| **外壳** | 用先验因子执行检索的算法 | 加权 BM25；OAC 有序对齐 | 算法超参（**不是**模型参数） |

为什么必须分开（三条实际后果）：

1. **同一个网络可以产出不同规模的因子**：5 个参数的网络可以产出 35,948 个 token 权重
   （低秩参数化：`w(t) = softplus(a1·idf + a2·log df + … + b)`）；反过来一个大投影头可能只产出
   一个 32 维向量。**"内核要小"约束的是参数量，不是因子规模。**
2. **决定优化器选择**：ES / 黑盒搜索的可行性由**参数量 d** 决定（`O(d)`~`O(d²)` 次评测），
   与因子维度无关。d 在几十~几百用 ES 很划算；到了上万就必须回到梯度。
3. **决定工件契约**：工件必须**分别**记录「先验因子规格」与「网络参数量 / 权重哈希」，
   否则无法判断"效果变化来自更强的网络，还是更多的因子"。

> 历史澄清：此前说"内核参数量 = 2"（`a1`、`a2`）其实是**两个 2 恰好重合**——那时网络参数就是
> 那两个标量、因子也是那两个标量。引入真正的网络后二者必然分叉，本节的术语从此刻生效。

## 1. 三层：实验台 / 外壳（检索管线） / 内核（先验因子提供者）

```text
┌─ 实验台（评测标准，docs/STANDARD.md）───────────────┐
│ 规范数据+指纹 · 指标口径 · 同轮同题 · 成本分列 · 报告 │
│  └─ 外壳（检索管线，可设计算法）─────────────────┐   │
│     token 化 → 基本特征（tf/df/长度/窗口）         │   │
│              + 内核特征（先验权重/时序相关/关联/向量）│
│              → 打分融合 → 索引与检索               │   │
│     └─ 内核（离线训练的特征提供者）──────────────┐  │   │
│        语料统计 / GPU 训练 → 工件（可审计、CPU 运行）│  │   │
└──────────────────────────────────────────────────┘
```

- **内核**只提供「特征」：token 的先验权重、token 间的时序/关联强度、概念 id、向量等，
  以离线工件形式落地；它不决定怎么融合、怎么检索。
- **外壳**是检索管线：token 化 → 取基本特征 + 内核特征 → 设计打分融合 → 建索引/检索。
  同一份内核特征可以被不同的外壳算法消费，互换外壳是**可比**的（同数据同 k）。
- **实验台**是外层评测标准：数据契约、指标、runner、报告、公平性；内核与外壳都不许改它。

**一条推论**：内核的一切「学习」都必须落在**离线工件**上，运行时只做查表 / 稀疏矩阵运算。
这既满足「运行时非 GPU」，也让内核可审计（工件带哈希、来源、训练数据指纹）。

## 2. 内核的四个部件

### 2.1 认知 token（Cognitive Token）

不是字面词，而是「带语义身份与权重的检索单元」。一个认知 token 至少四层：

| 层 | 内容 | 学习方式（离线） | 运行时形态 |
|---|---|---|---|
| 字面 | 分词单元（词 / 双字 / 子词） | 分词器（现有：jieba / Unicode） | 查表 |
| 概念 | 概念 id（同义、上下位、跨语言归并） | 对 embedding 聚类 / 本体映射 | token → concept 列表 |
| 类型 | 实体 / 动作 / 属性 / 关系… | 小分类器（GPU 训练） | token → 类型权重向量 |
| 权重 | 该 token 的检索先验重要性 | 统计（IDF）或学习（GPU 训练的权重头） | token → f32 |

第一版可以只做「字面 + 概念 + 权重」三层；**类型层是可消融项**，先留接口不做实现。

### 2.2 token 关联：链 / 树 / 图的统一

用户提的三种结构不是三选一，而是**同一个带类型的有向加权多重图**的不同边子集：

| 结构 | 边语义 | 例子 | 退化/消融方式 |
|---|---|---|---|
| 链（chain） | 时序 / 相邻（order-1、order-2 转移） | 「参数→保存→生效」的顺序 | 只保留 `seq` 边 |
| 树（tree） | 层级 / 依存 / 上下位 | 短语结构、概念上下位 | 只保留 `hier` 边 |
| 图（graph） | 一般关联（共现、语义近邻、学习关系） | 「重启 ↔ 生效」「磨损 ↔ 补偿」 | 只保留 `assoc` 边 |

统一表示：

```text
节点 V    = 认知 token（概念 id 可以复用同一节点，带类型）
边   E    = (src, dst, type ∈ {seq, hier, assoc}, weight f32)
```

**为什么统一成图**：图是链和树的超集；结构消融只需在运行时开关边类型，不需要三套实现。
这也让「链状 / 树状 / 图」变成一个可测量的实验变量，而不是口头选择。

### 2.3 搜索权重：从 token 权重到文档分数

```text
基础权重：  w0(t) = 统计权重（IDF 或学到的 token 先验权重）
传播：      w(t) = w0(t) + α · Σ_{s ∈ N_in(t)} edge_weight(s→t, type) · w0(s)   （跳数 ≤ 2）
打分：      score(q,d) = Σ_{t ∈ q} w(t) · tf_norm(t, d)      （tf_norm 沿用 BM25 的饱和与长度归一）
可选融合：  + β · cos(emb(q), emb(d))                        （dense 先验，见 §7 基线）
```

- 传播是**离线可预计算 + 在线稀疏运算**：查询侧只对查询 token 做 ≤2 跳传播（每 token 保留 top-k 邻居），
  所以在线成本是 `O(|q| · k · hops)`，与语料规模无关。
- 边权重与 α / β / hops 都写进 `AdapterInfo.config` 与工件 manifest，不允许隐式调参。

### 2.4 训练（GPU）与运行（CPU）分离

| 阶段 | 环境 | 产出（离线工件） |
|---|---|---|
| 语料统计 | CPU 即可 | 词表、DF/IDF、共现计数 |
| 图学习 | **GPU**（可选）：对比学习 / 蒸馏出边权；或 PPMI+SVD 无监督 | `graph.bin`（节点表 + 边表 + 权重） |
| 概念层 | GPU：embedding 聚类 | `concept.bin`（token → concept + 亲和度） |
| 类型/权重头 | GPU 训练（需要 train split） | `token_weights.bin` |
| 运行时 | **CPU**（无 GPU） | 只读上述工件 + 稀疏传播打分 |

验收必须包含一条「**GPU 训练 → CPU 运行**」的端到端：同一份工件在 CPU 上打分与训练环境一致（容差内）。

## 3. 数据与泄漏（最关键的一条）

- **无监督关联**（PPMI、SVD、embedding 近邻）不接触 qrels，可以从评测语料本身构建——这与 BM25 用
  语料算 IDF 同性质；但必须在工件 manifest 里声明「语料内统计」（transductive），并注明语料指纹。
- **有监督训练**（学边权、学 token 权重）只能用 train split。当前四个数据集只有 test qrels
  （scifact 有 `qrels/train.tsv`，尚未接入规范格式）⇒ 需要扩展数据契约：可选
  `train_queries.jsonl` / `train_qrels.tsv`，训练工具读、**适配器永远读不到**。
- 预训练模型（bge / MiniLM 等）可能见过 BEIR / NQ 数据，这是**污染风险**，必须在工件 manifest 的
  `contamination` 字段写明（模型卡里的训练数据范围 + 我们的判断），不能装作不存在。

## 4. 工件契约（外壳唯一需要扩展的地方）

`sl run` 要把内核工件当一等公民记录与校验：

```text
KernelArtifact {
  id, kind ∈ {embedding, graph, concept, token_weights, vocab},
  # —— 先验因子规格（内核的**输出**）——
  prior_factor: { name, kind, shape, dtype, count },
  # —— 内核网络本身（模型的**参数**）——
  network: { arch, params, checkpoint_sha256 },
  path, sha256, dtype, shape,
  trained_on: [数据集指纹 + split],        // 无监督则写「语料内统计」+ 指纹
  encoder: { model_id, revision, dim },     // 需要编码器时
  contamination: "...",                     // 预训练数据是否可能覆盖评测语料
  runtime: { device: cpu, precision: f32 }, // 运行时要求
  license, notes
}
```

**先验因子的两条硬约束**（由 `sl_core::types::PriorFactor` 强制，强制点登记在 `docs/STANDARD.md` §11）：

1. **`kind` 是封闭集合**：`token_confidence_list` / `concept` / `field`。新增一种要同时改类型、工件构建器与消费它的外壳，否则解析期直接失败。
2. **只许表达证据，不许表达用法**：`alpha` / `weight` / `top_k` 这类描述「怎么用」的字段一律被拒绝——用法归外壳的算法超参，这样换外壳不必重定因子。

因子字段缺失或为 `null` 表示该工件不提供因子，外壳据此走无因子路径。

- `AdapterInfo` 增加 `artifacts: [KernelArtifact]`，写进 `run.json`；报告里显示工件哈希前 12 位。
- 工件缺失 / 哈希不符 → 构造适配器时直接失败（同 rs 二进制的处理）。
- `sl artifact verify <run.json>` 离线复核（换机器后确认工件没被换掉）。

## 5. 里程碑（每个都有可判定的验收）

| 里程碑 | 内容 | 判据 |
|---|---|---|
| **M0 外壳补齐** | `KernelArtifact` + `train` split 契约 + `sl artifact verify` + dense 内核适配器（离线编码、CPU 精确内积） | 四个数据集跑通，run.json 有完整工件记录，CPU 单查询延迟与内存可报告 |
| **M1 关联图内核** | 无监督 PPMI+SVD 建图（词表 + top-k `assoc` 边）→ CPU 稀疏传播打分 | 与 `bm25` / `faq-bm25` / M0 的 dense 同一张表对比；离线工件可复现（哈希稳定）；单查询延迟 < 5ms |
| **M2 认知 token** | 概念聚类层（token→concept）接入，消融「有/无概念层」 | 同数据集同 k 的消融表；结论标注证据强度 |
| **M3 结构消融** | 链（seq）/树（hier）/图（assoc）边类型开关 | 三类边组合的消融表；回答「哪种结构在哪些语料上有效」 |
| **M4 学习边权** | GPU 上用 train split 学边权 / token 权重 + 蒸馏到稀疏工件；CPU 运行对拍 | 训练/评测隔离可审计；CPU 与训练环境打分之一致；效果相对 M1 的提升（或明确「没测出差异」） |

## 6. 与现有结果的关系（M0 之前的证据）

用 `bge-small-zh-v1.5` / `all-MiniLM-L6-v2` 在 CPU 上做纯 dense 检索（离线编码 + GPU 算相似度，
仅作为**基线**，尚未进本仓的正式对比表）：

| 数据集 | dense 单路 nDCG@10 | 当前最好的非内核方案 |
|---|---:|---|
| qbqtc（中文） | **0.8838** | `faq-bm25` 0.8697 |
| scifact（英文） | **0.6810** | `rs` 0.6785 |

这说明「内核换成深度学习」在中文短文本上已有可观增益；但这两个数字**不是**正式结论——
编码器训练数据可能覆盖评测语料（污染），且未按本仓工件契约登记。M0 会把它们变成合规的对照行。

## 7. 成本口径（新增，写进 STANDARD §5）

- **离线成本**：训练/编码时长、GPU 型号、语料规模；工件大小；
- **在线成本**：索引构建（含工件加载）、单查询延迟（**CPU**，p50/p95）、内存峰值；
- 报告里离线与在线成本分列，不合成单一分数；若某内核只能在 GPU 跑，直接判不合规（运行时要 CPU）。

## 8. M1 实现规格：`seq` 内核（token 时序相关度 v0）

> 定位（暂定）：内核先只做「**token 时序相关度**」——查询 token 之间的顺序/邻接关系，如何影响
> 每个 token 的检索权重与文档打分。结构上只用「链」（`seq` 边），树/图留到后续里程碑。
> v0 的权重来自**语料内统计**（PPMI，无监督、不看 qrels）；M4 再把权重换成 GPU 学习/蒸馏的版本，
> 工件格式与运行时打分不变 => 可以逐位对拍「统计先验 vs 学习先验」。

### 8.1 打分定义（v0）

```text
tokenize(q) -> [t1..tn]                      # 与现有适配器同一分词器
w0(t)        = idf(t)                        # 来自工件（语料内统计）
unigram:  s_uni(d) = Σ_t w0(t) · tf_norm(t, d)        # tf_norm 用 BM25 饱和 + 长度归一（k1=1.5, b=0.75）
seq 对:   P = {(q_i, q_{i+1})} ∪ {(q_i, q_{i+2})}      # 相邻 + 隔一
时序分:   s_seq(d) = Σ_{(a,b) ∈ P, e(a,b) > 0} e(a,b) · sat(c_d(a, b, W))
          e(a,b)   = 工件里的 PPMI 转移权（order=1 或 2），查不到记 0
          c_d(a,b,W) = 文档里满足 pos_a < pos_b 且 pos_b - pos_a ≤ W 的匹配次数
          sat(c)   = 1 + ln(c)  (c ≥ 1)，c 计入上限 3
总分:     score(q,d) = s_uni(d) + λ_seq · s_seq(d)
```

消融开关（都进 `AdapterInfo.config`）：

PPMI 权重（`method=ppmi_topk`）：

```text
c(a,b)    = 文档内 order 阶（1/2）有序共现计数
pmi(a,b)  = ln( c(a,b)^0.75 · N_pair / ( c(a)^0.75 · c(b)^0.75 ) )
weight    = max(0, pmi(a,b) − ln(ppmi_shift))
筛选      = c(a,b) ≥ min_df，且每个 src 只保留 weight 最大的 top_k 条出边
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `lambda_seq` | 0.5 | 时序分权重（0 = 退回纯 BM25） |
| `window` | 8 | 有序匹配的最大 token 距离 |
| `orders` | [1,2] | 用哪些阶的转移边（`[1]` = 只相邻） |
| `ordered` | true | `false` 时改为无序邻近（同样窗口），作为「顺序是否重要」的对照 |
| `max_matches` | 3 | `c` 的截断，避免长文档靠重复刷分 |

### 8.2 工件格式（`data/<name>/artifacts/seq-v0/`）

```text
manifest.json    元数据（见下）
seq.bin          词表 + idf + 边表（小端）
docs.txt         文档 id 列表（一行一个，顺序与 doc_tokens.bin 一致）
doc_tokens.bin   每篇文档的 token id 序列（由构建器用 Python 分词器切好，避免两端分词差异）
```

> **为什么要 `doc_tokens.bin`**：词表由构建器（Python）产，若让 Rust 侧再切一次文档，
> Python jieba 与 jieba-rs 的极少数切词差异会让文档 token 与词表错位（OOV、位置偏移）。
> 把文档侧 token 序列固化进工件，两端只共享同一份结果；查询侧仍由 Rust 现切（差异只会丢个别
> OOV 查询词，可接受且有测试覆盖）。

`manifest.json` 字段（必须能被 `KernelArtifact` 契约消费）：

```json
{
  "id": "seq-v0",
  "kind": "seq_graph",
  "method": "ppmi_topk",
  "weights_origin": "corpus_ppmi",
  "source_corpus": {"dataset": "<name>", "fingerprint": "<数据集 manifest 指纹>"},
  "params": {"orders": [1, 2], "top_k": 20, "window": 8, "min_df": 2, "ppmi_shift": 5, "smoothing": 0.75},
  "counts": {"vocab": 0, "edges": 0},
  "files": {"seq.bin": {"sha256": "<hex>", "bytes": 0}},
  "runtime": {"device": "cpu", "precision": "f32"},
  "trained_on": "语料内统计（无监督，未使用 qrels）",
  "contamination": "无预训练模型参与，无外部语料",
  "license": "由本仓从数据集语料构建",
  "notes": ""
}
```

`seq.bin`（小端，无对齐填充）：

```text
seq.bin（小端）：
  magic     8 字节  b"SLSEQv1\0"
  vocab     u32 n，随后 n 条：u16 len + UTF-8 bytes（按 token 字典序升序存，便于二分查找）
  idf       n × f32
  edges     u32 m，随后 m 条：u32 src, u32 dst, u8 order, f32 weight
            边按 (src, dst, order) 升序；src/dst 是词表下标；order ∈ {1,2}

doc_tokens.bin（小端）：
  magic     8 字节  b"SLDOCv1\0"
  docs      u32 d
  每篇：u32 len，随后 len 个 u32 token id（按文档内出现顺序，长度 = 该文档 token 数）
```

`docs.txt` 与 `doc_tokens.bin` 的文档必须一一对应且顺序一致；适配器加载时必须校验：
文档数一致、id 集合与数据集语料一致（不一致直接报错，不做静默对齐）。

### 8.3 交付物与验收

| 交付 | 位置 | 判据 |
|---|---|---|
| 工件构建器 | `tools/kernel/build_seq.py`（Python + uv；CPU 可跑，无 GPU 依赖） | 同一数据集构建两遍字节一致；manifest 的 `seq.bin` / `doc_tokens.bin` sha256 与文件一致；`sl artifact verify` 通过 |
| Python 参考打分 | `tools/kernel/ref_score.py` | 给定数据集/查询/工件，输出 top-k（doc id + 分数），供 Rust 对拍 |
| Rust 适配器 | `crates/sl-adapters/src/seq.rs`（名字 `seq`） | 与 Python 参考实现对同一 fixture 的 top-k 与分数（1e-6 相对误差）一致；CPU 单查询 < 5ms（nq-sub 规模） |
| 效果对比 | `docs/RESULTS.md` 新增一节 | 四数据集上与 `bm25` / `faq-bm25` 同表；至少一组消融（`lambda_seq=0` vs 0.5、`ordered` true/false、`orders=[1]` vs `[1,2]`） |

### 8.4 约束

- v0 构建器**不读 qrels**（只看 `corpus.jsonl`）；`trained_on` 必须如实写「语料内统计」。
- 工件按数据集生成（`source_corpus.fingerprint` 绑定），换数据集不能复用。
- 运行时必须纯 CPU；构建器允许可选 GPU，但 v0 不需要。

## 9. 范围决定（2026-11-09）：MVP 只做 token 化 + 时序相关性

按讨论收敛：**深度学习先不铺开**，当前目标是把「token 化 + 时序相关性」这条内核链路跑通并如实记账。

- 内核 v0 的**先验因子** = 语料统计的时序边权（PPMI，order=1/2）+ 词表/idf + 固化的文档 token 序列
  （注意：v0 没有神经网络，因子由统计直接算出；引入网络后因子改由网络产出，外壳不变）；
- 不做：概念层、类型层、树/图结构、大模型训练、在线编码；
- 学习版（轻量：例如在 train split 上学边权，或小规模自监督向量）**只换工件**，外壳打分公式、指标口径、报告格式不变；
- 结果如实记录：统计版时序分在四个数据集上**没有增益**（`docs/RESULTS.md` §8），这是 MVP 的基线，不是「时序相关度无效」的结论；
  学习版要回答的是同一问题，且必须相对这个基线比较。
- **kernel v1 已试**（`docs/PREREG-M2.md`、`docs/RESULTS.md` §9）：在 train 上学 2 个时序权重，
  预注册口径得到边界解（权重=0）、非零性冒烟未过 ⇒ 该变体作废；诊断显示时序特征本身有信号
  （单独 AUC 0.79）但「加性 bonus」的集成方式学不出增益，且 scifact test 的分辨率只有 0.0143。
  ⇒ 下一步应动**集成方式或测量分辨率**，不是参数数量。
- **先验因子 v0 已试**（`docs/RESULTS.md` §13）：`token → 关联 token + 置信度`（P0 统计基线，
  网络参数量 0）。固定 alpha 总体变差（nDCG 0.6647→0.6460），但**硬漏子集 31 条救回 5 条**；
  逐查询 oracle 上限 **+2.36pp nDCG / +2.33pp hit@10**（超过 Δ_min=1.43pp）⇒ 方向可判定，
  且只有约 9% 查询适合扩展 ⇒ 必须学**逐查询/逐 token 置信度**（P1），不能用全局 alpha。
- **语法（词性）因子已试并证伪**（`docs/RESULTS.md` §19）：在 train split 的 heldout 上，
  `w(t) = Σ_k p(k|t)·λ_k`（k = UPOS 17 类，λ 在 outer-train 拟合）的 Δ nDCG@10 = **−0.00018**，
  低于「df 分层置换」对照的 +0.01505 ⇒ 增益可被「任意 df 保序分箱 + K 个自由乘子」完整解释。
  随后又测了**结构匹配**用法（`docs/RESULTS.md` §21）：只用词性筛出「构成短语」的查询相邻对
  （ADJ+NOUN、NOUN+NOUN 等）给相邻性奖励，Δ = **+0.00011**，且不如「随机抽同样数量的相邻对」
  （−0.00255）⇒ 也是零。**语法因子（乘子 / 结构匹配两种用法）均不进外壳**；
  查询路由与谓词门控未测。
  **不过这两节的代理精度随后被测过（`docs/RESULTS.md` §22）：词形级查表在 EWT test 上
  89.47%、CRAFT 域内 95.20%；为此新建的上下文标注器达到 EWT 91.91% / CRAFT 96.70%，
  仍**未通过**「EWT ≥95% 且 CRAFT ≥97%」的验收门槛，因此逐次词性的检索实验未运行。
  ⇒ 词性这条线的状态是「证据为负、且工具不足」，**不能**判为「词性无用」。
- **kernel v1.1 已试**（`docs/PREREG-M2b.md`、`docs/RESULTS.md` §10）：目标改成业务口径的
  **搜索缺失率**、优化方式改成**精确网格搜索**（不用代理损失），train 上 1402 个网格点最多多命中
  1 条查询，test 上 **0 条查询的命中状态变化**（Δhit@10 = 0.0000）。
  ⇒ 结论：**2 个加性时序权重对「救漏」没有作用**；缺失率的两类来源（重排可救 9.3%、词汇不匹配 10.3%）
  分别指向重排层与召回/表示层，**不属于「内核只学时序关联」的范围**。

## 10. 概念归并内核（`concept-v0`）：先验因子由**学习**产出

这是第一个「因子由学到的权重决定」的内核：先验因子的内容不是统计量的直接复制，
而是一个带参数的判定谓词在语料上的输出。结果见 [`docs/RESULTS.md`](RESULTS.md) §15。

### 10.1 边界：什么是固定的，什么是学出来的

| 层 | 内容 | 谁决定 |
|---|---|---|
| 候选对空间 | 共享前缀 ≥ 较长词的 `lcp_ratio`（默认 0.7）、`len ≥ 4`、长度差 ≤ 8 | **结构约束**（语言无关，不可学习） |
| 特征 | `p_t_given_u` / `p_u_given_t` / `jaccard` / `log_df_ratio` / `len_ratio` / `extra_chars` / `tf_cosine` / `ppmi` | 语料统计（无监督） |
| 判定 | `merge(t,u) = σ(w·x + b) ≥ 0.5` | **内核网络**：`w`、`b` 在 outer-train 上学出来 |
| 归并结果 | `token → 规范形式`（组内 df 最大者为规范形式） | 谓词在语料上的输出 = **先验因子** |
| 怎么用 | 硬归并、是否叠加字段加权 | 外壳（算法超参） |

没有任何英文后缀表：`-s` / `-ed` / `-ing` 这类知识要么出现在学到的 `w` 里，要么根本不出现。
参数量 = 8 + 1 = **9**，在 `AGENTS.md` 定的 ≤10k 预算内。

候选空间用**共享前缀比例**而不是「严格前缀」，是因为严格前缀同时犯两个错：放进
`the`/`therapies` 这种毫无形态关系的对（内容词被并进停用词），又漏掉词干变化的
`-y→-ies`（`antibody` 不是 `antibodies` 的前缀）。实测覆盖从 6/228 提到 145/228，
停用词污染从 127 个降到 1 个（`docs/RESULTS.md` §17.3）。

两个候选规则都保留为**显式参数**（`rule = lcp | prefix`），因为已发布的 `concept-v0`
用的是 `prefix`，要能被逐字节复现；规则记在工件的 `params.candidate_rule` 里，
缺了就停车，不猜默认值。

### 10.2 训练与评测隔离

- 权重只在 **outer-train**（663 条，train split 内按 `FNV-1a(qid, seed=20261110) % 5 != 0`）上学，
  目标就是 harness 指标 nDCG@10，用坐标搜索（不用代理损失，避免代理与指标不一致）；
- **heldout**（146 条）只用于判决，不参与任何一步拟合；
- **默认不读 test qrels**：`sl concept-eval` 的默认划分是 train split 内的 heldout，
  工件构建器只读 `corpus.jsonl`；
- **`--split test` 需要预注册**：它读完整数据集（含 test qrels），只在写定了判据的
  迁移检验里用一次（已用过一次：`docs/PREREG-concept-transfer.md` → scifact 权重 → arguana）。

### 10.3 工件（`data/<dataset>/artifacts/concept-<ver>/`）

```text
manifest.json    id / kind=concept_map / weights_origin=learned_weights /
                 prior_factor{name,kind=concept,shape,dtype,count} /
                 network{arch,params,checkpoint_sha256,features,weights,intercept} /
                 counts / files(sha256) / runtime / trained_on / contamination
concepts.json    {"entries": {token: [[规范形式, 1.0]]}}     ← 先验因子本体
weights.json     学到的 w / b、特征缩放、候选规则、目标与划分
                 （network.checkpoint_sha256 指向它；缺 candidate_rule.rule 时构建器停车）
```

同一数据集可以有多个版本共存（`concept-v0` / `concept-v1`，差别见 §10.5），
评测用 `sl concept-eval --artifact <id>` 指定。

`concepts.json` 与 `exp-ppmi-v0` / `field-v0` **同构**（`token → [[目标, 置信度]]`），
所以外壳只多一条查表路径，不多一套解析；置信度恒为 1.0（硬归并）——
软归并（`tf_eff = tf_exact + λ·tf_variant`）已试过 λ∈{0.25,0.5,0.75}，
两套划分上都不优于 λ=1（`docs/RESULTS.md` §18 #3），因此外壳不提供该开关。

### 10.4 外壳怎么消费

`concept-bm25`（`crates/sl-adapters/src/field_bm25.rs`）在**索引期与查询期都**把 token 映射到
规范形式，再进同一份 `PositionalIndex`；idf 定义、`k1`/`b`、同分排序与 `bm25` 完全相同。
映射为空时必须与 `bm25` 逐查询一致，由 `concept_mode_with_empty_map_matches_bm25` 守着。

### 10.5 工件版本

| 工件 | 候选规则 | 归并 token | heldout Δ nDCG@10 | 说明 |
|---|---|---:|---|---|
| `concept-v0` | `prefix`（严格前缀） | 3617 | +0.01872（未过带） | 首个版本，保留用于复现与对照 |
| `concept-v1` | `lcp`（共享前缀 ≥70%） | 8506 | +0.02487（过带） | 当前版本，`concepts.json` sha256 `6a5fde5538f9c10e…` |

`concept-v1` 的增益通过了**等价对照**（`docs/RESULTS.md` §20）：保留被归并 token 集合、
组大小分布与规范形式规则，只随机打乱分组，20 次随机归并的 heldout Δ 均值 **−0.01501**
（区间 [−0.02812, −0.00502]），真实值 `+0.02487` 超出该零分布。⇒ 增益来自「按形态归并」
这件事本身，不是「任意分组」的产物。

两个工件之间**唯一的差别**是候选规则（权重都在这条空间上重训过）。v1 相对 v0 的直接配对差
是 +0.00615（2SE 0.028），**仍落在噪声带内**——所以不能声称「v1 的规则被证明更好」。

### 10.6 这个内核做不到什么

- **它只改词项身份，不改词项权重**：给 per-token 乘系数这条路已被证伪三次
  （`docs/RESULTS.md` §14 的 field per-token 因子、§15.2 的 `field_idf` / `idf_lift` / `saliency`）；
- **候选空间依赖「共享前缀」这个结构**，只覆盖**后缀屈折**形态（英文/德文）。中文没有这种
  形态关系，该候选空间在中文语料上没有意义，未测；
- **它不是局部操作**：归并会改变文档侧词项的 df/idf，影响会泄漏到与形态无关的查询
  （`docs/RESULTS.md` §17.4 有实例）；
- **同义/上下位不在它范围内**：`cancer` 与 `carcinoma` 不是前缀关系，需要 `concept` 的另一种
  实现（本体映射或 embedding 聚类），属于后续里程碑。

## 11. 神经稀疏先验与中英双语解耦扩散（`learned-sparse` & `bilingual-sparse`）

预注册见 [`docs/PREREG-learned-sparse.md`](PREREG-learned-sparse.md) 与 [`docs/PREREG-bilingual-sparse-hub.md`](PREREG-bilingual-sparse-hub.md)，实测记录见 [`docs/RESULTS.md`](RESULTS.md) §36 与 §37。

这是目前 PriorSeek 的核心架构底座，完美贯彻了“内核负责学习并产出先验因子，外壳负责极简组织执行”的第一性原理：

### 11.1 三层物理分离架构

| 层级 | 部件 | 职责 | 实现载体 |
|---|---|---|---|
| **概念层** | `BilingualHub` (双语概念映射枢纽) | 负责跨语言概念对齐（如 `cancer <-> 癌症`），严格隔离语法，不承担单语扩散 | `tools/kernel/bilingual_hub.py`<br>`crates/sl-engine/src/bilingual.rs` |
| **内核层** | `Language-Specific Diffusion Heads` (语系特化扩散头) | 英文走 30k 维英文词表，中文走 32k 维中文词表。同一个词严格在同语系内完成词权加权与上下文语义扩充，杜绝交叉污染 | `naver/splade-cocondenser-ensembledistil`<br>`naver/neuclir22-splade-zh` |
| **外壳层** | `SparseInvertedIndex` (纯 CPU 带权倒排索引) | 消费离线二进制先验工件 `sparse.bin`，纯 CPU 线性内积：`sum(w_q * w_d)`。**外壳完全零超参** | `crates/sl-engine/src/sparse.rs`<br>`crates/sl-adapters/src/bilingual_sparse.rs` |

### 11.2 先验因子契约与工件格式

- 工件：`data/{dataset}/artifacts/bilingual-sparse-v1/`
  - `sparse.bin`：小端二进制紧凑倒排向量（魔数 `SLSPARv1` + 文档数 + 词表大小 + 变长词项 `(token_id: u32, weight: f32)`）；
  - `manifest.json`：符合 `PriorFactor` 封闭规范，声明形状、SHA256、来源模型与训练集，无任何用法字段；
  - `bilingual_lexicon.json`：双语对齐映射字典。

### 11.3 核心收益与物理归因

1. **外壳零超参**：
   - 彻底摆脱传统 BM25 经验参数 `k1`、`b`，无需任何人工衰减系数。词语的语法作用、判别力与语义扩展全部由离线神经网络在海量检索任务中学得。
2. **语系隔离消除语义稀释**：
   - 英文是屈折语，中文是孤立语。分语言扩散保证了词元在自身语言体系内的上下文自洽性，多语言 Token 映射层则负责跨语言搭桥。
3. **微秒级极致性能**：
   - 充分利用倒排索引天然的稀疏剪枝优势，检索时仅扫描有交集的极少数文档项；
   - 中文查询 p50 延迟 **26 μs (0.026 ms)**，英文查询 p50 延迟 **28 μs (0.028 ms)**，建索引只需数毫秒，内存仅占用几兆字节，完全消除运行时对 GPU/Python 的依赖。

