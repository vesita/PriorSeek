//! 数据契约：数据集产出 [`Dataset`]，适配器只接受 [`Doc`] / [`Query`]。
//!
//! 这里的类型一旦改动，所有数据集构建脚本与所有适配器都要同步 —— 它就是「对比标准」的
//! 结构层。语义层（怎么评、怎么比）见仓库根 `docs/STANDARD.md`。

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// 语料里的一篇文档。`title` 可以为空（纯文本语料）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Doc {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    pub text: String,
}

impl Doc {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: String::new(),
            text: text.into(),
        }
    }

    /// 参与检索的全文。`title` 非空时拼在前面 —— 与 BEIR 的通行做法一致。
    pub fn full_text(&self) -> String {
        if self.title.is_empty() {
            self.text.clone()
        } else {
            format!("{}\n{}", self.title, self.text)
        }
    }
}

/// 一条查询。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
}

/// 相关性判定：`qid -> docid -> rel`。`rel = 0` 表示判定为不相关（可以出现，也可以省略）。
pub type Qrels = BTreeMap<String, BTreeMap<String, u32>>;

/// 各文件的内容摘要，由数据集构建器写入，用来判「这份规范数据还是不是原来那份」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub sha256: String,
    pub bytes: u64,
}

/// 数据集规模。`scored_queries` = qrels 里至少有一个正相关文档的查询数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub docs: usize,
    pub queries: usize,
    pub qrels: usize,
    pub scored_queries: usize,
}

/// 数据集清单：来源、构建参数、规模、文件摘要。
///
/// **跨机器 / 跨时间的可比性全靠它**：两个 run 的 `manifest_sha256` 不同就不是同一道题。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    /// 原始数据来源（URL、仓库、论文），一行一个。
    pub source: Vec<String>,
    /// 谁按什么参数构建的（人类可读，含子集口径时写死 pool 大小与 seed）。
    pub built_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub notes: String,
    pub counts: Counts,
    pub files: BTreeMap<String, FileDigest>,
    /// 可选训练划分的文件摘要（`train_queries.jsonl` / `train_qrels.tsv`）。
    ///
    /// **不参与 [`Manifest::fingerprint`]**：训练数据换一版、测试题没变，两次运行在 test 上仍然可比；
    /// 但它必须在训练工具里被校验，否则「训练用了哪份数据」不可审计。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub train_files: BTreeMap<String, FileDigest>,
}

impl Manifest {
    /// 清单整体摘要：对「参与比对的四个文件摘要」做 SHA-256。
    /// 不含 `built_by` / `notes`，因为它们不影响题目本身。
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update([0]);
        hasher.update(self.version.as_bytes());
        hasher.update([0]);
        for (file, digest) in &self.files {
            hasher.update(file.as_bytes());
            hasher.update([0]);
            hasher.update(digest.sha256.as_bytes());
            hasher.update([0]);
        }
        hasher.update(self.counts.docs.to_le_bytes());
        hasher.update(self.counts.queries.to_le_bytes());
        hasher.update(self.counts.qrels.to_le_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// 一份可以直接拿来跑的规范数据集。
#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    pub qrels: Qrels,
    pub manifest: Manifest,
}

impl Dataset {
    pub fn doc_ids(&self) -> HashSet<&str> {
        self.docs.iter().map(|d| d.id.as_str()).collect()
    }

    /// 该查询的正相关文档（`rel > 0`），按 `rel` 降序、doc id 升序给稳定顺序。
    pub fn relevant(&self, qid: &str) -> Vec<(&str, u32)> {
        let Some(map) = self.qrels.get(qid) else {
            return Vec::new();
        };
        let mut rows: Vec<(&str, u32)> = map
            .iter()
            .filter(|(_, rel)| **rel > 0)
            .map(|(doc, rel)| (doc.as_str(), *rel))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        rows
    }

    /// 参与计分的查询：qrels 里至少有一条正相关判定。
    pub fn scored_query_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .qrels
            .iter()
            .filter(|(_, map)| map.values().any(|rel| *rel > 0))
            .map(|(qid, _)| qid.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// 契约自检：qrels 指向的文档必须都在语料里，查询必须都在查询表里。
    /// 返回 (被丢弃的 qrels 条数, 被丢弃的查询条数, 详情)。被丢弃的条目会在加载时剔除。
    pub fn prune_inconsistent(&mut self) -> PruneReport {
        let doc_ids: HashSet<String> = self.docs.iter().map(|d| d.id.clone()).collect();
        let query_ids: BTreeSet<String> = self.queries.iter().map(|q| q.id.clone()).collect();

        let mut missing_docs: Vec<(String, String)> = Vec::new();
        let mut missing_queries: Vec<String> = Vec::new();

        let qrels = std::mem::take(&mut self.qrels);
        let mut kept: Qrels = BTreeMap::new();
        for (qid, map) in qrels {
            if !query_ids.contains(&qid) {
                missing_queries.push(qid);
                continue;
            }
            let mut kept_map = BTreeMap::new();
            for (docid, rel) in map {
                if doc_ids.contains(&docid) {
                    kept_map.insert(docid, rel);
                } else {
                    missing_docs.push((qid.clone(), docid));
                }
            }
            if !kept_map.is_empty() {
                kept.insert(qid, kept_map);
            }
        }
        self.qrels = kept;

        let scored = self.scored_query_ids().len();
        let mut report = PruneReport {
            dropped_qrels: missing_docs.len(),
            dropped_queries: missing_queries.len(),
            missing_docs,
            missing_queries,
            ..Default::default()
        };
        report.scored_queries = scored;
        report
    }

    /// 从 manifest 与当前内容重算规模，返回 (与 manifest 一致的 Counts, 是否一致)。
    pub fn recount(&self) -> (Counts, bool) {
        let qrels: usize = self.qrels.values().map(|m| m.len()).sum();
        let counts = Counts {
            docs: self.docs.len(),
            queries: self.queries.len(),
            qrels,
            scored_queries: self.scored_query_ids().len(),
        };
        let same = counts == self.manifest.counts;
        (counts, same)
    }
}

/// [`Dataset::prune_inconsistent`] 的结果。
#[derive(Debug, Clone, Default)]
pub struct PruneReport {
    pub dropped_qrels: usize,
    pub dropped_queries: usize,
    pub scored_queries: usize,
    pub missing_docs: Vec<(String, String)>,
    pub missing_queries: Vec<String>,
}

impl PruneReport {
    pub fn is_clean(&self) -> bool {
        self.dropped_qrels == 0 && self.dropped_queries == 0
    }
}

/// 先验因子的种类。封闭集合：新增一种要同时改这里、工件构建器和消费它的外壳。
///
/// 每种都对应 BM25 原理上做不到的一件事，见 `docs/KERNEL.md` §0 与 `docs/STANDARD.md` §10。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorKind {
    /// `token → (关联词, 置信度)`：解决同义/近义但词形不同的词汇不匹配。
    TokenConfidenceList,
    /// `token → 概念 id`：把同义、上下位、跨语言变体归并到同一检索单元。
    Concept,
    /// `token → 出现字段（title / abstract）`：标题命中与正文命中不再同价。
    Field,
}

/// 因子的数值类型。目前只允许 `f32`：外壳按单精度消费，声明成别的会让跨实现比对失真。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactorDtype {
    F32,
}

/// 因子形状的一个维度：要么是命名轴（`n_query_tokens`），要么是固定尺寸（`3`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShapeDim {
    Size(u64),
    Named(String),
}

/// 先验因子规格：内核网络的**输出**，外壳检索时消费。
///
/// 只表达事实/证据，不表达用法。`deny_unknown_fields` 是这条约束的强制点：
/// 任何 `alpha` / `weight` / `top_k` 这类描述「怎么用」的字段都会在解析时被拒绝。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorFactor {
    /// 因子名，进 `run.json` 供人辨认。
    pub name: String,
    pub kind: FactorKind,
    pub shape: Vec<ShapeDim>,
    pub dtype: FactorDtype,
    /// 因子条目数（如 scifact 上 35,758 条 token→关联词对）。
    pub count: u64,
}

impl PriorFactor {
    /// 构造点校验：非法因子在这里就被拒绝，不留给外壳去兜底。
    pub fn verify(&self) -> anyhow::Result<()> {
        if self.name.trim().is_empty() {
            anyhow::bail!("先验因子缺 name");
        }
        if self.count == 0 {
            anyhow::bail!("先验因子 `{}` 的 count = 0，空因子不应登记", self.name);
        }
        if self.shape.is_empty() {
            anyhow::bail!("先验因子 `{}` 的 shape 为空", self.name);
        }
        Ok(())
    }
}

/// 内核工件：深度学习内核的离线产物（embedding / graph / concept / token weights）。
///
/// 准入规则见 `docs/STANDARD.md` §10：运行时必须是 CPU；工件必须带哈希与训练来源，
/// 缺失或校验失败时适配器构造即失败，不允许「跑出结果再解释」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelArtifact {
    /// 工件 id（如 `seq-v0`）。
    pub id: String,
    /// 工件类型：`seq_graph` / `embedding` / `concept` / `token_weights` / `vocab`。
    pub kind: String,
    /// 权重来源：`corpus_ppmi` / `gpu_learned` / `pretrained:<model-id>` 等。
    pub weights_origin: String,
    /// 工件目录（相对仓库根或绝对路径），用于复核。
    pub dir: String,
    /// `manifest.json` 自身内容的 sha256。
    pub manifest_sha256: String,
    /// 工件内容文件 → sha256（来自 manifest 的 `files`）。
    pub files: BTreeMap<String, String>,
    /// 训练/构建来源（数据集指纹 + split，或「语料内统计」）。
    pub trained_on: String,
    /// 预训练污染声明（是否可能见过评测语料）。
    pub contamination: String,
    /// 运行时设备要求，必须包含 `cpu` 才准入。
    pub runtime_device: String,
    /// 该工件产出的先验因子规格。`None` = 该工件不提供因子（如纯 vocab 工件）。
    #[serde(default)]
    pub prior_factor: Option<PriorFactor>,
}

impl KernelArtifact {
    /// 从工件目录读取 `manifest.json`，校验必备字段并算出哈希。
    ///
    /// 任何一项不合格都直接报错——这是「内核工件可审计」的强制点。
    pub fn from_dir(dir: &std::path::Path) -> anyhow::Result<Self> {
        use sha2::{Digest, Sha256};
        let manifest_path = dir.join("manifest.json");
        let bytes = std::fs::read(&manifest_path).map_err(|err| {
            anyhow::anyhow!(
                "读取工件 {} 失败：{err}（先跑 tools/kernel/build_seq.py）",
                manifest_path.display()
            )
        })?;
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|err| anyhow::anyhow!("{} 不是合法 JSON：{err}", manifest_path.display()))?;

        let get_str = |key: &str| -> anyhow::Result<String> {
            manifest
                .get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    anyhow::anyhow!("工件 manifest 缺字段 `{key}`：{}", manifest_path.display())
                })
        };

        let runtime_device = manifest
            .get("runtime")
            .and_then(|runtime| runtime.get("device"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !runtime_device.contains("cpu") {
            anyhow::bail!(
                "工件 {} 的 runtime.device = {:?}，不符合「运行时必须 CPU」的准入条件（STANDARD §10）",
                dir.display(),
                runtime_device
            );
        }

        let mut files = BTreeMap::new();
        if let Some(map) = manifest.get("files").and_then(|v| v.as_object()) {
            for (name, digest) in map {
                let expected = digest
                    .get("sha256")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("工件 manifest 的 files.{name} 缺 sha256"))?;
                let path = dir.join(name);
                let content = std::fs::read(&path).map_err(|err| {
                    anyhow::anyhow!("读取工件文件 {} 失败：{err}", path.display())
                })?;
                let mut hasher = Sha256::new();
                hasher.update(&content);
                let actual = format!("{:x}", hasher.finalize());
                if actual != expected {
                    anyhow::bail!(
                        "工件文件 {} 的 sha256 与 manifest 不符（期望 {}，实际 {}）——工件被改过",
                        path.display(),
                        expected,
                        actual
                    );
                }
                files.insert(name.clone(), actual);
            }
        }
        if files.is_empty() {
            anyhow::bail!(
                "工件 {} 的 manifest 没有 files 摘要，无法审计",
                dir.display()
            );
        }

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        // 因子规格是可选登记项：字段缺失或显式 null 都表示「该工件不提供因子」。
        // 一旦写了，就必须完整符合契约——非法因子在构造工件时就失败，不留到检索期。
        let prior_factor = match manifest.get("prior_factor") {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                let factor: PriorFactor = serde_json::from_value(value.clone()).map_err(|err| {
                    anyhow::anyhow!(
                        "工件 {} 的 prior_factor 不符合因子契约：{err}。\
                         因子只表达证据，不允许出现 alpha / weight / top_k 这类用法字段",
                        dir.display()
                    )
                })?;
                factor.verify()?;
                Some(factor)
            }
        };
        Ok(Self {
            id: get_str("id")?,
            kind: get_str("kind")?,
            weights_origin: get_str("weights_origin")?,
            dir: dir.display().to_string(),
            manifest_sha256: format!("{:x}", hasher.finalize()),
            files,
            trained_on: get_str("trained_on")?,
            contamination: get_str("contamination")?,
            runtime_device,
            prior_factor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个最小可审计工件目录：一份内容文件 + 带摘要的 manifest。
    /// `extra` 里的键会覆盖 / 追加到 manifest 顶层。
    fn artifact_dir(extra: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"payload";
        std::fs::write(dir.path().join("data.bin"), payload).unwrap();
        let digest = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(payload);
            format!("{:x}", hasher.finalize())
        };
        let mut manifest = serde_json::json!({
            "id": "test-v0",
            "kind": "seq_graph",
            "weights_origin": "corpus_ppmi",
            "trained_on": "语料内统计",
            "contamination": "无",
            "runtime": {"device": "cpu"},
            "files": {"data.bin": {"sha256": digest}},
        });
        if let (Some(base), Some(extra)) = (manifest.as_object_mut(), extra.as_object()) {
            for (key, value) in extra {
                base.insert(key.clone(), value.clone());
            }
        }
        std::fs::write(
            dir.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        dir
    }

    fn parse_factor(value: serde_json::Value) -> anyhow::Result<PriorFactor> {
        let factor: PriorFactor = serde_json::from_value(value)?;
        factor.verify()?;
        Ok(factor)
    }

    #[test]
    fn all_registered_kinds_parse() {
        for (kind_json, expected) in [
            ("token_confidence_list", FactorKind::TokenConfidenceList),
            ("concept", FactorKind::Concept),
            ("field", FactorKind::Field),
        ] {
            let factor = parse_factor(serde_json::json!({
                "name": "f",
                "kind": kind_json,
                "shape": ["n_query_tokens", 3],
                "dtype": "f32",
                "count": 10,
            }))
            .unwrap();
            assert_eq!(factor.kind, expected);
            assert_eq!(factor.dtype, FactorDtype::F32);
            assert_eq!(factor.shape.len(), 2);
        }
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let error = parse_factor(serde_json::json!({
            "name": "f",
            "kind": "embedding_v2",
            "shape": ["n", 3],
            "dtype": "f32",
            "count": 10,
        }))
        .unwrap_err();
        assert!(
            error.to_string().contains("unknown variant"),
            "未知 kind 必须在解析期被拒绝，实际：{error}"
        );
    }

    #[test]
    fn usage_fields_are_rejected() {
        for usage_field in ["alpha", "weight", "top_k"] {
            let mut payload = serde_json::json!({
                "name": "f",
                "kind": "token_confidence_list",
                "shape": ["n_query_tokens", 3],
                "dtype": "f32",
                "count": 10,
            });
            payload[usage_field] = serde_json::json!(0.5);
            let error = parse_factor(payload).unwrap_err();
            assert!(
                error.to_string().contains("unknown field"),
                "因子里的用法字段 `{usage_field}` 必须在解析期被拒绝，实际：{error}"
            );
        }
    }

    #[test]
    fn zero_count_and_empty_shape_are_rejected() {
        let zero = parse_factor(serde_json::json!({
            "name": "f",
            "kind": "concept",
            "shape": ["n_query_tokens", 3],
            "dtype": "f32",
            "count": 0,
        }))
        .unwrap_err();
        assert!(zero.to_string().contains("count = 0"), "实际：{zero}");

        let empty_shape = parse_factor(serde_json::json!({
            "name": "f",
            "kind": "concept",
            "shape": [],
            "dtype": "f32",
            "count": 1,
        }))
        .unwrap_err();
        assert!(
            empty_shape.to_string().contains("shape 为空"),
            "实际：{empty_shape}"
        );
    }

    #[test]
    fn missing_factor_is_none_but_invalid_factor_fails_closed() {
        let absent = artifact_dir(serde_json::json!({}));
        assert_eq!(
            KernelArtifact::from_dir(absent.path())
                .unwrap()
                .prior_factor,
            None
        );

        let explicit_null = artifact_dir(serde_json::json!({"prior_factor": null}));
        assert_eq!(
            KernelArtifact::from_dir(explicit_null.path())
                .unwrap()
                .prior_factor,
            None
        );

        // 同一份 manifest，只把因子换成非法值，构造必须失败而不是静默忽略。
        let broken = artifact_dir(serde_json::json!({
            "prior_factor": {
                "name": "query_expansion",
                "kind": "token_confidence_list",
                "shape": ["n_query_tokens", 3],
                "dtype": "f32",
                "count": 10,
                "alpha": 0.3
            }
        }));
        let error = KernelArtifact::from_dir(broken.path()).unwrap_err();
        assert!(
            error.to_string().contains("不符合因子契约"),
            "非法因子必须在工件构造期失败，实际：{error}"
        );
    }

    #[test]
    fn valid_factor_is_recorded_on_artifact() {
        let dir = artifact_dir(serde_json::json!({
            "prior_factor": {
                "name": "query_expansion",
                "kind": "token_confidence_list",
                "shape": ["n_query_tokens", 3],
                "dtype": "f32",
                "count": 35758
            }
        }));
        let artifact = KernelArtifact::from_dir(dir.path()).unwrap();
        let factor = artifact.prior_factor.expect("合法因子应被登记");
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
        assert_eq!(factor.count, 35758);
        assert_eq!(
            factor.shape[0],
            ShapeDim::Named("n_query_tokens".to_string())
        );
        assert_eq!(factor.shape[1], ShapeDim::Size(3));
    }

    /// 真实工件的契约对拍：需要仓库内 `data/scifact/artifacts/exp-ppmi-v0` 就绪。
    /// 只验证 `PriorFactor` 契约本身；该工件缺 `weights_origin`，
    /// 不满足 `KernelArtifact` 的完整准入条件，因此这里直接读它的 manifest。
    /// 手动跑：`cargo test -p sl-core real_expansion_artifact -- --ignored`
    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_expansion_artifact_factor_matches_contract() {
        let path = std::path::Path::new("../../data/scifact/artifacts/exp-ppmi-v0/manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("真实工件缺失")).unwrap();
        let factor: PriorFactor =
            serde_json::from_value(manifest["prior_factor"].clone()).expect("真实因子必须符合契约");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
        assert_eq!(factor.dtype, FactorDtype::F32);
        assert_eq!(factor.count, 35758);
        assert_eq!(
            factor.shape,
            vec![
                ShapeDim::Named("n_query_tokens".to_string()),
                ShapeDim::Size(3)
            ]
        );
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_gated_expansion_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/gated-exp-v1");
        let artifact = KernelArtifact::from_dir(dir).expect("gated-exp-v1 必须通过完整准入校验");
        assert_eq!(artifact.id, "gated-exp-v1");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_saliency_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/saliency-v0");
        let artifact = KernelArtifact::from_dir(dir).expect("saliency-v0 必须通过完整准入校验");
        assert_eq!(artifact.id, "saliency-v0");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_sparse_routing_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/sparse-route-v1");
        let artifact = KernelArtifact::from_dir(dir).expect("sparse-route-v1 必须通过完整准入校验");
        assert_eq!(artifact.id, "sparse-route-v1");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_flow_cost_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/flow-cost-v0");
        let artifact = KernelArtifact::from_dir(dir).expect("flow-cost-v0 必须通过完整准入校验");
        assert_eq!(artifact.id, "flow-cost-v0");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_logic_graph_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/logic-graph-v0");
        let artifact = KernelArtifact::from_dir(dir).expect("logic-graph-v0 必须通过完整准入校验");
        assert_eq!(artifact.id, "logic-graph-v0");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_synonym_concept_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/concept-synonym-v1");
        let artifact =
            KernelArtifact::from_dir(dir).expect("concept-synonym-v1 必须通过完整准入校验");
        assert_eq!(artifact.id, "concept-synonym-v1");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::Concept);
    }

    #[test]
    #[ignore = "依赖 data/ 下的真实工件，手动验证用"]
    fn real_dense_artifact_verifies() {
        let dir = std::path::Path::new("../../data/scifact/artifacts/dense-v0");
        let artifact = KernelArtifact::from_dir(dir).expect("dense-v0 必须通过完整准入校验");
        assert_eq!(artifact.id, "dense-v0");
        assert_eq!(artifact.runtime_device, "cpu");
        let factor = artifact.prior_factor.expect("必须包含 prior_factor");
        factor.verify().unwrap();
        assert_eq!(factor.kind, FactorKind::TokenConfidenceList);
    }
}
