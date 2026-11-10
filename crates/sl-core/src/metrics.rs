//! 指标口径 —— 「同一个分数可比」的前提。
//!
//! 口径（全部写死在这里，也写死在 `docs/STANDARD.md`）：
//!
//! - **只对「至少有一个正相关文档」的查询计分**；其余查询进 `queries_skipped`，不拉平均。
//! - 所有指标都是**宏平均**：每个查询等权，先算单查询再平均。
//! - 相关性是分级的：`rel` 取 qrels 里的整数；`rel > 0` 才算相关。
//! - `Recall@k = |相关 ∩ top-k| / |相关|`（分母是全部相关文档数，不受 k 截断）。
//! - `MRR@k` 取第一个正相关文档的 1/rank（rank 从 1 起），rank > k 记 0。
//! - `nDCG@k` 增益 `2^rel - 1`，折损 `log2(i+2)`；IDCG 取该查询全部相关文档按 rel 降序的前 k 个。
//! - `MAP@k`（平均精度）的 AP 分母是**全部相关文档数**：`AP@k = (1/|R|) Σ_{i≤k, rel_i>0} P@i`。
//! - 排序里的重复 doc id 先按「保留首次出现」去重；不在 qrels 里的 doc 视为不相关。

use crate::types::Qrels;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// 对比表默认报的 k 值。`k` 超过本轮实际召回深度时会被截断到实际深度。
pub const DEFAULT_RECALL_KS: [usize; 5] = [1, 3, 10, 50, 100];
pub const DEFAULT_MRR_AT: usize = 10;
pub const DEFAULT_NDCG_AT: usize = 10;
pub const DEFAULT_MAP_AT: usize = 100;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// recall@k，键是 k。
    pub recall: BTreeMap<usize, f64>,
    /// 查询命中率 `hit@k`：该查询的 top-k 里**至少有一条**相关文档的查询占比。
    /// 缺失率 `miss@k = 1 − hit@k` —— 这是「用户搜一次能不能找到东西」的业务口径。
    pub hit: BTreeMap<usize, f64>,
    pub mrr_at: usize,
    pub mrr: f64,
    pub ndcg_at: usize,
    pub ndcg: f64,
    pub map_at: usize,
    pub map: f64,
    /// 参与计分的查询数（qrels 里至少有一个正相关文档）。
    pub queries_scored: usize,
    /// 被跳过的查询数（没有正相关文档，或 qrels 里根本没有它）。
    pub queries_skipped: usize,
}

/// 单查询明细，写进运行记录，便于复盘「差在哪条查询上」。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQuery {
    pub qid: String,
    /// 第一个正相关文档的排名（从 1 起）；没命中为 None。
    pub first_rel_rank: Option<usize>,
    pub ndcg: f64,
    pub ap: f64,
    pub recall: BTreeMap<usize, f64>,
    /// 前 `max_k` 名里命中的相关文档数。
    pub hits: usize,
}

/// 对一轮检索结果算全套指标。
///
/// `rankings` 是 `qid -> 有序 doc id 列表`；`k` 是本轮实际请求的召回深度。
/// 返回 (汇总指标, 单查询明细)，明细按 qid 升序。
pub fn evaluate_rankings(
    rankings: &BTreeMap<String, Vec<String>>,
    qrels: &Qrels,
    k: usize,
) -> (Metrics, Vec<PerQuery>) {
    let recall_ks: Vec<usize> = DEFAULT_RECALL_KS.iter().copied().filter(|rk| *rk <= k).collect();
    let mrr_at = DEFAULT_MRR_AT.min(k.max(1));
    let ndcg_at = DEFAULT_NDCG_AT.min(k.max(1));
    let map_at = DEFAULT_MAP_AT.min(k.max(1));

    let mut per_query = Vec::new();
    let mut skipped = 0usize;

    let mut recall_sums: BTreeMap<usize, f64> = recall_ks.iter().map(|rk| (*rk, 0.0)).collect();
    let mut hit_sums: BTreeMap<usize, f64> = recall_ks.iter().map(|rk| (*rk, 0.0)).collect();
    let mut mrr_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut ap_sum = 0.0;

    for (qid, rel_map) in qrels {
        let relevant: BTreeMap<&str, u32> =
            rel_map.iter().filter(|(_, rel)| **rel > 0).map(|(d, r)| (d.as_str(), *r)).collect();
        if relevant.is_empty() {
            // qrels 判定这条查询没有正相关文档：它无法参与效果指标，只计数、不拉平均。
            skipped += 1;
            continue;
        }
        // qrels 里有、但本轮没有结果的查询按「空排序」计 0 分，不能静默丢出分母。
        let empty: Vec<String> = Vec::new();
        let ranking = rankings.get(qid).unwrap_or(&empty);

        let deduped = dedupe(ranking);
        let top = &deduped[..deduped.len().min(k)];

        let mut hits = 0usize;
        let mut first_rel_rank = None;
        for (idx, doc) in top.iter().enumerate() {
            if relevant.contains_key(doc.as_str()) {
                hits += 1;
                if first_rel_rank.is_none() {
                    first_rel_rank = Some(idx + 1);
                }
            }
        }

        for rk in &recall_ks {
            let seen = top.iter().take(*rk).filter(|d| relevant.contains_key(d.as_str())).count();
            let value = seen as f64 / relevant.len() as f64;
            *recall_sums.get_mut(rk).expect("recall k 已初始化") += value;
            // 命中率只看「有没有至少一条」，与 recall 的分母无关。
            *hit_sums.get_mut(rk).expect("hit k 已初始化") += if seen > 0 { 1.0 } else { 0.0 };
        }

        let mrr = match first_rel_rank {
            Some(rank) if rank <= mrr_at => 1.0 / rank as f64,
            _ => 0.0,
        };
        mrr_sum += mrr;

        let ndcg = ndcg_at_k(top, &relevant, ndcg_at);
        ndcg_sum += ndcg;

        let ap = average_precision_at_k(top, &relevant, map_at);
        ap_sum += ap;

        let query_recall: BTreeMap<usize, f64> = recall_ks
            .iter()
            .map(|rk| {
                let seen = top.iter().take(*rk).filter(|d| relevant.contains_key(d.as_str())).count();
                (*rk, seen as f64 / relevant.len() as f64)
            })
            .collect();

        per_query.push(PerQuery {
            qid: qid.clone(),
            first_rel_rank,
            ndcg,
            ap,
            recall: query_recall,
            hits,
        });
    }

    let scored = per_query.len();

    let mut metrics = Metrics {
        recall: BTreeMap::new(),
        hit: BTreeMap::new(),
        mrr_at,
        mrr: if scored == 0 { 0.0 } else { mrr_sum / scored as f64 },
        ndcg_at,
        ndcg: if scored == 0 { 0.0 } else { ndcg_sum / scored as f64 },
        map_at,
        map: if scored == 0 { 0.0 } else { ap_sum / scored as f64 },
        queries_scored: scored,
        queries_skipped: skipped,
    };
    for (rk, sum) in recall_sums {
        metrics.recall.insert(rk, if scored == 0 { 0.0 } else { sum / scored as f64 });
    }
    for (rk, sum) in hit_sums {
        metrics.hit.insert(rk, if scored == 0 { 0.0 } else { sum / scored as f64 });
    }
    per_query.sort_by(|a, b| a.qid.cmp(&b.qid));
    (metrics, per_query)
}

fn dedupe(ranking: &[String]) -> Vec<String> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::with_capacity(ranking.len());
    for doc in ranking {
        if seen.insert(doc.as_str()) {
            out.push(doc.clone());
        }
    }
    out
}

fn ndcg_at_k(top: &[String], relevant: &BTreeMap<&str, u32>, k: usize) -> f64 {
    let dcg: f64 = top
        .iter()
        .take(k)
        .enumerate()
        .map(|(idx, doc)| {
            let rel = relevant.get(doc.as_str()).copied().unwrap_or(0);
            gain(rel) / ((idx + 2) as f64).log2()
        })
        .sum();
    let mut ideal: Vec<u32> = relevant.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(idx, rel)| gain(*rel) / ((idx + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn gain(rel: u32) -> f64 {
    (2f64).powi(rel as i32) - 1.0
}

fn average_precision_at_k(top: &[String], relevant: &BTreeMap<&str, u32>, k: usize) -> f64 {
    let mut hits = 0usize;
    let mut sum = 0.0;
    for (idx, doc) in top.iter().take(k).enumerate() {
        if relevant.contains_key(doc.as_str()) {
            hits += 1;
            sum += hits as f64 / (idx + 1) as f64;
        }
    }
    if relevant.is_empty() {
        0.0
    } else {
        sum / relevant.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qrels(rows: &[(&str, &str, u32)]) -> Qrels {
        let mut map: Qrels = BTreeMap::new();
        for (qid, doc, rel) in rows {
            map.entry((*qid).to_string()).or_default().insert((*doc).to_string(), *rel);
        }
        map
    }

    fn rankings(rows: &[(&str, &[&str])]) -> BTreeMap<String, Vec<String>> {
        rows.iter()
            .map(|(qid, docs)| ((*qid).to_string(), docs.iter().map(|d| (*d).to_string()).collect()))
            .collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn single_relevant_doc_at_rank_one() {
        let r = rankings(&[("q1", &["d1", "d2"])]);
        let (m, per) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1)]), 10);
        assert_eq!(m.queries_scored, 1);
        assert_eq!(m.recall[&1], 1.0);
        assert_eq!(m.recall[&3], 1.0);
        assert_eq!(m.mrr, 1.0);
        assert_eq!(m.ndcg, 1.0);
        assert_eq!(m.map, 1.0);
        assert_eq!(per[0].first_rel_rank, Some(1));
    }

    #[test]
    fn single_relevant_doc_at_rank_two() {
        let r = rankings(&[("q1", &["d2", "d1"])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1)]), 10);
        assert_eq!(m.recall[&1], 0.0);
        assert_eq!(m.recall[&3], 1.0);
        assert!(close(m.mrr, 0.5));
        // DCG = 1/log2(3)，IDCG = 1
        assert!(close(m.ndcg, 1.0 / 3f64.log2()));
        assert!(close(m.map, 0.5));
    }

    #[test]
    fn graded_relevance_uses_exponential_gain() {
        // rel: a=2, b=1；排序 [b, a]
        let r = rankings(&[("q1", &["b", "a"])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "a", 2), ("q1", "b", 1)]), 10);
        let dcg = 1.0 / 2f64.log2() + 3.0 / 3f64.log2();
        let idcg = 3.0 / 2f64.log2() + 1.0 / 3f64.log2();
        assert!(close(m.ndcg, dcg / idcg), "ndcg={} expected={}", m.ndcg, dcg / idcg);
        assert!(close(m.recall[&1], 0.5));
        assert!(close(m.recall[&3], 1.0));
        assert!(close(m.map, 1.0)); // 两个都在前两位，AP=1
    }

    #[test]
    fn average_precision_matches_hand_computation() {
        // 排序 [d2, d1, d3]，相关 {d1, d3}
        let r = rankings(&[("q1", &["d2", "d1", "d3"])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1), ("q1", "d3", 1)]), 10);
        let ap = (1.0 / 2.0 + 2.0 / 3.0) / 2.0;
        assert!(close(m.map, ap), "map={} expected={}", m.map, ap);
    }

    #[test]
    fn k_truncation_limits_recall_but_not_denominator() {
        let r = rankings(&[("q1", &["x", "y", "d1"])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1)]), 3);
        assert_eq!(m.recall.keys().copied().collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(m.recall[&1], 0.0);
        assert_eq!(m.recall[&3], 1.0);
    }

    #[test]
    fn duplicate_doc_ids_are_deduped_keeping_first_occurrence() {
        let r = rankings(&[("q1", &["d1", "d1", "d2"])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1)]), 10);
        assert_eq!(m.recall[&1], 1.0);
        assert!(close(m.ndcg, 1.0));
    }

    #[test]
    fn queries_without_relevant_docs_are_skipped_not_averaged() {
        let r = rankings(&[("q1", &["d1"]), ("q2", &["d2"])]);
        let rel = qrels(&[("q1", "d1", 1), ("q2", "d2", 0)]);
        let (m, per) = evaluate_rankings(&r, &rel, 10);
        assert_eq!(m.queries_scored, 1);
        assert_eq!(m.queries_skipped, 1);
        assert_eq!(per.len(), 1);
        assert_eq!(m.recall[&1], 1.0);
    }

    #[test]
    fn empty_ranking_scores_zero() {
        let r = rankings(&[("q1", &[])]);
        let (m, _) = evaluate_rankings(&r, &qrels(&[("q1", "d1", 1)]), 10);
        assert_eq!(m.recall[&1], 0.0);
        assert_eq!(m.mrr, 0.0);
        assert_eq!(m.ndcg, 0.0);
        assert_eq!(m.map, 0.0);
        assert_eq!(m.queries_scored, 1);
    }

    #[test]
    fn hit_rate_counts_queries_not_documents() {
        // q1：两条相关文档都在前 3（recall@3 = 1.0, hit@1 = 1.0）
        // q2：唯一相关文档在第 2（recall@1 = 0.0, hit@1 = 0.0）
        let r = rankings(&[("q1", &["d1", "d2"]), ("q2", &["x", "d3"])]);
        let rel = qrels(&[("q1", "d1", 1), ("q1", "d2", 1), ("q2", "d3", 1)]);
        let (m, _) = evaluate_rankings(&r, &rel, 10);
        assert_eq!(m.hit[&1], 0.5); // q1 命中、q2 未命中
        assert_eq!(m.hit[&3], 1.0);
        // q1 的 recall@1 是 0.5（两条相关只命中一条），hit@1 却是 1.0 —— 两者口径不同
        assert_eq!(m.recall[&1], 0.25);
    }

    #[test]
    fn macro_average_weights_every_query_equally() {
        // q1 满分，q2 全空
        let r = rankings(&[("q1", &["d1"]), ("q2", &["x"])]);
        let rel = qrels(&[("q1", "d1", 1), ("q2", "d2", 1)]);
        let (m, _) = evaluate_rankings(&r, &rel, 10);
        assert_eq!(m.recall[&1], 0.5);
        assert_eq!(m.mrr, 0.5);
        assert_eq!(m.ndcg, 0.5);
        assert_eq!(m.map, 0.5);
    }
}
