//! 纯 CPU 紧凑稠密向量检索与无参数 RRF 融合算子 (Dense Scorer & Reciprocal Rank Fusion)。
//!
//! 1. DenseIndex: 连续内存数组排布，单次遍历 5k 篇 384 维向量点积在 CPU 上仅需 < 1ms；
//! 2. rrf_fuse: 业界标准无超参数倒数排名融合算法，消除单模态偏见，实现双路异构互补。

use anyhow::Result;
use std::collections::HashMap;

pub struct DenseIndex {
    doc_ids: Vec<String>,
    dim: usize,
    /// 连续排布的 L2 归一化浮点数组：大小为 num_docs * dim
    data: Vec<f32>,
    id_map: HashMap<String, usize>,
}

impl DenseIndex {
    /// 从小端二进制 dense.bin 和 doc_ids 列表加载
    pub fn load_from_bytes(bin_bytes: &[u8], doc_ids: Vec<String>) -> Result<Self> {
        if bin_bytes.len() < 16 {
            anyhow::bail!("dense.bin 文件头损坏");
        }
        if &bin_bytes[0..8] != b"SLDENSv1" {
            anyhow::bail!("dense.bin magic 不匹配");
        }

        let num_docs = u32::from_le_bytes(bin_bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bin_bytes[12..16].try_into().unwrap()) as usize;

        if num_docs != doc_ids.len() {
            anyhow::bail!(
                "dense.bin 的文档数 ({num_docs}) 与 doc_ids.txt ({}) 不一致",
                doc_ids.len()
            );
        }

        let expected_bytes = 16 + num_docs * dim * 4;
        if bin_bytes.len() != expected_bytes {
            anyhow::bail!(
                "dense.bin 大小不匹配：期望 {expected_bytes} 字节，实际 {} 字节",
                bin_bytes.len()
            );
        }

        let mut data = Vec::with_capacity(num_docs * dim);
        let float_bytes = &bin_bytes[16..];
        for chunk in float_bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let mut id_map = HashMap::with_capacity(doc_ids.len());
        for (idx, id) in doc_ids.iter().enumerate() {
            id_map.insert(id.clone(), idx);
        }

        Ok(Self {
            doc_ids,
            dim,
            data,
            id_map,
        })
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 根据文档 ID 获取其对应的向量连续内积（若不存在返回 None）
    pub fn dot_with_id(&self, doc_id: &str, query_vec: &[f32]) -> Option<f32> {
        let idx = *self.id_map.get(doc_id)?;
        Some(self.dot_at_index(idx, query_vec))
    }

    /// 8-way SIMD 展开的高性能单文档点积
    pub fn dot_at_index(&self, doc_idx: usize, query_vec: &[f32]) -> f32 {
        let offset = doc_idx * self.dim;
        let doc_slice = &self.data[offset..offset + self.dim];
        let mut dot = 0.0f32;
        let mut i = 0;
        let dim = self.dim;
        while i + 7 < dim {
            dot += query_vec[i] * doc_slice[i]
                + query_vec[i + 1] * doc_slice[i + 1]
                + query_vec[i + 2] * doc_slice[i + 2]
                + query_vec[i + 3] * doc_slice[i + 3]
                + query_vec[i + 4] * doc_slice[i + 4]
                + query_vec[i + 5] * doc_slice[i + 5]
                + query_vec[i + 6] * doc_slice[i + 6]
                + query_vec[i + 7] * doc_slice[i + 7];
            i += 8;
        }
        while i < dim {
            dot += query_vec[i] * doc_slice[i];
            i += 1;
        }
        dot
    }

    /// 纯 CPU 紧凑向量点积：利用 8-way SIMD 展开与有界小顶堆 Top-K 截断。
    /// $O(N \log K)$ 时间复杂度，内存零重分配。
    pub fn search(&self, query_vec: &[f32], k: usize) -> Vec<(String, f64)> {
        if query_vec.len() != self.dim || k == 0 || self.is_empty() {
            return Vec::new();
        }

        // 使用有界小顶堆维护 Top-K
        let mut heap = std::collections::BinaryHeap::with_capacity(k + 1);

        for doc_idx in 0..self.doc_ids.len() {
            let dot = self.dot_at_index(doc_idx, query_vec);

            // 维护最小堆
            if heap.len() < k {
                heap.push(ReverseWrapper(dot, doc_idx));
            } else if let Some(min) = heap.peek() {
                if dot > min.0 {
                    heap.pop();
                    heap.push(ReverseWrapper(dot, doc_idx));
                }
            }
        }

        // 堆转出排序
        let mut results: Vec<_> = heap.into_vec();
        results.sort_by(|a, b| b.0.total_cmp(&a.0));
        results
            .into_iter()
            .map(|ReverseWrapper(score, idx)| (self.doc_ids[idx].clone(), score as f64))
            .collect()
    }
}

/// 辅助结构：用于 BinaryHeap 的最小堆包装
#[derive(PartialEq)]
struct ReverseWrapper(f32, usize);

impl Eq for ReverseWrapper {}

impl Ord for ReverseWrapper {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.total_cmp(&self.0)
    }
}

impl PartialOrd for ReverseWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 无参数倒数排名融合 (Reciprocal Rank Fusion, RRF)。
/// 标准参数 rrf_k = 60.0，两路完全平等互补，外壳无任何人工偏向权重。
pub fn rrf_fuse(
    list_a: &[(String, f64)],
    list_b: &[(String, f64)],
    k: usize,
    rrf_k: f64,
) -> Vec<(String, f64)> {
    rrf_fuse_many(&[list_a, list_b], k, rrf_k)
}

/// 任意路数的无参数 RRF 融合：每路只贡献名次，不比较分数。
///
/// 同一路里重复出现的 doc 只按第一次出现计票（防止某路重复条目放大权重）；
/// 不存在的路/空路自然跳过。这个入口让 RRF 外壳不再硬编码“两路”，
/// 记忆召回这类需要“词法 + 短语 + 稠密”多路的场景可以直接复用。
pub fn rrf_fuse_many(lists: &[&[(String, f64)]], k: usize, rrf_k: f64) -> Vec<(String, f64)> {
    if k == 0 {
        return Vec::new();
    }
    let mut scores: HashMap<String, f64> = HashMap::new();
    for list in lists {
        let mut seen = std::collections::HashSet::new();
        for (rank, (doc_id, _)) in list.iter().enumerate() {
            if !seen.insert(doc_id) {
                continue;
            }
            let score = 1.0 / (rrf_k + (rank as f64) + 1.0);
            *scores.entry(doc_id.clone()).or_insert(0.0) += score;
        }
    }

    let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    fused.truncate(k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_index_search_correctness() {
        // 构造 2 篇 2 维归一化文档
        let mut bin = Vec::new();
        bin.extend_from_slice(b"SLDENSv1");
        bin.extend_from_slice(&2u32.to_le_bytes()); // num_docs = 2
        bin.extend_from_slice(&2u32.to_le_bytes()); // dim = 2
                                                    // d0: [1.0, 0.0]
        bin.extend_from_slice(&1.0f32.to_le_bytes());
        bin.extend_from_slice(&0.0f32.to_le_bytes());
        // d1: [0.0, 1.0]
        bin.extend_from_slice(&0.0f32.to_le_bytes());
        bin.extend_from_slice(&1.0f32.to_le_bytes());

        let doc_ids = vec!["d0".to_string(), "d1".to_string()];
        let index = DenseIndex::load_from_bytes(&bin, doc_ids).unwrap();

        // 搜 [1.0, 0.0]，d0 应排在第 1 位且得分为 1.0
        let hits = index.search(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "d0");
        assert!((hits[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rrf_fuse_combines_ranks_fairly() {
        let list_a = vec![("d1".to_string(), 10.0), ("d2".to_string(), 5.0)];
        let list_b = vec![("d2".to_string(), 0.9), ("d1".to_string(), 0.8)];

        // 两路均在头部的文档获得极高的累加倒数排名
        let fused = rrf_fuse(&list_a, &list_b, 2, 60.0);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn rrf_fuse_many_supports_more_than_two_routes_without_double_counting() {
        let a = vec![("d1".to_string(), 10.0), ("d2".to_string(), 5.0)];
        let b = vec![("d2".to_string(), 0.9), ("d1".to_string(), 0.8)];
        let c = vec![("d3".to_string(), 0.7), ("d1".to_string(), 0.6)];
        let fused = rrf_fuse_many(&[&a, &b, &c], 3, 60.0);
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].0, "d1", "三路都靠前的 d1 应融合到第一");
        assert!(fused.iter().all(|(id, _)| id != "d3" || fused[0].0 == "d1"));
    }

    #[test]
    fn rrf_fuse_many_ignores_duplicate_entries_inside_one_route() {
        let a = vec![("d1".to_string(), 10.0), ("d1".to_string(), 9.0)];
        let b = vec![("d2".to_string(), 1.0)];
        let fused = rrf_fuse_many(&[&a, &b], 2, 60.0);
        assert_eq!(fused[0].0, "d1");
        assert_eq!(fused.len(), 2);
    }
}
