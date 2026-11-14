//! 神经稀疏带权倒排索引 (SparseInvertedIndex)。
//!
//! 专为消费神经先验词权因子（如 SPLADE 等神经稀疏模型输出）设计：
//! 离线将每篇文档编码为稀疏词项与权重的元组列表，构建紧凑带权倒排表；
//! 在线检索执行纯 CPU 带权累加与有界小顶堆 Top-K 截断，
//! 零超参数（无 k1/b、无阻尼衰减、无比例参数），零运行时深度学习推理。

use anyhow::{bail, ensure, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// 神经稀疏带权倒排索引
#[derive(Clone, Debug)]
pub struct SparseInvertedIndex {
    pub vocab_size: usize,
    pub doc_ids: Vec<String>,
    /// postings[token_id] -> [(doc_idx, weight)]
    pub postings: Vec<Vec<(u32, f32)>>,
}

#[derive(PartialEq)]
struct ReverseScoredDoc(f32, usize);

impl Eq for ReverseScoredDoc {}

impl Ord for ReverseScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}

impl PartialOrd for ReverseScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SparseInvertedIndex {
    /// 从小端二进制字节流 `sparse.bin` 与文档 ID 列表构建带权倒排索引。
    pub fn from_bytes(bytes: &[u8], doc_ids: Vec<String>) -> Result<Self> {
        if bytes.len() < 16 {
            bail!("sparse.bin 头部长度不足 (至少 16 字节)");
        }

        let magic = &bytes[0..8];
        ensure!(
            magic == b"SLSPARv1",
            "无效的 sparse.bin 魔数: {:?}",
            String::from_utf8_lossy(magic)
        );

        let num_docs = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let vocab_size = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;

        ensure!(
            num_docs == doc_ids.len(),
            "sparse.bin 中的文档数 ({}) 与 doc_ids.txt ({}) 不一致",
            num_docs,
            doc_ids.len()
        );

        let mut postings: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vocab_size];
        let mut offset = 16;

        for doc_idx in 0..num_docs {
            if offset + 4 > bytes.len() {
                bail!("sparse.bin 在读取文档 {} 词项条目数时提前结束", doc_idx);
            }
            let num_entries =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            let entries_bytes_len = num_entries * 8; // u32 token_id + f32 weight
            if offset + entries_bytes_len > bytes.len() {
                bail!("sparse.bin 在读取文档 {} 的词项内容时提前结束", doc_idx);
            }

            for _ in 0..num_entries {
                let token_id =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
                let weight = f32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?);
                offset += 8;

                if token_id < vocab_size {
                    postings[token_id].push((doc_idx as u32, weight));
                }
            }
        }

        Ok(Self {
            vocab_size,
            doc_ids,
            postings,
        })
    }

    /// 执行神经稀疏内积打分并返回 Top-K (doc_id, score)。
    ///
    /// 打分公式为纯粹的带权内积：
    /// score(q, d) = sum_{t in q intersect d} w(t, q) * w(t, d)
    /// 零超参数，有界小顶堆 Top-K 截断。
    pub fn search(&self, query_terms: &[(u32, f32)], k: usize) -> Vec<(String, f64)> {
        if query_terms.is_empty() || k == 0 || self.doc_ids.is_empty() {
            return Vec::new();
        }

        let num_docs = self.doc_ids.len();
        let mut scores = vec![0.0f32; num_docs];

        // 稀疏倒排带权累加
        for &(token_id, q_weight) in query_terms {
            let tid = token_id as usize;
            if tid < self.vocab_size {
                for &(doc_idx, d_weight) in &self.postings[tid] {
                    scores[doc_idx as usize] += q_weight * d_weight;
                }
            }
        }

        // 有界小顶堆维护 Top-K
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for (doc_idx, &score) in scores.iter().enumerate() {
            if score <= 0.0 {
                continue;
            }
            if heap.len() < k {
                heap.push(ReverseScoredDoc(score, doc_idx));
            } else if let Some(min) = heap.peek() {
                if score > min.0 {
                    heap.pop();
                    heap.push(ReverseScoredDoc(score, doc_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_vec();
        results.sort_by(|a, b| b.0.total_cmp(&a.0));
        results
            .into_iter()
            .map(|ReverseScoredDoc(score, idx)| (self.doc_ids[idx].clone(), score as f64))
            .collect()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_index_roundtrip() {
        // 构造一个测试 sparse.bin
        let mut bin = Vec::new();
        bin.extend_from_slice(b"SLSPARv1");
        bin.extend_from_slice(&2u32.to_le_bytes()); // num_docs = 2
        bin.extend_from_slice(&10u32.to_le_bytes()); // vocab_size = 10

        // doc 0: token 1 w=1.5, token 3 w=2.0
        bin.extend_from_slice(&2u32.to_le_bytes());
        bin.extend_from_slice(&1u32.to_le_bytes());
        bin.extend_from_slice(&1.5f32.to_le_bytes());
        bin.extend_from_slice(&3u32.to_le_bytes());
        bin.extend_from_slice(&2.0f32.to_le_bytes());

        // doc 1: token 1 w=0.5, token 5 w=3.0
        bin.extend_from_slice(&2u32.to_le_bytes());
        bin.extend_from_slice(&1u32.to_le_bytes());
        bin.extend_from_slice(&0.5f32.to_le_bytes());
        bin.extend_from_slice(&5u32.to_le_bytes());
        bin.extend_from_slice(&3.0f32.to_le_bytes());

        let doc_ids = vec!["docA".to_string(), "docB".to_string()];
        let index = SparseInvertedIndex::from_bytes(&bin, doc_ids).unwrap();

        assert_eq!(index.num_docs(), 2);
        assert_eq!(index.vocab_size, 10);

        // query: token 1 w=1.0, token 3 w=1.0
        // docA: 1*1.5 + 1*2.0 = 3.5
        // docB: 1*0.5 + 0 = 0.5
        let res = index.search(&[(1, 1.0), (3, 1.0)], 5);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, "docA");
        assert!((res[0].1 - 3.5).abs() < 1e-5);
        assert_eq!(res[1].0, "docB");
        assert!((res[1].1 - 0.5).abs() < 1e-5);
    }
}
