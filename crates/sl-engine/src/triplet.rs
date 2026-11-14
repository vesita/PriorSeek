//! 三元组因果检索核心引擎 (Triplet Causal Retrieval)
//! 
//! 本模块实现核心数据结构：将文档视为一组因果三元组，取代词频索引。

use std::collections::{HashMap, HashSet};

/// 三元组单元：(S, R, O)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Triplet {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// 基于三元组的倒排索引
pub struct TripletIndex {
    // 谓词 -> 文档列表
    pub relation_postings: HashMap<String, HashSet<String>>,
    // 实体 -> 参与的文档列表
    pub entity_postings: HashMap<String, HashSet<String>>,
}

impl TripletIndex {
    pub fn new() -> Self {
        Self {
            relation_postings: HashMap::new(),
            entity_postings: HashMap::new(),
        }
    }

    pub fn index_doc(&mut self, doc_id: &str, triplets: Vec<Triplet>) {
        for t in triplets {
            self.relation_postings
                .entry(t.relation)
                .or_default()
                .insert(doc_id.to_string());
            self.entity_postings
                .entry(t.subject)
                .or_default()
                .insert(doc_id.to_string());
            self.entity_postings
                .entry(t.object)
                .or_default()
                .insert(doc_id.to_string());
        }
    }

    /// 核心检索算子：直接进行逻辑合取，实现逻辑级的反事实防御
    pub fn search(&self, q: Triplet) -> HashSet<String> {
        let s_docs = self.entity_postings.get(&q.subject);
        let r_docs = self.relation_postings.get(&q.relation);
        let o_docs = self.entity_postings.get(&q.object);

        // 如果三个节点任何一个缺失，则逻辑路径断开，返回空集 (即完美防御)
        match (s_docs, r_docs, o_docs) {
            (Some(s), Some(r), Some(o)) => {
                s.intersection(r).cloned().collect::<HashSet<_>>()
                    .intersection(o).cloned().collect()
            }
            _ => HashSet::new(),
        }
    }
}
