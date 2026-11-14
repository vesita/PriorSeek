//! 中英双语概念映射枢纽 (Bilingual Lexicon Hub)。
//!
//! 在外壳侧加载内核构建的双语对齐先验词典 (bilingual_lexicon.json)，
//! 为跨语言概念对齐提供无开销查表，与单语倒排执行严格物理隔离。

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BilingualLexicon {
    pub en_to_zh: HashMap<String, Vec<String>>,
    pub zh_to_en: HashMap<String, Vec<String>>,
}

impl BilingualLexicon {
    /// 从 JSON 字符串加载双语对齐词典。
    pub fn from_json(json_str: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct RawLexicon {
            en_to_zh: HashMap<String, Vec<String>>,
            zh_to_en: HashMap<String, Vec<String>>,
        }

        let raw: RawLexicon =
            serde_json::from_str(json_str).context("解析 bilingual_lexicon.json 失败")?;

        Ok(Self {
            en_to_zh: raw.en_to_zh,
            zh_to_en: raw.zh_to_en,
        })
    }

    /// 给定源语言词元，查找目标语言对应项。
    pub fn map_token(&self, token: &str, src_lang: &str) -> Vec<String> {
        let clean = token.trim();
        if src_lang == "en" {
            self.en_to_zh
                .get(&clean.to_lowercase())
                .cloned()
                .unwrap_or_default()
        } else if src_lang == "zh" {
            self.zh_to_en.get(clean).cloned().unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// 自动推断语言并进行双向跨语言映射（zh -> en, en -> zh）。
    pub fn map_auto(&self, token: &str) -> Vec<String> {
        let clean = token.trim();
        let is_zh = clean.chars().any(|c| matches!(c as u32, 0x4E00..=0x9FFF));
        if is_zh {
            self.zh_to_en.get(clean).cloned().unwrap_or_default()
        } else {
            self.en_to_zh
                .get(&clean.to_lowercase())
                .cloned()
                .unwrap_or_default()
        }
    }

    /// 内置开箱即用的高置信度工程与科学双语对齐词典。
    pub fn core() -> Self {
        let pairs: &[(&str, &[&str])] = &[
            // 计算机工程与软件开发常用核心概念
            ("config", &["配置"]),
            ("configuration", &["配置"]),
            ("request", &["请求"]),
            ("response", &["响应", "回复"]),
            ("database", &["数据库"]),
            ("db", &["数据库"]),
            ("initialize", &["初始化"]),
            ("init", &["初始化"]),
            ("function", &["函数", "功能"]),
            ("func", &["函数"]),
            ("fn", &["函数"]),
            ("parameter", &["参数"]),
            ("param", &["参数"]),
            ("argument", &["参数", "实参"]),
            ("arg", &["参数"]),
            ("error", &["错误", "异常"]),
            ("err", &["错误"]),
            ("message", &["消息", "报文"]),
            ("msg", &["消息"]),
            ("index", &["索引"]),
            ("idx", &["索引"]),
            ("search", &["搜索", "检索"]),
            ("retrieval", &["检索", "召回"]),
            ("query", &["查询"]),
            ("cache", &["缓存"]),
            ("user", &["用户"]),
            ("server", &["服务", "服务器"]),
            ("service", &["服务"]),
            ("client", &["客户端", "客户"]),
            ("connection", &["连接"]),
            ("connect", &["连接"]),
            ("conn", &["连接"]),
            ("auth", &["认证", "授权", "鉴权"]),
            ("authentication", &["认证", "鉴权"]),
            ("authorize", &["授权"]),
            ("file", &["文件"]),
            ("document", &["文档", "文献"]),
            ("doc", &["文档"]),
            ("system", &["系统"]),
            ("network", &["网络"]),
            ("thread", &["线程"]),
            ("process", &["进程"]),
            ("memory", &["内存"]),
            ("mem", &["内存"]),
            ("model", &["模型"]),
            ("algorithm", &["算法"]),
            ("data", &["数据"]),
            ("log", &["日志"]),
            ("logger", &["日志"]),
            ("test", &["测试"]),
            ("async", &["异步"]),
            ("sync", &["同步"]),
            ("lock", &["锁"]),
            ("task", &["任务"]),
            ("worker", &["工作者", "执行者"]),
            ("queue", &["队列"]),
            ("token", &["词元", "标记", "令牌"]),
            ("vector", &["向量"]),
            ("sparse", &["稀疏"]),
            ("dense", &["稠密"]),
            ("kernel", &["内核"]),
            ("shell", &["外壳"]),
            ("weight", &["权重"]),
            ("score", &["得分", "分数"]),
            ("rank", &["排位", "排名"]),
            ("fusion", &["融合"]),
            // 基础科学与生物医学
            ("cancer", &["癌症", "肿瘤"]),
            ("tumor", &["肿瘤", "癌症"]),
            ("cell", &["细胞"]),
            ("gene", &["基因"]),
            ("protein", &["蛋白质", "蛋白"]),
            ("therapy", &["疗法", "治疗"]),
            ("patient", &["患者", "病人"]),
            ("disease", &["疾病"]),
            ("antibody", &["抗体"]),
            ("antigen", &["抗原"]),
            ("infection", &["感染"]),
            ("virus", &["病毒"]),
            ("receptor", &["受体"]),
            ("expression", &["表达"]),
            ("immune", &["免疫"]),
        ];

        let mut en_to_zh: HashMap<String, Vec<String>> = HashMap::new();
        let mut zh_to_en: HashMap<String, Vec<String>> = HashMap::new();

        for &(en, zh_list) in pairs {
            let en_str = en.to_string();
            let mut list = Vec::new();
            for &zh in zh_list {
                list.push(zh.to_string());
                zh_to_en
                    .entry(zh.to_string())
                    .or_default()
                    .push(en_str.clone());
            }
            en_to_zh.insert(en_str, list);
        }

        Self { en_to_zh, zh_to_en }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilingual_lexicon_roundtrip() {
        let json_data = r#"{
            "en_to_zh": {
                "cancer": ["癌症", "肿瘤"]
            },
            "zh_to_en": {
                "细胞": ["cell"]
            }
        }"#;

        let lex = BilingualLexicon::from_json(json_data).unwrap();
        assert_eq!(lex.map_token("cancer", "en"), vec!["癌症", "肿瘤"]);
        assert_eq!(lex.map_token("CANCER", "en"), vec!["癌症", "肿瘤"]);
        assert_eq!(lex.map_token("细胞", "zh"), vec!["cell"]);
        assert!(lex.map_token("unknown", "en").is_empty());
    }
}
