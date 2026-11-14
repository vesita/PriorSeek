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
