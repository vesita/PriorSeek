//! 文本特征提取：归一化、单字、相邻双字、分词（jieba）。
//!
//! 这是本 crate 里**唯一**的清洗与切分点：FAQ 五方案与自实现 BM25 都从这里的
//! 输出切特征，因此「中文标点被过滤」「大小写被归一」各自只实现一次。
//!
//! **适用范围**：`normalize` / `chars` / `bigrams` / `words` 对齐的是
//! `~/coding/hub/faq-search` 的 Python 实现，为中文 FAQ 语料设计。英文语料上用
//! 字符级方案（`faq-char` / `faq-bigram`）会退化成「字母覆盖率」，区分度很低，
//! 这是预期行为而不是 bug；英文语料请用 `bm25` / `rs`。
//!
//! 与 Python 侧的一处刻意差异：Python `normalize` 保留 Unicode 大类 L/N/**M**
//! （组合符号），这里只保留 Alphabetic/Numeric（≈ L+N）。组合符号只在 NFD 形式的
//! 拉丁文/梵文等文本里出现，中文语料里不出现；为它引入一个 Unicode 大类依赖
//! 不划算，因此不做。语料若含 NFD 组合字符，两边会差一个字符。

use jieba_rs::Jieba;
use std::collections::HashSet;
use std::sync::LazyLock;

/// 归一化：去掉标点 / 空白 / 控制字符 / 符号，再做大小写归一。
///
/// 对应 Python 的 `faq_search.normalize.normalize`：丢弃 Unicode 大类 P（标点）、
/// Z（空白/分隔）、C（控制/未分配）、S（符号），保留 L/N（含汉字），最后 `casefold`。
/// 例：`"改完参数，用不用重开系统？"` → `"改完参数用不用重开系统"`。
pub fn normalize(text: &str) -> String {
    let kept: String = text.chars().filter(|ch| ch.is_alphanumeric()).collect();
    kept.to_lowercase()
}

/// 一篇文本在三种特征空间上的切分结果。归一化只做一次。
///
/// 索引侧每篇文档都要切三种特征；若 `chars` / `bigrams` / `words` 各自归一化一次，
/// 同一篇文本会被归一化三遍，纯属浪费。三种特征的**定义**仍然只有这一份 ——
/// 下面三个公开函数只是「先 normalize、再取其中一项」的薄封装。
pub struct Features {
    pub chars: Vec<String>,
    pub bigrams: Vec<String>,
    pub words: Vec<String>,
}

/// 归一化一次，切出三种特征。
pub fn features(text: &str) -> Features {
    let normalized = normalize(text);
    Features {
        chars: chars_of(&normalized),
        bigrams: bigrams_of(&normalized),
        words: words_of(&normalized),
    }
}

/// 切单字。标点与空白已在 [`normalize`] 里去掉。
pub fn chars(text: &str) -> Vec<String> {
    chars_of(&normalize(text))
}

/// 切相邻双字。长度为 1 的输入切不出双字，返回空列表 —— 调用方必须处理这种空特征
/// （FAQ 方案与 BM25 都在 scorer 层退到单字，不是返回 0 分）。
pub fn bigrams(text: &str) -> Vec<String> {
    bigrams_of(&normalize(text))
}

fn chars_of(normalized: &str) -> Vec<String> {
    normalized.chars().map(|ch| ch.to_string()).collect()
}

fn bigrams_of(normalized: &str) -> Vec<String> {
    let cs: Vec<char> = normalized.chars().collect();
    (0..cs.len().saturating_sub(1))
        .map(|i| format!("{}{}", cs[i], cs[i + 1]))
        .collect()
}

/// 停用词：虚字、疑问词、语气词。它们在几乎所有 FAQ 里都出现，参与打分只会制造噪声。
///
/// 与 Python 侧 `faq_search.tokenize.STOPWORDS` 逐条一致 —— 两边不一致会让
/// `word` / `hybrid` 方案的查询特征集合不同，top-k 直接对不上。
pub const STOPWORDS: [&str; 59] = [
    "的",
    "了",
    "是",
    "在",
    "有",
    "和",
    "与",
    "或",
    "吗",
    "呢",
    "吧",
    "啊",
    "什么",
    "怎么",
    "怎样",
    "怎么办",
    "如何",
    "多少",
    "哪些",
    "哪个",
    "是否",
    "要不要",
    "不要",
    "需要",
    "应该",
    "可以",
    "这个",
    "那个",
    "后",
    "时候",
    "要",
    "不",
    "会",
    "能",
    "就",
    "都",
    "也",
    "还",
    "又",
    "再",
    "很",
    "太",
    "更",
    "最",
    "一",
    "个",
    "我",
    "你",
    "他",
    "它",
    "这",
    "那",
    "里",
    "上",
    "下",
    "中",
    "内",
    "外",
    "前",
];

static STOPWORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOPWORDS.iter().copied().collect());

/// 全进程共享一个 jieba 实例（词典只加载一次）。
fn jieba() -> &'static Jieba {
    static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);
    &JIEBA
}

/// 分词并滤掉停用词。
///
/// 对应 Python 的 `faq_search.tokenize.words`：先 [`normalize`]，再用 jieba
/// **精确模式 + HMM**（与 Python `jieba.cut` 默认参数一致）切词，丢掉空白 token
/// 与停用词。归一化后为空时返回空列表（调用方负责兜底）。
///
/// 不照搬 Python 的「jieba 不可用则退到单字」：Rust 侧 jieba-rs 是编译进来的硬依赖，
/// 不存在「装不上」的状态，降级分支没有触发条件。
pub fn words(text: &str) -> Vec<String> {
    words_of(&normalize(text))
}

fn words_of(normalized: &str) -> Vec<String> {
    if normalized.is_empty() {
        return Vec::new();
    }
    jieba()
        .cut(normalized, true)
        .into_iter()
        .map(|token| token.word)
        .filter(|word| !word.trim().is_empty() && !STOPWORD_SET.contains(*word))
        .map(str::to_string)
        .collect()
}

/// 该字符是否属于 CJK 文字（汉字 / 假名 / 谚文）。
fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30FF     // 平假名 + 片假名
        | 0x3400..=0x4DBF   // 汉字扩展 A
        | 0x4E00..=0x9FFF   // 汉字基本区
        | 0xAC00..=0xD7AF   // 谚文音节
        | 0xF900..=0xFAFF   // 汉字兼容区
        | 0x20000..=0x2FA1F // 汉字扩展 B~F + 兼容补充
    )
}

/// 「含 CJK 的块」判定：块内出现任意一个 CJK 字符即为真。
fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

/// BM25 用的词项切分：Unicode 字母数字连续串小写化；含 CJK 的块再用 jieba 分词。
///
/// 索引与查询**共用这一个函数**（`bm25` 适配器的两条路径都只调它），否则查询侧
/// 与文档侧词表不同源，IDF 与命中都会错位。英文/数字串整体成为一个词项；含 CJK
/// 的连续串交给 jieba，切出的词再统一小写。
pub fn bm25_terms(text: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut block_start: Option<usize> = None;
    let push_block = |block: &str, terms: &mut Vec<String>| {
        if block.is_empty() {
            return;
        }
        if contains_cjk(block) {
            for token in jieba().cut(block, true) {
                let word = token.word.trim();
                if !word.is_empty() {
                    terms.push(word.to_lowercase());
                }
            }
        } else {
            terms.push(block.to_lowercase());
        }
    };
    for (offset, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            block_start.get_or_insert(offset);
        } else if let Some(start) = block_start.take() {
            push_block(&text[start..offset], &mut terms);
        }
    }
    if let Some(start) = block_start {
        push_block(&text[start..], &mut terms);
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_punctuation_and_folds_case() {
        assert_eq!(
            normalize("改完参数，用不用重开系统？"),
            "改完参数用不用重开系统"
        );
        assert_eq!(normalize("PDF 报表"), "pdf报表");
        assert_eq!(normalize("  a+b=c  "), "abc");
        assert_eq!(normalize("？？！"), "");
    }

    #[test]
    fn bigrams_need_two_chars() {
        assert!(bigrams("刀").is_empty());
        assert_eq!(bigrams("刀片"), vec!["刀片"]);
        assert_eq!(bigrams("abc"), vec!["ab", "bc"]);
    }

    #[test]
    fn words_drops_stopwords_and_keeps_content_words() {
        // 「怎么」是停用词，「设置」不是；jieba 精确模式把两者切成两个词。
        assert_eq!(words("怎么设置"), vec!["设置"]);
        assert!(words("怎么").is_empty());
        assert!(words("？？").is_empty());
    }

    #[test]
    fn bm25_terms_lowercases_ascii_runs_and_segments_cjk() {
        assert_eq!(bm25_terms("PDF Report_v2"), vec!["pdf", "report", "v2"]);
        // 中文块交给 jieba：具体切法由词典决定，这里只断言切出来的词能拼回原块。
        let terms = bm25_terms("报警阈值");
        assert!(!terms.is_empty());
        assert_eq!(terms.concat(), "报警阈值");
    }
}
