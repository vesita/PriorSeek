"""与 Rust `crates/sl-adapters/src/tokenize.rs::bm25_terms` 等价的 Python 分词。

口径（两端必须逐条一致，否则查询词与工件词表会错位）：

1. 按「Unicode 字母数字连续串」切块：字符属于字母（L*）或数字（N*）就连在一起，
   其余字符（标点、空白、符号、控制字符）都是分隔符；
2. 不含 CJK 的块整体小写化后作为一个词项（英文 / 数字 / 代码标识符不会被拆开）；
3. 含任意 CJK 字符的块交给 jieba **精确模式 + HMM**（`jieba.cut(block, HMM=True)`，
   与 jieba-rs 的 `cut(block, true)` 对应），切出的词去空白、小写化；
4. 不做停用词过滤（`bm25_terms` 也不做；停用词只属于 FAQ 方案）。

**已知的刻意差异**（不影响本仓四个数据集，语料里没有组合符号）：

- Rust 的 `char::is_alphanumeric()` 还包含 Other_Alphabetic 里的组合符号（如天城文
  元音符号，Unicode 大类 M），Python 侧只按 L*/N* 判定；NFD 形式的拉丁文/梵文文本上
  两边会差一个字符。中文与英文语料里不出现这类字符。
- 大小写归并用 Python `str.lower()`，与 Rust `char::to_lowercase()` 在常见字符上一致。

产物 `doc_tokens.bin` 里的文档 token 序列由本模块切好固化，Rust 侧不再切文档；
查询侧仍由 Rust 现切，极少数 jieba / jieba-rs 切词差异只会让个别查询词 OOV。
"""

from __future__ import annotations

import logging
import unicodedata

import jieba

# jieba 首次加载词典会往 stderr 打「Building prefix dict ...」，构建日志里不需要；
# 真正错误仍会以 WARNING 以上级别输出。
jieba.setLogLevel(logging.WARNING)

# CJK 文字区段，与 Rust `is_cjk` 逐条一致（汉字 / 假名 / 谚文）。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),   # 平假名 + 片假名
    (0x3400, 0x4DBF),   # 汉字扩展 A
    (0x4E00, 0x9FFF),   # 汉字基本区
    (0xAC00, 0xD7AF),   # 谚文音节
    (0xF900, 0xFAFF),   # 汉字兼容区
    (0x20000, 0x2FA1F), # 汉字扩展 B~F + 兼容补充
)


def is_cjk(ch: str) -> bool:
    """单个字符是否属于 CJK 文字（汉字 / 假名 / 谚文）。"""
    cp = ord(ch)
    return any(low <= cp <= high for low, high in _CJK_RANGES)


def contains_cjk(text: str) -> bool:
    """块内出现任意一个 CJK 字符即为真。"""
    return any(is_cjk(ch) for ch in text)


def _is_alnum(ch: str) -> bool:
    """字母数字判定：Unicode 大类 L*（字母）或 N*（数字）。

    不用 `str.isalnum()`：它在个别 Unicode 版本上会把「带圈数字」等算进来，而这里
    要的是与 Rust `is_alphanumeric()` 相同的 L*/N* 口径（差异见模块 docstring）。
    """
    return unicodedata.category(ch)[0] in ("L", "N")


def _push_block(block: str, terms: list[str]) -> None:
    if not block:
        return
    if contains_cjk(block):
        for word in jieba.lcut(block, HMM=True):
            word = word.strip()
            if word:
                terms.append(word.lower())
    else:
        terms.append(block.lower())


def bm25_terms(text: str) -> list[str]:
    """BM25 词项切分：Unicode 字母数字连续串小写化；含 CJK 的块再用 jieba 分词。

    索引（文档）与查询共用这一个函数；`build_seq.py` 用它切文档并把结果写进
    `doc_tokens.bin`，`ref_score.py` 用它切查询。
    """
    terms: list[str] = []
    block_start: int | None = None
    for index, ch in enumerate(text):
        if _is_alnum(ch):
            if block_start is None:
                block_start = index
        elif block_start is not None:
            _push_block(text[block_start:index], terms)
            block_start = None
    if block_start is not None:
        _push_block(text[block_start:], terms)
    return terms
