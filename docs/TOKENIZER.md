# tokenizer 现状与版本

> 本仓的「字面层」事实源。当前有两套切分并存，**它们是不同的实验变量，不能混读**。

## v0：词级（默认，`bm25` / `faq-*` / `seq` 在用）

- 实现：`crates/sl-adapters/src/tokenize.rs`（唯一清洗/切分点）+ `tools/kernel/tokenizer.py`（镜像，用于固化文档 token）
- 规则：保留字母数字并小写；字母数字连续串 = 一个 token；含 CJK 的块交给 jieba（`bm25_terms`）；
  `words` / `chars` / `bigrams` 三种特征供 FAQ 五方案使用
- 局限：无词干化、无子词；`monocyte` 与 `monocytes` 是不同 token

## v1：LLM BPE（`bm25-llm` 在用）

- 实现：`crates/sl-adapters/src/tokenizer_llm.rs`（Rust）+ Python `tokenizers` 库（工件构建/对拍）
- 词表：Qwen2.5-0.5B 的 `tokenizer.json`（BPE，151,665），sha256 钉死在 `fetch_tokenizer.py`
- 完整性：`fetch_tokenizer.py` 下载 + 校验；`LlmTokenizer::from_path` 与同目录 `manifest.json` 比对
- 双端对拍：`crates/sl-adapters/tests/llm_tokenizer_parity.rs`（20 条用例，token id 与 token 串逐位相同）
- 实测：**纯词法 BM25 上更差**（nDCG@10 −0.0297，miss@10 +2.67pp，见 `RESULTS.md` §11），
  但互补（hit@10 救回 11 条 / 弄丢 19 条）

## 换 tokenizer 的规矩

1. 换 tokenizer = 换实验变量 ⇒ `AdapterInfo.config` 必须记录 `tokenizer_id` / `sha256` / `vocab_size`；
2. 工件（`seq.bin` / `doc_tokens.bin`）与 tokenizer 绑定，换 tokenizer 必须重建工件，不能复用；
3. 必须过双端对拍，否则查询侧与文档侧会静默错位；
4. 不能拿 test 集选 tokenizer（在 train 上选，test 上一次评）。
