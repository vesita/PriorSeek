#!/usr/bin/env python3
"""中英双语 Token 映射枢纽 (Bilingual Token Alignment Hub)。

作为跨语言的语义枢纽层 (Semantic Hub)：
- 负责将 Token / 概念在多语言之间进行干净映射（Token -> 对齐语言 Token）；
- 严格与语系特化扩散网络（Language-Specific Diffusion）解耦：
  映射层只做概念对齐，语系特化扩散网络专心在单语语法与词表内进行同语系扩散，
  从而彻底消除多语言全局词表的语义交叉污染与词表膨胀。
"""

from __future__ import annotations

import json
from pathlib import Path


class BilingualHub:
    """中英双语概念映射枢纽。"""

    def __init__(self, custom_lexicon: dict[str, list[str]] | None = None):
        self._en_to_zh: dict[str, list[str]] = {}
        self._zh_to_en: dict[str, list[str]] = {}
        self._init_core_lexicon()
        if custom_lexicon:
            for k, vals in custom_lexicon.items():
                self.add_pair(k, vals)

    def _init_core_lexicon(self) -> None:
        """初始化高置信度核心双语对齐概念。"""
        core_pairs = [
            # 基础生物医学与通用科学
            ("cell", ["细胞"]),
            ("cellular", ["细胞", "细胞的"]),
            ("cancer", ["癌症", "肿瘤"]),
            ("tumor", ["肿瘤", "癌症"]),
            ("gene", ["基因"]),
            ("protein", ["蛋白质", "蛋白"]),
            ("disease", ["疾病", "病症"]),
            ("therapy", ["疗法", "治疗"]),
            ("treatment", ["治疗", "处理"]),
            ("patient", ["患者", "病人"]),
            ("antibody", ["抗体"]),
            ("antigen", ["抗原"]),
            ("infection", ["感染"]),
            ("virus", ["病毒"]),
            ("bacteria", ["细菌"]),
            ("mutation", ["突变", "变异"]),
            ("receptor", ["受体"]),
            ("expression", ["表达"]),
            ("inhibition", ["抑制"]),
            ("activation", ["激活"]),
            ("increase", ["增加", "上升"]),
            ("decrease", ["减少", "下降"]),
            ("regulation", ["调控", "调节"]),
            ("mechanism", ["机制", "机理"]),
            ("immune", ["免疫"]),
            ("response", ["反应", "响应"]),
            ("structure", ["结构"]),
            ("function", ["功能"]),
            ("tissue", ["组织"]),
            ("organ", ["器官"]),
            ("blood", ["血液"]),
            ("brain", ["大脑", "脑"]),
            ("bone", ["骨", "骨骼"]),
            ("drug", ["药物", "药品"]),
            ("clinical", ["临床"]),
            ("trial", ["试验", "试用"]),
            # 计算机与检索
            ("search", ["搜索", "检索"]),
            ("retrieval", ["检索", "召回"]),
            ("query", ["查询", "搜索词"]),
            ("index", ["索引"]),
            ("document", ["文档", "文献"]),
            ("model", ["模型"]),
            ("network", ["网络"]),
            ("algorithm", ["算法"]),
            ("data", ["数据"]),
            ("system", ["系统"]),
            ("weight", ["权重"]),
            ("score", ["得分", "分数"]),
            ("kernel", ["内核"]),
            ("shell", ["外壳"]),
            ("token", ["词元", "标记"]),
            ("vector", ["向量"]),
            ("sparse", ["稀疏"]),
            ("dense", ["稠密", "密集"]),
            ("fusion", ["融合"]),
            ("rank", ["排位", "排名", "等级"]),
            # 经济与社会
            ("bank", ["银行"]),
            ("stock", ["股票", "证券"]),
            ("finance", ["金融", "财务"]),
            ("market", ["市场"]),
            ("economy", ["经济"]),
            ("price", ["价格"]),
            ("cost", ["成本"]),
            ("risk", ["风险"]),
            ("credit", ["信用", "信贷"]),
            ("insurance", ["保险"]),
        ]
        for en_term, zh_terms in core_pairs:
            self.add_pair(en_term, zh_terms)

    def add_pair(self, en_term: str, zh_terms: list[str]) -> None:
        en = en_term.strip().lower()
        self._en_to_zh.setdefault(en, [])
        for zh in zh_terms:
            z = zh.strip()
            if z and z not in self._en_to_zh[en]:
                self._en_to_zh[en].append(z)
            self._zh_to_en.setdefault(z, [])
            if en not in self._zh_to_en[z]:
                self._zh_to_en[z].append(en)

    def align(self, token: str, src_lang: str, target_lang: str) -> list[str]:
        """将源语言 Token 映射到目标语言的一组对齐 Token。"""
        t = token.strip()
        if src_lang == target_lang:
            return [t]
        if src_lang == "en" and target_lang == "zh":
            return self._en_to_zh.get(t.lower(), [])
        if src_lang == "zh" and target_lang == "en":
            return self._zh_to_en.get(t, [])
        return []

    def export_lexicon(self, out_path: Path) -> None:
        data = {
            "en_to_zh": self._en_to_zh,
            "zh_to_en": self._zh_to_en,
            "counts": {
                "en_terms": len(self._en_to_zh),
                "zh_terms": len(self._zh_to_en),
            },
        }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    hub = BilingualHub()
    print(f"BilingualHub 初始化完成: {len(hub._en_to_zh)} 英文词条, {len(hub._zh_to_en)} 中文词条")
    print("测试 'cell' ->", hub.align("cell", "en", "zh"))
    print("测试 '检索' ->", hub.align("检索", "zh", "en"))
