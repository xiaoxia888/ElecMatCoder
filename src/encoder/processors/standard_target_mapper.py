"""
标准到壁厚/外径对照目标标准映射器。

用途：将一阶段/二阶段得到的标准信息映射到壁厚对照表支持的 5 个目标标准之一：
- SHT3405
- HGT20553II
- HGT20553Ia
- AB3619
- AB3610

推荐输入：`decisions["STANDARD"]` 的对象数组。
例如：
[
    {"BODY": "HG/T20553", "GRADE": "Ia"},
    {"BODY": "GB/T12459", "GRADE": "II"},
]

也兼容：
- 单个字符串
- 字符串列表
- 单个标准对象

说明：
- 这里做的是“目标对照标准映射”，不是通用标准提取。
- 输入最好已经是标准字段，而不是整条原始描述。
- 返回空字符串表示未命中，留给上层人工兜底或补规则。
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

TARGET_STANDARDS = (
    "SHT3405",
    "HGT20553II",
    "HGT20553Ia",
    "AB3619",
    "AB3610",
)

_ROMAN_REPLACEMENTS = {
    "Ⅰ": "I",
    "Ⅱ": "II",
    "Ⅲ": "III",
    "Ⅳ": "IV",
    "Ⅴ": "V",
    "Ⅵ": "VI",
    "Ⅶ": "VII",
    "Ⅷ": "VIII",
    "Ⅸ": "IX",
    "Ⅹ": "X",
}


class StandardTargetMapper:
    """将标准信息映射为壁厚/外径换算所需的目标标准。"""

    def __init__(self, target_standards: Sequence[str] | None = None):
        self.target_standards = tuple(target_standards or TARGET_STANDARDS)

    def map_to_target(self, standards: Any) -> str:
        """
        将标准信息映射到目标标准。

        推荐输入：
        - `[{"BODY": "HG/T20553", "GRADE": "Ia"}, ...]`

        兼容输入：
        - `"HG/T20553(Ia);GB/T12459"`
        - `["ASME B16.9", "HG/T20553(Ia)"]`
        - `{"BODY": "ASME B36.10M"}`
        """
        tokens = self._collect_tokens(standards)
        if not tokens:
            return ""

        joined = "|".join(tokens)

        # 1. HG/T 20553 系列，优先区分 Ia / II。
        if self._contains_any(joined, "HGT20553LA", "HGT20553IA"):
            return "HGT20553Ia"
        if self._contains_any(joined, "HGT20553II"):
            return "HGT20553II"
        if self._contains_any(joined, "HGT20553"):
            return "HGT20553Ia"

        # 2. SH/T 石化管道系列（壁厚对照表统一归 SHT3405）。
        if self._contains_any(joined, "SHT3405"):
            return "SHT3405"
        if self._contains_any(joined, "SHT3406", "SHT3408", "SHT3410", "SHT3419"):
            return "SHT3405"

        # 3. ASME B36 管道尺寸。
        if self._contains_any(joined, "ASMEB3619", "AB3619", "B3619M", "B3619"):
            return "AB3619"
        if self._contains_any(joined, "ASMEB3610", "AB3610", "B3610M", "B3610"):
            return "AB3610"

        # 4. ASME B16 管件/法兰，外径体系回落到 AB3610。
        if self._contains_any(
            joined,
            "ASMEB169", "AB169", "B169",
            "ASMEB1610", "AB1610", "B1610",
            "ASMEB1611", "AB1611", "B1611",
            "ASMEB165", "AB165", "B165",
            "ASMEB1647", "AB1647", "B1647",
            "ASMEB1648", "AB1648", "B1648",
            "ASMEB1625", "AB1625", "B1625",
        ):
            return "AB3610"

        # 5. GB/T 12771 不锈钢焊管，多按 HG/T 20553 Ia 体系。
        if self._contains_any(joined, "GBT12771"):
            return "HGT20553Ia"

        # 6. GB/T 12459 / 13401 对焊管件。
        if self._contains_any(joined, "GBT12459", "GBT13401"):
            if self._contains_any(joined, "SHT"):
                return "SHT3405"
            if self._contains_any(joined, "GBT12459II", "GBT13401II"):
                return "HGT20553II"
            if self._contains_any(joined, "HGT20553", "HGT20592", "HGT20615", "HGT20538"):
                return "HGT20553Ia"
            return "AB3610"

        # 7. 钢管类国标。
        if self._contains_any(joined, "GBT8163", "GBT9711", "GBT3087", "GBT9948"):
            if self._contains_any(joined, "SHT"):
                return "SHT3405"
            if self._contains_any(joined, "HGT20553", "HGT20538"):
                return "HGT20553Ia"
            return "SHT3405"

        if self._contains_any(joined, "GBT14976", "GBT5310", "GBT4237", "GBT711", "GBT713", "GBT3274"):
            if self._contains_any(joined, "SHT"):
                return "SHT3405"
            if self._contains_any(joined, "ASMEB169", "AB169", "B169", "ASMEB313", "AB313", "B313"):
                return "AB3610"
            return "SHT3405"

        # 8. HG/T 法兰/管件。
        if self._contains_any(joined, "HGT20592", "HGT20615", "HGT20538"):
            return "HGT20553Ia"

        # 9. NB/T 压力容器配套。
        if self._contains_any(joined, "NBT47008", "NBT47010"):
            if self._contains_any(joined, "MSSSP97", "MSSSP95", "MSSSP75", "MS97", "MS95", "MS75"):
                return "AB3610"
            if self._contains_any(joined, "GBT19326II", "GBT14383II"):
                return "HGT20553II"
            if self._contains_any(joined, "HGT"):
                return "HGT20553Ia"
            if self._contains_any(joined, "SHT", "GBT4334C"):
                return "SHT3405"
            return "HGT20553Ia"

        # 10. GB/T 12228 通用管法兰。
        if self._contains_any(joined, "GBT12228"):
            if self._contains_any(joined, "HGT20615", "HGT205615"):
                return "HGT20553Ia"
            return "SHT3405"

        # 11. GB/T 19326 / 14383 锻制管件。
        if self._contains_any(joined, "GBT19326", "GBT14383"):
            if self._contains_any(joined, "GBT19326II", "GBT14383II"):
                return "HGT20553II"
            return "HGT20553Ia"

        # 12. MSS 系列。
        if self._contains_any(joined, "MSSSP97", "MSSSP95", "MSSSP75", "MS97", "MS95", "MS75"):
            return "AB3610"

        # 13. 其他 EN/BS/ASME 工程标准。
        if self._contains_any(joined, "BS3799", "DINEN1127", "EN10305"):
            return "AB3610"

        # 14. 补充标准。
        if self._contains_any(joined, "ASMEB163", "AB163", "B163", "ASMEB313", "AB313", "B313"):
            return "AB3610"
        if self._contains_any(joined, "GBT17395", "GBT3091", "GBT4334C"):
            return "SHT3405"

        return ""

    def _collect_tokens(self, standards: Any) -> List[str]:
        items = self._coerce_to_list(standards)
        tokens: List[str] = []
        for item in items:
            for token in self._tokens_from_item(item):
                if token and token not in tokens:
                    tokens.append(token)
        return tokens

    def _coerce_to_list(self, standards: Any) -> List[Any]:
        if standards is None:
            return []
        if self._is_nan(standards):
            return []
        if isinstance(standards, (str, dict)):
            return [standards]
        if isinstance(standards, Sequence):
            return list(standards)
        return [standards]

    def _tokens_from_item(self, item: Any) -> List[str]:
        if item is None or self._is_nan(item):
            return []

        if isinstance(item, dict):
            body = str(item.get("BODY") or "").strip()
            grade = str(item.get("GRADE") or "").strip()
            appendix = str(item.get("APPENDIX") or "").strip()
            method = str(item.get("METHOD") or "").strip()

            return self._build_tokens(body, grade, appendix, method)

        text = str(item).strip()
        if not text:
            return []

        return self._build_tokens(text, "", "", "")

    def _build_tokens(self, body: str, grade: str, appendix: str, method: str) -> List[str]:
        body_n = self._normalize_fragment(body)
        grade_n = self._normalize_fragment(grade)
        appendix_n = self._normalize_fragment(appendix)
        method_n = self._normalize_fragment(method)

        tokens: List[str] = []
        for token in [
            body_n,
            grade_n,
            appendix_n,
            method_n,
            body_n + grade_n,
            body_n + appendix_n,
            body_n + method_n,
            body_n + grade_n + appendix_n,
            body_n + grade_n + method_n,
            body_n + grade_n + appendix_n + method_n,
        ]:
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    @staticmethod
    def _contains_any(text: str, *patterns: str) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _normalize_fragment(text: str) -> str:
        s = str(text or "").strip().upper()
        for src, dst in _ROMAN_REPLACEMENTS.items():
            s = s.replace(src, dst)
        s = s.replace("SERIAL", "SERIES")
        s = s.replace("LA", "IA") if "20553LA" in s else s
        return re.sub(r"[^A-Z0-9]", "", s)

    @staticmethod
    def _is_nan(value: Any) -> bool:
        return isinstance(value, float) and math.isnan(value)


@lru_cache(maxsize=1)
def get_standard_target_mapper() -> StandardTargetMapper:
    return StandardTargetMapper()


def map_standard_to_target(standards: Any) -> str:
    """便捷函数：将标准信息映射到目标标准。"""
    return get_standard_target_mapper().map_to_target(standards)
