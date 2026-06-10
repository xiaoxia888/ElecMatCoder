from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FlangeSpecRfComboMatch:
    raw: str
    full_span: Tuple[int, int]
    first_value: str
    second_value: str
    first_span: Tuple[int, int]
    second_span: Tuple[int, int]


class ComboFallbackExtractor:
    """
    公共组合兜底提取器。

    当前只负责识别这类法兰型号规格片段：
    - SO100(B)-10RF
    - SO100(B)- 10 RF
    - HG/T20592SO20(A)-25 RF06Cr19Ni10

    语义由各字段处理器自己解释：
    - SIZE 取 first_value 作为 DN 兜底
    - PRESSURE 取 second_value 作为 PN 兜底
    """

    _FLANGE_SPEC_RF_PATTERN = re.compile(
        r'(?i)'
        r'(?:[A-Z]{1,8}\s*)?'
        r'(\d+(?:\.\d+)?)'
        r'\s*(?:\([A-Z]\))?'
        r'\s*-\s*'
        r'(\d+(?:\.\d+)?)'
        r'\s*RF'
    )

    @classmethod
    def extract_flange_spec_rf_combos(
        cls,
        text: str,
        *,
        allow_rf_right_glue: bool,
    ) -> List[FlangeSpecRfComboMatch]:
        source = str(text or "")
        results: List[FlangeSpecRfComboMatch] = []
        for m in cls._FLANGE_SPEC_RF_PATTERN.finditer(source):
            if not cls._passes_left_boundary(source, m.start()):
                continue
            if not allow_rf_right_glue and not cls._passes_strict_right_boundary(source, m.end()):
                continue
            results.append(
                FlangeSpecRfComboMatch(
                    raw=m.group(0).strip(),
                    full_span=(m.start(), m.end()),
                    first_value=m.group(1),
                    second_value=m.group(2),
                    first_span=(m.start(1), m.end(1)),
                    second_span=(m.start(2), m.end(2)),
                )
            )
        return results

    @staticmethod
    def _passes_left_boundary(source: str, start: int) -> bool:
        if start <= 0:
            return True
        left_char = source[start - 1]
        first_char = source[start]
        if first_char.isalpha():
            return not left_char.isalpha()
        if first_char.isdigit():
            return not left_char.isdigit()
        return True

    @staticmethod
    def _passes_strict_right_boundary(source: str, end: int) -> bool:
        if end >= len(source):
            return True
        return not source[end].isalnum()
