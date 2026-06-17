"""
磅级处理器

第一版只做：
1. 强规则：CL / CLASS / LB / LBS / # / PN / MPa / BAR
2. 组合锚点：两侧都必须是合法磅级 token，允许 `/ ; ， ,`

明确不做：
1. RF 组合（如 16RF / RF 16），但允许法兰型号规格中的兜底场景：100(B)-10RF
2. 裸数字兜底
3. 最终只输出 PN / CL 两个系列
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .combo_fallback_extractor import ComboFallbackExtractor


@dataclass
class RulePressureExtraction:
    values: List[str]
    pressure_code: str
    matched_texts: List[str]
    matched_spans: List[Tuple[int, int]]
    consumed_spans: List[Tuple[int, int]] = field(default_factory=list)
    ordered_items: List[Dict[str, str]] = field(default_factory=list)
    cleared: bool = False
    clear_reason: str = ""


class PressureProcessor:
    """磅级处理器"""

    DEFAULT_CLASS_VALUES = (
        "150", "300", "400", "600", "900",
        "1500", "2500", "3000", "6000", "9000",
    )
    DEFAULT_PN_VALUES = (
        "2.5", "6", "10", "16", "20", "25", "40", "50", "63", "68",
        "100", "110", "150", "160", "250", "260", "320", "400", "420",
    )

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.class_values = tuple(
            str(v) for v in self.config.get("class_values", self.DEFAULT_CLASS_VALUES)
        )
        self.pn_values = tuple(
            str(v) for v in self.config.get("pn_values", self.DEFAULT_PN_VALUES)
        )
        self.class_value_set = set(self.class_values)
        self.pn_value_set = set(self.pn_values)
        self._build_patterns()

    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "encoder_config.yaml"
        else:
            config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                return full_config.get("pressure_processing", {})
        return {}

    def _build_patterns(self) -> None:
        hash_value_pattern = "|".join(
            sorted((re.escape(v) for v in self.class_values), key=len, reverse=True)
        )
        # CL / CLASS：只认数字等级，天然过滤 Class I / Class II
        self.cl_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])CL\s*\.?\s*(\d+)(?![A-Z0-9])"
        )
        self.class_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])CLASS\s*(\d+)(?![A-Z0-9])"
        )
        self.lb_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])(\d+)\s*LB(?:S)?(?![A-Z0-9])"
        )
        self.hash_pattern = re.compile(
            rf"(?i)(?<![A-Z0-9])({hash_value_pattern})\s*#(?![A-Z0-9])"
        )
        self.encoded_class_pattern = re.compile(
            rf"(?i)(?<![A-Z0-9])C\s*({hash_value_pattern})(?![A-Z0-9])"
        )
        self.pn_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])PN\s*(\d+(?:\.\d+)?)"
        )

        # MPa / BAR 只认独立压力 token；比较符、FV/6Bar 这类场景在命中后再排除
        self.mpa_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])(\d+(?:\.\d+)?)\s*MPA(?![A-Z0-9])"
        )
        self.bar_pattern = re.compile(
            r"(?i)(?<![A-Z0-9])(\d+(?:\.\d+)?)\s*BAR(?![A-Z0-9])"
        )

        self.combo_separator_pattern = re.compile(r"\s*(?:/|;|,|，)\s*")
        self.slash_separator_pattern = re.compile(r"\s*/\s*")
        self.comparison_chars = set("><=≤≥")
        self.pressure_token_pattern = re.compile(
            rf"(?i)("
            rf"CL\s*\.?\s*\d+"
            rf"|CLASS\s*\d+"
            rf"|C\s*(?:{hash_value_pattern})"
            rf"|\d+\s*LB(?:S)?"
            rf"|(?:{hash_value_pattern})\s*#"
            rf"|PN\s*\d+(?:\.\d+)?"
            rf"|\d+(?:\.\d+)?\s*MPA"
            rf"|\d+(?:\.\d+)?\s*BAR"
            rf")"
        )

    def process(self, value: str) -> str:
        """
        处理单个磅级值或规则已抽取的组合值，返回标准编码。
        """
        if not value:
            return ""
        normalized = self._normalize_pressure_token(value, allow_prefix=False)
        if normalized:
            return normalized

        parts = [p.strip() for p in self.combo_separator_pattern.split(str(value).strip()) if p.strip()]
        if len(parts) >= 2:
            normalized_parts: List[str] = []
            for part in parts:
                normalized_part = self._normalize_pressure_token(part, allow_prefix=False)
                if not normalized_part:
                    return str(value).strip().upper()
                if normalized_part not in normalized_parts:
                    normalized_parts.append(normalized_part)
            return "/".join(normalized_parts)

        return str(value).strip().upper()

    def extract_by_rules(self, text: str, blocked_spans: Optional[List[Tuple[int, int]]] = None) -> RulePressureExtraction:
        source = str(text or "")
        values: List[str] = []
        matched_texts: List[str] = []
        matched_spans: List[Tuple[int, int]] = []
        ordered_items: List[Dict[str, str]] = []
        consumed_spans: List[Tuple[int, int]] = []
        blocked_spans = blocked_spans or []

        def overlaps(span: Tuple[int, int]) -> bool:
            return any(start < span[1] and span[0] < end for start, end in blocked_spans) or any(start < span[1] and span[0] < end for start, end in consumed_spans)

        def add_hit(raw_value: str, normalized_value: str, span: Tuple[int, int]) -> None:
            if overlaps(span):
                return
            if normalized_value not in values:
                values.append(normalized_value)
            raw_text = raw_value.strip()
            if raw_text and raw_text not in matched_texts:
                matched_texts.append(raw_text)
            if span not in matched_spans:
                matched_spans.append(span)
            ordered_items.append({"type": "PRESSURE", "value": normalized_value, "span": span})
            consumed_spans.append(span)

        token_matches: List[Tuple[int, int, str, str]] = []
        for m in self.pressure_token_pattern.finditer(source):
            span = m.span(1)
            raw = m.group(1)
            normalized = self._normalize_pressure_token(
                raw,
                allow_prefix=self._allow_prefix_normalization(source, span, raw),
            )
            if not normalized:
                continue
            if not self._is_valid_match_context(source, span, raw):
                continue
            token_matches.append((span[0], span[1], raw, normalized))

        i = 0
        while i < len(token_matches):
            start, end, raw, normalized = token_matches[i]
            combo_raw_parts = [raw]
            combo_normalized_parts = [normalized]
            combo_end = end
            j = i + 1
            while j < len(token_matches):
                next_start, next_end, next_raw, next_normalized = token_matches[j]
                separator_text = source[combo_end:next_start]
                if not self.slash_separator_pattern.fullmatch(separator_text):
                    break
                combo_raw_parts.append(next_raw)
                combo_normalized_parts.append(next_normalized)
                combo_end = next_end
                j += 1

            if len(combo_raw_parts) >= 2:
                span = (start, combo_end)
                raw_combo = source[start:combo_end]
                add_hit(raw_combo, "/".join(combo_normalized_parts), span)
                i = j
                continue

            span = (start, end)
            add_hit(raw, normalized, span)
            i += 1

        if not values:
            for raw, normalized, span in self._extract_flange_spec_pn_rf_fallback(source):
                add_hit(raw, normalized, span)

        ordered_items_sorted = sorted(ordered_items, key=lambda x: (x["span"][0], x["span"][1]))
        deduped_items: List[Dict[str, object]] = []
        seen_values: set[str] = set()
        for item in ordered_items_sorted:
            value = str(item["value"])
            if value in seen_values:
                continue
            deduped_items.append(item)
            seen_values.add(value)

        if len(deduped_items) >= 2:
            return RulePressureExtraction(
                values=[],
                pressure_code="",
                matched_texts=matched_texts,
                matched_spans=matched_spans,
                consumed_spans=[],
                ordered_items=[],
                cleared=True,
                clear_reason="multiple_pressure_without_slash",
            )

        pressure_code = "/".join(str(item["value"]) for item in deduped_items)
        consumed_value_spans = self._derive_consumed_spans_from_ordered_items(source, deduped_items)
        return RulePressureExtraction(
            values=[str(item["value"]) for item in deduped_items],
            pressure_code=pressure_code,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
            consumed_spans=consumed_value_spans,
            ordered_items=[{"type": "PRESSURE", "value": str(item["value"])} for item in deduped_items],
        )

    @staticmethod
    def _derive_consumed_spans_from_ordered_items(text: str, ordered_items: List[Dict[str, object]]) -> List[Tuple[int, int]]:
        consumed_spans: List[Tuple[int, int]] = []
        numeric_token_re = re.compile(r'\d+(?:\.\d+)?')
        for item in ordered_items:
            span = item.get("span")
            if not span or not isinstance(span, tuple) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or end > len(text) or start >= end:
                continue
            for token_match in numeric_token_re.finditer(text[start:end]):
                candidate = (start + token_match.start(), start + token_match.end())
                if candidate not in consumed_spans:
                    consumed_spans.append(candidate)
        return consumed_spans

    def _normalize_pressure_token(self, value: str, allow_prefix: bool = False) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        upper = raw.upper().replace("＃", "#")
        upper = re.sub(r"\s+", " ", upper).strip()

        m = re.fullmatch(r"CL\s*\.?\s*(\d+)", upper, re.IGNORECASE)
        if m:
            value_part = self._normalize_common_numeric(m.group(1), self.class_values, allow_prefix=allow_prefix)
            return f"C{value_part}" if value_part else ""
        m = re.fullmatch(r"CLASS\s*(\d+)", upper, re.IGNORECASE)
        if m:
            value_part = self._normalize_common_numeric(m.group(1), self.class_values, allow_prefix=allow_prefix)
            return f"C{value_part}" if value_part else ""
        m = re.fullmatch(r"(\d+)\s*LB(?:S)?", upper, re.IGNORECASE)
        if m:
            value_part = self._normalize_common_numeric(m.group(1), self.class_values, allow_prefix=False)
            return f"C{value_part}" if value_part else ""
        m = self.hash_pattern.fullmatch(upper)
        if m:
            return f"C{m.group(1)}"
        m = self.encoded_class_pattern.fullmatch(upper)
        if m:
            return f"C{m.group(1)}"
        m = re.fullmatch(r"PN\s*(\d+(?:\.\d+)?)", upper, re.IGNORECASE)
        if m:
            value_part = self._normalize_common_numeric(m.group(1), self.pn_values, allow_prefix=allow_prefix)
            return f"PN{value_part}" if value_part else ""
        m = self.mpa_pattern.fullmatch(upper)
        if m:
            value_part = self._normalize_mpa_to_pn_value(m.group(1))
            return f"PN{value_part}" if value_part else ""
        m = self.bar_pattern.fullmatch(upper)
        if m:
            value_part = self._normalize_bar_to_pn_value(m.group(1))
            return f"PN{value_part}" if value_part else ""
        return ""

    def _normalize_common_numeric(
        self,
        raw_numeric: str,
        allowed_values: Tuple[str, ...],
        allow_prefix: bool = False,
    ) -> str:
        numeric = str(raw_numeric or "").strip()
        if not numeric:
            return ""
        if numeric in allowed_values:
            return numeric
        # 允许处理粘连场景：CL30002.规格 -> C3000；CL3000200X15 -> C3000
        if allow_prefix:
            for allowed in sorted(allowed_values, key=len, reverse=True):
                if numeric.startswith(allowed):
                    remainder = numeric[len(allowed):]
                    if remainder:
                        return allowed
        return ""

    def _normalize_mpa_to_pn_value(self, raw_numeric: str) -> str:
        try:
            numeric = float(str(raw_numeric).strip())
        except Exception:
            return ""
        candidate = numeric * 10
        if abs(candidate - round(candidate)) < 1e-9:
            value = str(int(round(candidate)))
        else:
            value = str(candidate).rstrip("0").rstrip(".")
        return value if value in self.pn_value_set else ""

    def _normalize_bar_to_pn_value(self, raw_numeric: str) -> str:
        value = str(raw_numeric).strip().rstrip("0").rstrip(".") if "." in str(raw_numeric) else str(raw_numeric).strip()
        return value if value in self.pn_value_set else ""

    def _is_valid_match_context(self, source: str, span: Tuple[int, int], raw: str) -> bool:
        raw_upper = raw.upper()

        if "MPA" in raw_upper or "BAR" in raw_upper:
            idx = span[0] - 1
            while idx >= 0 and source[idx].isspace():
                idx -= 1
            if idx >= 0 and source[idx] in self.comparison_chars:
                return False
            if idx >= 0 and source[idx] == "/":
                left = source[:idx].rstrip()
                left_token_match = re.search(r'([A-Za-z0-9.#]+)\s*$', left)
                if left_token_match:
                    left_token = left_token_match.group(1)
                    if not self._normalize_pressure_token(left_token):
                        return False
                else:
                    return False

        return True

    def _allow_prefix_normalization(self, source: str, span: Tuple[int, int], raw: str) -> bool:
        upper = raw.upper().strip()
        if not (upper.startswith("CL") or upper.startswith("CLASS") or upper.startswith("PN")):
            return False
        if span[1] >= len(source):
            return False
        next_char = source[span[1]]
        return next_char in ".Xx×*;/,，:："

    def _extract_flange_spec_pn_rf_fallback(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """
        兜底识别法兰型号规格中的 PN + RF 组合，例如：
        - SO100(B)-10RF
        - SO100(B)- 10 RF
        - SO25(A)-16 RF06Cr19Ni10

        第二段数字解释为 PN 等级。
        这里允许 RF 右侧继续粘连材质串，不要求 RF 后必须是边界。
        """
        results: List[Tuple[str, str, Tuple[int, int]]] = []
        for combo in ComboFallbackExtractor.extract_flange_spec_rf_combos(
            text,
            allow_rf_right_glue=True,
        ):
            pn_value = self._normalize_common_numeric(combo.second_value, self.pn_values, allow_prefix=False)
            if not pn_value:
                continue
            results.append((combo.raw, f"PN{pn_value}", (combo.second_span[0], combo.full_span[1])))
        return results


_processor_instance: Optional[PressureProcessor] = None


def get_pressure_processor() -> PressureProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = PressureProcessor()
    return _processor_instance
