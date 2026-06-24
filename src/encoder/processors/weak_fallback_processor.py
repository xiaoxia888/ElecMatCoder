"""弱兜底处理器。

只承接已经从主规则器中拆出的“近似乱捞”规则：
1. 尺寸：末尾裸整数 -> 疑似 DN
2. 壁厚：单个小数 mm -> 疑似壁厚
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .size_processor import RuleSizeExtraction, SizeProcessor
from .thickness_processor import RuleThicknessExtraction, ThicknessProcessor


class WeakFallbackProcessor:
    """统一承接尺寸/壁厚的最弱兜底规则。"""

    DECIMAL_MM_FALLBACK_PATTERN = re.compile(
        r'(?i)(?<![A-Za-z0-9])(\d+\.\d+)\s*(MM|毫米)\b'
    )

    def __init__(
        self,
        *,
        size_processor: Optional[SizeProcessor] = None,
        thickness_processor: Optional[ThicknessProcessor] = None,
    ) -> None:
        self.size_processor = size_processor or SizeProcessor()
        self.thickness_processor = thickness_processor or ThicknessProcessor(enable_rule_layered=False)

    def apply_size_tail_dn_fallback(
        self,
        text: str,
        size_result: RuleSizeExtraction,
        *,
        blocked_spans: Optional[list[Tuple[int, int]]] = None,
    ) -> RuleSizeExtraction:
        if size_result.dn or size_result.od or size_result.inch:
            return size_result

        source = str(text or "")
        normalized = source.replace('”', '"').replace('“', '"').replace('″', '"')
        tail_dn_block = self.size_processor._extract_tail_dn_fallback_block(normalized)
        if not tail_dn_block:
            return size_result
        blocked_spans = blocked_spans or []
        span = tail_dn_block["span"]
        for start, end in blocked_spans:
            if span[0] < end and start < span[1]:
                return size_result

        dn_value = self.size_processor._normalize_number_text(tail_dn_block["dn"])
        matched_texts = list(size_result.matched_texts)
        matched_spans = list(size_result.matched_spans)
        ordered_items = list(size_result.ordered_items)

        raw_text = str(tail_dn_block["raw"] or "").strip()
        if raw_text and raw_text not in matched_texts:
            matched_texts.append(raw_text)
        if span not in matched_spans:
            matched_spans.append(span)

        candidate = {"type": "DN", "value": dn_value, "span": span}
        if candidate not in ordered_items:
            ordered_items.append(candidate)
        sorted_ordered_items = sorted(ordered_items, key=lambda x: (x["span"][0], x["span"][1]))
        dn_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "DN"]
        od_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "OD"]
        inch_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "INCH"]
        length_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "LENGTH"]
        consumed_spans = self.size_processor._derive_consumed_spans_from_ordered_items(normalized, sorted_ordered_items)

        code_values = dn_values if dn_values else []
        size_code = (
            self.size_processor.format_code([float(v) for v in code_values if re.fullmatch(r'\d+(?:\.\d+)?', v)])
            if code_values else ""
        )
        if length_values:
            size_code = self.size_processor._append_length_suffix(size_code, self.size_processor._extract_length_prefix(length_values[0]))

        return RuleSizeExtraction(
            dn=dn_values,
            od=od_values,
            inch=inch_values,
            length=length_values,
            size_code=size_code,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
            consumed_spans=consumed_spans,
            ordered_items=[
                {"type": item["type"], "value": item["value"]}
                for item in sorted_ordered_items
            ],
        )

    def apply_thickness_decimal_mm_fallback(
        self,
        text: str,
        thickness_result: RuleThicknessExtraction,
        *,
        blocked_spans: Optional[list[Tuple[int, int]]] = None,
    ) -> RuleThicknessExtraction:
        if thickness_result.schedule or thickness_result.mm:
            return thickness_result

        source = str(text or "")
        normalized = source.replace('”', '"').replace('“', '"').replace('″', '"')
        blocked_spans = blocked_spans or []

        def _overlaps_blocked(span: Tuple[int, int]) -> bool:
            for start, end in blocked_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        for match in self.DECIMAL_MM_FALLBACK_PATTERN.finditer(normalized):
            span = (match.start(), match.end())
            value_span = match.span(1)
            if _overlaps_blocked(value_span):
                continue
            try:
                numeric_value = float(match.group(1))
            except ValueError:
                continue
            if not (1 < numeric_value < 20):
                continue

            value = f"{self.thickness_processor._normalize_number(match.group(1))}MM"
            return RuleThicknessExtraction(
                schedule=[],
                mm=[value],
                thickness_code=value,
                matched_texts=[match.group(0)],
                matched_spans=[span],
                consumed_spans=[value_span],
                ordered_items=[{"type": "MM", "value": self.thickness_processor._normalize_number(match.group(1))}],
            )

        return thickness_result


_weak_fallback_processor_instance: Optional[WeakFallbackProcessor] = None


def get_weak_fallback_processor() -> WeakFallbackProcessor:
    global _weak_fallback_processor_instance
    if _weak_fallback_processor_instance is None:
        _weak_fallback_processor_instance = WeakFallbackProcessor()
    return _weak_fallback_processor_instance
