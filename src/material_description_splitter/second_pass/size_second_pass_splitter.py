# -*- coding: utf-8 -*-
"""Second-pass size evidence consumer."""

from __future__ import annotations

import re
from typing import Any

from .models import SizeSecondPassResult
from .size_surface_matcher import ParsedSizeItem, SizeSurfaceMatcher


class SizeSecondPassSplitter:
    def __init__(self, matcher: SizeSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or SizeSurfaceMatcher()

    @staticmethod
    def _dedupe_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for span in spans:
            if span not in result:
                result.append(span)
        return result

    @staticmethod
    def _find_invalid_dn_decimal_items(items: list[ParsedSizeItem]) -> list[ParsedSizeItem]:
        invalid_items: list[ParsedSizeItem] = []
        for item in items:
            if item.field != "DN":
                continue
            values = item.values or ([item.value] if item.value else [])
            if any("." in str(value or "") for value in values):
                invalid_items.append(item)
        return invalid_items

    def _render_size_result(self, size_result: Any) -> str:
        texts = self.matcher._expand_texts(self.matcher._normalize_size_result(size_result))
        return " ; ".join(texts)

    def _find_invalid_dn_decimal_texts(self, size_result: object) -> list[str]:
        invalid_texts: list[str] = []
        for text in self.matcher._normalize_size_result(size_result):
            if "DN" not in str(text or "").upper():
                continue
            matches = re.findall(
                r"(?i)\bDN\s*:?\s*([0-9]+(?:\.[0-9]+)?(?:\s*[xX×*]\s*[0-9]+(?:\.[0-9]+)?)*)",
                text,
            )
            for value in matches:
                parts = [part.strip() for part in re.split(r"\s*[xX×*]\s*", value) if part.strip()]
                if any("." in self.matcher._clean_number(part) for part in parts):
                    invalid_texts.append(str(text).strip())
                    break
        return invalid_texts

    def analyze(self, text: str, size_result: object, size_code: str = "") -> SizeSecondPassResult:
        clean_text = str(text or "").strip()
        clean_result = self._render_size_result(size_result)
        clean_code = str(size_code or "").strip().upper()
        normalized_result = self.matcher._normalize_size_result(size_result)
        if not clean_text:
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason="描述为空",
            )
        if not normalized_result and not clean_code:
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason="尺寸结果为空",
            )

        invalid_dn_texts = self._find_invalid_dn_decimal_texts(clean_result)
        if invalid_dn_texts:
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason=f"DN尺寸不能为小数: {' | '.join(invalid_dn_texts)}",
            )

        items = self.matcher.parse_size_items(size_result, clean_code)
        if not items:
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason="无法解析尺寸结果",
            )

        invalid_dn_items = self._find_invalid_dn_decimal_items(items)
        if invalid_dn_items:
            invalid_text = " | ".join(item.raw for item in invalid_dn_items)
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason=f"DN尺寸不能为小数: {invalid_text}",
                items=[item.to_result_item() for item in items],
                unmatched_items=[item.to_result_item() for item in invalid_dn_items],
            )

        anchored_hits = []
        fallback_hits = []
        unmatched_items: list[ParsedSizeItem] = []
        consumed_spans: list[tuple[int, int]] = []

        # 先消费显式/共享锚点证据；每个 item 只需一个命中。
        for item in items:
            hit = self.matcher.find_first_anchored_hit(
                clean_text,
                item,
                consumed_spans=consumed_spans,
            )
            if hit is not None:
                anchored_hits.append(hit)
                consumed_spans.append((hit.start, hit.end))
            else:
                unmatched_items.append(item)

        # 仅对完全没有锚点证据的 item 走裸数字兜底；每个 item 只取一个命中。
        still_unmatched: list[ParsedSizeItem] = []
        for item in unmatched_items:
            hits = self.matcher.match_bare(clean_text, item, consumed_spans=consumed_spans)
            if hits:
                hit = hits[0]
                fallback_hits.append(hit)
                consumed_spans.append((hit.start, hit.end))
            else:
                still_unmatched.append(item)

        result_items = [item.to_result_item() for item in items]
        unmatched_result_items = [item.to_result_item() for item in still_unmatched]
        consumed_spans = self._dedupe_spans(consumed_spans)
        if still_unmatched:
            missing = " | ".join(item.raw for item in still_unmatched)
            return SizeSecondPassResult(
                text=clean_text,
                size_result=clean_result,
                size_code=clean_code,
                passed=False,
                reason=f"未命中尺寸证据: {missing}",
                items=result_items,
                anchored_hits=anchored_hits,
                fallback_hits=fallback_hits,
                unmatched_items=unmatched_result_items,
                consumed_spans=consumed_spans,
                fallback_used=bool(fallback_hits),
            )

        reason = "命中尺寸锚点表达"
        if fallback_hits:
            reason += "，并使用裸数字兜底"
        return SizeSecondPassResult(
            text=clean_text,
            size_result=clean_result,
            size_code=clean_code,
            passed=True,
            reason=reason,
            items=result_items,
            anchored_hits=anchored_hits,
            fallback_hits=fallback_hits,
            unmatched_items=[],
            consumed_spans=consumed_spans,
            fallback_used=bool(fallback_hits),
        )
