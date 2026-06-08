# -*- coding: utf-8 -*-
"""Second-pass thickness evidence consumer."""

from __future__ import annotations

from typing import Any
from typing import Iterable

from .models import ThicknessSecondPassResult
from .thickness_surface_matcher import ThicknessSurfaceMatcher


class ThicknessSecondPassSplitter:
    def __init__(self, matcher: ThicknessSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or ThicknessSurfaceMatcher()

    @staticmethod
    def _dedupe_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for span in spans:
            if span not in result:
                result.append(span)
        return result

    @staticmethod
    def _mm_has_too_many_decimals(value: str) -> bool:
        text = str(value or "").strip()
        if "." not in text:
            return False
        decimal = text.split(".", 1)[1].rstrip("0")
        return len(decimal) > 1

    def _render_thickness_result(self, thickness_result: Any) -> str:
        texts = self.matcher._normalize_values(thickness_result)
        return " ; ".join(texts)

    def analyze(
        self,
        text: str,
        thickness_result: object,
        thickness_code: str = "",
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> ThicknessSecondPassResult:
        clean_text = str(text or "").strip()
        clean_result = self._render_thickness_result(thickness_result)
        clean_code = str(thickness_code or "").strip().upper()
        initial_consumed = list(consumed_spans)
        normalized_result = self.matcher._normalize_values(thickness_result)
        if not clean_text:
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason="描述为空",
                consumed_spans=initial_consumed,
            )
        if not normalized_result and not clean_code:
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason="壁厚结果为空",
                consumed_spans=initial_consumed,
            )

        items = self.matcher.parse_thickness_items(thickness_result, clean_code)
        if not items:
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason="无法解析壁厚结果",
                consumed_spans=initial_consumed,
            )

        if len(items) >= 3:
            item_text = " | ".join(item.raw for item in items)
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason=f"存在3个及以上壁厚项: {item_text}",
                items=[item.to_result_item() for item in items],
                consumed_spans=initial_consumed,
            )

        invalid_mm_items = [item for item in items if item.field == "MM" and self._mm_has_too_many_decimals(item.value)]
        if invalid_mm_items:
            invalid_text = " | ".join(item.raw for item in invalid_mm_items)
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason=f"壁厚小数位超过1位: {invalid_text}",
                items=[item.to_result_item() for item in items],
                unmatched_items=[item.to_result_item() for item in invalid_mm_items],
                consumed_spans=initial_consumed,
            )

        anchored_hits, fallback_hits, unmatched_items, final_consumed = self.matcher.allocate_anchored_hits(
            clean_text,
            items,
            consumed_spans=initial_consumed,
        )

        result_items = [item.to_result_item() for item in items]
        unmatched_result_items = [item.to_result_item() for item in unmatched_items]
        final_consumed = self._dedupe_spans(final_consumed)
        if unmatched_items:
            missing = " | ".join(item.raw for item in unmatched_items)
            return ThicknessSecondPassResult(
                text=clean_text,
                thickness_result=clean_result,
                thickness_code=clean_code,
                passed=False,
                reason=f"未命中壁厚证据: {missing}",
                items=result_items,
                anchored_hits=anchored_hits,
                fallback_hits=fallback_hits,
                unmatched_items=unmatched_result_items,
                consumed_spans=final_consumed,
                fallback_used=bool(fallback_hits),
            )

        reason = "命中壁厚锚点表达"
        if fallback_hits:
            reason += "，并使用裸值兜底"
        return ThicknessSecondPassResult(
            text=clean_text,
            thickness_result=clean_result,
            thickness_code=clean_code,
            passed=True,
            reason=reason,
            items=result_items,
            anchored_hits=anchored_hits,
            fallback_hits=fallback_hits,
            unmatched_items=[],
            consumed_spans=final_consumed,
            fallback_used=bool(fallback_hits),
        )
