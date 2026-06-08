# -*- coding: utf-8 -*-
"""Second-pass pressure evidence consumer."""

from __future__ import annotations

from typing import Iterable

from .models import PressureSecondPassResult
from .pressure_surface_matcher import PressureSurfaceMatcher


class PressureSecondPassSplitter:
    def __init__(self, matcher: PressureSurfaceMatcher | None = None) -> None:
        self.matcher = matcher or PressureSurfaceMatcher()

    @staticmethod
    def _dedupe_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for span in spans:
            if span not in result:
                result.append(span)
        return result

    def analyze(
        self,
        text: str,
        pressure_result: object,
        pressure_code: str = "",
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> PressureSecondPassResult:
        clean_text = str(text or "").strip()
        clean_result = str(pressure_result or "").strip() if not isinstance(pressure_result, (list, tuple)) else ""
        clean_code = str(pressure_code or "").strip().upper()
        initial_consumed = list(consumed_spans)
        if not clean_text:
            return PressureSecondPassResult(
                text=clean_text,
                pressure_result=clean_result,
                pressure_code=clean_code,
                passed=False,
                reason="描述为空",
                consumed_spans=initial_consumed,
            )
        if not pressure_result and not clean_code:
            return PressureSecondPassResult(
                text=clean_text,
                pressure_result=clean_result,
                pressure_code=clean_code,
                passed=False,
                reason="磅级结果为空",
                consumed_spans=initial_consumed,
            )

        items = self.matcher.parse_pressure_items(pressure_result, clean_code)
        if not items:
            return PressureSecondPassResult(
                text=clean_text,
                pressure_result=clean_result,
                pressure_code=clean_code,
                passed=False,
                reason="无法解析磅级结果",
                consumed_spans=initial_consumed,
            )

        anchored_hits, unmatched_items, final_consumed = self.matcher.allocate_anchored_hits(
            clean_text,
            items,
            consumed_spans=initial_consumed,
        )

        result_items = [item.to_result_item() for item in items]
        unmatched_result_items = [item.to_result_item() for item in unmatched_items]
        final_consumed = self._dedupe_spans(final_consumed)
        if unmatched_items:
            missing = " | ".join(item.raw for item in unmatched_items)
            return PressureSecondPassResult(
                text=clean_text,
                pressure_result=clean_result,
                pressure_code=clean_code,
                passed=False,
                reason=f"未命中磅级证据: {missing}",
                items=result_items,
                anchored_hits=anchored_hits,
                unmatched_items=unmatched_result_items,
                consumed_spans=final_consumed,
            )

        return PressureSecondPassResult(
            text=clean_text,
            pressure_result=clean_result,
            pressure_code=clean_code,
            passed=True,
            reason="命中磅级锚点表达",
            items=result_items,
            anchored_hits=anchored_hits,
            unmatched_items=[],
            consumed_spans=final_consumed,
        )
