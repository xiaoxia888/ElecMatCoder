# -*- coding: utf-8 -*-
"""Evidence matcher for pressure second-pass checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .models import PressureSecondPassItem, PressureSurfaceHit

TOKEN_HEAD = r"(?:(?<=^)|(?<=[^A-Z0-9])|(?<=[xX×*/-]))"
TOKEN_TAIL = r"(?=\s*(?:$|[xX×*/-]|[^A-Z0-9]))"


@dataclass
class ParsedPressureItem:
    field: str
    raw: str
    value: str = ""
    anchored_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    bare_values: list[str] = field(default_factory=list)

    def to_result_item(self) -> PressureSecondPassItem:
        return PressureSecondPassItem(field=self.field, raw=self.raw, value=self.value)


class PressureSurfaceMatcher:
    def parse_pressure_items(self, pressure_result: object, pressure_code: str = "") -> list[ParsedPressureItem]:
        values = self._normalize_values(pressure_result)
        code = str(pressure_code or "").strip().upper()
        if not values and code:
            values = [code]
        items: list[ParsedPressureItem] = []
        for value in values:
            item = self._parse_single_item(value)
            if item is not None:
                items.append(item)
        return items

    def match_anchored(self, text: str, item: ParsedPressureItem) -> list[PressureSurfaceHit]:
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        hits: list[PressureSurfaceHit] = []
        for alias, pattern in item.anchored_patterns:
            for match in pattern.finditer(upper_text):
                group_index = 1 if pattern.groups >= 1 else 0
                start = match.start(group_index)
                end = match.end(group_index)
                hits.append(
                    PressureSurfaceHit(
                        field=item.field,
                        alias=alias,
                        start=start,
                        end=end,
                        text=raw_text[start:end],
                        kind="anchored",
                    )
                )
        return self._dedupe_hits(hits)

    def find_first_anchored_hit(
        self,
        text: str,
        item: ParsedPressureItem,
        *,
        consumed_spans: list[tuple[int, int]] | None = None,
    ) -> PressureSurfaceHit | None:
        consumed = list(consumed_spans or [])
        for hit in self.match_anchored(text, item):
            if self._overlaps_consumed(hit.start, hit.end, consumed):
                continue
            return hit
        return None

    def find_first_bare_hit(
        self,
        text: str,
        item: ParsedPressureItem,
        *,
        consumed_spans: list[tuple[int, int]] | None = None,
    ) -> PressureSurfaceHit | None:
        consumed = list(consumed_spans or [])
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        for value in item.bare_values:
            pattern = re.compile(rf'(?<!\d)({re.escape(value)})(?!\d)', re.IGNORECASE)
            for match in pattern.finditer(upper_text):
                start = match.start(1)
                end = match.end(1)
                if self._overlaps_consumed(start, end, consumed):
                    continue
                return PressureSurfaceHit(
                    field=item.field,
                    alias=value,
                    start=start,
                    end=end,
                    text=raw_text[start:end],
                    kind="fallback",
                )
        return None

    def allocate_anchored_hits(
        self,
        text: str,
        items: list[ParsedPressureItem],
        *,
        consumed_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[list[PressureSurfaceHit], list[ParsedPressureItem], list[tuple[int, int]]]:
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        consumed = list(consumed_spans or [])
        allocated: list[PressureSurfaceHit] = []
        unmatched: list[ParsedPressureItem] = []
        for item in items:
            anchored_hit = self.find_first_anchored_hit(raw_text, item, consumed_spans=consumed)
            if anchored_hit is not None:
                allocated.append(anchored_hit)
                consumed.append((anchored_hit.start, anchored_hit.end))
                continue
            fallback_hit = self.find_first_bare_hit(raw_text, item, consumed_spans=consumed)
            if fallback_hit is None:
                unmatched.append(item)
                continue
            allocated.append(fallback_hit)
            consumed.append((fallback_hit.start, fallback_hit.end))
        return allocated, unmatched, consumed

    def _parse_single_item(self, value: str) -> ParsedPressureItem | None:
        text = str(value or "").strip().upper()
        if not text:
            return None

        # PN series
        match = re.fullmatch(r'PN\s*(\d+(?:\.\d+)?)', text)
        if match:
            number = match.group(1)
            return ParsedPressureItem(
                field="PN",
                raw=text,
                value=f"PN{number}",
                anchored_patterns=self._build_pn_patterns(number),
                bare_values=[number],
            )

        # CL series from result, e.g. CL3000
        match = re.fullmatch(r'CL\s*(\d+)', text)
        if match:
            number = match.group(1)
            return ParsedPressureItem(
                field="CLASS",
                raw=text,
                value=f"C{number}",
                anchored_patterns=self._build_class_patterns(number),
                bare_values=[number],
            )

        # Encoded class, e.g. C3000 / C150
        match = re.fullmatch(r'C(\d+)', text)
        if match:
            number = match.group(1)
            return ParsedPressureItem(
                field="CLASS",
                raw=text,
                value=f"C{number}",
                anchored_patterns=self._build_class_patterns(number),
                bare_values=[number],
            )
        return None

    @staticmethod
    def _build_pn_patterns(number: str) -> list[tuple[str, re.Pattern[str]]]:
        return [
            (f"PN{number}", re.compile(rf'{TOKEN_HEAD}(PN\s*{re.escape(number)}){TOKEN_TAIL}', re.IGNORECASE)),
        ]

    @staticmethod
    def _build_class_patterns(number: str) -> list[tuple[str, re.Pattern[str]]]:
        return [
            (f"CL{number}", re.compile(rf'{TOKEN_HEAD}(CL\s*\.?\s*{re.escape(number)}){TOKEN_TAIL}', re.IGNORECASE)),
            (f"CLASS{number}", re.compile(rf'{TOKEN_HEAD}(CLASS\s*{re.escape(number)}){TOKEN_TAIL}', re.IGNORECASE)),
            (f"{number}LB", re.compile(rf'{TOKEN_HEAD}({re.escape(number)}\s*LBS?){TOKEN_TAIL}', re.IGNORECASE)),
            (f"{number}#", re.compile(rf'{TOKEN_HEAD}({re.escape(number)}\s*#){TOKEN_TAIL}', re.IGNORECASE)),
        ]

    @staticmethod
    def _normalize_values(value: object) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(item or "").strip() for item in value if str(item or "").strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _dedupe_hits(hits: list[PressureSurfaceHit]) -> list[PressureSurfaceHit]:
        dedup: dict[tuple[str, str, int, int], PressureSurfaceHit] = {}
        for hit in hits:
            dedup[(hit.field, hit.alias.upper(), hit.start, hit.end)] = hit
        result = list(dedup.values())
        result.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias.upper()))
        return result

    @staticmethod
    def _overlaps_consumed(start: int, end: int, consumed_spans: list[tuple[int, int]]) -> bool:
        for consumed_start, consumed_end in consumed_spans:
            if start < consumed_end and end > consumed_start:
                return True
        return False
