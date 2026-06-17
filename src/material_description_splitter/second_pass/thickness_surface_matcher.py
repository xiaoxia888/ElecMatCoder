# -*- coding: utf-8 -*-
"""Evidence matcher for thickness second-pass checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from .models import ThicknessSecondPassItem, ThicknessSurfaceHit


X_SEP = r"(?:\s*[xX×*]\s*)"
TOKEN_HEAD = r"(?:(?<=^)|(?<=[^A-Z0-9])|(?<=[xX×*/]))"
TOKEN_TAIL = r"(?=\s*(?:$|[xX×*/]|[^A-Z0-9]))"


@dataclass
class ParsedThicknessItem:
    field: str
    raw: str
    value: str = ""
    values: list[str] = field(default_factory=list)
    anchored_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    bare_values: list[str] = field(default_factory=list)

    def to_result_item(self) -> ThicknessSecondPassItem:
        return ThicknessSecondPassItem(
            field=self.field,
            raw=self.raw,
            value=self.value,
            values=list(self.values),
        )


class ThicknessSurfaceMatcher:
    def parse_thickness_items(self, thickness_result: object, thickness_code: str = "") -> list[ParsedThicknessItem]:
        texts = self._expand_texts(self._normalize_values(thickness_result))
        if not texts:
            return []

        items: list[ParsedThicknessItem] = []
        for text in texts:
            items.extend(self._extract_schedule_items(text))
            items.extend(self._extract_mm_items(text))
        return items

    def allocate_anchored_hits(
        self,
        text: str,
        items: list[ParsedThicknessItem],
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> tuple[list[ThicknessSurfaceHit], list[ThicknessSurfaceHit], list[ParsedThicknessItem], list[tuple[int, int]]]:
        anchored_allocated: list[ThicknessSurfaceHit] = []
        fallback_allocated: list[ThicknessSurfaceHit] = []
        unmatched: list[ParsedThicknessItem] = []
        consumed = list(consumed_spans)

        for item in items:
            anchored_hit = self.find_first_anchored_hit(text, item, consumed_spans=consumed)
            if anchored_hit is not None:
                anchored_allocated.append(anchored_hit)
                consumed.append((anchored_hit.start, anchored_hit.end))
                continue

            fallback_hit = self.find_first_bare_hit(text, item, consumed_spans=consumed)
            if fallback_hit is None:
                unmatched.append(item)
                continue
            fallback_allocated.append(fallback_hit)
            consumed.append((fallback_hit.start, fallback_hit.end))
        return anchored_allocated, fallback_allocated, unmatched, consumed

    def find_first_anchored_hit(
        self,
        text: str,
        item: ParsedThicknessItem,
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> ThicknessSurfaceHit | None:
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        consumed = list(consumed_spans)
        candidates: list[ThicknessSurfaceHit] = []
        for alias, pattern in item.anchored_patterns:
            for match in pattern.finditer(upper_text):
                group_index = 1 if pattern.groups >= 1 else 0
                start = match.start(group_index)
                end = match.end(group_index)
                if self._overlaps_consumed(start, end, consumed):
                    continue
                candidates.append(
                    ThicknessSurfaceHit(
                        field=item.field,
                        alias=alias,
                        start=start,
                        end=end,
                        text=raw_text[start:end],
                        kind="anchored",
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias.upper()))
        return candidates[0]

    def find_first_bare_hit(
        self,
        text: str,
        item: ParsedThicknessItem,
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> ThicknessSurfaceHit | None:
        consumed = list(consumed_spans)
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        for value in item.bare_values:
            # Prevent integer items like `5` from matching the tail of decimal values like `4.5`.
            pattern = re.compile(rf'(?<![\d.])({re.escape(value)})(?![\d.])', re.IGNORECASE)
            for match in pattern.finditer(upper_text):
                start = match.start(1)
                end = match.end(1)
                if self._overlaps_consumed(start, end, consumed):
                    continue
                return ThicknessSurfaceHit(
                    field=item.field,
                    alias=value,
                    start=start,
                    end=end,
                    text=raw_text[start:end],
                    kind="fallback",
                )
        return None

    def _extract_schedule_items(self, text: str) -> list[ParsedThicknessItem]:
        payload = self._strip_head_label(text, "SCHEDULE")
        if "SCH" not in payload.upper() and "S" not in payload.upper():
            return []

        items: list[ParsedThicknessItem] = []
        schedule_token_re = re.compile(
            rf'{TOKEN_HEAD}(XXS|XS|STD|(?:SCH\s*-?|S\s*-?)\s*\d+\s*S?){TOKEN_TAIL}',
            re.IGNORECASE,
        )
        for match in schedule_token_re.finditer(payload.upper()):
            piece = match.group(1)
            norm = self._normalize_schedule(piece)
            if not norm:
                continue
            items.append(
                ParsedThicknessItem(
                    field="SCHEDULE",
                    raw=piece,
                    value=norm,
                    values=[norm],
                    anchored_patterns=self._build_schedule_patterns(norm),
                    bare_values=[] if norm in {"STD", "XS", "XXS"} else [re.sub(r'[^0-9.]', '', norm)],
                )
            )
        return items

    def _extract_mm_items(self, text: str) -> list[ParsedThicknessItem]:
        raw_text = str(text or "")
        text_upper = raw_text.upper()
        if not (
            re.match(r'(?i)^\s*(MM|THK)\s*:', raw_text)
            or "MM" in text_upper
            or "THK" in text_upper
        ):
            return []
        payload = self._strip_head_label(text, "MM")
        payload = self._strip_head_label(payload, "THK")
        pieces = [p.strip() for p in re.split(X_SEP, payload) if p.strip()]
        if not pieces:
            pieces = [payload.strip()]

        items: list[ParsedThicknessItem] = []
        for piece in pieces:
            value = self._extract_number(piece)
            if not value:
                continue
            items.append(
                ParsedThicknessItem(
                    field="MM",
                    raw=piece,
                    value=value,
                    values=[value],
                    anchored_patterns=self._build_mm_patterns(value),
                    bare_values=[value],
                )
            )
        return items

    @staticmethod
    def _normalize_values(value: object) -> list[str]:
        if isinstance(value, dict):
            items = value.get("_ITEMS")
            if isinstance(items, list) and items:
                result: list[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").strip().upper()
                    item_value = str(item.get("value") or "").strip()
                    if not item_type or not item_value:
                        continue
                    result.append(f"{item_type}: {item_value}")
                if result:
                    return result

            result = []
            for key in ("MM", "INCH", "SCHEDULE", "SERIES", "BWG"):
                values = value.get(key)
                if isinstance(values, list):
                    cleaned = [str(item or "").strip() for item in values if str(item or "").strip()]
                    if cleaned:
                        result.append(f"{key}: {' x '.join(cleaned)}")
                elif values not in (None, "", []):
                    result.append(f"{key}: {str(values).strip()}")
            return result
        if isinstance(value, (list, tuple)):
            return [str(item or "").strip() for item in value if str(item or "").strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _expand_texts(texts: list[str]) -> list[str]:
        expanded: list[str] = []
        for text in texts:
            raw = str(text or "").strip()
            if not raw:
                continue
            for chunk in [part.strip() for part in raw.split(";") if part.strip()]:
                slash_parts = [part.strip() for part in re.split(r"\s*/\s*", chunk) if part.strip()]
                expanded.extend(slash_parts or [chunk])
        return expanded

    @staticmethod
    def _strip_head_label(text: str, label: str) -> str:
        pattern = re.compile(rf'^\s*{re.escape(label)}\s*:\s*', re.IGNORECASE)
        return pattern.sub("", str(text or "").strip())

    @staticmethod
    def _normalize_schedule(value: str) -> str:
        text = str(value or "").strip().upper()
        text = re.sub(r"\s+", "", text)
        text = text.replace("SCHEDULE", "").replace("SCH", "S")
        if not text:
            return ""
        if text in {"STD", "XS", "XXS"}:
            return text
        if re.fullmatch(r"S\d+(?:S)?", text):
            return text
        return ""

    @staticmethod
    def _extract_number(value: str) -> str:
        match = re.search(r'(\d+(?:\.\d+)?)', str(value or ""))
        if not match:
            return ""
        return match.group(1)

    @staticmethod
    def _build_schedule_patterns(norm: str) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        if norm in {"STD", "XS", "XXS"}:
            patterns.append(
                (
                    norm,
                    re.compile(
                        rf'{TOKEN_HEAD}({re.escape(norm)}){TOKEN_TAIL}',
                        re.IGNORECASE,
                    ),
                )
            )
            return patterns

        match = re.fullmatch(r'S(\d+)(S?)', norm)
        if not match:
            return patterns
        number = match.group(1)
        suffix = match.group(2)
        if suffix:
            patterns.append(
                (
                    norm,
                    re.compile(
                        rf'{TOKEN_HEAD}((?:SCH\s*-?|S\s*-?)\s*{re.escape(number)}\s*S){TOKEN_TAIL}',
                        re.IGNORECASE,
                    ),
                )
            )
        else:
            patterns.append(
                (
                    norm,
                    re.compile(
                        rf'{TOKEN_HEAD}((?:SCH\s*-?|S\s*-?)\s*{re.escape(number)}){TOKEN_TAIL}',
                        re.IGNORECASE,
                    ),
                )
            )
        return patterns

    @staticmethod
    def _build_mm_patterns(value: str) -> list[tuple[str, re.Pattern[str]]]:
        number_pattern = ThicknessSurfaceMatcher._build_equivalent_numeric_pattern(value)
        return [
            # 数值等价匹配：8 可命中 8mm / 8.0mm / 8.00mm，
            # 但仍不能误命中 18mm 或 4.8mm 的尾部。
            (f"{value}MM", re.compile(rf'(?<![\d.])(({number_pattern})\s*MM)(?![A-Z0-9.])', re.IGNORECASE)),
            (f"{value}THK", re.compile(rf'(?<![A-Z0-9])(THK\s*=?\s*({number_pattern}))(?!\d)', re.IGNORECASE)),
        ]

    @staticmethod
    def _build_equivalent_numeric_pattern(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return re.escape(text)

        if "." not in text:
            return rf"{re.escape(text)}(?:\.0+)?"

        integer, decimal = text.split(".", 1)
        decimal = decimal.rstrip("0")
        if not decimal:
            return rf"{re.escape(integer)}(?:\.0+)?"
        return rf"{re.escape(integer)}\.{re.escape(decimal)}0*"

    @staticmethod
    def _overlaps_consumed(start: int, end: int, consumed_spans: list[tuple[int, int]]) -> bool:
        for consumed_start, consumed_end in consumed_spans:
            if start < consumed_end and end > consumed_start:
                return True
        return False
