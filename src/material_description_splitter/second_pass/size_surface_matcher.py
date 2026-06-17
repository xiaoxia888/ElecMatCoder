# -*- coding: utf-8 -*-
"""Evidence matcher for size second-pass checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from .models import SizeSecondPassItem, SizeSurfaceHit


X_SEP = r"(?:\s*[xX×*]\s*)"
TOKEN_HEAD = r"(?:(?<=^)|(?<=[^A-Z0-9])|(?<=[xX×*/]))"
TOKEN_TAIL = r"(?=\s*(?:$|[xX×*/]|[^A-Z0-9]))"
INCH_TOKEN_RE = r'(?:\d+\s*-\s*\d+/\d+|\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)'


@dataclass
class ParsedSizeItem:
    field: str
    raw: str
    value: str = ""
    values: list[str] = field(default_factory=list)
    anchored_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    bare_values: list[str] = field(default_factory=list)

    def to_result_item(self) -> SizeSecondPassItem:
        return SizeSecondPassItem(
            field=self.field,
            raw=self.raw,
            value=self.value,
            values=list(self.values),
        )


class SizeSurfaceMatcher:
    def parse_size_items(self, size_result: object, size_code: str = "") -> list[ParsedSizeItem]:
        structured_items = self._extract_structured_items(size_result)
        if structured_items:
            return self._dedupe_items(structured_items)

        texts = self._expand_texts(self._normalize_size_result(size_result))
        if not texts:
            return []

        items: list[ParsedSizeItem] = []
        for text in texts:
            # 1. Composite DN expressions: `DN: 350 x 20`, `DN350X20`.
            items.extend(self._extract_dn_composites(text))

            # 2. Composite inch expressions: `16"x 1"`, `2x1"`.
            items.extend(self._extract_inch_composites(text))

            # 3. Anchored DN/OD/LENGTH/single-inch tokens.
            items.extend(self._extract_dn_items(text))
            items.extend(self._extract_od_items(text))
            items.extend(self._extract_length_items(text))
            items.extend(self._extract_single_inch_items(text))

        deduped = self._dedupe_items(items)
        if deduped:
            return deduped

        # 4. Bare numeric fallback only when no anchored item can be parsed at all.
        fallback_items: list[ParsedSizeItem] = []
        for text in texts:
            fallback_items.extend(self._extract_bare_items(text))
        return self._dedupe_items(fallback_items)

    def _extract_structured_items(self, size_result: object) -> list[ParsedSizeItem]:
        if not isinstance(size_result, dict):
            return []

        items: list[ParsedSizeItem] = []

        dn_values = [self._clean_number(v) for v in self._ensure_list(size_result.get("DN")) if self._clean_number(v)]
        od_values = [self._clean_number(v) for v in self._ensure_list(size_result.get("OD")) if self._clean_number(v)]
        inch_values = [self._normalize_inch_value(v) for v in self._ensure_list(size_result.get("INCH")) if self._normalize_inch_value(v)]
        length_values = [self._clean_number(v) for v in self._ensure_list(size_result.get("LENGTH")) if self._clean_number(v)]

        if len(dn_values) >= 2:
            left, right = dn_values[0], dn_values[1]
            if "." not in left:
                items.append(self._build_dn_composite_item(left, right, is_left=True))
            if "." not in right:
                items.append(self._build_dn_composite_item(left, right, is_left=False))
        else:
            for value in dn_values:
                items.append(
                    ParsedSizeItem(
                        field="DN",
                        raw=f"DN: {value}",
                        value=value,
                        values=[value],
                        anchored_patterns=[(f"DN{value}", self._compile_dn_pattern(value))],
                        bare_values=[value],
                    )
                )

        for value in od_values:
            items.append(
                ParsedSizeItem(
                    field="OD",
                    raw=f"OD: {value}",
                    value=value,
                    values=[value],
                    anchored_patterns=[(f"OD{value}", self._compile_od_pattern(value))],
                    bare_values=[value],
                )
            )

        if len(inch_values) >= 2:
            left, right = inch_values[0], inch_values[1]
            raw = f"INCH: {left} x {right}"
            items.append(self._build_inch_composite_item(raw, left, right, is_left=True))
            items.append(self._build_inch_composite_item(raw, left, right, is_left=False))
        else:
            for value in inch_values:
                raw = f"INCH: {value} in"
                items.append(
                    ParsedSizeItem(
                        field="INCH",
                        raw=raw,
                        value=value,
                        values=[value],
                        anchored_patterns=self._build_single_inch_patterns(raw, value),
                        bare_values=[value],
                    )
                )

        for value in length_values:
            items.append(
                ParsedSizeItem(
                    field="LENGTH",
                    raw=f"LENGTH: {value}MM",
                    value=value,
                    values=[value],
                    anchored_patterns=self._build_length_patterns(value),
                    bare_values=self._build_length_fallback_values(value),
                )
            )

        if items:
            return items

        raw_items = size_result.get("_ITEMS")
        if not isinstance(raw_items, list):
            return []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().upper()
            item_value = str(item.get("value") or "").strip()
            if not item_type or not item_value:
                continue
            if item_type == "DN":
                value = self._clean_number(item_value)
                if value and "." not in value:
                    items.append(
                        ParsedSizeItem(
                            field="DN",
                            raw=f"DN: {value}",
                            value=value,
                            values=[value],
                            anchored_patterns=[(f"DN{value}", self._compile_dn_pattern(value))],
                            bare_values=[value],
                        )
                    )
            elif item_type == "OD":
                value = self._clean_number(item_value)
                if value:
                    items.append(
                        ParsedSizeItem(
                            field="OD",
                            raw=f"OD: {value}",
                            value=value,
                            values=[value],
                            anchored_patterns=[(f"OD{value}", self._compile_od_pattern(value))],
                            bare_values=[value],
                        )
                    )
            elif item_type == "INCH":
                value = self._normalize_inch_value(item_value)
                if value:
                    raw = f"INCH: {value} in"
                    items.append(
                        ParsedSizeItem(
                            field="INCH",
                            raw=raw,
                            value=value,
                            values=[value],
                            anchored_patterns=self._build_single_inch_patterns(raw, value),
                            bare_values=[value],
                        )
                    )
            elif item_type == "LENGTH":
                value = self._clean_number(item_value)
                if value:
                    items.append(
                        ParsedSizeItem(
                            field="LENGTH",
                            raw=f"LENGTH: {value}MM",
                            value=value,
                            values=[value],
                            anchored_patterns=self._build_length_patterns(value),
                            bare_values=self._build_length_fallback_values(value),
                        )
                    )
        return items

    def match_anchored(self, text: str, item: ParsedSizeItem) -> list[SizeSurfaceHit]:
        hits: list[SizeSurfaceHit] = []
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        for alias, pattern in item.anchored_patterns:
            for match in pattern.finditer(upper_text):
                group_index = 1 if pattern.groups >= 1 else 0
                start = match.start(group_index)
                end = match.end(group_index)
                hits.append(
                    SizeSurfaceHit(
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
        item: ParsedSizeItem,
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> SizeSurfaceHit | None:
        consumed = list(consumed_spans)
        for hit in self.match_anchored(text, item):
            if self._overlaps_consumed(hit.start, hit.end, consumed):
                continue
            return hit
        return None

    def match_bare(
        self,
        text: str,
        item: ParsedSizeItem,
        *,
        consumed_spans: Iterable[tuple[int, int]] = (),
    ) -> list[SizeSurfaceHit]:
        consumed = list(consumed_spans)
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        for value in item.bare_values:
            pattern = self._compile_bare_number_pattern(value)
            for match in pattern.finditer(upper_text):
                start = match.start(1)
                end = match.end(1)
                if self._overlaps_consumed(start, end, consumed):
                    continue
                return [
                    SizeSurfaceHit(
                        field=item.field,
                        alias=value,
                        start=start,
                        end=end,
                        text=raw_text[start:end],
                        kind="fallback",
                    )
                ]
        return []

    def _extract_inch_composites(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        payload = self._strip_head_label(text, "INCH")
        payload_upper = payload.upper()
        labeled_as_inch = bool(re.match(r'(?i)^\s*INCH\s*:', str(text or "")))
        if not re.search(r'[\"″]|\bIN(?:CH)?\b', payload_upper) and not labeled_as_inch:
            return []

        composite_pattern = re.compile(
            rf'({INCH_TOKEN_RE}\s*(?:["″]|\bIN(?:CH)?\b)?\s*[xX×*]\s*{INCH_TOKEN_RE}\s*(?:["″]|\bIN(?:CH)?\b)?)',
            re.IGNORECASE,
        )
        for match in composite_pattern.finditer(payload):
            raw = match.group(1).strip()
            values = [
                self._normalize_inch_value(part)
                for part in re.split(r'[xX×*]', raw)
                if self._normalize_inch_value(part)
            ]
            if len(values) < 2:
                continue
            left, right = values[0], values[1]
            items.append(self._build_inch_composite_item(raw, left, right, is_left=True))
            items.append(self._build_inch_composite_item(raw, left, right, is_left=False))
        return items

    def _extract_dn_composites(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        payload = self._strip_head_label(text, "DN")
        payload_upper = payload.upper()
        if "DN" not in str(text or "").upper() and not re.search(X_SEP, payload_upper):
            return items

        composite_patterns = [
            re.compile(rf'(?i)\bDN\s*:?\s*(\d+(?:\.\d+)?)\s*{X_SEP}\s*(\d+(?:\.\d+)?)\b'),
            re.compile(rf'(?i)\b(\d+(?:\.\d+)?)\s*{X_SEP}\s*DN\s*:?\s*(\d+(?:\.\d+)?)\b'),
            re.compile(rf'(?i)^\s*(\d+(?:\.\d+)?)\s*{X_SEP}\s*(\d+(?:\.\d+)?)\s*$'),
        ]
        for pattern in composite_patterns:
            for match in pattern.finditer(text if pattern.pattern.startswith('(?i)\\bDN') else payload):
                left = self._clean_number(match.group(1))
                right = self._clean_number(match.group(2))
                if not left or not right:
                    continue
                if "." not in left:
                    items.append(self._build_dn_composite_item(left, right, is_left=True))
                if "." not in right:
                    items.append(self._build_dn_composite_item(left, right, is_left=False))
                return items
        return items

    def _extract_dn_items(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        for match in re.finditer(rf'(?i){TOKEN_HEAD}DN\s*:?\s*(\d+(?:\.\d+)?){TOKEN_TAIL}', text):
            value = self._clean_number(match.group(1))
            raw = match.group(0).strip()
            items.append(
                ParsedSizeItem(
                    field="DN",
                    raw=raw,
                    value=value,
                    values=[value],
                    anchored_patterns=[(raw, self._compile_dn_pattern(value))],
                    bare_values=[value],
                )
            )
        return items

    def _extract_od_items(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        for match in re.finditer(rf'(?i){TOKEN_HEAD}(?:OD\s*|[Φφ])(\d+(?:\.\d+)?){TOKEN_TAIL}', text):
            value = self._clean_number(match.group(1))
            raw = match.group(0).strip()
            items.append(
                ParsedSizeItem(
                    field="OD",
                    raw=raw,
                    value=value,
                    values=[value],
                    anchored_patterns=[(raw, self._compile_od_pattern(value))],
                    bare_values=[value],
                )
            )
        return items

    def _extract_length_items(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        patterns = [
            re.compile(r'(?i)\bLENGTH\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MM|毫米|CM|厘米|M|米)?\b'),
            re.compile(r'(?i)\bLEN\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MM|毫米|CM|厘米|M|米)?\b'),
            re.compile(r'(?i)(?<![A-Z0-9])L\s*=\s*(\d+(?:\.\d+)?)\s*(MM|毫米|CM|厘米|M|米)?\b'),
            re.compile(r'长度\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(MM|毫米|CM|厘米|M|米)?'),
            re.compile(r'(?i)\bCUT\s*[-]?\s*TO\s*(\d+(?:\.\d+)?)\s*(MM|毫米|CM|厘米|M|米)?\b'),
            re.compile(r'(?i)(\d+(?:\.\d+)?)\s*MM\s*LENGTH\b'),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw_value = match.group(1)
                unit = match.group(2) if pattern.groups >= 2 else "MM"
                value = self._normalize_length_value(raw_value, unit or "")
                if not value:
                    continue
                if self._is_mm_length_unit(unit or "MM") and not self._is_valid_mm_length(value):
                    continue
                raw = match.group(0).strip()
                items.append(
                    ParsedSizeItem(
                        field="LENGTH",
                        raw=raw,
                        value=value,
                        values=[value],
                        anchored_patterns=self._build_length_patterns(value),
                        bare_values=self._build_length_fallback_values(value),
                    )
                )
        return items

    def _extract_single_inch_items(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        labeled_as_inch = bool(re.match(r'(?i)^\s*INCH\s*:', str(text or "")))
        # Avoid re-parsing inch values already included in a composite payload.
        if re.search(rf'(?i){INCH_TOKEN_RE}\s*"?\s*[xX×*]\s*{INCH_TOKEN_RE}\s*"?', text):
            payload = self._strip_head_label(text, "INCH")
            if re.search(rf'(?i){INCH_TOKEN_RE}\s*"?\s*[xX×*]\s*{INCH_TOKEN_RE}\s*"?', payload):
                return items
        for match in re.finditer(rf'({INCH_TOKEN_RE})\s*(?:[\"″]|\bIN(?:CH)?\b)', text, re.IGNORECASE):
            value = self._normalize_inch_value(match.group(1))
            raw = match.group(0).strip()
            items.append(
                ParsedSizeItem(
                    field="INCH",
                    raw=raw,
                    value=value,
                    values=[value],
                    anchored_patterns=self._build_single_inch_patterns(raw, value),
                    bare_values=[value],
                )
            )
        if items or not labeled_as_inch:
            return items

        payload = self._strip_head_label(text, "INCH")
        value = self._normalize_inch_value(payload)
        if not value:
            return items
        items.append(
            ParsedSizeItem(
                field="INCH",
                raw=payload,
                value=value,
                values=[value],
                anchored_patterns=self._build_single_inch_patterns(payload, value),
                bare_values=[value],
            )
        )
        return items

    def _extract_bare_items(self, text: str) -> list[ParsedSizeItem]:
        items: list[ParsedSizeItem] = []
        for match in re.finditer(r'(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?![A-Za-z0-9.])', text):
            value = self._clean_number(match.group(1))
            raw = match.group(1)
            items.append(
                ParsedSizeItem(
                    field="BARE",
                    raw=raw,
                    value=value,
                    values=[value],
                    anchored_patterns=[],
                    bare_values=[value],
                )
            )
        return self._dedupe_items(items)

    @staticmethod
    def _strip_head_label(text: str, label: str) -> str:
        pattern = re.compile(rf'^\s*{re.escape(label)}\s*:\s*', re.IGNORECASE)
        return pattern.sub("", str(text or "").strip())

    @staticmethod
    def _expand_texts(texts: list[str]) -> list[str]:
        expanded: list[str] = []
        for text in texts:
            raw = str(text or "").strip()
            if not raw:
                continue
            parts = [part.strip() for part in raw.split(";") if part.strip()]
            if not parts:
                parts = [raw]
            for part in parts:
                slash_parts = SizeSurfaceMatcher._split_slash_size_parts(part)
                expanded.extend(slash_parts or [part])
        return expanded

    @staticmethod
    def _split_slash_size_parts(text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw or "/" not in raw:
            return [raw] if raw else []

        upper = raw.upper()
        if upper.startswith("INCH:"):
            return [raw]
        if re.fullmatch(rf'{INCH_TOKEN_RE}\s*(?:["″]|\bIN(?:CH)?\b)?', upper, re.IGNORECASE):
            return [raw]

        return [part.strip() for part in re.split(r"\s*/\s*", raw) if part.strip()]

    @staticmethod
    def _normalize_size_result(size_result: object) -> list[str]:
        if isinstance(size_result, dict):
            result = []
            for key in ("DN", "OD", "INCH", "LENGTH"):
                values = size_result.get(key)
                if isinstance(values, list):
                    cleaned = [str(item or "").strip() for item in values if str(item or "").strip()]
                    if cleaned:
                        result.append(f"{key}: {' x '.join(cleaned)}")
                elif values not in (None, "", []):
                    result.append(f"{key}: {str(values).strip()}")
            if result:
                return result

            items = size_result.get("_ITEMS")
            if isinstance(items, list) and items:
                fallback: list[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").strip().upper()
                    item_value = str(item.get("value") or "").strip()
                    if not item_type or not item_value:
                        continue
                    fallback.append(f"{item_type}: {item_value}")
                if fallback:
                    return fallback
            return result
        if isinstance(size_result, (list, tuple)):
            result = [str(item or "").strip() for item in size_result if str(item or "").strip()]
            return result
        text = str(size_result or "").strip()
        return [text] if text else []

    @staticmethod
    def _ensure_list(value: object) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, list):
            return [str(item or "").strip() for item in value if str(item or "").strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _clean_number(value: str) -> str:
        return (
            str(value or "")
            .strip()
            .upper()
            .replace(" ", "")
            .replace('"', "")
            .replace("″", "")
        )

    @staticmethod
    def _normalize_decimal_text(value: float) -> str:
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @classmethod
    def _normalize_length_value(cls, value: str, unit: str = "") -> str:
        text = cls._clean_number(value)
        if not text:
            return ""
        try:
            numeric = float(text)
        except ValueError:
            return ""
        unit_u = str(unit or "").strip().upper()
        if unit_u in ("", "MM", "毫米"):
            return cls._normalize_decimal_text(numeric)
        if unit_u in ("CM", "厘米"):
            return cls._normalize_decimal_text(numeric * 10)
        if unit_u in ("M", "米"):
            return cls._normalize_decimal_text(numeric * 1000)
        return cls._normalize_decimal_text(numeric)

    @staticmethod
    def _is_mm_length_unit(unit: str) -> bool:
        return str(unit or "").strip().upper() in ("", "MM", "毫米")

    @classmethod
    def _is_valid_mm_length(cls, value: str) -> bool:
        try:
            return float(cls._clean_number(value)) > 100
        except ValueError:
            return False

    @classmethod
    def _build_length_fallback_values(cls, value: str) -> list[str]:
        if not cls._is_valid_mm_length(value):
            return []
        return [cls._clean_number(value)]

    @classmethod
    def _build_length_patterns(cls, value: str) -> list[tuple[str, re.Pattern[str]]]:
        mm_text = cls._clean_number(value)
        if not mm_text:
            return []

        try:
            mm_value = float(mm_text)
        except ValueError:
            return []

        variants: list[tuple[str, str, str]] = []
        if cls._is_valid_mm_length(mm_text):
            variants.append((mm_text, "MM", "MM"))
            variants.append((mm_text, "毫米", "毫米"))

        cm_value = cls._normalize_decimal_text(mm_value / 10)
        variants.append((cm_value, "CM", "CM"))
        variants.append((cm_value, "厘米", "厘米"))

        m_value = cls._normalize_decimal_text(mm_value / 1000)
        variants.append((m_value, "M", "M"))
        variants.append((m_value, "米", "米"))

        patterns: list[tuple[str, re.Pattern[str]]] = []
        seen: set[tuple[str, str]] = set()
        explicit_templates = [
            r'(?i)\bLENGTH\s*[:=]?\s*({value})\s*{unit}\b',
            r'(?i)\bLEN\s*[:=]?\s*({value})\s*{unit}\b',
            r'(?i)(?<![A-Z0-9])L\s*=\s*({value})\s*{unit}\b',
            r'长度\s*[:：]?\s*({value})\s*{unit}',
            r'(?i)\bCUT\s*[-]?\s*TO\s*({value})\s*{unit}\b',
        ]
        for numeric_text, unit_text, alias_unit in variants:
            key = (numeric_text, alias_unit)
            if key in seen:
                continue
            seen.add(key)
            for template in explicit_templates:
                pattern = re.compile(template.format(value=re.escape(numeric_text), unit=re.escape(unit_text)))
                patterns.append((f"{numeric_text}{alias_unit}", pattern))
        return patterns

    @staticmethod
    def _normalize_inch_value(value: str) -> str:
        text = str(value or "").strip().upper().replace('"', "").replace("″", "")
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _inch_value_variants(value: str) -> list[str]:
        text = SizeSurfaceMatcher._normalize_inch_value(value)
        if not text:
            return []
        variants = {text}
        decimal = SizeSurfaceMatcher._inch_to_decimal(text)
        if decimal:
            variants.add(decimal)
        if "-" in text:
            variants.add(text.replace("-", " "))
        if " " in text:
            variants.add(text.replace(" ", "-"))
        return sorted(variants, key=len, reverse=True)

    @staticmethod
    def _inch_to_decimal(value: str) -> str:
        text = SizeSurfaceMatcher._normalize_inch_value(value)
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return text
        mixed = re.fullmatch(r"(\d+)[ -](\d+)/(\d+)", text)
        if mixed:
            whole = int(mixed.group(1))
            num = int(mixed.group(2))
            den = int(mixed.group(3))
            return f"{whole + num / den:g}"
        frac = re.fullmatch(r"(\d+)/(\d+)", text)
        if frac:
            num = int(frac.group(1))
            den = int(frac.group(2))
            return f"{num / den:g}"
        return ""

    @staticmethod
    def _build_dn_composite_item(left: str, right: str, *, is_left: bool) -> ParsedSizeItem:
        value = left if is_left else right
        return ParsedSizeItem(
            field="DN",
            raw=f"DN{value}",
            value=value,
            values=[value],
            anchored_patterns=SizeSurfaceMatcher._build_dn_composite_patterns(left, right, is_left=is_left),
            bare_values=[value],
        )

    @staticmethod
    def _build_dn_composite_patterns(left: str, right: str, *, is_left: bool) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        value = left if is_left else right
        patterns.append((f"DN{value}", SizeSurfaceMatcher._compile_dn_pattern(value)))
        if is_left:
            patterns.append(
                (
                    f"{left}xDN{right}",
                    re.compile(
                        rf'{TOKEN_HEAD}({re.escape(left)})\s*{X_SEP}\s*DN\s*:?\s*{re.escape(right)}{TOKEN_TAIL}',
                        re.IGNORECASE,
                    ),
                )
            )
        else:
            patterns.append(
                (
                    f"DN{left}x{right}",
                    re.compile(
                        rf'{TOKEN_HEAD}DN\s*:?\s*{re.escape(left)}\s*{X_SEP}\s*({re.escape(right)}){TOKEN_TAIL}',
                        re.IGNORECASE,
                    ),
                )
            )
        return patterns

    @staticmethod
    def _build_inch_composite_item(raw: str, left: str, right: str, *, is_left: bool) -> ParsedSizeItem:
        value = left if is_left else right
        return ParsedSizeItem(
            field="INCH",
            raw=raw,
            value=value,
            values=[value],
            anchored_patterns=SizeSurfaceMatcher._build_inch_composite_patterns(left, right, is_left=is_left),
            bare_values=[value],
        )

    @staticmethod
    def _build_single_inch_patterns(raw: str, value: str) -> list[tuple[str, re.Pattern[str]]]:
        patterns: list[tuple[str, re.Pattern[str]]] = []
        for variant in SizeSurfaceMatcher._inch_value_variants(value):
            patterns.append(
                (
                    raw,
                    re.compile(rf'{TOKEN_HEAD}({re.escape(variant)}\s*(?:[\"″]|\bIN(?:CH)?\b)){TOKEN_TAIL}', re.IGNORECASE),
                )
            )
        return patterns

    @staticmethod
    def _build_inch_composite_patterns(left: str, right: str, *, is_left: bool) -> list[tuple[str, re.Pattern[str]]]:
        left_variants = SizeSurfaceMatcher._inch_value_variants(left)
        right_variants = SizeSurfaceMatcher._inch_value_variants(right)
        left_alt = "|".join(re.escape(v) for v in left_variants)
        right_alt = "|".join(re.escape(v) for v in right_variants)
        alias = left if is_left else right
        if is_left:
            pattern = re.compile(
                rf'{TOKEN_HEAD}(({left_alt}))\s*(?:[\"″]|\bIN(?:CH)?\b)?\s*{X_SEP}\s*(?:{right_alt})\s*(?:[\"″]|\bIN(?:CH)?\b)?{TOKEN_TAIL}',
                re.IGNORECASE,
            )
        else:
            pattern = re.compile(
                rf'{TOKEN_HEAD}(?:{left_alt})\s*(?:[\"″]|\bIN(?:CH)?\b)?\s*{X_SEP}\s*(({right_alt}))\s*(?:[\"″]|\bIN(?:CH)?\b)?{TOKEN_TAIL}',
                re.IGNORECASE,
            )
        return [(alias, pattern)]

    @staticmethod
    def _compile_dn_pattern(value: str) -> re.Pattern[str]:
        return re.compile(rf'{TOKEN_HEAD}((?:DN)\s*:?\s*{re.escape(value)}){TOKEN_TAIL}', re.IGNORECASE)

    @staticmethod
    def _compile_od_pattern(value: str) -> re.Pattern[str]:
        return re.compile(rf'{TOKEN_HEAD}((?:OD\s*|[Φφ])\s*{re.escape(value)}){TOKEN_TAIL}', re.IGNORECASE)

    @staticmethod
    def _compile_mm_pattern(value: str) -> re.Pattern[str]:
        return re.compile(rf'{TOKEN_HEAD}({re.escape(value)}\s*MM){TOKEN_TAIL}', re.IGNORECASE)

    @staticmethod
    def _compile_single_inch_pattern(value: str) -> re.Pattern[str]:
        variants = SizeSurfaceMatcher._inch_value_variants(value)
        body = "|".join(re.escape(v) for v in variants)
        return re.compile(rf'{TOKEN_HEAD}(({body})\s*(?:[\"″]|\bIN(?:CH)?\b)){TOKEN_TAIL}', re.IGNORECASE)

    @staticmethod
    def _compile_inch_composite_pattern(values: list[str]) -> re.Pattern[str]:
        pieces = [rf'{re.escape(value)}\s*[\"″]?' for value in values]
        body = X_SEP.join(pieces)
        return re.compile(rf'{TOKEN_HEAD}({body}){TOKEN_TAIL}', re.IGNORECASE)

    @staticmethod
    def _compile_bare_number_pattern(value: str) -> re.Pattern[str]:
        return re.compile(rf'(?<!\d)({re.escape(value)})(?!\d)', re.IGNORECASE)

    @staticmethod
    def _overlaps_consumed(start: int, end: int, consumed_spans: list[tuple[int, int]]) -> bool:
        for consumed_start, consumed_end in consumed_spans:
            if start < consumed_end and end > consumed_start:
                return True
        return False

    @staticmethod
    def _dedupe_items(items: list[ParsedSizeItem]) -> list[ParsedSizeItem]:
        dedup: dict[tuple[str, str, tuple[str, ...]], ParsedSizeItem] = {}
        for item in items:
            key = (item.field, item.value or item.raw.upper(), tuple(item.values))
            if key not in dedup:
                dedup[key] = ParsedSizeItem(
                    field=item.field,
                    raw=item.raw,
                    value=item.value,
                    values=list(item.values),
                    anchored_patterns=list(item.anchored_patterns),
                    bare_values=list(item.bare_values),
                )
                continue
            existing = dedup[key]
            existing.anchored_patterns.extend(item.anchored_patterns)
            existing.bare_values.extend(item.bare_values)
            if len(item.raw) > len(existing.raw):
                existing.raw = item.raw
        result = list(dedup.values())
        result.sort(key=lambda item: (-len(item.raw), item.field, item.raw.upper()))
        return result

    @staticmethod
    def _dedupe_hits(hits: list[SizeSurfaceHit]) -> list[SizeSurfaceHit]:
        dedup: dict[tuple[str, int, int, str], SizeSurfaceHit] = {}
        for hit in hits:
            key = (hit.field, hit.start, hit.end, hit.kind)
            existing = dedup.get(key)
            if existing is None or len(hit.alias) > len(existing.alias):
                dedup[key] = hit
        result = list(dedup.values())
        result.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias.upper(), item.kind))
        return result
