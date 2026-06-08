# -*- coding: utf-8 -*-
"""Encoding-driven strong-surface matcher for standard second-pass checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from .models import StandardSurfaceHit


FLEX_GAP = r"[\s.]*"
PREFIX_GAP = r"[\s./-]*"
SPACE_GAP = r"\s*"
SEPARATOR_BOUNDARY = r"[\s,，;；|｜:./()（）\[\]{}\\_-]"
STANDARD_CODE_SEPARATORS = re.compile(r"[|｜;；,，]+")


@dataclass
class ParsedStandardCode:
    raw_code: str
    family: str
    core: str
    suffix: str


class StandardSurfaceMatcher:
    def __init__(
        self,
        config_path: str | Path | None = None,
        surface_map_path: str | Path | None = None,
    ) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "standard_second_pass.yaml"
        )
        self.surface_map_path = (
            Path(surface_map_path)
            if surface_map_path is not None
            else Path(__file__).resolve().parent.parent / "config" / "standard_surface_map.yaml"
        )
        self._load_config()

    def split_codes(self, standard_code: str) -> list[str]:
        parts = [self._clean_token(x) for x in STANDARD_CODE_SEPARATORS.split(str(standard_code or ""))]
        return [part for part in parts if part]

    def parse_code(self, code: str) -> ParsedStandardCode | None:
        clean = self._clean_token(code)
        if not clean:
            return None
        family = self._match_family(clean)
        if not family:
            return None
        rest = clean[len(family):]
        family_rule = self.get_family_rule(family)
        core_mode = str(family_rule.get("core_mode", "direct"))
        core, suffix_rest = self._split_core_and_suffix(rest, core_mode)
        suffix = self._extract_suffix(suffix_rest)
        if not core:
            return None
        return ParsedStandardCode(raw_code=clean, family=family, core=core, suffix=suffix)

    def get_family_rule(self, family: str) -> dict[str, Any]:
        return dict(self.family_rules.get(self._clean_token(family), {}))

    def find_base_hits(self, text: str, parsed: ParsedStandardCode) -> list[StandardSurfaceHit]:
        clean_code = parsed.raw_code
        family = parsed.family
        core = parsed.core

        hits: list[StandardSurfaceHit] = []

        # 1. existing high-precision locator patterns from standard_surface_map
        base_code = f"{family}{core}"
        for pattern in self.locator_patterns.get(base_code, []):
            for match in pattern.finditer(text):
                hits.append(
                    StandardSurfaceHit(
                        code=clean_code,
                        field="BASE",
                        alias=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        text=text[match.start():match.end()],
                    )
                )

        # 2. dynamic family+core patterns
        family_rule = self.get_family_rule(family)
        prefixed_patterns = self._build_prefixed_patterns(family, core, family_rule)
        for alias, pattern in prefixed_patterns:
            for match in pattern.finditer(text):
                hits.append(
                    StandardSurfaceHit(
                        code=clean_code,
                        field="BASE",
                        alias=alias,
                        start=match.start(),
                        end=match.end(),
                        text=text[match.start():match.end()],
                    )
                )

        # 3. bare core patterns if allowed
        if family_rule.get("bare_core_allowed", False):
            for alias, pattern in self._build_bare_core_patterns(core, family_rule):
                for match in pattern.finditer(text):
                    hits.append(
                        StandardSurfaceHit(
                            code=clean_code,
                            field="BASE",
                            alias=alias,
                            start=match.start(),
                            end=match.end(),
                            text=text[match.start():match.end()],
                        )
                    )

        return self._dedupe_hits(hits)

    def find_prefix_hits(
        self,
        text: str,
        parsed: ParsedStandardCode,
        base_hit: StandardSurfaceHit,
    ) -> tuple[list[StandardSurfaceHit], list[StandardSurfaceHit]]:
        expected: list[StandardSurfaceHit] = []
        conflicting: list[StandardSurfaceHit] = []

        # 1. prefix may already be included in the matched base span, e.g. "ASME B16.11"
        # Only treat it as a valid in-span prefix when it starts at the beginning
        # of the matched base span. This avoids false conflicts such as the "B"
        # inside "GB/T12459" being treated as an AB-family prefix.
        for family, rule in self.family_rules.items():
            for alias, pattern in self._build_prefix_patterns(family, rule):
                for match in pattern.finditer(base_hit.text):
                    if match.start() != 0:
                        continue
                    abs_start = base_hit.start + match.start()
                    abs_end = base_hit.start + match.end()
                    hit = StandardSurfaceHit(
                        code=parsed.raw_code,
                        field="PREFIX",
                        alias=alias,
                        start=abs_start,
                        end=abs_end,
                        text=text[abs_start:abs_end],
                    )
                    if family == parsed.family:
                        expected.append(hit)
                    else:
                        conflicting.append(hit)

        # 2. for bare-core cases, prefix may appear immediately before the core
        lookback_start = max(0, base_hit.start - 20)
        prefix_zone = text[lookback_start:base_hit.start]
        for family, rule in self.family_rules.items():
            for alias, pattern in self._build_prefix_patterns(family, rule):
                for match in pattern.finditer(prefix_zone):
                    abs_start = lookback_start + match.start()
                    abs_end = lookback_start + match.end()
                    gap_text = text[abs_end:base_hit.start]
                    hit = StandardSurfaceHit(
                        code=parsed.raw_code,
                        field="PREFIX",
                        alias=alias,
                        start=abs_start,
                        end=abs_end,
                        text=text[abs_start:abs_end],
                    )
                    if any(not ch.isspace() for ch in gap_text):
                        continue
                    if family == parsed.family:
                        expected.append(hit)
                    else:
                        conflicting.append(hit)
        return self._dedupe_hits(expected), self._dedupe_hits(conflicting)

    def find_suffix_hits(self, text: str, parsed: ParsedStandardCode, base_hit: StandardSurfaceHit) -> list[StandardSurfaceHit]:
        if not parsed.suffix:
            return []
        suffix_key = parsed.suffix.upper()
        if suffix_key in self.composite_suffix_rules:
            return self._find_composite_suffix_hits(text, parsed, base_hit, suffix_key)
        suffix_rule = self.suffix_patterns.get(suffix_key)
        if not suffix_rule:
            return []
        return self._find_single_suffix_hits(text, parsed, base_hit, suffix_key, suffix_rule)

    def find_suspicious_suffix_hits(
        self,
        text: str,
        parsed: ParsedStandardCode,
        base_hit: StandardSurfaceHit,
        consumed_hits: Iterable[StandardSurfaceHit] = (),
    ) -> list[StandardSurfaceHit]:
        consumed_spans = [(hit.start, hit.end) for hit in consumed_hits]
        suspicious: list[StandardSurfaceHit] = []
        for suffix_key, rule in self.suffix_patterns.items():
            if suffix_key == parsed.suffix.upper():
                continue
            if rule.get("mode") != "nearby":
                continue
            for alias in self._iter_suspicious_aliases(rule):
                pattern = self._compile_suffix_pattern(alias)
                for match in pattern.finditer(text):
                    abs_start = match.start()
                    abs_end = match.end()
                    if self._overlaps_consumed(abs_start, abs_end, consumed_spans):
                        continue
                    suspicious.append(
                        StandardSurfaceHit(
                            code=parsed.raw_code,
                            field="SUFFIX",
                            alias=suffix_key,
                            start=abs_start,
                            end=abs_end,
                            text=text[abs_start:abs_end],
                        )
                    )
        return self._dedupe_hits(suspicious)

    def suffix_supported(self, suffix: str) -> bool:
        clean = self._clean_token(suffix)
        return clean in self.suffix_patterns or clean in self.composite_suffix_rules

    def suffix_satisfied(self, suffix: str, hits: Iterable[StandardSurfaceHit]) -> bool:
        suffix_key = self._clean_token(suffix)
        hit_list = list(hits)
        if not suffix_key:
            return True
        if suffix_key in self.composite_suffix_rules:
            rule = self.composite_suffix_rules.get(suffix_key, {})
            primary_key = self._clean_token(rule.get("primary", ""))
            appendix_key = self._clean_token(rule.get("appendix", ""))
            aliases = {self._clean_token(hit.alias) for hit in hit_list}
            return primary_key in aliases and appendix_key in aliases
        return bool(hit_list)

    def family_requires_prefix(self, family: str) -> bool:
        return bool(self.get_family_rule(family).get("prefix_required", False))

    def family_allows_bare_core(self, family: str) -> bool:
        return bool(self.get_family_rule(family).get("bare_core_allowed", False))

    def _find_single_suffix_hits(
        self,
        text: str,
        parsed: ParsedStandardCode,
        base_hit: StandardSurfaceHit,
        suffix_key: str,
        suffix_rule: dict[str, Any],
    ) -> list[StandardSurfaceHit]:
        hits: list[StandardSurfaceHit] = []
        window = int(suffix_rule.get("window", self.suffix_window_default))
        segment_start = base_hit.end
        segment_end = min(len(text), base_hit.end + self._suffix_scan_length(suffix_rule, window))
        segment = text[segment_start:segment_end]
        for alias in suffix_rule.get("patterns", []):
            pattern = self._compile_suffix_pattern(alias)
            for match in pattern.finditer(segment):
                if match.start() > window:
                    continue
                abs_start = segment_start + match.start()
                abs_end = segment_start + match.end()
                hits.append(
                    StandardSurfaceHit(
                        code=parsed.raw_code,
                        field="SUFFIX",
                        alias=alias,
                        start=abs_start,
                        end=abs_end,
                        text=text[abs_start:abs_end],
                    )
                )
        return self._dedupe_hits(hits)

    def _find_composite_suffix_hits(
        self,
        text: str,
        parsed: ParsedStandardCode,
        base_hit: StandardSurfaceHit,
        suffix_key: str,
    ) -> list[StandardSurfaceHit]:
        rule = self.composite_suffix_rules.get(suffix_key, {})
        primary_key = self._clean_token(rule.get("primary", ""))
        appendix_key = self._clean_token(rule.get("appendix", ""))
        primary_rule = self.suffix_patterns.get(primary_key, {})
        appendix_rule = self.suffix_patterns.get(appendix_key, {})
        hits: list[StandardSurfaceHit] = []

        primary_window = int(rule.get("primary_window", primary_rule.get("window", self.suffix_window_default)))
        appendix_window = int(rule.get("appendix_window", appendix_rule.get("window", self.appendix_window_default)))

        primary_hits = self._find_suffix_hits_by_rule(text, parsed, base_hit, primary_key, primary_rule, primary_window)
        appendix_hits = self._find_suffix_hits_by_rule(text, parsed, base_hit, appendix_key, appendix_rule, appendix_window)
        hits.extend(primary_hits)
        hits.extend(appendix_hits)
        return self._dedupe_hits(hits)

    def _find_suffix_hits_by_rule(
        self,
        text: str,
        parsed: ParsedStandardCode,
        base_hit: StandardSurfaceHit,
        suffix_key: str,
        suffix_rule: dict[str, Any],
        window: int,
    ) -> list[StandardSurfaceHit]:
        hits: list[StandardSurfaceHit] = []
        segment_start = base_hit.end
        segment_end = min(len(text), base_hit.end + self._suffix_scan_length(suffix_rule, window))
        segment = text[segment_start:segment_end]
        for alias in suffix_rule.get("patterns", []):
            pattern = self._compile_suffix_pattern(alias)
            for match in pattern.finditer(segment):
                if match.start() > window:
                    continue
                abs_start = segment_start + match.start()
                abs_end = segment_start + match.end()
                hits.append(
                    StandardSurfaceHit(
                        code=parsed.raw_code,
                        field="SUFFIX",
                        alias=suffix_key,
                        start=abs_start,
                        end=abs_end,
                        text=text[abs_start:abs_end],
                    )
                )
        return self._dedupe_hits(hits)

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.meta = data.get("meta", {}) or {}
        self.suffix_window_default = int(self.meta.get("suffix_window_default", 6))
        self.appendix_window_default = int(self.meta.get("appendix_window_default", 30))
        self.roman_series_suffixes = [
            self._clean_token(item) for item in (self.meta.get("roman_series_suffixes", []) or []) if self._clean_token(item)
        ]
        self.family_rules: dict[str, dict[str, Any]] = {
            self._clean_token(key): (value or {}) for key, value in (data.get("family_rules", {}) or {}).items()
        }
        raw_suffix_patterns = {
            self._clean_token(key): (value or {}) for key, value in (data.get("suffix_patterns", {}) or {}).items()
        }
        self.suffix_patterns = self._expand_suffix_patterns(raw_suffix_patterns)
        self.composite_suffix_rules: dict[str, dict[str, Any]] = {
            self._clean_token(key): (value or {}) for key, value in (data.get("composite_suffix_rules", {}) or {}).items()
        }
        self.family_order = sorted(self.family_rules.keys(), key=len, reverse=True)

        with self.surface_map_path.open("r", encoding="utf-8") as f:
            surface_data = yaml.safe_load(f) or {}
        raw_locator = surface_data.get("locator_codes", {}) or {}
        self.locator_patterns: dict[str, list[re.Pattern[str]]] = {}
        for code, entry in raw_locator.items():
            clean_code = self._clean_token(code)
            patterns = [re.compile(str(p), re.IGNORECASE) for p in (entry.get("patterns", []) or []) if str(p).strip()]
            if patterns:
                self.locator_patterns[clean_code] = patterns

    def _expand_suffix_patterns(self, raw_suffix_patterns: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        suffix_patterns: dict[str, dict[str, Any]] = {}
        roman_template = raw_suffix_patterns.pop("ROMAN_SERIES", None)
        for key, value in raw_suffix_patterns.items():
            suffix_patterns[key] = dict(value or {})
        if roman_template:
            for suffix in self.roman_series_suffixes:
                suffix_patterns[suffix] = self._build_roman_suffix_rule(suffix, roman_template)
        return suffix_patterns

    def _build_roman_suffix_rule(self, suffix: str, template: dict[str, Any]) -> dict[str, Any]:
        suffix_key = self._clean_token(suffix)
        unicode_value = self._roman_unicode_char(suffix_key)
        expanded_patterns = []
        for raw_pattern in template.get("patterns", []) or []:
            pattern = str(raw_pattern or "")
            if not pattern:
                continue
            expanded_patterns.append(pattern.format(suffix=suffix_key, unicode=unicode_value))
        rule = dict(template or {})
        rule["patterns"] = expanded_patterns
        return rule

    @staticmethod
    def _roman_unicode_char(suffix: str) -> str:
        roman_unicode_map = {
            "I": "Ⅰ",
            "II": "Ⅱ",
            "III": "Ⅲ",
            "IV": "Ⅳ",
            "V": "Ⅴ",
            "VI": "Ⅵ",
            "VII": "Ⅶ",
            "VIII": "Ⅷ",
            "IX": "Ⅸ",
            "X": "Ⅹ",
        }
        return roman_unicode_map.get(str(suffix or "").upper(), str(suffix or "").upper())

    def _match_family(self, clean_code: str) -> str:
        for family in self.family_order:
            if clean_code.startswith(family):
                return family
        return ""

    def _extract_suffix(self, rest: str) -> str:
        candidates = sorted(
            [key for key in self.suffix_patterns.keys() if key != "APPENDIX_B"] + list(self.composite_suffix_rules.keys()),
            key=len,
            reverse=True,
        )
        upper_rest = self._clean_token(rest)
        for suffix in candidates:
            if upper_rest.endswith(suffix):
                return suffix
        return ""

    def _split_core_and_suffix(self, rest: str, core_mode: str) -> tuple[str, str]:
        clean_rest = self._clean_token(rest)
        if not clean_rest:
            return "", ""
        if core_mode == "direct":
            match = re.match(r"^(\d+(?:\.\d+)?)(.*)$", clean_rest)
            if match:
                return match.group(1), match.group(2)
            return clean_rest, ""
        if core_mode in {"asme_b_compact", "mss_sp_compact"}:
            match = re.match(r"^(\d+)(.*)$", clean_rest)
            if match:
                return match.group(1), match.group(2)
            return clean_rest, ""
        return clean_rest, ""

    def _build_prefixed_patterns(self, family: str, core: str, family_rule: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
        core_mode = str(family_rule.get("core_mode", "direct"))
        prefix_patterns = family_rule.get("prefix_patterns", []) or []
        core_pattern = self._build_core_pattern(core, core_mode)
        out: list[tuple[str, re.Pattern[str]]] = []
        for prefix in prefix_patterns:
            prefix_piece = self._build_prefix_pattern(prefix)
            if not prefix_piece:
                continue
            pattern = re.compile(rf"{prefix_piece}{SPACE_GAP}{core_pattern}", re.IGNORECASE)
            out.append((prefix, pattern))
        return out

    def _build_bare_core_patterns(self, core: str, family_rule: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
        core_mode = str(family_rule.get("core_mode", "direct"))
        if core_mode != "direct":
            return []
        core_pattern = self._build_core_pattern(core, core_mode)
        pattern = re.compile(rf"(?<!\d)({core_pattern})(?!\d)", re.IGNORECASE)
        return [(core, pattern)]

    def _build_prefix_patterns(self, family: str, family_rule: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
        out: list[tuple[str, re.Pattern[str]]] = []
        for prefix in family_rule.get("prefix_patterns", []) or []:
            piece = self._build_prefix_pattern(prefix)
            if not piece:
                continue
            out.append((prefix, re.compile(piece, re.IGNORECASE)))
        return out

    @staticmethod
    def _build_prefix_pattern(prefix: str) -> str:
        chars = []
        for ch in str(prefix or "").upper():
            if ch in {" ", "."}:
                chars.append(FLEX_GAP)
            elif ch in {"/", "-"}:
                chars.append(PREFIX_GAP)
            else:
                chars.append(re.escape(ch))
        return "".join(chars)

    def _build_core_pattern(self, core: str, mode: str) -> str:
        clean_core = str(core or "").upper()
        if mode == "direct":
            parts = []
            for ch in clean_core:
                if ch == ".":
                    parts.append(r"\s*\.\s*")
                else:
                    parts.append(re.escape(ch))
            return "".join(parts)
        if mode == "asme_b_compact":
            if not clean_core.isdigit() or len(clean_core) < 3:
                return re.escape(clean_core)
            major = clean_core[:2]
            minor = clean_core[2:]
            suffix = r"(?:M)?"
            return rf"{re.escape(major)}\s*\.\s*{re.escape(minor)}{suffix}"
        if mode == "mss_sp_compact":
            return re.escape(clean_core)
        return re.escape(clean_core)

    @staticmethod
    def _compile_literal_pattern(alias: str) -> re.Pattern[str]:
        parts = []
        for ch in str(alias or "").upper():
            if ch in {" ", "."}:
                parts.append(FLEX_GAP)
            elif ch in {"/", "-"}:
                parts.append(PREFIX_GAP)
            else:
                parts.append(re.escape(ch))
        return re.compile("".join(parts), re.IGNORECASE)

    def _compile_suffix_pattern(self, alias: str) -> re.Pattern[str]:
        pattern = self._compile_literal_pattern(alias).pattern
        if self._needs_strict_separator_boundary(alias):
            return re.compile(
                rf"(?:(?<=^)|(?<={SEPARATOR_BOUNDARY}))({pattern})(?:(?=$)|(?={SEPARATOR_BOUNDARY}))",
                re.IGNORECASE,
            )
        if self._needs_non_letter_boundary(alias):
            return re.compile(rf"(?<![A-Za-z])({pattern})(?![A-Za-z])", re.IGNORECASE)
        prefix_guard = ""
        suffix_guard = ""
        first_char = self._first_meaningful_char(alias)
        last_char = self._last_meaningful_char(alias)
        if first_char.isalpha():
            prefix_guard = r"(?<![A-Za-z])"
        elif first_char.isdigit():
            prefix_guard = r"(?<!\d)"
        if last_char.isalpha():
            suffix_guard = r"(?![A-Za-z])"
        elif last_char.isdigit():
            suffix_guard = r"(?!\d)"
        return re.compile(rf"{prefix_guard}({pattern}){suffix_guard}", re.IGNORECASE)

    @staticmethod
    def _dedupe_hits(hits: list[StandardSurfaceHit]) -> list[StandardSurfaceHit]:
        dedup: dict[tuple[str, str, int, int], StandardSurfaceHit] = {}
        for hit in hits:
            dedup[(hit.field, hit.alias.upper(), hit.start, hit.end)] = hit
        items = list(dedup.values())
        items.sort(key=lambda item: (item.start, -(item.end - item.start), item.alias.upper()))
        return items

    def _suffix_scan_length(self, suffix_rule: dict[str, Any], window: int) -> int:
        patterns = suffix_rule.get("patterns", []) or []
        longest = max((len(str(item or "")) for item in patterns), default=0)
        return max(window, window + longest + 4)

    @staticmethod
    def _needs_strict_separator_boundary(alias: str) -> bool:
        compact = re.sub(r"[\s.()/\-]", "", str(alias or "")).upper()
        return compact in {"A", "B"} or bool(re.fullmatch(r"[IVX]", compact))

    @staticmethod
    def _needs_non_letter_boundary(alias: str) -> bool:
        compact = re.sub(r"[\s.()/\-]", "", str(alias or "")).upper()
        return compact in {"IA", "IB", "LA"} or bool(re.fullmatch(r"[IVX]{2,}", compact))

    @staticmethod
    def _first_meaningful_char(alias: str) -> str:
        for ch in str(alias or ""):
            if not ch.isspace():
                return ch
        return ""

    @staticmethod
    def _last_meaningful_char(alias: str) -> str:
        for ch in reversed(str(alias or "")):
            if not ch.isspace():
                return ch
        return ""

    @staticmethod
    def _overlaps_consumed(start: int, end: int, consumed_spans: list[tuple[int, int]]) -> bool:
        for consumed_start, consumed_end in consumed_spans:
            if start < consumed_end and end > consumed_start:
                return True
        return False

    @staticmethod
    def _iter_suspicious_aliases(suffix_rule: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        for raw in suffix_rule.get("patterns", []) or []:
            alias = str(raw or "")
            compact = re.sub(r"[\s.()/\-]", "", alias).upper()
            # 裸单字母 A/B 全描述扫描噪声过大，疑似残留时只保留强写法。
            if compact in {"A", "B"}:
                continue
            aliases.append(alias)
        return aliases

    @staticmethod
    def _clean_token(value: str) -> str:
        return str(value or "").strip().upper()
