# -*- coding: utf-8 -*-
"""Strong-surface matcher for second-pass material auto-pass."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import yaml

from .models import MaterialSurfaceHit

BOUNDARY_CLASS = r"A-Z0-9\u4e00-\u9fff"
SUFFIX_CODES = ("ZN", "CE")


class MaterialSurfaceMatcher:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "material_second_pass.yaml"
        )
        self._load_config()

    def split_material_code(self, material_code: str) -> tuple[str, str]:
        code = self._clean_code(material_code)
        if not code:
            return "", ""
        if code in self.common_materials:
            return code, ""
        for suffix in SUFFIX_CODES:
            if code.endswith(suffix):
                base = code[: -len(suffix)]
                if base in self.common_materials:
                    return base, suffix
        return code, ""

    def is_supported_code(self, material_code: str) -> bool:
        clean_code = self._clean_code(material_code)
        base, suffix = self.split_material_code(clean_code)
        if not base or base not in self.common_materials:
            return False
        if not suffix:
            return clean_code == base or clean_code in self.common_materials
        if suffix not in self.suffix_surfaces:
            return False
        enabled_codes = self.suffix_enabled_codes.get(suffix, set())
        if enabled_codes and clean_code not in enabled_codes:
            return False
        return True

    def match_base_surfaces(self, text: str, base_code: str) -> list[MaterialSurfaceHit]:
        patterns = self.base_patterns.get(base_code, ())
        return self._collect_hits(text, base_code, patterns, kind="base")

    def match_suffix_surfaces(self, text: str, suffix_code: str) -> list[MaterialSurfaceHit]:
        patterns = self.suffix_patterns.get(suffix_code, ())
        return self._collect_hits(text, suffix_code, patterns, kind="suffix")


    def find_any_suffix_hits(self, text: str) -> dict[str, list[MaterialSurfaceHit]]:
        found: dict[str, list[MaterialSurfaceHit]] = {}
        for suffix_code, patterns in self.suffix_patterns.items():
            hits = self._collect_hits(text, suffix_code, patterns, kind="suffix")
            if hits:
                found[suffix_code] = hits
        return found

    def match_combined_code(self, text: str, material_code: str) -> list[MaterialSurfaceHit]:
        code = self._clean_code(material_code)
        if not code:
            return []
        patterns = [(code, self._compile_alias_pattern(code))]
        return self._collect_hits(text, code, patterns, kind="combined")

    def find_conflict_hits(
        self,
        text: str,
        *,
        excluded_codes: Iterable[str],
    ) -> dict[str, list[MaterialSurfaceHit]]:
        excluded = {self._clean_code(code) for code in excluded_codes if self._clean_code(code)}
        conflicts: dict[str, list[MaterialSurfaceHit]] = {}
        for code, patterns in self.base_patterns.items():
            if code in excluded:
                continue
            hits = self._collect_hits(text, code, patterns, kind="base")
            if hits:
                conflicts[code] = hits
        return conflicts

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_common = data.get("common_materials", {}) or {}
        raw_suffix = data.get("suffix_surfaces", {}) or {}
        self.common_materials: dict[str, list[str]] = {}
        self.suffix_surfaces: dict[str, list[str]] = {}
        self.suffix_enabled_codes: dict[str, set[str]] = {}
        for code, aliases in raw_common.items():
            code_str = self._clean_code(code)
            if not code_str or not isinstance(aliases, list):
                continue
            uniq = self._dedupe_aliases(aliases)
            if uniq:
                self.common_materials[code_str] = uniq
        for code, aliases in raw_suffix.items():
            code_str = self._clean_code(code)
            if not code_str or not isinstance(aliases, list):
                continue
            uniq = self._dedupe_aliases(aliases)
            if uniq:
                self.suffix_surfaces[code_str] = uniq
        raw_suffix_enabled = data.get("suffix_enabled_codes", {}) or {}
        for suffix_code, codes in raw_suffix_enabled.items():
            suffix_key = self._clean_code(suffix_code)
            if not suffix_key or not isinstance(codes, list):
                continue
            self.suffix_enabled_codes[suffix_key] = {self._clean_code(code) for code in codes if self._clean_code(code)}
        self.base_patterns = {
            code: [(alias, self._compile_alias_pattern(alias)) for alias in aliases]
            for code, aliases in self.common_materials.items()
        }
        self.suffix_patterns = {
            code: [(alias, self._compile_alias_pattern(alias)) for alias in aliases]
            for code, aliases in self.suffix_surfaces.items()
        }

    @staticmethod
    def _clean_code(value: str) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _dedupe_aliases(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            alias = str(value or "").strip()
            if not alias:
                continue
            key = alias.upper()
            if key in seen:
                continue
            seen.add(key)
            result.append(alias)
        result.sort(key=lambda item: (-len(item), item.upper()))
        return result

    @staticmethod
    def _compile_alias_pattern(alias: str) -> re.Pattern[str]:
        escaped = re.escape(alias.upper())
        if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
            pattern = rf"({escaped})"
        elif alias.isdigit() or alias.isalpha():
            pattern = rf"(?<![{BOUNDARY_CLASS}])({escaped})(?![{BOUNDARY_CLASS}])"
        else:
            # 混合字母数字材质采用非对称边界：
            # - 若首字符是字母，则左侧不允许继续粘字母
            # - 若首字符是数字，则左侧不允许继续粘数字
            # - 若尾字符是字母，则右侧不允许继续粘字母
            # - 若尾字符是数字，则右侧不允许继续粘数字
            #
            # 这样 CF415GB/T13401 可视为合法命中，
            # 但像 ACF415 / CF4157 这种同类字符继续粘连仍会被挡住。
            left_guard = MaterialSurfaceMatcher._build_side_guard(alias[0], is_left=True)
            right_guard = MaterialSurfaceMatcher._build_side_guard(alias[-1], is_left=False)
            pattern = rf"{left_guard}({escaped}){right_guard}"
        return re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _build_side_guard(ch: str, *, is_left: bool) -> str:
        upper = str(ch or "").upper()
        if upper.isalpha():
            return r"(?<![A-Z])" if is_left else r"(?![A-Z])"
        if upper.isdigit():
            return r"(?<!\d)" if is_left else r"(?!\d)"
        return rf"(?<![{BOUNDARY_CLASS}])" if is_left else rf"(?![{BOUNDARY_CLASS}])"

    @staticmethod
    def _prune_overlaps(hits: list[MaterialSurfaceHit]) -> list[MaterialSurfaceHit]:
        ordered = sorted(hits, key=lambda item: (item.start, -(item.end - item.start), item.alias.upper()))
        kept: list[MaterialSurfaceHit] = []
        for hit in ordered:
            if any(not (hit.end <= existing.start or hit.start >= existing.end) for existing in kept):
                continue
            kept.append(hit)
        return kept

    def _collect_hits(
        self,
        text: str,
        code: str,
        patterns: Iterable[tuple[str, re.Pattern[str]]],
        *,
        kind: str,
    ) -> list[MaterialSurfaceHit]:
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        hits: list[MaterialSurfaceHit] = []
        for alias, pattern in patterns:
            for match in pattern.finditer(upper_text):
                hits.append(
                    MaterialSurfaceHit(
                        code=code,
                        alias=alias,
                        start=match.start(1),
                        end=match.end(1),
                        text=raw_text[match.start(1): match.end(1)],
                        kind=kind,
                    )
                )
        return self._prune_overlaps(hits)
