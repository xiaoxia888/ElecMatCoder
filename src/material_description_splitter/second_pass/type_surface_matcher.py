# -*- coding: utf-8 -*-
"""Strong-surface matcher for second-pass type auto-pass checks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import yaml

from .models import TypeSurfaceHit

BOUNDARY_CLASS = r"A-Z0-9\u4e00-\u9fff"
IGNORED_IN_ALIAS = {" ", "."}
FLEX_GAP_PATTERN = r"[\s.]*"


class TypeSurfaceMatcher:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "type_second_pass.yaml"
        )
        self._load_config()

    def is_supported_code(self, type_code: str) -> bool:
        return self._clean_code(type_code) in self.common_types

    def get_more_specific_codes(self, type_code: str) -> list[str]:
        clean_code = self._clean_code(type_code)
        return list(self.more_specific_codes.get(clean_code, []))

    def match_direct(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.direct_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "DIRECT", patterns)

    def match_body(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.body_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "BODY", patterns)

    def match_manu(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.manu_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "MANU", patterns)

    def match_conn(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.conn_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "CONN", patterns)

    def match_seal(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.seal_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "SEAL", patterns)

    def match_angle(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.angle_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "ANGLE", patterns)

    def match_radius(self, text: str, type_code: str) -> list[TypeSurfaceHit]:
        patterns = self.radius_patterns.get(self._clean_code(type_code), ())
        return self._collect_hits(text, self._clean_code(type_code), "RADIUS", patterns)

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_common = data.get("common_types", {}) or {}
        self.common_types: dict[str, dict] = {}
        self.more_specific_codes: dict[str, list[str]] = {}
        self.direct_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.body_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.manu_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.conn_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.seal_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.angle_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}
        self.radius_patterns: dict[str, list[tuple[str, re.Pattern[str], re.Pattern[str] | None]]] = {}

        for code, meta in raw_common.items():
            clean_code = self._clean_code(code)
            if not clean_code or not isinstance(meta, dict):
                continue
            self.common_types[clean_code] = meta
            self.more_specific_codes[clean_code] = [self._clean_code(v) for v in meta.get("block_if_matches", []) or [] if self._clean_code(v)]

            direct_aliases = self._dedupe_aliases(meta.get("direct", []) or [])
            self.direct_patterns[clean_code] = [
                (
                    alias,
                    self._compile_alias_pattern(alias),
                    self._compile_start_glue_pattern(alias),
                )
                for alias in direct_aliases
            ]
            body_aliases = self._dedupe_aliases(meta.get("body", []) or [])
            self.body_patterns[clean_code] = [
                (
                    alias,
                    self._compile_alias_pattern(alias),
                    self._compile_start_glue_pattern(alias),
                )
                for alias in body_aliases
            ]
            manu_aliases = self._dedupe_aliases(meta.get("manu", []) or [])
            self.manu_patterns[clean_code] = [
                (alias, self._compile_alias_pattern(alias), None) for alias in manu_aliases
            ]
            conn_aliases = self._dedupe_aliases(meta.get("conn", []) or [])
            self.conn_patterns[clean_code] = [
                (alias, self._compile_alias_pattern(alias), None) for alias in conn_aliases
            ]
            seal_aliases = self._dedupe_aliases(meta.get("seal", []) or [])
            self.seal_patterns[clean_code] = [
                (alias, self._compile_alias_pattern(alias), None) for alias in seal_aliases
            ]
            angle_aliases = self._dedupe_aliases(meta.get("angle", []) or [])
            self.angle_patterns[clean_code] = [
                (alias, self._compile_angle_pattern(alias), None) for alias in angle_aliases
            ]
            radius_aliases = self._dedupe_aliases(meta.get("radius", []) or [])
            self.radius_patterns[clean_code] = [
                (alias, self._compile_alias_pattern(alias), None) for alias in radius_aliases
            ]

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
            key = TypeSurfaceMatcher._normalize_alias_key(alias)
            if key in seen:
                continue
            seen.add(key)
            result.append(alias)
        result.sort(key=lambda item: (-len(item), item.upper()))
        return result

    @staticmethod
    def _compile_alias_pattern(alias: str) -> re.Pattern[str]:
        normalized = TypeSurfaceMatcher._normalize_alias_key(alias)
        escaped = FLEX_GAP_PATTERN.join(
            re.escape(ch) for ch in alias.upper() if ch not in IGNORED_IN_ALIAS
        )
        if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
            pattern = rf"({escaped})"
        elif normalized == "SO":
            # FSO 的 SO 经常写进类似 HG/T20592SO150(A) 这类规格串里：
            # 左边允许数字/符号贴边，右边允许数字或 "("。
            pattern = rf"(?<![A-Z])({escaped})(?=$|[^A-Z]|\d|\()"
        else:
            pattern = rf"(?<![{BOUNDARY_CLASS}])({escaped})(?![{BOUNDARY_CLASS}])"
        return re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _compile_start_glue_pattern(alias: str) -> re.Pattern[str] | None:
        if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
            return None
        escaped = FLEX_GAP_PATTERN.join(
            re.escape(ch) for ch in alias.upper() if ch not in IGNORED_IN_ALIAS
        )
        return re.compile(rf"^\s*({escaped})", re.IGNORECASE)

    @staticmethod
    def _compile_angle_pattern(alias: str) -> re.Pattern[str]:
        compact = [ch for ch in alias.upper() if ch not in IGNORED_IN_ALIAS]
        escaped = FLEX_GAP_PATTERN.join(re.escape(ch) for ch in compact)
        left_guard = TypeSurfaceMatcher._edge_guard(compact[0], side="left") if compact else ""
        right_guard = TypeSurfaceMatcher._edge_guard(compact[-1], side="right") if compact else ""
        # 角度字段允许和非同类字符贴边，是否禁止粘连由 alias 首尾实际字符类型决定。
        pattern = rf"{left_guard}({escaped}){right_guard}"
        return re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _edge_guard(ch: str, side: str) -> str:
        if ch.isdigit():
            return r"(?<!\d)" if side == "left" else r"(?!\d)"
        if "A" <= ch <= "Z":
            return r"(?<![A-Z])" if side == "left" else r"(?![A-Z])"
        return ""

    @staticmethod
    def _normalize_alias_key(value: str) -> str:
        raw = str(value or "").strip().upper()
        return "".join(ch for ch in raw if ch not in IGNORED_IN_ALIAS)

    @staticmethod
    def _prune_overlaps(hits: list[TypeSurfaceHit]) -> list[TypeSurfaceHit]:
        ordered = sorted(hits, key=lambda item: (item.start, -(item.end - item.start), item.alias.upper()))
        kept: list[TypeSurfaceHit] = []
        for hit in ordered:
            if any(not (hit.end <= existing.start or hit.start >= existing.end) for existing in kept):
                continue
            kept.append(hit)
        return kept

    def _collect_hits(
        self,
        text: str,
        code: str,
        field: str,
        patterns: Iterable[tuple[str, re.Pattern[str], re.Pattern[str] | None]],
    ) -> list[TypeSurfaceHit]:
        raw_text = str(text or "")
        upper_text = raw_text.upper()
        hits: list[TypeSurfaceHit] = []
        for alias, pattern, start_glue_pattern in patterns:
            for match in pattern.finditer(upper_text):
                hits.append(
                    TypeSurfaceHit(
                        code=code,
                        field=field,
                        alias=alias,
                        start=match.start(1),
                        end=match.end(1),
                        text=raw_text[match.start(1): match.end(1)],
                    )
                )
            if field in {"DIRECT", "BODY"} and start_glue_pattern is not None:
                start_match = start_glue_pattern.search(upper_text)
                if start_match:
                    hits.append(
                        TypeSurfaceHit(
                            code=code,
                            field=field,
                            alias=alias,
                            start=start_match.start(1),
                            end=start_match.end(1),
                            text=raw_text[start_match.start(1): start_match.end(1)],
                        )
                    )
        return self._prune_overlaps(hits)
