# -*- coding: utf-8 -*-
"""Detector for glued standard spans."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from .models import DifficultyFeature, GlueHit


LEFT_RIGHT_SEPARATORS = set(" \t\r\n,;，；")
RIGHT_SUFFIX_PATTERNS = (
    (re.compile(r"\((?:[IVX]{1,4}|[A-Z]{1,3}|\d{4})\)"), False),
    # Series / SER 视为规范合法后缀表达，可继续带常见等级尾缀。
    # 例如 GB/T12459Series II、GB/T12459SER II、GB/T12459SeriesA。
    (re.compile(r"(?i)(?:SER(?:IES)?)\s*(?:IA|IB|II|III|IV|A|B)?"), True),
    (re.compile(r"(?:IA|IB|II|III|IV|A|B)"), True),
)


class StandardGlueDetector:
    """Detect whether standard spans are glued with neighboring text."""

    TAG_NAME = "standard_glue"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.locator_codes = self._load_locator_codes(config_path)

    def analyze(self, text: str) -> DifficultyFeature:
        hits: list[GlueHit] = []
        for code, match in self._find_effective_matches(text):
            left_glue = False
            end_after_suffix = self._consume_right_suffix(text, match.end())
            right_glue = self._is_right_glued(text, end_after_suffix)
            if not right_glue:
                continue

            token_start, token_end = self._expand_token_bounds(text, match.start(), end_after_suffix)
            token = text[token_start:token_end]
            note = self._build_note(text, code, match.start(), match.end(), end_after_suffix, left_glue, right_glue)
            hits.append(
                GlueHit(
                    tag=self.TAG_NAME,
                    code_group="standard_codes",
                    code=code,
                    token=token,
                    start=match.start(),
                    end=end_after_suffix,
                    note=note,
                )
            )

        # de-duplicate same code/span
        dedup: dict[tuple[str, int, int], GlueHit] = {}
        for hit in hits:
            dedup[(hit.code, hit.start, hit.end)] = hit
        final_hits = list(dedup.values())

        return DifficultyFeature(
            name=self.TAG_NAME,
            matched=bool(final_hits),
            reason="规范主体右侧在合法后缀之后仍与其他字符粘连" if final_hits else "",
            hits=final_hits,
        )

    def _find_effective_matches(self, text: str) -> list[tuple[str, re.Match[str]]]:
        raw_matches: list[tuple[str, re.Match[str]]] = []
        for code, entry in self.locator_codes.items():
            for pattern in entry["patterns"]:
                raw_matches.extend((code, match) for match in re.finditer(pattern, text, re.IGNORECASE))

        # suppress shorter matches fully contained in longer matches across all codes
        raw_matches.sort(key=lambda item: (-(item[1].end() - item[1].start()), item[1].start(), item[0]))
        kept: list[tuple[str, re.Match[str]]] = []
        for code, match in raw_matches:
            if any(prev.start() <= match.start() and match.end() <= prev.end() for _, prev in kept):
                continue
            kept.append((code, match))
        return kept

    @staticmethod
    def _is_left_glued(text: str, start: int) -> bool:
        if start <= 0:
            return False
        return text[start - 1].isalnum()

    def _consume_right_suffix(self, text: str, end: int) -> int:
        pos = end
        while pos < len(text):
            matched = False
            tail = text[pos:]

            # 用户规则：所有 "-后缀" 都算合法后缀。
            # 因此只要标准主体后紧跟 '-'，就一直吞到下一个分隔符为止。
            if tail.startswith("-"):
                next_pos = pos + 1
                while next_pos < len(text) and text[next_pos] not in LEFT_RIGHT_SEPARATORS:
                    next_pos += 1
                pos = next_pos
                matched = True
                continue

            for pattern, require_boundary in RIGHT_SUFFIX_PATTERNS:
                m = pattern.match(tail)
                if not m:
                    continue
                next_pos = pos + len(m.group(0))
                if require_boundary:
                    if next_pos < len(text) and text[next_pos].isalnum():
                        continue
                pos = next_pos
                matched = True
                break
            if not matched:
                break
        return pos

    @staticmethod
    def _is_right_glued(text: str, end: int) -> bool:
        if end >= len(text):
            return False
        return text[end].isalnum()

    @staticmethod
    def _expand_token_bounds(text: str, start: int, end: int) -> tuple[int, int]:
        left = start
        while left > 0 and text[left - 1] not in LEFT_RIGHT_SEPARATORS:
            left -= 1
        right = end
        while right < len(text) and text[right] not in LEFT_RIGHT_SEPARATORS:
            right += 1
        return left, right

    @staticmethod
    def _build_note(
        text: str,
        code: str,
        start: int,
        end: int,
        end_after_suffix: int,
        left_glue: bool,
        right_glue: bool,
    ) -> str:
        body = text[start:end]
        matched = text[start:end_after_suffix]
        return f"规范主体 {body} 右侧在合法后缀之后仍与其他字符粘连，当前片段为 {matched}"

    @staticmethod
    def _load_locator_codes(config_path: str | Path | None) -> dict[str, dict[str, Any]]:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config" / "standard_surface_map.yaml"
        else:
            config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        locator_codes = data.get("locator_codes", {}) or {}
        out: dict[str, dict[str, Any]] = {}
        for code, entry in locator_codes.items():
            patterns = entry.get("patterns", []) or []
            if not patterns:
                continue
            out[str(code)] = {
                "patterns": [str(p) for p in patterns if str(p).strip()],
            }
        return out
