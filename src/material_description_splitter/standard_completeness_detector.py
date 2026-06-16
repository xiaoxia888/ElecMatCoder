# -*- coding: utf-8 -*-
"""Detector for standard presence completeness."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import DifficultyFeature, GlueHit


NON_ALNUM_LEFT = r"(?<![A-Za-z0-9])"


class StandardCompletenessDetector:
    """只判断描述中是否疑似存在规范表达。"""

    TAG_NAME = "standard_completeness"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "description_completeness.yaml"
        )
        self.allow_levels: set[str] = set()
        self.block_levels: set[str] = set()
        self.strong_patterns: list[re.Pattern[str]] = []
        self.weak_patterns: list[re.Pattern[str]] = []
        self._load_config()

    def analyze(self, text: str) -> DifficultyFeature:
        clean_text = str(text or "")
        if not clean_text.strip():
            return self._feature_for_level("missing")

        if self._match_any(self.strong_patterns, clean_text):
            return self._feature_for_level("strong")

        if self._match_any(self.weak_patterns, clean_text):
            return self._feature_for_level("weak")

        return self._feature_for_level("missing")

    def _feature_for_level(self, level: str) -> DifficultyFeature:
        if level in self.allow_levels and level not in self.block_levels:
            return DifficultyFeature(name=self.TAG_NAME, matched=False)

        if level != "missing":
            return DifficultyFeature(name=self.TAG_NAME, matched=level in self.block_levels)

        reason = "未命中规范表达"
        return DifficultyFeature(
            name=self.TAG_NAME,
            matched=True,
            reason=reason,
            hits=[
                GlueHit(
                    tag=self.TAG_NAME,
                    code_group="standard_presence",
                    code="missing",
                    token="",
                    start=-1,
                    end=-1,
                    note=f"STANDARD: {reason}",
                )
            ],
        )

    @staticmethod
    def _match_any(patterns: list[re.Pattern[str]], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        standard_cfg = (((data.get("checks") or {}).get("standard")) or {}) if isinstance(data, dict) else {}
        self.allow_levels = {str(item).strip() for item in (standard_cfg.get("allow_levels") or []) if str(item).strip()}
        self.block_levels = {str(item).strip() for item in (standard_cfg.get("block_levels") or []) if str(item).strip()}

        families = standard_cfg.get("families") or {}
        strong_patterns: list[re.Pattern[str]] = []
        weak_patterns: list[re.Pattern[str]] = []

        for family_cfg in families.values():
            if not isinstance(family_cfg, dict):
                continue
            mode = str(family_cfg.get("mode", "")).strip().lower()
            if mode == "prefix_core":
                prefixes = [str(item).strip() for item in (family_cfg.get("prefixes") or []) if str(item).strip()]
                cores = [str(item).strip() for item in (family_cfg.get("cores") or []) if str(item).strip()]
                strong_patterns.extend(self._build_prefix_core_strong_patterns(prefixes, cores))
                weak_patterns.extend(self._build_prefix_core_weak_patterns(cores))
            elif mode == "direct":
                patterns = [str(item).strip() for item in (family_cfg.get("patterns") or []) if str(item).strip()]
                strong_patterns.extend(re.compile(pattern, re.IGNORECASE) for pattern in patterns)

        self.strong_patterns = strong_patterns
        self.weak_patterns = weak_patterns

    @staticmethod
    def _build_prefix_core_strong_patterns(prefixes: list[str], cores: list[str]) -> list[re.Pattern[str]]:
        patterns: list[re.Pattern[str]] = []
        for prefix in prefixes:
            escaped_prefix = StandardCompletenessDetector._surface_to_pattern(prefix)
            for core in cores:
                escaped_core = StandardCompletenessDetector._surface_to_pattern(core)
                patterns.append(
                    re.compile(
                        rf"{NON_ALNUM_LEFT}{escaped_prefix}\s*{escaped_core}",
                        re.IGNORECASE,
                    )
                )
        return patterns

    @staticmethod
    def _build_prefix_core_weak_patterns(cores: list[str]) -> list[re.Pattern[str]]:
        patterns: list[re.Pattern[str]] = []
        for core in cores:
            escaped_core = StandardCompletenessDetector._surface_to_pattern(core)
            patterns.append(
                re.compile(
                    rf"{NON_ALNUM_LEFT}{escaped_core}(?!\d)",
                    re.IGNORECASE,
                )
            )
        return patterns

    @staticmethod
    def _surface_to_pattern(text: str) -> str:
        pieces: list[str] = []
        for ch in str(text or ""):
            if ch.isspace():
                pieces.append(r"\s*")
            elif ch in {"/", ".", "-", "_"}:
                pieces.append(rf"\s*{re.escape(ch)}\s*")
            else:
                pieces.append(re.escape(ch))
        return "".join(pieces)
