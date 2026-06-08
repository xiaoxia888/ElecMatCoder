# -*- coding: utf-8 -*-
"""Material rule matcher based on strong/weak/fallback mappings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


GRADE_NORMALIZE_RE = re.compile(r"\bGRADE\b|\bGR\b\.?", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"[\s\-_/;,，；]+")
KEEP_CHAR_RE = re.compile(r"[A-Z0-9#+\u4e00-\u9fff]")
ROMAN_FULL_TO_HALF = str.maketrans({
    "Ⅰ": "I",
    "Ⅱ": "II",
    "Ⅲ": "III",
    "Ⅳ": "IV",
    "Ⅴ": "V",
    "Ⅵ": "VI",
    "Ⅶ": "VII",
    "Ⅷ": "VIII",
    "Ⅸ": "IX",
    "Ⅹ": "X",
})
DEFAULT_MODEL_ONLY_SUFFIX = {
    "CE": ["CE", "ANTI-H2S", "NACE"],
    "ZN": ["ZN", "GALV", "GALVANIZED"],
}
DEFAULT_MODEL_ONLY_GRADE_SUFFIX = {
    "I": ["I", "Gr.I", "GR.I"],
    "II": ["II", "Gr.II", "GR.II"],
    "III": ["III", "Gr.III", "GR.III"],
}
DEFAULT_RULE_POLICY = {
    "force_model_on_multi_candidates": True,
    "clear_result_when_force_model": True,
}


@dataclass
class MaterialRuleHit:
    layer: str
    target: str
    alias: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialRuleMatchResult:
    text: str
    matched: bool
    layer: str | None = None
    candidates: list[str] = field(default_factory=list)
    hits: list[MaterialRuleHit] = field(default_factory=list)
    force_model: bool = False
    force_model_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "matched": self.matched,
            "layer": self.layer,
            "candidates": self.candidates,
            "hits": [hit.to_dict() for hit in self.hits],
            "force_model": self.force_model,
            "force_model_reasons": self.force_model_reasons,
        }


class MaterialRuleMatcher:
    """Match material literals with strong / weak / fallback rule layers."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "material_value_mapping.yaml"
        )
        self._load_config()

    def match(self, text: str) -> MaterialRuleMatchResult:
        force_model_reasons = self._detect_force_model_reasons(text)
        force_model = bool(force_model_reasons)

        combo_hits = self._match_combo_layer(text)
        strong_hits = self._match_layer(text, self.strong_value_mapping, "strong", require_boundary=False)
        weak_hits = self._match_layer(text, self.weak_value_mapping, "weak", require_boundary=True)
        fallback_hits: list[MaterialRuleHit] = []
        chosen_layer: str | None = None
        chosen_hits: list[MaterialRuleHit] = []

        if combo_hits:
            chosen_layer = "combo"
            chosen_hits = combo_hits
        elif strong_hits:
            chosen_layer = "strong"
            chosen_hits = strong_hits
        elif weak_hits:
            chosen_layer = "weak"
            chosen_hits = weak_hits
        else:
            fallback_hits = self._match_layer(
                text, self.fallback_value_mapping, "fallback", require_boundary=True
            )
            if fallback_hits:
                chosen_layer = "fallback"
                chosen_hits = fallback_hits

        candidates = self._dedupe_targets(chosen_hits)
        if self.rule_policy.get("force_model_on_multi_candidates", True) and len(candidates) >= 2:
            force_model = True
            reason = "命中多个不同材质编码，规则不拍板最终结果"
            if reason not in force_model_reasons:
                force_model_reasons.append(reason)
        return MaterialRuleMatchResult(
            text=text,
            matched=bool(chosen_hits),
            layer=chosen_layer,
            candidates=candidates,
            hits=chosen_hits,
            force_model=force_model,
            force_model_reasons=force_model_reasons,
        )

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_strong = data.get("strong_value_mapping", {}) or {}
        raw_weak = data.get("weak_value_mapping", {}) or {}
        raw_fallback = data.get("fallback_value_mapping", {}) or {}
        raw_combo = data.get("combo_value_mapping", {}) or {}

        combo_raw = self._merge_raw_mappings(
            raw_combo,
            self._extract_combo_mapping(raw_strong, raw_weak, raw_fallback),
        )
        strong_raw = self._remove_combo_targets(raw_strong)
        weak_raw = self._remove_combo_targets(raw_weak)
        fallback_raw = self._remove_combo_targets(raw_fallback)

        self.combo_value_mapping = self._prepare_mapping(combo_raw)
        self.strong_value_mapping = self._prepare_mapping(strong_raw)
        self.weak_value_mapping = self._prepare_mapping(weak_raw)
        self.fallback_value_mapping = self._prepare_mapping(fallback_raw)
        raw_rule_policy = data.get("rule_policy", {}) or {}
        self.rule_policy = {
            "force_model_on_multi_candidates": bool(
                raw_rule_policy.get(
                    "force_model_on_multi_candidates",
                    DEFAULT_RULE_POLICY["force_model_on_multi_candidates"],
                )
            ),
            "clear_result_when_force_model": bool(
                raw_rule_policy.get(
                    "clear_result_when_force_model",
                    DEFAULT_RULE_POLICY["clear_result_when_force_model"],
                )
            ),
        }
        self.model_only_suffix = self._prepare_suffix_map(
            data.get("model_only_suffix")
            or data.get("special_req_suffix")
            or DEFAULT_MODEL_ONLY_SUFFIX
        )
        self.model_only_grade_suffix = self._prepare_suffix_map(
            data.get("model_only_grade_suffix")
            or data.get("grade_suffix")
            or DEFAULT_MODEL_ONLY_GRADE_SUFFIX
        )

    @staticmethod
    def _merge_raw_mappings(*raw_mappings: dict[str, list[str]]) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for raw_mapping in raw_mappings:
            for target, aliases in raw_mapping.items():
                target_str = str(target)
                bucket = merged.setdefault(target_str, [])
                if isinstance(aliases, list):
                    bucket.extend(str(alias) for alias in aliases if str(alias).strip())
        return merged

    @staticmethod
    def _extract_combo_mapping(*raw_mappings: dict[str, list[str]]) -> dict[str, list[str]]:
        combo: dict[str, list[str]] = {}
        for raw_mapping in raw_mappings:
            for target, aliases in raw_mapping.items():
                target_str = str(target)
                if "/" not in target_str:
                    continue
                existing = combo.setdefault(target_str, [])
                if isinstance(aliases, list):
                    existing.extend(str(alias) for alias in aliases if str(alias).strip())
        return combo

    @staticmethod
    def _remove_combo_targets(raw_mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        filtered: dict[str, list[str]] = {}
        for target, aliases in raw_mapping.items():
            target_str = str(target)
            if "/" in target_str:
                continue
            filtered[target_str] = aliases
        return filtered

    @staticmethod
    def _prepare_mapping(raw_mapping: dict[str, list[str]]) -> dict[str, list[tuple[str, str]]]:
        prepared: dict[str, list[tuple[str, str]]] = {}
        for target, aliases in raw_mapping.items():
            if not isinstance(aliases, list):
                continue
            unique_pairs: list[tuple[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for alias in aliases:
                alias_str = str(alias).strip()
                if not alias_str:
                    continue
                alias_key = MaterialRuleMatcher._normalize_match_key(alias_str)
                if not alias_key:
                    continue
                pair = (alias_str, alias_key)
                if pair in seen:
                    continue
                seen.add(pair)
                unique_pairs.append(pair)
            unique_pairs.sort(key=lambda item: (-len(item[1]), item[0]))
            prepared[str(target)] = unique_pairs
        return prepared

    @staticmethod
    def _prepare_suffix_map(raw_mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        prepared: dict[str, list[str]] = {}
        for key, values in raw_mapping.items():
            if not isinstance(values, list):
                continue
            normalized = []
            for value in values:
                value_str = str(value).strip()
                if not value_str:
                    continue
                normalized.append(value_str)
            prepared[str(key)] = sorted(set(v for v in normalized if v))
        return prepared

    @staticmethod
    def _normalize_match_key(text: str) -> str:
        upper = text.upper().replace("（", "(").replace("）", ")")
        upper = GRADE_NORMALIZE_RE.sub("GR.", upper)
        chars: list[str] = []
        for ch in upper:
            if KEEP_CHAR_RE.fullmatch(ch):
                chars.append(ch)
        return "".join(chars)

    @staticmethod
    def _build_unbounded_regex(alias: str) -> re.Pattern[str]:
        parts = [re.escape(part) for part in re.split(r"[\s\-_/;,，；()（）]+", alias) if part]
        if not parts:
            return re.compile(r"$^")
        body = r"[\s\-_/;,，；()（）]*".join(parts)
        return re.compile(rf"({body})", re.IGNORECASE)

    def _match_layer(
        self,
        text: str,
        mapping: dict[str, list[tuple[str, str]]],
        layer: str,
        require_boundary: bool,
    ) -> list[MaterialRuleHit]:
        if require_boundary:
            return self._match_boundary_layer(text, mapping, layer)

        raw_hits: list[MaterialRuleHit] = []

        for target, aliases in mapping.items():
            for alias, alias_key in aliases:
                pattern = self._build_unbounded_regex(alias)
                for match in pattern.finditer(text):
                    orig_start = match.start(1)
                    orig_end = match.end(1)
                    raw_hits.append(
                        MaterialRuleHit(
                            layer=layer,
                            target=target,
                            alias=alias,
                            start=orig_start,
                            end=orig_end,
                            text=text[orig_start:orig_end],
                        )
                    )

        if not raw_hits:
            return []

        # Strong layer should prefer the most specific literal, not the YAML order.
        # We first prefer earlier hits, and for the same start prefer the longest alias/span.
        raw_hits.sort(
            key=lambda hit: (
                hit.start,
                -(hit.end - hit.start),
                -len(self._normalize_match_key(hit.alias)),
                hit.end,
            )
        )

        chosen_hits: list[MaterialRuleHit] = []
        occupied: list[tuple[int, int]] = []
        for hit in raw_hits:
            if self._overlaps(occupied, hit.start, hit.end):
                continue
            occupied.append((hit.start, hit.end))
            chosen_hits.append(hit)

        chosen_hits.sort(key=lambda hit: (hit.start, hit.end))
        return chosen_hits

    @staticmethod
    def _build_combo_regex(alias: str) -> re.Pattern[str]:
        pattern = re.escape(alias.strip())
        pattern = pattern.replace(r"\ ", r"\s*")
        pattern = pattern.replace(r"\/", r"\s*/\s*")
        pattern = pattern.replace(r"\+", r"\s*\+\s*")
        pattern = pattern.replace(r"\-", r"\s*-\s*")
        return re.compile(f"({pattern})", re.IGNORECASE)

    def _match_combo_layer(self, text: str) -> list[MaterialRuleHit]:
        hits: list[MaterialRuleHit] = []
        occupied: list[tuple[int, int]] = []
        for target, aliases in self.combo_value_mapping.items():
            for alias, _alias_key in aliases:
                pattern = self._build_combo_regex(alias)
                for match in pattern.finditer(text):
                    start, end = match.start(1), match.end(1)
                    if self._overlaps(occupied, start, end):
                        continue
                    occupied.append((start, end))
                    hits.append(
                        MaterialRuleHit(
                            layer="combo",
                            target=target,
                            alias=alias,
                            start=start,
                            end=end,
                            text=text[start:end],
                        )
                    )
        hits.sort(key=lambda hit: (hit.start, -(hit.end - hit.start), hit.end))
        chosen_hits: list[MaterialRuleHit] = []
        final_occupied: list[tuple[int, int]] = []
        for hit in hits:
            if self._overlaps(final_occupied, hit.start, hit.end):
                continue
            final_occupied.append((hit.start, hit.end))
            chosen_hits.append(hit)
        chosen_hits.sort(key=lambda hit: (hit.start, hit.end))
        return chosen_hits

    @staticmethod
    def _build_boundary_regex(alias: str) -> re.Pattern[str]:
        parts = [re.escape(part) for part in re.split(r"[\s\-_/;,，；()（）]+", alias) if part]
        if not parts:
            return re.compile(r"$^")
        body = r"[\s\-_/;,，；()（）]*".join(parts)
        pattern = rf"(?<![A-Za-z0-9\u4e00-\u9fff])({body})(?![A-Za-z0-9\u4e00-\u9fff])"
        return re.compile(pattern, re.IGNORECASE)

    def _match_boundary_layer(
        self,
        text: str,
        mapping: dict[str, list[tuple[str, str]]],
        layer: str,
    ) -> list[MaterialRuleHit]:
        hits: list[MaterialRuleHit] = []
        occupied: list[tuple[int, int]] = []

        for target, aliases in mapping.items():
            for alias, _alias_key in aliases:
                pattern = self._build_boundary_regex(alias)
                for match in pattern.finditer(text):
                    start, end = match.start(1), match.end(1)
                    if self._overlaps(occupied, start, end):
                        continue
                    occupied.append((start, end))
                    hits.append(
                        MaterialRuleHit(
                            layer=layer,
                            target=target,
                            alias=alias,
                            start=start,
                            end=end,
                            text=text[start:end],
                        )
                    )

        hits.sort(key=lambda hit: (hit.start, hit.end))
        return hits

    @staticmethod
    def _overlaps(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
        for occ_start, occ_end in occupied:
            if not (end <= occ_start or start >= occ_end):
                return True
        return False

    def _detect_force_model_reasons(self, text: str) -> list[str]:
        reasons: list[str] = []
        normalized_text = text.translate(ROMAN_FULL_TO_HALF)

        for suffix_name, variants in self.model_only_suffix.items():
            for variant in variants:
                if variant and self._contains_suffix_variant(normalized_text, variant, require_boundary=False):
                    reasons.append(f"命中后缀语义 {suffix_name}")
                    break

        for grade_name, variants in self.model_only_grade_suffix.items():
            for variant in variants:
                if variant and self._contains_suffix_variant(normalized_text, variant, require_boundary=True):
                    reasons.append(f"命中等级语义 {grade_name}")
                    break

        return reasons

    @staticmethod
    def _contains_suffix_variant(text: str, variant: str, require_boundary: bool) -> bool:
        pattern = (
            MaterialRuleMatcher._build_boundary_regex(variant)
            if require_boundary
            else MaterialRuleMatcher._build_unbounded_regex(variant)
        )
        return bool(pattern.search(text))

    @staticmethod
    def _dedupe_targets(hits: list[MaterialRuleHit]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.target not in seen:
                seen.add(hit.target)
                ordered.append(hit.target)
        return ordered
