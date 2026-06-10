from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

from src.encoder.processors.rule_extraction import (
    build_structured_rule_entities,
    extract_size_and_thickness_by_rules,
)

logger = logging.getLogger(__name__)


def _is_size_empty_for_rule_fallback(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return not any(value.get(key) for key in ("DN", "OD", "INCH"))


def _is_thickness_empty_for_rule_fallback(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return not any(value.get(key) for key in ("MM", "SCHEDULE", "BWG", "INCH"))


def _is_pressure_empty_for_rule_fallback(value: Any) -> bool:
    return value in (None, "", [], {})


def _copy_structural_field(value: Any) -> Any:
    return copy.deepcopy(value)


def _build_prompt_context(size_value: Any, thickness_value: Any) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    if not _is_size_empty_for_rule_fallback(size_value):
        context["SIZE"] = _copy_structural_field(size_value)
    if not _is_thickness_empty_for_rule_fallback(thickness_value):
        context["THICKNESS"] = _copy_structural_field(thickness_value)
    return context


class StructuralFieldResolver:
    """
    结构字段统一决策器。

    规则：
    1. 可通过配置决定是否先走规则层
    2. SIZE 为空时回退大模型；仅有 LENGTH 也视为 SIZE 为空
    3. THICKNESS 为空时回退大模型
    4. PRESSURE 只有在 THICKNESS 和 PRESSURE 都为空时才回退大模型
    """

    def __init__(
        self,
        *,
        prompt_extractor: Optional[Any] = None,
        rule_config: Optional[Dict[str, Any]] = None,
    ):
        self.prompt_extractor = prompt_extractor
        self.rule_config = copy.deepcopy(rule_config or {})
        self.rules_enabled = bool(self.rule_config.get("enabled", False))

    @classmethod
    def from_configs(
        cls,
        *,
        prompt_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
    ) -> "StructuralFieldResolver":
        prompt_extractor = None
        prompt_cfg = copy.deepcopy(prompt_config or {})
        if prompt_cfg.get("enabled", False):
            from .structural_prompt_extractor import StructuralPromptExtractor

            prompt_extractor = StructuralPromptExtractor(prompt_cfg)
        return cls(prompt_extractor=prompt_extractor, rule_config=rule_config)

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        raw_text = str(text or "")
        if not raw_text.strip():
            return None

        rule_structural: Optional[Dict[str, Any]] = None
        sources: Dict[str, str] = {}

        if self.rules_enabled:
            rule_result = extract_size_and_thickness_by_rules(raw_text)
            rule_structural = build_structured_rule_entities(rule_result, original_text=raw_text)
            for field in ("SIZE", "THICKNESS", "PRESSURE"):
                sources[field] = "rule_extraction"

        if not self.prompt_extractor:
            if rule_structural is None:
                return None
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            return merged

        if rule_structural is None:
            prompt_structural = self.prompt_extractor.extract_with_context(raw_text)
            if not isinstance(prompt_structural, dict):
                return None
            prompt_structural["_sources"] = {
                "SIZE": "prompt_extraction",
                "THICKNESS": "prompt_extraction",
                "PRESSURE": "prompt_extraction",
            }
            return prompt_structural

        size_value = rule_structural.get("SIZE")
        thickness_value = rule_structural.get("THICKNESS")
        pressure_value = rule_structural.get("PRESSURE")

        need_size_model = _is_size_empty_for_rule_fallback(size_value)
        need_thickness_model = _is_thickness_empty_for_rule_fallback(thickness_value)
        need_pressure_model = need_thickness_model and _is_pressure_empty_for_rule_fallback(pressure_value)

        if not (need_size_model or need_thickness_model or need_pressure_model):
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            return merged

        prompt_context = _build_prompt_context(size_value, thickness_value)
        prompt_structural = self.prompt_extractor.extract_with_context(raw_text, context=prompt_context)
        if not isinstance(prompt_structural, dict):
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            return merged

        merged = copy.deepcopy(rule_structural)
        merged["_raw"] = str(prompt_structural.get("_raw", "") or "")
        merged["_status"] = copy.deepcopy(prompt_structural.get("_status", {}) or {})
        merged["_errors"] = copy.deepcopy(prompt_structural.get("_errors", {}) or {})

        if need_size_model:
            sources["SIZE"] = "prompt_extraction"
            prompt_size = prompt_structural.get("SIZE")
            if not _is_size_empty_for_rule_fallback(prompt_size):
                merged["SIZE"] = _copy_structural_field(prompt_size)

        if need_thickness_model:
            sources["THICKNESS"] = "prompt_extraction"
            prompt_thickness = prompt_structural.get("THICKNESS")
            if not _is_thickness_empty_for_rule_fallback(prompt_thickness):
                merged["THICKNESS"] = _copy_structural_field(prompt_thickness)

        if need_pressure_model:
            sources["PRESSURE"] = "prompt_extraction"
            prompt_pressure = prompt_structural.get("PRESSURE")
            if not _is_pressure_empty_for_rule_fallback(prompt_pressure):
                merged["PRESSURE"] = _copy_structural_field(prompt_pressure)

        merged["_sources"] = sources
        return merged
