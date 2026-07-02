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


def _clean_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(config or {})
    cleaned.pop("enabled", None)
    cleaned.pop("share_group", None)
    return cleaned


def _flatten_config_keys(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (config or {}).items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_config_keys(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _build_shared_group_mismatch_message(
    *,
    share_group: str,
    previous_field: str,
    current_field: str,
    previous_config: Dict[str, Any],
    current_config: Dict[str, Any],
) -> str:
    prev_flat = _flatten_config_keys(previous_config)
    curr_flat = _flatten_config_keys(current_config)

    missing_in_current = sorted(key for key in prev_flat.keys() if key not in curr_flat)
    extra_in_current = sorted(key for key in curr_flat.keys() if key not in prev_flat)
    different_values = sorted(
        key for key in prev_flat.keys() & curr_flat.keys()
        if prev_flat[key] != curr_flat[key]
    )

    parts = [
        f"结构字段共享模型组“{share_group}”配置不一致，不能共用一次调用。",
        f"已存在字段: {previous_field}；当前字段: {current_field}。",
    ]
    if missing_in_current:
        parts.append(f"{current_field} 缺少配置: {', '.join(missing_in_current)}。")
    if extra_in_current:
        parts.append(f"{current_field} 额外配置了: {', '.join(extra_in_current)}。")
    if different_values:
        parts.append(f"以下键取值不同: {', '.join(different_values)}。")
    return " ".join(parts)


def _build_field_context(size_value: Any, thickness_value: Any, pressure_value: Any) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    if not _is_size_empty_for_rule_fallback(size_value):
        context["SIZE"] = _copy_structural_field(size_value)
    if not _is_thickness_empty_for_rule_fallback(thickness_value):
        context["THICKNESS"] = _copy_structural_field(thickness_value)
    if not _is_pressure_empty_for_rule_fallback(pressure_value):
        context["PRESSURE"] = _copy_structural_field(pressure_value)
    return context


def _uses_prompt_orchestrator(config: Dict[str, Any]) -> bool:
    backend = str((config or {}).get("backend") or "").strip()
    if backend == "openai_compatible":
        return True
    if backend == "ollama":
        return any(key in (config or {}) for key in ("prompt_version", "thickness_prompt_version"))
    return False


class StructuralFieldResolver:
    """
    结构字段统一决策器。

    规则：
    1. 可通过配置决定是否先走规则层
    2. SIZE 为空时回退大模型；仅有 LENGTH 也视为 SIZE 为空
    3. THICKNESS 为空时回退大模型
    4. PRESSURE 在 THICKNESS 和 PRESSURE 都为空时回退大模型；
       如果规则主动清空 PRESSURE，也单独回退大模型
    """

    def __init__(
        self,
        *,
        extractor_groups: Optional[list[Dict[str, Any]]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
    ):
        self.extractor_groups = list(extractor_groups or [])
        self.rule_config = copy.deepcopy(rule_config or {})
        self.rules_enabled = bool(self.rule_config.get("enabled", False))

    @classmethod
    def from_configs(
        cls,
        *,
        size_model_config: Optional[Dict[str, Any]] = None,
        thickness_model_config: Optional[Dict[str, Any]] = None,
        pressure_model_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
    ) -> "StructuralFieldResolver":
        from .structural_field_model_extractor import StructuralFieldModelExtractor
        from .structural_prompt_extractor import StructuralPromptExtractor

        extractor_groups: list[Dict[str, Any]] = []

        field_model_configs = {
            "SIZE": copy.deepcopy(size_model_config or {}),
            "THICKNESS": copy.deepcopy(thickness_model_config or {}),
            "PRESSURE": copy.deepcopy(pressure_model_config or {}),
        }
        has_field_level_config = any(field_model_configs.values())

        if has_field_level_config:
            grouped_fields: Dict[str, Dict[str, Any]] = {}
            for field, cfg in field_model_configs.items():
                if not cfg or not cfg.get("enabled", False):
                    continue
                share_group = str(cfg.get("share_group") or "").strip()
                group_key = share_group or f"__{field}__"
                if group_key not in grouped_fields:
                    grouped_fields[group_key] = {
                        "config": copy.deepcopy(cfg),
                        "fields": [],
                        "first_field": field,
                    }
                elif share_group and _clean_model_config(grouped_fields[group_key]["config"]) != _clean_model_config(cfg):
                    raise RuntimeError(
                        _build_shared_group_mismatch_message(
                            share_group=share_group,
                            previous_field=str(grouped_fields[group_key]["first_field"]),
                            current_field=field,
                            previous_config=_clean_model_config(grouped_fields[group_key]["config"]),
                            current_config=_clean_model_config(cfg),
                        )
                    )
                grouped_fields[group_key]["fields"].append(field)
            for grouped in grouped_fields.values():
                extractor_config = grouped["config"]
                extractor_cls = (
                    StructuralPromptExtractor
                    if _uses_prompt_orchestrator(extractor_config)
                    else StructuralFieldModelExtractor
                )
                extractor_groups.append(
                    {
                        "fields": list(grouped["fields"]),
                        "extractor": extractor_cls(extractor_config),
                    }
                )
        return cls(extractor_groups=extractor_groups, rule_config=rule_config)

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

        if not self.extractor_groups:
            if rule_structural is None:
                return None
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            return merged

        if rule_structural is None:
            field_context: Dict[str, Any] = {}
            model_structural: Dict[str, Any] = {
                "_raw": "",
                "_status": {},
                "_errors": {},
                "_usage": {},
            }
            sources = {}

            for group in self.extractor_groups:
                partial_result = group["extractor"].extract_with_context(
                    raw_text,
                    context=field_context,
                    run_size_length="SIZE" in group["fields"],
                    run_thickness="THICKNESS" in group["fields"],
                    run_pressure="PRESSURE" in group["fields"],
                )
                if not isinstance(partial_result, dict):
                    continue

                raw_chunk = str(partial_result.get("_raw", "") or "")
                if raw_chunk:
                    model_structural["_raw"] = (
                        f'{model_structural["_raw"]}\n\n{raw_chunk}'.strip()
                        if model_structural["_raw"]
                        else raw_chunk
                    )
                model_structural["_status"].update(copy.deepcopy(partial_result.get("_status", {}) or {}))
                model_structural["_errors"].update(copy.deepcopy(partial_result.get("_errors", {}) or {}))
                model_structural["_usage"].update(copy.deepcopy(partial_result.get("_usage", {}) or {}))

                if "SIZE" in group["fields"] and not _is_size_empty_for_rule_fallback(partial_result.get("SIZE")):
                    model_structural["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                    field_context["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                    sources["SIZE"] = "prompt_extraction"
                if "THICKNESS" in group["fields"] and not _is_thickness_empty_for_rule_fallback(partial_result.get("THICKNESS")):
                    model_structural["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                    field_context["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                    sources["THICKNESS"] = "prompt_extraction"
                if "PRESSURE" in group["fields"] and not _is_pressure_empty_for_rule_fallback(partial_result.get("PRESSURE")):
                    model_structural["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                    field_context["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                    sources["PRESSURE"] = "prompt_extraction"

            if not any(key in model_structural for key in ("SIZE", "THICKNESS", "PRESSURE")):
                return None
            model_structural["_sources"] = sources
            return model_structural

        size_value = rule_structural.get("SIZE")
        thickness_value = rule_structural.get("THICKNESS")
        pressure_value = rule_structural.get("PRESSURE")
        rule_flags = rule_structural.get("_rule_flags") if isinstance(rule_structural.get("_rule_flags"), dict) else {}
        pressure_rule_flag = rule_flags.get("PRESSURE") if isinstance(rule_flags.get("PRESSURE"), dict) else {}
        pressure_was_cleared = bool(pressure_rule_flag.get("cleared"))

        need_size_model = _is_size_empty_for_rule_fallback(size_value)
        need_thickness_model = _is_thickness_empty_for_rule_fallback(thickness_value)
        need_pressure_model = _is_pressure_empty_for_rule_fallback(pressure_value) and (
            need_thickness_model or pressure_was_cleared
        )

        if not (need_size_model or need_thickness_model or need_pressure_model):
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            return merged

        field_context = _build_field_context(size_value, thickness_value, pressure_value)
        model_structural: Dict[str, Any] = {
            "_raw": "",
            "_status": {},
            "_errors": {},
            "_usage": {},
        }

        for group in self.extractor_groups:
            fields = set(group.get("fields") or [])
            run_size = need_size_model and "SIZE" in fields
            run_thickness = need_thickness_model and "THICKNESS" in fields
            run_pressure = need_pressure_model and "PRESSURE" in fields
            if not (run_size or run_thickness or run_pressure):
                continue

            partial_result = group["extractor"].extract_with_context(
                raw_text,
                context=field_context,
                run_size_length=run_size,
                run_thickness=run_thickness,
                run_pressure=run_pressure,
            )
            if not isinstance(partial_result, dict):
                continue

            raw_chunk = str(partial_result.get("_raw", "") or "")
            if raw_chunk:
                model_structural["_raw"] = (
                    f'{model_structural["_raw"]}\n\n{raw_chunk}'.strip()
                    if model_structural["_raw"]
                    else raw_chunk
                )
            model_structural["_status"].update(copy.deepcopy(partial_result.get("_status", {}) or {}))
            model_structural["_errors"].update(copy.deepcopy(partial_result.get("_errors", {}) or {}))
            model_structural["_usage"].update(copy.deepcopy(partial_result.get("_usage", {}) or {}))

            if run_size and not _is_size_empty_for_rule_fallback(partial_result.get("SIZE")):
                model_structural["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                field_context["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
            if run_thickness and not _is_thickness_empty_for_rule_fallback(partial_result.get("THICKNESS")):
                model_structural["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                field_context["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
            if run_pressure and not _is_pressure_empty_for_rule_fallback(partial_result.get("PRESSURE")):
                model_structural["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                field_context["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))

        merged = copy.deepcopy(rule_structural)
        merged["_raw"] = str(model_structural.get("_raw", "") or "")
        merged["_status"] = copy.deepcopy(model_structural.get("_status", {}) or {})
        merged["_errors"] = copy.deepcopy(model_structural.get("_errors", {}) or {})

        if need_size_model:
            sources["SIZE"] = "prompt_extraction"
            prompt_size = model_structural.get("SIZE")
            if not _is_size_empty_for_rule_fallback(prompt_size):
                merged["SIZE"] = _copy_structural_field(prompt_size)

        if need_thickness_model:
            sources["THICKNESS"] = "prompt_extraction"
            prompt_thickness = model_structural.get("THICKNESS")
            if not _is_thickness_empty_for_rule_fallback(prompt_thickness):
                merged["THICKNESS"] = _copy_structural_field(prompt_thickness)

        if need_pressure_model:
            sources["PRESSURE"] = "prompt_extraction"
            prompt_pressure = model_structural.get("PRESSURE")
            if not _is_pressure_empty_for_rule_fallback(prompt_pressure):
                merged["PRESSURE"] = _copy_structural_field(prompt_pressure)

        merged["_sources"] = sources
        return merged
