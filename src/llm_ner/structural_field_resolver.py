from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

from src.llm_ner.complex_structural_trigger import ComplexStructuralTrigger
from src.encoder.processors.rule_extraction import (
    build_structured_rule_entities,
    extract_size_and_thickness_by_rules,
)
from src.domain.structural_v2 import (
    has_v2_pressure,
    has_v2_size,
    has_v2_thickness,
    is_structural_v2,
)

logger = logging.getLogger(__name__)


def _is_size_empty_for_rule_fallback(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return not bool(value.get("_ITEMS"))


def _is_thickness_empty_for_rule_fallback(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return not bool(value.get("_ITEMS"))


def _is_pressure_empty_for_rule_fallback(value: Any) -> bool:
    return value in (None, "", [], {})


def _copy_structural_field(value: Any) -> Any:
    return copy.deepcopy(value)


def _clean_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(config or {})
    cleaned.pop("enabled", None)
    cleaned.pop("share_group", None)
    return cleaned


def _build_complex_prompt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config or {})
    if not cfg.get("enabled", False):
        return cfg
    cfg.setdefault("backend", "openai_compatible")
    cfg.setdefault("api", "chat_completions")
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("max_tokens", 1536)
    cfg.setdefault("timeout", 60)
    cfg.setdefault("prompt_version", "lined_jacketed_v1")
    cfg.setdefault("thinking_mode", "disabled")
    cfg.setdefault("reasoning_effort", "high")
    cfg.setdefault("max_workers", 1)
    return cfg


def _summarize_prompt_diagnostics(value: Any, *, max_len: int = 800) -> Any:
    if not isinstance(value, dict):
        return value
    summarized: Dict[str, Any] = {}
    for key, item in value.items():
        text = str(item)
        summarized[key] = text if len(text) <= max_len else text[:max_len] + "...[truncated]"
    return summarized


def _summarize_structural_result(value: Dict[str, Any]) -> Dict[str, Any]:
    if is_structural_v2(value):
        return {
            "schema_version": "v2",
            "ITEMS": value.get("ITEMS") or [],
            "LENGTH": value.get("LENGTH") or "",
            "PRESSURE": value.get("PRESSURE") or "",
            "sources": value.get("_sources") or {},
        }
    return {
        "SIZE_ITEMS": value.get("SIZE_ITEMS") or [],
        "LENGTH": value.get("LENGTH") or "",
        "THICKNESS_ITEMS": value.get("THICKNESS_ITEMS") or [],
        "PRESSURE": value.get("PRESSURE") or "",
        "sources": value.get("_sources") or {},
        "complex_structure": value.get("_complex_structure") or {},
    }


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


def _structural_extractor_source(config: Dict[str, Any]) -> str:
    return "prompt_extraction" if _uses_prompt_orchestrator(config) else "finetuned_structural_model"


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
        complex_prompt_config: Optional[Dict[str, Any]] = None,
    ):
        self.extractor_groups = list(extractor_groups or [])
        self.rule_config = copy.deepcopy(rule_config or {})
        self.rules_enabled = bool(self.rule_config.get("enabled", False))
        self.complex_prompt_config = _build_complex_prompt_config(complex_prompt_config or {})
        self.complex_prompt_enabled = bool(self.complex_prompt_config.get("enabled", False))
        self.complex_trigger = (
            ComplexStructuralTrigger.from_config(self.complex_prompt_config)
            if self.complex_prompt_enabled
            else None
        )
        self._complex_extractor: Optional[Any] = None

    @classmethod
    def from_configs(
        cls,
        *,
        size_model_config: Optional[Dict[str, Any]] = None,
        thickness_model_config: Optional[Dict[str, Any]] = None,
        pressure_model_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
        complex_prompt_config: Optional[Dict[str, Any]] = None,
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
                        "source": _structural_extractor_source(extractor_config),
                    }
                )
        return cls(
            extractor_groups=extractor_groups,
            rule_config=rule_config,
            complex_prompt_config=complex_prompt_config,
        )

    def _get_complex_extractor(self) -> Any:
        if not self.complex_prompt_enabled:
            return None
        if self._complex_extractor is None:
            from .structural_prompt_extractor import StructuralPromptExtractor

            self._complex_extractor = StructuralPromptExtractor(self.complex_prompt_config)
        return self._complex_extractor

    def _extract_complex_prompt(self, raw_text: str, trigger_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        extractor = self._get_complex_extractor()
        if extractor is None:
            return None
        try:
            result = extractor.extract_with_context(
                raw_text,
                context=None,
                run_size_length=True,
                run_thickness=True,
                run_pressure=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[复杂结构提示词] 调用失败，回退普通结构字段路径: %s", exc)
            return None
        if not isinstance(result, dict):
            return None
        has_value = (
            has_v2_size(result) or has_v2_thickness(result) or has_v2_pressure(result)
            if is_structural_v2(result)
            else (
                not _is_size_empty_for_rule_fallback(result.get("SIZE"))
                or not _is_thickness_empty_for_rule_fallback(result.get("THICKNESS"))
                or not _is_pressure_empty_for_rule_fallback(result.get("PRESSURE"))
            )
        )
        if not has_value:
            logger.warning(
                "[复杂结构提示词] 命中触发但未抽取到结构字段，回退普通结构字段路径: "
                "trigger=%s, status=%s, errors=%s",
                trigger_info,
                _summarize_prompt_diagnostics(result.get("_status") or {}),
                _summarize_prompt_diagnostics(result.get("_errors") or {}),
            )
            return None
        result["_sources"] = {
            field: "complex_prompt_extraction"
            for field, present in (
                ("SIZE", has_v2_size(result) if is_structural_v2(result) else not _is_size_empty_for_rule_fallback(result.get("SIZE"))),
                ("THICKNESS", has_v2_thickness(result) if is_structural_v2(result) else not _is_thickness_empty_for_rule_fallback(result.get("THICKNESS"))),
                ("PRESSURE", has_v2_pressure(result) if is_structural_v2(result) else not _is_pressure_empty_for_rule_fallback(result.get("PRESSURE"))),
            )
            if present
        }
        result["_complex_structural_trigger"] = trigger_info
        return result

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        raw_text = str(text or "")
        if not raw_text.strip():
            return None

        if self.complex_trigger is not None:
            trigger = self.complex_trigger.match(raw_text)
            if trigger is not None:
                logger.debug("[结构字段路径] 命中复杂结构触发，优先走复杂提示词: %s", trigger.to_dict())
                complex_result = self._extract_complex_prompt(raw_text, trigger.to_dict())
                if complex_result is not None:
                    logger.debug("[结构字段路径] 使用复杂提示词结果: %s", _summarize_structural_result(complex_result))
                    return complex_result
                logger.debug("[结构字段路径] 复杂提示词无可用结果，继续普通结构字段路径")

        rule_structural: Optional[Dict[str, Any]] = None
        sources: Dict[str, str] = {}

        if self.rules_enabled:
            rule_result = extract_size_and_thickness_by_rules(raw_text)
            rule_structural = build_structured_rule_entities(rule_result, original_text=raw_text)
            for field in ("SIZE", "THICKNESS", "PRESSURE"):
                sources[field] = "rule_extraction"
            logger.debug("[结构字段路径] 规则器结果: %s", _summarize_structural_result(rule_structural))

        if not self.extractor_groups:
            if rule_structural is None:
                return None
            merged = copy.deepcopy(rule_structural)
            merged["_sources"] = sources
            merged["_raw"] = ""
            merged["_status"] = {}
            merged["_errors"] = {}
            logger.debug("[结构字段路径] 最终使用规则器结果: %s", _summarize_structural_result(merged))
            return merged

        if rule_structural is None:
            field_context: Dict[str, Any] = {}
            model_structural: Dict[str, Any] = {
                "_raw": "",
                "_status": {},
                "_errors": {},
                "_usage": {},
                "_extract_confidence_v2": {},
            }
            sources = {}

            for group in self.extractor_groups:
                group_source = str(group.get("source") or "prompt_extraction")
                partial_result = group["extractor"].extract_with_context(
                    raw_text,
                    context=field_context,
                    run_size_length="SIZE" in group["fields"],
                    run_thickness="THICKNESS" in group["fields"],
                    run_pressure="PRESSURE" in group["fields"],
                )
                if not isinstance(partial_result, dict):
                    continue

                if is_structural_v2(partial_result):
                    partial_result = copy.deepcopy(partial_result)
                    partial_result["_sources"] = {
                        field: group_source
                        for field, present in (
                            ("SIZE", has_v2_size(partial_result)),
                            ("THICKNESS", has_v2_thickness(partial_result)),
                            ("PRESSURE", has_v2_pressure(partial_result)),
                        )
                        if present
                    }
                    logger.debug(
                        "[结构字段路径] 使用V2模型结果，不转换为旧结构: %s",
                        _summarize_structural_result(partial_result),
                    )
                    return partial_result

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
                model_structural["_extract_confidence_v2"].update(
                    copy.deepcopy(partial_result.get("_extract_confidence_v2", {}) or {})
                )

                if "SIZE" in group["fields"] and not _is_size_empty_for_rule_fallback(partial_result.get("SIZE")):
                    model_structural["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                    field_context["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                    sources["SIZE"] = group_source
                if "THICKNESS" in group["fields"] and not _is_thickness_empty_for_rule_fallback(partial_result.get("THICKNESS")):
                    model_structural["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                    field_context["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                    sources["THICKNESS"] = group_source
                if "PRESSURE" in group["fields"] and not _is_pressure_empty_for_rule_fallback(partial_result.get("PRESSURE")):
                    model_structural["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                    field_context["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                    sources["PRESSURE"] = group_source

            if not any(key in model_structural for key in ("SIZE", "THICKNESS", "PRESSURE")):
                return None
            model_structural["_sources"] = sources
            logger.debug("[结构字段路径] 无规则器，最终使用模型/提示词结果: %s", _summarize_structural_result(model_structural))
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
            logger.debug("[结构字段路径] 规则器字段完整，无需模型补提: %s", _summarize_structural_result(merged))
            return merged

        field_context = _build_field_context(size_value, thickness_value, pressure_value)
        model_structural: Dict[str, Any] = {
            "_raw": "",
            "_status": {},
            "_errors": {},
            "_usage": {},
            "_field_sources": {},
            "_extract_confidence_v2": {},
        }

        for group in self.extractor_groups:
            fields = set(group.get("fields") or [])
            group_source = str(group.get("source") or "prompt_extraction")
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

            if is_structural_v2(partial_result):
                partial_result = copy.deepcopy(partial_result)
                partial_result["_sources"] = {
                    field: group_source
                    for field, present in (
                        ("SIZE", has_v2_size(partial_result)),
                        ("THICKNESS", has_v2_thickness(partial_result)),
                        ("PRESSURE", has_v2_pressure(partial_result)),
                    )
                    if present
                }
                logger.debug(
                    "[结构字段路径] 模型返回V2，V2整体优先于旧规则结果: %s",
                    _summarize_structural_result(partial_result),
                )
                return partial_result

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
            model_structural["_extract_confidence_v2"].update(
                copy.deepcopy(partial_result.get("_extract_confidence_v2", {}) or {})
            )

            if run_size and not _is_size_empty_for_rule_fallback(partial_result.get("SIZE")):
                model_structural["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                field_context["SIZE"] = _copy_structural_field(partial_result.get("SIZE"))
                model_structural["_field_sources"]["SIZE"] = group_source
            if run_thickness and not _is_thickness_empty_for_rule_fallback(partial_result.get("THICKNESS")):
                model_structural["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                field_context["THICKNESS"] = _copy_structural_field(partial_result.get("THICKNESS"))
                model_structural["_field_sources"]["THICKNESS"] = group_source
            if run_pressure and not _is_pressure_empty_for_rule_fallback(partial_result.get("PRESSURE")):
                model_structural["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                field_context["PRESSURE"] = _copy_structural_field(partial_result.get("PRESSURE"))
                model_structural["_field_sources"]["PRESSURE"] = group_source

        merged = copy.deepcopy(rule_structural)
        merged["_raw"] = str(model_structural.get("_raw", "") or "")
        merged["_status"] = copy.deepcopy(model_structural.get("_status", {}) or {})
        merged["_errors"] = copy.deepcopy(model_structural.get("_errors", {}) or {})
        merged["_extract_confidence_v2"] = copy.deepcopy(
            model_structural.get("_extract_confidence_v2", {}) or {}
        )

        if need_size_model:
            prompt_size = model_structural.get("SIZE")
            if not _is_size_empty_for_rule_fallback(prompt_size):
                merged["SIZE"] = _copy_structural_field(prompt_size)
                sources["SIZE"] = model_structural.get("_field_sources", {}).get("SIZE", "finetuned_structural_model")

        if need_thickness_model:
            prompt_thickness = model_structural.get("THICKNESS")
            if not _is_thickness_empty_for_rule_fallback(prompt_thickness):
                merged["THICKNESS"] = _copy_structural_field(prompt_thickness)
                sources["THICKNESS"] = model_structural.get("_field_sources", {}).get("THICKNESS", "finetuned_structural_model")

        if need_pressure_model:
            prompt_pressure = model_structural.get("PRESSURE")
            if not _is_pressure_empty_for_rule_fallback(prompt_pressure):
                merged["PRESSURE"] = _copy_structural_field(prompt_pressure)
                sources["PRESSURE"] = model_structural.get("_field_sources", {}).get("PRESSURE", "finetuned_structural_model")

        merged["_sources"] = sources
        logger.debug("[结构字段路径] 规则器+模型补提合并结果: %s", _summarize_structural_result(merged))
        return merged
