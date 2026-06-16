# -*- coding: utf-8 -*-
"""Unified routing pipeline: stage1 -> stage2 -> project-frequency."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from .difficulty_levels import DIFF_EASY, DIFF_HARD, DIFF_SECOND_EASY, normalize_difficulty_level
from .platform_integration import build_base_difficulty
from .project_frequency_detector import ProjectFrequencyDetector
from .second_pass import PlatformSecondPassRunner

_second_pass_runner = PlatformSecondPassRunner()
_project_frequency_detector = ProjectFrequencyDetector()


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def _resolve_routing_text(result_dict: dict[str, Any]) -> str:
    if not isinstance(result_dict, dict):
        return ""
    processed_text = _clean(result_dict.get("processed_text"))
    if processed_text:
        return processed_text
    return _clean(result_dict.get("original_text"))


def _dedupe_checks(checks: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for check in checks:
        field = _clean(check.get("field"))
        reason = _clean(check.get("reason"))
        stage = _clean(check.get("stage"))
        rule = _clean(check.get("rule"))
        key = (field, reason, stage, rule)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "field": field,
                "reason": reason,
                "stage": stage,
                "rule": rule,
            }
        )
    return result


def _make_check(field: str, reason: str, *, stage: str, rule: str) -> dict[str, str]:
    return {
        "field": _clean(field),
        "reason": _clean(reason),
        "stage": _clean(stage),
        "rule": _clean(rule),
    }


def _field_from_code_group(code_group: str, default_field: str = "OTHER") -> str:
    normalized = _clean(code_group).lower()
    mapping = {
        "type_code": "TYPE",
        "material_code": "MATERIAL",
        "standard_code": "STANDARD",
        "structure": "STRUCTURE",
        "naked_spec": "STRUCTURE",
    }
    return mapping.get(normalized, default_field)


def _stage1_failed_checks(features: Any) -> list[dict[str, str]]:
    if not isinstance(features, list):
        return []
    checks: list[dict[str, str]] = []
    for feature in features:
        if not isinstance(feature, dict) or not feature.get("matched"):
            continue
        name = _clean(feature.get("name"))
        reason = _clean(feature.get("reason"))
        hits = feature.get("hits") if isinstance(feature.get("hits"), list) else []

        if name == "type_glue":
            checks.append(_make_check("TYPE", reason, stage="stage1", rule=name))
            continue
        if name == "standard_glue":
            checks.append(_make_check("STANDARD", reason, stage="stage1", rule=name))
            continue
        if name == "standard_completeness":
            checks.append(_make_check("STANDARD", reason, stage="stage1", rule=name))
            continue
        if name == "special_token":
            checks.append(_make_check("SPECIAL", reason, stage="stage1", rule=name))
            continue

        if hits:
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                note = _clean(hit.get("note")) or reason
                code = _clean(hit.get("code"))
                code_group = _clean(hit.get("code_group"))
                if name == "structure_completeness":
                    field = "SIZE" if code == "SIZE" else "THICKNESS/PRESSURE"
                elif name == "uncommon_code":
                    field = _field_from_code_group(code_group)
                elif name == "project_frequency":
                    field = _field_from_code_group(code_group)
                else:
                    field = _field_from_code_group(code_group, "STRUCTURE")
                checks.append(_make_check(field, note, stage="stage1", rule=name))
            continue

        checks.append(_make_check("STRUCTURE", reason, stage="stage1", rule=name))
    return _dedupe_checks(checks)


def _compat_stage1_payload(stage1_summary: dict[str, Any], base_difficulty: dict[str, Any]) -> dict[str, Any]:
    return {
        "level": stage1_summary.get("stage1_level"),
        "difficulty": stage1_summary.get("stage1_level"),
        "reason_text": _clean(base_difficulty.get("reason_text")),
        "reasons": list(base_difficulty.get("reasons", [])) if isinstance(base_difficulty.get("reasons"), list) else [],
        "failed_checks": copy.deepcopy(stage1_summary.get("failed_checks") or []),
        "passed_checks": [],
    }


def build_stage1_routing(
    text: str,
    *,
    normalized_text: str | None = None,
    type_code: str = "",
    material_code: str = "",
    standard_code: str = "",
    standard_codes: Iterable[str] | None = None,
    enable_code_rules: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = build_base_difficulty(
        text,
        normalized_text=normalized_text,
        type_code=type_code,
        material_code=material_code,
        standard_code=standard_code,
        standard_codes=standard_codes,
        enable_code_rules=enable_code_rules,
    )
    level = normalize_difficulty_level(base.get("difficulty"))
    failed_checks = _stage1_failed_checks(base.get("features"))
    reason_text = _clean(base.get("reason_text"))
    summary = {
        "stage1_level": level,
        "final_level": level,
        "decision_stage": "stage1",
        "need_review": level in (DIFF_HARD, DIFF_EASY),
        "reason_text": reason_text if reason_text else ("一阶段通过" if level == DIFF_EASY else ""),
        "failed_checks": failed_checks,
        "passed_checks": [],
    }
    return summary, _compat_stage1_payload(summary, base)


def build_stage2_routing(result_dict: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "text": _resolve_routing_text(result_dict),
        "stage1_difficulty": ((result_dict.get("difficulty_split") or {}) if isinstance(result_dict.get("difficulty_split"), dict) else {}).get("level"),
        "fields": result_dict.get("fields", {}) if isinstance(result_dict.get("fields"), dict) else {},
    }
    return _second_pass_runner.analyze_payload_summary(payload)


def attach_routing(result_dict: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result_dict, dict):
        return result_dict
    raw_text = _clean(result_dict.get("original_text"))
    routing_text = _resolve_routing_text(result_dict)
    stage1_summary, stage1_payload = build_stage1_routing(
        routing_text or raw_text,
        normalized_text=routing_text or raw_text,
    )
    result_dict["difficulty_split"] = stage1_payload

    if stage1_summary.get("stage1_level") != DIFF_EASY:
        result_dict["second_pass"] = None
        result_dict["routing"] = stage1_summary
        result_dict["need_review"] = bool(result_dict.get("need_review")) or bool(stage1_summary.get("need_review"))
        return result_dict

    stage2_summary = build_stage2_routing(result_dict)
    result_dict["second_pass"] = stage2_summary
    result_dict["routing"] = stage2_summary
    result_dict["need_review"] = bool(result_dict.get("need_review")) or bool(stage2_summary.get("need_review"))
    return result_dict


def _project_frequency_checks(feature: Any) -> list[dict[str, str]]:
    if not isinstance(feature, dict) or not feature.get("matched"):
        return []
    checks: list[dict[str, str]] = []
    hits = feature.get("hits") if isinstance(feature.get("hits"), list) else []
    reason = _clean(feature.get("reason"))
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        checks.append(
            _make_check(
                _field_from_code_group(_clean(hit.get("code_group")), "PROJECT"),
                _clean(hit.get("note")) or reason,
                stage="project_frequency",
                rule="project_frequency",
            )
        )
    if not checks and reason:
        checks.append(_make_check("PROJECT", reason, stage="project_frequency", rule="project_frequency"))
    return _dedupe_checks(checks)


def apply_project_frequency(results: list[dict[str, Any]], project_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for project_name, result_dict in zip(project_names, results):
        rows.append(
            {
                "project": _clean(project_name),
                "type_code": _clean(
                    ((((result_dict.get("fields") or {}).get("TYPE") or {}).get("stage2_output") or {}).get("code"))
                    if isinstance(result_dict, dict)
                    else ""
                ),
                "material_code": _clean(
                    ((((result_dict.get("fields") or {}).get("MATERIAL") or {}).get("stage2_output") or {}).get("code"))
                    if isinstance(result_dict, dict)
                    else ""
                ),
            }
        )

    features = _project_frequency_detector.analyze_rows(rows)
    finalized: list[dict[str, Any]] = []
    for result_dict, feature in zip(results, features):
        updated = copy.deepcopy(result_dict)
        routing = copy.deepcopy(updated.get("routing")) if isinstance(updated.get("routing"), dict) else None
        if not isinstance(routing, dict) or not feature.matched:
            finalized.append(updated)
            continue

        current_level = normalize_difficulty_level(routing.get("final_level"))
        if current_level == DIFF_HARD:
            finalized.append(updated)
            continue

        project_checks = _project_frequency_checks(feature.to_dict())
        reason_text = _clean(routing.get("reason_text"))
        project_reason = " | ".join(check["reason"] for check in project_checks if _clean(check.get("reason")))
        if project_reason:
            reason_text = f"{reason_text} | {project_reason}" if reason_text else project_reason

        routing["final_level"] = DIFF_EASY if current_level == DIFF_SECOND_EASY else current_level
        routing["decision_stage"] = "project_frequency"
        routing["need_review"] = True
        routing["reason_text"] = reason_text
        routing["failed_checks"] = _dedupe_checks([*(routing.get("failed_checks") or []), *project_checks])
        updated["routing"] = routing
        updated["need_review"] = True
        finalized.append(updated)
    return finalized
