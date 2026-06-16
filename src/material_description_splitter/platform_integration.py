# -*- coding: utf-8 -*-
"""Platform-facing helpers for difficulty splitting."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .difficulty_splitter import MaterialDifficultySplitter
from .difficulty_levels import DIFF_EASY, DIFF_HARD
from .models import DifficultyFeature, DifficultyResult
from .project_frequency_detector import ProjectFrequencyDetector

DEFAULT_PROJECT_NAME = "__DEFAULT_PROJECT__"

_splitter = MaterialDifficultySplitter()
_project_frequency_detector = ProjectFrequencyDetector()


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def _feature_reason_text(feature: DifficultyFeature) -> str:
    if not feature.matched:
        return ""
    hit_notes = [str(hit.note).strip() for hit in feature.hits if str(hit.note).strip()]
    if hit_notes:
        unique_notes: list[str] = []
        seen: set[str] = set()
        for note in hit_notes:
            if note in seen:
                continue
            seen.add(note)
            unique_notes.append(note)
        return "；".join(unique_notes)
    return str(feature.reason or "").strip()


def format_reason_text(result: DifficultyResult) -> str:
    parts: list[str] = []
    for feature in result.features:
        part = _feature_reason_text(feature)
        if part:
            parts.append(part)
    return " | ".join(parts)


def build_base_difficulty(
    text: str,
    *,
    normalized_text: str | None = None,
    type_code: str = "",
    material_code: str = "",
    standard_code: str = "",
    standard_codes: Iterable[str] | None = None,
    enable_code_rules: bool = False,
) -> Dict[str, Any]:
    clean_text = _clean_text(text)
    if not clean_text:
        return {
            "difficulty": "",
            "reason_text": "",
            "reasons": [],
            "is_difficult": False,
            "features": [],
            "project_frequency": {"matched": False, "reason": "", "hits": []},
        }

    result = _splitter.analyze(
        clean_text,
        type_code=_clean_text(type_code),
        material_code=_clean_text(material_code),
        standard_code=(
            [_clean_text(code) for code in (standard_codes or []) if _clean_text(code)]
            if standard_codes is not None
            else _clean_text(standard_code)
        ),
        normalized_text=_clean_text(normalized_text) or clean_text,
        enable_code_rules=enable_code_rules,
    )
    return {
        "difficulty": DIFF_HARD if result.is_difficult else DIFF_EASY,
        "reason_text": format_reason_text(result),
        "reasons": list(result.reasons),
        "is_difficult": bool(result.is_difficult),
        "features": [feature.to_dict() for feature in result.features],
        "project_frequency": {"matched": False, "reason": "", "hits": []},
    }


def finalize_batch_difficulty(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    row_list = list(items)
    if not row_list:
        return []

    clean_projects = [_clean_text(row.get("project_name", "")) for row in row_list]
    has_any_project = any(clean_projects)

    project_rows: list[dict[str, str]] = []
    effective_projects: list[str] = []
    for idx, row in enumerate(row_list):
        text = _clean_text(row.get("text", ""))
        project_name = clean_projects[idx]
        if text:
            if not has_any_project:
                project_name = DEFAULT_PROJECT_NAME
            elif not project_name:
                project_name = DEFAULT_PROJECT_NAME
        else:
            project_name = ""
        effective_projects.append(project_name)
        project_rows.append(
            {
                "project": project_name,
                "type_code": _clean_text(row.get("type_code", "")),
                "material_code": _clean_text(row.get("material_code", "")),
            }
        )

    project_features = _project_frequency_detector.analyze_rows(project_rows)

    final_results: list[Dict[str, Any]] = []
    for idx, row in enumerate(row_list):
        text = _clean_text(row.get("text", ""))
        if not text:
            final_results.append(
                {
                    "difficulty": "",
                    "reason_text": "",
                    "reasons": [],
                    "is_difficult": False,
                    "features": [],
                    "project_frequency": {"matched": False, "reason": "", "hits": []},
                    "project_name": "",
                }
            )
            continue

        base = row.get("base_difficulty")
        if not isinstance(base, dict):
            base = build_base_difficulty(
                text,
                type_code=_clean_text(row.get("type_code", "")),
                material_code=_clean_text(row.get("material_code", "")),
                standard_code=_clean_text(row.get("standard_code", "")),
                standard_codes=row.get("standard_codes") if isinstance(row.get("standard_codes"), (list, tuple)) else None,
                enable_code_rules=False,
            )

        project_feature = project_features[idx] if idx < len(project_features) else DifficultyFeature(
            name="project_frequency",
            matched=False,
        )
        reasons: list[str] = []
        base_reason_text = _clean_text(base.get("reason_text", ""))
        if base_reason_text:
            reasons.append(base_reason_text)
        project_reason_text = _feature_reason_text(project_feature)
        if project_reason_text:
            reasons.append(project_reason_text)

        final_results.append(
            {
                "difficulty": DIFF_HARD if (bool(base.get("is_difficult")) or project_feature.matched) else DIFF_EASY,
                "reason_text": " | ".join(reasons),
                "reasons": list(base.get("reasons", [])),
                "is_difficult": bool(base.get("is_difficult")) or project_feature.matched,
                "features": list(base.get("features", [])),
                "project_frequency": project_feature.to_dict(),
                "project_name": effective_projects[idx],
            }
        )

    return final_results
