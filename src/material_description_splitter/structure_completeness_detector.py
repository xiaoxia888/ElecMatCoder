# -*- coding: utf-8 -*-
"""Detect missing stage-1 structural anchors from description text only."""

from __future__ import annotations

from .models import DifficultyFeature, GlueHit
from .stage1_structure_checker import Stage1StructureChecker


class StructureCompletenessDetector:
    """基于一阶段规则抽取结果，判断尺寸/壁厚/磅级结构是否完整。"""

    def __init__(self, checker: Stage1StructureChecker | None = None) -> None:
        self.checker = checker or Stage1StructureChecker()

    def analyze(self, text: str, *, normalized_text: str | None = None) -> DifficultyFeature:
        result = self.checker.analyze(text, normalized_text=normalized_text)
        if result.is_structurally_complete:
            return DifficultyFeature(name="structure_completeness", matched=False)

        hits: list[GlueHit] = []
        if "SIZE" in result.required_missing_fields:
            hits.append(
                GlueHit(
                    tag="missing_size",
                    code_group="structure",
                    code="SIZE",
                    token="",
                    start=-1,
                    end=-1,
                    note="SIZE: 未命中尺寸字段",
                )
            )
        if "THICKNESS_OR_PRESSURE" in result.required_missing_fields:
            hits.append(
                GlueHit(
                    tag="missing_thickness_or_pressure",
                    code_group="structure",
                    code="THICKNESS_OR_PRESSURE",
                    token="",
                    start=-1,
                    end=-1,
                    note="THICKNESS/PRESSURE: 未命中壁厚或磅级字段",
                )
            )

        return DifficultyFeature(
            name="structure_completeness",
            matched=bool(hits),
            reason="；".join(hit.note for hit in hits),
            hits=hits,
        )
