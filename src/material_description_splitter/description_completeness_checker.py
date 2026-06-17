# -*- coding: utf-8 -*-
"""Unified entry for description completeness checks."""

from __future__ import annotations

from dataclasses import dataclass

from src.tokenizer_utils.preprocessor import TextPreprocessor

from .models import DifficultyFeature
from .standard_completeness_detector import StandardCompletenessDetector
from .structure_completeness_detector import StructureCompletenessDetector


@dataclass
class DescriptionCompletenessResult:
    text: str
    normalized_text: str
    is_complete: bool
    reasons: list[str]
    features: list[DifficultyFeature]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "is_complete": self.is_complete,
            "reasons": list(self.reasons),
            "features": [feature.to_dict() for feature in self.features],
        }


class DescriptionCompletenessChecker:
    """描述完整性校验统一入口。"""

    def __init__(
        self,
        *,
        preprocessor: TextPreprocessor | None = None,
        standard_detector: StandardCompletenessDetector | None = None,
        structure_detector: StructureCompletenessDetector | None = None,
    ) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.standard_detector = standard_detector or StandardCompletenessDetector()
        self.structure_detector = structure_detector or StructureCompletenessDetector()

    def analyze(self, text: str, *, normalized_text: str | None = None) -> DescriptionCompletenessResult:
        raw_text = str(text or "")
        normalized = str(normalized_text or "").strip() or self.preprocessor.process(raw_text)

        features = [
            self.standard_detector.analyze(normalized),
            self.structure_detector.analyze(raw_text, normalized_text=normalized),
        ]
        reasons = [feature.reason for feature in features if feature.matched and feature.reason]
        return DescriptionCompletenessResult(
            text=raw_text,
            normalized_text=normalized,
            is_complete=not reasons,
            reasons=reasons,
            features=features,
        )
