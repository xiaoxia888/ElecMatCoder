# -*- coding: utf-8 -*-
"""Entry point for difficulty splitting."""

from __future__ import annotations

from typing import Iterable

from .anchor_missing_detector import AnchorMissingDetector
from .models import DifficultyResult
from .special_token_detector import SpecialTokenDetector
from .standard_glue_detector import StandardGlueDetector
from .type_glue_detector import TypeGlueDetector
from .uncommon_code_detector import UncommonCodeDetector


class MaterialDifficultySplitter:
    """Split material descriptions by difficulty features."""

    def __init__(self) -> None:
        self.type_glue_detector = TypeGlueDetector()
        self.standard_glue_detector = StandardGlueDetector()
        self.anchor_missing_detector = AnchorMissingDetector()
        self.special_token_detector = SpecialTokenDetector()
        self.uncommon_code_detector = UncommonCodeDetector()

    def analyze(
        self,
        text: str,
        type_code: str = "",
        material_code: str = "",
        standard_code: str | Iterable[str] = "",
    ) -> DifficultyResult:
        features = [
            self.type_glue_detector.analyze(text),
            self.standard_glue_detector.analyze(text),
            self.anchor_missing_detector.analyze(text),
            self.special_token_detector.analyze(text),
            self.uncommon_code_detector.analyze(
                text,
                type_code=type_code,
                material_code=material_code,
                standard_code=standard_code,
            ),
        ]
        reasons = [feature.reason for feature in features if feature.matched and feature.reason]
        return DifficultyResult(
            text=text,
            is_difficult=bool(reasons),
            reasons=reasons,
            features=features,
        )
