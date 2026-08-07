# -*- coding: utf-8 -*-
"""Detect descriptions containing substantial parenthesized content."""

from __future__ import annotations

import re

from .models import DifficultyFeature, GlueHit


class ParenthesizedContentDetector:
    """Mark non-trivial parenthesized content as a stage-one difficulty."""

    FEATURE_NAME = "parenthesized_content"
    MIN_CONTENT_LENGTH = 5
    PATTERN = re.compile(r"\(([^()]*)\)|（([^（）]*)）")

    def analyze(self, text: str) -> DifficultyFeature:
        raw_text = str(text or "")
        hits: list[GlueHit] = []

        for match in self.PATTERN.finditer(raw_text):
            content = (match.group(1) if match.group(1) is not None else match.group(2) or "").strip()
            content_length = len(re.sub(r"\s+", "", content))
            if content_length < self.MIN_CONTENT_LENGTH:
                continue
            hits.append(
                GlueHit(
                    tag=self.FEATURE_NAME,
                    code_group="parenthesized_content",
                    code=content,
                    token=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    note=f"括号内包含较长内容: {match.group(0)}",
                )
            )

        return DifficultyFeature(
            name=self.FEATURE_NAME,
            matched=bool(hits),
            reason="括号内包含至少5个字符，需要结合上下文判断" if hits else "",
            hits=hits,
        )
