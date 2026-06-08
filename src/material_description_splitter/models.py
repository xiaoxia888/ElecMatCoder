# -*- coding: utf-8 -*-
"""Shared models for material description difficulty splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GlueHit:
    """A single glued-code hit inside one token."""

    tag: str
    code_group: str
    code: str
    token: str
    start: int
    end: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DifficultyFeature:
    """One difficulty feature result."""

    name: str
    matched: bool
    reason: str = ""
    hits: list[GlueHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matched": self.matched,
            "reason": self.reason,
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass
class DifficultyResult:
    """Final split result."""

    text: str
    is_difficult: bool
    reasons: list[str] = field(default_factory=list)
    features: list[DifficultyFeature] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "is_difficult": self.is_difficult,
            "reasons": self.reasons,
            "features": [feature.to_dict() for feature in self.features],
        }
