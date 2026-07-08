from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ComplexStructuralTriggerResult:
    matched: bool
    source: str = ""
    value: str = ""
    pattern: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": bool(self.matched),
            "source": self.source,
            "value": self.value,
            "pattern": self.pattern,
        }


class ComplexStructuralTrigger:
    """Rule gate for lined / jacketed structural prompt routing."""

    BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "in_out_marker",
            re.compile(r"(?i)\bIN\s*[:：].{0,160}\bOUT\s*[:：]"),
        ),
        (
            "pipe_lined",
            re.compile(r"(?i)\bPIPE\s*,?\s*LINED\b|\bLINED\s+BE\b"),
        ),
        (
            "layered_parenthesized_thickness",
            re.compile(
                r"\d+(?:\.\d+)?\s*[xX×*]\s*\d+(?:\.\d+)?\s*"
                r"\(\s*\d+(?:\.\d+)?\s*\)"
            ),
        ),
        (
            "layered_plus_mm",
            re.compile(r"(?i)\d+(?:\.\d+)?\s*MM\s*\+\s*\d+(?:\.\d+)?\s*MM"),
        ),
        (
            "multi_material_thk",
            re.compile(
                r"(?i)\b[A-Z0-9#./+-]{2,}\s+THK\s*=\s*\d+(?:\.\d+)?\s*MM?"
                r".{0,120}\b[A-Z0-9#./+-]{2,}\s+THK\s*=\s*\d+(?:\.\d+)?\s*MM?"
            ),
        ),
    )

    def __init__(self, keywords: List[str]):
        self.keywords = [str(item or "").strip() for item in keywords if str(item or "").strip()]

    @staticmethod
    def _is_ascii_word_keyword(keyword: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_+-]+", keyword or ""))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ComplexStructuralTrigger":
        raw_keywords = (config or {}).get("trigger_keywords") or []
        if not isinstance(raw_keywords, list):
            raw_keywords = []
        return cls([str(item) for item in raw_keywords])

    def match(self, text: str) -> Optional[ComplexStructuralTriggerResult]:
        raw = str(text or "")
        if not raw.strip():
            return None

        upper = raw.upper()
        for keyword in self.keywords:
            if self._is_ascii_word_keyword(keyword):
                pattern = re.compile(rf"(?<![A-Z0-9_]){re.escape(keyword)}(?![A-Z0-9_])", re.IGNORECASE)
                if pattern.search(raw):
                    return ComplexStructuralTriggerResult(True, "keyword", keyword, pattern.pattern)
                continue
            if keyword.upper() in upper:
                return ComplexStructuralTriggerResult(True, "keyword", keyword, "")

        for name, pattern in self.BUILTIN_PATTERNS:
            if pattern.search(raw):
                return ComplexStructuralTriggerResult(True, f"builtin:{name}", "", pattern.pattern)

        return None
