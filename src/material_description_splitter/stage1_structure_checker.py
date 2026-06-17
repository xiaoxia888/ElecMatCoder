# -*- coding: utf-8 -*-
"""Stage-1 structure completeness checker driven by deterministic rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from src.encoder.processors import (
    PressureProcessor,
    SizeProcessor,
    ThicknessProcessor,
    extract_size_and_thickness_by_rules,
)
from src.tokenizer_utils.preprocessor import TextPreprocessor


@dataclass
class Stage1StructureCheckResult:
    text: str
    normalized_text: str
    presence: dict[str, bool]
    missing_fields: list[str]
    required_missing_fields: list[str]
    is_structurally_complete: bool
    structured_fields: dict[str, Any]
    raw_rule_results: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "presence": dict(self.presence),
            "missing_fields": list(self.missing_fields),
            "required_missing_fields": list(self.required_missing_fields),
            "is_structurally_complete": self.is_structurally_complete,
            "structured_fields": self.structured_fields,
            "raw_rule_results": self.raw_rule_results,
        }


class Stage1StructureChecker:
    """
    一阶段结构完整性检查器。

    当前只判断三类字段是否存在：
    - SIZE
    - THICKNESS
    - PRESSURE

    当前“最小完整”口径：
    - 必须有 SIZE
    - THICKNESS / PRESSURE 至少命中一个
    """

    SUSPECTED_THICKNESS_RE = re.compile(
        r"(?<![A-Za-z0-9])(?:THK\s*[:=]?\s*)?\d+(?:\.\d+)?\s*MM(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    LENGTH_LEFT_CONTEXT_RE = re.compile(
        r"(?i)(?:长度|长[:：=]?\s*|LEN(?:GTH)?[:：=]?\s*|CUT\s*TO\s*|LONG\b|L[:：=]\s*)$"
    )
    LENGTH_RANGE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:~|-|至|TO)\s*\d+(?:\.\d+)?\s*MM", re.IGNORECASE)

    def __init__(
        self,
        *,
        preprocessor: TextPreprocessor | None = None,
        size_processor: SizeProcessor | None = None,
        thickness_processor: ThicknessProcessor | None = None,
        pressure_processor: PressureProcessor | None = None,
    ) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.size_processor = size_processor or SizeProcessor()
        self.thickness_processor = thickness_processor or ThicknessProcessor(enable_rule_layered=False)
        self.pressure_processor = pressure_processor or PressureProcessor()

    def analyze(self, text: str, *, normalized_text: str | None = None) -> Stage1StructureCheckResult:
        raw_text = str(text or "")
        normalized = str(normalized_text or "").strip() or self.preprocessor.process(raw_text)
        rule_result = extract_size_and_thickness_by_rules(
            normalized,
            size_processor=self.size_processor,
            thickness_processor=self.thickness_processor,
            pressure_processor=self.pressure_processor,
            apply_residual_guard=False,
        )

        strict_thickness_present = bool(rule_result.thickness.mm or rule_result.thickness.schedule)
        suspected_thickness_present = False
        if not strict_thickness_present:
            suspected_thickness_present = self._has_suspected_thickness_presence(normalized)

        presence = {
            "SIZE": bool(rule_result.size.dn or rule_result.size.od or rule_result.size.inch or rule_result.size.length),
            "THICKNESS": strict_thickness_present or suspected_thickness_present,
            "PRESSURE": bool(rule_result.pressure.pressure_code),
        }

        missing_fields = [field for field, present in presence.items() if not present]
        required_missing_fields: list[str] = []
        if not presence["SIZE"]:
            required_missing_fields.append("SIZE")
        if not (presence["THICKNESS"] or presence["PRESSURE"]):
            required_missing_fields.append("THICKNESS_OR_PRESSURE")

        return Stage1StructureCheckResult(
            text=raw_text,
            normalized_text=normalized,
            presence=presence,
            missing_fields=missing_fields,
            required_missing_fields=required_missing_fields,
            is_structurally_complete=not required_missing_fields,
            structured_fields={
                "SIZE": {
                    "DN": list(rule_result.size.dn),
                    "OD": list(rule_result.size.od),
                    "INCH": list(rule_result.size.inch),
                    "LENGTH": list(rule_result.size.length),
                },
                "THICKNESS": {
                    "MM": [str(value).replace("MM", "") for value in rule_result.thickness.mm],
                    "SCHEDULE": list(rule_result.thickness.schedule),
                    "PRESENCE_SOURCE": (
                        "strict_rule"
                        if strict_thickness_present
                        else ("suspected_presence" if suspected_thickness_present else "missing")
                    ),
                },
                "PRESSURE": {
                    "VALUES": list(rule_result.pressure.values),
                    "CODE": rule_result.pressure.pressure_code,
                },
            },
            raw_rule_results={
                "SIZE": asdict(rule_result.size),
                "THICKNESS": asdict(rule_result.thickness),
                "PRESSURE": asdict(rule_result.pressure),
                "THICKNESS_PRESENCE": {
                    "strict_rule": strict_thickness_present,
                    "suspected_presence": suspected_thickness_present,
                },
            },
        )

    def _has_suspected_thickness_presence(self, text: str) -> bool:
        if not text:
            return False
        if self.LENGTH_RANGE_RE.search(text):
            return False

        for match in self.SUSPECTED_THICKNESS_RE.finditer(text):
            if self._looks_like_length_context(text, match.start(), match.end()):
                continue
            return True
        return False

    def _looks_like_length_context(self, text: str, start: int, end: int) -> bool:
        left_window = text[max(0, start - 24):start]
        right_window = text[end:min(len(text), end + 16)]
        compact = text[max(0, start - 24):min(len(text), end + 16)]

        if self.LENGTH_LEFT_CONTEXT_RE.search(left_window):
            return True
        if re.search(r"(?i)MM\s*(?:长|长度|LONG\b|LEN(?:GTH)?\b)", compact):
            return True
        if re.search(r"(?i)(?:~|至|TO|-)\s*\d+(?:\.\d+)?\s*MM", right_window):
            return True
        return False
