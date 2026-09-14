# -*- coding: utf-8 -*-
"""High-precision source evidence audit for structural model omissions.

This module intentionally does not repair model output. It extracts only
strongly anchored size, thickness and pressure surfaces, then compares them
with stage-1 output so routing rules can flag likely omissions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Iterable

from src.encoder.processors.structural_rule_config import get_common_dn_values
from src.encoder.processors.structural_numeric_rules import looks_like_od_wall_thickness


NUMBER = r"\d+(?:\.\d+)?"
FRACTION = r"(?:\d+\s+)?\d+/\d+|\d+(?:\.\d+)?"
SEP = r"\s*[xX×*]\s*"


@dataclass(frozen=True)
class SourceEvidence:
    field: str
    item_type: str
    value: str
    raw: str
    rule: str
    start: int
    end: int

    @property
    def key(self) -> tuple[str, str]:
        return self.item_type, self.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceEvidenceAuditResult:
    text: str
    evidence: dict[str, list[SourceEvidence]] = field(default_factory=dict)
    model_items: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    missing: dict[str, list[SourceEvidence]] = field(default_factory=dict)

    @property
    def has_missing(self) -> bool:
        return any(self.missing.values())

    @property
    def reasons(self) -> list[str]:
        labels = {"SIZE": "尺寸", "THICKNESS": "壁厚", "PRESSURE": "磅级", "LENGTH": "长度"}
        result: list[str] = []
        for field_name in ("SIZE", "THICKNESS", "PRESSURE", "LENGTH"):
            items = self.missing.get(field_name) or []
            if not items:
                continue
            values = " | ".join(f"{item.item_type}:{item.value}" for item in items)
            result.append(f"{labels[field_name]}疑似漏提取: {values}")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "has_missing": self.has_missing,
            "reasons": self.reasons,
            "evidence": {
                key: [item.to_dict() for item in values]
                for key, values in self.evidence.items()
            },
            "model_items": {
                key: [{"type": item_type, "value": value} for item_type, value in values]
                for key, values in self.model_items.items()
            },
            "missing": {
                key: [item.to_dict() for item in values]
                for key, values in self.missing.items()
            },
        }


class SourceEvidenceCompletenessChecker:
    """Compare high-precision source evidence against stage-1 output."""

    _DN_EXPLICIT_PAIR = re.compile(
        rf"(?<![A-Z0-9])DN\s*({NUMBER}){SEP}DN\s*({NUMBER})(?!\d)",
        re.IGNORECASE,
    )
    _DN_COMPACT_INTEGER_PAIR = re.compile(
        r"(?<![A-Z0-9])DN\s*(\d+)(?![\d.])\s*[xX×*]\s*(\d+)"
        r"(?!\s*(?:MM|毫米|''|\"|IN(?:CH(?:ES)?)?))(?![A-Z0-9./])",
        re.IGNORECASE,
    )
    _DN_DECIMAL_WALL = re.compile(
        r"(?<![A-Z0-9])DN\s*(\d+)(?![\d.])\s*[xX×*]\s*(\d+\.\d+)\s*(?:MM|毫米)?"
        r"(?!\s*(?:''|\"|IN(?:CH(?:ES)?)?))(?![A-Z0-9.])",
        re.IGNORECASE,
    )
    _BARE_INTEGER_PAIR = re.compile(
        r"(?<![A-Z0-9./])([0-9]+)\s*[xX×*]\s*([0-9]+)"
        r"(?!\s*(?:MM|毫米|''|\"|IN(?:CH(?:ES)?)?))(?![A-Z0-9./])",
        re.IGNORECASE,
    )
    _BARE_DECIMAL_WALL = re.compile(
        r"(?<![A-Z0-9./])([0-9]+(?:\.[0-9]+)?)\s*[xX×*]\s*([0-9]+\.[0-9]+)"
        r"(?!\s*(?:MM|毫米|''|\"|IN(?:CH(?:ES)?)?))(?![A-Z0-9./])",
        re.IGNORECASE,
    )
    _DN_SINGLE = re.compile(rf"(?<![A-Z0-9])DN\s*:?\s*({NUMBER})(?!\d)", re.IGNORECASE)
    _NPS_PAIR = re.compile(
        rf"(?<![A-Z0-9])NPS\s*({FRACTION}){SEP}(?:NPS\s*)?({FRACTION})(?![\d/])",
        re.IGNORECASE,
    )
    _NPS_SINGLE = re.compile(rf"(?<![A-Z0-9])NPS\s*:?\s*({FRACTION})(?![\d/])", re.IGNORECASE)
    _INCH_PAIR = re.compile(
        rf"(?<![\d.])({FRACTION})\s*(?:''|\"|IN(?:CH(?:ES)?)?){SEP}"
        rf"({FRACTION})\s*(?:''|\"|IN(?:CH(?:ES)?)?)(?![A-Z])",
        re.IGNORECASE,
    )
    _INCH_SINGLE = re.compile(
        rf"(?<![\d.])({FRACTION})\s*(?:''|\"|IN(?:CH(?:ES)?)?)(?![A-Z])",
        re.IGNORECASE,
    )
    _OD_PAIR = re.compile(
        rf"(?:[\u03a6φØ∅]|(?<![A-Z])OD\s*)\s*({NUMBER}){SEP}"
        rf"(?:[\u03a6φØ∅]|OD\s*)\s*({NUMBER})",
        re.IGNORECASE,
    )
    _OD_NUMERIC_PAIR = re.compile(
        rf"(?:[\u03a6φØ∅]|(?<![A-Z])OD\s*)\s*({NUMBER}){SEP}({NUMBER})(\s*(?:MM|毫米))?"
        rf"(?!\s*[xX×*]\s*{NUMBER})",
        re.IGNORECASE,
    )
    _OD_WALL = re.compile(
        rf"(?:[\u03a6φØ∅]|(?<![A-Z])OD\s*)\s*({NUMBER}){SEP}({NUMBER})\s*MM\b",
        re.IGNORECASE,
    )
    _OD_SINGLE = re.compile(
        rf"(?:[\u03a6φØ∅]|(?<![A-Z])OD\s*:?\s*)\s*({NUMBER})(?!\d)",
        re.IGNORECASE,
    )
    _LENGTH = re.compile(
        rf"(?<![A-Z])(?:L|LENGTH|长度)\s*[:=]\s*({NUMBER})\s*(MM|CM|M)\b",
        re.IGNORECASE,
    )

    _SCHEDULE_TOKEN = r"(?:SCH\s*\.?\s*\d+S?|S\s*[-.]?\s*\d+S?|STD|XXS|XS)"
    _SCHEDULE_PAIR = re.compile(
        rf"(?<![A-Z0-9])({_SCHEDULE_TOKEN})\s*[xX×*/]\s*({_SCHEDULE_TOKEN})(?![A-Z0-9])",
        re.IGNORECASE,
    )
    _SCHEDULE_SHORTHAND_PAIR = re.compile(
        r"(?<![A-Z0-9])((?:SCH\s*\.?|S\s*[-.]?\s*)\d+S?)\s*[/xX×*]\s*(\d+S?)(?![A-Z0-9])",
        re.IGNORECASE,
    )
    _SCHEDULE_SINGLE = re.compile(rf"(?<![A-Z0-9])({_SCHEDULE_TOKEN})(?![A-Z0-9])", re.IGNORECASE)
    _MM_PAIR = re.compile(
        rf"(?<![A-Z0-9.])({NUMBER})\s*(?:MM)?\s*[xX×/]\s*({NUMBER})\s*MM\b",
        re.IGNORECASE,
    )
    _THK_PAIR = re.compile(
        rf"(?<![A-Z])(?:THK|THICKNESS|T|WT|壁厚|厚度)\s*[:=]\s*"
        rf"({NUMBER})\s*[xX×/]\s*({NUMBER})\s*(?:MM)?",
        re.IGNORECASE,
    )
    _THK_SINGLE = re.compile(
        rf"(?<![A-Z])(?:THK|THICKNESS|T|WT|壁厚|厚度)\s*[:=]\s*({NUMBER})\s*(?:MM)?",
        re.IGNORECASE,
    )
    _MM_BEFORE_SIZE = re.compile(
        rf"(?<![\d.])({NUMBER})\s*MM\s*(?=(?:DN\s*)?\d+\s*[xX×*]\s*(?:DN\s*)?\d+)",
        re.IGNORECASE,
    )

    _PN = re.compile(r"(?<![A-Z0-9])PN\s*\.?\s*(\d+(?:\.\d+)?)(?!\d)", re.IGNORECASE)
    _CLASS = re.compile(r"(?<![A-Z0-9])(?:CL|CLASS)\s*\.?\s*(\d+)(?!\d)", re.IGNORECASE)
    _IMPERIAL_PRESSURE = re.compile(r"(?<![\d.])(\d+)\s*(#|LBS?)(?![A-Z])", re.IGNORECASE)
    _METRIC_PRESSURE = re.compile(rf"(?<![\d.])({NUMBER})\s*(MPA|BAR)(?![A-Z])", re.IGNORECASE)

    _CLASS_VALUES = {"125", "150", "300", "400", "600", "900", "1500", "2500", "3000", "6000", "9000"}
    _SCHEDULE_VALUES = {
        "5", "5S", "10", "10S", "20", "20S", "30", "30S", "40", "40S",
        "60", "60S", "80", "80S", "100", "100S", "120", "120S", "140",
        "140S", "160", "160S",
    }

    _MODEL_ITEM = re.compile(
        rf"\b(DN|OD|INCH|MM|SCHEDULE|LENGTH|PRESSURE)\s*:?\s*"
        rf"(SCH\s*\d+S?|S\s*-?\s*\d+S?|STD|XS|XXS|PN\s*\d+(?:\.\d+)?|"
        rf"(?:CL|CLASS|C)\s*\d+|{FRACTION})",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        infer_common_dn_pairs: bool = True,
        infer_bare_decimal_wall: bool = True,
        common_dn_values: Iterable[int] | None = None,
    ) -> None:
        self.infer_common_dn_pairs = infer_common_dn_pairs
        self.infer_bare_decimal_wall = infer_bare_decimal_wall
        configured = get_common_dn_values() if common_dn_values is None else common_dn_values
        self.common_dn_values = frozenset(int(value) for value in configured)

    def audit(
        self,
        text: str,
        structural_result: Any = None,
        *,
        size_result: Any = None,
        thickness_result: Any = None,
        pressure_result: Any = None,
    ) -> SourceEvidenceAuditResult:
        clean_text = str(text or "")
        evidence = self.extract_evidence(clean_text)
        model_items = self.extract_model_items(
            structural_result,
            size_result=size_result,
            thickness_result=thickness_result,
            pressure_result=pressure_result,
        )
        missing: dict[str, list[SourceEvidence]] = {}
        for field_name in ("SIZE", "THICKNESS", "PRESSURE"):
            remaining = Counter(model_items.get(field_name) or [])
            field_missing: list[SourceEvidence] = []
            for item in evidence.get(field_name) or []:
                if remaining[item.key] > 0:
                    remaining[item.key] -= 1
                else:
                    field_missing.append(item)
            missing[field_name] = field_missing
        missing["LENGTH"] = []
        return SourceEvidenceAuditResult(
            text=clean_text,
            evidence=evidence,
            model_items=model_items,
            missing=missing,
        )

    def extract_evidence(self, text: str) -> dict[str, list[SourceEvidence]]:
        clean_text = self._prepare_source_text(str(text or ""))
        result = {"SIZE": [], "THICKNESS": [], "PRESSURE": [], "LENGTH": []}
        size_blocked: list[tuple[int, int]] = []
        thickness_blocked: list[tuple[int, int]] = []
        pressure_blocked: list[tuple[int, int]] = []

        self._extract_pairs(
            clean_text,
            self._DN_EXPLICIT_PAIR,
            result["SIZE"],
            "DN",
            "explicit_dn_pair",
            size_blocked,
        )
        if self.infer_common_dn_pairs:
            self._extract_compact_dn_pairs(clean_text, result["SIZE"], size_blocked)
        if self.infer_bare_decimal_wall:
            self._extract_dn_decimal_walls(
                clean_text,
                result["SIZE"],
                result["THICKNESS"],
                size_blocked,
                thickness_blocked,
            )
        self._extract_pairs(clean_text, self._NPS_PAIR, result["SIZE"], "INCH", "explicit_nps_pair", size_blocked)
        self._extract_pairs(clean_text, self._INCH_PAIR, result["SIZE"], "INCH", "explicit_inch_pair", size_blocked)
        self._extract_pairs(clean_text, self._OD_PAIR, result["SIZE"], "OD", "explicit_od_pair", size_blocked)

        self._extract_od_numeric_pairs(
            clean_text,
            result["SIZE"],
            result["THICKNESS"],
            size_blocked,
            thickness_blocked,
        )

        for match in self._OD_WALL.finditer(clean_text):
            if self._overlaps(match.span(), size_blocked):
                continue
            result["SIZE"].append(self._item("SIZE", "OD", match.group(1), match, "explicit_od_wall"))
            result["THICKNESS"].append(
                self._item("THICKNESS", "MM", match.group(2), match, "explicit_od_wall")
            )
            size_blocked.append(match.span())
            thickness_blocked.append(match.span())

        self._extract_singles(clean_text, self._DN_SINGLE, result["SIZE"], "DN", "explicit_dn", size_blocked)
        for match in self._NPS_SINGLE.finditer(clean_text):
            if self._overlaps(match.span(), size_blocked) or self._is_explanatory_size_context(clean_text, match):
                continue
            result["SIZE"].append(self._item("SIZE", "INCH", match.group(1), match, "explicit_nps"))
            size_blocked.append(match.span())
        for match in self._INCH_SINGLE.finditer(clean_text):
            if self._overlaps(match.span(), size_blocked) or self._is_explanatory_size_context(clean_text, match):
                continue
            result["SIZE"].append(self._item("SIZE", "INCH", match.group(1), match, "explicit_inch"))
            size_blocked.append(match.span())
        self._extract_singles(clean_text, self._OD_SINGLE, result["SIZE"], "OD", "explicit_od", size_blocked)

        if self.infer_common_dn_pairs:
            self._extract_bare_common_dn_pairs(clean_text, result["SIZE"], size_blocked)
        if self.infer_bare_decimal_wall:
            self._extract_bare_decimal_walls(
                clean_text,
                result["SIZE"],
                result["THICKNESS"],
                size_blocked,
                thickness_blocked,
            )

        for match in self._LENGTH.finditer(clean_text):
            value = self._length_to_mm(match.group(1), match.group(2))
            if value:
                result["LENGTH"].append(
                    SourceEvidence("LENGTH", "LENGTH", value, match.group(0), "explicit_length", *match.span())
                )

        self._extract_schedule_pairs(clean_text, result["THICKNESS"], thickness_blocked)
        self._extract_pairs(
            clean_text,
            self._THK_PAIR,
            result["THICKNESS"],
            "MM",
            "explicit_thickness_pair",
            thickness_blocked,
        )
        self._extract_pairs(
            clean_text,
            self._MM_PAIR,
            result["THICKNESS"],
            "MM",
            "explicit_mm_pair",
            thickness_blocked,
        )
        self._extract_singles(
            clean_text,
            self._THK_SINGLE,
            result["THICKNESS"],
            "MM",
            "explicit_thickness",
            thickness_blocked,
        )
        self._extract_singles(
            clean_text,
            self._MM_BEFORE_SIZE,
            result["THICKNESS"],
            "MM",
            "mm_before_size_pair",
            thickness_blocked,
        )
        for match in self._SCHEDULE_SINGLE.finditer(clean_text):
            if self._overlaps(match.span(), thickness_blocked):
                continue
            normalized = self._normalize_schedule(match.group(1))
            if not self._valid_schedule(normalized) or self._is_non_wall_schedule_context(clean_text, match):
                continue
            result["THICKNESS"].append(
                SourceEvidence(
                    "THICKNESS",
                    "SCHEDULE",
                    normalized,
                    match.group(0),
                    "explicit_schedule",
                    *match.span(),
                )
            )
            thickness_blocked.append(match.span())

        for match in self._PN.finditer(clean_text):
            result["PRESSURE"].append(self._item("PRESSURE", "PN", match.group(1), match, "explicit_pn"))
            pressure_blocked.append(match.span())
        for match in self._CLASS.finditer(clean_text):
            if self._overlaps(match.span(), pressure_blocked):
                continue
            if match.group(1) not in self._CLASS_VALUES:
                continue
            result["PRESSURE"].append(
                SourceEvidence(
                    "PRESSURE", "CLASS", f"CL{int(match.group(1))}", match.group(0), "explicit_class", *match.span()
                )
            )
            pressure_blocked.append(match.span())
        for match in self._IMPERIAL_PRESSURE.finditer(clean_text):
            if self._overlaps(match.span(), pressure_blocked) or self._is_technical_range(clean_text, match.start()):
                continue
            if match.group(1) not in self._CLASS_VALUES:
                continue
            result["PRESSURE"].append(
                SourceEvidence(
                    "PRESSURE", "CLASS", f"CL{int(match.group(1))}", match.group(0), "explicit_imperial_pressure", *match.span()
                )
            )
            pressure_blocked.append(match.span())
        for match in self._METRIC_PRESSURE.finditer(clean_text):
            if self._overlaps(match.span(), pressure_blocked) or self._is_design_pressure(clean_text, match.start()):
                continue
            normalized = self._normalize_metric_pressure(match.group(1), match.group(2))
            if normalized:
                result["PRESSURE"].append(
                    SourceEvidence(
                        "PRESSURE", "PN", normalized, match.group(0), "explicit_metric_pressure", *match.span()
                    )
                )
                pressure_blocked.append(match.span())

        # Single mentions are deduplicated, while explicit pairs retain their
        # multiplicity so DN50x50 and SCH40xSCH40 can expose one missing side.
        for field_name, items in result.items():
            result[field_name] = self._dedupe_non_pair_items(items)
        return result

    def extract_model_items(
        self,
        structural_result: Any = None,
        *,
        size_result: Any = None,
        thickness_result: Any = None,
        pressure_result: Any = None,
    ) -> dict[str, list[tuple[str, str]]]:
        result = {"SIZE": [], "THICKNESS": [], "PRESSURE": [], "LENGTH": []}
        parsed = self._parse_json(structural_result)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("output"), dict):
                parsed = parsed["output"]
            items = parsed.get("ITEMS")
            if isinstance(items, list):
                for position in items:
                    if not isinstance(position, dict):
                        continue
                    self._append_model_dict_items(result["SIZE"], position.get("SIZE"), "SIZE")
                    self._append_model_dict_items(result["THICKNESS"], position.get("THICKNESS"), "THICKNESS")
                self._append_top_level(parsed, result)
            else:
                self._append_model_dict_items(
                    result["SIZE"], parsed.get("SIZE_ITEMS") or self._ordered_items(parsed.get("SIZE")), "SIZE"
                )
                self._append_model_dict_items(
                    result["THICKNESS"],
                    parsed.get("THICKNESS_ITEMS") or self._ordered_items(parsed.get("THICKNESS")),
                    "THICKNESS",
                )
                self._append_top_level(parsed, result)
        elif structural_result not in (None, ""):
            self._append_model_text(result, str(structural_result))

        self._append_model_value(result["SIZE"], size_result, "SIZE")
        self._append_model_value(result["THICKNESS"], thickness_result, "THICKNESS")
        self._append_model_value(result["PRESSURE"], pressure_result, "PRESSURE")
        return result

    def _append_model_value(self, target: list[tuple[str, str]], value: Any, field_name: str) -> None:
        parsed = self._parse_json(value)
        if isinstance(parsed, (dict, list)):
            if isinstance(parsed, dict) and isinstance(parsed.get("ITEMS"), list):
                key = "SIZE" if field_name == "SIZE" else "THICKNESS"
                for position in parsed["ITEMS"]:
                    if isinstance(position, dict):
                        self._append_model_dict_items(target, position.get(key), field_name)
                if field_name == "PRESSURE":
                    self._append_model_scalar(target, "PRESSURE", parsed.get("PRESSURE"))
            else:
                items = parsed
                if isinstance(parsed, dict):
                    item_key = "SIZE_ITEMS" if field_name == "SIZE" else "THICKNESS_ITEMS"
                    items = parsed.get(item_key) or self._ordered_items(parsed)
                self._append_model_dict_items(target, items, field_name)
        elif value not in (None, ""):
            temp = {"SIZE": [], "THICKNESS": [], "PRESSURE": [], "LENGTH": []}
            self._append_model_text(temp, str(value))
            target.extend(temp[field_name])

    def _append_model_text(self, result: dict[str, list[tuple[str, str]]], text: str) -> None:
        clean_text = str(text or "")
        seen: set[tuple[str, str, str, int, int]] = set()

        def append(field_name: str, item_type: str, value: str, match: re.Match[str]) -> None:
            if item_type == "SCHEDULE":
                normalized = self._normalize_schedule(value)
                if not self._valid_schedule(normalized):
                    return
            else:
                normalized = self._normalize_number(value)
            key = (field_name, item_type, normalized, match.start(), match.end())
            if key in seen:
                return
            seen.add(key)
            result[field_name].append((item_type, normalized))

        for match in re.finditer(rf"\b(DN|OD)\s*:?\s*({NUMBER})(?!\d)", clean_text, re.I):
            append("SIZE", match.group(1).upper(), match.group(2), match)
        for match in re.finditer(rf"\bINCH\s*:?\s*({FRACTION})(?![\d/])", clean_text, re.I):
            append("SIZE", "INCH", match.group(1), match)
        for match in re.finditer(rf"(?<![\d.])({FRACTION})\s*(?:''|\")", clean_text, re.I):
            append("SIZE", "INCH", match.group(1), match)
        for match in re.finditer(rf"\bMM\s*:?\s*({NUMBER})(?!\d)", clean_text, re.I):
            append("THICKNESS", "MM", match.group(1), match)
        for match in re.finditer(rf"(?<![\d.])({NUMBER})\s*MM\b", clean_text, re.I):
            append("THICKNESS", "MM", match.group(1), match)
        for match in re.finditer(
            rf"(?:\bSCHEDULE\s*:\s*)?\b(SCH\s*\d+S?|S\s*-\s*\d+S?|STD|XXS|XS)\b",
            clean_text,
            re.I,
        ):
            append("THICKNESS", "SCHEDULE", match.group(1), match)
        # Platform exports pressure as a bare PN/CL value.
        for match in re.finditer(r"(?<![A-Z0-9])(PN\s*\d+(?:\.\d+)?|(?:CL|CLASS|C)\s*\d+)(?!\d)", clean_text, re.I):
            self._append_model_scalar(result["PRESSURE"], "PRESSURE", match.group(1))

    def _append_top_level(self, parsed: dict[str, Any], result: dict[str, list[tuple[str, str]]]) -> None:
        self._append_model_scalar(result["PRESSURE"], "PRESSURE", parsed.get("PRESSURE"))
        length = parsed.get("LENGTH")
        if length not in (None, ""):
            result["LENGTH"].append(("LENGTH", self._normalize_number(str(length).upper().replace("MM", ""))))

    def _append_model_dict_items(self, target: list[tuple[str, str]], items: Any, field_name: str) -> None:
        if isinstance(items, dict):
            expanded: list[dict[str, Any]] = []
            for item_type, values in items.items():
                if str(item_type).startswith("_"):
                    continue
                for value in values if isinstance(values, list) else [values]:
                    expanded.append({"type": item_type, "value": value})
            items = expanded
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or item.get("TYPE") or "").strip().upper()
            value = str(item.get("value") or item.get("VALUE") or "").strip()
            if not item_type or not value:
                continue
            if field_name == "SIZE" and item_type in {"DN", "OD", "INCH"}:
                target.append((item_type, self._normalize_number(value)))
            elif field_name == "THICKNESS" and item_type == "MM":
                target.append(("MM", self._normalize_number(value)))
            elif field_name == "THICKNESS" and item_type == "SCHEDULE":
                target.append(("SCHEDULE", self._normalize_schedule(value)))

    def _append_model_scalar(self, target: list[tuple[str, str]], _: str, value: Any) -> None:
        text = str(value or "").strip().upper()
        if not text:
            return
        match = re.fullmatch(r"PN\s*(\d+(?:\.\d+)?)", text)
        if match:
            target.append(("PN", f"PN{self._normalize_number(match.group(1))}"))
            return
        match = re.fullmatch(r"(?:CL|CLASS|C)\s*(\d+)", text)
        if match:
            target.append(("CLASS", f"CL{int(match.group(1))}"))
            return
        match = re.fullmatch(r"(\d+)\s*(?:#|LBS?)", text)
        if match:
            target.append(("CLASS", f"CL{int(match.group(1))}"))
            return
        match = re.fullmatch(rf"({NUMBER})\s*(MPA|BAR)", text)
        if match:
            normalized = self._normalize_metric_pressure(match.group(1), match.group(2))
            if normalized:
                target.append(("PN", normalized))

    def _extract_schedule_pairs(
        self,
        text: str,
        target: list[SourceEvidence],
        blocked: list[tuple[int, int]],
    ) -> None:
        for pattern, rule in (
            (self._SCHEDULE_PAIR, "explicit_schedule_pair"),
            (self._SCHEDULE_SHORTHAND_PAIR, "explicit_schedule_shorthand_pair"),
        ):
            for match in pattern.finditer(text):
                if self._overlaps(match.span(), blocked):
                    continue
                first = self._normalize_schedule(match.group(1))
                second_raw = match.group(2)
                if rule.endswith("shorthand_pair"):
                    suffix = "S" if str(second_raw).upper().endswith("S") else ""
                    second = f"SCH{re.sub(r'\D', '', second_raw)}{suffix}"
                else:
                    second = self._normalize_schedule(second_raw)
                if not self._valid_schedule(first) or not self._valid_schedule(second):
                    continue
                target.extend(
                    [
                        SourceEvidence("THICKNESS", "SCHEDULE", first, match.group(0), rule, *match.span()),
                        SourceEvidence("THICKNESS", "SCHEDULE", second, match.group(0), rule, *match.span()),
                    ]
                )
                blocked.append(match.span())

    def _extract_compact_dn_pairs(
        self,
        text: str,
        target: list[SourceEvidence],
        blocked: list[tuple[int, int]],
    ) -> None:
        for match in self._DN_COMPACT_INTEGER_PAIR.finditer(text):
            if self._overlaps(match.span(), blocked):
                continue
            second = int(match.group(2))
            if second not in self.common_dn_values:
                continue
            target.extend(
                [
                    self._item("SIZE", "DN", match.group(1), match, "configured_dn_compact_pair"),
                    self._item("SIZE", "DN", match.group(2), match, "configured_dn_compact_pair"),
                ]
            )
            blocked.append(match.span())

    def _extract_od_numeric_pairs(
        self,
        text: str,
        size_target: list[SourceEvidence],
        thickness_target: list[SourceEvidence],
        size_blocked: list[tuple[int, int]],
        thickness_blocked: list[tuple[int, int]],
    ) -> None:
        for match in self._OD_NUMERIC_PAIR.finditer(text):
            if self._overlaps(match.span(), size_blocked):
                continue
            explicit_mm = bool(str(match.group(3) or "").strip())
            configured_size_pair = self._both_configured_integer_dn(match.group(1), match.group(2))
            is_wall = explicit_mm or (
                not configured_size_pair
                and looks_like_od_wall_thickness(match.group(1), match.group(2))
            )
            size_target.append(
                self._item(
                    "SIZE",
                    "OD",
                    match.group(1),
                    match,
                    "explicit_od_wall_ratio" if is_wall else "explicit_od_numeric_pair",
                )
            )
            if is_wall:
                thickness_target.append(
                    self._item("THICKNESS", "MM", match.group(2), match, "explicit_od_wall_ratio")
                )
                thickness_blocked.append(match.span())
            else:
                size_target.append(
                    self._item("SIZE", "OD", match.group(2), match, "explicit_od_numeric_pair")
                )
            size_blocked.append(match.span())

    def _both_configured_integer_dn(self, first: str, second: str) -> bool:
        if not str(first).isdigit() or not str(second).isdigit():
            return False
        return int(first) in self.common_dn_values and int(second) in self.common_dn_values

    def _extract_dn_decimal_walls(
        self,
        text: str,
        size_target: list[SourceEvidence],
        thickness_target: list[SourceEvidence],
        size_blocked: list[tuple[int, int]],
        thickness_blocked: list[tuple[int, int]],
    ) -> None:
        for match in self._DN_DECIMAL_WALL.finditer(text):
            if self._overlaps(match.span(), size_blocked):
                continue
            size_target.append(self._item("SIZE", "DN", match.group(1), match, "explicit_dn_decimal_wall"))
            thickness_target.append(
                self._item("THICKNESS", "MM", match.group(2), match, "explicit_dn_decimal_wall")
            )
            size_blocked.append(match.span())
            thickness_blocked.append(match.span())

    def _extract_bare_common_dn_pairs(
        self,
        text: str,
        target: list[SourceEvidence],
        blocked: list[tuple[int, int]],
    ) -> None:
        for match in self._BARE_INTEGER_PAIR.finditer(text):
            if self._overlaps(match.span(), blocked) or self._is_schedule_pair_context(text, match):
                continue
            first, second = int(match.group(1)), int(match.group(2))
            if first not in self.common_dn_values or second not in self.common_dn_values:
                continue
            target.extend(
                [
                    self._item("SIZE", "DN", match.group(1), match, "configured_common_dn_pair"),
                    self._item("SIZE", "DN", match.group(2), match, "configured_common_dn_pair"),
                ]
            )
            blocked.append(match.span())

    def _extract_bare_decimal_walls(
        self,
        text: str,
        size_target: list[SourceEvidence],
        thickness_target: list[SourceEvidence],
        size_blocked: list[tuple[int, int]],
        thickness_blocked: list[tuple[int, int]],
    ) -> None:
        for match in self._BARE_DECIMAL_WALL.finditer(text):
            if (
                self._overlaps(match.span(), size_blocked)
                or self._overlaps(match.span(), thickness_blocked)
                or self._is_schedule_pair_context(text, match)
            ):
                continue
            first = self._normalize_number(match.group(1))
            common_dn_first = "." not in first and int(first) in self.common_dn_values
            if not common_dn_first and not looks_like_od_wall_thickness(
                match.group(1),
                match.group(2),
                minimum_ratio=15,
            ):
                continue
            if common_dn_first:
                size_target.append(
                    self._item("SIZE", "DN", match.group(1), match, "configured_dn_decimal_wall")
                )
                size_blocked.append(match.span())
            thickness_target.append(
                self._item("THICKNESS", "MM", match.group(2), match, "bare_decimal_wall")
            )
            thickness_blocked.append(match.span())

    @staticmethod
    def _is_schedule_pair_context(text: str, match: re.Match[str]) -> bool:
        prefix = text[max(0, match.start() - 8):match.start()].upper()
        suffix = text[match.end():min(len(text), match.end() + 8)].upper()
        return bool(
            re.search(r"(?:SCH|S\s*[-.]?)\s*$", prefix)
            or re.match(r"\s*S\b", suffix)
        )

    def _extract_pairs(
        self,
        text: str,
        pattern: re.Pattern[str],
        target: list[SourceEvidence],
        item_type: str,
        rule: str,
        blocked: list[tuple[int, int]],
    ) -> None:
        field_name = "THICKNESS" if item_type in {"MM", "SCHEDULE"} else "SIZE"
        for match in pattern.finditer(text):
            if self._overlaps(match.span(), blocked):
                continue
            target.extend(
                [
                    self._item(field_name, item_type, match.group(1), match, rule),
                    self._item(field_name, item_type, match.group(2), match, rule),
                ]
            )
            blocked.append(match.span())

    def _extract_singles(
        self,
        text: str,
        pattern: re.Pattern[str],
        target: list[SourceEvidence],
        item_type: str,
        rule: str,
        blocked: list[tuple[int, int]],
    ) -> None:
        field_name = "THICKNESS" if item_type in {"MM", "SCHEDULE"} else "SIZE"
        for match in pattern.finditer(text):
            if self._overlaps(match.span(), blocked):
                continue
            target.append(self._item(field_name, item_type, match.group(1), match, rule))
            blocked.append(match.span())

    def _item(
        self,
        field_name: str,
        item_type: str,
        value: str,
        match: re.Match[str],
        rule: str,
    ) -> SourceEvidence:
        if item_type == "SCHEDULE":
            normalized = self._normalize_schedule(value)
        elif item_type == "PN":
            normalized = f"PN{self._normalize_number(value)}"
        else:
            normalized = self._normalize_number(value)
        return SourceEvidence(field_name, item_type, normalized, match.group(0), rule, *match.span())

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        text = str(value or "").strip()
        if not text or text[0] not in "[{":
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _ordered_items(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return value.get("_ITEMS") or value.get("ordered_items") or value

    @staticmethod
    def _normalize_number(value: Any) -> str:
        text = str(value or "").strip().replace(" ", "")
        if "/" in text:
            return text
        try:
            number = Decimal(text)
        except InvalidOperation:
            return text.upper()
        normalized = format(number.normalize(), "f")
        return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized

    @staticmethod
    def _normalize_schedule(value: Any) -> str:
        text = re.sub(r"[\s.]", "", str(value or "").upper())
        if text in {"STD", "XS", "XXS"}:
            return text
        if text.startswith("SCH"):
            return text
        text = re.sub(r"^S-?", "SCH", text)
        return text

    @classmethod
    def _valid_schedule(cls, value: str) -> bool:
        if value in {"STD", "XS", "XXS"}:
            return True
        match = re.fullmatch(r"SCH(\d+S?)", value)
        return bool(match and match.group(1) in cls._SCHEDULE_VALUES)

    @staticmethod
    def _is_non_wall_schedule_context(text: str, match: re.Match[str]) -> bool:
        prefix = text[max(0, match.start() - 20):match.start()].upper()
        suffix = text[match.end():min(len(text), match.end() + 20)].upper()
        # Manufacturer standard and end-preparation instructions are not the
        # product's nominal wall-thickness evidence.
        if re.search(r"(?:MFR(?:'S)?|MANUFACTURER)\s*$", prefix):
            return True
        if re.search(r"(?:BEVELL?ED\s+TO|MACHINED\s+TO)\s*$", prefix):
            return True
        if re.match(r"\s*(?:ON|AT)\s+(?:BOTH|DOUBLE)\s+ENDS?", suffix):
            return True
        return False

    @classmethod
    def _normalize_metric_pressure(cls, number: str, unit: str) -> str:
        try:
            value = Decimal(str(number)) * (Decimal("10") if str(unit).upper() == "MPA" else Decimal("1"))
        except InvalidOperation:
            return ""
        normalized = cls._normalize_number(value)
        return f"PN{normalized}"

    @classmethod
    def _length_to_mm(cls, number: str, unit: str) -> str:
        try:
            value = Decimal(str(number)) * {"MM": 1, "CM": 10, "M": 1000}[str(unit).upper()]
        except (InvalidOperation, KeyError):
            return ""
        return cls._normalize_number(value)

    @staticmethod
    def _overlaps(span: tuple[int, int], blocked: Iterable[tuple[int, int]]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in blocked)

    @staticmethod
    def _is_design_pressure(text: str, start: int) -> bool:
        prefix = text[max(0, start - 20):start].upper()
        return bool(
            re.search(r"(?:设计|工作|试验|操作|最高允许)压力\s*[:：=]?\s*$", prefix)
            or re.search(r"(?:PRESSURE|DESIGN\s*PRESSURE)\s*[:=]\s*$", prefix)
            or re.search(r"[<>]=?\s*$|[≤≥]\s*$", prefix)
        )

    @staticmethod
    def _is_technical_range(text: str, start: int) -> bool:
        prefix = text[max(0, start - 12):start].upper()
        return bool(re.search(r"(?:AARH|RA|ROUGHNESS)\s*[:=]?\s*$", prefix))

    @staticmethod
    def _prepare_source_text(text: str) -> str:
        # Numbered clauses are frequently glued to a decimal wall thickness:
        # ``Ø5.03.连接方式`` means ``Ø5.0 3.连接方式``, not 5.03 mm.
        return re.sub(
            r"(\d+\.\d)(\d)(?=\.\s*(?:连接|焊接|执行|安装|除锈|材质|规格))",
            r"\1 \2",
            text,
        )

    @staticmethod
    def _is_explanatory_size_context(text: str, match: re.Match[str]) -> bool:
        left = text.rfind("(", 0, match.start())
        right = text.find(")", match.end())
        if left < 0 or right < 0:
            return False
        content = text[left + 1:right].upper()
        return bool(
            re.search(
                r"STRAIGHT\s+TANGENT|FOR\s+NPS|AND\s+BELOW|BOTH\s+ENDS?|"
                r"切线长|两端|双端|以下",
                content,
            )
        )

    @staticmethod
    def _dedupe_non_pair_items(items: list[SourceEvidence]) -> list[SourceEvidence]:
        """Keep the strongest required multiplicity, not repeated mentions.

        An explicit ``DN50x50`` requires two DN50 values, but repeating that
        same specification elsewhere in the description still requires two,
        not four. A standalone repetition also cannot increase a pair count.
        """
        ordered = sorted(items, key=lambda current: (current.start, current.end, current.item_type))
        groups: dict[tuple[int, int, str], list[SourceEvidence]] = {}
        singles: list[SourceEvidence] = []
        for item in ordered:
            if "pair" in item.rule:
                groups.setdefault((item.start, item.end, item.rule), []).append(item)
            else:
                singles.append(item)

        required: Counter[tuple[str, str]] = Counter()
        representatives: dict[tuple[str, str], list[SourceEvidence]] = {}
        for group in groups.values():
            counts = Counter(item.key for item in group)
            for key, count in counts.items():
                if count > required[key]:
                    required[key] = count
                    representatives[key] = [item for item in group if item.key == key][:count]
        for item in singles:
            if required[item.key] < 1:
                required[item.key] = 1
                representatives[item.key] = [item]

        selected = [item for values in representatives.values() for item in values]
        return sorted(selected, key=lambda current: (current.start, current.end, current.item_type))
