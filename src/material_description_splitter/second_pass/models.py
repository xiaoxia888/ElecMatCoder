# -*- coding: utf-8 -*-
"""Result models for second-pass material auto-pass splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SizeSecondPassItem:
    field: str
    raw: str
    value: str = ""
    values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SizeSurfaceHit:
    field: str
    alias: str
    start: int
    end: int
    text: str
    kind: str = "anchored"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SizeSecondPassResult:
    text: str
    size_result: str
    size_code: str
    passed: bool
    reason: str = ""
    items: list[SizeSecondPassItem] = field(default_factory=list)
    anchored_hits: list[SizeSurfaceHit] = field(default_factory=list)
    fallback_hits: list[SizeSurfaceHit] = field(default_factory=list)
    unmatched_items: list[SizeSecondPassItem] = field(default_factory=list)
    consumed_spans: list[tuple[int, int]] = field(default_factory=list)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "size_result": self.size_result,
            "size_code": self.size_code,
            "passed": self.passed,
            "reason": self.reason,
            "items": [item.to_dict() for item in self.items],
            "anchored_hits": [hit.to_dict() for hit in self.anchored_hits],
            "fallback_hits": [hit.to_dict() for hit in self.fallback_hits],
            "unmatched_items": [item.to_dict() for item in self.unmatched_items],
            "consumed_spans": list(self.consumed_spans),
            "fallback_used": self.fallback_used,
        }


@dataclass
class ThicknessSecondPassItem:
    field: str
    raw: str
    value: str = ""
    values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThicknessSurfaceHit:
    field: str
    alias: str
    start: int
    end: int
    text: str
    kind: str = "anchored"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThicknessSecondPassResult:
    text: str
    thickness_result: str
    thickness_code: str
    passed: bool
    reason: str = ""
    items: list[ThicknessSecondPassItem] = field(default_factory=list)
    anchored_hits: list[ThicknessSurfaceHit] = field(default_factory=list)
    fallback_hits: list[ThicknessSurfaceHit] = field(default_factory=list)
    unmatched_items: list[ThicknessSecondPassItem] = field(default_factory=list)
    consumed_spans: list[tuple[int, int]] = field(default_factory=list)
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "thickness_result": self.thickness_result,
            "thickness_code": self.thickness_code,
            "passed": self.passed,
            "reason": self.reason,
            "items": [item.to_dict() for item in self.items],
            "anchored_hits": [hit.to_dict() for hit in self.anchored_hits],
            "fallback_hits": [hit.to_dict() for hit in self.fallback_hits],
            "unmatched_items": [item.to_dict() for item in self.unmatched_items],
            "consumed_spans": list(self.consumed_spans),
            "fallback_used": self.fallback_used,
        }


@dataclass
class PressureSecondPassItem:
    field: str
    raw: str
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PressureSurfaceHit:
    field: str
    alias: str
    start: int
    end: int
    text: str
    kind: str = "anchored"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PressureSecondPassResult:
    text: str
    pressure_result: str
    pressure_code: str
    passed: bool
    reason: str = ""
    items: list[PressureSecondPassItem] = field(default_factory=list)
    anchored_hits: list[PressureSurfaceHit] = field(default_factory=list)
    unmatched_items: list[PressureSecondPassItem] = field(default_factory=list)
    consumed_spans: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pressure_result": self.pressure_result,
            "pressure_code": self.pressure_code,
            "passed": self.passed,
            "reason": self.reason,
            "items": [item.to_dict() for item in self.items],
            "anchored_hits": [hit.to_dict() for hit in self.anchored_hits],
            "unmatched_items": [item.to_dict() for item in self.unmatched_items],
            "consumed_spans": list(self.consumed_spans),
        }


@dataclass
class MaterialSurfaceHit:
    code: str
    alias: str
    start: int
    end: int
    text: str
    kind: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialSecondPassResult:
    text: str
    material_code: str
    passed: bool
    reason: str = ""
    base_code: str = ""
    suffix_code: str = ""
    base_hits: list[MaterialSurfaceHit] = field(default_factory=list)
    suffix_hits: list[MaterialSurfaceHit] = field(default_factory=list)
    conflict_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "material_code": self.material_code,
            "passed": self.passed,
            "reason": self.reason,
            "base_code": self.base_code,
            "suffix_code": self.suffix_code,
            "base_hits": [hit.to_dict() for hit in self.base_hits],
            "suffix_hits": [hit.to_dict() for hit in self.suffix_hits],
            "conflict_codes": list(self.conflict_codes),
        }


@dataclass
class TypeSurfaceHit:
    code: str
    field: str
    alias: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TypeSecondPassResult:
    text: str
    type_code: str
    passed: bool
    reason: str = ""
    matched_path: str = ""
    direct_hits: list[TypeSurfaceHit] = field(default_factory=list)
    body_hits: list[TypeSurfaceHit] = field(default_factory=list)
    manu_hits: list[TypeSurfaceHit] = field(default_factory=list)
    conn_hits: list[TypeSurfaceHit] = field(default_factory=list)
    seal_hits: list[TypeSurfaceHit] = field(default_factory=list)
    angle_hits: list[TypeSurfaceHit] = field(default_factory=list)
    radius_hits: list[TypeSurfaceHit] = field(default_factory=list)
    blocking_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type_code": self.type_code,
            "passed": self.passed,
            "reason": self.reason,
            "matched_path": self.matched_path,
            "direct_hits": [hit.to_dict() for hit in self.direct_hits],
            "body_hits": [hit.to_dict() for hit in self.body_hits],
            "manu_hits": [hit.to_dict() for hit in self.manu_hits],
            "conn_hits": [hit.to_dict() for hit in self.conn_hits],
            "seal_hits": [hit.to_dict() for hit in self.seal_hits],
            "angle_hits": [hit.to_dict() for hit in self.angle_hits],
            "radius_hits": [hit.to_dict() for hit in self.radius_hits],
            "blocking_code": self.blocking_code,
        }


@dataclass
class StandardSurfaceHit:
    code: str
    field: str
    alias: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StandardCodeCheck:
    raw_code: str
    category: str = ""
    family: str = ""
    core: str = ""
    suffix: str = ""
    passed: bool = False
    reason: str = ""
    prefix_status: str = ""
    base_hits: list[StandardSurfaceHit] = field(default_factory=list)
    prefix_hits: list[StandardSurfaceHit] = field(default_factory=list)
    suffix_hits: list[StandardSurfaceHit] = field(default_factory=list)
    suspicious_suffix_hits: list[StandardSurfaceHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_code": self.raw_code,
            "category": self.category,
            "family": self.family,
            "core": self.core,
            "suffix": self.suffix,
            "passed": self.passed,
            "reason": self.reason,
            "prefix_status": self.prefix_status,
            "base_hits": [hit.to_dict() for hit in self.base_hits],
            "prefix_hits": [hit.to_dict() for hit in self.prefix_hits],
            "suffix_hits": [hit.to_dict() for hit in self.suffix_hits],
            "suspicious_suffix_hits": [hit.to_dict() for hit in self.suspicious_suffix_hits],
        }


@dataclass
class StandardSecondPassResult:
    text: str
    standard_code: str
    passed: bool
    reason: str = ""
    checks: list[StandardCodeCheck] = field(default_factory=list)
    unmatched_standard_candidates: list[str] = field(default_factory=list)
    has_unmatched_standard_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "standard_code": self.standard_code,
            "passed": self.passed,
            "reason": self.reason,
            "checks": [item.to_dict() for item in self.checks],
            "unmatched_standard_candidates": list(self.unmatched_standard_candidates),
            "has_unmatched_standard_risk": self.has_unmatched_standard_risk,
        }
