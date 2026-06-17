# -*- coding: utf-8 -*-
"""Unified second-pass runner for platform payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..difficulty_levels import DIFF_EASY, DIFF_HARD, DIFF_SECOND_EASY, normalize_difficulty_level
from .material_second_pass_splitter import MaterialSecondPassSplitter
from .pressure_second_pass_splitter import PressureSecondPassSplitter
from .size_second_pass_splitter import SizeSecondPassSplitter
from .standard_second_pass_splitter import StandardSecondPassSplitter
from .thickness_second_pass_splitter import ThicknessSecondPassSplitter
from .type_second_pass_splitter import TypeSecondPassSplitter


@dataclass
class PlatformSecondPassRunner:
    size_splitter: SizeSecondPassSplitter = field(default_factory=SizeSecondPassSplitter)
    thickness_splitter: ThicknessSecondPassSplitter = field(default_factory=ThicknessSecondPassSplitter)
    pressure_splitter: PressureSecondPassSplitter = field(default_factory=PressureSecondPassSplitter)
    material_splitter: MaterialSecondPassSplitter = field(default_factory=MaterialSecondPassSplitter)
    type_splitter: TypeSecondPassSplitter = field(default_factory=TypeSecondPassSplitter)
    standard_splitter: StandardSecondPassSplitter = field(default_factory=StandardSecondPassSplitter)

    def analyze_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self._clean(payload.get("text") or payload.get("original_text"))
        stage1_difficulty = normalize_difficulty_level(payload.get("stage1_difficulty") if payload.get("stage1_difficulty") is not None else payload.get("difficulty"))
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}

        extracted = {
            "SIZE": self._extract_stage1_value(fields.get("SIZE")),
            "THICKNESS": self._extract_stage1_value(fields.get("THICKNESS")),
            "PRESSURE": self._extract_stage1_value(fields.get("PRESSURE")),
            "MATERIAL_CODE": self._extract_code(fields.get("MATERIAL")),
            "TYPE_CODE": self._extract_code(fields.get("TYPE")),
            "STANDARD_ITEMS": self._extract_standard_items(fields.get("STANDARD")),
        }
        return self.analyze(
            text=text,
            stage1_difficulty=stage1_difficulty,
            size_value=extracted["SIZE"],
            thickness_value=extracted["THICKNESS"],
            pressure_value=extracted["PRESSURE"],
            material_code=extracted["MATERIAL_CODE"],
            type_code=extracted["TYPE_CODE"],
            standard_items=extracted["STANDARD_ITEMS"],
        )

    def analyze_payload_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self._clean(payload.get("text") or payload.get("original_text"))
        stage1_difficulty = normalize_difficulty_level(
            payload.get("stage1_difficulty") if payload.get("stage1_difficulty") is not None else payload.get("difficulty")
        )
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        extracted = {
            "SIZE": self._extract_stage1_value(fields.get("SIZE")),
            "THICKNESS": self._extract_stage1_value(fields.get("THICKNESS")),
            "PRESSURE": self._extract_stage1_value(fields.get("PRESSURE")),
            "MATERIAL_CODE": self._extract_code(fields.get("MATERIAL")),
            "TYPE_CODE": self._extract_code(fields.get("TYPE")),
            "STANDARD_ITEMS": self._extract_standard_items(fields.get("STANDARD")),
        }
        detailed = self.analyze(
            text=text,
            stage1_difficulty=stage1_difficulty,
            size_value=extracted["SIZE"],
            thickness_value=extracted["THICKNESS"],
            pressure_value=extracted["PRESSURE"],
            material_code=extracted["MATERIAL_CODE"],
            type_code=extracted["TYPE_CODE"],
            standard_items=extracted["STANDARD_ITEMS"],
        )
        return self._summarize_result(
            detailed,
            size_value=extracted["SIZE"],
            thickness_value=extracted["THICKNESS"],
            pressure_value=extracted["PRESSURE"],
            material_code=extracted["MATERIAL_CODE"],
            type_code=extracted["TYPE_CODE"],
            standard_items=extracted["STANDARD_ITEMS"],
        )

    def analyze(
        self,
        *,
        text: str,
        stage1_difficulty: Any = None,
        size_value: Any = None,
        thickness_value: Any = None,
        pressure_value: Any = None,
        material_code: str = "",
        type_code: str = "",
        standard_items: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        clean_text = self._clean(text)
        clean_difficulty = normalize_difficulty_level(stage1_difficulty)
        results: dict[str, Any] = {}
        skipped_fields: dict[str, str] = {}

        if clean_difficulty is not None and clean_difficulty != DIFF_EASY:
            for field_name in ("SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "TYPE", "STANDARD"):
                if self._field_provided(field_name, size_value, thickness_value, pressure_value, material_code, type_code, standard_items):
                    skipped_fields[field_name] = f"一阶段非简单: {clean_difficulty}"
            return {
                "text": clean_text,
                "stage1_difficulty": clean_difficulty,
                "final_level": DIFF_HARD,
                "results": results,
                "skipped_fields": skipped_fields,
            }

        consumed_spans: list[tuple[int, int]] = []

        if not self._is_empty_value(size_value):
            size_result = self.size_splitter.analyze(clean_text, size_value)
            results["SIZE"] = size_result.to_dict()
            consumed_spans = list(size_result.consumed_spans)

        if not self._is_empty_value(thickness_value):
            thickness_result = self.thickness_splitter.analyze(
                clean_text,
                thickness_value,
                consumed_spans=consumed_spans,
            )
            results["THICKNESS"] = thickness_result.to_dict()
            consumed_spans = list(thickness_result.consumed_spans)

        if not self._is_empty_value(pressure_value):
            pressure_result = self.pressure_splitter.analyze(
                clean_text,
                pressure_value,
                consumed_spans=consumed_spans,
            )
            results["PRESSURE"] = pressure_result.to_dict()

        material_code = self._clean(material_code)
        if material_code:
            results["MATERIAL"] = self.material_splitter.analyze(clean_text, material_code).to_dict()

        type_code = self._clean(type_code)
        if type_code:
            results["TYPE"] = self.type_splitter.analyze(clean_text, type_code).to_dict()

        normalized_standard_items = self._normalize_standard_items(standard_items)
        if normalized_standard_items:
            results["STANDARD"] = self.standard_splitter.analyze_items(clean_text, normalized_standard_items).to_dict()

        final_level = self._build_final_level(
            clean_difficulty,
            results,
            size_value=size_value,
            thickness_value=thickness_value,
            pressure_value=pressure_value,
            material_code=material_code,
            type_code=type_code,
            standard_items=normalized_standard_items,
        )
        return {
            "text": clean_text,
            "stage1_difficulty": clean_difficulty,
            "final_level": final_level,
            "results": results,
            "skipped_fields": skipped_fields,
        }

    @staticmethod
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    @classmethod
    def _extract_stage1_value(cls, field_obj: Any) -> Any:
        if not isinstance(field_obj, dict):
            return None
        stage1_raw = field_obj.get("stage1_raw")
        if isinstance(stage1_raw, dict):
            value = stage1_raw.get("value")
            if not cls._is_empty_value(value):
                return value
        stage2_input = field_obj.get("stage2_input")
        if isinstance(stage2_input, dict):
            value = stage2_input.get("value")
            if not cls._is_empty_value(value):
                return value
        return None

    @classmethod
    def _extract_code(cls, field_obj: Any) -> str:
        if not isinstance(field_obj, dict):
            return ""
        stage2_output = field_obj.get("stage2_output")
        if isinstance(stage2_output, dict):
            return cls._clean(stage2_output.get("code"))
        return ""

    @classmethod
    def _extract_standard_items(cls, field_obj: Any) -> list[dict[str, str]]:
        if not isinstance(field_obj, dict):
            return []
        stage2_input = field_obj.get("stage2_input")
        value = stage2_input.get("value") if isinstance(stage2_input, dict) else None
        return cls._normalize_standard_items(value)

    @classmethod
    def _normalize_standard_items(cls, items: Any) -> list[dict[str, str]]:
        if not isinstance(items, list):
            return []
        normalized: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            code = cls._clean(item.get("BODY") or item.get("code"))
            if not code:
                continue
            normalized.append(
                {
                    "code": code,
                    "category": cls._clean(item.get("CATEGORY") or item.get("category")),
                }
            )
        return normalized

    @classmethod
    def _is_empty_value(cls, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return cls._clean(value) == ""
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).startswith("_"):
                    continue
                if not cls._is_empty_value(item):
                    return False
            return True
        return False

    @staticmethod
    def _field_provided(
        field_name: str,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> bool:
        if field_name == "SIZE":
            return not PlatformSecondPassRunner._is_empty_value(size_value)
        if field_name == "THICKNESS":
            return not PlatformSecondPassRunner._is_empty_value(thickness_value)
        if field_name == "PRESSURE":
            return not PlatformSecondPassRunner._is_empty_value(pressure_value)
        if field_name == "MATERIAL":
            return bool(material_code)
        if field_name == "TYPE":
            return bool(type_code)
        if field_name == "STANDARD":
            return bool(standard_items)
        return False

    @classmethod
    def _second_easy_field_presence(
        cls,
        *,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> dict[str, bool]:
        return {
            "SIZE": not cls._is_empty_value(size_value),
            "THICKNESS": not cls._is_empty_value(thickness_value),
            "PRESSURE": not cls._is_empty_value(pressure_value),
            "MATERIAL": bool(cls._clean(material_code)),
            "TYPE": bool(cls._clean(type_code)),
            "STANDARD": bool(standard_items),
        }

    @classmethod
    def _is_second_easy_eligible(
        cls,
        *,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> bool:
        presence = cls._second_easy_field_presence(
            size_value=size_value,
            thickness_value=thickness_value,
            pressure_value=pressure_value,
            material_code=material_code,
            type_code=type_code,
            standard_items=standard_items,
        )
        base_required = presence["TYPE"] and presence["SIZE"] and presence["MATERIAL"] and presence["STANDARD"]
        thickness_or_pressure = presence["THICKNESS"] or presence["PRESSURE"]
        return base_required and thickness_or_pressure

    @classmethod
    def _build_final_level(
        cls,
        stage1_difficulty: int | None,
        results: dict[str, Any],
        *,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> int:
        if stage1_difficulty is not None and stage1_difficulty != DIFF_EASY:
            return DIFF_HARD
        material_result = results.get("MATERIAL") if isinstance(results, dict) else None
        material_reason = cls._clean(material_result.get("reason")) if isinstance(material_result, dict) else ""
        if material_reason.startswith("文本命中后缀表达，但编码缺少后缀"):
            return DIFF_HARD
        provided_results = [payload for payload in results.values() if isinstance(payload, dict)]
        if not provided_results:
            return DIFF_EASY if stage1_difficulty == DIFF_EASY else DIFF_HARD
        if not cls._is_second_easy_eligible(
            size_value=size_value,
            thickness_value=thickness_value,
            pressure_value=pressure_value,
            material_code=material_code,
            type_code=type_code,
            standard_items=standard_items,
        ):
            return DIFF_EASY
        all_passed = all(bool(payload.get("passed")) for payload in provided_results)
        if stage1_difficulty == DIFF_EASY and all_passed:
            return DIFF_SECOND_EASY
        return DIFF_EASY

    @staticmethod
    def _make_check(field: str, reason: str) -> dict[str, str]:
        return {
            "field": str(field or "").strip(),
            "reason": str(reason or "").strip(),
            "stage": "second_pass",
            "rule": "second_pass",
        }

    @staticmethod
    def _field_label(field_name: str) -> str:
        mapping = {
            "TYPE": "种类",
            "SIZE": "尺寸",
            "THICKNESS": "壁厚",
            "PRESSURE": "磅级",
            "MATERIAL": "材质",
            "STANDARD": "规范",
            "THICKNESS_OR_PRESSURE": "壁厚或磅级",
        }
        return mapping.get(str(field_name or "").strip(), str(field_name or "").strip())

    @classmethod
    def _build_missing_required_checks(
        cls,
        *,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> list[str]:
        presence = cls._second_easy_field_presence(
            size_value=size_value,
            thickness_value=thickness_value,
            pressure_value=pressure_value,
            material_code=material_code,
            type_code=type_code,
            standard_items=standard_items,
        )
        missing: list[str] = []
        for field_name in ("TYPE", "SIZE", "MATERIAL", "STANDARD"):
            if not presence[field_name]:
                missing.append(field_name)
        if not (presence["THICKNESS"] or presence["PRESSURE"]):
            missing.append("THICKNESS_OR_PRESSURE")
        return missing

    @classmethod
    def _summarize_result(
        cls,
        detailed: dict[str, Any],
        *,
        size_value: Any,
        thickness_value: Any,
        pressure_value: Any,
        material_code: str,
        type_code: str,
        standard_items: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        stage1_level = normalize_difficulty_level(detailed.get("stage1_difficulty"))
        final_level = normalize_difficulty_level(detailed.get("final_level"))
        results = detailed.get("results") if isinstance(detailed.get("results"), dict) else {}

        failed_checks: list[dict[str, str]] = []
        passed_checks: list[str] = []
        for field, payload in results.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("passed"):
                passed_checks.append(str(field))
                continue
            reason = cls._clean(payload.get("reason"))
            if reason:
                failed_checks.append(cls._make_check(str(field), reason))

        missing_required_checks = cls._build_missing_required_checks(
            size_value=size_value,
            thickness_value=thickness_value,
            pressure_value=pressure_value,
            material_code=material_code,
            type_code=type_code,
            standard_items=standard_items,
        )
        missing_required_labels = [cls._field_label(field_name) for field_name in missing_required_checks]

        if failed_checks:
            reason_text = " | ".join(
                f"{cls._field_label(check['field'])}: {check['reason']}"
                for check in failed_checks
                if check["reason"]
            )
        elif final_level == DIFF_SECOND_EASY:
            reason_text = "二次分流全部通过"
        elif missing_required_checks:
            reason_text = f"未满足自动通过条件: 缺少 {'、'.join(missing_required_labels)}"
        elif passed_checks:
            reason_text = "二次分流已回查，维持中等"
        else:
            reason_text = "未进入二次分流"

        return {
            "stage1_level": stage1_level,
            "stage1_difficulty": stage1_level,
            "final_level": final_level,
            "decision_stage": "second_pass" if stage1_level == DIFF_EASY else "stage1",
            "need_review": final_level != DIFF_SECOND_EASY,
            "reason_text": reason_text,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "missing_required_checks": missing_required_checks,
        }
