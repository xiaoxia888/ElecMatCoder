# -*- coding: utf-8 -*-
"""Detect uncommon encoded type/material codes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .models import DifficultyFeature, GlueHit


class UncommonCodeDetector:
    """Tag uncommon type/material/standard codes after encoding."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "common_code_mapping.yaml"
        )
        self._load_config()

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.common_type_codes = {str(v).strip() for v in data.get("common_type_codes", []) if str(v).strip()}
        self.common_material_codes = {
            str(v).strip() for v in data.get("common_material_codes", []) if str(v).strip()
        }
        self.common_standard_codes = {
            self._normalize_standard_code(str(v).strip())
            for v in data.get("common_standard_codes", [])
            if str(v).strip()
        }

    @staticmethod
    def _normalize_standard_code(value: str) -> str:
        # 常见规范比较只关心编码语义，不关心大小写差异，例如 Ia/IA。
        return value.strip().upper()

    @staticmethod
    def _normalize_standard_codes(value: str | Iterable[str] | None) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("|", "｜").split("｜")]
            return [part for part in parts if part]
        normalized: list[str] = []
        for item in value:
            item_text = str(item).strip()
            if item_text:
                normalized.append(item_text)
        return normalized

    def analyze(
        self,
        text: str,
        type_code: str = "",
        material_code: str = "",
        standard_code: str | Iterable[str] | None = "",
    ) -> DifficultyFeature:
        hits: list[GlueHit] = []
        reasons: list[str] = []

        norm_type = (type_code or "").strip()
        norm_material = (material_code or "").strip()
        norm_standard_codes = self._normalize_standard_codes(standard_code)

        if norm_type and norm_type not in self.common_type_codes:
            reasons.append("种类编码不常见")
            hits.append(
                GlueHit(
                    tag="uncommon_type_code",
                    code_group="type_code",
                    code=norm_type,
                    token=norm_type,
                    start=-1,
                    end=-1,
                    note=f"种类编码 {norm_type} 不在常见种类编码表中",
                )
            )

        if norm_material and norm_material not in self.common_material_codes:
            reasons.append("材质编码不常见")
            hits.append(
                GlueHit(
                    tag="uncommon_material_code",
                    code_group="material_code",
                    code=norm_material,
                    token=norm_material,
                    start=-1,
                    end=-1,
                    note=f"材质编码 {norm_material} 不在常见材质编码表中",
                )
            )

        uncommon_standard_codes = [
            code
            for code in norm_standard_codes
            if self._normalize_standard_code(code) not in self.common_standard_codes
        ]
        if uncommon_standard_codes:
            reasons.append(f"规范编码不常见: {'｜'.join(uncommon_standard_codes)}")
            for norm_standard in uncommon_standard_codes:
                hits.append(
                    GlueHit(
                        tag="uncommon_standard_code",
                        code_group="standard_code",
                        code=norm_standard,
                        token=norm_standard,
                        start=-1,
                        end=-1,
                        note=f"规范编码 {norm_standard} 不在常见规范编码表中",
                    )
                )

        reason = "、".join(reasons)
        return DifficultyFeature(
            name="uncommon_code",
            matched=bool(hits),
            reason=reason,
            hits=hits,
        )
