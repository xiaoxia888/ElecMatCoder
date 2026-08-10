# -*- coding: utf-8 -*-
"""MATERIAL 编码器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


MATERIAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "material_mapping.yaml"


@dataclass
class MaterialEncodingResult:
    code: str = ""
    resolved: bool = False
    strategy: str = ""
    value: str = ""
    special_req: List[str] = field(default_factory=list)
    reason: str = ""
    matched_code: str = ""
    matched_value: str = ""
    matched_suffixes: List[str] = field(default_factory=list)


class MaterialEncoder:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else MATERIAL_CONFIG
        self.config = self._load_yaml(self.config_path)
        self.value_mapping = self.config.get("value_mapping", {}) or {}
        self.special_req_suffix = self.config.get("special_req_suffix", {}) or {}
        self.composition_overrides = self._build_composition_overrides(
            self.config.get("composition_overrides", {}) or {}
        )
        self.reverse_value_mapping = self._build_reverse_value_mapping(self.value_mapping)
        self.reverse_special_req_suffix = self._build_reverse_special_req_mapping(self.special_req_suffix)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _build_reverse_value_mapping(value_mapping: Dict[str, Iterable[str]]) -> Dict[str, str]:
        reverse: Dict[str, str] = {}
        for code, aliases in value_mapping.items():
            code = str(code).strip()
            for alias in aliases or []:
                alias = str(alias).strip()
                if alias:
                    reverse[alias.upper()] = code
        return reverse

    @staticmethod
    def _build_reverse_special_req_mapping(special_req_suffix: Dict[str, Iterable[str]]) -> Dict[str, str]:
        reverse: Dict[str, str] = {}
        for suffix, aliases in special_req_suffix.items():
            suffix = str(suffix).strip()
            for alias in aliases or []:
                alias = str(alias).strip()
                if alias:
                    reverse[alias.upper()] = suffix
        return reverse

    @staticmethod
    def _build_composition_overrides(overrides: Dict[str, Any]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for source, target in overrides.items():
            source_code = str(source).strip().upper()
            target_code = str(target).strip()
            if source_code and target_code:
                normalized[source_code] = target_code
        return normalized

    def apply_composition_override(self, combined_code: str) -> str:
        """对已完成默认拼接的材质编码执行完整键精确覆盖。"""
        code = str(combined_code or "").strip()
        if not code:
            return ""
        return self.composition_overrides.get(code.upper(), code)

    def encode(self, material_item: Dict[str, Any]) -> MaterialEncodingResult:
        value = str(material_item.get("VALUE") or "").strip()
        special_req = self._as_list(material_item.get("SPECIAL_REQ"))
        result = MaterialEncodingResult(value=value, special_req=special_req)

        if not value:
            result.reason = "empty_value"
            return result

        base_code = self.reverse_value_mapping.get(value.upper(), "")
        if not base_code:
            result.reason = "value_not_mapped"
            return result

        suffixes: List[str] = []
        seen = set()
        for req in special_req:
            suffix = self.reverse_special_req_suffix.get(req.upper(), "")
            if not suffix:
                result.reason = "special_req_not_mapped"
                return result
            if suffix not in seen:
                suffixes.append(suffix)
                seen.add(suffix)

        result.code = f"{base_code}{''.join(suffixes)}"
        result.resolved = True
        result.strategy = "material_mapping"
        result.reason = "matched_material_mapping"
        result.matched_code = base_code
        result.matched_value = value
        result.matched_suffixes = suffixes
        return result


def get_material_encoder(config_path: Optional[str] = None) -> MaterialEncoder:
    return MaterialEncoder(config_path=config_path)
