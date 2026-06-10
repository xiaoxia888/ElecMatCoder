# -*- coding: utf-8 -*-
"""TYPE 编码器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import yaml

TYPE_COMBO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "type_combo_mapping.yaml"
TYPE_RULE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "type_rule_mapping.yaml"


@dataclass
class TypeEncodingResult:
    code: str = ""
    resolved: bool = False
    strategy: str = ""
    body: str = ""
    angle: str = ""
    radius: str = ""
    manu: List[str] = field(default_factory=list)
    conn: List[str] = field(default_factory=list)
    seal: List[str] = field(default_factory=list)
    reason: str = ""
    matched_key: str = ""
    tried_keys: List[str] = field(default_factory=list)


class TypeEncoder:
    def __init__(self, combo_config_path: Optional[str] = None, rule_config_path: Optional[str] = None):
        self.combo_config_path = Path(combo_config_path) if combo_config_path else TYPE_COMBO_CONFIG
        self.rule_config_path = Path(rule_config_path) if rule_config_path else TYPE_RULE_CONFIG
        self.combo_config = self._load_yaml(self.combo_config_path)
        self.rule_config = self._load_yaml(self.rule_config_path)
        self.reverse_combo_mapping = self._build_reverse_mapping(self.combo_config)
        self.manu_priority = self.rule_config.get("manu_priority", [])
        self.conn_priority = self.rule_config.get("conn_priority", [])

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _build_reverse_mapping(mapping_config: Dict[str, Any]) -> Dict[str, str]:
        reverse: Dict[str, str] = {}
        for code, aliases in mapping_config.items():
            values = aliases if isinstance(aliases, list) else [aliases]
            for value in values:
                text = str(value).strip()
                if text:
                    reverse[text] = str(code).strip()
        return reverse

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _first_nonempty(values: Iterable[str]) -> str:
        for value in values:
            text = str(value).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _get_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _pick_by_priority(self, values: Iterable[str], priority: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        if not normalized:
            return []
        for item in priority:
            if item in seen:
                return [item]
        return normalized[:1]

    def _normalize_manu(self, manu: Iterable[str]) -> List[str]:
        return self._pick_by_priority(manu, self.manu_priority)

    def _normalize_conn(self, conn: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in conn:
            text = str(value).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        if not normalized:
            return []

        composite_conn = next(
            (
                item for item in normalized
                if any(marker in item for marker in ('/', '(', ')'))
            ),
            ''
        )
        if composite_conn:
            return [composite_conn]

        return self._pick_by_priority(normalized, self.conn_priority)

    @staticmethod
    def _normalize_angle(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_radius(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _build_key(parts: Iterable[str]) -> str:
        return ";".join(str(part).strip() for part in parts if str(part).strip())

    def _build_lookup_keys(self, *, body: str, angle: str, radius: str, seal: str, conn: str, manu: str) -> List[str]:
        key = self._build_key([body, angle, radius, seal, conn, manu])
        return [key] if key else []

    def _resolve_by_rules(self, *, body: str, angle: str, radius: str, seal: str, conn: str, manu: str) -> str:
        body_codes = self._get_dict(self.rule_config.get("body_codes"))
        geometry_body_codes = self._get_dict(self.rule_config.get("geometry_body_codes"))
        angle_codes = self._get_dict(self.rule_config.get("angle_codes"))
        radius_codes = self._get_dict(self.rule_config.get("radius_codes"))
        seal_codes = self._get_dict(self.rule_config.get("seal_codes"))
        conn_codes = self._get_dict(self.rule_config.get("conn_codes"))
        manu_codes = self._get_dict(self.rule_config.get("manu_codes"))
        body_manu_overrides = self._get_dict(self.rule_config.get("body_manu_code_overrides"))
        body_implicit_conn = self._get_dict(self.rule_config.get("body_implicit_conn"))

        base_code = ""
        if body in geometry_body_codes:
            if not angle or angle not in angle_codes:
                return ""
            angle_code = str(angle_codes[angle]).strip()
            body_code = str(geometry_body_codes[body]).strip()
            if radius:
                if radius not in radius_codes:
                    return ""
                radius_code = str(radius_codes[radius]).strip()
            else:
                radius_code = ""
            base_code = f"{angle_code}{body_code}{radius_code}"
        else:
            base_code = str(body_codes.get(body, "")).strip()
            if not base_code:
                return ""

        if manu:
            body_override = self._get_dict(body_manu_overrides.get(body))
            if manu in body_override:
                return str(body_override[manu]).strip()

        implicit_conn_values = {
            str(item).strip()
            for item in self._as_list(body_implicit_conn.get(body))
            if str(item).strip()
        }
        if conn and conn in implicit_conn_values:
            conn = ""

        seal_suffix = ""
        if seal:
            if seal not in seal_codes:
                return ""
            seal_suffix = str(seal_codes[seal]).strip()

        conn_suffix = ""
        if conn:
            if conn not in conn_codes:
                return ""
            conn_suffix = str(conn_codes[conn]).strip()

        manu_suffix = ""
        if manu:
            if manu not in manu_codes:
                return ""
            manu_suffix = str(manu_codes[manu]).strip()

        return f"{base_code}{seal_suffix}{conn_suffix}{manu_suffix}"

    def encode(self, type_dict: Dict[str, Any]) -> TypeEncodingResult:
        body = str(type_dict.get("BODY", "") or "").strip()
        geometry = type_dict.get("GEOMETRY") or {}
        angle = self._normalize_angle(geometry.get("ANGLE", ""))
        radius = self._normalize_radius(geometry.get("RADIUS", ""))
        manu = self._normalize_manu(self._as_list(type_dict.get("MANU")))
        conn = self._normalize_conn(self._as_list(type_dict.get("CONN")))
        seal = self._as_list(type_dict.get("SEAL"))
        seal_value = self._first_nonempty(seal)
        conn_value = self._first_nonempty(conn)
        manu_value = self._first_nonempty(manu)

        result = TypeEncodingResult(body=body, angle=angle, radius=radius, manu=manu, conn=conn, seal=seal)
        if not body:
            result.reason = "empty_body"
            return result

        lookup_keys = self._build_lookup_keys(body=body, angle=angle, radius=radius, seal=seal_value, conn=conn_value, manu=manu_value)
        result.tried_keys = lookup_keys
        for key in lookup_keys:
            code = self.reverse_combo_mapping.get(key)
            if not code:
                continue
            result.code = code
            result.resolved = True
            result.strategy = "combo_mapping"
            result.reason = "matched_combo_mapping"
            result.matched_key = key
            return result

        geometry_body_codes = self._get_dict(self.rule_config.get("geometry_body_codes"))
        if angle and body not in geometry_body_codes:
            result.strategy = "unresolved"
            result.reason = "angle_not_supported_for_non_geometry_body"
            return result

        code = self._resolve_by_rules(
            body=body,
            angle=angle,
            radius=radius,
            seal=seal_value,
            conn=conn_value,
            manu=manu_value,
        )
        if code:
            result.code = code
            result.resolved = True
            result.strategy = "rule_mapping"
            result.reason = "matched_rule_mapping"
            result.matched_key = self._build_key([body, angle, radius, seal_value, conn_value, manu_value])
            return result

        result.strategy = "unresolved"
        result.reason = "no_combo_mapping_or_rule_mapping"
        return result


def get_type_encoder(combo_config_path: Optional[str] = None, rule_config_path: Optional[str] = None) -> TypeEncoder:
    return TypeEncoder(combo_config_path=combo_config_path, rule_config_path=rule_config_path)
