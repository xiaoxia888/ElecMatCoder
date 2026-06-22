# -*- coding: utf-8 -*-
"""TYPE 编码器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import yaml

TYPE_COMBO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "type_combo_mapping.yaml"
TYPE_RULE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "type_rule_mapping.yaml"
DEFAULT_TYPE_KEY_ORDER = ["FLANGE_STYLE", "BODY", "ANGLE", "RADIUS", "SEAL", "CONN", "MANU"]


@dataclass
class TypeEncodingResult:
    code: str = ""
    resolved: bool = False
    strategy: str = ""
    flange_style: str = ""
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
        self.type_key_order = [
            str(item).strip().upper()
            for item in (self.rule_config.get("type_key_order") or DEFAULT_TYPE_KEY_ORDER)
            if str(item).strip()
        ] or list(DEFAULT_TYPE_KEY_ORDER)

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

    def _dedupe_values(self, values: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized

    def _normalize_manu(self, manu: Iterable[str]) -> List[str]:
        return self._dedupe_values(manu)

    def _normalize_conn(self, conn: Iterable[str]) -> List[str]:
        return self._dedupe_values(conn)

    @staticmethod
    def _normalize_angle(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_radius(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_flange_style(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _flatten_key_parts(parts: Iterable[Any]) -> List[str]:
        flattened: List[str] = []
        for part in parts:
            if part in (None, "", []):
                continue
            if isinstance(part, list):
                for item in part:
                    text = str(item).strip()
                    if text:
                        flattened.append(text)
                continue
            text = str(part).strip()
            if text:
                flattened.append(text)
        return flattened

    def _build_key(self, parts: Iterable[Any]) -> str:
        return ";".join(self._flatten_key_parts(parts))

    def _build_key_from_components(
        self,
        *,
        flange_style: str,
        body: str,
        angle: str,
        radius: str,
        seal: List[str],
        conn: List[str],
        manu: List[str],
    ) -> str:
        component_map = {
            "FLANGE_STYLE": flange_style,
            "BODY": body,
            "ANGLE": angle,
            "RADIUS": radius,
            "SEAL": seal,
            "CONN": conn,
            "MANU": manu,
        }
        ordered_parts = [component_map.get(key, "") for key in self.type_key_order]
        return self._build_key(ordered_parts)

    def _build_lookup_keys(
        self,
        *,
        flange_style: str,
        body: str,
        angle: str,
        radius: str,
        seal: List[str],
        conn: List[str],
        manu: List[str],
    ) -> List[str]:
        key = self._build_key_from_components(
            flange_style=flange_style,
            body=body,
            angle=angle,
            radius=radius,
            seal=seal,
            conn=conn,
            manu=manu,
        )
        return [key] if key else []

    def _resolve_by_rules(self, *, flange_style: str, body: str, angle: str, radius: str, seal: List[str], conn: List[str], manu: List[str]) -> str:
        flange_style_codes = self._get_dict(self.rule_config.get("flange_style_codes"))
        body_codes = self._get_dict(self.rule_config.get("body_codes"))
        geometry_body_codes = self._get_dict(self.rule_config.get("geometry_body_codes"))
        angle_codes = self._get_dict(self.rule_config.get("angle_codes"))
        radius_codes = self._get_dict(self.rule_config.get("radius_codes"))
        seal_codes = self._get_dict(self.rule_config.get("seal_codes"))
        conn_codes = self._get_dict(self.rule_config.get("conn_codes"))
        manu_codes = self._get_dict(self.rule_config.get("manu_codes"))
        body_manu_overrides = self._get_dict(self.rule_config.get("body_manu_code_overrides"))
        body_implicit_conn = self._get_dict(self.rule_config.get("body_implicit_conn"))
        flange_prefix = ""

        if flange_style:
            flange_prefix = str(flange_style_codes.get(flange_style, "")).strip()
            if not flange_prefix:
                return ""

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

        if len(manu) == 1:
            body_override = self._get_dict(body_manu_overrides.get(body))
            manu_value = manu[0]
            if manu_value in body_override:
                override_code = str(body_override[manu_value]).strip()
                return f"{flange_prefix}{override_code}" if override_code else ""

        implicit_conn_values = {
            str(item).strip()
            for item in self._as_list(body_implicit_conn.get(body))
            if str(item).strip()
        }
        conn = [item for item in conn if item not in implicit_conn_values]

        seal_suffix_parts: List[str] = []
        for seal_value in seal:
            if seal_value not in seal_codes:
                return ""
            seal_suffix_parts.append(str(seal_codes[seal_value]).strip())

        conn_suffix_parts: List[str] = []
        for conn_value in conn:
            if conn_value not in conn_codes:
                return ""
            conn_suffix_parts.append(str(conn_codes[conn_value]).strip())

        manu_suffix_parts: List[str] = []
        for manu_value in manu:
            if manu_value not in manu_codes:
                return ""
            manu_suffix_parts.append(str(manu_codes[manu_value]).strip())

        return f"{flange_prefix}{base_code}{''.join(seal_suffix_parts)}{''.join(conn_suffix_parts)}{''.join(manu_suffix_parts)}"

    def encode(self, type_dict: Dict[str, Any]) -> TypeEncodingResult:
        flange_style = self._normalize_flange_style(type_dict.get("FLANGE_STYLE", ""))
        body = str(type_dict.get("BODY", "") or "").strip()
        geometry = type_dict.get("GEOMETRY") or {}
        angle = self._normalize_angle(geometry.get("ANGLE", ""))
        radius = self._normalize_radius(geometry.get("RADIUS", ""))
        manu = self._normalize_manu(self._as_list(type_dict.get("MANU")))
        conn = self._normalize_conn(self._as_list(type_dict.get("CONN")))
        seal = self._dedupe_values(self._as_list(type_dict.get("SEAL")))

        result = TypeEncodingResult(flange_style=flange_style, body=body, angle=angle, radius=radius, manu=manu, conn=conn, seal=seal)
        if not body:
            result.reason = "empty_body"
            return result

        lookup_keys = self._build_lookup_keys(
            flange_style=flange_style,
            body=body,
            angle=angle,
            radius=radius,
            seal=seal,
            conn=conn,
            manu=manu,
        )
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
            flange_style=flange_style,
            body=body,
            angle=angle,
            radius=radius,
            seal=seal,
            conn=conn,
            manu=manu,
        )
        if code:
            result.code = code
            result.resolved = True
            result.strategy = "rule_mapping"
            result.reason = "matched_rule_mapping"
            result.matched_key = self._build_key_from_components(
                flange_style=flange_style,
                body=body,
                angle=angle,
                radius=radius,
                seal=seal,
                conn=conn,
                manu=manu,
            )
            return result

        result.strategy = "unresolved"
        result.reason = "no_combo_mapping_or_rule_mapping"
        return result


def get_type_encoder(combo_config_path: Optional[str] = None, rule_config_path: Optional[str] = None) -> TypeEncoder:
    return TypeEncoder(combo_config_path=combo_config_path, rule_config_path=rule_config_path)
