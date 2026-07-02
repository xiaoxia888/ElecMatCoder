from __future__ import annotations

import re
from typing import Any, Dict, List


class StructuralFieldOutputNormalizer:
    """统一归一化结构字段模型输出。"""

    SIZE_KEYS = ("DN", "OD", "INCH", "LENGTH")
    THICKNESS_KEYS = ("MM", "SCHEDULE", "BWG", "INCH")
    ITEM_TYPES = {
        "SIZE_ITEMS": set(SIZE_KEYS),
        "THICKNESS_ITEMS": set(THICKNESS_KEYS),
    }

    @classmethod
    def empty_result(cls) -> Dict[str, Any]:
        return {
            "SIZE": {key: [] for key in cls.SIZE_KEYS},
            "SIZE_ITEMS": [],
            "LENGTH": "",
            "THICKNESS": {key: [] for key in cls.THICKNESS_KEYS},
            "THICKNESS_ITEMS": [],
            "PRESSURE": "",
        }

    @classmethod
    def normalize(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        result = cls.empty_result()

        size = parsed.get("SIZE")
        if isinstance(size, dict):
            for key in cls.SIZE_KEYS:
                result["SIZE"][key] = cls.normalize_list(size.get(key))

        result["SIZE_ITEMS"] = cls.normalize_items(
            parsed.get("SIZE_ITEMS"),
            cls.ITEM_TYPES["SIZE_ITEMS"],
        )
        top_level_length = cls.normalize_item_value("LENGTH", parsed.get("LENGTH", ""))
        result["LENGTH"] = top_level_length
        if top_level_length:
            result["SIZE"]["LENGTH"] = cls.normalize_list([top_level_length])
            if ("LENGTH", top_level_length) not in {
                (str(item.get("type", "")).strip().upper(), str(item.get("value", "")).strip())
                for item in result["SIZE_ITEMS"]
            }:
                result["SIZE_ITEMS"].append({"type": "LENGTH", "value": top_level_length})
        if result["SIZE_ITEMS"]:
            result["SIZE"] = cls.group_items(result["SIZE_ITEMS"], cls.SIZE_KEYS)
            result["LENGTH"] = result["SIZE"]["LENGTH"][0] if result["SIZE"]["LENGTH"] else top_level_length

        thickness = parsed.get("THICKNESS")
        if isinstance(thickness, dict):
            for key in cls.THICKNESS_KEYS:
                result["THICKNESS"][key] = cls.normalize_list(thickness.get(key))

        result["THICKNESS_ITEMS"] = cls.normalize_items(
            parsed.get("THICKNESS_ITEMS"),
            cls.ITEM_TYPES["THICKNESS_ITEMS"],
        )
        if result["THICKNESS_ITEMS"]:
            result["THICKNESS"] = cls.group_items(result["THICKNESS_ITEMS"], cls.THICKNESS_KEYS)

        pressure = parsed.get("PRESSURE")
        result["PRESSURE"] = "" if pressure in (None, [], {}) else str(pressure).strip()
        return result

    @staticmethod
    def normalize_list(value: Any) -> List[str]:
        if value in (None, "", []):
            return []
        if not isinstance(value, list):
            value = [value]
        result: List[str] = []
        seen = set()
        for item in value:
            if item in (None, ""):
                continue
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @classmethod
    def normalize_items(cls, value: Any, allowed_types: set[str]) -> List[Dict[str, str]]:
        if value in (None, "", []):
            return []
        if not isinstance(value, list):
            return []
        result: List[Dict[str, str]] = []
        seen = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().upper()
            item_value = cls.normalize_item_value(item_type, item.get("value", ""))
            if not item_type or not item_value or item_type not in allowed_types:
                continue
            key = (item_type, item_value)
            if key in seen:
                continue
            seen.add(key)
            result.append({"type": item_type, "value": item_value})
        return result

    @staticmethod
    def normalize_item_value(item_type: str, raw_value: Any) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""

        kind = str(item_type or "").strip().upper()
        normalized = text.replace("”", "\"").replace("“", "\"").replace("″", "\"").strip()

        if kind == "DN":
            matched = re.fullmatch(r"(?i)DN\s*(\d+(?:\.\d+)?)", normalized)
            if matched:
                return matched.group(1)
            return normalized

        if kind == "OD":
            matched = re.fullmatch(r"(?i)(?:OD|[ΦφФфØø])\s*(\d+(?:\.\d+)?)", normalized)
            if matched:
                return matched.group(1)
            return normalized

        if kind == "INCH":
            normalized = re.sub(r"(?i)^NPS\s*", "", normalized)
            if normalized.endswith('"'):
                normalized = normalized[:-1].strip()
            return re.sub(r"\s+", "", normalized)

        if kind in {"MM", "LENGTH"}:
            matched = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*MM)?", normalized, flags=re.IGNORECASE)
            if matched:
                return matched.group(1)
            return normalized.upper()

        if kind == "BWG":
            matched = re.fullmatch(r"(?i)(?:BWG\s*)?(\d+(?:\.\d+)?)", normalized)
            if matched:
                return matched.group(1)
            return normalized.upper()

        return normalized.upper() if kind in {"SCHEDULE", "SERIES"} else normalized

    @staticmethod
    def group_items(items: List[Dict[str, str]], keys: tuple[str, ...]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {key: [] for key in keys}
        for item in items:
            item_type = str(item.get("type", "")).strip().upper()
            item_value = str(item.get("value", "")).strip()
            if not item_type or not item_value or item_type not in grouped:
                continue
            grouped[item_type].append(item_value)
        return grouped
