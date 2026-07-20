from __future__ import annotations

import re
from typing import Any, Dict, List


class StructuralFieldOutputNormalizer:
    """统一归一化结构字段模型输出。"""

    SIZE_KEYS = ("DN", "OD", "INCH", "LENGTH")
    THICKNESS_KEYS = ("MM", "SCHEDULE", "BWG", "INCH")
    ROLE_KEYS = {"BASE", "LINING", "INNER", "OUTER"}
    ITEM_TYPES = {
        "SIZE_ITEMS": set(SIZE_KEYS),
        "THICKNESS_ITEMS": set(THICKNESS_KEYS),
    }

    @classmethod
    def empty_result(cls) -> Dict[str, Any]:
        return {
            "SIZE": {"_ITEMS": []},
            "SIZE_ITEMS": [],
            "LENGTH": "",
            "THICKNESS": {"_ITEMS": []},
            "THICKNESS_ITEMS": [],
            "PRESSURE": "",
        }

    @classmethod
    def normalize(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        result = cls.empty_result()
        structure_kind = cls.normalize_structure_kind(parsed.get("STRUCTURE_KIND"))
        complex_meta: Dict[str, Any] = {}

        raw_size_items = parsed.get("SIZE_ITEMS")
        size_items_with_role = cls.normalize_items_with_role(
            raw_size_items,
            cls.ITEM_TYPES["SIZE_ITEMS"],
        )
        if size_items_with_role:
            complex_meta["SIZE_ITEMS_WITH_ROLE"] = size_items_with_role
        result["SIZE_ITEMS"] = (
            size_items_with_role
            if size_items_with_role
            else cls.normalize_items(raw_size_items, cls.ITEM_TYPES["SIZE_ITEMS"])
        )
        top_level_length = cls.normalize_item_value("LENGTH", parsed.get("LENGTH", ""))
        result["LENGTH"] = top_level_length
        if top_level_length:
            if ("LENGTH", top_level_length) not in {
                (str(item.get("type", "")).strip().upper(), str(item.get("value", "")).strip())
                for item in result["SIZE_ITEMS"]
            }:
                result["SIZE_ITEMS"].append({"type": "LENGTH", "value": top_level_length})
        if result["SIZE_ITEMS"]:
            result["SIZE"] = cls.group_ordered_items_only(result["SIZE_ITEMS"], cls.SIZE_KEYS)
            length_values = [
                str(item.get("value") or "").strip()
                for item in result["SIZE_ITEMS"]
                if str(item.get("type") or "").strip().upper() == "LENGTH" and str(item.get("value") or "").strip()
            ]
            result["LENGTH"] = length_values[0] if length_values else top_level_length

        raw_thickness_items = parsed.get("THICKNESS_ITEMS")
        thickness_items_with_role = cls.normalize_items_with_role(
            raw_thickness_items,
            cls.ITEM_TYPES["THICKNESS_ITEMS"],
        )
        if thickness_items_with_role:
            complex_meta["THICKNESS_ITEMS_WITH_ROLE"] = thickness_items_with_role
        result["THICKNESS_ITEMS"] = (
            thickness_items_with_role
            if thickness_items_with_role
            else cls.normalize_items(raw_thickness_items, cls.ITEM_TYPES["THICKNESS_ITEMS"])
        )
        if result["THICKNESS_ITEMS"]:
            result["THICKNESS"] = cls.group_ordered_items_only(result["THICKNESS_ITEMS"], cls.THICKNESS_KEYS)

        pressure = parsed.get("PRESSURE")
        result["PRESSURE"] = "" if pressure in (None, [], {}) else str(pressure).strip()
        if structure_kind:
            complex_meta["STRUCTURE_KIND"] = structure_kind
        if complex_meta:
            result["_complex_structure"] = complex_meta
        return result

    @staticmethod
    def normalize_structure_kind(value: Any) -> str:
        text = str(value or "").strip().upper()
        return text if text in {"NORMAL", "LINED", "JACKETED"} else ""

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

    @classmethod
    def normalize_items_with_role(cls, value: Any, allowed_types: set[str]) -> List[Dict[str, str]]:
        if value in (None, "", []):
            return []
        if not isinstance(value, list):
            return []
        result: List[Dict[str, str]] = []
        seen = set()
        has_any_role = False
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().upper()
            item_value = cls.normalize_item_value(item_type, item.get("value", ""))
            item_role = str(item.get("role", "") or "").strip().upper()
            if not item_type or not item_value or item_type not in allowed_types:
                continue
            has_role = item_role in cls.ROLE_KEYS
            if has_role:
                has_any_role = True
            key = (item_role if has_role else "", item_type, item_value)
            if key in seen:
                continue
            seen.add(key)
            payload = {"type": item_type, "value": item_value}
            if has_role:
                payload = {"role": item_role, **payload}
            result.append(payload)
        return result if has_any_role else []

    @classmethod
    def fold_complex_items(cls, items: List[Dict[str, str]], structure_kind: str) -> List[Dict[str, str]]:
        if not items:
            return []
        if structure_kind == "LINED":
            return cls._fold_lined_items(items)
        if structure_kind == "JACKETED":
            return cls._fold_jacketed_items(items)
        return [{"type": item["type"], "value": item["value"]} for item in items]

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        if value and value not in values:
            values.append(value)

    @classmethod
    def _fold_lined_items(cls, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        role_values: Dict[str, Dict[str, List[str]]] = {}
        first_role_index: Dict[str, int] = {}

        for index, item in enumerate(items):
            role = str(item.get("role", "") or "").upper()
            if role not in {"BASE", "LINING"}:
                continue
            item_type = item["type"]
            first_role_index.setdefault(item_type, index)
            grouped = role_values.setdefault(item_type, {"BASE": [], "LINING": []})
            grouped[role].append(item["value"])

        folded_by_type: Dict[str, List[Dict[str, str]]] = {}
        for item_type, grouped in role_values.items():
            bases = grouped["BASE"]
            linings = grouped["LINING"]
            folded: List[Dict[str, str]] = []
            for index in range(max(len(bases), len(linings))):
                base_value = bases[index] if index < len(bases) else ""
                lining_value = linings[index] if index < len(linings) else ""
                value = (
                    f"{base_value}/{lining_value}"
                    if base_value and lining_value
                    else base_value or lining_value
                )
                if value:
                    folded.append({"type": item_type, "value": value})
            folded_by_type[item_type] = folded

        result: List[Dict[str, str]] = []
        emitted_types = set()
        for index, item in enumerate(items):
            item_type = item["type"]
            role = str(item.get("role", "") or "").upper()
            if role in {"BASE", "LINING"}:
                if item_type not in emitted_types and index == first_role_index[item_type]:
                    result.extend(folded_by_type[item_type])
                    emitted_types.add(item_type)
                continue
            result.append({"type": item_type, "value": item["value"]})
        return result

    @classmethod
    def _fold_jacketed_items(cls, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        current_role = ""
        current_type = ""
        current_values: List[str] = []

        def flush() -> None:
            nonlocal current_role, current_type, current_values
            if current_type and current_values:
                result.append({"type": current_type, "value": "x".join(current_values)})
            current_role = ""
            current_type = ""
            current_values = []

        for item in items:
            item_type = item["type"]
            item_value = item["value"]
            role = str(item.get("role", "") or "").upper()
            group_role = role if role in {"INNER", "OUTER"} else "BASE"
            if current_values and (group_role != current_role or item_type != current_type):
                flush()
            current_role = group_role
            current_type = item_type
            cls._append_unique(current_values, item_value)

        flush()
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
            if kind == "MM" and re.fullmatch(
                r"\d+(?:\.\d+)?(?:\s*[xX×*/]\s*\d+(?:\.\d+)?)+",
                normalized,
            ):
                return re.sub(r"\s+", "", normalized.replace("×", "x").replace("*", "x").replace("X", "x"))
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
    def group_ordered_items_only(items: List[Dict[str, str]], keys: tuple[str, ...]) -> Dict[str, List[str]]:
        ordered_items: List[Dict[str, str]] = []
        for item in items:
            item_type = str(item.get("type", "")).strip().upper()
            item_value = str(item.get("value", "")).strip()
            if not item_type or not item_value or item_type not in keys:
                continue
            ordered_item = {"type": item_type, "value": item_value}
            item_role = str(item.get("role", "") or "").strip().upper()
            if item_role in StructuralFieldOutputNormalizer.ROLE_KEYS:
                ordered_item = {"role": item_role, **ordered_item}
            ordered_items.append(ordered_item)
        return {"_ITEMS": ordered_items} if ordered_items else {}
