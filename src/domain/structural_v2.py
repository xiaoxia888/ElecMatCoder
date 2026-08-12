from __future__ import annotations

import copy
from typing import Any, Dict, List


STRUCTURAL_V2_FIELD = "STRUCTURAL"


def is_structural_v2(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("_schema_version") == "v2"
        and isinstance(value.get("ITEMS"), list)
    )


def has_v2_size(value: Any) -> bool:
    if not is_structural_v2(value):
        return False
    if str(value.get("LENGTH") or "").strip():
        return True
    return any(isinstance(item, dict) and bool(item.get("SIZE")) for item in value["ITEMS"])


def has_v2_thickness(value: Any) -> bool:
    if not is_structural_v2(value):
        return False
    return any(isinstance(item, dict) and bool(item.get("THICKNESS")) for item in value["ITEMS"])


def has_v2_pressure(value: Any) -> bool:
    return is_structural_v2(value) and bool(str(value.get("PRESSURE") or "").strip())


def canonical_structural_v2(value: Any) -> Dict[str, Any]:
    if not is_structural_v2(value):
        return {}
    return {
        "_schema_version": "v2",
        "ITEMS": copy.deepcopy(value.get("ITEMS") or []),
        "LENGTH": str(value.get("LENGTH") or "").strip(),
        "PRESSURE": str(value.get("PRESSURE") or "").strip(),
    }


def ordered_v2_items(value: Any) -> List[Dict[str, Any]]:
    """Return positional items in encoding order while retaining SCOPE metadata."""
    if not is_structural_v2(value):
        return []
    role_order = {
        "SINGLE": 0,
        "MAIN": 10,
        "END_A": 10,
        "BRANCH": 20,
        "END_B": 20,
    }
    indexed = [
        (index, item)
        for index, item in enumerate(value["ITEMS"])
        if isinstance(item, dict)
    ]
    indexed.sort(key=lambda pair: (role_order.get(str(pair[1].get("ROLE") or ""), 99), pair[0]))
    return [item for _, item in indexed]
