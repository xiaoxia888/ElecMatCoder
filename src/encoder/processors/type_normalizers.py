# -*- coding: utf-8 -*-
"""TYPE 结构字段的通用规范化。"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any


_NUMERIC_RADIUS_RE = re.compile(r"^(\d+)(?:\.(\d+))?D$", re.IGNORECASE)
_NUMERIC_ANGLE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def normalize_type_angle(value: Any) -> str:
    """二阶段角度最多保留两位小数，使用常规四舍五入。"""
    text = str(value or "").strip()
    if not text or not _NUMERIC_ANGLE_RE.fullmatch(text):
        return text

    try:
        rounded = Decimal(text).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return text

    normalized = format(rounded, "f").rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def normalize_type_radius(value: Any) -> str:
    """规范数字倍径，保留 LR、SR 等非数字半径原值。"""
    text = str(value or "").strip().upper()
    match = _NUMERIC_RADIUS_RE.fullmatch(text)
    if not match:
        return text

    integer_part = match.group(1)
    fractional_part = (match.group(2) or "").rstrip("0")
    if fractional_part:
        return f"{integer_part}.{fractional_part}D"
    return f"{integer_part}D"
