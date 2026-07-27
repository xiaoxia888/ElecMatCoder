# -*- coding: utf-8 -*-
"""TYPE 结构字段的通用规范化。"""

from __future__ import annotations

import re
from typing import Any


_NUMERIC_RADIUS_RE = re.compile(r"^(\d+)(?:\.(\d+))?D$", re.IGNORECASE)


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
