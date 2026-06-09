# -*- coding: utf-8 -*-
"""Shared difficulty level constants and compatibility helpers."""

from __future__ import annotations

from typing import Any

DIFF_HARD = 0
DIFF_EASY = 1
DIFF_SECOND_EASY = 2

DIFF_LABELS = {
    DIFF_HARD: "困难",
    DIFF_EASY: "简单",
    DIFF_SECOND_EASY: "二次简单",
}

_TEXT_TO_LEVEL = {
    "困难": DIFF_HARD,
    "简单": DIFF_EASY,
    "二次简单": DIFF_SECOND_EASY,
    "0": DIFF_HARD,
    "1": DIFF_EASY,
    "2": DIFF_SECOND_EASY,
}


def normalize_difficulty_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return DIFF_EASY if value else DIFF_HARD
    if isinstance(value, int):
        return value if value in DIFF_LABELS else None
    text = str(value).strip()
    if not text:
        return None
    if text in _TEXT_TO_LEVEL:
        return _TEXT_TO_LEVEL[text]
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed in DIFF_LABELS else None


def difficulty_label(value: Any) -> str:
    level = normalize_difficulty_level(value)
    return DIFF_LABELS.get(level, "")
