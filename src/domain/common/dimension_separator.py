from __future__ import annotations

import re


# 工程描述中常见的尺寸乘号及 OCR/全角变体。
MULTIPLICATION_SEPARATOR_CHARS = "xX×*＊∗﹡✕✖⨉"
MULTIPLICATION_SEPARATOR_PATTERN = f"[{re.escape(MULTIPLICATION_SEPARATOR_CHARS)}]"
MULTIPLICATION_GLYPH_CHARS = "×*＊∗﹡✕✖⨉"
MULTIPLICATION_GLYPH_PATTERN = f"[{re.escape(MULTIPLICATION_GLYPH_CHARS)}]"
_MULTIPLICATION_SEPARATOR_RE = re.compile(MULTIPLICATION_SEPARATOR_PATTERN)
_MULTIPLICATION_GLYPH_RE = re.compile(MULTIPLICATION_GLYPH_PATTERN)
_MULTIPLICATION_SEPARATOR_FULL_RE = re.compile(
    rf"^{MULTIPLICATION_SEPARATOR_PATTERN}$"
)


def is_multiplication_separator(value: object) -> bool:
    """判断单个字符是否为受支持的尺寸乘号。"""
    return bool(_MULTIPLICATION_SEPARATOR_FULL_RE.fullmatch(str(value or "")))


def normalize_multiplication_separators(
    text: object,
    replacement: str = "×",
) -> str:
    """将尺寸乘号及其常见变体统一为指定字符。"""
    return _MULTIPLICATION_SEPARATOR_RE.sub(replacement, str(text or ""))


def normalize_multiplication_glyphs(
    text: object,
    replacement: str = "×",
) -> str:
    """归一乘号和星号字形，但保留普通英文 ``x/X``。"""
    return _MULTIPLICATION_GLYPH_RE.sub(replacement, str(text or ""))


def split_by_multiplication_separator(text: object) -> list[str]:
    """按尺寸乘号拆分文本并移除空片段。"""
    return [
        part.strip()
        for part in _MULTIPLICATION_SEPARATOR_RE.split(str(text or ""))
        if part.strip()
    ]
