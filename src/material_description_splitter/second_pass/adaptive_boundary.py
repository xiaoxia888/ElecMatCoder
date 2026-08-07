"""Adaptive token boundaries for structural evidence matching."""

from __future__ import annotations


def adaptive_guards(surface: str) -> tuple[str, str]:
    """Build guards from the first and last ASCII alphanumeric characters."""
    chars = [ch for ch in str(surface or "").upper() if ch.isascii() and ch.isalnum()]
    if not chars:
        return "", ""

    first, last = chars[0], chars[-1]
    left = r"(?<![A-Z])" if first.isalpha() else r"(?<!\d)"
    right = r"(?![A-Z])" if last.isalpha() else r"(?!\d)"
    return left, right
