#!/usr/bin/env python3
"""Stable entry point for the pre-split type dataset converter."""

from __future__ import annotations

import runpy
from pathlib import Path


CONVERTER = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "按8类拆分数据集"
    / "种类"
    / "convert_presplit_type_to_llamafactory.py"
)


if __name__ == "__main__":
    if not CONVERTER.is_file():
        raise FileNotFoundError(f"种类转换脚本不存在: {CONVERTER}")
    runpy.run_path(str(CONVERTER), run_name="__main__")
