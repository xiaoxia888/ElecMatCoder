# -*- coding: utf-8 -*-

"""
python scripts/查看一阶段结构完整性.py '90度法兰弯头, PTFElined GB/T 8163-20, RF, PN16, HG/T20538, SMLS , DN100, 4.0 mm'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.material_description_splitter.stage1_structure_checker import Stage1StructureChecker


def main() -> None:
    parser = argparse.ArgumentParser(description="查看一阶段结构完整性（当前只检查尺寸/壁厚/磅级）")
    parser.add_argument("text", nargs="?", help="待分析的材料描述")
    parser.add_argument("--text", dest="text_flag", help="待分析的材料描述")
    args = parser.parse_args()

    text = args.text_flag or args.text
    if not text:
        raise SystemExit("请提供待分析的描述文本。")

    checker = Stage1StructureChecker()
    result = checker.analyze(text)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
