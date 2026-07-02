#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集"
DEFAULT_INPUT = BASE / "管件_主词优先专项增强.json"
DEFAULT_OUTPUT = BASE / "管件_主体纠偏补充训练集.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出管件主体纠偏补充训练集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def to_clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        input_text = (row.get("input") or "").strip()
        if not input_text or input_text in seen:
            continue
        seen.add(input_text)
        clean_rows.append(
            {
                "input": row["input"],
                "output": row["output"],
            }
        )
    return clean_rows


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    clean_rows = to_clean_rows(rows)
    args.output.write_text(
        json.dumps(clean_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成: {args.output}")
    print(f"样本数: {len(clean_rows)}")


if __name__ == "__main__":
    main()
