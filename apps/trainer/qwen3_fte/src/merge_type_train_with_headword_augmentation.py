#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集"
DEFAULT_TRAIN = BASE / "种类_train.json"
DEFAULT_AUG = BASE / "管件_主词优先专项增强.json"
DEFAULT_OUTPUT = BASE / "种类_train_主词增强.json"
DEFAULT_OUTPUT_META = BASE / "种类_train_主词增强_带标识.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把主词增强样本并入种类训练集")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--aug", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-meta", type=Path, default=DEFAULT_OUTPUT_META)
    return parser.parse_args()


def strip_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"input": row["input"], "output": row["output"]} for row in rows]


def dedupe(base: list[dict[str, Any]], aug: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {row["input"].strip() for row in base}
    merged = list(base)
    for row in aug:
        key = row["input"].strip()
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def main() -> None:
    args = parse_args()
    train = json.loads(args.train.read_text(encoding="utf-8"))
    aug = json.loads(args.aug.read_text(encoding="utf-8"))
    merged_meta = dedupe(train, aug)
    merged = strip_meta(merged_meta)
    args.output_meta.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"原训练集: {len(train)}")
    print(f"增强样本: {len(aug)}")
    print(f"合并后: {len(merged)}")
    print(f"已生成: {args.output}")
    print(f"已生成: {args.output_meta}")


if __name__ == "__main__":
    main()
