#!/usr/bin/env python3
"""Build the first standalone original-grade dataset from six 304H forms."""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern


QWEN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "材质规范"
DEFAULT_TRAIN_SOURCE = SOURCE_DIR / "材质规范_train.json"
DEFAULT_VAL_SOURCE = SOURCE_DIR / "材质规范_val.json"
DEFAULT_OUTPUT_DIR = SOURCE_DIR / "原始牌号"
DEFAULT_TRAIN_OUTPUT = DEFAULT_OUTPUT_DIR / "材质规范_原始牌号_304H六牌号_train.json"
DEFAULT_VAL_OUTPUT = DEFAULT_OUTPUT_DIR / "材质规范_原始牌号_304H六牌号_val.json"


@dataclass(frozen=True)
class GradeRule:
    name: str
    value: str
    pattern: Pattern[str]


# Specific suffix forms must precede the generic WP304H rule.
GRADE_RULES = (
    GradeRule(
        "ASTM A403 WP304H-WX",
        "WP304H-WX",
        re.compile(r"\bWP304H\s*-\s*WX\b", re.IGNORECASE),
    ),
    GradeRule(
        "ASTM A403 WP304H-S",
        "WP304H-S",
        re.compile(r"\bWP304H\s*-\s*S\b", re.IGNORECASE),
    ),
    GradeRule(
        "ASTM A182 F304H",
        "F304H",
        re.compile(r"\bF304H\b", re.IGNORECASE),
    ),
    GradeRule(
        "ASTM A403 WP304H",
        "WP304H",
        re.compile(r"\bWP304H\b", re.IGNORECASE),
    ),
    GradeRule(
        "07Cr19Ni10",
        "07Cr19Ni10",
        re.compile(r"(?<![A-Za-z0-9])07Cr19Ni10(?![A-Za-z0-9])", re.IGNORECASE),
    ),
    GradeRule(
        "S30409",
        "S30409",
        re.compile(r"(?<![A-Za-z0-9])S30409(?![A-Za-z0-9])", re.IGNORECASE),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="转换首批六类 304H 原始牌号数据")
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    return parser.parse_args()


def normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def match_grade_rule(description: str) -> GradeRule | None:
    for rule in GRADE_RULES:
        if rule.pattern.search(description):
            return rule
    return None


def convert_row(row: dict[str, Any], rule: GradeRule) -> dict[str, Any]:
    converted = copy.deepcopy(row)
    output = converted.get("output")
    if not isinstance(output, dict):
        raise ValueError("缺少 output 对象")
    materials = output.get("MATERIAL")
    if not isinstance(materials, list) or len(materials) != 1:
        raise ValueError("首批转换仅接受单一 MATERIAL")
    material = materials[0]
    if not isinstance(material, dict):
        raise ValueError("MATERIAL 项结构错误")
    if material.get("ROLE") != "MAIN" or material.get("VALUE") != "304H":
        raise ValueError(
            f"首批源标签必须为 MAIN/304H，实际为 "
            f"{material.get('ROLE')}/{material.get('VALUE')}",
        )
    material["VALUE"] = rule.value
    return converted


def convert_dataset(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    source_rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError(f"数据集顶层必须为数组: {path}")

    converted_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_inputs: set[str] = set()
    for index, row in enumerate(source_rows):
        if not isinstance(row, dict):
            continue
        description = str(row.get("input") or "")
        rule = match_grade_rule(description)
        if rule is None:
            continue
        try:
            converted = convert_row(row, rule)
        except ValueError as exc:
            raise ValueError(f"{path.name} row={index}: {exc}") from exc

        input_key = normalized_text(description)
        if input_key in seen_inputs:
            raise ValueError(f"{path.name} 存在重复描述: {description}")
        seen_inputs.add(input_key)
        converted_rows.append(converted)
        counts[rule.value] += 1

    return converted_rows, counts


def validate_converted_rows(rows: list[dict[str, Any]]) -> None:
    allowed_values = {rule.value for rule in GRADE_RULES}
    for index, row in enumerate(rows):
        description = str(row.get("input") or "")
        rule = match_grade_rule(description)
        material = row["output"]["MATERIAL"][0]
        actual_value = material.get("VALUE")
        if rule is None:
            raise ValueError(f"row={index}: 转换结果不再命中任何首批规则")
        if actual_value != rule.value:
            raise ValueError(
                f"row={index}: expected={rule.value}, actual={actual_value}",
            )
        if actual_value not in allowed_values or material.get("ROLE") != "MAIN":
            raise ValueError(f"row={index}: MATERIAL 输出不合法")


def assert_split_isolation(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    train_inputs = {normalized_text(row["input"]) for row in train_rows}
    val_inputs = {normalized_text(row["input"]) for row in val_rows}
    if overlap := train_inputs & val_inputs:
        raise ValueError(f"训练集与验证集存在重复描述: {len(overlap)}")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    train_rows, train_counts = convert_dataset(args.train_source)
    val_rows, val_counts = convert_dataset(args.val_source)
    validate_converted_rows(train_rows)
    validate_converted_rows(val_rows)
    assert_split_isolation(train_rows, val_rows)

    write_json(args.train_output, train_rows)
    write_json(args.val_output, val_rows)
    print(
        json.dumps(
            {
                "train_source": str(args.train_source),
                "val_source": str(args.val_source),
                "train_output": str(args.train_output),
                "val_output": str(args.val_output),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_value_counts": dict(sorted(train_counts.items())),
                "val_value_counts": dict(sorted(val_counts.items())),
                "train_val_overlap": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
