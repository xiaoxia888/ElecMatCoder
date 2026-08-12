#!/usr/bin/env python3
"""Convert pre-split size V2 datasets to LLaMAFactory Alpaca files.

The converter validates and converts only. It never re-splits, deduplicates,
reorders, normalizes, or drops source rows.

Example:
python apps/trainer/qwen3_fte/src/convert_presplit_size_v2_to_llamafactory.py \
  --train-input apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2已划分/尺寸壁厚磅级V2_train.json \
  --val-input apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2已划分/尺寸壁厚磅级V2_val.json \
  --prompt apps/trainer/qwen3_fte/prompt/尺寸壁厚磅级提示词.txt \
  --output-dir apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/尺寸壁厚磅级/V2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TRAIN_DATASET_NAME = "train"
VAL_DATASET_NAME = "val"
TRAIN_FILE_NAME = f"{TRAIN_DATASET_NAME}.json"
VAL_FILE_NAME = f"{VAL_DATASET_NAME}.json"
REPORT_FILE_NAME = "尺寸壁厚磅级V2_转换报告.json"

OUTPUT_KEYS = {"ITEMS", "LENGTH", "PRESSURE"}
ITEM_KEYS = {"SCOPE", "ROLE", "SIZE", "THICKNESS"}
VALUE_KEYS = {"type", "value"}
ALLOWED_SCOPES = {"BODY", "INNER", "OUTER", "LINING"}
ALLOWED_ROLES = {"SINGLE", "MAIN", "BRANCH", "END_A", "END_B"}
ALLOWED_SIZE_TYPES = {"DN", "OD", "INCH"}
ALLOWED_THICKNESS_TYPES = {"MM", "SCHEDULE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将已划分的尺寸壁厚磅级 V2 数据转换为 LLaMAFactory Alpaca 格式"
    )
    parser.add_argument("--train-input", type=Path, required=True, help="已划分训练集 JSON")
    parser.add_argument("--val-input", type=Path, required=True, help="已划分验证集 JSON")
    parser.add_argument("--prompt", type=Path, required=True, help="instruction 提示词文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出文件夹")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {path}: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} 顶层必须是对象数组")
    return data


def validate_value_items(
    value: Any,
    *,
    allowed_types: set[str],
    location: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{location} 必须是数组")
    validated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(value):
        entry_location = f"{location}[{index}]"
        if not isinstance(entry, dict) or set(entry) != VALUE_KEYS:
            fields = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
            raise ValueError(f"{entry_location} 字段异常: {fields}")
        value_type = entry["type"]
        item_value = entry["value"]
        if value_type not in allowed_types:
            raise ValueError(f"{entry_location} type 无效: {value_type!r}")
        if not isinstance(item_value, str) or not item_value.strip():
            raise ValueError(f"{entry_location} value 为空或不是字符串")
        pair = (value_type, item_value)
        if pair in seen:
            raise ValueError(f"{location} 存在重复项: {pair}")
        seen.add(pair)
        validated.append(entry)
    return validated


def validate_output(output: Any, *, source: Path, row_index: int) -> dict[str, Any]:
    location = f"{source} 第 {row_index + 1} 条 output"
    if not isinstance(output, dict) or set(output) != OUTPUT_KEYS:
        fields = sorted(output) if isinstance(output, dict) else type(output).__name__
        raise ValueError(f"{location} 顶层字段异常: {fields}")
    if not isinstance(output["LENGTH"], str):
        raise ValueError(f"{location}.LENGTH 必须是字符串")
    if not isinstance(output["PRESSURE"], str):
        raise ValueError(f"{location}.PRESSURE 必须是字符串")

    items = output["ITEMS"]
    if not isinstance(items, list):
        raise ValueError(f"{location}.ITEMS 必须是数组")
    positions: set[tuple[str, str]] = set()
    for item_index, item in enumerate(items):
        item_location = f"{location}.ITEMS[{item_index}]"
        if not isinstance(item, dict) or set(item) != ITEM_KEYS:
            fields = sorted(item) if isinstance(item, dict) else type(item).__name__
            raise ValueError(f"{item_location} 字段异常: {fields}")
        scope = item["SCOPE"]
        role = item["ROLE"]
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"{item_location}.SCOPE 无效: {scope!r}")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{item_location}.ROLE 无效: {role!r}")
        position = (scope, role)
        if position in positions:
            raise ValueError(f"{location}.ITEMS 存在重复位置: {scope}+{role}")
        positions.add(position)

        sizes = validate_value_items(
            item["SIZE"],
            allowed_types=ALLOWED_SIZE_TYPES,
            location=f"{item_location}.SIZE",
        )
        thicknesses = validate_value_items(
            item["THICKNESS"],
            allowed_types=ALLOWED_THICKNESS_TYPES,
            location=f"{item_location}.THICKNESS",
        )
        if not sizes and not thicknesses:
            raise ValueError(f"{item_location} 的 SIZE 和 THICKNESS 不能同时为空")
    return output


def length_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "max": max(values, default=0),
        "average": round(sum(values) / len(values), 2) if values else 0,
    }


def convert_rows(
    rows: list[dict[str, Any]], *, instruction: str, source: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    converted: list[dict[str, str]] = []
    scope_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    size_type_counts: Counter[str] = Counter()
    thickness_type_counts: Counter[str] = Counter()
    length_present = 0
    pressure_present = 0
    input_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    input_lengths: list[int] = []
    output_lengths: list[int] = []

    for row_index, row in enumerate(rows):
        location = f"{source} 第 {row_index + 1} 条"
        if set(row) != {"input", "output"}:
            raise ValueError(f"{location} 顶层字段异常: {sorted(row)}")
        input_text = row["input"]
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"{location}.input 为空或不是字符串")
        output = validate_output(row["output"], source=source, row_index=row_index)
        output_text = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        converted.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )

        for item in output["ITEMS"]:
            scope_counts[item["SCOPE"]] += 1
            role_counts[item["ROLE"]] += 1
            size_type_counts.update(entry["type"] for entry in item["SIZE"])
            thickness_type_counts.update(entry["type"] for entry in item["THICKNESS"])
        length_present += bool(output["LENGTH"])
        pressure_present += bool(output["PRESSURE"])
        input_counts[input_text] += 1
        pair_counts[(input_text, output_text)] += 1
        input_lengths.append(len(input_text))
        output_lengths.append(len(output_text))

    return converted, {
        "source_rows": len(rows),
        "output_rows": len(converted),
        "rows_preserved": len(rows) == len(converted),
        "unique_inputs": len(input_counts),
        "duplicate_input_rows": sum(count - 1 for count in input_counts.values()),
        "exact_duplicate_rows": sum(count - 1 for count in pair_counts.values()),
        "scope_counts": dict(sorted(scope_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "size_type_counts": dict(sorted(size_type_counts.items())),
        "thickness_type_counts": dict(sorted(thickness_type_counts.items())),
        "length_present_rows": length_present,
        "pressure_present_rows": pressure_present,
        "input_char_length": length_summary(input_lengths),
        "output_char_length": length_summary(output_lengths),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_info() -> dict[str, Any]:
    columns = {"prompt": "instruction", "query": "input", "response": "output"}
    return {
        TRAIN_DATASET_NAME: {
            "file_name": TRAIN_FILE_NAME,
            "formatting": "alpaca",
            "columns": columns,
        },
        VAL_DATASET_NAME: {
            "file_name": VAL_FILE_NAME,
            "formatting": "alpaca",
            "columns": columns,
        },
    }


def main() -> int:
    args = parse_args()
    train_input = args.train_input.expanduser().resolve()
    val_input = args.val_input.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for label, path in (
        ("训练集", train_input),
        ("验证集", val_input),
        ("提示词", prompt_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}文件不存在: {path}")
    if train_input == val_input:
        raise ValueError("训练集和验证集不能使用同一个文件")

    instruction = prompt_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"提示词为空: {prompt_path}")

    train_rows = load_rows(train_input)
    val_rows = load_rows(val_input)
    train_converted, train_stats = convert_rows(
        train_rows, instruction=instruction, source=train_input
    )
    val_converted, val_stats = convert_rows(
        val_rows, instruction=instruction, source=val_input
    )

    train_output = output_dir / TRAIN_FILE_NAME
    val_output = output_dir / VAL_FILE_NAME
    dataset_info_output = output_dir / "dataset_info.json"
    report_output = output_dir / REPORT_FILE_NAME
    source_paths = {train_input, val_input, prompt_path}
    output_paths = {train_output, val_output, dataset_info_output, report_output}
    collisions = source_paths & output_paths
    if collisions:
        raise ValueError(
            "输出路径会覆盖输入文件: "
            + ", ".join(str(path) for path in sorted(collisions))
        )

    # Start writing only after both source files pass complete validation.
    write_json_atomic(train_output, train_converted)
    write_json_atomic(val_output, val_converted)
    write_json_atomic(dataset_info_output, build_dataset_info())

    train_inputs = {row["input"] for row in train_converted}
    val_inputs = {row["input"] for row in val_converted}
    train_pairs = {(row["input"], row["output"]) for row in train_converted}
    val_pairs = {(row["input"], row["output"]) for row in val_converted}
    report = {
        "mode": "pre_split_passthrough",
        "guarantees": {
            "resplit": False,
            "deduplicate": False,
            "reorder": False,
            "normalize_labels": False,
            "drop_rows": False,
        },
        "source": {
            "train": str(train_input),
            "val": str(val_input),
            "prompt": str(prompt_path),
        },
        "output": {
            "directory": str(output_dir),
            "train": str(train_output),
            "val": str(val_output),
            "dataset_info": str(dataset_info_output),
            "report": str(report_output),
        },
        "dataset_names": {
            "train": TRAIN_DATASET_NAME,
            "val": VAL_DATASET_NAME,
        },
        "statistics": {
            "train": train_stats,
            "val": val_stats,
            "instruction_chars": len(instruction),
            "input_overlap": len(train_inputs & val_inputs),
            "exact_pair_overlap": len(train_pairs & val_pairs),
        },
        "source_sha256": {
            "train": sha256_file(train_input),
            "val": sha256_file(val_input),
            "prompt": sha256_file(prompt_path),
        },
        "sha256": {
            "train": sha256_file(train_output),
            "val": sha256_file(val_output),
            "dataset_info": sha256_file(dataset_info_output),
        },
    }
    write_json_atomic(report_output, report)

    print(f"训练集: {len(train_converted)} -> {train_output}")
    print(f"验证集: {len(val_converted)} -> {val_output}")
    print(f"提示词字符数: {len(instruction)}")
    print(f"输入交叉: {report['statistics']['input_overlap']}")
    print(f"精确样本交叉: {report['statistics']['exact_pair_overlap']}")
    print(f"转换报告: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
