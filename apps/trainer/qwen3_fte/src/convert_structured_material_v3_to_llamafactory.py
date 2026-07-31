#!/usr/bin/env python3
"""Convert structured material v3 datasets to LLaMAFactory Alpaca files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    QWEN_ROOT
    / "output"
    / "按8类拆分数据集"
    / "材质规范"
    / "结构化原始牌号"
    / "重新划分_v3"
)
DEFAULT_TRAIN_INPUT = SOURCE_DIR / "材质规范_结构化原始牌号_train.json"
DEFAULT_VAL_INPUT = SOURCE_DIR / "材质规范_结构化原始牌号_val.json"
DEFAULT_PROMPT = (
    QWEN_ROOT / "prompt" / "material_standard_extraction_sft_instruction_v3.txt"
)
DEFAULT_OUTPUT_DIR = (
    QWEN_ROOT
    / "output"
    / "按8类拆分llamafactory数据集"
    / "材质规范_v3"
)

TRAIN_DATASET_NAME = "材质规范_train"
VAL_DATASET_NAME = "材质规范_val"
TRAIN_FILE_NAME = f"{TRAIN_DATASET_NAME}.json"
VAL_FILE_NAME = f"{VAL_DATASET_NAME}.json"
REPORT_FILE_NAME = "材质规范_v3_转换报告.json"

OUTPUT_KEYS = {"MATERIAL", "STANDARD"}
MATERIAL_KEYS = {"PART", "VALUE", "SPECIAL_REQ"}
ALLOWED_PARTS = {"BODY", "LINING", "INNER_PIPE", "OUTER_PIPE", "FLANGE"}
ALLOWED_SPECIAL_REQS = {
    "NACE",
    "GALVANIZED",
    "ANTI-H2S",
    "ANTI-HIC",
    "ANTI-SCC",
    "CE",
    "3PE",
    "4PE",
    "PE",
    "EP",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是数组")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} 数组中存在非对象元素")
    return data


def validate_output(output: Any, *, source: Path, index: int) -> dict[str, Any]:
    location = f"{source} 第 {index + 1} 条"
    if not isinstance(output, dict) or set(output) != OUTPUT_KEYS:
        fields = sorted(output) if isinstance(output, dict) else type(output).__name__
        raise ValueError(f"{location} output 字段异常: {fields}")

    materials = output["MATERIAL"]
    if not isinstance(materials, list):
        raise ValueError(f"{location} MATERIAL 不是数组")
    for material_index, material in enumerate(materials):
        item_location = f"{location} MATERIAL[{material_index}]"
        if not isinstance(material, dict) or set(material) != MATERIAL_KEYS:
            raise ValueError(f"{item_location} 字段异常")
        if material["PART"] not in ALLOWED_PARTS:
            raise ValueError(f"{item_location} PART 无效: {material['PART']}")
        if not isinstance(material["VALUE"], str) or not material["VALUE"].strip():
            raise ValueError(f"{item_location} VALUE 为空或不是字符串")
        special_reqs = material["SPECIAL_REQ"]
        if not isinstance(special_reqs, list) or not all(
            isinstance(value, str) and value in ALLOWED_SPECIAL_REQS
            for value in special_reqs
        ):
            raise ValueError(f"{item_location} SPECIAL_REQ 无效: {special_reqs}")
        if len(special_reqs) != len(set(special_reqs)):
            raise ValueError(f"{item_location} SPECIAL_REQ 存在重复值")

    standards = output["STANDARD"]
    if not isinstance(standards, list):
        raise ValueError(f"{location} STANDARD 不是数组")
    standard_values: list[str] = []
    for standard_index, standard in enumerate(standards):
        item_location = f"{location} STANDARD[{standard_index}]"
        if (
            not isinstance(standard, dict)
            or set(standard) != {"BODY"}
            or not isinstance(standard["BODY"], str)
            or not standard["BODY"].strip()
        ):
            raise ValueError(f"{item_location} 结构无效")
        standard_values.append(standard["BODY"])
    if len(standard_values) != len(set(standard_values)):
        raise ValueError(f"{location} STANDARD 存在重复值")
    return output


def convert_rows(
    rows: list[dict[str, Any]], *, instruction: str, source: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    converted: list[dict[str, str]] = []
    part_counts: Counter[str] = Counter()
    value_counts: Counter[str] = Counter()
    special_req_counts: Counter[str] = Counter()
    standard_counts: Counter[str] = Counter()
    unique_inputs: set[str] = set()
    unique_outputs: set[str] = set()
    exact_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    input_lengths: list[int] = []
    output_lengths: list[int] = []

    for index, row in enumerate(rows):
        if set(row) != {"input", "output"}:
            raise ValueError(
                f"{source} 第 {index + 1} 条顶层字段异常: {sorted(row)}"
            )
        input_text = row["input"]
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"{source} 第 {index + 1} 条 input 为空或不是字符串")
        input_text = input_text.strip()
        output = validate_output(row["output"], source=source, index=index)
        output_text = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        converted.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )

        for material in output["MATERIAL"]:
            part_counts[material["PART"]] += 1
            value_counts[material["VALUE"]] += 1
            special_req_counts.update(material["SPECIAL_REQ"])
        standard_counts.update(item["BODY"] for item in output["STANDARD"])
        unique_inputs.add(input_text)
        unique_outputs.add(output_text)
        pair = (input_text, output_text)
        if pair in exact_pairs:
            duplicate_pairs += 1
        exact_pairs.add(pair)
        input_lengths.append(len(input_text))
        output_lengths.append(len(output_text))

    return converted, {
        "rows": len(converted),
        "unique_inputs": len(unique_inputs),
        "unique_outputs": len(unique_outputs),
        "duplicate_pairs": duplicate_pairs,
        "part_counts": dict(sorted(part_counts.items())),
        "unique_values": len(value_counts),
        "value_counts_top100": dict(value_counts.most_common(100)),
        "special_req_counts": dict(sorted(special_req_counts.items())),
        "unique_standards": len(standard_counts),
        "standard_counts_top100": dict(standard_counts.most_common(100)),
        "input_char_length": length_summary(input_lengths),
        "output_char_length": length_summary(output_lengths),
    }


def length_summary(lengths: list[int]) -> dict[str, float | int]:
    return {
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "average": round(sum(lengths) / len(lengths), 2) if lengths else 0,
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
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
    }
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
    parser = argparse.ArgumentParser(
        description="将材质规范 v3 数据转换为 LLaMAFactory Alpaca 格式"
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--val-input", type=Path, default=DEFAULT_VAL_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    train_input = args.train_input.expanduser().resolve()
    val_input = args.val_input.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

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

    write_json_atomic(train_output, train_converted)
    write_json_atomic(val_output, val_converted)
    write_json_atomic(dataset_info_output, build_dataset_info())

    train_pairs = {(row["input"], row["output"]) for row in train_converted}
    val_pairs = {(row["input"], row["output"]) for row in val_converted}
    train_inputs = {row["input"] for row in train_converted}
    val_inputs = {row["input"] for row in val_converted}
    report = {
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
        },
        "format": {
            "formatting": "alpaca",
            "output_is_compact_json_string": True,
        },
        "dataset_names": {
            "train": TRAIN_DATASET_NAME,
            "val": VAL_DATASET_NAME,
        },
        "statistics": {
            "train": train_stats,
            "val": val_stats,
            "exact_pair_overlap": len(train_pairs & val_pairs),
            "input_overlap": len(train_inputs & val_inputs),
            "instruction_chars": len(instruction),
        },
        "sha256": {
            "train": sha256_file(train_output),
            "val": sha256_file(val_output),
            "dataset_info": sha256_file(dataset_info_output),
        },
    }
    write_json_atomic(report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
