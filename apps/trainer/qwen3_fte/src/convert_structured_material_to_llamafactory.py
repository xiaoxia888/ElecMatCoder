#!/usr/bin/env python3
"""Convert structured original-grade datasets to LlamaFactory Alpaca files.

The existing train/validation split is preserved. Generated ``dataset_info.json``
allows the output directory to be passed directly as LlamaFactory's dataset_dir.

Example:
    python apps/trainer/qwen3_fte/src/convert_structured_material_to_llamafactory.py
"""

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
    / "重新划分_v2"
)
DEFAULT_TRAIN_INPUT = SOURCE_DIR / "材质规范_结构化原始牌号_train.json"
DEFAULT_VAL_INPUT = SOURCE_DIR / "材质规范_结构化原始牌号_val.json"
DEFAULT_PROMPT = (
    QWEN_ROOT / "prompt" / "material_standard_extraction_sft_instruction_v2.txt"
)
DEFAULT_OUTPUT_DIR = (
    QWEN_ROOT
    / "output"
    / "按8类拆分llamafactory数据集"
    / "材质规范_结构化原始牌号"
)

TRAIN_DATASET_NAME = "材质规范_结构化原始牌号_train"
VAL_DATASET_NAME = "材质规范_结构化原始牌号_val"
TRAIN_FILE_NAME = f"{TRAIN_DATASET_NAME}.json"
VAL_FILE_NAME = f"{VAL_DATASET_NAME}.json"
REPORT_FILE_NAME = "材质规范_结构化原始牌号_转换报告.json"

ALLOWED_PARTS = {"BODY", "LINING", "INNER_PIPE", "OUTER_PIPE", "FLANGE"}
ALLOWED_RELATIONS = {
    "SINGLE",
    "COMPOSITE",
    "DUAL_CERTIFIED",
    "EQUIVALENT",
    "ALTERNATIVE",
}
MATERIAL_KEYS = {"PART", "STANDARD", "GRADE", "CLASS", "SPECIAL_REQ"}
OUTPUT_KEYS = {"MATERIAL", "STANDARD", "MATERIAL_RELATION"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是数组")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} 数组中存在非对象元素")
    return data


def validate_output(output: Any, *, source: Path, index: int) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError(f"{source} 第 {index + 1} 条 output 不是对象")
    if set(output) != OUTPUT_KEYS:
        raise ValueError(
            f"{source} 第 {index + 1} 条 output 字段异常: {sorted(output)}"
        )

    materials = output.get("MATERIAL")
    if not isinstance(materials, list):
        raise ValueError(f"{source} 第 {index + 1} 条 MATERIAL 不是数组")
    for material_index, material in enumerate(materials):
        if not isinstance(material, dict) or set(material) != MATERIAL_KEYS:
            raise ValueError(
                f"{source} 第 {index + 1} 条 MATERIAL[{material_index}] 字段异常"
            )
        if material["PART"] not in ALLOWED_PARTS:
            raise ValueError(
                f"{source} 第 {index + 1} 条 PART 无效: {material['PART']}"
            )
        if not isinstance(material["SPECIAL_REQ"], list):
            raise ValueError(
                f"{source} 第 {index + 1} 条 SPECIAL_REQ 不是数组"
            )

    standards = output.get("STANDARD")
    if not isinstance(standards, list) or not all(
        isinstance(item, dict)
        and set(item) == {"BODY"}
        and isinstance(item["BODY"], str)
        for item in standards
    ):
        raise ValueError(f"{source} 第 {index + 1} 条 STANDARD 结构无效")

    relation = output.get("MATERIAL_RELATION")
    if relation not in ALLOWED_RELATIONS:
        raise ValueError(
            f"{source} 第 {index + 1} 条 MATERIAL_RELATION 无效: {relation}"
        )
    return output


def convert_rows(
    rows: list[dict[str, Any]], *, instruction: str, source: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    converted: list[dict[str, str]] = []
    relation_counts: Counter[str] = Counter()
    part_counts: Counter[str] = Counter()
    unique_inputs: set[str] = set()
    unique_outputs: set[str] = set()
    exact_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    input_lengths: list[int] = []
    output_lengths: list[int] = []

    for index, row in enumerate(rows):
        if "input" not in row or "output" not in row:
            raise ValueError(f"{source} 第 {index + 1} 条缺少 input 或 output")
        input_text = str(row["input"]).strip()
        if not input_text:
            raise ValueError(f"{source} 第 {index + 1} 条 input 为空")
        output = validate_output(row["output"], source=source, index=index)
        output_text = json.dumps(
            output,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        converted.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )

        relation_counts[output["MATERIAL_RELATION"]] += 1
        part_counts.update(
            material["PART"] for material in output.get("MATERIAL", [])
        )
        unique_inputs.add(input_text)
        unique_outputs.add(output_text)
        pair = (input_text, output_text)
        if pair in exact_pairs:
            duplicate_pairs += 1
        exact_pairs.add(pair)
        input_lengths.append(len(input_text))
        output_lengths.append(len(output_text))

    statistics = {
        "rows": len(converted),
        "unique_inputs": len(unique_inputs),
        "unique_outputs": len(unique_outputs),
        "duplicate_pairs": duplicate_pairs,
        "relation_counts": dict(sorted(relation_counts.items())),
        "part_counts": dict(sorted(part_counts.items())),
        "input_char_length": {
            "min": min(input_lengths, default=0),
            "max": max(input_lengths, default=0),
            "average": round(sum(input_lengths) / len(input_lengths), 2)
            if input_lengths
            else 0,
        },
        "output_char_length": {
            "min": min(output_lengths, default=0),
            "max": max(output_lengths, default=0),
            "average": round(sum(output_lengths) / len(output_lengths), 2)
            if output_lengths
            else 0,
        },
    }
    return converted, statistics


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


def dataset_info() -> dict[str, Any]:
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
        description="将结构化原始牌号训练集转换为LlamaFactory Alpaca格式"
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
    write_json_atomic(dataset_info_output, dataset_info())

    train_pairs = {
        (row["input"], row["output"]) for row in train_converted
    }
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
            "columns": {
                "instruction": "instruction",
                "input": "input",
                "output": "output",
            },
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
