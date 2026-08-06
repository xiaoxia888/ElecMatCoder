#!/usr/bin/env python3
"""Convert pre-split type datasets to LLaMAFactory Alpaca files.

This converter never re-splits, deduplicates, reorders, normalizes, or drops rows.
It validates the six source files and writes both per-category and merged datasets.

Example:
python apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/convert_presplit_type_to_llamafactory.py \
    --input-dir apps/trainer/qwen3_fte/output/按8类拆分数据集/种类 \
    --prompt apps/trainer/qwen3_fte/prompt/type_extraction_sft_instruction_v3.txt \
    --output-dir apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_已划分

python apps/trainer/qwen3_fte/src/convert_presplit_type_to_llamafactory.py \
    --prompt apps/trainer/qwen3_fte/prompt/种类微调提示词.txt \
    --flange-train apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/法兰_train.json \
    --flange-val apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/法兰_val.json \
    --pipe-train apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/直管_train.json \
    --pipe-val apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/直管_val.json \
    --fitting-train apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/管件_train.json \
    --fitting-val apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/管件_val.json \
    --output-dir apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类/0806
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类"
DEFAULT_OUTPUT_DIR = (
    QWEN_ROOT / "output" / "按8类拆分llamafactory数据集" / "种类_已划分"
)

CATEGORIES = ("法兰", "直管", "管件")
SPLITS = ("train", "val")
DATASET_NAMES = {"train": "种类_train", "val": "种类_val"}
REPORT_FILE_NAME = "种类_已划分转换报告.json"

TYPE_KEYS = {
    "法兰": {"BODY", "CONN", "SEAL"},
    "直管": {"BODY", "FLANGE_STYLE", "MANU"},
    "管件": {"BODY", "GEOMETRY", "FLANGE_STYLE", "MANU", "CONN"},
}


def read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"文件顶层必须是对象数组: {path}")
    return value


def validate_string_list(value: Any, location: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{location} 必须是非空字符串数组")


def validate_output(output: Any, category: str, location: str) -> dict[str, Any]:
    if not isinstance(output, dict) or set(output) != {"CATEGORY", "TYPE"}:
        raise ValueError(f"{location} output 字段必须严格为 CATEGORY、TYPE")
    if output["CATEGORY"] != category:
        raise ValueError(
            f"{location} CATEGORY={output['CATEGORY']!r}，与来源分类 {category!r} 不一致"
        )
    type_value = output["TYPE"]
    if not isinstance(type_value, dict) or set(type_value) != TYPE_KEYS[category]:
        actual = sorted(type_value) if isinstance(type_value, dict) else type(type_value).__name__
        raise ValueError(f"{location} TYPE 字段异常: {actual}")
    if not isinstance(type_value["BODY"], str) or not type_value["BODY"].strip():
        raise ValueError(f"{location} TYPE.BODY 不能为空")

    if category == "法兰":
        validate_string_list(type_value["CONN"], f"{location} TYPE.CONN")
        validate_string_list(type_value["SEAL"], f"{location} TYPE.SEAL")
    elif category == "直管":
        if not isinstance(type_value["FLANGE_STYLE"], str):
            raise ValueError(f"{location} TYPE.FLANGE_STYLE 必须是字符串")
        validate_string_list(type_value["MANU"], f"{location} TYPE.MANU")
    else:
        geometry = type_value["GEOMETRY"]
        if not isinstance(geometry, dict) or set(geometry) != {"ANGLE", "RADIUS"}:
            raise ValueError(f"{location} TYPE.GEOMETRY 字段异常")
        if not all(isinstance(geometry[key], str) for key in ("ANGLE", "RADIUS")):
            raise ValueError(f"{location} TYPE.GEOMETRY 的值必须是字符串")
        if not isinstance(type_value["FLANGE_STYLE"], str):
            raise ValueError(f"{location} TYPE.FLANGE_STYLE 必须是字符串")
        validate_string_list(type_value["MANU"], f"{location} TYPE.MANU")
        validate_string_list(type_value["CONN"], f"{location} TYPE.CONN")
    return output


def convert_file(
    source: Path, category: str, instruction: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_json_array(source)
    converted: list[dict[str, str]] = []
    bodies: Counter[str] = Counter()
    exact_pairs: Counter[tuple[str, str]] = Counter()
    input_counts: Counter[str] = Counter()
    extra_field_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        location = f"{source} 第 {index + 1} 条"
        missing_fields = {"input", "output"} - set(row)
        if missing_fields:
            raise ValueError(f"{location} 缺少字段: {sorted(missing_fields)}")
        extra_field_counts.update(set(row) - {"input", "output"})
        input_text = row["input"]
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"{location} input 不能为空")
        output = validate_output(row["output"], category, location)
        output_text = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        converted.append(
            {"instruction": instruction, "input": input_text, "output": output_text}
        )
        bodies[output["TYPE"]["BODY"]] += 1
        exact_pairs[(input_text, output_text)] += 1
        input_counts[input_text] += 1

    return converted, {
        "source_rows": len(rows),
        "output_rows": len(converted),
        "rows_preserved": len(rows) == len(converted),
        "unique_inputs": len(input_counts),
        "duplicate_input_rows": sum(count - 1 for count in input_counts.values()),
        "exact_duplicate_rows": sum(count - 1 for count in exact_pairs.values()),
        "ignored_metadata_fields": dict(sorted(extra_field_counts.items())),
        "body_distribution": dict(bodies.most_common()),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_info() -> dict[str, Any]:
    columns = {"prompt": "instruction", "query": "input", "response": "output"}
    return {
        DATASET_NAMES[split]: {
            "file_name": f"{DATASET_NAMES[split]}.json",
            "formatting": "alpaca",
            "columns": columns,
        }
        for split in SPLITS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将已划分的法兰、直管、管件 train/val 原样转换为 LLaMAFactory 格式"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"已划分种类数据集目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"LLaMAFactory 输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help="instruction 提示词文件（必填）",
    )
    for category in CATEGORIES:
        option = {"法兰": "flange", "直管": "pipe", "管件": "fitting"}[category]
        for split in SPLITS:
            parser.add_argument(
                f"--{option}-{split}",
                type=Path,
                help=f"{category}_{split}.json；未指定时从 --input-dir 读取",
            )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
    instruction = prompt_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"提示词为空: {prompt_path}")

    source_paths: dict[tuple[str, str], Path] = {}
    for category in CATEGORIES:
        option = {"法兰": "flange", "直管": "pipe", "管件": "fitting"}[category]
        for split in SPLITS:
            supplied = getattr(args, f"{option}_{split}")
            source = supplied or input_dir / f"{category}_{split}.json"
            source = source.expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"{category} {split} 文件不存在: {source}")
            source_paths[(category, split)] = source

    if len(set(source_paths.values())) != len(source_paths):
        raise ValueError("六个 train/val 参数中存在重复的输入文件路径")
    planned_outputs = {
        output_dir / f"{category}_{split}.json"
        for category in CATEGORIES
        for split in SPLITS
    } | {
        output_dir / f"{DATASET_NAMES[split]}.json" for split in SPLITS
    } | {
        output_dir / "dataset_info.json",
        output_dir / REPORT_FILE_NAME,
    }
    collisions = set(source_paths.values()) & planned_outputs
    if collisions:
        raise ValueError(
            "输出路径会覆盖输入文件: "
            + ", ".join(str(path) for path in sorted(collisions))
        )

    converted_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    converted_files: dict[tuple[str, str], list[dict[str, str]]] = {}
    report_files: dict[str, Any] = {}
    for split in SPLITS:
        report_files[split] = {}
        for category in CATEGORIES:
            source = source_paths[(category, split)]
            converted, stats = convert_file(source, category, instruction)
            category_output = output_dir / f"{category}_{split}.json"
            converted_files[(category, split)] = converted
            converted_by_split[split].extend(converted)
            report_files[split][category] = {
                "source": str(source),
                "output": str(category_output),
                "statistics": stats,
            }

    # Only start writing after all six source files pass validation.
    output_paths: list[Path] = []
    for split in SPLITS:
        for category in CATEGORIES:
            category_output = output_dir / f"{category}_{split}.json"
            write_json_atomic(category_output, converted_files[(category, split)])
            output_paths.append(category_output)

    merged_outputs: dict[str, Path] = {}
    for split in SPLITS:
        merged_output = output_dir / f"{DATASET_NAMES[split]}.json"
        write_json_atomic(merged_output, converted_by_split[split])
        output_paths.append(merged_output)
        merged_outputs[split] = merged_output

    info_output = output_dir / "dataset_info.json"
    write_json_atomic(info_output, dataset_info())
    output_paths.append(info_output)

    train_inputs = {row["input"] for row in converted_by_split["train"]}
    val_inputs = {row["input"] for row in converted_by_split["val"]}
    train_pairs = {(row["input"], row["output"]) for row in converted_by_split["train"]}
    val_pairs = {(row["input"], row["output"]) for row in converted_by_split["val"]}
    report = {
        "mode": "pre_split_passthrough",
        "guarantees": {
            "resplit": False,
            "deduplicate": False,
            "reorder_within_source_file": False,
            "normalize_labels": False,
            "drop_rows": False,
            "merge_order": list(CATEGORIES),
        },
        "prompt": str(prompt_path),
        "instruction_chars": len(instruction),
        "output_directory": str(output_dir),
        "files": report_files,
        "merged": {
            split: {
                "output": str(merged_outputs[split]),
                "rows": len(converted_by_split[split]),
            }
            for split in SPLITS
        },
        "overlap_audit": {
            "input_overlap": len(train_inputs & val_inputs),
            "exact_pair_overlap": len(train_pairs & val_pairs),
            "overlap_is_report_only": True,
        },
        "sha256": {path.name: sha256_file(path) for path in output_paths},
        "source_sha256": {
            f"{category}_{split}": sha256_file(source_paths[(category, split)])
            for split in SPLITS
            for category in CATEGORIES
        },
    }
    report_output = output_dir / REPORT_FILE_NAME
    write_json_atomic(report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
