#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/将编码映射JSON转换为单字段训练集.py \
  --field TYPE \
  --output /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/按8类拆分数据集/type_single_field_train.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_JSON_PATH = Path(
    # "/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/"
    "/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/按8类拆分数据集/encoding_mappings.updated.json"
)

FIELD_INSTRUCTIONS = {
    "TYPE": "你是工业管道材料种类编码助手。请根据输入的原始种类描述，输出唯一的标准化种类编码。只输出编码，不要解释。",
    "SIZE": "你是工业管道材料尺寸编码助手。请根据输入的原始尺寸描述，输出唯一的标准化尺寸编码。只输出编码，不要解释。",
    "THICKNESS": "你是工业管道材料壁厚编码助手。请根据输入的原始壁厚描述，输出唯一的标准化壁厚编码。只输出编码，不要解释。",
    "PRESSURE": "你是工业管道材料磅级编码助手。请根据输入的原始磅级描述，输出唯一的标准化磅级编码。只输出编码，不要解释。",
    "MATERIAL": "你是工业管道材料材质编码助手。请根据输入的原始材质描述，输出唯一的标准化材质编码。只输出编码，不要解释。",
    "STANDARD": "你是工业管道材料规范编码助手。请根据输入的原始规范描述，输出唯一的标准化规范编码。只输出编码，不要解释。",
}
ALL_FIELDS = ["TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD"]
ALL_FIELDS_INSTRUCTION = "你是工业管道材料字段编码助手。请根据字段类型和原始字段值，输出唯一的标准化编码。只输出编码，不要解释。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 encoding_mappings.json 转换为单字段归一化训练集。"
    )
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help="encoding_mappings.json 路径",
    )
    parser.add_argument(
        "--field",
        default="TYPE",
        choices=sorted(FIELD_INSTRUCTIONS.keys()) + ["ALL"],
        help="要导出的字段，默认 TYPE；ALL 表示合并六个字段为一个统一训练集",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出训练集 JSON 路径，默认在原文件同目录生成 <FIELD>_single_field_train.json",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="自定义 instruction，不传则使用字段默认 instruction",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="是否打乱样本顺序",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="打乱时使用的随机种子，默认 42",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="输出字段格式。text=直接输出编码；json=输出 {FIELD_CODE: 编码} JSON 字符串。",
    )
    return parser


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def resolve_output_path(json_path: Path, field: str, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    if field == "ALL":
        return json_path.with_name("all_fields_single_field_train.json")
    return json_path.with_name(f"{field.lower()}_single_field_train.json")


def load_mappings(json_path: Path, field: str) -> dict[str, list[str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须是对象")

    field_block = payload.get(field)
    if not isinstance(field_block, dict):
        raise ValueError(f"JSON 中不存在字段块: {field}")

    mappings = field_block.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(f"{field}.mappings 不存在或格式错误")

    normalized: dict[str, list[str]] = {}
    for normalized_code, raw_values in mappings.items():
        code = normalize_text(normalized_code)
        if not code:
            continue
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        seen: set[str] = set()
        cleaned_values: list[str] = []
        for raw_value in raw_values:
            raw_text = normalize_text(raw_value)
            if not raw_text or raw_text in seen:
                continue
            seen.add(raw_text)
            cleaned_values.append(raw_text)
        if cleaned_values:
            normalized[code] = cleaned_values
    return normalized


def build_output_text(field: str, normalized_code: str, output_format: str) -> str:
    if output_format == "text":
        return normalized_code
    return json.dumps({f"{field}_CODE": normalized_code}, ensure_ascii=False)


def build_samples(
    mappings: dict[str, list[str]],
    *,
    field: str,
    instruction: str,
    output_format: str,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for normalized_code, raw_values in mappings.items():
        output_text = build_output_text(field, normalized_code, output_format)
        for raw_value in raw_values:
            key = (raw_value, output_text)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            samples.append(
                {
                    "instruction": instruction,
                    "input": raw_value,
                    "output": output_text,
                }
            )
    return samples


def build_multi_field_samples(
    payload: dict[str, Any],
    *,
    instruction: str,
    output_format: str,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for field in ALL_FIELDS:
        mappings = load_mappings_from_payload(payload, field)
        for normalized_code, raw_values in mappings.items():
            output_text = build_output_text(field, normalized_code, output_format)
            for raw_value in raw_values:
                input_text = f"字段类型: {field}\n原始值: {raw_value}"
                key = (input_text, output_text)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                samples.append(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_text,
                    }
                )
    return samples


def load_mappings_from_payload(payload: dict[str, Any], field: str) -> dict[str, list[str]]:
    field_block = payload.get(field)
    if not isinstance(field_block, dict):
        return {}

    mappings = field_block.get("mappings")
    if not isinstance(mappings, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for normalized_code, raw_values in mappings.items():
        code = normalize_text(normalized_code)
        if not code:
            continue
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        seen: set[str] = set()
        cleaned_values: list[str] = []
        for raw_value in raw_values:
            raw_text = normalize_text(raw_value)
            if not raw_text or raw_text in seen:
                continue
            seen.add(raw_text)
            cleaned_values.append(raw_text)
        if cleaned_values:
            normalized[code] = cleaned_values
    return normalized


def main() -> None:
    args = build_parser().parse_args()
    json_path = Path(args.json).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    field = str(args.field or "").strip().upper()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须是对象")

    if field == "ALL":
        instruction = normalize_text(args.instruction) or ALL_FIELDS_INSTRUCTION
        samples = build_multi_field_samples(
            payload,
            instruction=instruction,
            output_format=args.output_format,
        )
        normalized_code_count = sum(len(load_mappings_from_payload(payload, name)) for name in ALL_FIELDS)
    else:
        instruction = normalize_text(args.instruction) or FIELD_INSTRUCTIONS[field]
        mappings = load_mappings_from_payload(payload, field)
        if not mappings:
            raise ValueError(f"JSON 中不存在字段块或 mappings 为空: {field}")
        samples = build_samples(
            mappings,
            field=field,
            instruction=instruction,
            output_format=args.output_format,
        )
        normalized_code_count = len(mappings)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(samples)

    output_path = resolve_output_path(json_path, field, args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "field": field,
                "json": str(json_path),
                "output": str(output_path),
                "normalized_code_count": normalized_code_count,
                "sample_count": len(samples),
                "output_format": args.output_format,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
