#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_JSON_PATH = Path(
    "/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/"
    "apps/trainer/qwen3_fte/output/按8类拆分数据集/encoding_mappings.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 Excel 中的“原始种类 / 标准化种类”，增量更新 encoding_mappings.json 的 TYPE.mappings。"
    )
    parser.add_argument("excel", help="Excel 文件路径")
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help="原始 encoding_mappings.json 路径",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出 JSON 路径，默认在原 JSON 同目录生成 encoding_mappings.updated.json",
    )
    parser.add_argument(
        "--sheet",
        default="",
        help="Excel sheet 名称，默认读取第一个 sheet",
    )
    parser.add_argument(
        "--raw-col",
        default="原始种类",
        help="原始种类列名",
    )
    parser.add_argument(
        "--normalized-col",
        default="标准化种类",
        help="标准化种类列名",
    )
    return parser


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def load_excel_rows(excel_path: Path, sheet: str, raw_col: str, normalized_col: str) -> list[tuple[str, str]]:
    if sheet:
        df = pd.read_excel(excel_path, sheet_name=sheet)
    else:
        df = pd.read_excel(excel_path)

    missing = [col for col in (raw_col, normalized_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少列: {missing}；实际列为: {list(df.columns)}")

    rows: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        raw_type = normalize_text(row.get(raw_col))
        normalized_type = normalize_text(row.get(normalized_col))
        if not raw_type or not normalized_type:
            continue
        rows.append((raw_type, normalized_type))
    return rows


def ensure_type_mappings(payload: dict[str, Any]) -> dict[str, list[str]]:
    type_block = payload.get("TYPE")
    if not isinstance(type_block, dict):
        type_block = {}
        payload["TYPE"] = type_block

    mappings = type_block.get("mappings")
    if not isinstance(mappings, dict):
        mappings = {}
        type_block["mappings"] = mappings

    normalized_mappings: dict[str, list[str]] = {}
    for key, values in mappings.items():
        normalized_key = normalize_text(key)
        if not normalized_key:
            continue
        if not isinstance(values, list):
            values = [values]
        seen: set[str] = set()
        merged_values: list[str] = []
        for value in values:
            normalized_value = normalize_text(value)
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            merged_values.append(normalized_value)
        normalized_mappings[normalized_key] = merged_values

    type_block["mappings"] = normalized_mappings
    return normalized_mappings


def merge_rows_into_mappings(
    rows: list[tuple[str, str]],
    mappings: dict[str, list[str]],
) -> dict[str, int]:
    skipped_same_pair = 0
    appended_raw_count = 0
    created_standard_count = 0

    for raw_type, normalized_type in rows:
        if raw_type == normalized_type:
            skipped_same_pair += 1
            continue

        existing = mappings.get(normalized_type)
        if existing is None:
            mappings[normalized_type] = [raw_type]
            created_standard_count += 1
            continue

        if raw_type not in existing:
            existing.append(raw_type)
            appended_raw_count += 1

    return {
        "skipped_same_pair": skipped_same_pair,
        "appended_raw_count": appended_raw_count,
        "created_standard_count": created_standard_count,
    }


def refresh_type_meta(payload: dict[str, Any], mappings: dict[str, list[str]]) -> None:
    type_block = payload.setdefault("TYPE", {})
    type_block["unique_outputs"] = len(mappings)
    type_block["unique_pairs"] = sum(len(values) for values in mappings.values())


def resolve_output_path(json_path: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    return json_path.with_name(f"{json_path.stem}.updated{json_path.suffix}")


def main() -> None:
    args = build_parser().parse_args()
    excel_path = Path(args.excel).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    output_path = resolve_output_path(json_path, args.output).expanduser().resolve()

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    rows = load_excel_rows(excel_path, args.sheet, args.raw_col, args.normalized_col)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须是对象")

    mappings = ensure_type_mappings(payload)
    stats = merge_rows_into_mappings(rows, mappings)
    refresh_type_meta(payload, mappings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "excel": str(excel_path),
                "json": str(json_path),
                "output": str(output_path),
                "excel_rows": len(rows),
                "type_unique_outputs": payload["TYPE"]["unique_outputs"],
                "type_unique_pairs": payload["TYPE"]["unique_pairs"],
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
