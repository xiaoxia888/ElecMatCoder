#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/统计尺寸壁厚磅级训练集分布.py \
  --input-json /Users/guoxi/Documents/尺寸壁厚磅级C1训练集.json \
  --output /Users/guoxi/Documents/尺寸壁厚磅级C1训练集统计.xlsx
  
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    return data


def build_size_key(item: dict[str, Any]) -> str:
    item_type = clean_text(item.get("type")).upper()
    value = clean_text(item.get("value"))
    return f"{item_type}{value}" if item_type and value else ""


def build_thickness_key(item: dict[str, Any]) -> str:
    item_type = clean_text(item.get("type")).upper()
    value = clean_text(item.get("value"))
    return f"{item_type}{value}" if item_type and value else ""


def build_counter_df(counter: Counter[str]) -> pd.DataFrame:
    rows = [{"项": value, "频次": count} for value, count in counter.most_common()]
    return pd.DataFrame(rows, columns=["项", "频次"])


def build_item_counter_df(counter: Counter[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {"type": item_type, "value": value, "频次": count}
        for (item_type, value), count in counter.most_common()
    ]
    return pd.DataFrame(rows, columns=["type", "value", "频次"])


def analyze_dataset(data: list[dict[str, Any]]) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]], Counter[str]]:
    size_counter: Counter[tuple[str, str]] = Counter()
    thickness_counter: Counter[tuple[str, str]] = Counter()
    pressure_counter: Counter[str] = Counter()

    for row in data:
        output = row.get("output") or {}
        if not isinstance(output, dict):
            continue

        size_items = output.get("SIZE_ITEMS") or []
        if isinstance(size_items, list):
            for item in size_items:
                if not isinstance(item, dict):
                    continue
                item_type = clean_text(item.get("type")).upper()
                value = clean_text(item.get("value"))
                if item_type and value:
                    size_counter[(item_type, value)] += 1

        thickness_items = output.get("THICKNESS_ITEMS") or []
        if isinstance(thickness_items, list):
            for item in thickness_items:
                if not isinstance(item, dict):
                    continue
                item_type = clean_text(item.get("type")).upper()
                value = clean_text(item.get("value"))
                if item_type and value:
                    thickness_counter[(item_type, value)] += 1

        pressure = clean_text(output.get("PRESSURE"))
        if pressure:
            pressure_counter[pressure] += 1

    return size_counter, thickness_counter, pressure_counter


def build_output_path(input_path: Path, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        return output_path if output_path.suffix.lower() == ".xlsx" else output_path.with_suffix(".xlsx")
    return input_path.with_name(f"{input_path.stem}_统计.xlsx")


def export_excel(
    output_path: Path,
    size_counter: Counter[tuple[str, str]],
    thickness_counter: Counter[tuple[str, str]],
    pressure_counter: Counter[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        build_item_counter_df(size_counter).to_excel(writer, sheet_name="尺寸", index=False)
        build_item_counter_df(thickness_counter).to_excel(writer, sheet_name="壁厚", index=False)
        build_counter_df(pressure_counter).to_excel(writer, sheet_name="磅级", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计尺寸/壁厚/磅级训练集分布并导出 Excel")
    parser.add_argument("--input-json", required=True, help="输入训练集 JSON 路径")
    parser.add_argument("--output", help="输出 Excel 路径，默认与输入同目录自动生成")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = build_output_path(input_path, args.output)

    data = load_dataset(input_path)
    size_counter, thickness_counter, pressure_counter = analyze_dataset(data)
    export_excel(output_path, size_counter, thickness_counter, pressure_counter)

    print(f"已生成: {output_path}")
    print(f"尺寸项数: {sum(size_counter.values())}，去重后: {len(size_counter)}")
    print(f"壁厚项数: {sum(thickness_counter.values())}，去重后: {len(thickness_counter)}")
    print(f"磅级项数: {sum(pressure_counter.values())}，去重后: {len(pressure_counter)}")


if __name__ == "__main__":
    main()
