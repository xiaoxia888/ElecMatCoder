#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成尺寸、壁厚、磅级训练集。

输入：
- Excel/CSV 中的原始描述列
- SIZE_原始结果
- THICKNESS_原始结果
- PRESSURE_原始结果

输出：
- JSON 数组
- 每条数据格式：
  {
    "input": "原始描述",
    "output": {
      "SIZE_ITEMS": [...],
      "LENGTH": "",
      "THICKNESS_ITEMS": [...],
      "PRESSURE": ""
    }
  }

示例：
python scripts/生成尺寸壁厚磅级编码训练集.py \
  --excel /Users/guoxi/Documents/训练数据.xlsx \
  --output /Users/guoxi/Downloads/尺寸壁厚磅级训练集.json

python /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/scripts/生成尺寸壁厚磅级编码训练集.py \
  --excel /Users/guoxi/Documents/尺寸壁厚磅级C1训练集.xlsx \
  --output /Users/guoxi/Documents/尺寸壁厚磅级C1训练集.json \
  --unresolved-output /Users/guoxi/Documents/尺寸壁厚磅级C1训练集未定位明细.json
  
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "tokenizer_utils" / "preprocessor.py").is_file():
            return candidate
    raise RuntimeError(f"未找到项目根目录: {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer_utils.preprocessor import TextPreprocessor


SIZE_TYPES = {"DN", "OD", "INCH", "LENGTH"}
THICKNESS_TYPES = {"MM", "SCHEDULE", "INCH", "BWG", "SERIES"}
SPECIAL_SCHEDULE_TOKENS = {"STD", "XS", "XXS"}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def read_table(path: Path, sheet_name: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str).fillna("")
    if suffix in {".xlsx", ".xls"}:
        kwargs: dict[str, Any] = {"dtype": str}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        return pd.read_excel(path, **kwargs).fillna("")
    raise ValueError(f"不支持的文件类型: {path}；仅支持 .csv / .xlsx / .xls")


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}；实际列为: {list(df.columns)}")


def normalize_number_text(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    if re.fullmatch(r"\d+\.\d+", text):
        return text.rstrip("0").rstrip(".")
    return text


def normalize_inch_text(value: str) -> str:
    text = clean_text(value).upper()
    text = text.replace("NPS", "")
    text = text.replace("INCH", "")
    text = text.replace("IN", "")
    text = text.replace('"', "")
    text = text.replace("”", "")
    text = text.replace("″", "")
    return clean_text(text)


def inch_value_to_decimal(value: str) -> str | None:
    text = normalize_inch_text(value)
    if not text:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return normalize_number_text(text)

    mixed = re.fullmatch(r"(\d+)-(\d+)/(\d+)", text)
    if mixed:
        whole = int(mixed.group(1))
        numerator = int(mixed.group(2))
        denominator = int(mixed.group(3))
        if denominator == 0:
            return None
        return normalize_number_text(str(whole + numerator / denominator))

    fraction = re.fullmatch(r"(\d+)/(\d+)", text)
    if fraction:
        numerator = int(fraction.group(1))
        denominator = int(fraction.group(2))
        if denominator == 0:
            return None
        return normalize_number_text(str(numerator / denominator))

    return None


def inch_value_to_mixed_fraction(value: str) -> str | None:
    text = normalize_inch_text(value)
    if not text:
        return None
    mixed = re.fullmatch(r"(\d+)-(\d+)/(\d+)", text)
    if mixed:
        return f"{mixed.group(1)} {mixed.group(2)}/{mixed.group(3)}"
    return None


def maybe_restore_decimal_inch(text: str, value: str) -> str:
    decimal_value = inch_value_to_decimal(value)
    if not decimal_value:
        return value

    # 原文若明确用小数英寸表达，则训练集回写成小数，而不是分数。
    patterns = [
        rf"(?<![\d.]){re.escape(decimal_value)}\s*[\"”″]",
        rf"(?<![\d.]){re.escape(decimal_value)}\s*(?:INCH|IN)\b",
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return decimal_value
    return value


def normalize_size_value(size_type: str, value: str) -> str:
    text = clean_text(value)
    upper = text.upper()
    if size_type == "DN":
        upper = re.sub(r"^DN\s*", "", upper)
        return normalize_number_text(upper)
    if size_type == "OD":
        upper = re.sub(r"^(?:OD\s*[:=]?\s*|[ΦφØø])", "", upper)
        upper = re.sub(r"MM$", "", upper).strip()
        return normalize_number_text(upper)
    if size_type == "INCH":
        return normalize_inch_text(text)
    if size_type == "LENGTH":
        upper = re.sub(r"^(?:LENGTH|LEN|L)\s*[:=]?\s*", "", upper)
        upper = re.sub(r"MM$", "", upper).strip()
        normalized = normalize_number_text(upper)
        return f"{normalized}MM" if normalized else ""
    return text


def normalize_schedule_value(value: str) -> str:
    text = clean_text(value).upper()
    text = text.replace(" ", "")
    text = text.replace("SCH.", "SCH")
    text = text.replace("SCH-", "SCH")
    if text in SPECIAL_SCHEDULE_TOKENS:
        return text

    special_match = re.fullmatch(r"(?:SCH|S-)?(STD|XS|XXS)", text)
    if special_match:
        return special_match.group(1)

    match = re.fullmatch(r"SCH(\d+(?:\.\d+)?S?)", text)
    if match:
        return f"SCH{match.group(1)}"

    match = re.fullmatch(r"S-(\d+(?:\.\d+)?S?)", text)
    if match:
        return f"SCH{match.group(1)}"

    match = re.fullmatch(r"S(\d+(?:\.\d+)?S?)", text)
    if match:
        return f"SCH{match.group(1)}"

    match = re.fullmatch(r"(\d+(?:\.\d+)?S?)", text)
    if match:
        return f"SCH{match.group(1)}"

    return text


def normalize_thickness_value(thickness_type: str, value: str) -> str:
    text = clean_text(value)
    upper = text.upper()
    if thickness_type == "MM":
        upper = re.sub(r"^(?:T|THK)\s*[:=]?\s*", "", upper)
        upper = re.sub(r"MM$", "", upper).strip()
        return normalize_number_text(upper)
    if thickness_type == "SCHEDULE":
        return normalize_schedule_value(text)
    if thickness_type == "INCH":
        return normalize_inch_text(text)
    if thickness_type == "BWG":
        return normalize_number_text(re.sub(r"BWG$", "", upper).strip())
    if thickness_type == "SERIES":
        return upper.replace(" ", "")
    return text


def normalize_pressure_value(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^\s*PRESSURE\s*:\s*", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def split_value_list(value_text: str) -> list[str]:
    text = clean_text(value_text)
    if not text:
        return []
    return [clean_text(part) for part in re.split(r"\s*[xX×]\s*", text) if clean_text(part)]


def split_thickness_value_list(value_text: str) -> list[str]:
    text = clean_text(value_text)
    if not text:
        return []

    protected = text
    placeholders = {
        "XXS": "__QQS__",
        "XS": "__QS__",
        "STD": "__STD_TOKEN__",
    }
    for token, placeholder in placeholders.items():
        protected = re.sub(token, placeholder, protected, flags=re.IGNORECASE)

    parts = [clean_text(part) for part in re.split(r"\s*[xX×]\s*", protected) if clean_text(part)]
    restored: list[str] = []
    for part in parts:
        value = part
        for token, placeholder in placeholders.items():
            value = value.replace(placeholder, token)
        restored.append(value)
    return restored


def parse_size_field(size_text: str) -> tuple[list[dict[str, str]], str]:
    items: list[dict[str, str]] = []
    length_value = ""
    for segment in re.split(r"\s*[;；]\s*", clean_text(size_text)):
        if not segment:
            continue
        match = re.match(r"^\s*([A-Za-z]+)\s*:\s*(.+?)\s*$", segment)
        if not match:
            continue
        raw_type = match.group(1).upper()
        raw_values = match.group(2)
        if raw_type not in SIZE_TYPES:
            continue
        if raw_type == "LENGTH":
            length_parts = split_value_list(raw_values)
            if length_parts:
                length_value = normalize_size_value("LENGTH", length_parts[0])
            continue
        for raw_value in split_value_list(raw_values):
            normalized = normalize_size_value(raw_type, raw_value)
            if normalized:
                items.append({"type": raw_type, "value": normalized, "_raw": raw_value})
    return items, length_value


def parse_thickness_field(thickness_text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for segment in re.split(r"\s*[;；]\s*", clean_text(thickness_text)):
        if not segment:
            continue
        match = re.match(r"^\s*([A-Za-z]+)\s*:\s*(.+?)\s*$", segment)
        if not match:
            continue
        raw_type = match.group(1).upper()
        raw_values = match.group(2)
        if raw_type not in THICKNESS_TYPES:
            continue
        for raw_value in split_thickness_value_list(raw_values):
            normalized = normalize_thickness_value(raw_type, raw_value)
            if normalized:
                items.append({"type": raw_type, "value": normalized, "_raw": raw_value})
    return items


def find_schedule_candidates(text: str, normalized_value: str) -> list[str]:
    if normalized_value in SPECIAL_SCHEDULE_TOKENS:
        return [normalized_value]

    match = re.fullmatch(r"SCH(\d+(?:\.\d+)?S?)", normalized_value)
    if not match:
        return [normalized_value]

    core = match.group(1)
    digits_only = core[:-1] if core.endswith("S") else core
    has_suffix_s = core.endswith("S")
    candidates = {
        f"SCH{core}",
        f"SCH {core}",
        f"SCH.{core}",
        f"SCH. {core}",
        f"SCH-{core}",
        f"S-{core}",
        f"S{core}",
    }
    if has_suffix_s:
        candidates.update(
            {
                f"SCH{digits_only}",
                f"SCH {digits_only}",
                f"SCH.{digits_only}",
                f"SCH. {digits_only}",
                f"SCH-{digits_only}",
                f"S-{digits_only}",
                f"S{digits_only}",
                f"SCH {digits_only} S",
                f"SCH.{digits_only}S",
                f"SCH.{digits_only} S",
                f"SCH. {digits_only}S",
                f"SCH. {digits_only} S",
                f"S-{digits_only}S",
                f"S-{digits_only} S",
                f"S{digits_only}S",
                f"S{digits_only} S",
                f"{digits_only}S",
                f"{digits_only} S",
            }
        )
    return [candidate for candidate in candidates if candidate]


def locate_item_position(
    text: str,
    item_type: str,
    normalized_value: str,
    raw_value: str,
    start_at: int = 0,
) -> tuple[int, int] | None:
    patterns: list[str] = []

    if item_type == "DN":
        patterns = [
            rf"DN\s*{re.escape(normalized_value)}(?![\d.])",
            rf"(?<![\d.]){re.escape(normalized_value)}(?![\d.])",
        ]
    elif item_type == "OD":
        patterns = [
            rf"(?:OD\s*[:=]?\s*|[ΦφØø])\s*{re.escape(normalized_value)}(?![\d.])",
            rf"(?<![\d.]){re.escape(normalized_value)}(?![\d.])",
        ]
    elif item_type == "INCH":
        value = re.escape(normalized_value)
        patterns = [
            rf"NPS\s*{value}(?![\d/.-])",
            rf"{value}\s*(?:INCH|IN)\b",
            rf"{value}\s*[\"”″]",
        ]
        decimal_value = inch_value_to_decimal(normalized_value)
        if decimal_value and decimal_value != normalized_value:
            decimal_pattern = re.escape(decimal_value)
            patterns.extend(
                [
                    rf"NPS\s*{decimal_pattern}(?![\d/.-])",
                    rf"{decimal_pattern}\s*(?:INCH|IN)\b",
                    rf"{decimal_pattern}\s*[\"”″]",
                ]
            )
        mixed_fraction_value = inch_value_to_mixed_fraction(normalized_value)
        if mixed_fraction_value:
            mixed_pattern = re.escape(mixed_fraction_value).replace(r"\ ", r"\s+")
            patterns.extend(
                [
                    rf"NPS\s*{mixed_pattern}(?![\d/.-])",
                    rf"{mixed_pattern}\s*(?:INCH|IN)\b",
                    rf"{mixed_pattern}\s*[\"”″]",
                    rf"(?<![\d/.-]){mixed_pattern}(?![\d/.-])",
                ]
            )
    elif item_type == "MM":
        value = re.escape(normalized_value)
        mm_unit = r"(?:MM|OMM|0MM)\b"
        patterns = [
            rf"(?:THK|T)\s*[:=\-]?\s*{value}\s*{mm_unit}",
            rf"{value}\s*{mm_unit}",
            rf"(?<![\d.]){value}(?![\d.])",
        ]
        raw_decimal = re.sub(r"(?<=\d)(?:OMM|MM)$", "", clean_text(raw_value).upper()).strip()
        if re.fullmatch(r"\d+\.\d+", raw_decimal):
            raw_decimal_pattern = re.escape(raw_decimal)
            int_part = re.escape(raw_decimal.split(".", 1)[0])
            patterns.extend(
                [
                    rf"(?:THK|T)\s*[:=\-]?\s*{raw_decimal_pattern}\s*{mm_unit}",
                    rf"{raw_decimal_pattern}\s*{mm_unit}",
                    rf"(?<![\d.]){raw_decimal_pattern}(?![\d.])",
                    rf"(?:THK|T)\s*[:=\-]?\s*{int_part}\s*\.\s*(?:0\s*)?{mm_unit}",
                    rf"{int_part}\s*\.\s*(?:0\s*)?{mm_unit}",
                    rf"(?<![\d.]){int_part}\s*\.\s*(?:0\s*)?(?![\d.])",
                ]
            )
    elif item_type == "SCHEDULE":
        patterns = [re.escape(candidate) for candidate in find_schedule_candidates(text, normalized_value)]
    elif item_type == "BWG":
        patterns = [rf"{re.escape(normalized_value)}\s*BWG\b"]
    elif item_type == "SERIES":
        patterns = [re.escape(normalized_value)]

    if raw_value:
        raw_candidate = clean_text(raw_value).upper().replace(" ", "")
        if raw_candidate and re.escape(raw_candidate) not in patterns:
            patterns.append(re.escape(raw_candidate))

    for pattern in patterns:
        match = re.search(pattern, text[start_at:], flags=re.IGNORECASE)
        if match:
            start = start_at + match.start()
            end = start_at + match.end()
            return start, end
    return None


def sort_items_by_text_order(text: str, items: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ordered: list[tuple[int, int, int, dict[str, str]]] = []
    cursor = 0
    unresolved: list[dict[str, str]] = []

    for index, item in enumerate(items):
        located = locate_item_position(
            text=text,
            item_type=item["type"],
            normalized_value=item["value"],
            raw_value=item.get("_raw", ""),
            start_at=cursor,
        )
        if located is None:
            located = locate_item_position(
                text=text,
                item_type=item["type"],
                normalized_value=item["value"],
                raw_value=item.get("_raw", ""),
                start_at=0,
            )
        if located is None:
            unresolved.append({"type": item["type"], "value": item["value"], "raw": item.get("_raw", "")})
            continue
        start, end = located
        cursor = end
        ordered.append((start, end, index, item))

    ordered.sort(key=lambda row: (row[0], row[1], row[2]))

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, _, item in ordered:
        key = (item["type"], item["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"type": item["type"], "value": item["value"]})
    return deduped, unresolved


def build_record(
    preprocessor: TextPreprocessor,
    description: str,
    size_text: str,
    thickness_text: str,
    pressure_text: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    processed_text = preprocessor.process(description).upper()
    size_items, length_value = parse_size_field(size_text)
    thickness_items = parse_thickness_field(thickness_text)
    ordered_size_items, unresolved_size_items = sort_items_by_text_order(processed_text, size_items)
    for item in ordered_size_items:
        if item["type"] == "INCH":
            item["value"] = maybe_restore_decimal_inch(processed_text, item["value"])
    ordered_thickness_items, unresolved_thickness_items = sort_items_by_text_order(processed_text, thickness_items)

    unresolved_items = []
    if unresolved_size_items:
        unresolved_items.append({"field": "SIZE_ITEMS", "items": unresolved_size_items})
    if unresolved_thickness_items:
        unresolved_items.append({"field": "THICKNESS_ITEMS", "items": unresolved_thickness_items})

    if unresolved_items:
        return None, {
            "input": description,
            "processed_text": processed_text,
            "size_text": size_text,
            "thickness_text": thickness_text,
            "pressure_text": pressure_text,
            "unresolved": unresolved_items,
        }

    return {
        "input": description,
        "output": {
            "SIZE_ITEMS": ordered_size_items,
            "LENGTH": length_value,
            "THICKNESS_ITEMS": ordered_thickness_items,
            "PRESSURE": normalize_pressure_value(pressure_text),
        },
    }, None


def build_output_path(excel_path: Path, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        return output_path if output_path.suffix.lower() == ".json" else output_path.with_suffix(".json")
    return excel_path.with_name(f"{excel_path.stem}_尺寸壁厚磅级训练集.json")


def build_unresolved_output_path(output_path: Path, unresolved_output: str | None) -> Path:
    if unresolved_output:
        path = Path(unresolved_output)
        return path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    return output_path.with_name(f"{output_path.stem}_未定位明细.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成尺寸、壁厚、磅级训练集 JSON")
    parser.add_argument("--excel", required=True, help="输入文件路径，支持 csv/xlsx/xls")
    parser.add_argument("--sheet", help="sheet 名，CSV 可不传")
    parser.add_argument("--desc-col", default="原始描述", help="原始描述列名")
    parser.add_argument("--size-col", default="SIZE_原始结果", help="尺寸列名")
    parser.add_argument("--thickness-col", default="THICKNESS_原始结果", help="壁厚列名")
    parser.add_argument("--pressure-col", default="PRESSURE_原始结果", help="磅级列名")
    parser.add_argument("--output", help="输出 JSON 路径")
    parser.add_argument("--unresolved-output", help="未定位描述输出 JSON 路径")
    parser.add_argument("--skip-empty", action="store_true", help="跳过尺寸、壁厚、磅级都为空的记录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excel_path = Path(args.excel)
    df = read_table(excel_path, args.sheet)
    validate_columns(df, [args.desc_col, args.size_col, args.thickness_col, args.pressure_col])

    preprocessor = TextPreprocessor()
    records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        description = clean_text(row.get(args.desc_col, ""))
        size_text = clean_text(row.get(args.size_col, ""))
        thickness_text = clean_text(row.get(args.thickness_col, ""))
        pressure_text = clean_text(row.get(args.pressure_col, ""))

        if not description:
            continue
        if args.skip_empty and not any([size_text, thickness_text, pressure_text]):
            continue

        record, unresolved = build_record(preprocessor, description, size_text, thickness_text, pressure_text)
        if unresolved is not None:
            unresolved_records.append(unresolved)
            continue
        if record is not None:
            records.append(record)

    output_path = build_output_path(excel_path, args.output)
    unresolved_output_path = build_unresolved_output_path(output_path, args.unresolved_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    unresolved_output_path.write_text(json.dumps(unresolved_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已生成: {output_path}")
    print(f"训练集条数: {len(records)}")
    print(f"未定位描述条数: {len(unresolved_records)}")
    print(f"未定位明细: {unresolved_output_path}")


if __name__ == "__main__":
    main()
