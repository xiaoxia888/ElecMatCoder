#!/usr/bin/env python3
"""
根据现有字段编码列，生成“壁厚转换后材料代码”。

规则：
1. 读取 Excel 中 6 个字段编码列：
   - excel2_TYPE_原始编码
   - excel2_SIZE_原始编码
   - excel2_THICKNESS_原始编码
   - excel2_PRESSURE_原始编码
   - excel2_MATERIAL_原始编码
   - excel2_STANDARD_原始编码
2. 仅当“尺寸”为单值、且“壁厚”为单值时，才尝试按平台当前壁厚对照表换算。
3. 若尺寸为异径（如 50x20）或壁厚为多段（如 XXSxXS），则“壁厚转换后材料代码”留空。
4. 尺寸中的长度前缀（如 50L200）会忽略长度，仅保留 DN50 参与换算。
5. 壁厚换算成功后，按：
   TYPE + SIZE + 转换后壁厚 + PRESSURE + MATERIAL + STANDARD
   拼接生成新的材料代码。
6. 若换算失败，则“壁厚转换后材料代码”留空。

示例：
python scripts/生成壁厚转换后材料代码.py \
  --excel /Users/guoxi/Documents/demo.xlsx \
  --output /Users/guoxi/Documents/demo_壁厚换算.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder.processors.thickness_table_processor import get_thickness_table_processor


DEFAULT_TYPE_COL = "excel2_TYPE_原始编码"
DEFAULT_SIZE_COL = "excel2_SIZE_原始编码"
DEFAULT_THICKNESS_COL = "excel2_THICKNESS_原始编码"
DEFAULT_PRESSURE_COL = "excel2_PRESSURE_原始编码"
DEFAULT_MATERIAL_COL = "excel2_MATERIAL_原始编码"
DEFAULT_STANDARD_COL = "excel2_STANDARD_原始编码"
DEFAULT_ORIGINAL_CODE_COL = "excel2_原始总编码"
OUTPUT_COL = "壁厚转换后材料代码"
CONVERTED_FLAG_COL = "是否转换"
NO_CONVERT_REASON_COL = "未转换原因"
MATCHED_STANDARD_COL = "壁厚换算匹配规范"


@dataclass
class ConversionOutcome:
    final_code: str
    converted: str
    reason: str
    matched_standard: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于现有字段编码列生成壁厚转换后材料代码。")
    parser.add_argument("--excel", required=True, help="输入 Excel 路径（.xlsx / .xls）。")
    parser.add_argument("--sheet", default=None, help="工作簿名称，默认读取第一个工作簿。")
    parser.add_argument("--output", default=None, help="输出 Excel 路径，默认在原文件名后追加“_壁厚换算”。")
    parser.add_argument("--type-col", default=DEFAULT_TYPE_COL, help=f"种类编码列，默认: {DEFAULT_TYPE_COL}")
    parser.add_argument("--size-col", default=DEFAULT_SIZE_COL, help=f"尺寸编码列，默认: {DEFAULT_SIZE_COL}")
    parser.add_argument("--thickness-col", default=DEFAULT_THICKNESS_COL, help=f"壁厚编码列，默认: {DEFAULT_THICKNESS_COL}")
    parser.add_argument("--pressure-col", default=DEFAULT_PRESSURE_COL, help=f"磅级编码列，默认: {DEFAULT_PRESSURE_COL}")
    parser.add_argument("--material-col", default=DEFAULT_MATERIAL_COL, help=f"材质编码列，默认: {DEFAULT_MATERIAL_COL}")
    parser.add_argument("--standard-col", default=DEFAULT_STANDARD_COL, help=f"规范编码列，默认: {DEFAULT_STANDARD_COL}")
    parser.add_argument("--original-code-col", default=DEFAULT_ORIGINAL_CODE_COL, help=f"原始总编码列，默认: {DEFAULT_ORIGINAL_CODE_COL}")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}；实际列为: {list(df.columns)}")


def read_excel(path: str, sheet: str | None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"仅支持 Excel 文件，当前文件为: {path}")
    kwargs = {"dtype": str}
    if sheet:
        kwargs["sheet_name"] = sheet
    return pd.read_excel(file_path, **kwargs).fillna("")


def build_output_path(input_path: str, output_path: str | None) -> Path:
    if output_path:
        out = Path(output_path)
        if out.suffix.lower() not in {".xlsx", ".xls"}:
            out = out.with_suffix(".xlsx")
        return out
    source = Path(input_path)
    return source.with_name(f"{source.stem}_壁厚换算.xlsx")


def parse_dn_values_from_size(size_code: str) -> list[str]:
    text = normalize_text(size_code)
    if not text:
        return []

    size_part = re.sub(r"L\d+(?:\.\d+)?", "", text, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"[xX×]", size_part) if part.strip()]
    result: list[str] = []
    for part in parts:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)", part)
        if not match:
            return []
        dn = match.group(1)
        if "." in dn:
            dn = dn.rstrip("0").rstrip(".")
        result.append(dn)
    return result


def format_converted_mm(mm_value: str) -> str:
    text = normalize_text(mm_value)
    if not text:
        return ""
    return text if text.upper().endswith("MM") else f"{text}MM"


def is_explicit_mm_code(thickness_code: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?MM", normalize_text(thickness_code), flags=re.IGNORECASE))


def build_thickness_items(thickness_code: str) -> list[dict[str, str]]:
    processor = get_thickness_table_processor()
    return processor.build_thickness_items(thickness_code)


def split_thickness_items(items: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    schedule_items: list[dict[str, str]] = []
    mm_items: list[dict[str, str]] = []
    other_items: list[dict[str, str]] = []
    for item in items:
        item_type = str(item.get("type") or "").strip().upper()
        if item_type == "SCHEDULE":
            schedule_items.append(item)
        elif item_type == "MM":
            mm_items.append(item)
        else:
            other_items.append(item)
    return schedule_items, mm_items, other_items


def get_item_display_value(item: dict[str, Any]) -> str:
    return normalize_text(item.get("normalized") or item.get("value") or "")


def build_converted_thickness(
    processor: Any,
    standard_code: str,
    dn_values: list[str],
    thickness_items: list[dict[str, str]],
) -> tuple[str, str, str, str]:
    if not thickness_items:
        return "", "否", "壁厚为空", ""

    schedule_items, mm_items, other_items = split_thickness_items(thickness_items)
    if other_items:
        return "", "否", "壁厚存在无法识别的类型", ""

    if len(dn_values) != 1 and len(dn_values) != len(thickness_items):
        return "", "否", "尺寸与壁厚数量不满足1对多或N对N", ""

    if schedule_items and not standard_code:
        return "", "否", "存在SCHEDULE壁厚，但规范为空", ""

    if len(schedule_items) == 0 and len(mm_items) == len(thickness_items):
        return "", "否", "原壁厚已是MM值，无需转换", ""

    pairs: list[tuple[str, dict[str, str]]] = []
    if len(dn_values) == 1:
        pairs = [(dn_values[0], item) for item in thickness_items]
    else:
        pairs = list(zip(dn_values, thickness_items))

    converted_parts: list[str] = []
    matched_standard_infos: list[str] = []
    for dn_value, item in pairs:
        item_type = str(item.get("type") or "").strip().upper()
        display_value = get_item_display_value(item)
        if item_type == "MM":
            converted_parts.append(format_converted_mm(display_value))
            continue
        if item_type != "SCHEDULE":
            return "", "否", f"壁厚类型不支持转换: {display_value}", ""

        lookup_detail = processor.lookup_mm_detail(standard_code, dn_value, display_value)
        target_standard = normalize_text(lookup_detail.get("target_standard", ""))
        matched_standard = normalize_text(lookup_detail.get("matched_standard", ""))
        looked_up_mm = normalize_text(lookup_detail.get("mm", ""))
        lookup_reason = normalize_text(lookup_detail.get("reason", ""))

        if target_standard or matched_standard:
            info = f"{display_value}(DN{dn_value})"
            if target_standard and matched_standard:
                info += f": {target_standard} -> {matched_standard}"
            elif target_standard:
                info += f": {target_standard}"
            matched_standard_infos.append(info)

        if not looked_up_mm:
            if not target_standard:
                return "", "否", f"规范未映射到壁厚对照目标标准: {standard_code}", ""
            if lookup_reason == "thickness_value_empty":
                matched_text = matched_standard or target_standard
                return (
                    "",
                    "否",
                    f"壁厚对照表已命中，但壁厚值为空: 原规范={standard_code} -> 目标规范={target_standard} -> 匹配规范={matched_text}；壁厚={display_value}；DN{dn_value}",
                    "；".join(matched_standard_infos),
                )
            return (
                "",
                "否",
                f"壁厚对照表未命中: 原规范={standard_code} -> 目标规范={target_standard}；壁厚={display_value}；DN{dn_value}",
                "；".join(matched_standard_infos),
            )

        converted_parts.append(format_converted_mm(looked_up_mm))

    deduped_parts: list[str] = []
    for part in converted_parts:
        if not part:
            continue
        if part not in deduped_parts:
            deduped_parts.append(part)

    if not deduped_parts:
        return "", "否", "壁厚换算后为空", "；".join(matched_standard_infos)

    final_thickness = deduped_parts[0] if len(deduped_parts) == 1 else "X".join(deduped_parts)
    return final_thickness, "是", "", "；".join(matched_standard_infos)


def build_converted_code(
    type_code: str,
    size_code: str,
    converted_thickness: str,
    pressure_code: str,
    material_code: str,
    standard_code: str,
) -> ConversionOutcome:
    return "".join(
        [
            normalize_text(type_code),
            normalize_text(size_code),
            normalize_text(converted_thickness),
            normalize_text(pressure_code),
            normalize_text(material_code),
            normalize_text(standard_code),
        ]
    )


def build_original_code(
    row: pd.Series,
    original_code_col: str,
    type_col: str,
    size_col: str,
    thickness_col: str,
    pressure_col: str,
    material_col: str,
    standard_col: str,
) -> str:
    original_code = normalize_text(row.get(original_code_col, ""))
    if original_code:
        return original_code
    return build_converted_code(
        type_code=row.get(type_col, ""),
        size_code=row.get(size_col, ""),
        converted_thickness=row.get(thickness_col, ""),
        pressure_code=row.get(pressure_col, ""),
        material_code=row.get(material_col, ""),
        standard_code=row.get(standard_col, ""),
    )


def convert_row(
    row: pd.Series,
    original_code_col: str,
    type_col: str,
    size_col: str,
    thickness_col: str,
    pressure_col: str,
    material_col: str,
    standard_col: str,
) -> str:
    original_code = build_original_code(
        row,
        original_code_col,
        type_col,
        size_col,
        thickness_col,
        pressure_col,
        material_col,
        standard_col,
    )
    type_code = normalize_text(row.get(type_col, ""))
    size_code = normalize_text(row.get(size_col, ""))
    thickness_code = normalize_text(row.get(thickness_col, ""))
    pressure_code = normalize_text(row.get(pressure_col, ""))
    material_code = normalize_text(row.get(material_col, ""))
    standard_code = normalize_text(row.get(standard_col, ""))

    if not size_code or not thickness_code:
        return ConversionOutcome(
            final_code=original_code,
            converted="否",
            reason="尺寸或壁厚为空",
            matched_standard="",
        )

    dn_values = parse_dn_values_from_size(size_code)
    if not dn_values:
        return ConversionOutcome(
            final_code=original_code,
            converted="否",
            reason="尺寸包含多个值、不是单一DN，或只包含长度信息",
            matched_standard="",
        )

    processor = get_thickness_table_processor()
    thickness_items = build_thickness_items(thickness_code)
    converted_thickness, converted_flag, reason, matched_standard = build_converted_thickness(
        processor=processor,
        standard_code=standard_code,
        dn_values=dn_values,
        thickness_items=thickness_items,
    )
    if converted_flag != "是":
        return ConversionOutcome(
            final_code=original_code,
            converted=converted_flag,
            reason=reason,
            matched_standard=matched_standard,
        )

    return ConversionOutcome(
        final_code=build_converted_code(
            type_code=type_code,
            size_code=size_code,
            converted_thickness=converted_thickness,
            pressure_code=pressure_code,
            material_code=material_code,
            standard_code=standard_code,
        ),
        converted="是",
        reason="",
        matched_standard=matched_standard,
    )


def main() -> None:
    args = parse_args()
    df = read_excel(args.excel, args.sheet)
    validate_columns(
        df,
        [
            args.original_code_col,
            args.type_col,
            args.size_col,
            args.thickness_col,
            args.pressure_col,
            args.material_col,
            args.standard_col,
        ],
    )

    outcomes = df.apply(
        convert_row,
        axis=1,
        args=(
            args.original_code_col,
            args.type_col,
            args.size_col,
            args.thickness_col,
            args.pressure_col,
            args.material_col,
            args.standard_col,
        ),
    )
    df[OUTPUT_COL] = outcomes.map(lambda item: item.final_code)
    df[CONVERTED_FLAG_COL] = outcomes.map(lambda item: item.converted)
    df[NO_CONVERT_REASON_COL] = outcomes.map(lambda item: item.reason)
    df[MATCHED_STANDARD_COL] = outcomes.map(lambda item: item.matched_standard)

    output_path = build_output_path(args.excel, args.output)
    df.to_excel(output_path, index=False)
    print(f"已生成: {output_path}")
    print(f"总行数: {len(df)}")
    print(f"成功换算行数: {(df[CONVERTED_FLAG_COL] == '是').sum()}")


if __name__ == "__main__":
    main()
