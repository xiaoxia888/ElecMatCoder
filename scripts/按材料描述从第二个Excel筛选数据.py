#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据第一个 Excel 中的材料描述，从第二个 Excel 中筛选命中行，并输出新的 Excel。

规则：
1. 读取第一个文件中的“材料描述”列，生成匹配集合
2. 读取第二个文件中的“材料描述”列
3. 将第二个文件中“材料描述”出现在第一个文件中的整行全部保留
4. 输出结果保留第二个文件的全部原始列与原始顺序

默认列名均为“材料描述”，可通过参数覆盖。

示例：
python scripts/按材料描述从第二个Excel筛选数据.py \
    --excel1 /Users/guoxi/Documents/描述集合.xlsx \
    --excel2 /Users/guoxi/Documents/原始明细.xlsx \
    --output /Users/guoxi/Downloads/筛选结果.xlsx

如需指定 sheet：
python scripts/按材料描述从第二个Excel筛选数据.py \
    --excel1 /Users/guoxi/Documents/描述集合.xlsx \
    --excel2 /Users/guoxi/Documents/原始明细.xlsx \
    --excel1-sheet Sheet1 \
    --excel2-sheet Sheet1 \
    --excel1-desc-col 材料描述 \
    --excel2-desc-col 材料描述 \
    --output /Users/guoxi/Downloads/筛选结果.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    """做基础归一化，避免换行、全角空格、重复空格影响匹配。"""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = text.replace("\u3000", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.strip().split())


def read_table(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, dtype=str).fillna("")
    if suffix in {".xlsx", ".xls"}:
        kwargs: dict[str, Any] = {"dtype": str}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        return pd.read_excel(file_path, **kwargs).fillna("")
    raise ValueError(f"不支持的文件类型: {path}；仅支持 .csv / .xlsx / .xls")


def validate_column(df: pd.DataFrame, file_label: str, col_name: str, arg_name: str) -> None:
    if col_name not in df.columns:
        raise ValueError(
            f"{file_label} 缺少列: {col_name}；"
            f"请通过参数 {arg_name} 指定正确列名。"
            f"实际列为: {list(df.columns)}"
        )


def build_output_path(excel2: str, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        if output_path.suffix.lower() not in {".xlsx", ".xls"}:
            output_path = output_path.with_suffix(".xlsx")
        return output_path
    excel2_path = Path(excel2)
    return excel2_path.with_name(f"{excel2_path.stem}_按材料描述筛选.xlsx")


def build_safe_output_col_name(df: pd.DataFrame, requested_name: str) -> str:
    """避免和 excel2 原有列重名。"""
    if requested_name not in df.columns:
        return requested_name
    return f"{requested_name}(excel1)"


def build_desc_to_category_map(df: pd.DataFrame, desc_col: str, category_col: str) -> dict[str, str]:
    """按材料描述汇总 excel1 分类列；同一描述多个分类时去重后拼接。"""
    desc_to_categories: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        desc = normalize_text(row.get(desc_col, ""))
        category = normalize_text(row.get(category_col, ""))
        if not desc or not category:
            continue
        desc_to_categories.setdefault(desc, []).append(category)

    desc_to_category_text: dict[str, str] = {}
    for desc, categories in desc_to_categories.items():
        unique_categories: list[str] = []
        seen: set[str] = set()
        for category in categories:
            if category not in seen:
                seen.add(category)
                unique_categories.append(category)
        desc_to_category_text[desc] = " | ".join(unique_categories)
    return desc_to_category_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据第一个 Excel 的材料描述，从第二个 Excel 中筛选命中行")
    parser.add_argument("--excel1", required=True, help="第一个文件路径，提供材料描述集合，支持 csv/xlsx/xls")
    parser.add_argument("--excel2", required=True, help="第二个文件路径，被筛选的数据源，支持 csv/xlsx/xls")
    parser.add_argument("--excel1-sheet", help="第一个 Excel 的 sheet 名，CSV 可不传")
    parser.add_argument("--excel2-sheet", help="第二个 Excel 的 sheet 名，CSV 可不传")
    parser.add_argument("--excel1-desc-col", default="材料描述", help="第一个文件的材料描述列名")
    parser.add_argument("--excel2-desc-col", default="材料描述", help="第二个文件的材料描述列名")
    parser.add_argument("--excel1-category-col", help="第一个文件的分类列名；若传入，则会回填到输出文件中")
    parser.add_argument("--output", help="输出文件路径，默认在第二个文件同目录生成 *_按材料描述筛选.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df1 = read_table(args.excel1, args.excel1_sheet)
    df2 = read_table(args.excel2, args.excel2_sheet)

    validate_column(df1, "excel1", args.excel1_desc_col, "--excel1-desc-col")
    validate_column(df2, "excel2", args.excel2_desc_col, "--excel2-desc-col")
    if args.excel1_category_col:
        validate_column(df1, "excel1", args.excel1_category_col, "--excel1-category-col")

    desc_set = {
        normalized
        for normalized in df1[args.excel1_desc_col].map(normalize_text)
        if normalized
    }

    normalized_desc2 = df2[args.excel2_desc_col].map(normalize_text)
    result_df = df2[normalized_desc2.isin(desc_set)].copy()

    if args.excel1_category_col:
        output_category_col = build_safe_output_col_name(result_df, args.excel1_category_col)
        desc_to_category = build_desc_to_category_map(df1, args.excel1_desc_col, args.excel1_category_col)
        result_df[output_category_col] = result_df[args.excel2_desc_col].map(
            lambda value: desc_to_category.get(normalize_text(value), "")
        )

    output_path = build_output_path(args.excel2, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(output_path, index=False)

    print(f"已生成: {output_path}")
    print(f"excel1 去重后材料描述数: {len(desc_set)}")
    print(f"excel2 原始行数: {len(df2)}")
    print(f"命中输出行数: {len(result_df)}")


if __name__ == "__main__":
    main()
