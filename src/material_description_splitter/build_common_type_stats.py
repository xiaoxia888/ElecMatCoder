# -*- coding: utf-8 -*-
"""统计常见种类编码及出现次数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def build_type_stats(input_file: str, output_file: str | None = None) -> Path:
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path, usecols=["标准化种类", "修正种类"])
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path, usecols=["标准化种类", "修正种类"])
    else:
        raise ValueError(f"不支持的输入文件类型: {input_path.suffix}")

    values: list[str] = []
    for _, row in df.iterrows():
        corrected = _clean_cell(row.get("修正种类", ""))
        standardized = _clean_cell(row.get("标准化种类", ""))
        values.append(corrected or standardized)

    series = pd.Series(values, name="种类编码")
    total_rows = len(series)
    non_empty = series[series != ""]
    stats = (
        non_empty.value_counts()
        .rename_axis("种类编码")
        .reset_index(name="出现次数")
    )
    stats["占比"] = (stats["出现次数"] / total_rows).round(6)
    stats["累计占比"] = stats["占比"].cumsum().round(6)
    stats["是否空值"] = False

    empty_count = int((series == "").sum())
    empty_row = pd.DataFrame(
        [
            {
                "种类编码": "",
                "出现次数": empty_count,
                "占比": round(empty_count / total_rows, 6) if total_rows else 0,
                "累计占比": "",
                "是否空值": True,
            }
        ]
    )

    summary = pd.DataFrame(
        [
            {"指标": "总行数", "值": total_rows},
            {"指标": "非空种类数", "值": int((series != "").sum())},
            {"指标": "空值数", "值": empty_count},
            {"指标": "去重种类数", "值": int(non_empty.nunique())},
        ]
    )

    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_种类编码统计.xlsx")
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() == ".csv":
            output_path = output_path.with_suffix(".xlsx")
        if output_path.suffix.lower() not in {".xlsx", ".xls"}:
            output_path = output_path.with_suffix(".xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        stats.to_excel(writer, sheet_name="种类编码统计", index=False)
        empty_row.to_excel(writer, sheet_name="空值情况", index=False)
        summary.to_excel(writer, sheet_name="说明", index=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="统计常见种类编码及出现次数")
    parser.add_argument("--input-file", required=True, help="输入文件，支持 csv/xlsx/xls")
    parser.add_argument("--output-file", help="输出文件，统一为 xlsx")
    args = parser.parse_args()

    output = build_type_stats(args.input_file, args.output_file)
    print(str(output))


if __name__ == "__main__":
    main()
