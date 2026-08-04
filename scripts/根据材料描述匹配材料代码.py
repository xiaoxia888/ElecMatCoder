#!/usr/bin/env python3
"""
根据两个 Excel/CSV 按材料描述匹配材料代码。

输入:
1. excel1: 仅要求存在材料描述列，保留其全部原始列
2. excel2: 要求存在材料描述列和材料代码列

规则:
1. 按材料描述匹配材料代码
2. 如果 excel1 某条描述在 excel2 命中多行，且这些行的材料代码不同:
   - 取 excel2 中出现的第一条代码作为返回值
   - 用单独列标记为编码冲突
3. 如果命中多行但材料代码都相同:
   - 正常返回该代码
   - 不标记冲突

输出:
- 新的 Excel 文件
- 保留 excel1 的所有原列
- 新增:
  - 匹配材料代码
  - 匹配命中行数
  - 编码冲突标记
  - 候选材料代码
  - 二次分流最终难度
  - 分流原因

python scripts/根据材料描述匹配材料代码.py \
    --excel1 /Users/guoxi/Documents/C1库运行/法兰/法兰总数据0724.xlsx \
    --excel2 '/Users/guoxi/Downloads/编码结果 (53).csv'\
    --excel1-desc-col 材料描述 \
    --excel2-desc-col 原始描述 \
    --excel2-code-col 原始总编码 \
    --output /Users/guoxi/Documents/C1库运行/法兰/法兰总数据0731-匹配后.xlsx

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
    
import pandas as pd


@dataclass
class MatchSummary:
    selected_code: str
    hit_count: int
    conflict: str
    candidate_codes: str
    selected_second_pass_level: str
    selected_reason: str
    selected_row: dict[str, str]


def normalize_text(value: object) -> str:
    """做基础归一化，避免换行、全角空格、重复空格影响精确匹配。"""
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
        kwargs = {"dtype": str}
        if sheet_name:
            kwargs["sheet_name"] = sheet_name
        return pd.read_excel(file_path, **kwargs).fillna("")
    raise ValueError(f"不支持的文件类型: {path}；仅支持 .csv / .xlsx / .xls")


def validate_columns(df: pd.DataFrame, file_label: str, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{file_label} 缺少列: {missing}；实际列为: {list(df.columns)}")


def warn_if_optional_columns_missing(df: pd.DataFrame, file_label: str, optional: Iterable[str]) -> None:
    missing = [col for col in optional if col not in df.columns]
    if missing:
        print(f"提示: {file_label} 缺少可选列 {missing}，对应输出将留空。")


def build_desc_index(
    df: pd.DataFrame,
    desc_col: str,
    code_col: str,
    second_pass_col: str,
    reason_col: str,
) -> dict[str, list[dict[str, str]]]:
    """保留 excel2 中同一描述对应记录的出现顺序，便于“取第一条”。"""
    index: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        desc = normalize_text(row.get(desc_col, ""))
        code = normalize_text(row.get(code_col, ""))
        if not desc:
            continue
        index.setdefault(desc, []).append(
            {
                "code": code,
                "second_pass_level": normalize_text(row.get(second_pass_col, "")),
                "reason": normalize_text(row.get(reason_col, "")),
                "row_data": {str(col): normalize_text(row.get(col, "")) for col in df.columns},
            }
        )
    return index


def summarize_matches(records: list[dict[str, str]]) -> MatchSummary:
    effective_codes = [record["code"] for record in records if record.get("code")]
    if not effective_codes:
        first_record = records[0] if records else {}
        return MatchSummary(
            selected_code="",
            hit_count=len(records),
            conflict="",
            candidate_codes="",
            selected_second_pass_level=str(first_record.get("second_pass_level", "")),
            selected_reason=str(first_record.get("reason", "")),
            selected_row=dict(first_record.get("row_data", {}) or {}),
        )

    unique_codes: list[str] = []
    seen: set[str] = set()
    for code in effective_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    conflict = "是" if len(unique_codes) > 1 else ""
    first_record = next((record for record in records if record.get("code")), records[0] if records else {})
    return MatchSummary(
        selected_code=effective_codes[0],
        hit_count=len(records),
        conflict=conflict,
        candidate_codes=" | ".join(unique_codes),
        selected_second_pass_level=str(first_record.get("second_pass_level", "")),
        selected_reason=str(first_record.get("reason", "")),
        selected_row=dict(first_record.get("row_data", {}) or {}),
    )


def match_codes(
    df_source: pd.DataFrame,
    df_lookup: pd.DataFrame,
    source_desc_col: str,
    source_desc_fallback_col: str | None,
    lookup_desc_col: str,
    lookup_code_col: str,
    lookup_second_pass_col: str,
    lookup_reason_col: str,
) -> pd.DataFrame:
    lookup_index = build_desc_index(
        df_lookup,
        lookup_desc_col,
        lookup_code_col,
        lookup_second_pass_col,
        lookup_reason_col,
    )
    result_df = df_source.copy()

    match_codes_list: list[str] = []
    hit_count_list: list[int] = []
    conflict_list: list[str] = []
    candidate_codes_list: list[str] = []
    second_pass_level_list: list[str] = []
    reason_list: list[str] = []
    matched_lookup_rows: list[dict[str, str]] = []
    fallback_match_count = 0

    for _, row in df_source.iterrows():
        desc = normalize_text(row.get(source_desc_col, ""))
        matched_records = lookup_index.get(desc, [])
        if not matched_records and source_desc_fallback_col:
            fallback_desc = normalize_text(row.get(source_desc_fallback_col, ""))
            if fallback_desc and fallback_desc != desc:
                matched_records = lookup_index.get(fallback_desc, [])
                if matched_records:
                    fallback_match_count += 1
        summary = summarize_matches(matched_records)
        match_codes_list.append(summary.selected_code)
        hit_count_list.append(summary.hit_count)
        conflict_list.append(summary.conflict)
        candidate_codes_list.append(summary.candidate_codes)
        second_pass_level_list.append(summary.selected_second_pass_level)
        reason_list.append(summary.selected_reason)
        matched_lookup_rows.append(summary.selected_row)

    result_df["匹配材料代码"] = match_codes_list
    result_df["匹配命中行数"] = hit_count_list
    result_df["编码冲突标记"] = conflict_list
    result_df["候选材料代码"] = candidate_codes_list
    result_df["二次分流最终难度"] = second_pass_level_list
    result_df["分流原因"] = reason_list

    result_df[""] = ""

    lookup_columns = [f"excel2_{col}" for col in df_lookup.columns]
    lookup_rows = [
        {f"excel2_{col}": row_data.get(str(col), "") for col in df_lookup.columns}
        for row_data in matched_lookup_rows
    ]
    lookup_df = pd.DataFrame(lookup_rows, columns=lookup_columns)
    merged_df = pd.concat([result_df, lookup_df], axis=1)
    merged_df.attrs["fallback_match_count"] = fallback_match_count
    return merged_df


def build_output_path(path1: str, output_path: str | None) -> Path:
    if output_path:
        out = Path(output_path)
        if out.suffix.lower() not in {".xlsx", ".xls"}:
            out = out.with_suffix(".xlsx")
        return out
    source = Path(path1)
    return source.with_name(f"{source.stem}_匹配材料代码.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据材料描述匹配材料代码，并输出新的 Excel 文件")
    parser.add_argument("--excel1", required=True, help="源文件路径，支持 csv/xlsx/xls")
    parser.add_argument("--excel2", required=True, help="对照文件路径，支持 csv/xlsx/xls")
    parser.add_argument("--excel1-sheet", help="excel1 sheet 名，CSV 可不传")
    parser.add_argument("--excel2-sheet", help="excel2 sheet 名，CSV 可不传")
    parser.add_argument("--excel1-desc-col", default="材料描述", help="excel1 的材料描述列名")
    parser.add_argument(
        "--excel1-desc-fallback-col",
        default="材料描述(多行)",
        help="excel1 的兜底材料描述列名。主描述没匹配上时，会再用这个列尝试匹配；传空字符串可关闭",
    )
    parser.add_argument("--excel2-desc-col", default="材料描述", help="excel2 的材料描述列名")
    parser.add_argument("--excel2-code-col", default="材料代码", help="excel2 的材料代码列名")
    parser.add_argument("--excel2-second-pass-col", default="二次分流最终难度", help="excel2 的二次分流最终难度列名")
    parser.add_argument("--excel2-reason-col", default="分流原因", help="excel2 的分流原因列名")
    parser.add_argument("--output", help="输出文件路径，默认在 excel1 同目录生成 *_匹配材料代码.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df1 = read_table(args.excel1, args.excel1_sheet)
    df2 = read_table(args.excel2, args.excel2_sheet)
    fallback_col = (args.excel1_desc_fallback_col or "").strip() or None

    validate_columns(df1, "excel1", [args.excel1_desc_col])
    if fallback_col and fallback_col not in df1.columns:
        print(f"提示: excel1 缺少可选列 ['{fallback_col}']，将跳过兜底描述匹配。")
        fallback_col = None
    validate_columns(
        df2,
        "excel2",
        [
            args.excel2_desc_col,
            args.excel2_code_col,
        ],
    )
    warn_if_optional_columns_missing(
        df2,
        "excel2",
        [
            args.excel2_second_pass_col,
            args.excel2_reason_col,
        ],
    )

    out_df = match_codes(
        df_source=df1,
        df_lookup=df2,
        source_desc_col=args.excel1_desc_col,
        source_desc_fallback_col=fallback_col,
        lookup_desc_col=args.excel2_desc_col,
        lookup_code_col=args.excel2_code_col,
        lookup_second_pass_col=args.excel2_second_pass_col,
        lookup_reason_col=args.excel2_reason_col,
    )

    output_path = build_output_path(args.excel1, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)

    total = len(out_df)
    matched = int((out_df["匹配材料代码"].astype(str).str.strip() != "").sum())
    conflicted = int((out_df["编码冲突标记"].astype(str).str.strip() == "是").sum())
    fallback_matched = int(out_df.attrs.get("fallback_match_count", 0))

    print(f"源文件总行数: {total}")
    print(f"成功匹配行数: {matched}")
    print(f"兜底描述匹配行数: {fallback_matched}")
    print(f"编码冲突行数: {conflicted}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
