# -*- coding: utf-8 -*-
"""读取平台导出的 Excel，按当前系统分流逻辑重算分流最终难度与分流原因。

python scripts/重算Excel分流结果.py /Users/guoxi/Documents/直 管总数据.xlsx

python scripts/重算Excel分流结果.py 你的文件.xlsx -o 输出文件.xlsx

python scripts/重算Excel分流结果.py 你的文件.xlsx --sheet Sheet1

"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.material_description_splitter.platform_integration import build_base_difficulty, finalize_batch_difficulty
from src.material_description_splitter.second_pass import PlatformSecondPassRunner


OUTPUT_DIFFICULTY_HEADER = "分流最终难度（0=困难，2=简单）"
LEGACY_DIFFICULTY_HEADERS = (
    OUTPUT_DIFFICULTY_HEADER,
    "分流最终难度（0=困难，1=中等，2=简单）",
    "分流最终难度（0=困难，1=简单，2=二次简单）",
)
OUTPUT_REASON_HEADER = "分流原因"

PROJECT_COL = "项目名称"
TEXT_COL = "原始描述"
TYPE_CODE_COL = "TYPE_原始编码"
SIZE_RESULT_COL = "SIZE_原始结果"
THICKNESS_RESULT_COL = "THICKNESS_原始结果"
PRESSURE_RESULT_COL = "PRESSURE_原始结果"
MATERIAL_CODE_COL = "MATERIAL_原始编码"
STANDARD_RESULT_COL = "STANDARD_原始结果"
STANDARD_CODE_COL = "STANDARD_原始编码"
MODEL_CONFIDENCE_COLS = ("模型置信分", "excel2_模型置信分")
FINAL_CODE_COLS = ("原始总编码", "excel2_原始总编码")

EMPTY_MARKERS = {"", "—", "-", "nan", "none", "null"}
STANDARD_SPLIT_RE = re.compile(r"\s*[;；]\s*")
STANDARD_ITEM_RE = re.compile(r"^(?P<code>.*?)(?:[（(](?P<category>[^）)]+)[）)])?$")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in EMPTY_MARKERS else text


def parse_confidence(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        score = float(text)
    except (TypeError, ValueError):
        return None
    if is_percent or score > 1:
        score /= 100
    return score if 0 <= score <= 1 else None


def get_existing_column(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def split_segments(text: str) -> list[str]:
    raw = clean_text(text)
    if not raw:
        return []
    return [part.strip() for part in STANDARD_SPLIT_RE.split(raw) if part.strip()]


def parse_standard_items(display_text: Any, code_text: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    for segment in split_segments(clean_text(display_text)):
        match = STANDARD_ITEM_RE.match(segment)
        if not match:
            items.append({"code": segment, "category": ""})
            continue
        code = clean_text(match.group("code"))
        category = clean_text(match.group("category"))
        if code:
            items.append({"code": code, "category": category})

    if items:
        return items

    code_value = clean_text(code_text)
    if code_value:
        return [{"code": code_value, "category": ""}]
    return []


def parse_standard_codes(display_text: Any, code_text: Any) -> list[str]:
    items = parse_standard_items(display_text, code_text)
    codes = [clean_text(item.get("code")) for item in items if clean_text(item.get("code"))]
    return codes


def build_route_reason(*, difficulty_split: dict[str, Any], second_pass: dict[str, Any]) -> str:
    final_level = second_pass.get("final_level")
    stage1_level = second_pass.get("stage1_difficulty")
    if final_level == 0 and stage1_level == 0:
        return clean_text(difficulty_split.get("reason_text")) or "未提供一阶段分流原因"

    results = second_pass.get("results") if isinstance(second_pass.get("results"), dict) else {}
    failed_parts: list[str] = []
    for field in ("SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "TYPE", "STANDARD", "RESULT_SET", "OUTPUT", "CONFIDENCE"):
        payload = results.get(field)
        if not isinstance(payload, dict):
            continue
        if payload.get("passed") is not False:
            continue
        reason = clean_text(payload.get("reason"))
        if reason:
            failed_parts.append(f"{field}: {reason}")
    if failed_parts:
        return " | ".join(failed_parts)

    if final_level == 2:
        return "无需二次分流原因"
    if second_pass.get("final_level") is not None:
        return "未提供二次分流原因"
    return clean_text(difficulty_split.get("reason_text")) or "未提供原因说明"


def build_rows_for_base_difficulty(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        text = clean_text(row.get(TEXT_COL))
        type_code = clean_text(row.get(TYPE_CODE_COL))
        material_code = clean_text(row.get(MATERIAL_CODE_COL))
        standard_result = row.get(STANDARD_RESULT_COL)
        standard_code = row.get(STANDARD_CODE_COL)
        standard_codes = parse_standard_codes(standard_result, standard_code)
        rows.append(
            {
                "text": text,
                "project_name": clean_text(row.get(PROJECT_COL)),
                "type_code": type_code,
                "material_code": material_code,
                "standard_codes": standard_codes,
                "base_difficulty": build_base_difficulty(
                    text,
                    type_code=type_code,
                    material_code=material_code,
                    standard_codes=standard_codes,
                ),
            }
        )
    return rows


def recompute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    runner = PlatformSecondPassRunner()
    result_df = df.copy()
    base_rows = build_rows_for_base_difficulty(result_df)
    finalized_difficulties = finalize_batch_difficulty(base_rows)
    confidence_col = get_existing_column(result_df, MODEL_CONFIDENCE_COLS)
    final_code_col = get_existing_column(result_df, FINAL_CODE_COLS)

    new_levels: list[Any] = []
    new_reasons: list[str] = []

    for idx, (_, row) in enumerate(result_df.iterrows()):
        difficulty_split = finalized_difficulties[idx] if idx < len(finalized_difficulties) else {}
        stage1_level = difficulty_split.get("difficulty")
        confidence = parse_confidence(row.get(confidence_col)) if confidence_col else None
        final_code = clean_text(row.get(final_code_col)) if final_code_col else None
        second_pass = runner.analyze(
            text=clean_text(row.get(TEXT_COL)),
            stage1_difficulty=stage1_level,
            size_value=clean_text(row.get(SIZE_RESULT_COL)),
            thickness_value=clean_text(row.get(THICKNESS_RESULT_COL)),
            pressure_value=clean_text(row.get(PRESSURE_RESULT_COL)),
            material_code=clean_text(row.get(MATERIAL_CODE_COL)),
            type_code=clean_text(row.get(TYPE_CODE_COL)),
            standard_items=parse_standard_items(row.get(STANDARD_RESULT_COL), row.get(STANDARD_CODE_COL)),
            success=bool(final_code) if final_code_col else None,
            final_code=final_code,
            validate_output=bool(final_code_col),
            confidence=confidence,
            confidence_provided=bool(confidence_col),
        )

        new_levels.append(second_pass.get("final_level"))
        new_reasons.append(build_route_reason(difficulty_split=difficulty_split, second_pass=second_pass))

    target_difficulty_col = get_existing_column(result_df, LEGACY_DIFFICULTY_HEADERS)
    if target_difficulty_col is None:
        target_difficulty_col = OUTPUT_DIFFICULTY_HEADER
    if target_difficulty_col != OUTPUT_DIFFICULTY_HEADER and OUTPUT_DIFFICULTY_HEADER not in result_df.columns:
        result_df = result_df.rename(columns={target_difficulty_col: OUTPUT_DIFFICULTY_HEADER})
        target_difficulty_col = OUTPUT_DIFFICULTY_HEADER

    result_df[target_difficulty_col] = new_levels
    result_df[OUTPUT_REASON_HEADER] = new_reasons
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="读取平台导出的 Excel，并按当前系统分流逻辑重算分流结果。")
    parser.add_argument("excel_path", help="输入 Excel 路径")
    parser.add_argument("-o", "--output", help="输出 Excel 路径，默认在原文件名后追加 _重算分流")
    parser.add_argument("--sheet", help="要读取的 sheet 名，默认读取第一个 sheet")
    args = parser.parse_args()

    input_path = Path(args.excel_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    sheet_name: str | int | None = args.sheet if args.sheet else 0
    df = pd.read_excel(input_path, sheet_name=sheet_name)

    required_columns = [
        PROJECT_COL,
        TEXT_COL,
        TYPE_CODE_COL,
        SIZE_RESULT_COL,
        THICKNESS_RESULT_COL,
        PRESSURE_RESULT_COL,
        MATERIAL_CODE_COL,
        STANDARD_RESULT_COL,
        STANDARD_CODE_COL,
    ]
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要列: {', '.join(missing)}")

    result_df = recompute_dataframe(df)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_重算分流.xlsx")
    )
    result_df.to_excel(output_path, index=False)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
