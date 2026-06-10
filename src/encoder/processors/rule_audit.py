from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .rule_extraction import (
    build_structured_rule_entities,
    extract_size_and_thickness_by_rules,
)
from .pressure_processor import PressureProcessor
from .size_processor import SizeProcessor
from .thickness_processor import ThicknessProcessor
from src.tokenizer_utils.preprocessor import TextPreprocessor


DEFAULT_TEXT_COLUMNS = ("材料描述", "材料描述(多行)")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的输入文件格式: {path.suffix}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"不支持的输出文件格式: {path.suffix}")


def _pick_text_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for col in DEFAULT_TEXT_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"未找到描述列，可选列：{', '.join(df.columns)}")


def _pick_truth(row: pd.Series, corrected_col: str, normalized_col: str) -> str:
    corrected_raw = row.get(corrected_col)
    normalized_raw = row.get(normalized_col)
    corrected = "" if pd.isna(corrected_raw) else str(corrected_raw).strip()
    if corrected:
        return corrected
    return "" if pd.isna(normalized_raw) else str(normalized_raw).strip()


def build_rule_audit_excel(
    input_excel: str | Path,
    output_excel: str | Path,
    text_column: Optional[str] = None,
) -> Path:
    input_path = Path(input_excel)
    output_path = Path(output_excel)

    df = _read_table(input_path)
    desc_col = _pick_text_column(df, text_column)

    size_processor = SizeProcessor()
    thickness_processor = ThicknessProcessor(enable_rule_layered=False)
    pressure_processor = PressureProcessor()
    preprocessor = TextPreprocessor()

    records = []
    for _, row in df.iterrows():
        text = str(row.get(desc_col) or "").strip()
        processed_text = preprocessor.process(text)
        result = extract_size_and_thickness_by_rules(
            processed_text,
            size_processor=size_processor,
            thickness_processor=thickness_processor,
            pressure_processor=pressure_processor,
        )
        size_result = result.size
        thickness_result = result.thickness
        pressure_result = result.pressure
        structured = build_structured_rule_entities(result, original_text=processed_text)
        encoded_size = size_processor.process(structured["SIZE"], original_text=processed_text)
        encoded_thickness = thickness_processor.process(structured["THICKNESS"], original_text=processed_text)
        encoded_pressure = pressure_processor.process(structured["PRESSURE"])

        records.append({
            "材料描述": text,
            "格式化描述": processed_text,
            "尺寸规则命中": " | ".join(size_result.matched_texts),
            "处理尺寸": encoded_size,
            "正确尺寸": _pick_truth(row, "修正尺寸", "标准化尺寸"),
            "壁厚规则命中": " | ".join(thickness_result.matched_texts),
            "处理壁厚": encoded_thickness,
            "正确壁厚": _pick_truth(row, "修正壁厚", "标准化壁厚"),
            "磅级规则命中": " | ".join(pressure_result.matched_texts),
            "处理磅级": encoded_pressure,
            "正确磅级": _pick_truth(row, "修正磅级", "标准化磅级"),
            "尺寸是否一致": "是" if encoded_size == _pick_truth(row, "修正尺寸", "标准化尺寸") else "否",
            "壁厚是否一致": "是" if encoded_thickness == _pick_truth(row, "修正壁厚", "标准化壁厚") else "否",
            "磅级是否一致": "是" if encoded_pressure == _pick_truth(row, "修正磅级", "标准化磅级") else "否",
            "尺寸JSON数据": json.dumps(structured["SIZE"], ensure_ascii=False),
            "壁厚JSON数据": json.dumps(structured["THICKNESS"], ensure_ascii=False),
            "磅级JSON数据": json.dumps({"PRESSURE": structured["PRESSURE"]}, ensure_ascii=False),
        })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(out_df, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="基于规则的尺寸/壁厚/磅级批量对照导出")
    parser.add_argument("--input-excel", required=True, help="输入 Excel 路径")
    parser.add_argument("--output-excel", required=True, help="输出 Excel 路径")
    parser.add_argument("--text-column", default=None, help="描述列名，默认自动选择 材料描述/材料描述(多行)")
    args = parser.parse_args()

    output = build_rule_audit_excel(
        input_excel=args.input_excel,
        output_excel=args.output_excel,
        text_column=args.text_column,
    )
    print(output)


if __name__ == "__main__":
    main()
