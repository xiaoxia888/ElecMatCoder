# -*- coding: utf-8 -*-
"""CLI for material rule matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .material_rule_matcher import MaterialRuleMatcher
from src.tokenizer_utils.preprocessor import TextPreprocessor


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _format_pretty(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"文本: {result['text']}")
    lines.append(f"规则是否命中: {'是' if result['matched'] else '否'}")
    lines.append(f"命中层级: {result['layer'] or '-'}")
    lines.append(f"候选材质: {' | '.join(result['candidates']) if result['candidates'] else '-'}")
    lines.append(f"是否强制走模型: {'是' if result['force_model'] else '否'}")
    if result["force_model_reasons"]:
        lines.append(f"强制走模型原因: {' | '.join(result['force_model_reasons'])}")

    if not result["hits"]:
        lines.append("命中详情: 无")
        return "\n".join(lines)

    lines.append("命中详情:")
    for idx, hit in enumerate(result["hits"], start=1):
        lines.append(f"{idx}. 层级: {hit['layer']}")
        lines.append(f"   - 目标值: {hit['target']}")
        lines.append(f"   - alias: {hit['alias']}")
        lines.append(f"   - 片段: {hit['text']}")
        lines.append(f"   - 位置: {hit['start']}:{hit['end']}")
    return "\n".join(lines)


def _run_batch(matcher: MaterialRuleMatcher, input_file: str, text_column: str, output_file: str | None) -> None:
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"不支持的输入文件类型: {input_path.suffix}")

    if text_column not in df.columns:
        raise ValueError(f"列 {text_column} 不存在，可用列为: {list(df.columns)}")

    preprocessor = TextPreprocessor()

    layers = []
    candidates = []
    force_model_list = []
    force_model_reasons = []
    detail_json = []
    hit_count = []
    hit_summary = []
    handled_materials = []
    correct_materials = []
    candidate_hit_correct = []
    direct_match_correct = []

    for _, row in df.iterrows():
        value = row.get(text_column, "")
        text = preprocessor.process(_clean_cell(value))
        result = matcher.match(text).to_dict()
        corrected = _clean_cell(row.get("修正材质", ""))
        standardized = _clean_cell(row.get("标准化材质", ""))
        correct_material = corrected or standardized
        layers.append(result["layer"] or "")
        candidates.append(" | ".join(result["candidates"]))
        force_model_list.append(result["force_model"])
        force_model_reasons.append(" | ".join(result["force_model_reasons"]))
        detail_json.append(json.dumps(result, ensure_ascii=False))
        hit_count.append(len(result["hits"]))
        clear_when_force_model = bool(
            getattr(matcher, "rule_policy", {}).get("clear_result_when_force_model", True)
        )
        handled_material = (
            ""
            if result["force_model"] and clear_when_force_model
            else " | ".join(result["candidates"])
        )
        handled_materials.append(handled_material)
        correct_materials.append(correct_material)
        candidate_hit_correct.append(bool(correct_material and correct_material in result["candidates"]))
        direct_match_correct.append(
            bool(
                correct_material
                and not result["force_model"]
                and len(result["candidates"]) == 1
                and result["candidates"][0] == correct_material
            )
        )

        parts: list[str] = []
        for hit in result["hits"]:
            parts.append(
                f"{hit['text']} [{hit['target']}] ({hit['layer']}) alias={hit['alias']}"
            )
        hit_summary.append(" | ".join(parts))

    out_df = df.copy()
    out_df["处理材质"] = handled_materials
    out_df["正确材质"] = correct_materials
    out_df["是否命中正确材质"] = candidate_hit_correct
    out_df["是否可直接作为结果"] = direct_match_correct
    out_df["材质规则_命中层级"] = layers
    out_df["材质规则_候选材质"] = candidates
    out_df["材质规则_是否强制走模型"] = force_model_list
    out_df["材质规则_强制走模型原因"] = force_model_reasons
    out_df["材质规则_命中数量"] = hit_count
    out_df["材质规则_命中详情"] = hit_summary
    out_df["材质规则_JSON"] = detail_json

    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_material_rule_match.xlsx")
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() == ".csv":
            output_path = output_path.with_suffix(".xlsx")

    if output_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"不支持的输出文件类型: {output_path.suffix}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="材质规则匹配", index=False)

    print(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="材质规则匹配测试入口")
    parser.add_argument("--text", help="待分析的单条材料描述")
    parser.add_argument("--input-file", help="批量输入文件，支持 csv/xlsx/xls")
    parser.add_argument("--text-column", default="材料描述", help="批量模式下的描述列名，默认 材料描述")
    parser.add_argument("--output-file", help="批量模式下的输出文件，输出统一为 xlsx")
    parser.add_argument("--pretty", action="store_true", help="单条模式下输出人工可读摘要")
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("必须提供 --text 或 --input-file")
    if args.text and args.input_file:
        parser.error("--text 与 --input-file 只能二选一")

    matcher = MaterialRuleMatcher()
    preprocessor = TextPreprocessor()
    if args.text:
        result = matcher.match(preprocessor.process(args.text)).to_dict()
        if args.pretty:
            print(_format_pretty(result))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _run_batch(
        matcher=matcher,
        input_file=args.input_file,
        text_column=args.text_column,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
    preprocessor = TextPreprocessor()
