# -*- coding: utf-8 -*-
"""CLI for difficulty splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .difficulty_levels import DIFF_EASY, DIFF_HARD, difficulty_label
from .difficulty_splitter import MaterialDifficultySplitter
from .project_frequency_detector import ProjectFrequencyDetector

# 项目列后续可按实际表头手工修改；列不存在时自动跳过项目低频检测
PROJECT_COLUMN = "项目名称"


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _pick_code(row: pd.Series, primary: str, corrected: str, standardized: str) -> str:
    primary_value = _clean_cell(row.get(primary, ""))
    if primary_value:
        return primary_value
    corrected_value = _clean_cell(row.get(corrected, ""))
    if corrected_value:
        return corrected_value
    return _clean_cell(row.get(standardized, ""))


def _analyze_one(
    splitter: MaterialDifficultySplitter,
    text: str,
    type_code: str = "",
    material_code: str = "",
    standard_code: str = "",
) -> dict:
    result = splitter.analyze(
        text,
        type_code=type_code,
        material_code=material_code,
        standard_code=standard_code,
    ).to_dict()
    result["difficulty"] = DIFF_HARD if result.get("is_difficult") else DIFF_EASY
    result["difficulty_label"] = difficulty_label(result["difficulty"])
    return result


def _format_pretty(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"文本: {result['text']}")
    lines.append(f"结论: {result.get('difficulty_label') or difficulty_label(result.get('difficulty')) or ('困难' if result['is_difficult'] else '简单')}")
    if result["reasons"]:
        lines.append(f"原因: {' | '.join(result['reasons'])}")

    matched_features = [feature for feature in result["features"] if feature["matched"]]
    if not matched_features:
        lines.append("命中: 无")
        return "\n".join(lines)

    lines.append("命中:")
    for idx, feature in enumerate(matched_features, start=1):
        lines.append(f"{idx}. {feature['name']} - {feature['reason']}")
        for hit in feature["hits"]:
            lines.append(f"   - 片段: {hit['token']}")
            lines.append(f"   - code: {hit['code']}")
            lines.append(f"   - 位置: {hit['start']}:{hit['end']}")
            lines.append(f"   - 说明: {hit['note']}")
    return "\n".join(lines)


def _format_feature_hits_inline(feature: dict) -> str:
    if not feature["matched"]:
        return ""

    parts: list[str] = []
    for hit in feature["hits"]:
        token = hit.get("token", "")
        code = hit.get("code", "")
        note = hit.get("note", "")
        fragment = token
        if code:
            fragment = f"{fragment} [{code}]"
        if note:
            fragment = f"{fragment} - {note}"
        parts.append(fragment)
    return " | ".join(parts)


def _build_feature_sheet_rows(text: str, analyzed: dict, row_no: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        "type_glue": [],
        "standard_glue": [],
        "anchor_missing": [],
        "special_token": [],
        "uncommon_code": [],
        "project_frequency": [],
    }
    for feature in analyzed.get("features", []):
        name = feature.get("name", "")
        if name not in out or not feature.get("matched"):
            continue
        for idx, hit in enumerate(feature.get("hits", []), start=1):
            out[name].append(
                {
                    "行号": row_no,
                    "文本": text,
                    "特征": name,
                    "命中序号": idx,
                    "片段": hit.get("token", ""),
                    "编码": hit.get("code", ""),
                    "开始": hit.get("start", ""),
                    "结束": hit.get("end", ""),
                    "说明": hit.get("note", ""),
                }
            )
    return out


def _write_excel_output(
    *,
    output_path: Path,
    detail_df: pd.DataFrame,
    type_glue_df: pd.DataFrame,
    standard_glue_df: pd.DataFrame,
    anchor_missing_df: pd.DataFrame,
    special_token_df: pd.DataFrame,
    uncommon_code_df: pd.DataFrame,
    project_frequency_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="明细", index=False)
        type_glue_df.to_excel(writer, sheet_name="种类粘连", index=False)
        standard_glue_df.to_excel(writer, sheet_name="规范粘连", index=False)
        anchor_missing_df.to_excel(writer, sheet_name="锚点缺失", index=False)
        special_token_df.to_excel(writer, sheet_name="特殊值", index=False)
        uncommon_code_df.to_excel(writer, sheet_name="不常见编码", index=False)
        project_frequency_df.to_excel(writer, sheet_name="项目低频", index=False)


def _run_batch(
    splitter: MaterialDifficultySplitter,
    input_file: str,
    text_column: str,
    output_file: str | None,
) -> None:
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

    results = []
    difficulty_list = []
    is_difficult_list = []
    reasons_list = []
    feature_name_list = []
    pretty_summary_list = []
    type_glue_list = []
    standard_glue_list = []
    anchor_missing_list = []
    special_token_list = []
    type_glue_matched = []
    standard_glue_matched = []
    anchor_missing_matched = []
    special_token_matched = []
    uncommon_code_list = []
    uncommon_code_matched = []
    project_frequency_list = []
    project_frequency_matched = []
    hit_count_list = []
    type_glue_rows: list[dict] = []
    standard_glue_rows: list[dict] = []
    anchor_missing_rows: list[dict] = []
    special_token_rows: list[dict] = []
    uncommon_code_rows: list[dict] = []
    project_frequency_rows: list[dict] = []
    project_rows: list[dict] = []
    analyzed_results: list[dict] = []
    for _, row in df.iterrows():
        text = _clean_cell(row.get(text_column, ""))
        type_code = _pick_code(row, "处理种类", "修正种类", "标准化种类")
        material_code = _pick_code(row, "处理材质", "修正材质", "标准化材质")
        standard_code = _pick_code(row, "处理规范", "修正规范", "标准化规范")
        analyzed = _analyze_one(
            splitter,
            text,
            type_code=type_code,
            material_code=material_code,
            standard_code=standard_code,
        )
        analyzed_results.append(analyzed)
        results.append(json.dumps(analyzed, ensure_ascii=False))
        difficulty_list.append(analyzed["difficulty"])
        is_difficult_list.append(analyzed["is_difficult"])
        reasons_list.append(" | ".join(analyzed["reasons"]))
        feature_name_list.append(" | ".join(feature["name"] for feature in analyzed["features"] if feature["matched"]))
        pretty_summary_list.append(_format_pretty(analyzed))

        feature_map = {feature["name"]: feature for feature in analyzed["features"]}
        type_glue_feature = feature_map.get("type_glue", {"matched": False, "hits": []})
        standard_glue_feature = feature_map.get("standard_glue", {"matched": False, "hits": []})
        anchor_missing_feature = feature_map.get("anchor_missing", {"matched": False, "hits": []})
        special_token_feature = feature_map.get("special_token", {"matched": False, "hits": []})
        uncommon_code_feature = feature_map.get("uncommon_code", {"matched": False, "hits": []})
        type_glue_list.append(_format_feature_hits_inline(type_glue_feature))
        standard_glue_list.append(_format_feature_hits_inline(standard_glue_feature))
        anchor_missing_list.append(_format_feature_hits_inline(anchor_missing_feature))
        special_token_list.append(_format_feature_hits_inline(special_token_feature))
        uncommon_code_list.append(_format_feature_hits_inline(uncommon_code_feature))
        type_glue_matched.append(bool(type_glue_feature.get("matched")))
        standard_glue_matched.append(bool(standard_glue_feature.get("matched")))
        anchor_missing_matched.append(bool(anchor_missing_feature.get("matched")))
        special_token_matched.append(bool(special_token_feature.get("matched")))
        uncommon_code_matched.append(bool(uncommon_code_feature.get("matched")))
        hit_count_list.append(sum(len(feature.get("hits", [])) for feature in analyzed.get("features", [])))
        project_rows.append(
            {
                "project": _clean_cell(row.get(PROJECT_COLUMN, "")) if PROJECT_COLUMN in df.columns else "",
                "type_code": type_code,
                "material_code": material_code,
            }
        )

        feature_rows = _build_feature_sheet_rows(text, analyzed, len(results))
        type_glue_rows.extend(feature_rows["type_glue"])
        standard_glue_rows.extend(feature_rows["standard_glue"])
        anchor_missing_rows.extend(feature_rows["anchor_missing"])
        special_token_rows.extend(feature_rows["special_token"])
        uncommon_code_rows.extend(feature_rows["uncommon_code"])

    project_features = (
        ProjectFrequencyDetector().analyze_rows(project_rows)
        if PROJECT_COLUMN in df.columns
        else []
    )
    for idx, analyzed in enumerate(analyzed_results):
        feature = project_features[idx] if idx < len(project_features) else None
        matched = bool(feature and feature.matched)
        project_frequency_matched.append(matched)
        project_frequency_list.append(
            _format_feature_hits_inline(feature.to_dict()) if matched and feature is not None else ""
        )
        if matched and feature is not None:
            analyzed["is_difficult"] = True
            analyzed["difficulty"] = DIFF_HARD
            analyzed["difficulty_label"] = difficulty_label(DIFF_HARD)
            if feature.reason:
                existing_reasons = list(analyzed.get("reasons", []))
                if feature.reason not in existing_reasons:
                    existing_reasons.append(feature.reason)
                analyzed["reasons"] = existing_reasons
            analyzed.setdefault("features", []).append(feature.to_dict())
            hit_count_list[idx] = int(hit_count_list[idx]) + len(feature.hits)
            existing_feature_names = feature_name_list[idx]
            project_name = feature.name
            feature_name_list[idx] = (
                f"{existing_feature_names} | {project_name}" if existing_feature_names else project_name
            )
            existing_reason_text = reasons_list[idx]
            project_reason_text = _format_feature_hits_inline(feature.to_dict())
            reasons_list[idx] = (
                f"{existing_reason_text} | {project_reason_text}" if existing_reason_text and project_reason_text else (project_reason_text or existing_reason_text)
            )
            difficulty_list[idx] = DIFF_HARD
            is_difficult_list[idx] = True
            text = _clean_cell(df.iloc[idx].get(text_column, ""))
            feature_rows = _build_feature_sheet_rows(text, analyzed, idx + 1)
            project_frequency_rows.extend(feature_rows["project_frequency"])
            results[idx] = json.dumps(analyzed, ensure_ascii=False)
            pretty_summary_list[idx] = _format_pretty(analyzed)

    out_df = df.copy()
    out_df["难度"] = difficulty_list
    out_df["是否困难"] = is_difficult_list
    out_df["命中特征"] = feature_name_list
    out_df["命中数量"] = hit_count_list
    out_df["判定原因"] = reasons_list
    out_df["种类粘连_是否命中"] = type_glue_matched
    out_df["种类粘连_命中详情"] = type_glue_list
    out_df["规范粘连_是否命中"] = standard_glue_matched
    out_df["规范粘连_命中详情"] = standard_glue_list
    out_df["锚点缺失_是否命中"] = anchor_missing_matched
    out_df["锚点缺失_命中详情"] = anchor_missing_list
    out_df["特殊值_是否命中"] = special_token_matched
    out_df["特殊值_命中详情"] = special_token_list
    out_df["不常见编码_是否命中"] = uncommon_code_matched
    out_df["不常见编码_命中详情"] = uncommon_code_list
    out_df["项目低频_是否命中"] = project_frequency_matched
    out_df["项目低频_命中详情"] = project_frequency_list
    out_df["分析摘要"] = pretty_summary_list
    out_df["分析结果JSON"] = results

    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_difficulty_split.xlsx")
    else:
        output_path = Path(output_file)
        if output_path.suffix.lower() == ".csv":
            output_path = output_path.with_suffix(".xlsx")

    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"不支持的输出文件类型: {output_path.suffix}")
    _write_excel_output(
        output_path=output_path,
        detail_df=out_df,
        type_glue_df=pd.DataFrame(type_glue_rows),
        standard_glue_df=pd.DataFrame(standard_glue_rows),
        anchor_missing_df=pd.DataFrame(anchor_missing_rows),
        special_token_df=pd.DataFrame(special_token_rows),
        uncommon_code_df=pd.DataFrame(uncommon_code_rows),
        project_frequency_df=pd.DataFrame(project_frequency_rows),
    )

    print(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="材料描述困难度分流（当前实现种类粘连、规范粘连、锚点缺失）")
    parser.add_argument("--text", help="待分析的单条材料描述")
    parser.add_argument("--input-file", help="批量输入文件，支持 csv/xlsx/xls")
    parser.add_argument("--text-column", default="材料描述", help="批量模式下的描述列名，默认 材料描述")
    parser.add_argument("--output-file", help="批量模式下的输出文件，支持 csv/xlsx/xls")
    parser.add_argument("--type-code", default="", help="单条模式下的种类编码，可选")
    parser.add_argument("--material-code", default="", help="单条模式下的材质编码，可选")
    parser.add_argument("--standard-code", default="", help="单条模式下的规范编码，可选")
    parser.add_argument("--pretty", action="store_true", help="单条模式下输出人工可读摘要")
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("必须提供 --text 或 --input-file")
    if args.text and args.input_file:
        parser.error("--text 与 --input-file 只能二选一")

    splitter = MaterialDifficultySplitter()
    if args.text:
        result = splitter.analyze(
            args.text,
            type_code=args.type_code,
            material_code=args.material_code,
            standard_code=args.standard_code,
        )
        result_dict = result.to_dict()
        if args.pretty:
            print(_format_pretty(result_dict))
        else:
            print(json.dumps(result_dict, ensure_ascii=False, indent=2))
        return

    _run_batch(
        splitter=splitter,
        input_file=args.input_file,
        text_column=args.text_column,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
