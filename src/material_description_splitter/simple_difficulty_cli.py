# -*- coding: utf-8 -*-
"""Simple batch difficulty labeling CLI.

python -m src.material_description_splitter.simple_difficulty_cli \
    --input-file /Users/guoxi/Downloads/B1-AI管道材料编码生成0511-0515.xlsx \
    --output-file /Users/guoxi/Downloads/B1-AI管道材料编码生成0511-0515-res.xlsx

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

from .difficulty_levels import DIFF_EASY, DIFF_HARD, difficulty_label, normalize_difficulty_level
from .difficulty_splitter import MaterialDifficultySplitter
from .models import DifficultyResult
from .project_frequency_detector import ProjectFrequencyDetector

TEXT_COLUMN = "原始描述"
TYPE_CODE_COLUMN = "TYPE_原始编码"
MATERIAL_CODE_COLUMN = "MATERIAL_原始编码"
STANDARD_CODE_COLUMN = "STANDARD_原始结果"
CORRECTNESS_COLUMN = "修正编码"
PROJECT_COLUMN = "项目名称"

# TEXT_COLUMN = "材料描述"
# TYPE_CODE_COLUMN = "子表.标准化种类"
# MATERIAL_CODE_COLUMN = "子表.标准化材质"
# STANDARD_CODE_COLUMN = "子表.原始规范"
# CORRECTNESS_COLUMN = "子表.修正后编码"
# # 项目列后续可按实际表头手工修改；列不存在时自动跳过项目低频检测
# PROJECT_COLUMN = "子表.项目名称"

DIFFICULTY_COLUMN = "难度"
REASON_COLUMN = "原因"
IS_CORRECT_COLUMN = "是否正确"


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的输入文件类型: {path.suffix}")


def _normalize_output_path(input_path: Path, output_file: str | None) -> Path:
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_difficulty.xlsx")
    if output_path.suffix.lower() == ".csv":
        output_path = output_path.with_suffix(".xlsx")
    if output_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"不支持的输出文件类型: {output_path.suffix}")
    return output_path


def _build_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_accuracy.png")


def _build_project_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_project_difficulty.png")


def _build_project_accuracy_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_project_accuracy.png")


def _build_project_total_accuracy_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_project_total_accuracy.png")


def _format_reason(result: DifficultyResult) -> str:
    parts: list[str] = []
    for feature in result.features:
        if not feature.matched:
            continue
        hit_notes = [str(hit.note).strip() for hit in feature.hits if str(hit.note).strip()]
        if hit_notes:
            unique_notes: list[str] = []
            seen: set[str] = set()
            for note in hit_notes:
                if note in seen:
                    continue
                seen.add(note)
                unique_notes.append(note)
            parts.append("；".join(unique_notes))
            continue
        if feature.reason:
            parts.append(feature.reason)
    return " | ".join(parts)


def _save_accuracy_chart(df: pd.DataFrame, chart_path: Path) -> None:
    valid_mask = df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")
    df = df.loc[valid_mask].copy()
    if df.empty:
        return

    summary = (
        df.groupby(DIFFICULTY_COLUMN, dropna=False)[IS_CORRECT_COLUMN]
        .agg(总数="count", 正确数="sum")
        .reset_index()
    )
    summary["正确率"] = summary.apply(
        lambda row: float(row["正确数"]) / float(row["总数"]) if row["总数"] else 0.0,
        axis=1,
    )

    order = [DIFF_EASY, DIFF_HARD]
    summary[DIFFICULTY_COLUMN] = pd.Categorical(summary[DIFFICULTY_COLUMN], categories=order, ordered=True)
    summary = summary.sort_values(DIFFICULTY_COLUMN)

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [difficulty_label(value) for value in summary[DIFFICULTY_COLUMN].tolist()]
    rates = (summary["正确率"] * 100).tolist()
    bars = ax.bar(labels, rates, color=["#4CAF50", "#FF9800"][: len(labels)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title("简单/困难情况下的正确率")

    for bar, rate, total, correct in zip(bars, rates, summary["总数"], summary["正确数"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.2f}%\n({int(correct)}/{int(total)})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _split_reason_tokens(reason_text: str) -> list[str]:
    if not reason_text:
        return []
    tokens = [part.strip() for part in reason_text.split("|")]
    return [token for token in tokens if token]


def _build_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid_df = df[df[DIFFICULTY_COLUMN] != ""].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["原因", "涉及条数", "占困难样本比例(%)"])

    difficult_df = valid_df[valid_df[DIFFICULTY_COLUMN] == DIFF_HARD].copy()
    if difficult_df.empty:
        return pd.DataFrame(columns=["原因", "涉及条数", "占困难样本比例(%)"])

    counter: dict[str, int] = {}
    for reason_text in difficult_df[REASON_COLUMN].fillna("").astype(str):
        tokens = []
        seen: set[str] = set()
        for token in _split_reason_tokens(reason_text):
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1

    summary = pd.DataFrame(
        [
            {
                "原因": reason,
                "涉及条数": count,
                "占困难样本比例(%)": round(count / len(difficult_df) * 100, 2),
            }
            for reason, count in counter.items()
        ]
    )
    if summary.empty:
        return pd.DataFrame(columns=["原因", "涉及条数", "占困难样本比例(%)"])
    return summary.sort_values(["涉及条数", "原因"], ascending=[False, True]).reset_index(drop=True)


def _build_project_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns:
        return pd.DataFrame()

    valid_df = df[df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["项目名称", "总数", "简单数", "困难数", "简单占比(%)", "困难占比(%)"]
        )

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell)
    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].replace("", "未填写项目")

    summary = (
        valid_df.groupby([PROJECT_COLUMN, DIFFICULTY_COLUMN], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if DIFF_EASY not in summary.columns:
        summary[DIFF_EASY] = 0
    if DIFF_HARD not in summary.columns:
        summary[DIFF_HARD] = 0

    summary["总数"] = summary[DIFF_EASY] + summary[DIFF_HARD]
    summary["简单占比(%)"] = summary.apply(
        lambda row: round(row[DIFF_EASY] / row["总数"] * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    summary["困难占比(%)"] = summary.apply(
        lambda row: round(row[DIFF_HARD] / row["总数"] * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    summary = summary.rename(
        columns={
            PROJECT_COLUMN: "项目名称",
            DIFF_EASY: "简单数",
            DIFF_HARD: "困难数",
        }
    )
    return summary.sort_values(["总数", "项目名称"], ascending=[False, True]).reset_index(drop=True)


def _build_project_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or IS_CORRECT_COLUMN not in df.columns:
        return pd.DataFrame()

    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[DIFFICULTY_COLUMN] != "") &
        (df[IS_CORRECT_COLUMN].notna())
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["项目名称", "简单总数", "简单正确数", "简单正确率(%)", "困难总数", "困难正确数", "困难正确率(%)"]
        )

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell)
    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].replace("", "未填写项目")

    pivot = (
        valid_df.groupby([PROJECT_COLUMN, DIFFICULTY_COLUMN], dropna=False)[IS_CORRECT_COLUMN]
        .agg(["count", "sum"])
        .reset_index()
    )
    if pivot.empty:
        return pd.DataFrame(
            columns=["项目名称", "简单总数", "简单正确数", "简单正确率(%)", "困难总数", "困难正确数", "困难正确率(%)"]
        )

    rows: list[dict[str, object]] = []
    for project_name, group in pivot.groupby(PROJECT_COLUMN):
        metrics = {
            "项目名称": project_name,
            "简单总数": 0,
            "简单正确数": 0,
            "简单正确率(%)": 0.0,
            "困难总数": 0,
            "困难正确数": 0,
            "困难正确率(%)": 0.0,
        }
        for _, row in group.iterrows():
            difficulty = normalize_difficulty_level(row[DIFFICULTY_COLUMN])
            total = int(row["count"])
            correct = int(row["sum"])
            if difficulty == DIFF_EASY:
                metrics["简单总数"] = total
                metrics["简单正确数"] = correct
                metrics["简单正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
            elif difficulty == DIFF_HARD:
                metrics["困难总数"] = total
                metrics["困难正确数"] = correct
                metrics["困难正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["简单总数", "困难总数", "项目名称"], ascending=[False, False, True]
    ).reset_index(drop=True)


def _build_project_total_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or IS_CORRECT_COLUMN not in df.columns:
        return pd.DataFrame()

    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[IS_CORRECT_COLUMN].notna())
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["项目名称", "总数", "正确数", "正确率(%)"])

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell)
    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].replace("", "未填写项目")

    summary = (
        valid_df.groupby(PROJECT_COLUMN, dropna=False)[IS_CORRECT_COLUMN]
        .agg(["count", "sum"])
        .reset_index()
        .rename(
            columns={
                PROJECT_COLUMN: "项目名称",
                "count": "总数",
                "sum": "正确数",
            }
        )
    )
    summary["正确率(%)"] = summary.apply(
        lambda row: round(float(row["正确数"]) / float(row["总数"]) * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    return summary.sort_values(["总数", "项目名称"], ascending=[False, True]).reset_index(drop=True)


def _save_project_difficulty_chart(project_summary: pd.DataFrame, chart_path: Path) -> None:
    if project_summary.empty:
        return

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_summary["项目名称"].astype(str).tolist()
    easy_counts = project_summary["简单数"].tolist()
    hard_counts = project_summary["困难数"].tolist()
    totals = project_summary["总数"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    ax.bar(x, easy_counts, color="#4CAF50", label="简单")
    ax.bar(x, hard_counts, bottom=easy_counts, color="#FF9800", label="困难")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("条数")
    ax.set_title("项目维度简单/困难数量分布")
    ax.legend()

    for idx, (easy, hard, total) in enumerate(zip(easy_counts, hard_counts, totals)):
        easy_ratio = round(easy / total * 100, 1) if total else 0.0
        hard_ratio = round(hard / total * 100, 1) if total else 0.0
        ax.text(
            idx,
            total + max(totals) * 0.01,
            f"总{int(total)}\n简{easy_ratio}% / 难{hard_ratio}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_project_accuracy_chart(project_accuracy_summary: pd.DataFrame, chart_path: Path) -> None:
    if project_accuracy_summary.empty:
        return

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_accuracy_summary["项目名称"].astype(str).tolist()
    easy_rates = project_accuracy_summary["简单正确率(%)"].tolist()
    hard_rates = project_accuracy_summary["困难正确率(%)"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], easy_rates, width=width, color="#4CAF50", label="简单")
    ax.bar([i + width / 2 for i in x], hard_rates, width=width, color="#FF9800", label="困难")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title("项目维度简单/困难正确率")
    ax.legend()

    for idx, (easy_rate, hard_rate) in enumerate(zip(easy_rates, hard_rates)):
        ax.text(idx - width / 2, easy_rate + 1, f"{easy_rate:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, hard_rate + 1, f"{hard_rate:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_project_total_accuracy_chart(project_total_accuracy_summary: pd.DataFrame, chart_path: Path) -> None:
    if project_total_accuracy_summary.empty:
        return

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_total_accuracy_summary["项目名称"].astype(str).tolist()
    rates = project_total_accuracy_summary["正确率(%)"].tolist()
    totals = project_total_accuracy_summary["总数"].tolist()
    corrects = project_total_accuracy_summary["正确数"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    bars = ax.bar(x, rates, color="#3B82F6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title("项目维度总体正确率")

    for bar, rate, total, correct in zip(bars, rates, totals, corrects):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.2f}%\n({int(correct)}/{int(total)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def run_batch(input_file: str, output_file: str | None = None) -> tuple[Path, list[Path]]:
    input_path = Path(input_file)
    df = _read_table(input_path)

    required = [TEXT_COLUMN, TYPE_CODE_COLUMN, MATERIAL_CODE_COLUMN, STANDARD_CODE_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，当前列为: {list(df.columns)}")

    splitter = MaterialDifficultySplitter()
    base_results: list[DifficultyResult | None] = []
    project_rows: list[dict[str, str]] = []
    valid_text_mask: list[bool] = []

    for _, row in df.iterrows():
        text = _clean_cell(row.get(TEXT_COLUMN, ""))
        type_code = _clean_cell(row.get(TYPE_CODE_COLUMN, ""))
        material_code = _clean_cell(row.get(MATERIAL_CODE_COLUMN, ""))
        standard_code = _clean_cell(row.get(STANDARD_CODE_COLUMN, ""))
        project_value = _clean_cell(row.get(PROJECT_COLUMN, "")) if PROJECT_COLUMN in df.columns else ""
        if not text:
            base_results.append(None)
            project_rows.append(
                {
                    "project": "",
                    "type_code": "",
                    "material_code": "",
                }
            )
            valid_text_mask.append(False)
            continue
        result = splitter.analyze(
            text,
            type_code=type_code,
            material_code=material_code,
            standard_code=standard_code,
        )
        base_results.append(result)
        project_rows.append(
            {
                "project": project_value,
                "type_code": type_code,
                "material_code": material_code,
            }
        )
        valid_text_mask.append(True)

    project_features = (
        ProjectFrequencyDetector().analyze_rows(project_rows)
        if PROJECT_COLUMN in df.columns
        else []
    )
    difficulties: list[str] = []
    reasons: list[str] = []
    for idx, result in enumerate(base_results):
        if result is None:
            difficulties.append("")
            reasons.append("")
            continue
        project_feature = project_features[idx] if idx < len(project_features) else None
        parts: list[str] = []
        base_reason = _format_reason(result)
        if base_reason:
            parts.append(base_reason)
        if project_feature and project_feature.matched:
            hit_notes = [str(hit.note).strip() for hit in project_feature.hits if str(hit.note).strip()]
            if hit_notes:
                parts.append("；".join(hit_notes))
            elif project_feature.reason:
                parts.append(project_feature.reason)
        difficulties.append(DIFF_HARD if (result.is_difficult or (project_feature and project_feature.matched)) else DIFF_EASY)
        reasons.append(" | ".join(parts))

    out_df = df.copy()
    out_df[DIFFICULTY_COLUMN] = difficulties
    out_df[REASON_COLUMN] = reasons

    output_path = _normalize_output_path(input_path, output_file)
    chart_paths: list[Path] = []
    if CORRECTNESS_COLUMN in out_df.columns:
        out_df[IS_CORRECT_COLUMN] = [
            (_clean_cell(value) == "") if valid else pd.NA
            for value, valid in zip(out_df[CORRECTNESS_COLUMN], valid_text_mask)
        ]

    reason_summary_df = _build_reason_summary(out_df)
    project_summary_df = _build_project_summary(out_df)
    project_accuracy_summary_df = _build_project_accuracy_summary(out_df)
    project_total_accuracy_summary_df = _build_project_total_accuracy_summary(out_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="分流结果", index=False)
        reason_summary_df.to_excel(writer, sheet_name="原因统计", index=False)
        if not project_summary_df.empty:
            project_summary_df.to_excel(writer, sheet_name="项目统计", index=False)
        if not project_accuracy_summary_df.empty:
            project_accuracy_summary_df.to_excel(writer, sheet_name="项目正确率", index=False)
        if not project_total_accuracy_summary_df.empty:
            project_total_accuracy_summary_df.to_excel(writer, sheet_name="项目总体正确率", index=False)

    if CORRECTNESS_COLUMN in out_df.columns:
        accuracy_chart_path = _build_chart_path(output_path)
        _save_accuracy_chart(out_df, accuracy_chart_path)
        chart_paths.append(accuracy_chart_path)

    if not project_summary_df.empty:
        project_chart_path = _build_project_chart_path(output_path)
        _save_project_difficulty_chart(project_summary_df, project_chart_path)
        chart_paths.append(project_chart_path)

    if not project_accuracy_summary_df.empty:
        project_accuracy_chart_path = _build_project_accuracy_chart_path(output_path)
        _save_project_accuracy_chart(project_accuracy_summary_df, project_accuracy_chart_path)
        chart_paths.append(project_accuracy_chart_path)

    if not project_total_accuracy_summary_df.empty:
        project_total_accuracy_chart_path = _build_project_total_accuracy_chart_path(output_path)
        _save_project_total_accuracy_chart(project_total_accuracy_summary_df, project_total_accuracy_chart_path)
        chart_paths.append(project_total_accuracy_chart_path)

    return output_path, chart_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="极简分流测试：固定列名读取材料描述、种类编码、材质编码，只输出难度和原因")
    parser.add_argument("--input-file", required=True, help="输入文件，支持 csv/xlsx/xls")
    parser.add_argument("--output-file", help="输出文件，默认 *_difficulty.xlsx")
    args = parser.parse_args()

    output_path, chart_paths = run_batch(args.input_file, args.output_file)
    print(str(output_path))
    for chart_path in chart_paths:
        print(str(chart_path))


if __name__ == "__main__":
    main()
