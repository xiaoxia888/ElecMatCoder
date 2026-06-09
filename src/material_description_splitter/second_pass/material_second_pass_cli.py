# -*- coding: utf-8 -*-
"""CLI for second-pass auto-pass verification.

python -m src.material_description_splitter.second_pass.material_second_pass_cli \
    --input-file /path/to/input.xlsx \
    --output-file /path/to/output.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

from ..difficulty_levels import DIFF_EASY, DIFF_HARD, DIFF_SECOND_EASY, difficulty_label, normalize_difficulty_level
from .material_second_pass_splitter import MaterialSecondPassSplitter
from .pressure_second_pass_splitter import PressureSecondPassSplitter
from .size_second_pass_splitter import SizeSecondPassSplitter
from .standard_second_pass_splitter import StandardSecondPassSplitter
from .thickness_second_pass_splitter import ThicknessSecondPassSplitter
from .type_second_pass_splitter import TypeSecondPassSplitter

TEXT_COLUMN = "原始描述"
MATERIAL_CODE_COLUMN = "MATERIAL_原始编码"
TYPE_CODE_COLUMN = "TYPE_原始编码"
STANDARD_RESULT_COLUMN = "STANDARD_原始结果"
STANDARD_CODE_COLUMN = "STANDARD_原始编码"
SIZE_RESULT_COLUMN = "SIZE_原始结果"
SIZE_CODE_COLUMN = "SIZE_原始编码"
THICKNESS_RESULT_COLUMN = "THICKNESS_原始结果"
THICKNESS_CODE_COLUMN = "THICKNESS_原始编码"
PRESSURE_RESULT_COLUMN = "PRESSURE_原始结果"
PRESSURE_CODE_COLUMN = "PRESSURE_原始编码"
DIFFICULTY_COLUMN = "难度"
REASON_COLUMN = "原因"
PROJECT_COLUMN = "项目名称"
CORRECTNESS_COLUMN = "修正编码"
IS_CORRECT_COLUMN = "是否正确"

SECOND_PASS_COLUMN = "材质二次分流"
SECOND_PASS_REASON_COLUMN = "材质二次分流原因"
SECOND_PASS_BASE_CODE_COLUMN = "材质二次主材编码"
SECOND_PASS_SUFFIX_CODE_COLUMN = "材质二次后缀编码"
SECOND_PASS_BASE_HITS_COLUMN = "材质二次主材命中"
SECOND_PASS_SUFFIX_HITS_COLUMN = "材质二次后缀命中"
SECOND_PASS_CONFLICT_COLUMN = "材质二次冲突编码"
SECOND_PASS_SKIP_COLUMN = "材质二次跳过原因"

TYPE_SECOND_PASS_COLUMN = "种类二次分流"
TYPE_SECOND_PASS_REASON_COLUMN = "种类二次分流原因"
TYPE_SECOND_PASS_PATH_COLUMN = "种类二次命中路径"
TYPE_SECOND_PASS_DIRECT_HITS_COLUMN = "种类二次直接命中"
TYPE_SECOND_PASS_BODY_HITS_COLUMN = "种类二次主体命中"
TYPE_SECOND_PASS_MANU_HITS_COLUMN = "种类二次工艺命中"
TYPE_SECOND_PASS_CONN_HITS_COLUMN = "种类二次连接命中"
TYPE_SECOND_PASS_SEAL_HITS_COLUMN = "种类二次密封面命中"
TYPE_SECOND_PASS_ANGLE_HITS_COLUMN = "种类二次角度命中"
TYPE_SECOND_PASS_RADIUS_HITS_COLUMN = "种类二次半径命中"
TYPE_SECOND_PASS_BLOCK_COLUMN = "种类二次阻塞编码"
TYPE_SECOND_PASS_SKIP_COLUMN = "种类二次跳过原因"

STANDARD_SECOND_PASS_COLUMN = "规范二次分流"
STANDARD_SECOND_PASS_REASON_COLUMN = "规范二次分流原因"
STANDARD_SECOND_PASS_CODES_COLUMN = "规范二次编码集合"
STANDARD_SECOND_PASS_BASE_HITS_COLUMN = "规范二次主体命中"
STANDARD_SECOND_PASS_PREFIX_STATUS_COLUMN = "规范二次前缀状态"
STANDARD_SECOND_PASS_SUFFIX_HITS_COLUMN = "规范二次后缀命中"
STANDARD_SECOND_PASS_SUSPICIOUS_SUFFIX_COLUMN = "规范二次疑似残留后缀"
STANDARD_SECOND_PASS_SKIP_COLUMN = "规范二次跳过原因"

SIZE_SECOND_PASS_COLUMN = "尺寸二次分流"
SIZE_SECOND_PASS_REASON_COLUMN = "尺寸二次分流原因"
SIZE_SECOND_PASS_ANCHORED_HITS_COLUMN = "尺寸二次锚点命中"
SIZE_SECOND_PASS_FALLBACK_HITS_COLUMN = "尺寸二次兜底命中"
SIZE_SECOND_PASS_CONSUMED_SPANS_COLUMN = "尺寸二次消费位置"
SIZE_SECOND_PASS_SKIP_COLUMN = "尺寸二次跳过原因"

THICKNESS_SECOND_PASS_COLUMN = "壁厚二次分流"
THICKNESS_SECOND_PASS_REASON_COLUMN = "壁厚二次分流原因"
THICKNESS_SECOND_PASS_ANCHORED_HITS_COLUMN = "壁厚二次锚点命中"
THICKNESS_SECOND_PASS_FALLBACK_HITS_COLUMN = "壁厚二次兜底命中"
THICKNESS_SECOND_PASS_CONSUMED_SPANS_COLUMN = "壁厚二次消费位置"
THICKNESS_SECOND_PASS_SKIP_COLUMN = "壁厚二次跳过原因"

PRESSURE_SECOND_PASS_COLUMN = "磅级二次分流"
PRESSURE_SECOND_PASS_REASON_COLUMN = "磅级二次分流原因"
PRESSURE_SECOND_PASS_HITS_COLUMN = "磅级二次命中"
PRESSURE_SECOND_PASS_CONSUMED_SPANS_COLUMN = "磅级二次消费位置"
PRESSURE_SECOND_PASS_SKIP_COLUMN = "磅级二次跳过原因"
FINAL_SECOND_PASS_LEVEL_COLUMN = "二次分流最终难度"


def _build_chart_path(output_path: Path, prefix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{prefix}_accuracy.png")


def _build_project_chart_path(output_path: Path, prefix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{prefix}_project_distribution.png")


def _build_project_accuracy_chart_path(output_path: Path, prefix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{prefix}_project_accuracy.png")


def _build_project_total_accuracy_chart_path(output_path: Path, prefix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{prefix}_project_total_accuracy.png")


def _build_final_accuracy_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_final_accuracy.png")


def _build_final_project_distribution_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_final_project_distribution.png")


def _build_final_project_accuracy_chart_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_final_project_accuracy.png")


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
        output_path = input_path.with_name(f"{input_path.stem}_second_pass.xlsx")
    if output_path.suffix.lower() == ".csv":
        output_path = output_path.with_suffix(".xlsx")
    if output_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"不支持的输出文件类型: {output_path.suffix}")
    return output_path


def _join_hit_aliases(hits: list[dict]) -> str:
    aliases: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        alias = _clean_cell(hit.get("alias", ""))
        if not alias or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return " | ".join(aliases)


def _format_spans(spans: list[tuple[int, int]]) -> str:
    if not spans:
        return ""
    return " | ".join(f"{start}:{end}" for start, end in spans)


def _build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(detail_df)
    rows = [
        {"指标": "总行数", "值": total_rows},
    ]
    summary_specs = [
        (
            "尺寸",
            SIZE_SECOND_PASS_COLUMN,
            SIZE_SECOND_PASS_REASON_COLUMN,
            SIZE_SECOND_PASS_SKIP_COLUMN,
        ),
        (
            "壁厚",
            THICKNESS_SECOND_PASS_COLUMN,
            THICKNESS_SECOND_PASS_REASON_COLUMN,
            THICKNESS_SECOND_PASS_SKIP_COLUMN,
        ),
        (
            "磅级",
            PRESSURE_SECOND_PASS_COLUMN,
            PRESSURE_SECOND_PASS_REASON_COLUMN,
            PRESSURE_SECOND_PASS_SKIP_COLUMN,
        ),
        (
            "材质",
            SECOND_PASS_COLUMN,
            SECOND_PASS_REASON_COLUMN,
            SECOND_PASS_SKIP_COLUMN,
        ),
        (
            "种类",
            TYPE_SECOND_PASS_COLUMN,
            TYPE_SECOND_PASS_REASON_COLUMN,
            TYPE_SECOND_PASS_SKIP_COLUMN,
        ),
        (
            "规范",
            STANDARD_SECOND_PASS_COLUMN,
            STANDARD_SECOND_PASS_REASON_COLUMN,
            STANDARD_SECOND_PASS_SKIP_COLUMN,
        ),
    ]
    for label, result_col, reason_col, skip_col in summary_specs:
        processed_mask = detail_df[skip_col].apply(_clean_cell) == ""
        processed_df = detail_df.loc[processed_mask].copy()
        passed_df = processed_df[processed_df[result_col] == "通过"]
        failed_df = processed_df[processed_df[result_col] == "不通过"]
        rows.extend(
            [
                {"指标": f"{label}进入二次分流行数", "值": len(processed_df)},
                {"指标": f"{label}二次通过行数", "值": len(passed_df)},
                {"指标": f"{label}二次不通过行数", "值": len(failed_df)},
                {
                    "指标": f"{label}二次通过率(%)",
                    "值": round(len(passed_df) / len(processed_df) * 100, 2) if len(processed_df) else 0.0,
                },
            ]
        )

        skip_counter = (
            detail_df[skip_col]
            .apply(_clean_cell)
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )
        for reason, count in skip_counter.items():
            rows.append({"指标": f"{label}跳过: {reason}", "值": int(count)})

        fail_counter = (
            failed_df[reason_col]
            .apply(_clean_cell)
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )
        for reason, count in fail_counter.items():
            rows.append({"指标": f"{label}不通过: {reason}", "值": int(count)})

    return pd.DataFrame(rows)


def _build_processed_only(detail_df: pd.DataFrame) -> pd.DataFrame:
    size_processed = detail_df[SIZE_SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    thickness_processed = detail_df[THICKNESS_SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    pressure_processed = detail_df[PRESSURE_SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    material_processed = detail_df[SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    type_processed = detail_df[TYPE_SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    standard_processed = detail_df[STANDARD_SECOND_PASS_SKIP_COLUMN].apply(_clean_cell) == ""
    processed_mask = (
        size_processed
        | thickness_processed
        | pressure_processed
        | material_processed
        | type_processed
        | standard_processed
    )
    return detail_df.loc[processed_mask].copy()


def _build_final_second_pass_level(detail_df: pd.DataFrame) -> list[str]:
    levels: list[int | str] = []
    for _, row in detail_df.iterrows():
        text = _clean_cell(row.get(TEXT_COLUMN, ""))
        difficulty = normalize_difficulty_level(row.get(DIFFICULTY_COLUMN, ""))
        size_result = _clean_cell(row.get(SIZE_RESULT_COLUMN, ""))
        thickness_result = _clean_cell(row.get(THICKNESS_RESULT_COLUMN, ""))
        pressure_result = _clean_cell(row.get(PRESSURE_RESULT_COLUMN, ""))
        size_pass = _clean_cell(row.get(SIZE_SECOND_PASS_COLUMN, ""))
        thickness_pass = _clean_cell(row.get(THICKNESS_SECOND_PASS_COLUMN, ""))
        pressure_pass = _clean_cell(row.get(PRESSURE_SECOND_PASS_COLUMN, ""))
        material_result = _clean_cell(row.get(SECOND_PASS_COLUMN, ""))
        type_result = _clean_cell(row.get(TYPE_SECOND_PASS_COLUMN, ""))
        standard_result = _clean_cell(row.get(STANDARD_SECOND_PASS_COLUMN, ""))
        if not text:
            levels.append("")
            continue
        if difficulty is not None and difficulty != DIFF_EASY:
            levels.append(DIFF_HARD)
            continue
        standard_value = _clean_cell(row.get(STANDARD_RESULT_COLUMN, "")) or _clean_cell(row.get(STANDARD_CODE_COLUMN, ""))
        required_presence_ok = bool(
            _clean_cell(row.get(TYPE_CODE_COLUMN, ""))
            and size_result
            and _clean_cell(row.get(MATERIAL_CODE_COLUMN, ""))
            and standard_value
        )
        thickness_or_pressure_present = bool(thickness_result or pressure_result)
        if not (required_presence_ok and thickness_or_pressure_present):
            levels.append(DIFF_EASY)
            continue
        checks = []
        if size_result:
            checks.append(size_pass == "通过")
        if thickness_result:
            checks.append(thickness_pass == "通过")
        if pressure_result:
            checks.append(pressure_pass == "通过")
        if material_result:
            checks.append(material_result == "通过")
        if type_result:
            checks.append(type_result == "通过")
        if standard_result:
            checks.append(standard_result == "通过")
        if checks and all(checks):
            levels.append(DIFF_SECOND_EASY)
        else:
            levels.append(DIFF_EASY)
    return levels


def _save_final_accuracy_chart(df: pd.DataFrame, chart_path: Path) -> None:
    if IS_CORRECT_COLUMN not in df.columns or FINAL_SECOND_PASS_LEVEL_COLUMN not in df.columns:
        return
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[FINAL_SECOND_PASS_LEVEL_COLUMN].apply(_clean_cell) != "") &
        (df[IS_CORRECT_COLUMN].notna())
    ].copy()
    if valid_df.empty:
        return

    summary = (
        valid_df.groupby(FINAL_SECOND_PASS_LEVEL_COLUMN, dropna=False)[IS_CORRECT_COLUMN]
        .agg(总数="count", 正确数="sum")
        .reset_index()
    )
    summary["正确率"] = summary.apply(
        lambda row: float(row["正确数"]) / float(row["总数"]) if row["总数"] else 0.0,
        axis=1,
    )
    order = [DIFF_SECOND_EASY, DIFF_EASY, DIFF_HARD]
    summary[FINAL_SECOND_PASS_LEVEL_COLUMN] = pd.Categorical(
        summary[FINAL_SECOND_PASS_LEVEL_COLUMN], categories=order, ordered=True
    )
    summary = summary.sort_values(FINAL_SECOND_PASS_LEVEL_COLUMN)

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [difficulty_label(value) for value in summary[FINAL_SECOND_PASS_LEVEL_COLUMN].tolist()]
    rates = (summary["正确率"] * 100).tolist()
    bars = ax.bar(labels, rates, color=["#2563EB", "#4CAF50", "#FF9800"][: len(labels)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title("二次分流最终难度下的正确率")

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


def _build_final_project_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or FINAL_SECOND_PASS_LEVEL_COLUMN not in df.columns:
        return pd.DataFrame()
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[FINAL_SECOND_PASS_LEVEL_COLUMN].apply(_clean_cell) != "")
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["项目名称", "总数", "二次简单数", "简单数", "困难数", "二次简单占比(%)", "简单占比(%)", "困难占比(%)"])

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell).replace("", "未填写项目")
    summary = (
        valid_df.groupby([PROJECT_COLUMN, FINAL_SECOND_PASS_LEVEL_COLUMN], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in [DIFF_SECOND_EASY, DIFF_EASY, DIFF_HARD]:
        if col not in summary.columns:
            summary[col] = 0
    summary["总数"] = summary[DIFF_SECOND_EASY] + summary[DIFF_EASY] + summary[DIFF_HARD]
    summary["二次简单占比(%)"] = summary.apply(lambda row: round(row[DIFF_SECOND_EASY] / row["总数"] * 100, 2) if row["总数"] else 0.0, axis=1)
    summary["简单占比(%)"] = summary.apply(lambda row: round(row[DIFF_EASY] / row["总数"] * 100, 2) if row["总数"] else 0.0, axis=1)
    summary["困难占比(%)"] = summary.apply(lambda row: round(row[DIFF_HARD] / row["总数"] * 100, 2) if row["总数"] else 0.0, axis=1)
    summary = summary.rename(columns={PROJECT_COLUMN: "项目名称", DIFF_SECOND_EASY: "二次简单数", DIFF_EASY: "简单数", DIFF_HARD: "困难数"})
    return summary.sort_values(["总数", "项目名称"], ascending=[False, True]).reset_index(drop=True)


def _build_final_project_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or FINAL_SECOND_PASS_LEVEL_COLUMN not in df.columns or IS_CORRECT_COLUMN not in df.columns:
        return pd.DataFrame()
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[FINAL_SECOND_PASS_LEVEL_COLUMN].apply(_clean_cell) != "") &
        (df[IS_CORRECT_COLUMN].notna())
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["项目名称", "二次简单正确率(%)", "简单正确率(%)", "困难正确率(%)"])

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell).replace("", "未填写项目")
    pivot = (
        valid_df.groupby([PROJECT_COLUMN, FINAL_SECOND_PASS_LEVEL_COLUMN], dropna=False)[IS_CORRECT_COLUMN]
        .agg(["count", "sum"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for project_name, group in pivot.groupby(PROJECT_COLUMN):
        metrics = {
            "项目名称": project_name,
            "二次简单总数": 0, "二次简单正确数": 0, "二次简单正确率(%)": 0.0,
            "简单总数": 0, "简单正确数": 0, "简单正确率(%)": 0.0,
            "困难总数": 0, "困难正确数": 0, "困难正确率(%)": 0.0,
        }
        for _, row in group.iterrows():
            level = normalize_difficulty_level(row[FINAL_SECOND_PASS_LEVEL_COLUMN])
            total = int(row["count"])
            correct = int(row["sum"])
            if level == DIFF_SECOND_EASY:
                metrics["二次简单总数"] = total
                metrics["二次简单正确数"] = correct
                metrics["二次简单正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
            elif level == DIFF_EASY:
                metrics["简单总数"] = total
                metrics["简单正确数"] = correct
                metrics["简单正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
            elif level == DIFF_HARD:
                metrics["困难总数"] = total
                metrics["困难正确数"] = correct
                metrics["困难正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["二次简单总数", "简单总数", "困难总数", "项目名称"], ascending=[False, False, False, True]).reset_index(drop=True)


def _save_final_project_distribution_chart(project_summary: pd.DataFrame, chart_path: Path) -> None:
    if project_summary.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_summary["项目名称"].astype(str).tolist()
    second_easy = project_summary["二次简单数"].tolist()
    easy = project_summary["简单数"].tolist()
    hard = project_summary["困难数"].tolist()
    totals = project_summary["总数"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    ax.bar(x, second_easy, color="#2563EB", label="二次简单")
    ax.bar(x, easy, bottom=second_easy, color="#4CAF50", label="简单")
    ax.bar(x, hard, bottom=[a + b for a, b in zip(second_easy, easy)], color="#FF9800", label="困难")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("条数")
    ax.set_title("项目维度二次简单/简单/困难数量分布")
    ax.legend()

    for idx, (s2, s1, h, total) in enumerate(zip(second_easy, easy, hard, totals)):
        s2_ratio = round(s2 / total * 100, 1) if total else 0.0
        s1_ratio = round(s1 / total * 100, 1) if total else 0.0
        h_ratio = round(h / total * 100, 1) if total else 0.0
        ax.text(
            idx,
            total + max(totals) * 0.01,
            f"总{int(total)}\n二简{s2_ratio}% / 简{s1_ratio}% / 难{h_ratio}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_final_project_accuracy_chart(project_accuracy_summary: pd.DataFrame, chart_path: Path) -> None:
    if project_accuracy_summary.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_accuracy_summary["项目名称"].astype(str).tolist()
    second_easy_rates = project_accuracy_summary["二次简单正确率(%)"].tolist()
    easy_rates = project_accuracy_summary["简单正确率(%)"].tolist()
    hard_rates = project_accuracy_summary["困难正确率(%)"].tolist()

    fig_width = max(10, min(18, len(labels) * 1.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    width = 0.25
    ax.bar([i - width for i in x], second_easy_rates, width=width, color="#2563EB", label="二次简单")
    ax.bar(x, easy_rates, width=width, color="#4CAF50", label="简单")
    ax.bar([i + width for i in x], hard_rates, width=width, color="#FF9800", label="困难")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title("项目维度二次简单/简单/困难正确率")
    ax.legend()

    for idx, (r2, r1, rh) in enumerate(zip(second_easy_rates, easy_rates, hard_rates)):
        ax.text(idx - width, r2 + 1, f"{r2:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(idx, r1 + 1, f"{r1:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width, rh + 1, f"{rh:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_pass_accuracy_chart(df: pd.DataFrame, result_col: str, skip_col: str, chart_path: Path, title: str) -> None:
    if IS_CORRECT_COLUMN not in df.columns:
        return
    valid_mask = (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) & (df[skip_col].apply(_clean_cell) == "")
    plot_df = df.loc[valid_mask].copy()
    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(result_col, dropna=False)[IS_CORRECT_COLUMN]
        .agg(总数="count", 正确数="sum")
        .reset_index()
    )
    summary["正确率"] = summary.apply(
        lambda row: float(row["正确数"]) / float(row["总数"]) if row["总数"] else 0.0,
        axis=1,
    )

    order = ["通过", "不通过"]
    summary[result_col] = pd.Categorical(summary[result_col], categories=order, ordered=True)
    summary = summary.sort_values(result_col)

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = summary[result_col].astype(str).tolist()
    rates = (summary["正确率"] * 100).tolist()
    bars = ax.bar(labels, rates, color=["#4CAF50", "#FF9800"][: len(labels)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title(title)

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


def _build_project_pass_summary(df: pd.DataFrame, result_col: str, skip_col: str) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns:
        return pd.DataFrame()
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[skip_col].apply(_clean_cell) == "")
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=["项目名称", "总数", "通过数", "不通过数", "通过占比(%)", "不通过占比(%)"])

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell)
    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].replace("", "未填写项目")

    summary = (
        valid_df.groupby([PROJECT_COLUMN, result_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "通过" not in summary.columns:
        summary["通过"] = 0
    if "不通过" not in summary.columns:
        summary["不通过"] = 0
    summary["总数"] = summary["通过"] + summary["不通过"]
    summary["通过占比(%)"] = summary.apply(
        lambda row: round(row["通过"] / row["总数"] * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    summary["不通过占比(%)"] = summary.apply(
        lambda row: round(row["不通过"] / row["总数"] * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    summary = summary.rename(
        columns={PROJECT_COLUMN: "项目名称", "通过": "通过数", "不通过": "不通过数"}
    )
    return summary.sort_values(["总数", "项目名称"], ascending=[False, True]).reset_index(drop=True)


def _build_project_pass_accuracy_summary(df: pd.DataFrame, result_col: str, skip_col: str) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or IS_CORRECT_COLUMN not in df.columns:
        return pd.DataFrame()
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[skip_col].apply(_clean_cell) == "") &
        (df[IS_CORRECT_COLUMN].notna())
    ].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["项目名称", "通过总数", "通过正确数", "通过正确率(%)", "不通过总数", "不通过正确数", "不通过正确率(%)"]
        )

    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].apply(_clean_cell)
    valid_df[PROJECT_COLUMN] = valid_df[PROJECT_COLUMN].replace("", "未填写项目")
    pivot = (
        valid_df.groupby([PROJECT_COLUMN, result_col], dropna=False)[IS_CORRECT_COLUMN]
        .agg(["count", "sum"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for project_name, group in pivot.groupby(PROJECT_COLUMN):
        metrics = {
            "项目名称": project_name,
            "通过总数": 0,
            "通过正确数": 0,
            "通过正确率(%)": 0.0,
            "不通过总数": 0,
            "不通过正确数": 0,
            "不通过正确率(%)": 0.0,
        }
        for _, row in group.iterrows():
            result = str(row[result_col])
            total = int(row["count"])
            correct = int(row["sum"])
            if result == "通过":
                metrics["通过总数"] = total
                metrics["通过正确数"] = correct
                metrics["通过正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
            elif result == "不通过":
                metrics["不通过总数"] = total
                metrics["不通过正确数"] = correct
                metrics["不通过正确率(%)"] = round(correct / total * 100, 2) if total else 0.0
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["通过总数", "不通过总数", "项目名称"], ascending=[False, False, True]).reset_index(drop=True)


def _build_project_total_accuracy_summary(df: pd.DataFrame, skip_col: str) -> pd.DataFrame:
    if PROJECT_COLUMN not in df.columns or IS_CORRECT_COLUMN not in df.columns:
        return pd.DataFrame()
    valid_df = df[
        (df[TEXT_COLUMN].apply(lambda v: _clean_cell(v) != "")) &
        (df[skip_col].apply(_clean_cell) == "") &
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
        .rename(columns={PROJECT_COLUMN: "项目名称", "count": "总数", "sum": "正确数"})
    )
    summary["正确率(%)"] = summary.apply(
        lambda row: round(float(row["正确数"]) / float(row["总数"]) * 100, 2) if row["总数"] else 0.0,
        axis=1,
    )
    return summary.sort_values(["总数", "项目名称"], ascending=[False, True]).reset_index(drop=True)


def _save_project_distribution_chart(project_summary: pd.DataFrame, chart_path: Path, title: str) -> None:
    if project_summary.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_summary["项目名称"].astype(str).tolist()
    pass_counts = project_summary["通过数"].tolist()
    fail_counts = project_summary["不通过数"].tolist()
    totals = project_summary["总数"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    ax.bar(x, pass_counts, color="#4CAF50", label="通过")
    ax.bar(x, fail_counts, bottom=pass_counts, color="#FF9800", label="不通过")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("条数")
    ax.set_title(title)
    ax.legend()

    for idx, (passed, failed, total) in enumerate(zip(pass_counts, fail_counts, totals)):
        pass_ratio = round(passed / total * 100, 1) if total else 0.0
        fail_ratio = round(failed / total * 100, 1) if total else 0.0
        ax.text(
            idx,
            total + max(totals) * 0.01,
            f"总{int(total)}\n过{pass_ratio}% / 不{fail_ratio}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_project_accuracy_chart(project_accuracy_summary: pd.DataFrame, chart_path: Path, title: str) -> None:
    if project_accuracy_summary.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = project_accuracy_summary["项目名称"].astype(str).tolist()
    pass_rates = project_accuracy_summary["通过正确率(%)"].tolist()
    fail_rates = project_accuracy_summary["不通过正确率(%)"].tolist()

    fig_width = max(10, min(18, len(labels) * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pass_rates, width=width, color="#4CAF50", label="通过")
    ax.bar([i + width / 2 for i in x], fail_rates, width=width, color="#FF9800", label="不通过")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("正确率(%)")
    ax.set_title(title)
    ax.legend()

    for idx, (pass_rate, fail_rate) in enumerate(zip(pass_rates, fail_rates)):
        ax.text(idx - width / 2, pass_rate + 1, f"{pass_rate:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, fail_rate + 1, f"{fail_rate:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)


def _save_project_total_accuracy_chart(project_total_accuracy_summary: pd.DataFrame, chart_path: Path, title: str) -> None:
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
    ax.set_title(title)

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


def run(input_file: str, output_file: str | None) -> tuple[Path, list[Path]]:
    input_path = Path(input_file)
    output_path = _normalize_output_path(input_path, output_file)
    df = _read_table(input_path)

    required = [TEXT_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}；当前可用列: {list(df.columns)}")

    material_splitter = MaterialSecondPassSplitter()
    type_splitter = TypeSecondPassSplitter()
    standard_splitter = StandardSecondPassSplitter()
    size_splitter = SizeSecondPassSplitter()
    thickness_splitter = ThicknessSecondPassSplitter()
    pressure_splitter = PressureSecondPassSplitter()

    size_second_pass_values: list[str] = []
    size_second_pass_reasons: list[str] = []
    size_anchored_hit_texts: list[str] = []
    size_fallback_hit_texts: list[str] = []
    size_consumed_span_texts: list[str] = []
    size_skip_reasons: list[str] = []

    thickness_second_pass_values: list[str] = []
    thickness_second_pass_reasons: list[str] = []
    thickness_anchored_hit_texts: list[str] = []
    thickness_fallback_hit_texts: list[str] = []
    thickness_consumed_span_texts: list[str] = []
    thickness_skip_reasons: list[str] = []

    pressure_second_pass_values: list[str] = []
    pressure_second_pass_reasons: list[str] = []
    pressure_hit_texts: list[str] = []
    pressure_consumed_span_texts: list[str] = []
    pressure_skip_reasons: list[str] = []

    second_pass_values: list[str] = []
    second_pass_reasons: list[str] = []
    base_codes: list[str] = []
    suffix_codes: list[str] = []
    base_hit_texts: list[str] = []
    suffix_hit_texts: list[str] = []
    conflict_texts: list[str] = []
    skip_reasons: list[str] = []

    type_second_pass_values: list[str] = []
    type_second_pass_reasons: list[str] = []
    type_matched_paths: list[str] = []
    type_direct_hit_texts: list[str] = []
    type_body_hit_texts: list[str] = []
    type_manu_hit_texts: list[str] = []
    type_conn_hit_texts: list[str] = []
    type_seal_hit_texts: list[str] = []
    type_angle_hit_texts: list[str] = []
    type_radius_hit_texts: list[str] = []
    type_blocking_codes: list[str] = []
    type_skip_reasons: list[str] = []

    standard_second_pass_values: list[str] = []
    standard_second_pass_reasons: list[str] = []
    standard_codes_texts: list[str] = []
    standard_base_hit_texts: list[str] = []
    standard_prefix_status_texts: list[str] = []
    standard_suffix_hit_texts: list[str] = []
    standard_suspicious_suffix_texts: list[str] = []
    standard_skip_reasons: list[str] = []

    has_difficulty = DIFFICULTY_COLUMN in df.columns

    for _, row in df.iterrows():
        text = _clean_cell(row.get(TEXT_COLUMN, ""))
        size_result = _clean_cell(row.get(SIZE_RESULT_COLUMN, ""))
        size_code = _clean_cell(row.get(SIZE_CODE_COLUMN, ""))
        thickness_result = _clean_cell(row.get(THICKNESS_RESULT_COLUMN, ""))
        thickness_code = _clean_cell(row.get(THICKNESS_CODE_COLUMN, ""))
        pressure_result = _clean_cell(row.get(PRESSURE_RESULT_COLUMN, ""))
        pressure_code = _clean_cell(row.get(PRESSURE_CODE_COLUMN, ""))
        material_code = _clean_cell(row.get(MATERIAL_CODE_COLUMN, ""))
        type_code = _clean_cell(row.get(TYPE_CODE_COLUMN, ""))
        standard_code = _clean_cell(row.get(STANDARD_RESULT_COLUMN, "")) or _clean_cell(row.get(STANDARD_CODE_COLUMN, ""))
        difficulty = normalize_difficulty_level(row.get(DIFFICULTY_COLUMN, "")) if has_difficulty else None

        if not text:
            size_second_pass_values.append("")
            size_second_pass_reasons.append("")
            size_anchored_hit_texts.append("")
            size_fallback_hit_texts.append("")
            size_consumed_span_texts.append("")
            size_skip_reasons.append("原始描述为空")
            thickness_second_pass_values.append("")
            thickness_second_pass_reasons.append("")
            thickness_anchored_hit_texts.append("")
            thickness_fallback_hit_texts.append("")
            thickness_consumed_span_texts.append("")
            thickness_skip_reasons.append("原始描述为空")
            pressure_second_pass_values.append("")
            pressure_second_pass_reasons.append("")
            pressure_hit_texts.append("")
            pressure_consumed_span_texts.append("")
            pressure_skip_reasons.append("原始描述为空")
            second_pass_values.append("")
            second_pass_reasons.append("")
            base_codes.append("")
            suffix_codes.append("")
            base_hit_texts.append("")
            suffix_hit_texts.append("")
            conflict_texts.append("")
            skip_reasons.append("原始描述为空")
            type_second_pass_values.append("")
            type_second_pass_reasons.append("")
            type_matched_paths.append("")
            type_direct_hit_texts.append("")
            type_body_hit_texts.append("")
            type_manu_hit_texts.append("")
            type_conn_hit_texts.append("")
            type_seal_hit_texts.append("")
            type_angle_hit_texts.append("")
            type_radius_hit_texts.append("")
            type_blocking_codes.append("")
            type_skip_reasons.append("原始描述为空")
            standard_second_pass_values.append("")
            standard_second_pass_reasons.append("")
            standard_codes_texts.append("")
            standard_base_hit_texts.append("")
            standard_prefix_status_texts.append("")
            standard_suffix_hit_texts.append("")
            standard_suspicious_suffix_texts.append("")
            standard_skip_reasons.append("原始描述为空")
            continue

        if has_difficulty and difficulty is not None and difficulty != DIFF_EASY:
            size_second_pass_values.append("")
            size_second_pass_reasons.append("")
            size_anchored_hit_texts.append("")
            size_fallback_hit_texts.append("")
            size_consumed_span_texts.append("")
            size_skip_reasons.append(f"一阶段非简单: {difficulty}")
            thickness_second_pass_values.append("")
            thickness_second_pass_reasons.append("")
            thickness_anchored_hit_texts.append("")
            thickness_fallback_hit_texts.append("")
            thickness_consumed_span_texts.append("")
            thickness_skip_reasons.append(f"一阶段非简单: {difficulty}")
            pressure_second_pass_values.append("")
            pressure_second_pass_reasons.append("")
            pressure_hit_texts.append("")
            pressure_consumed_span_texts.append("")
            pressure_skip_reasons.append(f"一阶段非简单: {difficulty}")
            second_pass_values.append("")
            second_pass_reasons.append("")
            base_codes.append("")
            suffix_codes.append("")
            base_hit_texts.append("")
            suffix_hit_texts.append("")
            conflict_texts.append("")
            skip_reasons.append(f"一阶段非简单: {difficulty}")
            type_second_pass_values.append("")
            type_second_pass_reasons.append("")
            type_matched_paths.append("")
            type_direct_hit_texts.append("")
            type_body_hit_texts.append("")
            type_manu_hit_texts.append("")
            type_conn_hit_texts.append("")
            type_seal_hit_texts.append("")
            type_angle_hit_texts.append("")
            type_radius_hit_texts.append("")
            type_blocking_codes.append("")
            type_skip_reasons.append(f"一阶段非简单: {difficulty}")
            standard_second_pass_values.append("")
            standard_second_pass_reasons.append("")
            standard_codes_texts.append("")
            standard_base_hit_texts.append("")
            standard_prefix_status_texts.append("")
            standard_suffix_hit_texts.append("")
            standard_suspicious_suffix_texts.append("")
            standard_skip_reasons.append(f"一阶段非简单: {difficulty}")
            continue

        if size_result or size_code:
            size_pass_result = size_splitter.analyze(text, size_result, size_code)
            size_second_pass_values.append("通过" if size_pass_result.passed else "不通过")
            size_second_pass_reasons.append(size_pass_result.reason)
            size_anchored_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in size_pass_result.anchored_hits]))
            size_fallback_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in size_pass_result.fallback_hits]))
            size_consumed_span_texts.append(_format_spans(size_pass_result.consumed_spans))
            size_skip_reasons.append("")
        else:
            size_second_pass_values.append("")
            size_second_pass_reasons.append("")
            size_anchored_hit_texts.append("")
            size_fallback_hit_texts.append("")
            size_consumed_span_texts.append("")
            size_skip_reasons.append("SIZE_原始结果和SIZE_原始编码都为空")

        thickness_input_spans = size_pass_result.consumed_spans if (size_result or size_code) else []
        if thickness_result or thickness_code:
            thickness_pass_result = thickness_splitter.analyze(
                text,
                thickness_result,
                thickness_code,
                consumed_spans=thickness_input_spans,
            )
            thickness_second_pass_values.append("通过" if thickness_pass_result.passed else "不通过")
            thickness_second_pass_reasons.append(thickness_pass_result.reason)
            thickness_anchored_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in thickness_pass_result.anchored_hits]))
            thickness_fallback_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in thickness_pass_result.fallback_hits]))
            thickness_consumed_span_texts.append(_format_spans(thickness_pass_result.consumed_spans))
            thickness_skip_reasons.append("")
        else:
            thickness_second_pass_values.append("")
            thickness_second_pass_reasons.append("")
            thickness_anchored_hit_texts.append("")
            thickness_fallback_hit_texts.append("")
            thickness_consumed_span_texts.append("")
            thickness_skip_reasons.append("THICKNESS_原始结果和THICKNESS_原始编码都为空")

        pressure_input_spans = list(thickness_input_spans)
        if thickness_result or thickness_code:
            pressure_input_spans = list(getattr(thickness_pass_result, "consumed_spans", []) or pressure_input_spans)
        if pressure_result or pressure_code:
            pressure_pass_result = pressure_splitter.analyze(
                text,
                pressure_result,
                pressure_code,
                consumed_spans=pressure_input_spans,
            )
            pressure_second_pass_values.append("通过" if pressure_pass_result.passed else "不通过")
            pressure_second_pass_reasons.append(pressure_pass_result.reason)
            pressure_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in pressure_pass_result.anchored_hits]))
            pressure_consumed_span_texts.append(_format_spans(pressure_pass_result.consumed_spans))
            pressure_skip_reasons.append("")
        else:
            pressure_second_pass_values.append("")
            pressure_second_pass_reasons.append("")
            pressure_hit_texts.append("")
            pressure_consumed_span_texts.append("")
            pressure_skip_reasons.append("PRESSURE_原始结果和PRESSURE_原始编码都为空")

        if material_code:
            result = material_splitter.analyze(text, material_code)
            second_pass_values.append("通过" if result.passed else "不通过")
            second_pass_reasons.append(result.reason)
            base_codes.append(result.base_code)
            suffix_codes.append(result.suffix_code)
            base_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in result.base_hits]))
            suffix_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in result.suffix_hits]))
            conflict_texts.append(" | ".join(result.conflict_codes))
            skip_reasons.append("")
        else:
            second_pass_values.append("")
            second_pass_reasons.append("")
            base_codes.append("")
            suffix_codes.append("")
            base_hit_texts.append("")
            suffix_hit_texts.append("")
            conflict_texts.append("")
            skip_reasons.append("MATERIAL_原始编码为空")

        if type_code:
            type_result = type_splitter.analyze(text, type_code)
            type_second_pass_values.append("通过" if type_result.passed else "不通过")
            type_second_pass_reasons.append(type_result.reason)
            type_matched_paths.append(type_result.matched_path)
            type_direct_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.direct_hits]))
            type_body_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.body_hits]))
            type_manu_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.manu_hits]))
            type_conn_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.conn_hits]))
            type_seal_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.seal_hits]))
            type_angle_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.angle_hits]))
            type_radius_hit_texts.append(_join_hit_aliases([hit.to_dict() for hit in type_result.radius_hits]))
            type_blocking_codes.append(type_result.blocking_code)
            type_skip_reasons.append("")
        else:
            type_second_pass_values.append("")
            type_second_pass_reasons.append("")
            type_matched_paths.append("")
            type_direct_hit_texts.append("")
            type_body_hit_texts.append("")
            type_manu_hit_texts.append("")
            type_conn_hit_texts.append("")
            type_seal_hit_texts.append("")
            type_angle_hit_texts.append("")
            type_radius_hit_texts.append("")
            type_blocking_codes.append("")
            type_skip_reasons.append("TYPE_原始编码为空")

        if standard_code:
            standard_result = standard_splitter.analyze(text, standard_code)
            standard_second_pass_values.append("通过" if standard_result.passed else "不通过")
            standard_second_pass_reasons.append(standard_result.reason)
            standard_codes_texts.append(" | ".join(item.raw_code for item in standard_result.checks))
            standard_base_hit_texts.append(
                " || ".join(
                    f"{item.raw_code}: {_join_hit_aliases([hit.to_dict() for hit in item.base_hits])}"
                    for item in standard_result.checks
                )
            )
            standard_prefix_status_texts.append(
                " | ".join(f"{item.raw_code}:{item.prefix_status}" for item in standard_result.checks)
            )
            standard_suffix_hit_texts.append(
                " || ".join(
                    f"{item.raw_code}: {_join_hit_aliases([hit.to_dict() for hit in item.suffix_hits])}"
                    for item in standard_result.checks
                )
            )
            standard_suspicious_suffix_texts.append(
                " || ".join(
                    f"{item.raw_code}: {_join_hit_aliases([hit.to_dict() for hit in item.suspicious_suffix_hits])}"
                    for item in standard_result.checks
                )
            )
            standard_skip_reasons.append("")
        else:
            standard_second_pass_values.append("")
            standard_second_pass_reasons.append("")
            standard_codes_texts.append("")
            standard_base_hit_texts.append("")
            standard_prefix_status_texts.append("")
            standard_suffix_hit_texts.append("")
            standard_suspicious_suffix_texts.append("")
            standard_skip_reasons.append("STANDARD_原始结果和STANDARD_原始编码都为空")

    detail_df = df.copy()
    detail_df[SIZE_SECOND_PASS_COLUMN] = size_second_pass_values
    detail_df[SIZE_SECOND_PASS_REASON_COLUMN] = size_second_pass_reasons
    detail_df[SIZE_SECOND_PASS_ANCHORED_HITS_COLUMN] = size_anchored_hit_texts
    detail_df[SIZE_SECOND_PASS_FALLBACK_HITS_COLUMN] = size_fallback_hit_texts
    detail_df[SIZE_SECOND_PASS_CONSUMED_SPANS_COLUMN] = size_consumed_span_texts
    detail_df[SIZE_SECOND_PASS_SKIP_COLUMN] = size_skip_reasons
    detail_df[THICKNESS_SECOND_PASS_COLUMN] = thickness_second_pass_values
    detail_df[THICKNESS_SECOND_PASS_REASON_COLUMN] = thickness_second_pass_reasons
    detail_df[THICKNESS_SECOND_PASS_ANCHORED_HITS_COLUMN] = thickness_anchored_hit_texts
    detail_df[THICKNESS_SECOND_PASS_FALLBACK_HITS_COLUMN] = thickness_fallback_hit_texts
    detail_df[THICKNESS_SECOND_PASS_CONSUMED_SPANS_COLUMN] = thickness_consumed_span_texts
    detail_df[THICKNESS_SECOND_PASS_SKIP_COLUMN] = thickness_skip_reasons
    detail_df[PRESSURE_SECOND_PASS_COLUMN] = pressure_second_pass_values
    detail_df[PRESSURE_SECOND_PASS_REASON_COLUMN] = pressure_second_pass_reasons
    detail_df[PRESSURE_SECOND_PASS_HITS_COLUMN] = pressure_hit_texts
    detail_df[PRESSURE_SECOND_PASS_CONSUMED_SPANS_COLUMN] = pressure_consumed_span_texts
    detail_df[PRESSURE_SECOND_PASS_SKIP_COLUMN] = pressure_skip_reasons
    detail_df[SECOND_PASS_COLUMN] = second_pass_values
    detail_df[SECOND_PASS_REASON_COLUMN] = second_pass_reasons
    detail_df[SECOND_PASS_BASE_CODE_COLUMN] = base_codes
    detail_df[SECOND_PASS_SUFFIX_CODE_COLUMN] = suffix_codes
    detail_df[SECOND_PASS_BASE_HITS_COLUMN] = base_hit_texts
    detail_df[SECOND_PASS_SUFFIX_HITS_COLUMN] = suffix_hit_texts
    detail_df[SECOND_PASS_CONFLICT_COLUMN] = conflict_texts
    detail_df[SECOND_PASS_SKIP_COLUMN] = skip_reasons
    detail_df[TYPE_SECOND_PASS_COLUMN] = type_second_pass_values
    detail_df[TYPE_SECOND_PASS_REASON_COLUMN] = type_second_pass_reasons
    detail_df[TYPE_SECOND_PASS_PATH_COLUMN] = type_matched_paths
    detail_df[TYPE_SECOND_PASS_DIRECT_HITS_COLUMN] = type_direct_hit_texts
    detail_df[TYPE_SECOND_PASS_BODY_HITS_COLUMN] = type_body_hit_texts
    detail_df[TYPE_SECOND_PASS_MANU_HITS_COLUMN] = type_manu_hit_texts
    detail_df[TYPE_SECOND_PASS_CONN_HITS_COLUMN] = type_conn_hit_texts
    detail_df[TYPE_SECOND_PASS_SEAL_HITS_COLUMN] = type_seal_hit_texts
    detail_df[TYPE_SECOND_PASS_ANGLE_HITS_COLUMN] = type_angle_hit_texts
    detail_df[TYPE_SECOND_PASS_RADIUS_HITS_COLUMN] = type_radius_hit_texts
    detail_df[TYPE_SECOND_PASS_BLOCK_COLUMN] = type_blocking_codes
    detail_df[TYPE_SECOND_PASS_SKIP_COLUMN] = type_skip_reasons
    detail_df[STANDARD_SECOND_PASS_COLUMN] = standard_second_pass_values
    detail_df[STANDARD_SECOND_PASS_REASON_COLUMN] = standard_second_pass_reasons
    detail_df[STANDARD_SECOND_PASS_CODES_COLUMN] = standard_codes_texts
    detail_df[STANDARD_SECOND_PASS_BASE_HITS_COLUMN] = standard_base_hit_texts
    detail_df[STANDARD_SECOND_PASS_PREFIX_STATUS_COLUMN] = standard_prefix_status_texts
    detail_df[STANDARD_SECOND_PASS_SUFFIX_HITS_COLUMN] = standard_suffix_hit_texts
    detail_df[STANDARD_SECOND_PASS_SUSPICIOUS_SUFFIX_COLUMN] = standard_suspicious_suffix_texts
    detail_df[STANDARD_SECOND_PASS_SKIP_COLUMN] = standard_skip_reasons
    detail_df[FINAL_SECOND_PASS_LEVEL_COLUMN] = _build_final_second_pass_level(detail_df)

    chart_paths: list[Path] = []
    if CORRECTNESS_COLUMN in detail_df.columns:
        detail_df[IS_CORRECT_COLUMN] = [
            (_clean_cell(value) == "") if _clean_cell(row.get(TEXT_COLUMN, "")) else pd.NA
            for value, (_, row) in zip(detail_df[CORRECTNESS_COLUMN], detail_df.iterrows())
        ]

    summary_df = _build_summary(detail_df)
    processed_df = _build_processed_only(detail_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总", index=False)
        detail_df.to_excel(writer, sheet_name="明细", index=False)
        processed_df.to_excel(writer, sheet_name="已处理样本", index=False)

    if CORRECTNESS_COLUMN in detail_df.columns:
        final_accuracy_chart = _build_final_accuracy_chart_path(output_path)
        _save_final_accuracy_chart(detail_df, final_accuracy_chart)
        chart_paths.append(final_accuracy_chart)

        final_project_summary = _build_final_project_summary(detail_df)
        if not final_project_summary.empty:
            chart_path = _build_final_project_distribution_chart_path(output_path)
            _save_final_project_distribution_chart(final_project_summary, chart_path)
            chart_paths.append(chart_path)

        final_project_accuracy = _build_final_project_accuracy_summary(detail_df)
        if not final_project_accuracy.empty:
            chart_path = _build_final_project_accuracy_chart_path(output_path)
            _save_final_project_accuracy_chart(final_project_accuracy, chart_path)
            chart_paths.append(chart_path)

    return output_path, chart_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="材质/种类/规范二次分流测试脚本")
    parser.add_argument("--input-file", required=True, help="输入 Excel/CSV 文件")
    parser.add_argument("--output-file", default=None, help="输出 Excel 文件路径")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path, chart_paths = run(args.input_file, args.output_file)
    print(output_path)
    for chart_path in chart_paths:
        print(chart_path)


if __name__ == "__main__":
    main()
