from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
from openpyxl import load_workbook


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "分析编码准确率难度与置信度.py"
SPEC = importlib.util.spec_from_file_location("encoding_accuracy_confidence_analysis", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def build_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "项目名称": ["P1", "P1", "P2", "P2", "P3"],
            "本项目材料代码": ["A105", "20GL", "B", "C", ""],
            "excel2_原始总编码": [" a105 ", "20/GL", "b", "", ""],
            "excel2_分流最终难度（0=困难，1=中等，2=简单）": ["0", "2", "1", "0=困难", ""],
            "excel2_分流原因": ["原因A", "原因B", "原因A | 原因C", "", ""],
            "excel2_模型置信分": ["99%", "0.98", "80", "40%", ""],
        }
    )


def prepare() -> pd.DataFrame:
    return MODULE.prepare_analysis_data(
        build_source(),
        project_col="项目名称",
        truth_col="本项目材料代码",
        pred_col="excel2_原始总编码",
        difficulty_col="excel2_分流最终难度（0=困难，1=中等，2=简单）",
        reason_col="excel2_分流原因",
        confidence_col="excel2_模型置信分",
    )


def test_code_comparison_preserves_business_separators() -> None:
    data = prepare()

    assert data.loc[0, "分析_是否正确"] == "正确"
    assert data.loc[1, "分析_是否正确"] == "错误"
    assert data.loc[4, "分析_是否正确"] == "无正确答案"


def test_confidence_and_difficulty_parsing() -> None:
    data = prepare()

    assert data.loc[0, "分析_模型置信分"] == 0.99
    assert data.loc[2, "分析_模型置信分"] == 0.8
    assert data.loc[3, "分析_难度"] == "困难"


def test_difficulty_and_project_summaries() -> None:
    data = prepare()
    valid = data[data["分析_正确答案有效"]]
    difficulty = MODULE.build_difficulty_summary(valid).set_index("难度")
    projects = MODULE.build_project_summary(
        valid,
        high_confidence_threshold=0.95,
        minimum_project_samples=30,
    ).set_index("项目")

    assert len(valid) == 4
    assert valid["分析_正确值"].mean() == 0.5
    assert difficulty.loc["困难", "样本数"] == 2
    assert difficulty.loc["困难", "准确率"] == 0.5
    assert projects.loc["P1", "准确率"] == 0.5
    assert projects.loc["P2", "准确率"] == 0.5


def test_calibration_and_workbook_export(tmp_path: Path) -> None:
    data = prepare()
    valid = data[data["分析_正确答案有效"]]
    calibration, metrics = MODULE.build_confidence_calibration(valid)
    output = tmp_path / "analysis.xlsx"

    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["brier"] <= 1
    assert 0 <= metrics["auc"] <= 1

    MODULE.write_analysis_workbook(
        output,
        {
            "难度分析": MODULE.build_difficulty_summary(valid),
            "置信度校准": calibration,
            "分析明细": data,
        },
    )
    assert output.exists()
    assert output.stat().st_size > 0
    workbook = load_workbook(output)
    difficulty_sheet = workbook["难度分析"]
    assert difficulty_sheet.freeze_panes == "A4"
    assert difficulty_sheet["A3"].border.left.style == "thin"
    assert difficulty_sheet["A4"].border.bottom.style == "thin"
