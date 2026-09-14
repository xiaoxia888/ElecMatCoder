from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "apps/trainer/qwen3_fte/src/generate_size_v2_error_contrast_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("generate_size_v2_error_contrast_dataset", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def find_sample(fragment: str):
    return next(sample for sample in MODULE.build_samples() if fragment in sample.input)


def test_tr_is_reducing_tee_not_reducer() -> None:
    sample = find_sample("TR, BE, GB/T12459II, GB/T13401, 3X2.5mm")
    assert [item["ROLE"] for item in sample.output["ITEMS"]] == ["MAIN", "BRANCH"]
    assert sample.output["ITEMS"][1]["SIZE"] == [{"type": "DN", "value": "20"}]
    assert sample.output["ITEMS"][1]["THICKNESS"] == [{"type": "MM", "value": "2.5"}]


def test_concentric_reducer_uses_end_roles() -> None:
    sample = find_sample("CON REDUCER 20 SMLS BW ASME B16.9 DN32xDN20")
    assert [item["ROLE"] for item in sample.output["ITEMS"]] == ["END_A", "END_B"]


def test_single_position_keeps_distinct_mm_thicknesses() -> None:
    sample = find_sample("45 Deg Elbow, P235GH, SMLS, TYPE5, BW, 21.3x2.77/2.27")
    assert sample.output["ITEMS"][0]["THICKNESS"] == [
        {"type": "MM", "value": "2.77"},
        {"type": "MM", "value": "2.27"},
    ]


def test_pressure_grade_and_design_pressure_are_negative_examples() -> None:
    material_grade = find_sample("A350 Gr.LF2 CL1")
    assert material_grade.output["PRESSURE"] == ""
    assert not any("设计压力：6Bar" in sample.input for sample in MODULE.build_samples())


def test_group_split_does_not_leak_contrast_variants() -> None:
    synthetic = MODULE.build_samples()
    actual = MODULE.Sample(
        input="真实项目错误样本 DN50",
        output=MODULE.output([MODULE.item("SINGLE", sizes=[("DN", "50")])]),
        category="真实项目错误-测试",
        group="actual-test",
        source="actual_project_error",
    )
    train, val = MODULE.split_samples(synthetic + [actual])
    assert train and val == [actual]
    assert all(sample.source == "synthetic_contrast" for sample in train)
    assert all(sample.source == "actual_project_error" for sample in val)


def test_actual_project_error_parsers() -> None:
    tr = MODULE.parse_actual_project_error(
        1,
        "TR, BE, GB/T12459II, GB/T13401, 3X2.5mm, 06Cr19Ni10, SMLS, GB/T14976 DN32X20",
        "32x20",
        "3MMX2.5MM",
        "",
        0,
        0,
        1,
    )
    multi_wall = MODULE.parse_actual_project_error(
        2,
        "45 Deg Elbow, P235GH(1.0345), SMLS, TYPE5, BW, 88.9x5.6/5.0, DIN 2605-1 DN80",
        "80",
        "5.6MMX5MM",
        "",
        1,
        0,
        1,
    )
    assert tr and tr.source == "actual_project_error"
    assert [current["ROLE"] for current in tr.output["ITEMS"]] == ["MAIN", "BRANCH"]
    assert multi_wall and multi_wall.output["ITEMS"][0]["THICKNESS"] == [
        {"type": "MM", "value": "5.6"},
        {"type": "MM", "value": "5"},
    ]
