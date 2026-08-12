from __future__ import annotations

from src.material_description_splitter.difficulty_levels import DIFF_EASY, DIFF_HARD, DIFF_SECOND_EASY
from src.material_description_splitter.difficulty_splitter import MaterialDifficultySplitter
from src.material_description_splitter.routing_pipeline import apply_project_frequency
from src.material_description_splitter.second_pass.platform_second_pass_runner import PlatformSecondPassRunner


BASE_ANALYZE_ARGS = {
    "text": "Blind Flange ASTM A105 RF CL150 ASME B16.5 DN80",
    "stage1_difficulty": DIFF_EASY,
    "size_value": {"DN": ["80"]},
    "pressure_value": "CL150",
    "material_code": "A105",
    "type_code": "BF",
    "standard_items": [{"code": "AB165", "category": "制造"}],
    "success": True,
    "final_code": "BF80C150A105AB165",
    "validate_output": True,
}


def test_valid_parenthesized_content_is_not_a_difficulty_rule() -> None:
    splitter = MaterialDifficultySplitter()

    for text in (
        "承插焊法兰 (PN25;SW;RF;NB/T47008 20) DN20",
        "带颈平焊法兰 GB/T 12228-A105(C≤0.3%) RF CL150 HG/T20615 DN25",
    ):
        result = splitter.analyze(text)
        assert not result.is_difficult
        assert all(feature.name != "parenthesized_content" for feature in result.features)


def test_naked_number_risk_cannot_be_cleared_by_later_model_output() -> None:
    splitter = MaterialDifficultySplitter()
    stage1 = splitter.analyze("Elbow 6 STD BW ASME B16.9")

    assert stage1.is_difficult
    assert any(feature.name == "anchor_missing" and feature.matched for feature in stage1.features)

    stage2 = PlatformSecondPassRunner().analyze(
        text="Elbow 6 STD BW ASME B16.9",
        stage1_difficulty=DIFF_HARD,
        type_code="6EL",
    )
    assert stage2["final_level"] == DIFF_HARD


def test_all_backchecks_and_high_confidence_produce_simple() -> None:
    result = PlatformSecondPassRunner().analyze(
        **BASE_ANALYZE_ARGS,
        confidence=0.95,
        confidence_provided=True,
    )

    assert result["final_level"] == DIFF_SECOND_EASY
    assert all(payload["passed"] for payload in result["results"].values())


def test_low_or_invalid_confidence_produces_hard() -> None:
    runner = PlatformSecondPassRunner()

    low = runner.analyze(**BASE_ANALYZE_ARGS, confidence=0.80, confidence_provided=True)
    invalid = runner.analyze(**BASE_ANALYZE_ARGS, confidence=None, confidence_provided=True)

    assert low["final_level"] == DIFF_HARD
    assert not low["results"]["CONFIDENCE"]["passed"]
    assert invalid["final_level"] == DIFF_HARD
    assert not invalid["results"]["CONFIDENCE"]["passed"]


def test_empty_final_code_produces_hard() -> None:
    args = {**BASE_ANALYZE_ARGS, "final_code": ""}
    result = PlatformSecondPassRunner().analyze(
        **args,
        confidence=0.95,
        confidence_provided=True,
    )

    assert result["final_level"] == DIFF_HARD
    assert not result["results"]["OUTPUT"]["passed"]


def test_project_low_frequency_directly_downgrades_to_hard() -> None:
    results = []
    projects = ["P1"] * 100
    for index in range(100):
        type_code = "RARE" if index == 0 else "BF"
        results.append(
            {
                "routing": {
                    "final_level": DIFF_SECOND_EASY,
                    "reason_text": "所有已提取字段回查通过",
                    "failed_checks": [],
                },
                "fields": {
                    "TYPE": {"stage2_output": {"code": type_code}},
                    "MATERIAL": {"stage2_output": {"code": "A105"}},
                },
            }
        )

    finalized = apply_project_frequency(results, projects)

    assert finalized[0]["routing"]["final_level"] == DIFF_HARD
    assert finalized[0]["routing"]["need_review"] is True
    assert finalized[1]["routing"]["final_level"] == DIFF_SECOND_EASY
