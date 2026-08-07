from __future__ import annotations

from src.material_description_splitter.difficulty_levels import DIFF_EASY
from src.material_description_splitter.second_pass.platform_second_pass_runner import PlatformSecondPassRunner


BASE_TEXT = (
    "中压碳钢焊接法兰1 材质：16MnD II NB/T 47008 CL150 5.5mm "
    "2 结构形式：WN-RF HG/T 20615 3 型号、规格：NPS2 CL150"
)
STANDARD_ITEMS = [
    {"code": "HGT20615", "category": "制造"},
    {"code": "NBT47008", "category": "材料"},
]


def test_material_grade_suffix_is_not_reused_as_standard_suffix() -> None:
    result = PlatformSecondPassRunner().analyze(
        text=BASE_TEXT,
        stage1_difficulty=DIFF_EASY,
        material_code="16MnDII",
        standard_items=STANDARD_ITEMS,
    )

    assert result["results"]["MATERIAL"]["passed"]
    assert result["results"]["STANDARD"]["passed"]
    assert all(
        not item["suspicious_suffix_hits"]
        for item in result["results"]["STANDARD"]["checks"]
    )


def test_separate_standard_suffix_is_still_reported() -> None:
    result = PlatformSecondPassRunner().analyze(
        text=f"{BASE_TEXT} HG/T 20615 II",
        stage1_difficulty=DIFF_EASY,
        material_code="16MnDII",
        standard_items=STANDARD_ITEMS,
    )

    assert result["results"]["MATERIAL"]["passed"]
    assert not result["results"]["STANDARD"]["passed"]
    assert "存在疑似未编码后缀" in result["results"]["STANDARD"]["reason"]
