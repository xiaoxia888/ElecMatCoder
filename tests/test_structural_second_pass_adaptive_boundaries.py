from src.material_description_splitter.second_pass.pressure_surface_matcher import PressureSurfaceMatcher
from src.material_description_splitter.second_pass.platform_second_pass_runner import PlatformSecondPassRunner
from src.material_description_splitter.second_pass.size_surface_matcher import SizeSurfaceMatcher
from src.material_description_splitter.second_pass.thickness_second_pass_splitter import ThicknessSecondPassSplitter


def test_xs_can_follow_numeric_standard_body() -> None:
    result = ThicknessSecondPassSplitter().analyze(
        "FLANGE ASTMA105N(Normalized)CL150RFWN ASMEB16.5XS DN80",
        {"SCHEDULE": ["XS"]},
    )

    assert result.passed


def test_dn_uses_alpha_left_and_numeric_right_boundaries() -> None:
    matcher = SizeSurfaceMatcher()
    item = matcher.parse_size_items({"_ITEMS": [{"type": "DN", "value": "50"}]})[0]

    assert matcher.find_first_anchored_hit("5DN50X", item) is not None
    assert matcher.find_first_anchored_hit("ADN50", item) is None
    assert matcher.find_first_anchored_hit("DN500", item) is None


def test_pressure_uses_adaptive_boundaries() -> None:
    matcher = PressureSurfaceMatcher()
    item = matcher.parse_pressure_items("CL150")[0]

    assert matcher.find_first_anchored_hit("5CL150RF", item) is not None
    assert matcher.find_first_anchored_hit("ACL150", item) is None
    assert matcher.find_first_anchored_hit("CL1500", item) is None


def test_size_bare_value_does_not_consume_thickness_decimal_tail() -> None:
    text = "等径三通;SMLS;BW;GB/T 8163 20;GB/T 13401 GB/T 12459 Series II;-;5.50mm50X40"

    result = PlatformSecondPassRunner().analyze(
        text=text,
        stage1_difficulty="简单",
        size_value={"DN": ["50", "40"]},
        thickness_value={"MM": ["5.5"]},
    )

    assert result["results"]["SIZE"]["passed"]
    assert [hit["text"] for hit in result["results"]["SIZE"]["fallback_hits"]] == ["50", "40"]
    assert result["results"]["THICKNESS"]["passed"]
    assert result["results"]["THICKNESS"]["anchored_hits"][0]["text"] == "5.50mm"
