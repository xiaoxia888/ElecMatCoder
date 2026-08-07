from src.material_description_splitter.second_pass.pressure_surface_matcher import PressureSurfaceMatcher
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
