from src.material_description_splitter.second_pass.material_surface_matcher import MaterialSurfaceMatcher
from src.material_description_splitter.second_pass.material_second_pass_splitter import MaterialSecondPassSplitter


def test_long_alphanumeric_material_alias_matches_without_boundaries() -> None:
    matcher = MaterialSurfaceMatcher()

    hits = matcher.match_base_surfaces("FLANGEASTMA105NCL150", "A105")

    assert [hit.text for hit in hits] == ["A105"]


def test_numeric_material_alias_keeps_boundaries() -> None:
    matcher = MaterialSurfaceMatcher()

    assert matcher.match_base_surfaces("A304B", "304") == []


def test_more_specific_target_suppresses_contained_shorter_conflict() -> None:
    result = MaterialSecondPassSplitter().analyze(
        "BLINDFLANGE ASTMA182GR.F316LCL150RFBLD ASMEB16.5 DN80",
        "316L",
    )

    assert result.passed
    assert result.conflict_codes == []


def test_shorter_target_is_rejected_when_text_contains_more_specific_material() -> None:
    result = MaterialSecondPassSplitter().analyze("ASTMA182GR.F316LCL150", "316")

    assert not result.passed
    assert "316L" in result.conflict_codes


def test_independent_shorter_material_hit_remains_a_conflict() -> None:
    result = MaterialSecondPassSplitter().analyze("F316L/F316", "316L")

    assert not result.passed
    assert "316" in result.conflict_codes


def test_a105n_and_a105_follow_same_specificity_rule() -> None:
    specific = MaterialSecondPassSplitter().analyze("ASTMA105NCL150", "A105N")
    shorter = MaterialSecondPassSplitter().analyze("ASTMA105NCL150", "A105")

    assert specific.passed
    assert not shorter.passed
    assert "A105N" in shorter.conflict_codes
