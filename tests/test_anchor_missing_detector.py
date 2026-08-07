from __future__ import annotations

import pytest

from src.material_description_splitter.anchor_missing_detector import AnchorMissingDetector


@pytest.fixture(scope="module")
def detector() -> AnchorMissingDetector:
    return AnchorMissingDetector()


@pytest.mark.parametrize(
    "text",
    [
        "DIN 17455;EN10374 WELD RE;DD01PPD 53x1.5-41x1.5 BW1.4301",
        "REDUCER 53/1.5 BW",
        "REDUCER 53*1.5 BW",
    ],
)
def test_integer_decimal_dimension_pairs_are_not_missing_anchors(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize(
    "text",
    [
        "设计温度150℃；设计压力：6Bar，插焊 DN150x80",
        "设计温度：150，插焊 DN150x80",
        "DESIGN TEMPERATURE 150 C, DN150x80",
    ],
)
def test_temperature_values_have_semantic_anchors(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize(
    "text",
    [
        "Bld Flg,ASME B16.5,ASTM A182 Gr.F316L,RF,150 Lbs.,DN80",
        "BLIND FLANGE RF 150 LB DN80",
        "BLIND FLANGE RF 150 PSI DN80",
        "BLIND FLANGE RF 150# DN80",
        "BLIND FLANGE RF #150 DN80",
        "BLIND FLANGE RF 150＃ DN80",
        "BLIND FLANGE RF ＃150 DN80",
    ],
)
def test_imperial_pressure_units_are_explicit_anchors(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize(
    "text",
    [
        "WN Flg,ASME B16.5,ASTM A105,RF/BW End,300 Lbs.,S40 DN25",
        "WN Flg,ASME B16.5,ASTM A105,RF/BW End,300 Lbs.,S-40 DN25",
        "WN Flg,ASME B16.5,ASTM A105,RF/BW End,300 Lbs.,S 40 DN25",
        "WN Flg,ASME B16.5,ASTM A105,RF/BW End,300 Lbs.,40S DN25",
        "WN Flg,ASME B16.5,ASTM A105,RF/BW End,300 Lbs.,40 S DN25",
    ],
)
def test_schedule_s_shorthand_is_an_explicit_anchor(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize(
    "text",
    [
        "不锈钢法兰盖 BL25-10 RF 06Cr19Ni10 HG/T 20592-2009 DN25",
        "不锈钢法兰盖 BL25(B)-10 RF 06Cr19Ni10 HG/T 20592-2009 DN25",
        "不锈钢法兰盖 BL25（B）-10 RF 06Cr19Ni10 HG/T 20592-2009 DN25",
    ],
)
def test_hyphenated_numeric_model_codes_are_not_naked_numbers(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize(
    "text",
    [
        "偏心短管DN50×D25 S40/80 BE/PE",
        "偏心短管DN50×D25 SCH40x80 BE/PE",
        "偏心短管DN50×D25 40/S80 BE/PE",
        "偏心短管DN50×D25 THK4*5 BE/PE",
    ],
)
def test_pair_anchor_is_shared_with_the_other_numeric_item(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert not result.matched


@pytest.mark.parametrize("text", ["REDUCER 50x80", "REDUCER 50/80", "REDUCER 53-1.5"])
def test_unanchored_ambiguous_pairs_remain_difficult(
    detector: AnchorMissingDetector,
    text: str,
) -> None:
    result = detector.analyze(text)

    assert result.matched
    assert any(hit.code_group == "naked_spec" for hit in result.hits)
