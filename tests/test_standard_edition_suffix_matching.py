from src.material_description_splitter.second_pass.standard_second_pass_splitter import StandardSecondPassSplitter


def _analyze(text: str):
    return StandardSecondPassSplitter().analyze_items(
        text,
        [{"code": "HGT20592B", "category": "制造"}],
    )


def test_standard_suffix_can_follow_four_digit_edition() -> None:
    assert _analyze("HG/T20592-2009B 法兰").passed
    assert _analyze("HG/T20592-2009 B系列 法兰").passed


def test_standard_suffix_can_follow_two_digit_edition() -> None:
    assert _analyze("HG/T20592-09B 法兰").passed
    assert _analyze("HG/T20592-09 B系列 法兰").passed


def test_standard_suffix_must_immediately_follow_edition() -> None:
    result = _analyze("HG/T20592-2009 RF BOLT 法兰")

    assert not result.passed
    assert "未命中规范后缀表达" in result.reason
