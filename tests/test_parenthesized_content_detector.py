from src.material_description_splitter.parenthesized_content_detector import ParenthesizedContentDetector


def test_long_ascii_parenthesized_content_is_difficult():
    result = ParenthesizedContentDetector().analyze(
        "3/4'' SOCKET WELD FLANGE 150# RF 125 - 250 μin AARH (3.2 - 6.3 μm Ra)"
    )

    assert result.matched
    assert result.hits[0].token == "(3.2 - 6.3 μm Ra)"


def test_short_parenthesized_codes_are_ignored():
    detector = ParenthesizedContentDetector()

    assert not detector.analyze("SW25(B)-25").matched
    assert not detector.analyze("HG/T20592(II)").matched


def test_five_characters_is_the_inclusive_threshold():
    detector = ParenthesizedContentDetector()

    assert not detector.analyze("法兰(ABCD)").matched
    assert not detector.analyze("法兰(A   B)").matched
    assert detector.analyze("法兰(ABCDE)").matched
    assert detector.analyze("法兰（ABCDE）").matched
