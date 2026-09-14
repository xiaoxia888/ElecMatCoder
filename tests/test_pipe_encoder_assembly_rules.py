from src.encoder.pipe_encoder import EncodedFieldResult, PipeEncoderBase, PipeEncodingResult


def _field(field_type: str, code: str) -> EncodedFieldResult:
    return EncodedFieldResult(field_type=field_type, code=code, codes=[code] if code else [])


def _assemble(*, category: str, thickness: str, pressure: str) -> PipeEncodingResult:
    encoder = object.__new__(PipeEncoderBase)
    encoder.FIELD_ORDER = ["TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL"]
    result = PipeEncodingResult(
        original_text="",
        material_category=category,
        fields={
            "TYPE": _field("TYPE", "P"),
            "SIZE": _field("SIZE", "50"),
            "THICKNESS": _field("THICKNESS", thickness),
            "PRESSURE": _field("PRESSURE", pressure),
            "MATERIAL": _field("MATERIAL", "A106B"),
        },
    )
    encoder._assemble_code(result)
    return result


def test_straight_pipe_keeps_thickness_and_omits_pressure_from_final_code() -> None:
    result = _assemble(category="直管", thickness="6MM", pressure="C150")

    assert result.final_code == "P506MMA106B"
    assert result.fields["PRESSURE"].code == "C150"
    assert result.fields["PRESSURE"].notes == ["直管同时存在壁厚和磅级时，最终编码仅保留壁厚"]


def test_straight_pipe_keeps_pressure_when_thickness_code_is_missing() -> None:
    result = _assemble(category="直管", thickness="", pressure="C150")

    assert result.final_code == "P50C150A106B"
    assert result.fields["PRESSURE"].notes == []


def test_non_straight_pipe_keeps_both_thickness_and_pressure() -> None:
    result = _assemble(category="管件", thickness="6MM", pressure="C150")

    assert result.final_code == "P506MMC150A106B"
    assert result.fields["PRESSURE"].notes == []
