from src.encoder.pipe_encoder_llm import LlmPipeEncoder
from src.encoder.processors.material_encoder import MaterialEncoder


def test_glass_lined_material_composition_exact_overrides() -> None:
    encoder = MaterialEncoder()

    assert encoder.apply_composition_override("20/GL") == "20GL"
    assert encoder.apply_composition_override("20/GL/A105") == "20GLA105"


def test_other_material_compositions_are_unchanged() -> None:
    encoder = MaterialEncoder()

    assert encoder.apply_composition_override("304/GL") == "304/GL"
    assert encoder.apply_composition_override("20/PTFE") == "20/PTFE"


def test_structured_material_applies_composition_override_after_default_join() -> None:
    encoder = LlmPipeEncoder.__new__(LlmPipeEncoder)
    encoder.material_encoder = MaterialEncoder()
    encoder.config = {
        "material_part_composition": {
            "separator": "/",
            "default_order": ["BODY", "LINING", "FLANGE"],
            "supported_combinations": [
                ["BODY", "LINING"],
                ["BODY", "FLANGE"],
                ["BODY", "LINING", "FLANGE"],
            ],
        }
    }

    result = encoder._process_material_structured(
        [
            {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
            {"PART": "LINING", "VALUE": "GLASS LINED", "SPECIAL_REQ": []},
            {"PART": "FLANGE", "VALUE": "ASTM A105", "SPECIAL_REQ": []},
        ]
    )

    assert result.code == "20GLA105"
    assert result.codes == ["20", "GL", "A105"]
