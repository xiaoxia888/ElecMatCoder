import unittest

from src.encoder.pipe_encoder import PipeEncoderBase
from src.encoder.processors.material_encoder import MaterialEncoder
from src.llm_ner.predictor import Qwen3Predictor
from src.llm_ner.stage1_orchestrator import Stage1FieldOrchestrator


class _MaterialPartEncoder(PipeEncoderBase):
    def __init__(self) -> None:
        super().__init__()
        self.material_encoder = MaterialEncoder()
        self.fallback_inputs: list[str] = []

    def _should_use_type_combined(self) -> bool:
        return False

    def _process_material_item_structured(self, item):
        encoded = self.material_encoder.encode(item)
        if not encoded.resolved:
            return None
        original = self._flatten_material_item_for_stage2(item)
        return {
            "original": original,
            "matched": original,
            "code": encoded.code,
            "similarity": 0.995,
            "is_exact": True,
            "need_review": False,
            "candidates": [],
        }

    def _process_single_value(self, field_type, value):
        self.fallback_inputs.append(value)
        code = {"UNMAPPED LINING": "UL"}.get(value, "")
        return {
            "original": value,
            "matched": value if code else "",
            "code": code,
            "similarity": 0.9 if code else 0.0,
            "is_exact": False,
            "need_review": not bool(code),
            "candidates": [],
        }


class MaterialPartEncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = _MaterialPartEncoder()

    def test_body_and_lining_are_encoded_separately(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "LINING", "VALUE": "PTFE", "SPECIAL_REQ": []},
            {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
        ])

        self.assertEqual(result.code, "20/PTFE")
        self.assertEqual([item["PART"] for item in result.stage2_input], ["BODY", "LINING"])
        self.assertFalse(result.need_review)

    def test_inner_pipe_precedes_outer_pipe(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "OUTER_PIPE", "VALUE": "20", "SPECIAL_REQ": []},
            {"PART": "INNER_PIPE", "VALUE": "S30408", "SPECIAL_REQ": []},
        ])

        self.assertEqual(result.code, "304/20")
        self.assertEqual(
            [item["PART"] for item in result.stage2_input],
            ["INNER_PIPE", "OUTER_PIPE"],
        )

    def test_body_and_flange_preserve_special_requirement(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "FLANGE", "VALUE": "ASTM A105", "SPECIAL_REQ": ["GALVANIZED"]},
            {"PART": "BODY", "VALUE": "FRP", "SPECIAL_REQ": []},
        ])

        self.assertEqual(result.code, "FRP/A105ZN")

    def test_multiple_coating_requirements_are_appended_in_input_order(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "BODY", "VALUE": "Q235B", "SPECIAL_REQ": ["PE", "EP"]},
        ])

        self.assertEqual(result.code, "Q235BPEEP")

    def test_unmapped_part_uses_existing_scalar_fallback(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
            {"PART": "LINING", "VALUE": "UNMAPPED LINING", "SPECIAL_REQ": []},
        ])

        self.assertEqual(self.encoder.fallback_inputs, ["UNMAPPED LINING"])
        self.assertEqual(result.code, "20/UL")

    def test_duplicate_part_requires_review(self) -> None:
        result = self.encoder._process_material_structured([
            {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
            {"PART": "BODY", "VALUE": "PTFE", "SPECIAL_REQ": []},
        ])

        self.assertTrue(result.need_review)
        self.assertEqual(result.detail_items[0]["part_structure_reason"], "duplicate_part")

    def test_legacy_role_is_normalized_to_body(self) -> None:
        entries = self.encoder._normalize_material_entries([
            {"ROLE": "MAIN", "VALUE": "ASTM A182 F51", "SPECIAL_REQ": ["NACE"]},
        ])

        self.assertEqual(entries, [{
            "PART": "BODY",
            "VALUE": "ASTM A182 F51",
            "SPECIAL_REQ": ["NACE"],
        }])

    def test_platform_payload_keeps_part_in_both_input_stages(self) -> None:
        material = [
            {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
            {"PART": "LINING", "VALUE": "PTFE", "SPECIAL_REQ": []},
        ]

        result = self.encoder.encode({"MATERIAL": material}, material_category="直管")
        payload = result.to_payload_dict()
        field = payload["fields"]["MATERIAL"]

        self.assertEqual(field["stage1_raw"]["value"], material)
        self.assertEqual(field["stage2_input"]["value"], material)
        self.assertEqual(field["stage2_output"]["code"], "20/PTFE")

    def test_global_special_req_supplement_does_not_cross_part_boundaries(self) -> None:
        decisions = {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "FRP", "SPECIAL_REQ": []},
                {"PART": "FLANGE", "VALUE": "ASTM A105", "SPECIAL_REQ": []},
            ]
        }
        predictor = object.__new__(Qwen3Predictor)

        predictor._apply_material_special_req_supplement(
            "FRP body with ASTM A105 galvanized flange",
            decisions,
        )

        self.assertEqual(decisions["MATERIAL"][0]["SPECIAL_REQ"], [])
        self.assertEqual(decisions["MATERIAL"][1]["SPECIAL_REQ"], [])

    def test_orchestrator_preserves_part_bound_special_requirement(self) -> None:
        decisions = {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "FRP", "SPECIAL_REQ": []},
                {
                    "PART": "FLANGE",
                    "VALUE": "ASTM A105",
                    "SPECIAL_REQ": ["GALVANIZED"],
                },
            ]
        }
        orchestrator = object.__new__(Stage1FieldOrchestrator)

        orchestrator._apply_material_special_req_supplement(
            "FRP body with ASTM A105 galvanized flange",
            decisions,
        )

        self.assertEqual(decisions["MATERIAL"][0]["SPECIAL_REQ"], [])
        self.assertEqual(
            decisions["MATERIAL"][1]["SPECIAL_REQ"],
            ["GALVANIZED"],
        )
        result = self.encoder._process_material_structured(decisions["MATERIAL"])
        self.assertEqual(result.code, "FRP/A105ZN")


if __name__ == "__main__":
    unittest.main()
