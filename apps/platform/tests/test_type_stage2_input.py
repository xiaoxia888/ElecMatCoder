import unittest

from src.encoder.pipe_encoder import PipeEncoderBase
from src.encoder.processors.regex_extractor import RegexExtractor
from src.encoder.processors.type_encoder import TypeEncoder
from src.encoder.processors.type_normalizers import normalize_type_radius
from src.llm_ner.predictor import Qwen3Predictor


class _CapturingTypeEncoder(PipeEncoderBase):
    def __init__(self) -> None:
        super().__init__()
        self.captured_fallback = ""
        self.captured_structured = None

    def _encode_type_value(self, merged_value, type_value=None):
        self.captured_fallback = merged_value
        self.captured_structured = type_value
        return "45RLTW", 0.9

    def _should_use_type_combined(self) -> bool:
        return True


class TypeStage2InputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = PipeEncoderBase()

    def test_type_payload_matches_training_json_exactly(self) -> None:
        value = {
            "FLANGE_STYLE": "",
            "BODY": "异径斜三通",
            "GEOMETRY": {"ANGLE": "45", "RADIUS": ""},
            "CONN": [],
            "SEAL": [],
            "MANU": ["WELDED", "WELDED"],
        }

        self.assertEqual(
            self.encoder._flatten_type_value_for_stage2(value),
            '{"BODY":"异径斜三通","ANGLE":"45","MANU":["WELDED"]}',
        )

    def test_type_payload_flattens_geometry_and_preserves_training_order(self) -> None:
        value = {
            "BODY": "弯头",
            "GEOMETRY": {"ANGLE": "90", "RADIUS": "LR"},
            "FLANGE_STYLE": "固定法兰",
            "CONN": ["BW"],
            "SEAL": ["RF"],
            "MANU": ["SMLS"],
        }

        self.assertEqual(
            self.encoder._normalize_type_stage2_payload(value),
            {
                "BODY": "弯头",
                "ANGLE": "90",
                "RADIUS": "LR",
                "FLANGE_STYLE": "固定法兰",
                "CONN": ["BW"],
                "SEAL": ["RF"],
                "MANU": ["SMLS"],
            },
        )

    def test_numeric_radius_removes_only_trailing_decimal_zeros(self) -> None:
        cases = {
            "1.20D": "1.2D",
            "1.000D": "1D",
            "1D": "1D",
            "2.50d": "2.5D",
            "LR": "LR",
            "SR": "SR",
            "1.20DN": "1.20DN",
        }

        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(normalize_type_radius(raw_value), expected)
                self.assertEqual(TypeEncoder._normalize_radius(raw_value), expected)

    def test_radius_is_normalized_in_stage1_snapshot_and_stage2_payload(self) -> None:
        stage1 = {
            "TYPE": {
                "BODY": "弯头",
                "GEOMETRY": {"ANGLE": "90", "RADIUS": "1.000D"},
            }
        }

        self.encoder._normalize_type_radius_in_entities(stage1)

        self.assertEqual(stage1["TYPE"]["GEOMETRY"]["RADIUS"], "1D")
        self.assertEqual(
            self.encoder._normalize_type_stage2_payload(stage1["TYPE"]),
            {"BODY": "弯头", "ANGLE": "90", "RADIUS": "1D"},
        )

    def test_regex_radius_normalization_removes_trailing_decimal_zeros(self) -> None:
        self.assertEqual(RegexExtractor._normalize_radius_code("R=1.200D"), "1.2D")
        self.assertEqual(RegexExtractor._normalize_radius_code("R 1.000D"), "1D")

    def test_encode_normalizes_radius_in_both_display_stages(self) -> None:
        encoder = _CapturingTypeEncoder()
        result = encoder.encode(
            {
                "TYPE": {
                    "BODY": "弯头",
                    "GEOMETRY": {"ANGLE": "90", "RADIUS": "1.000D"},
                    "MANU": ["WELDED"],
                }
            },
            material_category="管件",
        )

        type_result = result.fields["TYPE"]
        self.assertEqual(type_result.stage1_raw["GEOMETRY"]["RADIUS"], "1D")
        self.assertEqual(type_result.stage2_input["RADIUS"], "1D")
        self.assertEqual(
            encoder.captured_fallback,
            '{"BODY":"弯头","ANGLE":"90","RADIUS":"1D","MANU":["WELDED"]}',
        )

    def test_predictor_uses_canonical_value_label_for_type_only(self) -> None:
        predictor = object.__new__(Qwen3Predictor)
        predictor.backend = "mlx_service"
        predictor.stage2_system_prompt = None
        captured: list[str] = []
        predictor._call_model = lambda _system, user: captured.append(user) or {"_raw": "T"}

        predictor._encode_one_single_field_text("TYPE", '{"BODY":"三通"}')
        predictor._encode_one_single_field_text("MATERIAL", "A234 WPB")

        self.assertEqual(captured[0], '字段类型: TYPE\n规范值: {"BODY":"三通"}')
        self.assertEqual(captured[1], "字段类型: MATERIAL\n原始值: A234 WPB")

    def test_combined_type_path_sends_json_only_to_model_fallback(self) -> None:
        encoder = _CapturingTypeEncoder()
        entities = {
            "TYPE": {
                "BODY": "异径斜三通",
                "GEOMETRY": {"ANGLE": "45", "RADIUS": ""},
                "FLANGE_STYLE": "",
                "CONN": [],
                "SEAL": [],
                "MANU": ["WELDED"],
            }
        }

        result = encoder._process_type_combined(entities, {})

        self.assertEqual(
            encoder.captured_fallback,
            '{"BODY":"异径斜三通","ANGLE":"45","MANU":["WELDED"]}',
        )
        self.assertEqual(encoder.captured_structured["GEOMETRY"]["ANGLE"], "45")
        self.assertEqual(
            result.stage2_input,
            {"BODY": "异径斜三通", "ANGLE": "45", "MANU": ["WELDED"]},
        )


if __name__ == "__main__":
    unittest.main()
