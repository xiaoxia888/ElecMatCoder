import unittest

from src.encoder.pipe_encoder import PipeEncoderBase
from src.llm_ner.structural_field_output_normalizer import StructuralFieldOutputNormalizer
from src.material_description_splitter.second_pass.size_surface_matcher import SizeSurfaceMatcher
from src.material_description_splitter.second_pass.thickness_surface_matcher import ThicknessSurfaceMatcher


class StructuralV2RuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = PipeEncoderBase()

    def test_v2_normalization_is_mutually_exclusive_with_v1(self) -> None:
        result = StructuralFieldOutputNormalizer.normalize(
            {
                "ITEMS": [
                    {
                        "SCOPE": "BODY",
                        "ROLE": "MAIN",
                        "SIZE": [{"type": "DN", "value": "80"}],
                        "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
                    }
                ],
                "LENGTH": "",
                "PRESSURE": "",
            }
        )

        self.assertEqual(result["_schema_version"], "v2")
        self.assertIn("ITEMS", result)
        self.assertNotIn("SIZE_ITEMS", result)
        self.assertNotIn("THICKNESS_ITEMS", result)
        self.assertNotIn("SIZE", result)
        self.assertNotIn("THICKNESS", result)

    def test_main_branch_size_and_thickness_preserve_position(self) -> None:
        structural = {
            "_schema_version": "v2",
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "40"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH80"}],
                },
            ],
            "LENGTH": "100",
            "PRESSURE": "",
        }

        size = self.encoder._encode_structural_v2_size(structural)
        thickness = self.encoder._encode_structural_v2_thickness(structural)

        self.assertEqual(size.code, "50x40L100")
        self.assertEqual(thickness.code, "S40XS80")
        self.assertEqual(size.stage2_input["ITEMS"][0]["ROLE"], "MAIN")
        self.assertEqual(size.stage2_input["ITEMS"][1]["ROLE"], "BRANCH")

    def test_equal_end_codes_collapse_to_single_code(self) -> None:
        structural = {
            "_schema_version": "v2",
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_A",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [{"type": "MM", "value": "4"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_B",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [{"type": "MM", "value": "4"}],
                },
            ],
            "LENGTH": "",
            "PRESSURE": "",
        }

        self.assertEqual(self.encoder._encode_structural_v2_size(structural).code, "50")
        self.assertEqual(self.encoder._encode_structural_v2_thickness(structural).code, "4MM")

    def test_each_position_uses_highest_priority_thickness(self) -> None:
        structural = {
            "_schema_version": "v2",
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "150"}],
                    "THICKNESS": [
                        {"type": "SCHEDULE", "value": "SCH40"},
                        {"type": "MM", "value": "7.1"},
                    ],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "40"}],
                    "THICKNESS": [
                        {"type": "SCHEDULE", "value": "SCH80"},
                        {"type": "MM", "value": "5"},
                    ],
                },
            ],
            "LENGTH": "",
            "PRESSURE": "CL3000",
        }

        result = self.encoder._encode_structural_v2_thickness(structural)

        self.assertEqual(result.code, "7.1MMX5MM")
        self.assertEqual(len(result.stage2_input["ITEMS"][0]["THICKNESS"]), 2)

    def test_schedule_number_has_priority_over_std_series(self) -> None:
        structural = {
            "_schema_version": "v2",
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_A",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [
                        {"type": "SCHEDULE", "value": "STD"},
                        {"type": "SCHEDULE", "value": "SCH40"},
                    ],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_B",
                    "SIZE": [{"type": "DN", "value": "25"}],
                    "THICKNESS": [
                        {"type": "SCHEDULE", "value": "XS"},
                        {"type": "SCHEDULE", "value": "S80"},
                    ],
                },
            ],
            "LENGTH": "",
            "PRESSURE": "",
        }

        self.assertEqual(self.encoder._encode_structural_v2_thickness(structural).code, "S40XS80")

    def test_second_pass_matchers_read_v2_without_v1_conversion(self) -> None:
        structural = {
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [{"type": "MM", "value": "4"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "40"}],
                    "THICKNESS": [{"type": "MM", "value": "3"}],
                },
            ],
            "LENGTH": "",
            "PRESSURE": "",
        }

        sizes = SizeSurfaceMatcher().parse_size_items(structural)
        thicknesses = ThicknessSurfaceMatcher().parse_thickness_items(structural)

        self.assertEqual([item.value for item in sizes], ["50", "40"])
        self.assertEqual([item.value for item in thicknesses], ["4", "3"])


if __name__ == "__main__":
    unittest.main()
