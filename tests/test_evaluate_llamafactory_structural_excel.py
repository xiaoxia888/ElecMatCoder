import unittest

from apps.trainer.qwen3_fte.src.evaluate_llamafactory_model import (
    _structural_excel_columns,
)
from src.encoder.processors.pressure_processor import PressureProcessor
from src.encoder.processors.size_processor import SizeProcessor
from src.encoder.processors.thickness_processor import ThicknessProcessor


class StructuralExcelColumnsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processors = {
            "size_processor": SizeProcessor(),
            "thickness_processor": ThicknessProcessor(enable_rule_layered=False),
            "pressure_processor": PressureProcessor(),
        }

    def test_reducing_tee_preserves_roles_and_encodes_fields(self) -> None:
        predicted = {
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
            "LENGTH": "",
            "PRESSURE": "CL3000",
        }

        columns = _structural_excel_columns(
            predicted,
            original_text="",
            **self.processors,
        )

        self.assertEqual(columns["STRUCTURAL_尺寸原始值"], "DN：50 DN：40")
        self.assertEqual(columns["STRUCTURAL_尺寸编码"], "50x40")
        self.assertEqual(
            columns["STRUCTURAL_壁厚原始值"],
            "SCHEDULE：SCH40 SCHEDULE：SCH80",
        )
        self.assertEqual(columns["STRUCTURAL_壁厚编码"], "S40XS80")
        self.assertEqual(columns["STRUCTURAL_磅级编码"], "C3000")

    def test_size_prefers_dn_and_appends_length_after_platform_conversion(self) -> None:
        predicted = {
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "OD", "value": "48.3"}],
                    "THICKNESS": [{"type": "MM", "value": "3"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [
                        {"type": "OD", "value": "60.3"},
                        {"type": "DN", "value": "50"},
                    ],
                    "THICKNESS": [{"type": "MM", "value": "4"}],
                },
            ],
            "LENGTH": "120MM",
            "PRESSURE": "PN16",
        }

        columns = _structural_excel_columns(
            predicted,
            original_text="",
            **self.processors,
        )

        self.assertEqual(columns["STRUCTURAL_尺寸编码"], "50x40L120")
        self.assertEqual(columns["STRUCTURAL_壁厚编码"], "4MMX3MM")
        self.assertEqual(columns["STRUCTURAL_磅级编码"], "PN16")

    def test_inch_size_uses_platform_mapping_before_length_suffix(self) -> None:
        predicted = {
            "ITEMS": [
                {
                    "SCOPE": "BODY",
                    "ROLE": "SINGLE",
                    "SIZE": [{"type": "INCH", "value": "2"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "STD"}],
                }
            ],
            "LENGTH": "3000MM",
            "PRESSURE": "1.6MPA",
        }

        columns = _structural_excel_columns(
            predicted,
            original_text="",
            **self.processors,
        )

        self.assertEqual(columns["STRUCTURAL_尺寸编码"], "50L3000")
        self.assertEqual(columns["STRUCTURAL_磅级编码"], "PN16")


if __name__ == "__main__":
    unittest.main()
