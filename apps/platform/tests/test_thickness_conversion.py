import unittest

from src.encoder.pipe_encoder import (
    EncodedFieldResult,
    PipeEncoderBase,
    PipeEncodingResult,
)


class ThicknessConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = PipeEncoderBase()

    @staticmethod
    def _lined_thickness_value():
        return {
            "_ITEMS": [
                {"type": "SCHEDULE", "value": "SCH40", "role": "BASE"},
                {"type": "MM", "value": "4.0", "role": "BASE"},
                {"type": "MM", "value": "3.0", "role": "LINING"},
            ]
        }

    def test_table_items_preserve_layer_roles(self) -> None:
        items = self.encoder.thickness_table_processor.build_thickness_items(
            self._lined_thickness_value()
        )

        self.assertEqual(
            [item["role"] for item in items],
            ["BASE", "BASE", "LINING"],
        )

    def test_schedule_dedup_preserves_base_lining_separator(self) -> None:
        thickness_value = self._lined_thickness_value()
        result = PipeEncodingResult(
            original_text=(
                "无缝衬里钢管 HG/T20538-16 SCH40 "
                "THK=4.0mm PTFE THK=3.0mm DN50"
            )
        )
        result.fields["SIZE"] = EncodedFieldResult(
            field_type="SIZE",
            stage1_raw={"_ITEMS": [{"type": "DN", "value": "50"}]},
            code="50",
        )
        result.fields["STANDARD"] = EncodedFieldResult(
            field_type="STANDARD",
            stage1_raw=[{"BODY": "HG/T20538-16"}],
            code="HGT20538",
        )
        result.fields["THICKNESS"] = EncodedFieldResult(
            field_type="THICKNESS",
            stage1_raw=thickness_value,
            stage2_input=thickness_value,
            code=self.encoder.thickness_processor.process(thickness_value),
        )

        self.encoder._apply_thickness_mm_conversion(result)

        self.assertEqual(result.fields["THICKNESS"].code, "4MM/3MM")
        self.assertIn(
            "壁厚 SCH40 按 DN50 换算为 4MM",
            result.fields["THICKNESS"].notes,
        )


if __name__ == "__main__":
    unittest.main()
