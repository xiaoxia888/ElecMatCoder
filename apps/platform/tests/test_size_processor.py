import unittest

from src.encoder.pipe_encoder import EncodedFieldResult, PipeEncoderBase
from src.encoder.processors.rule_extraction import extract_size_and_thickness_by_rules
from src.encoder.processors.size_processor import SizeProcessor
from src.tokenizer_utils.preprocessor import TextPreprocessor
from src.domain.common.dimension_separator import (
    is_multiplication_separator,
    normalize_multiplication_separators,
    split_by_multiplication_separator,
)


class _StandardAwareSizeEncoder(PipeEncoderBase):
    def __init__(self) -> None:
        super().__init__()
        self.processing_trace = []
        self.received_standard_codes = []

    def _should_use_type_combined(self) -> bool:
        return False

    def _process_standard_multi(self, values, modifier_map=None, original_text=""):
        self.processing_trace.append("STANDARD")
        return EncodedFieldResult(
            field_type="STANDARD",
            code="DIN10357II",
            codes=["DIN10357II"],
            detail_items=[{"code": "DIN10357II", "base_code": "DIN10357"}],
        )

    def _encode_size_multi(self, values, original_text="", standard_codes=None):
        self.processing_trace.append("SIZE")
        self.received_standard_codes = list(standard_codes or [])
        code, need_review = self.size_processor.process_multi_with_review(
            values,
            original_text=original_text,
            standard_codes=standard_codes,
        )
        return EncodedFieldResult(
            field_type="SIZE",
            code=code,
            codes=[code] if code else [],
            need_review=need_review,
        )


class SizeProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = SizeProcessor()
        self.preprocessor = TextPreprocessor()

    def test_d_pair_can_follow_chinese_text(self) -> None:
        text = self.preprocessor.process(
            "同心大小头D20×D15×S40 12459Ⅰ 06Cr19Ni10 14976"
        )
        result = extract_size_and_thickness_by_rules(
            text,
            size_processor=self.processor,
        )

        self.assertEqual(result.size.dn, ["20", "15"])
        self.assertEqual(result.size.size_code, "20x15")
        self.assertEqual(
            result.size.ordered_items,
            [
                {"type": "DN", "value": "20"},
                {"type": "DN", "value": "15"},
            ],
        )
        self.assertEqual(result.thickness.schedule, ["S40"])

    def test_d_anchor_does_not_match_inside_english_word(self) -> None:
        result = self.processor.extract_by_rules("FRIEND20")

        self.assertEqual(result.dn, [])
        self.assertEqual(result.od, [])

    def test_astm_d_grade_is_not_treated_as_size(self) -> None:
        result = self.processor.extract_by_rules("ASTM D20")

        self.assertEqual(result.dn, [])
        self.assertEqual(result.od, [])

    def test_fractional_inch_pair_does_not_absorb_standard_decimal_tail(self) -> None:
        result = self.processor.extract_by_rules(
            'RED. TEE , CS A234-WPB , SMLS , BE , SCH80 , B16.9 3/4"X1/2"'
        )

        self.assertEqual(result.inch, ["3/4", "1/2"])
        self.assertEqual(
            result.ordered_items,
            [
                {"type": "INCH", "value": "3/4"},
                {"type": "INCH", "value": "1/2"},
            ],
        )

    def test_mixed_fraction_integer_part_is_limited_to_one_two_or_three(self) -> None:
        for whole in ("1", "2", "3"):
            with self.subTest(whole=whole):
                result = self.processor.extract_by_rules(f'RED. TEE {whole} 1/2"X1/2"')

                self.assertEqual(result.inch, [f"{whole}-1/2", "1/2"])

    def test_out_of_range_mixed_fraction_does_not_absorb_integer_part(self) -> None:
        result = self.processor.extract_by_rules('RED. TEE 9 3/4"X1/2"')

        self.assertEqual(result.inch, ["3/4", "1/2"])
        self.assertEqual(
            result.ordered_items,
            [
                {"type": "INCH", "value": "3/4"},
                {"type": "INCH", "value": "1/2"},
            ],
        )

    def test_nearest_od_fallback_can_search_upward(self) -> None:
        self.assertEqual(self.processor._od_to_dn(85), 80)
        self.assertEqual(self.processor._od_to_dn(70), 65)

    def test_nearest_od_fallback_rejects_dn_greater_than_input_od(self) -> None:
        processor = SizeProcessor()
        processor._od_to_dn_mapping = {18.0: 20, 21.0: 15}
        processor._od_candidate_rows = {
            18.0: [{"od": 18.0, "dn": 20}],
            21.0: [{"od": 21.0, "dn": 15}],
        }

        self.assertEqual(processor._od_to_dn(19), 15)

    def test_standard_specific_od_mapping_precedes_generic_fallback(self) -> None:
        self.assertEqual(
            self.processor._od_to_dn(23, standard_codes=["DIN10357"]),
            20,
        )
        self.assertEqual(self.processor._od_to_dn(23), 15)

    def test_en10357_a_series_od_mapping(self) -> None:
        expected = {
            13: 10,
            23: 20,
            85: 80,
            204: 200,
        }

        for od, dn in expected.items():
            with self.subTest(od=od):
                self.assertEqual(
                    self.processor._lookup_standard_od_dn(od, ["EN10357"]),
                    dn,
                )

    def test_din11850_supports_reihe_1_and_reihe_2(self) -> None:
        expected = {
            12: 10,
            13: 10,
            52: 50,
            53: 50,
        }

        for od, dn in expected.items():
            with self.subTest(od=od):
                self.assertEqual(
                    self.processor._lookup_standard_od_dn(od, ["DIN11850"]),
                    dn,
                )

    def test_din11866_a_series_od_mapping(self) -> None:
        self.assertEqual(
            self.processor._lookup_standard_od_dn(23, ["DIN11866A"]),
            20,
        )

    def test_en10220_series_1_od_mapping(self) -> None:
        expected = {
            10.2: 6,
            17.2: 10,
            60.3: 50,
            114.3: 100,
            273: 250,
            457: 450,
            610: 600,
        }

        for od, dn in expected.items():
            with self.subTest(od=od):
                self.assertEqual(
                    self.processor._lookup_standard_od_dn(od, ["EN10220"]),
                    dn,
                )

    def test_delivery_standards_do_not_force_od_mapping(self) -> None:
        for standard_code in (
            "DIN17455",
            "DIN17457",
            "EN102177",
            "EN102962",
            "EN10312",
            "ENI1127",
        ):
            with self.subTest(standard_code=standard_code):
                self.assertIsNone(
                    self.processor._lookup_standard_od_dn(85, [standard_code])
                )

    def test_structured_od_uses_final_standard_base_code(self) -> None:
        size_value = {
            "OD": ["23"],
            "_ITEMS": [{"type": "OD", "value": "23"}],
        }

        self.assertEqual(
            self.processor.process(
                size_value,
                standard_codes=["DIN10357"],
            ),
            "20",
        )

    def test_standard_is_processed_before_size_without_changing_code_order(self) -> None:
        encoder = _StandardAwareSizeEncoder()
        result = encoder.encode(
            {
                "SIZE": {
                    "OD": ["23"],
                    "_ITEMS": [{"type": "OD", "value": "23"}],
                },
                "STANDARD": [{"BODY": "DIN 10357", "GRADE": "II"}],
            },
            material_category="直管",
        )

        self.assertEqual(encoder.processing_trace, ["STANDARD", "SIZE"])
        self.assertEqual(encoder.received_standard_codes, ["DIN10357"])
        self.assertEqual(result.fields["SIZE"].code, "20")
        self.assertEqual(result.final_code, "20DIN10357II")

    def test_en10374_reducer_uses_nearest_valid_od_fallback(self) -> None:
        text = (
            "DIN 17455;EN10374 WELD 同心异径管;"
            "DD01TPD 85x2.0-70x2.0 BW1.4301"
        )
        result = extract_size_and_thickness_by_rules(
            text,
            size_processor=self.processor,
        )

        self.assertEqual(result.size.od, ["85", "70"])
        self.assertEqual(result.size.size_code, "80x65")
        self.assertEqual(result.thickness.mm, ["2MM", "2MM"])

    def test_common_dn_after_d_is_preserved_before_mm_thickness(self) -> None:
        text = r"弯头\90° D100×4.5 R1.5D GB12459Ⅱ 20# 8163"
        result = extract_size_and_thickness_by_rules(
            self.preprocessor.process(text),
            size_processor=self.processor,
        )

        self.assertEqual(result.size.dn, ["100"])
        self.assertEqual(result.size.od, [])
        self.assertEqual(result.size.size_code, "100")
        self.assertEqual(result.thickness.mm, ["4.5MM"])

    def test_common_dn_after_d_is_preserved_before_schedule(self) -> None:
        result = extract_size_and_thickness_by_rules(
            "弯头 D100×S40",
            size_processor=self.processor,
        )

        self.assertEqual(result.size.dn, ["100"])
        self.assertEqual(result.size.od, [])
        self.assertEqual(result.size.size_code, "100")
        self.assertEqual(result.thickness.schedule, ["S40"])

    def test_non_common_d_value_before_thickness_remains_od(self) -> None:
        result = extract_size_and_thickness_by_rules(
            "弯头 D108×4.5",
            size_processor=self.processor,
        )

        self.assertEqual(result.size.dn, [])
        self.assertEqual(result.size.od, ["108"])
        self.assertEqual(result.thickness.mm, ["4.5MM"])

    def test_dn_number_followed_by_dot_is_not_extracted(self) -> None:
        cases = (
            "DN2.0",
            "DN3.A100",
            "DN 2.材质:20#",
        )
        for text in cases:
            with self.subTest(text=text):
                result = extract_size_and_thickness_by_rules(
                    text,
                    size_processor=self.processor,
                )
                self.assertEqual(result.size.dn, [])

    def test_radius_dn_letters_do_not_create_section_number_size(self) -> None:
        text = (
            "1.名称:90º无缝弯头 R=1.5DN 2.材质:20# SMLS "
            "3.规格:D114.3×6.02 4.压力等级:Class 300"
        )
        normalized = self.preprocessor.process(text)
        result = extract_size_and_thickness_by_rules(
            normalized,
            size_processor=self.processor,
        )

        self.assertIn("R=1.5DN 2.材质", normalized)
        self.assertEqual(result.size.dn, [])
        self.assertEqual(result.size.od, ["114.3"])
        self.assertEqual(result.size.size_code, "100")
        self.assertEqual(result.thickness.mm, ["6.02MM"])

    def test_three_part_inch_chain_is_fully_extracted(self) -> None:
        text = 'Reducing TEE, ASTM A 420 Gr.WPL6,BW, SMLS, SCH XS, ASME B16.9 2"x1"x2"'
        result = extract_size_and_thickness_by_rules(
            text,
            size_processor=self.processor,
        )

        self.assertEqual(result.size.inch, ["2", "1", "2"])
        self.assertEqual(result.size.size_code, "50x25")
        self.assertEqual(
            result.size.ordered_items,
            [
                {"type": "INCH", "value": "2"},
                {"type": "INCH", "value": "1"},
                {"type": "INCH", "value": "2"},
            ],
        )

    def test_inch_chain_supports_multiplication_separator_variants(self) -> None:
        for separator in "xX×*＊∗﹡✕✖⨉":
            with self.subTest(separator=separator):
                self.assertTrue(is_multiplication_separator(separator))
                text = f'REDUCING TEE 2"{separator}1"{separator}2"'
                result = extract_size_and_thickness_by_rules(
                    text,
                    size_processor=self.processor,
                )
                self.assertEqual(result.size.inch, ["2", "1", "2"])
                self.assertEqual(result.size.size_code, "50x25")

        self.assertEqual(normalize_multiplication_separators("2＊1∗2"), "2×1×2")
        self.assertEqual(split_by_multiplication_separator("2✕1⨉2"), ["2", "1", "2"])


if __name__ == "__main__":
    unittest.main()
