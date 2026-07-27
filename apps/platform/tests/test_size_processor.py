import unittest

from src.encoder.processors.rule_extraction import extract_size_and_thickness_by_rules
from src.encoder.processors.size_processor import SizeProcessor
from src.tokenizer_utils.preprocessor import TextPreprocessor


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


if __name__ == "__main__":
    unittest.main()
