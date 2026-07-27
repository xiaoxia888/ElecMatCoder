import unittest

from src.tokenizer_utils.preprocessor import TextPreprocessor


class TextPreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = TextPreprocessor()

    def test_preserves_dn_followed_by_schedule(self) -> None:
        original = "低压流体输送用焊接钢管SAW,GB/T3091,DN20xXS Q235B+GALV"
        self.assertEqual(self.preprocessor.process(original), original)

    def test_preserves_supported_size_schedule_combinations(self) -> None:
        cases = (
            "DN20xXS",
            "DN20Xxxs",
            "DN20×STD",
            "DN20xSCH40",
            "DN20XS-40",
            "DN20xS40",
            "OD89xSTD",
            "Φ89xSCH40",
        )
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), original)

    def test_multiplication_normalization_still_applies(self) -> None:
        self.assertEqual(self.preprocessor.process("DN20*XS"), "DN20×XS")

    def test_preserves_numeric_reducing_size(self) -> None:
        self.assertEqual(self.preprocessor.process("DN20x15"), "DN20x15")

    def test_preserves_schedule_pair_with_omitted_second_prefix(self) -> None:
        cases = {
            "SCH20X80": "SCH20X80",
            "Sch20X80": "SCH20X80",
            (
                "三通 20, NB/T47008, SW, CL3000，GB/T14383-2008, "
                "I系列 Sch20X80 THK=10.0X4.5mm DN500×25"
            ): (
                "三通 20, NB/T47008, SW, CL3000;GB/T14383-2008, "
                "I系列 SCH20X80 THK=10.0X4.5mm DN500×25"
            ),
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), expected)

    def test_preserves_three_digit_s_dash_schedule(self) -> None:
        cases = (
            "S-100",
            "S-120",
            "S-140",
            "S-160",
            "90 度长半径弯头 DN250 S-100 CF415K GB/T13401",
        )
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), original)

    def test_does_not_partially_split_unknown_s_dash_number(self) -> None:
        self.assertEqual(self.preprocessor.process("PIPE S-1000 CF415K"), "PIPE S-1000 CF415K")

    def test_unknown_suffix_is_still_split(self) -> None:
        self.assertEqual(self.preprocessor.process("DN20xABC"), "DN20 xABC")

    def test_separates_numeric_token_from_roman_grade_marker(self) -> None:
        cases = {
            "S32168Gr.III": "S32168 Gr.III",
            "NB/T47010GR II": "NB/T47010 GR II",
            "NB/T47010GRII": "NB/T47010 GRII",
            "S32168Gr.IIINB/T47010": "S32168 Gr.III NB/T47010",
            "Q245RGR.I": "Q245RGR.I",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), expected)

    def test_does_not_separate_grade_marker_without_roman_grade(self) -> None:
        cases = (
            "NB/T47010GR",
            "NB/T47010GRADE",
            "NB/T47010GR.A",
        )
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), original)

    def test_separates_glued_standard_prefixes(self) -> None:
        cases = {
            "S32168Gr.IIINB/T47010ASMEB16.5": "S32168 Gr.III NB/T47010 ASMEB16.5",
            "Q245RGB/T713": "Q245R GB/T713",
            "A105ASTMA182": "A105 ASTMA182",
            "材料EN10217": "材料 EN10217",
            "A234MSSSP-75": "A234 MSSSP-75",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), expected)

    def test_does_not_split_plain_words_as_standard_prefixes(self) -> None:
        cases = (
            "BLENDED",
            "DINNER",
            "ISOLATION",
            "BASIC",
        )
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), original)

    def test_repairs_radius_ocr_only_in_strong_context(self) -> None:
        cases = {
            "R=l.0D": "R=1.0D",
            "R=I.0D": "R=1.0D",
            "R=i.25D": "R=1.25D",
            "弯头R=L.00D": "弯头R=1.00D",
            "R = l . 0D": "R = 1.0D",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), expected)

    def test_radius_ocr_does_not_modify_weak_or_unrelated_contexts(self) -> None:
        cases = (
            "GRADE=l.0D",
            "SR=I.0D",
            "R=I.D",
            "R=I.0DN",
            "R=LR",
        )
        for original in cases:
            with self.subTest(original=original):
                self.assertEqual(self.preprocessor.process(original), original)

    def test_repairs_radius_ocr_in_full_material_description(self) -> None:
        original = "90度焊接短半径弯头弯头(R=l.0D) SY/T5037-2018 DN1200×STD Q235B"
        expected = "90度焊接短半径弯头弯头(R=1.0D) SY/T5037-2018 DN1200×STD Q235B"
        self.assertEqual(self.preprocessor.process(original), expected)

    def test_does_not_split_decimal_radius_followed_by_dn_letters(self) -> None:
        original = "90º无缝弯头 R=1.5DN 2.材质:20# SMLS"

        self.assertEqual(self.preprocessor.process(original), original)


if __name__ == "__main__":
    unittest.main()
