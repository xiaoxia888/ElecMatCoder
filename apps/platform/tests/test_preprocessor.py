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

    def test_unknown_suffix_is_still_split(self) -> None:
        self.assertEqual(self.preprocessor.process("DN20xABC"), "DN20 xABC")


if __name__ == "__main__":
    unittest.main()
