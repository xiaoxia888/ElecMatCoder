import unittest

from src.encoder.processors.pressure_processor import PressureProcessor


class PressureProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = PressureProcessor()

    def test_prefix_class_wins_over_overlapping_reverse_class(self) -> None:
        result = self.processor.extract_by_rules(
            "法兰 DN100 FLANGE HG/T20615 20 CL.150 WN RF Sch 20"
        )

        self.assertEqual(result.values, ["C150"])
        self.assertEqual(result.pressure_code, "C150")
        self.assertEqual(result.matched_texts, ["CL.150"])

    def test_reverse_class_still_supported(self) -> None:
        result = self.processor.extract_by_rules("FLANGE DN100 150 CL RF")

        self.assertEqual(result.values, ["C150"])
        self.assertEqual(result.matched_texts, ["150 CL"])

    def test_prefix_pn_is_prioritized(self) -> None:
        result = self.processor.extract_by_rules("FLANGE 20 PN16 RF")

        self.assertEqual(result.values, ["PN16"])
        self.assertEqual(result.matched_texts, ["PN16"])

    def test_schedule_number_is_not_pressure(self) -> None:
        result = self.processor.extract_by_rules("FLANGE DN100 WN RF Sch 20")

        self.assertEqual(result.values, [])
        self.assertEqual(result.pressure_code, "")


if __name__ == "__main__":
    unittest.main()
