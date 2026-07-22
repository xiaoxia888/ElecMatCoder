import unittest

from src.encoder.pipe_encoder import PipeEncoderBase


class TypeManuStandardSupplementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = PipeEncoderBase()

    def test_all_configured_standard_codes_supplement_welded(self) -> None:
        standards = (
            "SY/T 5037-2018",
            "GB/T 12771",
            "GB/T 713",
            "GB/T 3091",
            "GB/T 13401",
            "EN 10305-2",
            "EN 10305-3",
            "EN 10305-5",
            "EN 10305-6",
            "GB/T 21832",
            "GB/T 13793",
            "AA 139",
            "AA 139M",
            "EN 10217-7",
            "GB/T 21835",
            "EN 10374",
            "EN 10217",
            "DIN 10374",
            "DIN 10357",
        )
        for standard in standards:
            with self.subTest(standard=standard):
                entities = {
                    "TYPE": {"BODY": "直管", "MANU": []},
                    "STANDARD": [{"BODY": standard}],
                }
                self.assertTrue(self.encoder._supplement_type_manu_from_standard(entities))
                self.assertEqual(entities["TYPE"]["MANU"], "WELDED")

    def test_specific_welding_process_suppresses_welded(self) -> None:
        for manu in ("ERW", "DSAW", "DSAWL", "DSAWH", "SAW", "SAWL", "SAWH", "LSAW", "EFW", "HFW"):
            with self.subTest(manu=manu):
                entities = {
                    "TYPE": {"BODY": "直管", "MANU": [manu]},
                    "STANDARD": [{"BODY": "GB/T 3091"}],
                }
                self.assertTrue(self.encoder._supplement_type_manu_from_standard(entities))
                self.assertEqual(entities["TYPE"]["MANU"], manu)

    def test_non_welding_process_is_kept_with_welded(self) -> None:
        entities = {
            "TYPE": {"BODY": "直管", "MANU": ["SMLS"]},
            "STANDARD": [{"BODY": "GB/T 3091"}],
        }
        self.assertTrue(self.encoder._supplement_type_manu_from_standard(entities))
        self.assertEqual(entities["TYPE"]["MANU"], ["SMLS", "WELDED"])

    def test_unrelated_standard_does_not_supplement(self) -> None:
        entities = {
            "TYPE": {"BODY": "直管", "MANU": []},
            "STANDARD": [{"BODY": "GB/T 8163"}],
        }
        self.assertFalse(self.encoder._supplement_type_manu_from_standard(entities))
        self.assertEqual(entities["TYPE"]["MANU"], [])

    def test_standard_grade_suffix_must_match_exactly(self) -> None:
        for standard in ("GB/T 12771-I", "GB/T 13401(II)"):
            with self.subTest(standard=standard):
                entities = {
                    "TYPE": {"BODY": "直管", "MANU": ["SMLS"]},
                    "STANDARD": [{"BODY": standard}],
                }
                self.assertFalse(self.encoder._supplement_type_manu_from_standard(entities))
                self.assertEqual(entities["TYPE"]["MANU"], ["SMLS"])

    def test_explicit_m_suffix_code_matches_independently(self) -> None:
        entities = {
            "TYPE": {"BODY": "直管", "MANU": []},
            "STANDARD": [{"BODY": "AA 139M"}],
        }
        self.assertTrue(self.encoder._supplement_type_manu_from_standard(entities))
        self.assertEqual(entities["TYPE"]["MANU"], "WELDED")

    def test_resolution_removes_generic_welded(self) -> None:
        self.assertEqual(
            self.encoder.regex_extractor.resolve_values("MANU", ["WELDED", "DSAWL", "DSAW", "SAW"]),
            ["DSAWL"],
        )


if __name__ == "__main__":
    unittest.main()
