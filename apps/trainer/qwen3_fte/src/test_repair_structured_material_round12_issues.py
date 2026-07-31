from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.repair_structured_material_round12_issues import (
    repair_row,
)


def _row(text: str, standard: str, grade: str, standards=None) -> dict:
    return {
        "input": text,
        "output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "STANDARD": standard,
                    "GRADE": grade,
                    "CLASS": "",
                    "SPECIAL_REQ": ["GALVANIZED"] if "Galv" in text else [],
                }
            ],
            "STANDARD": [{"BODY": body} for body in standards or []],
            "MATERIAL_RELATION": "SINGLE",
        },
    }


class RepairStructuredMaterialRound12IssuesTest(unittest.TestCase):
    def test_repairs_material_standard_omissions(self) -> None:
        cases = [
            ("1 1/4CrCL22A691/A691M", "", "1 1/4Cr", "ASTM A691", "1.25Cr"),
            ("ASTM A815 UNS S32205-S", "", "S32205", "ASTM A815", "S32205"),
            ("B622-(N10276)", "", "N10276", "ASTM B622", "N10276"),
        ]
        for text, old_std, old_grade, expected_std, expected_grade in cases:
            with self.subTest(text=text):
                repaired, _ = repair_row(_row(text, old_std, old_grade))
                material = repaired["output"]["MATERIAL"][0]
                self.assertEqual(material["STANDARD"], expected_std)
                self.assertEqual(material["GRADE"], expected_grade)

    def test_separates_galvanized_from_wpb_grade(self) -> None:
        repaired, _ = repair_row(
            _row("A234 WPBGalv.", "ASTM A234", "WPBGalv")
        )
        self.assertEqual(repaired["output"]["MATERIAL"][0]["GRADE"], "WPB")

    def test_removes_a420_from_product_standards(self) -> None:
        repaired, _ = repair_row(
            _row(
                "ASTM A420/420M WPL6",
                "ASTM A420",
                "WPL6",
                ["ASTM420"],
            )
        )
        self.assertEqual(repaired["output"]["STANDARD"], [])


if __name__ == "__main__":
    unittest.main()
