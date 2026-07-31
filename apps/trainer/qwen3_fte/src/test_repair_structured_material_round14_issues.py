from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.repair_structured_material_round14_issues import (
    repair_row,
)


def _row(text: str, standard: str, grade: str, special_req=None) -> dict:
    return {
        "input": text,
        "output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "STANDARD": standard,
                    "GRADE": grade,
                    "CLASS": "",
                    "SPECIAL_REQ": special_req or [],
                }
            ],
            "STANDARD": [],
            "MATERIAL_RELATION": "SINGLE",
        },
    }


class RepairStructuredMaterialRound14IssuesTest(unittest.TestCase):
    def test_splits_glass_lining_from_body(self) -> None:
        repaired, rules = repair_row(
            _row("INSTRUMENT TEE, 20 GLASS LINED, SH/T 3406", "", "20")
        )
        self.assertIn("SPLIT_20_GLASS_LINING", rules)
        self.assertEqual(repaired["output"]["MATERIAL_RELATION"], "COMPOSITE")
        self.assertEqual(
            [(item["PART"], item["GRADE"]) for item in repaired["output"]["MATERIAL"]],
            [("BODY", "20"), ("LINING", "GLASS")],
        )

    def test_does_not_duplicate_existing_glass_lining(self) -> None:
        row = _row("FLANGED PIPE, 20 GLASS LINED", "", "20")
        row["output"]["MATERIAL"].append(
            {
                "PART": "LINING",
                "STANDARD": "",
                "GRADE": "GLASS",
                "CLASS": "",
                "SPECIAL_REQ": [],
            }
        )
        row["output"]["MATERIAL_RELATION"] = "COMPOSITE"
        repaired, rules = repair_row(row)
        self.assertNotIn("SPLIT_20_GLASS_LINING", rules)
        self.assertEqual(len(repaired["output"]["MATERIAL"]), 2)

    def test_extracts_a691_fractional_chromium_grade(self) -> None:
        repaired, rules = repair_row(
            _row("ASTM A691 Gr.1 1/4 Cr22;ASME B36.10 Pipe", "ASTM A691", "")
        )
        self.assertIn("ASTM_A691_1_25CR_GRADE", rules)
        self.assertEqual(repaired["output"]["MATERIAL"][0]["GRADE"], "1.25Cr")

    def test_prefers_a312_tp304_over_parenthetical_uns(self) -> None:
        repaired, rules = repair_row(
            _row(
                "ASTM A312Grade TP304(UNS S30400), ASME B36.19M",
                "",
                "S30400",
                ["OXYGEN"],
            )
        )
        material = repaired["output"]["MATERIAL"][0]
        self.assertIn("ASTM_A312_TP304_PRECEDENCE", rules)
        self.assertEqual(material["STANDARD"], "ASTM A312")
        self.assertEqual(material["GRADE"], "TP304")
        self.assertEqual(material["SPECIAL_REQ"], ["OXYGEN"])


if __name__ == "__main__":
    unittest.main()
