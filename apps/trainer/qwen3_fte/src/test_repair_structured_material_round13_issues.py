from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.repair_structured_material_round13_issues import (
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
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [{"BODY": body} for body in standards or []],
            "MATERIAL_RELATION": "SINGLE",
        },
    }


class RepairStructuredMaterialRound13IssuesTest(unittest.TestCase):
    def test_removes_unevidenced_b1611(self) -> None:
        repaired, _ = repair_row(
            _row(
                "Nipple A182 F304 GB/T 14383(I)",
                "ASTM A182",
                "F304",
                ["GBT14383I", "AB1611"],
            )
        )
        self.assertEqual(repaired["output"]["STANDARD"], [{"BODY": "GBT14383I"}])

    def test_adds_explicit_astm_material_standards(self) -> None:
        a335, _ = repair_row(_row("F11 ASTM A335", "", "F11"))
        self.assertEqual(
            a335["output"]["MATERIAL"][0]["STANDARD"],
            "ASTM A335",
        )
        a672, _ = repair_row(_row("A672-C65 CL32", "", "C65"))
        self.assertEqual(
            a672["output"]["MATERIAL"][0]["STANDARD"],
            "ASTM A672",
        )


if __name__ == "__main__":
    unittest.main()
