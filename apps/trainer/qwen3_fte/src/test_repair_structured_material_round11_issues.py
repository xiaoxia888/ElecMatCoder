from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.repair_structured_material_round11_issues import (
    repair_row,
)


def _row(
    text: str,
    *,
    standard: str = "",
    grade: str = "",
    standards: list[str] | None = None,
) -> dict:
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


class RepairStructuredMaterialRound11IssuesTest(unittest.TestCase):
    def test_repairs_bare_3410_prefix(self) -> None:
        repaired, _ = repair_row(
            _row("三通 SW 3410 06Cr19Ni10", standards=["GBT3410"])
        )
        self.assertEqual(repaired["output"]["STANDARD"], [{"BODY": "SHT3410"}])

    def test_removes_non_coding_cl2_from_wp12(self) -> None:
        repaired, _ = repair_row(
            _row(
                "A234GR.WP12CL.2 SMLS",
                standard="ASTM A234",
                grade="WP12CL",
            )
        )
        self.assertEqual(repaired["output"]["MATERIAL"][0]["GRADE"], "WP12")

    def test_normalizes_source_grade_and_case(self) -> None:
        repaired, _ = repair_row(
            _row("22Cr17Ni12Mo2(316L)", grade="316L")
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "022Cr17Ni12Mo2",
        )

        repaired, _ = repair_row(_row("材质310s", grade="310s"))
        self.assertEqual(repaired["output"]["MATERIAL"][0]["GRADE"], "310S")

    def test_repairs_material_standard_omissions(self) -> None:
        a240, _ = repair_row(_row("A240GR.TP316", grade="TP316"))
        self.assertEqual(
            a240["output"]["MATERIAL"][0]["STANDARD"],
            "ASTM A240",
        )

        b423, _ = repair_row(
            _row("ASTM B423 GRADE N08825", grade="N08825")
        )
        self.assertEqual(
            b423["output"]["MATERIAL"][0]["STANDARD"],
            "ASTM B423",
        )

    def test_normalizes_standard_and_uns_grade(self) -> None:
        repaired, _ = repair_row(
            _row("20# GB/T 12459Ⅰ", grade="20", standards=["GB/T12459I"])
        )
        self.assertEqual(
            repaired["output"]["STANDARD"],
            [{"BODY": "GBT12459I"}],
        )

        repaired, _ = repair_row(
            _row(
                "ASTM-B564 UNSN04400",
                standard="ASTM B564",
                grade="UNS N04400",
            )
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "N04400",
        )

        repaired, _ = repair_row(
            _row(
                "ASTM B165 UNS N04400",
                standard="ASTM B165",
                grade="UNS N04400",
            )
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "N04400",
        )


if __name__ == "__main__":
    unittest.main()
