from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.repair_structured_material_round10_issues import (
    repair_row,
)


def _row(
    text: str,
    *,
    standard: str = "",
    grade: str = "",
    material_class: str = "",
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
                    "CLASS": material_class,
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [{"BODY": body} for body in standards or []],
            "MATERIAL_RELATION": "SINGLE",
        },
    }


class RepairStructuredMaterialRound10IssuesTest(unittest.TestCase):
    def test_prefers_explicit_chinese_grade_over_equivalent_grade(self) -> None:
        repaired, _ = repair_row(
            _row(
                "名称：022Cr17Ni12Mo2无缝钢管 材料或性能等级：316L",
                grade="316L",
            )
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "022Cr17Ni12Mo2",
        )

        repaired, _ = repair_row(
            _row("法兰06Cr19Ni10 (304) DN25", grade="304")
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "06Cr19Ni10",
        )

    def test_repairs_astm_material_standards_and_grades(self) -> None:
        a691, _ = repair_row(
            _row("ASTM A691Gr.1.25Cr CL22", standard="ASTM A691")
        )
        self.assertEqual(a691["output"]["MATERIAL"][0]["GRADE"], "1.25Cr")

        a182, _ = repair_row(
            _row("ASTM A182 Grade F 304 (UNS S30400)", grade="S30400")
        )
        self.assertEqual(
            a182["output"]["MATERIAL"][0],
            {
                "PART": "BODY",
                "STANDARD": "ASTM A182",
                "GRADE": "F304",
                "CLASS": "",
                "SPECIAL_REQ": [],
            },
        )

        a240, _ = repair_row(
            _row("ASTM A240 GRADE S32205", grade="S32205")
        )
        self.assertEqual(
            a240["output"]["MATERIAL"][0]["STANDARD"],
            "ASTM A240",
        )

    def test_repairs_product_standard_omissions_and_ocr(self) -> None:
        repaired, _ = repair_row(
            _row(
                "GB/T 9711 PSL1 L245 SH/T 3405",
                grade="L245",
                standards=["SHT3405"],
            )
        )
        self.assertEqual(
            repaired["output"]["STANDARD"],
            [{"BODY": "SHT3405"}, {"BODY": "GBT9711"}],
        )

        unchanged, _ = repair_row(
            _row(
                "GB/T 9711.1 L245",
                grade="L245",
                standards=["GBT9711.1"],
            )
        )
        self.assertEqual(
            unchanged["output"]["STANDARD"],
            [{"BODY": "GBT9711.1"}],
        )

        repaired, _ = repair_row(
            _row(
                "DIN 17455;11850 Weld Pipe",
                standards=["DIN17455"],
            )
        )
        self.assertEqual(
            repaired["output"]["STANDARD"],
            [{"BODY": "DIN17455"}, {"BODY": "DIN11850"}],
        )

        repaired, _ = repair_row(
            _row("SF304 GB/T1340I", standards=["GBT1340I"])
        )
        self.assertEqual(
            repaired["output"]["STANDARD"],
            [{"BODY": "GBT13401"}],
        )

    def test_removes_only_unevidenced_gbt3087(self) -> None:
        repaired, _ = repair_row(
            _row("GB/T12459 20", grade="20", standards=["GBT12459", "GBT3087"])
        )
        self.assertEqual(
            repaired["output"]["STANDARD"],
            [{"BODY": "GBT12459"}],
        )

        unchanged, _ = repair_row(
            _row("GB/T3087 20", grade="20", standards=["GBT3087"])
        )
        self.assertEqual(
            unchanged["output"]["STANDARD"],
            [{"BODY": "GBT3087"}],
        )

    def test_repairs_b564_n04400_and_wphc276_parentheses(self) -> None:
        repaired, _ = repair_row(
            _row("ASTM-B564 UNS NO4400", grade="NO4400")
        )
        material = repaired["output"]["MATERIAL"][0]
        self.assertEqual(material["STANDARD"], "ASTM B564")
        self.assertEqual(material["GRADE"], "N04400")

        repaired, _ = repair_row(
            _row(
                "B366-WPHC276(N10276)",
                standard="ASTM B366",
                grade="WPHC276(N10276",
            )
        )
        self.assertEqual(
            repaired["output"]["MATERIAL"][0]["GRADE"],
            "WPHC276(N10276)",
        )

    def test_cf415_does_not_override_explicit_material_20(self) -> None:
        repaired, _ = repair_row(
            _row(
                "GB/T14383,CF415,L=100mm 20 DN40",
                grade="CF415",
            )
        )
        self.assertEqual(repaired["output"]["MATERIAL"][0]["GRADE"], "20")

        unchanged, rules = repair_row(
            _row("DN300 S-20 CF415GB/T13401", grade="CF415")
        )
        self.assertEqual(unchanged["output"]["MATERIAL"][0]["GRADE"], "CF415")
        self.assertNotIn("EXPLICIT_20_OVERRIDES_CF415", rules)

    def test_size_series_is_not_material_class(self) -> None:
        repaired, _ = repair_row(
            _row(
                "同心异径管 150×100Ⅱ 304 GB/T 12459",
                grade="304",
                material_class="Gr.II",
            )
        )
        self.assertEqual(repaired["output"]["MATERIAL"][0]["CLASS"], "")


if __name__ == "__main__":
    unittest.main()
