import unittest

from apps.trainer.qwen3_fte.src.convert_structured_material_v3_to_v4 import (
    convert_row,
    minimize_material_value,
)


def row(value, input_text=None, special_req=None):
    return {
        "input": input_text or value,
        "output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": value,
                    "SPECIAL_REQ": special_req or [],
                }
            ],
            "STANDARD": [],
        },
    }


class ConvertStructuredMaterialV4Test(unittest.TestCase):
    def test_complete_tp_grade_drops_material_standard(self):
        self.assertEqual(
            minimize_material_value("ASTM A312 TP316"),
            ("TP316", "remove_material_standard"),
        )

    def test_complete_wp_and_f_grades_drop_material_standard(self):
        self.assertEqual(minimize_material_value("ASTM A403 WP316L")[0], "WP316L")
        self.assertEqual(minimize_material_value("ASTM A182 F304")[0], "F304")

    def test_standard_only_designation_keeps_number(self):
        self.assertEqual(minimize_material_value("ASTM A105")[0], "A105")

    def test_ambiguous_grade_keeps_material_standard(self):
        self.assertEqual(
            minimize_material_value("ASTM A106 Gr.B")[0],
            "ASTM A106 Gr.B",
        )
        self.assertEqual(
            minimize_material_value("ASTM A333 Gr.6")[0],
            "ASTM A333 Gr.6",
        )
        self.assertEqual(
            minimize_material_value("ASTM A516 Gr.70")[0],
            "ASTM A516 Gr.70",
        )
        self.assertEqual(
            minimize_material_value("ASTM A671 CC60")[0],
            "ASTM A671 CC60",
        )

    def test_api_gr_b_keeps_standard_but_x65_does_not(self):
        self.assertEqual(minimize_material_value("API 5L Gr.B")[0], "API 5L Gr.B")
        self.assertEqual(minimize_material_value("API 5L X65")[0], "X65")

    def test_alternatives_are_minimized_without_merging(self):
        self.assertEqual(
            minimize_material_value("ASTM A106 Gr.B or ASTM A53 Gr.B")[0],
            "ASTM A106 Gr.B or ASTM A53 Gr.B",
        )
        self.assertEqual(
            minimize_material_value("ASTM A234 WPB or 304")[0],
            "WPB or 304",
        )

    def test_explicit_international_grade_inside_parentheses_is_preferred(self):
        self.assertEqual(minimize_material_value("2205(A182 F51)")[0], "F51")

    def test_b_standard_alias_reduces_to_concrete_grade(self):
        self.assertEqual(minimize_material_value("B564(N10276)")[0], "N10276")
        self.assertEqual(
            minimize_material_value("B366-WPHC276(N10276)")[0],
            "WPHC276",
        )

    def test_unrelated_parenthetical_alias_is_not_rewritten(self):
        self.assertEqual(
            minimize_material_value("X6CrNiTi18-10(1.4541)")[0],
            "X6CrNiTi18-10(1.4541)",
        )

    def test_composite_keeps_context_dependent_inner_grade(self):
        self.assertEqual(
            minimize_material_value("ASTM A234 WPB (ASTM A516 Gr.65)")[0],
            "WPB (ASTM A516 Gr.65)",
        )

    def test_a269_a312_source_uses_existing_tp316_without_synthesis(self):
        converted, _ = convert_row(
            row(
                "ASTM A312 TP316",
                (
                    '伴热管安装 1/2" Tubing,A269-TP316,PE '
                    "ASTM A312 TP316（06Cr17Ni12Mo2）"
                ),
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "TP316",
        )

    def test_class_fragment_is_removed_from_wp11(self):
        converted, _ = convert_row(
            row(
                "ASTM A234 WP11Cl",
                "90度弯头 ASTM A234 WP11Cl.2 BE",
            )
        )
        self.assertEqual(converted["output"]["MATERIAL"][0]["VALUE"], "WP11")

    def test_explicit_hdpe_level_is_recovered(self):
        converted, _ = convert_row(
            row(
                "HDPE",
                "法兰盖 DN20 HDPE(PE100) SDR11",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "HDPE(PE100)",
        )

    def test_explicit_second_dual_prefix_is_recovered(self):
        converted, _ = convert_row(
            row(
                "ASTM A182 F316/316L",
                "ASTM A182 GR.F316 / F316L Dual Certified",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "F316/F316L",
        )

    def test_ss316l_is_not_truncated_to_s316l(self):
        converted, _ = convert_row(
            row(
                "S316L",
                "90度弯头 LR SMLS TUBE SS316L-EP(PN16)",
            )
        )
        self.assertEqual(converted["output"]["MATERIAL"][0]["VALUE"], "SS316L")

    def test_strong_4pe_context_adds_special_requirement(self):
        converted, audit = convert_row(
            row(
                "20",
                "DN250 PIPE GB/T 8163 20 钢管自带4PE加强级外防腐",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["SPECIAL_REQ"],
            ["4PE"],
        )
        self.assertTrue(audit["special_req_4pe_added"])

    def test_plain_pe_does_not_become_4pe(self):
        converted, audit = convert_row(
            row("20", "PIPE 20 PE SMLS")
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["SPECIAL_REQ"],
            [],
        )
        self.assertFalse(audit["special_req_4pe_added"])

    def test_ms97_without_source_evidence_is_removed(self):
        source = row("ASTM A182 F304/304L", 'Olet 6"*1" SW')
        source["output"]["STANDARD"] = [{"BODY": "MS97"}]
        converted, _ = convert_row(source)
        self.assertEqual(converted["output"]["STANDARD"], [])

    def test_ms97_with_source_evidence_is_kept(self):
        source = row("ASTM A182 F304", 'Olet 6"*1" MSS SP-97')
        source["output"]["STANDARD"] = [{"BODY": "MS97"}]
        converted, _ = convert_row(source)
        self.assertEqual(
            converted["output"]["STANDARD"],
            [{"BODY": "MS97"}],
        )

    def test_explicit_02s403_is_added(self):
        converted, _ = convert_row(
            row("Q235B", "异径三通 02S403钢制管件标准图集 Q235B")
        )
        self.assertEqual(
            converted["output"]["STANDARD"],
            [{"BODY": "02S403"}],
        )


if __name__ == "__main__":
    unittest.main()
