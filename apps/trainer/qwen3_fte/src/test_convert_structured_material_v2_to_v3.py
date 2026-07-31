import unittest

from apps.trainer.qwen3_fte.src.convert_structured_material_v2_to_v3 import (
    deduplicate_converted_rows,
    convert_row,
    source_row_removal_reason,
)


def old_item(
    standard="",
    grade="",
    material_class="",
    part="BODY",
    special_req=None,
):
    return {
        "PART": part,
        "STANDARD": standard,
        "GRADE": grade,
        "CLASS": material_class,
        "SPECIAL_REQ": special_req or [],
    }


def row(input_text, materials, relation="SINGLE", standards=None):
    return {
        "input": input_text,
        "output": {
            "MATERIAL": materials,
            "STANDARD": [{"BODY": value} for value in (standards or [])],
            "MATERIAL_RELATION": relation,
        },
    }


class ConvertStructuredMaterialV3Test(unittest.TestCase):
    def test_single_astm_designation_is_one_value(self):
        converted, _ = convert_row(
            row(
                "PIPE ASTM A312 TP304L ASME B36.19",
                [old_item("ASTM A312", "TP304L")],
                standards=["AB3619"],
            )
        )
        self.assertEqual(
            converted["output"],
            {
                "MATERIAL": [
                    {
                        "PART": "BODY",
                        "VALUE": "ASTM A312 TP304L",
                        "SPECIAL_REQ": [],
                    }
                ],
                "STANDARD": [{"BODY": "AB3619"}],
            },
        )

    def test_dual_chinese_grades_are_one_value(self):
        converted, _ = convert_row(
            row(
                "FLANGE S30408/S30403 NB/T47010 SH/T3406 GB/T4334-C",
                [old_item(grade="S30408"), old_item(grade="S30403")],
                relation="DUAL_CERTIFIED",
                standards=["NBT47010", "SHT3406", "GBT4334C"],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"],
            [
                {
                    "PART": "BODY",
                    "VALUE": "S30408/S30403",
                    "SPECIAL_REQ": [],
                }
            ],
        )

    def test_dual_astm_grade_keeps_shared_prefix_once(self):
        converted, _ = convert_row(
            row(
                "ASTM A403 WP304/304L",
                [
                    old_item("ASTM A403", "WP304"),
                    old_item("ASTM A403", "WP304L"),
                ],
                relation="DUAL_CERTIFIED",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "ASTM A403 WP304/304L",
        )

    def test_alternatives_are_not_fake_parts(self):
        converted, _ = convert_row(
            row(
                "A106 Gr.B or A53 Gr.B",
                [
                    old_item("ASTM A106", "Gr.B"),
                    old_item("ASTM A53", "Gr.B"),
                ],
                relation="ALTERNATIVE",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"],
            [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A106 Gr.B or ASTM A53 Gr.B",
                    "SPECIAL_REQ": [],
                }
            ],
        )

    def test_real_physical_parts_remain_separate(self):
        converted, _ = convert_row(
            row(
                "20/PTFE",
                [
                    old_item(grade="20", part="BODY"),
                    old_item(grade="PTFE", part="LINING"),
                ],
                relation="COMPOSITE",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"],
            [
                {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []},
                {"PART": "LINING", "VALUE": "PTFE", "SPECIAL_REQ": []},
            ],
        )

    def test_same_body_composite_becomes_one_complete_value(self):
        converted, _ = convert_row(
            row(
                "ASTM A234 WPB (A516 Gr.65)",
                [
                    old_item("ASTM A234", "WPB"),
                    old_item("ASTM A516", "Gr.65"),
                ],
                relation="COMPOSITE",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "ASTM A234 WPB (ASTM A516 Gr.65)",
        )

    def test_material_class_is_retained_inside_value(self):
        converted, _ = convert_row(
            row("15CrMo Gr.III", [old_item(grade="15CrMo", material_class="Gr.III")])
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"], "15CrMo Gr.III"
        )

    def test_source_recovers_outer_grade_around_astm_designation(self):
        converted, audit = convert_row(
            row(
                "法兰;2205(A182 F51);NB/T47010",
                [old_item("ASTM A182", "F51")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"], "2205(A182 F51)"
        )
        self.assertTrue(audit["source_value_recovered"])

    def test_source_recovers_compact_standard_grade_parentheses(self):
        converted, _ = convert_row(
            row(
                "WN FLANGE; B564(N10276); ASME B16.5",
                [old_item("ASTM B564", "N10276")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"], "B564(N10276)"
        )

    def test_discarded_cf_alias_is_not_restored(self):
        converted, audit = convert_row(
            row("无缝钢管 20(CF415)", [old_item(grade="20")])
        )
        self.assertEqual(converted["output"]["MATERIAL"][0]["VALUE"], "20")
        self.assertFalse(audit["source_value_recovered"])

    def test_product_name_before_parentheses_is_not_material(self):
        converted, audit = convert_row(
            row(
                "ELBOW(ASTM A403 WP304)",
                [old_item("ASTM A403", "WP304")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"], "ASTM A403 WP304"
        )
        self.assertFalse(audit["source_value_recovered"])

    def test_trailing_separator_before_parentheses_is_removed(self):
        converted, _ = convert_row(
            row(
                "PIPE ASTM B622-(N10276)",
                [old_item("ASTM B622", "N10276")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"], "B622(N10276)"
        )

    def test_duplicate_conflict_prefers_common_standard_encoding(self):
        first, _ = convert_row(
            row("same", [old_item(grade="Q235B")], standards=["GBT97111"])
        )
        second, _ = convert_row(
            row("same", [old_item(grade="Q235B")], standards=["GBT9711.1"])
        )
        rows, resolutions = deduplicate_converted_rows(
            [first, second],
            {"GBT97111": 1, "GBT9711.1": 20},
            "train",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["output"]["STANDARD"], [{"BODY": "GBT9711.1"}]
        )
        self.assertEqual(len(resolutions), 1)

    def test_wp11_class_fragment_is_removed(self):
        converted, _ = convert_row(
            row(
                "90度弯头 ASTM A234 WP11Cl.2",
                [old_item("ASTM A234", "WP11Cl")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "ASTM A234 WP11",
        )

    def test_hdpe_pe100_is_recovered_from_source(self):
        converted, _ = convert_row(
            row("法兰盖 HDPE(PE100) SDR11", [old_item(grade="HDPE")])
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "HDPE(PE100)",
        )

    def test_explicit_dual_prefix_is_recovered(self):
        converted, _ = convert_row(
            row(
                "ASTM A182 GR.F316 / F316L Dual Certified",
                [
                    old_item("ASTM A182", "F316"),
                    old_item("ASTM A182", "F316L"),
                ],
                relation="DUAL_CERTIFIED",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "ASTM A182 F316/F316L",
        )

    def test_ss316l_is_not_truncated(self):
        converted, _ = convert_row(
            row("弯头 SS316L-EP(PN16)", [old_item(grade="S316L")])
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "SS316L",
        )

    def test_ss31603_is_not_truncated_and_ep_suffix_is_not_inferred(self):
        converted, _ = convert_row(
            row(
                "材料：SS31603-TUBE-EP,GB/T 14976",
                [old_item(grade="S31603")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0],
            {
                "PART": "BODY",
                "VALUE": "SS31603",
                "SPECIAL_REQ": [],
            },
        )

    def test_ss_pipe_descriptor_does_not_rewrite_another_grade(self):
        converted, _ = convert_row(
            row(
                "SS PIPE ASTM A312 TP304/304L",
                [
                    old_item("ASTM A312", "TP304"),
                    old_item("ASTM A312", "TP304L"),
                ],
                relation="DUAL_CERTIFIED",
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "ASTM A312 TP304/304L",
        )

    def test_jis_sus_prefix_is_not_truncated(self):
        converted, _ = convert_row(
            row(
                "JISB2312 90度长半径弯头 SCH5S/SUS 316 DN40",
                [old_item(grade="316")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "SUS 316",
        )

    def test_jis_sus_f_dotted_form_is_canonicalized(self):
        converted, _ = convert_row(
            row(
                "法兰 JIS B2220 SUS.F.316L DN40",
                [old_item(grade="F316L")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "SUS F316L",
        )

    def test_jis_sus_prefix_is_not_inferred_without_source_evidence(self):
        converted, _ = convert_row(
            row("90度长半径弯头 316 DN40", [old_item(grade="316")])
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "316",
        )

    def test_jis_sus_prefix_preserves_source_grade_body(self):
        converted, _ = convert_row(
            row(
                "无缝不锈钢管 JIS G3459 SUS S31608TP DN50",
                [old_item(grade="S31608TP")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["VALUE"],
            "SUS S31608TP",
        )

    def test_strong_4pe_context_is_added(self):
        converted, _ = convert_row(
            row(
                "PIPE GB/T8163 20 钢管自带4PE加强级外防腐",
                [old_item(grade="20")],
            )
        )
        self.assertEqual(
            converted["output"]["MATERIAL"][0]["SPECIAL_REQ"],
            ["4PE"],
        )

    def test_ms97_without_source_evidence_is_removed(self):
        converted, _ = convert_row(
            row(
                'Olet 6"*1" SW A182 F304',
                [old_item("ASTM A182", "F304")],
                standards=["MS97"],
            )
        )
        self.assertEqual(converted["output"]["STANDARD"], [])

    def test_ms97_with_source_evidence_is_kept(self):
        converted, _ = convert_row(
            row(
                'Olet 6"*1" MSS SP-97 A182 F304',
                [old_item("ASTM A182", "F304")],
                standards=["MS97"],
            )
        )
        self.assertEqual(
            converted["output"]["STANDARD"],
            [{"BODY": "MS97"}],
        )

    def test_explicit_02s403_is_added(self):
        converted, _ = convert_row(
            row(
                "异径三通 02S403钢制管件标准图集 Q235B",
                [old_item(grade="Q235B")],
            )
        )
        self.assertEqual(
            converted["output"]["STANDARD"],
            [{"BODY": "02S403"}],
        )

    def test_ambiguous_a269_a312_row_is_removed(self):
        reason = source_row_removal_reason(
            'Tubing A269-TP316 ASTM A312 TP316（06Cr17Ni12Mo2）'
        )
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()
