from src.material_description_splitter.second_pass.source_evidence_completeness_checker import (
    SourceEvidenceCompletenessChecker,
)


CHECKER = SourceEvidenceCompletenessChecker()


def v2(items, pressure="", length=""):
    return {"ITEMS": items, "PRESSURE": pressure, "LENGTH": length}


def position(role, sizes=(), thicknesses=()):
    return {
        "SCOPE": "BODY",
        "ROLE": role,
        "SIZE": [{"type": item_type, "value": value} for item_type, value in sizes],
        "THICKNESS": [{"type": item_type, "value": value} for item_type, value in thicknesses],
    }


def test_explicit_pairs_match_complete_v2_output() -> None:
    result = CHECKER.audit(
        "异径三通 DN50xDN25 SCH40xSCH80 PN16",
        v2(
            [
                position("MAIN", [("DN", "50")], [("SCHEDULE", "SCH40")]),
                position("BRANCH", [("DN", "25")], [("SCHEDULE", "SCH80")]),
            ],
            pressure="PN16",
        ),
    )

    assert not result.has_missing


def test_explicit_pairs_report_missing_second_side() -> None:
    result = CHECKER.audit(
        "异径三通 DN50xDN25 SCH40xSCH80 PN16",
        v2([position("MAIN", [("DN", "50")], [("SCHEDULE", "SCH40")])], pressure="PN16"),
    )

    assert [(item.item_type, item.value) for item in result.missing["SIZE"]] == [("DN", "25")]
    assert [(item.item_type, item.value) for item in result.missing["THICKNESS"]] == [
        ("SCHEDULE", "SCH80")
    ]


def test_equal_pair_keeps_multiplicity() -> None:
    result = CHECKER.audit(
        "等径三通 DN50xDN50 SCH40xSCH40",
        v2([position("MAIN", [("DN", "50")], [("SCHEDULE", "SCH40")])]),
    )

    assert [(item.item_type, item.value) for item in result.missing["SIZE"]] == [("DN", "50")]
    assert [(item.item_type, item.value) for item in result.missing["THICKNESS"]] == [
        ("SCHEDULE", "SCH40")
    ]


def test_od_wall_and_dn_pair_are_kept_separately() -> None:
    result = CHECKER.audit(
        "夹套法兰 PN16 Φ32X3mm DN50X25",
        v2(
            [
                position("SINGLE", [("DN", "50")]),
                {
                    "SCOPE": "INNER",
                    "ROLE": "SINGLE",
                    "SIZE": [
                        {"type": "OD", "value": "32"},
                        {"type": "DN", "value": "25"},
                    ],
                    "THICKNESS": [{"type": "MM", "value": "3"}],
                },
            ],
            pressure="PN16",
        ),
    )

    assert result.missing["SIZE"] == []
    assert result.missing["THICKNESS"] == []


def test_single_od_anchor_uses_ratio_to_distinguish_dual_od() -> None:
    dual_od = CHECKER.audit("规格 Φ88.9×60.3", size_result="OD88.9 | OD60.3")
    od_wall = CHECKER.audit("规格 Φ88.9×5.6", size_result="OD88.9", thickness_result="5.6MM")

    assert [(item.item_type, item.value) for item in dual_od.evidence["SIZE"]] == [
        ("OD", "88.9"),
        ("OD", "60.3"),
    ]
    assert dual_od.evidence["THICKNESS"] == []
    assert not dual_od.has_missing
    assert [(item.item_type, item.value) for item in od_wall.evidence["SIZE"]] == [("OD", "88.9")]
    assert [(item.item_type, item.value) for item in od_wall.evidence["THICKNESS"]] == [("MM", "5.6")]
    assert not od_wall.has_missing


def test_explicit_mm_overrides_od_wall_ratio() -> None:
    result = CHECKER.audit("规格 Φ32×20mm", size_result="OD32", thickness_result="20MM")

    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [("OD", "32")]
    assert [(item.item_type, item.value) for item in result.evidence["THICKNESS"]] == [("MM", "20")]
    assert not result.has_missing


def test_single_od_anchor_prefers_two_configured_integer_sizes_over_ratio() -> None:
    result = CHECKER.audit("规格 Φ450×15", size_result="OD450 | OD15")

    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [
        ("OD", "450"),
        ("OD", "15"),
    ]
    assert result.evidence["THICKNESS"] == []
    assert not result.has_missing


def test_metric_pressure_is_normalized_but_design_pressure_is_ignored() -> None:
    grade = CHECKER.audit("法兰 1.6MPa DN50", v2([position("SINGLE", [("DN", "50")])], pressure="PN16"))
    design = CHECKER.audit("设计压力：6Bar DN50", v2([position("SINGLE", [("DN", "50")])]))

    assert grade.missing["PRESSURE"] == []
    assert design.evidence["PRESSURE"] == []


def test_ambiguous_standalone_mm_is_not_a_high_precision_evidence() -> None:
    result = CHECKER.audit("法兰 DN50 5.5mm", v2([position("SINGLE", [("DN", "50")])]))

    assert result.evidence["THICKNESS"] == []


def test_platform_display_columns_can_be_audited() -> None:
    result = CHECKER.audit(
        "异径三通 DN50xDN25 SCH40xSCH80 PN16",
        size_result="MAIN: DN50 | BRANCH: DN25",
        thickness_result="MAIN: SCHEDULE: SCH40 | BRANCH: SCHEDULE: SCH80",
        pressure_result="PN16",
    )

    assert not result.has_missing


def test_platform_suffix_value_format_can_be_audited() -> None:
    result = CHECKER.audit(
        "异径三通 DN50x25 8mmx7mm PN16",
        size_result="DN50 | DN25",
        thickness_result="8MM | 7MM",
        pressure_result="PN16",
    )

    assert not result.has_missing


def test_material_20_hash_is_not_pressure() -> None:
    result = CHECKER.audit("90°弯头 DN50 20# GB/T8163", size_result="DN50")

    assert result.evidence["PRESSURE"] == []


def test_compact_dn_integer_pair_uses_configured_common_values() -> None:
    result = CHECKER.audit(
        "弯头90° DN700×10 1.5D SHT3408 Q245R",
        size_result="DN700 | DN10",
    )

    assert not result.has_missing
    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [
        ("DN", "700"),
        ("DN", "10"),
    ]
    assert result.evidence["THICKNESS"] == []


def test_material_grade_and_manufacturer_std_are_not_schedule() -> None:
    material = CHECKER.audit("支管座 S22053 DN200X25", size_result="DN200 | DN25")
    manufacturer = CHECKER.audit('ELBOW PN25 MFR\'S STD 6 in', size_result='6"', pressure_result="PN25")

    assert material.evidence["THICKNESS"] == []
    assert manufacturer.evidence["THICKNESS"] == []


def test_class_material_grade_is_not_pressure() -> None:
    result = CHECKER.audit("A350 Gr.LF2 CL1 DN50", size_result="DN50")

    assert result.evidence["PRESSURE"] == []


def test_repeated_dn_pair_does_not_increase_required_count() -> None:
    result = CHECKER.audit(
        "支管座 DN25(B) 20 DN200X25",
        size_result="DN200 | DN25",
    )

    assert result.missing["SIZE"] == []


def test_numbered_clause_is_not_merged_into_wall_decimal() -> None:
    result = CHECKER.audit(
        "2.规格:Ø45×5.03.连接方式:焊接",
        size_result="OD45",
        thickness_result="5MM",
    )

    assert result.missing["THICKNESS"] == []


def test_explanatory_parentheses_do_not_add_product_size() -> None:
    result = CHECKER.audit(
        "BEND SCH80 (100mm tangent for NPS 4 and below, 150mm tangent for NPS 6);2 in",
        size_result='2"',
        thickness_result="SCH80",
    )

    assert result.missing["SIZE"] == []


def test_compact_dn_pair_uses_configured_second_size() -> None:
    result = CHECKER.audit(
        "TR 3X2.5mm DN32X20",
        size_result="DN32 | DN20",
        thickness_result="3MM | 2.5MM",
    )

    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [
        ("DN", "32"),
        ("DN", "20"),
    ]
    assert [(item.item_type, item.value) for item in result.evidence["THICKNESS"]] == [
        ("MM", "3"),
        ("MM", "2.5"),
    ]
    assert not result.has_missing


def test_bare_integer_pair_uses_configured_common_dn_values() -> None:
    result = CHECKER.audit("异径件 50x25", size_result="DN50 | DN25")

    assert not result.has_missing
    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [
        ("DN", "50"),
        ("DN", "25"),
    ]


def test_bare_integer_pair_requires_both_common_dn_values() -> None:
    result = CHECKER.audit("规格 50x23")

    assert result.evidence["SIZE"] == []


def test_bare_number_times_decimal_extracts_wall_and_common_dn_size() -> None:
    result = CHECKER.audit("规格 50x3.5", size_result="DN50", thickness_result="3.5MM")

    assert not result.has_missing
    assert [(item.item_type, item.value) for item in result.evidence["SIZE"]] == [("DN", "50")]
    assert [(item.item_type, item.value) for item in result.evidence["THICKNESS"]] == [("MM", "3.5")]


def test_bare_decimal_wall_does_not_guess_non_common_size_type() -> None:
    result = CHECKER.audit("规格 88.9x5.6", thickness_result="5.6MM")

    assert result.evidence["SIZE"] == []
    assert [(item.item_type, item.value) for item in result.evidence["THICKNESS"]] == [("MM", "5.6")]
    assert not result.has_missing


def test_common_dn_pair_does_not_consume_schedule_or_material_pair() -> None:
    schedule = CHECKER.audit("SCH40xSCH80", thickness_result="SCH40 | SCH80")
    material = CHECKER.audit("材质 304x20")

    assert schedule.evidence["SIZE"] == []
    assert not schedule.has_missing
    assert material.evidence["SIZE"] == []


def test_common_dn_pair_rejects_wall_and_inch_suffixes() -> None:
    wall = CHECKER.audit("DN50x25mm")
    inch = CHECKER.audit('50x25"')

    assert [(item.item_type, item.value) for item in wall.evidence["SIZE"]] == [("DN", "50")]
    assert [(item.item_type, item.value) for item in inch.evidence["SIZE"]] == [("INCH", "25")]


def test_new_inference_rules_can_be_disabled() -> None:
    checker = SourceEvidenceCompletenessChecker(
        infer_common_dn_pairs=False,
        infer_bare_decimal_wall=False,
    )

    result = checker.audit("50x25 80x3.5")

    assert result.evidence["SIZE"] == []
    assert result.evidence["THICKNESS"] == []
