from __future__ import annotations

import importlib.util
from pathlib import Path

from apps.trainer.qwen3_fte.src.apply_size_v2_group_decisions import (
    repair_approved_explicit_equivalent_specs,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "src"
    / "prepare_size_dataset_v2_conversion.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_size_dataset_v2_conversion", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def old_output(size, thickness):
    return {
        "SIZE_ITEMS": size,
        "LENGTH": "",
        "THICKNESS_ITEMS": thickness,
        "PRESSURE": "",
    }


def v2_row(text, items):
    return {
        "input": text,
        "output": {"ITEMS": items, "LENGTH": "", "PRESSURE": ""},
    }


def test_repairs_explicit_od_schedule_pairs_by_existing_roles():
    rows = [
        v2_row(
            "GB/T 8163;SH/T 3408 Red.Tee 60.3 SCH40-48.3 SCH80 BW20 DN50x40",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "50"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "40"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH80"}],
                },
            ],
        )
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert len(changes) == 1
    assert skips == []
    assert rows[0]["output"]["ITEMS"][0]["SIZE"] == [
        {"type": "OD", "value": "60.3"},
        {"type": "DN", "value": "50"},
    ]
    assert rows[0]["output"]["ITEMS"][1]["SIZE"] == [
        {"type": "OD", "value": "48.3"},
        {"type": "DN", "value": "40"},
    ]


def test_repairs_pipe_od_when_wall_and_dn_are_already_labeled():
    rows = [
        v2_row(
            "ASTM-A335-P11 /ASME B36.10 Pipe 33.4x4.55 PE DN25",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "SINGLE",
                    "SIZE": [{"type": "DN", "value": "25"}],
                    "THICKNESS": [{"type": "MM", "value": "4.55"}],
                }
            ],
        )
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert len(changes) == 1
    assert skips == []
    assert rows[0]["output"]["ITEMS"][0]["SIZE"] == [
        {"type": "OD", "value": "33.4"},
        {"type": "DN", "value": "25"},
    ]


def test_repairs_explicit_equal_tee_branch_spec():
    rows = [
        v2_row(
            "EQUAL TEE A403GR.WP316L SMLS BW ASME B16.9 DN200 S-10S X 200 S-10S",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "200"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH10S"}],
                }
            ],
        )
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert len(changes) == 1
    assert skips == []
    assert [item["ROLE"] for item in rows[0]["output"]["ITEMS"]] == ["MAIN", "BRANCH"]
    assert rows[0]["output"]["ITEMS"][1]["SIZE"] == [
        {"type": "DN", "value": "200"}
    ]


def test_schedule_pair_repair_does_not_consume_xs_or_material_fragments():
    rows = [
        v2_row(
            "对焊管接台 DN600xDN200 S-30xS-30 S32168NB/T47010",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "600"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH30"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "200"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH30"}],
                },
            ],
        )
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert changes == []
    assert skips == []


def test_repairs_explicit_schedule_pair_for_second_position():
    rows = [
        v2_row(
            "异径三通 DN250*80 Sch 10S Sch 10S2.连接形式：焊接",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "MAIN",
                    "SIZE": [{"type": "DN", "value": "250"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH10S"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "BRANCH",
                    "SIZE": [{"type": "DN", "value": "80"}],
                    "THICKNESS": [],
                },
            ],
        ),
        v2_row(
            "偏心异径管 DN300xDN200 S-20 x S-20",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_A",
                    "SIZE": [{"type": "DN", "value": "300"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH20"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_B",
                    "SIZE": [{"type": "DN", "value": "200"}],
                    "THICKNESS": [],
                },
            ],
        ),
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert len(changes) == 2
    assert skips == []
    assert rows[0]["output"]["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"}
    ]
    assert rows[1]["output"]["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH20"}
    ]


def test_schedule_pair_repair_handles_ocr_split_suffix_but_not_material_grade():
    rows = [
        v2_row(
            "异径管 DN100x80 SCH10SXSCH1 0S SF304",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_A",
                    "SIZE": [{"type": "DN", "value": "100"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH10S"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_B",
                    "SIZE": [{"type": "DN", "value": "80"}],
                    "THICKNESS": [],
                },
            ],
        ),
        v2_row(
            "异径管 DN100x80 S10S S3408",
            [
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_A",
                    "SIZE": [{"type": "DN", "value": "100"}],
                    "THICKNESS": [{"type": "SCHEDULE", "value": "SCH10S"}],
                },
                {
                    "SCOPE": "BODY",
                    "ROLE": "END_B",
                    "SIZE": [{"type": "DN", "value": "80"}],
                    "THICKNESS": [],
                },
            ],
        ),
    ]

    changes, skips = repair_approved_explicit_equivalent_specs(rows)

    assert len(changes) == 1
    assert skips == []
    assert rows[0]["output"]["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"}
    ]
    assert rows[1]["output"]["ITEMS"][1]["THICKNESS"] == []


def test_repairs_od_only_when_approved_equivalence_is_already_evidenced():
    evidence = v2_row(
        "弯头 OD323.9 DN300 SCH20",
        [
            {
                "SCOPE": "BODY",
                "ROLE": "SINGLE",
                "SIZE": [
                    {"type": "OD", "value": "323.9"},
                    {"type": "DN", "value": "300"},
                ],
                "THICKNESS": [{"type": "SCHEDULE", "value": "SCH20"}],
            }
        ],
    )
    target = v2_row(
        "GB/T 8163;SH/T 3408 Tee 323.9 SCH20 BW DN300",
        [
            {
                "SCOPE": "BODY",
                "ROLE": "MAIN",
                "SIZE": [{"type": "DN", "value": "300"}],
                "THICKNESS": [{"type": "SCHEDULE", "value": "SCH20"}],
            }
        ],
    )
    standard_number = v2_row(
        "90度弯头 ASME B16.9 SCH40 DN40",
        [
            {
                "SCOPE": "BODY",
                "ROLE": "SINGLE",
                "SIZE": [{"type": "DN", "value": "40"}],
                "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
            }
        ],
    )

    changes, skips = repair_approved_explicit_equivalent_specs(
        [evidence, target, standard_number]
    )

    assert len(changes) == 1
    assert skips == []
    assert target["output"]["ITEMS"][0]["SIZE"] == [
        {"type": "OD", "value": "323.9"},
        {"type": "DN", "value": "300"},
    ]
    assert standard_number["output"]["ITEMS"][0]["SIZE"] == [
        {"type": "DN", "value": "40"}
    ]


def test_restores_glued_mpa_product_pressure_without_touching_design_pressure():
    product = v2_row(
        "卫生级法兰 DIN11853-2 BF 1.6MPa1.4301 DN80",
        [
            {
                "SCOPE": "BODY",
                "ROLE": "SINGLE",
                "SIZE": [{"type": "DN", "value": "80"}],
                "THICKNESS": [],
            }
        ],
    )
    product["output"]["PRESSURE"] = "PN16"
    design = v2_row(
        "法兰 DN80 设计压力1.6MPa",
        [
            {
                "SCOPE": "BODY",
                "ROLE": "SINGLE",
                "SIZE": [{"type": "DN", "value": "80"}],
                "THICKNESS": [],
            }
        ],
    )
    design["output"]["PRESSURE"] = "PN16"

    changes, skips = repair_approved_explicit_equivalent_specs([product, design])

    assert len(changes) == 1
    assert skips == []
    assert product["output"]["PRESSURE"] == "1.6MPA"
    assert design["output"]["PRESSURE"] == "PN16"


def test_reducing_tee_expands_explicit_equal_thickness_to_main_and_branch():
    description = "REDUCING TEE, 20 GB/T 3087, SMLS, BW, SH/T 3408,S-40 X S-40 DN80x50"
    source = old_output(
        [{"type": "DN", "value": "80"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "80"}],
            "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "DN", "value": "50"}],
            "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
        },
    ]


def test_tee_with_one_thickness_defaults_to_main():
    description = "REDUCING TEE DN80x50 SCH40 20 GB/T8163 BW"
    source = old_output(
        [{"type": "DN", "value": "80"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"][0]["ROLE"] == "MAIN"
    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "SCHEDULE", "value": "SCH40"}]
    assert [item["ROLE"] for item in converted["ITEMS"]] == ["MAIN", "BRANCH"]


def test_equal_tee_keeps_equivalent_sizes_and_wall_in_main():
    description = "等径三通;SMLS;BW;SF304;φ26.9XSCH40S DN20"
    source = old_output(
        [{"type": "OD", "value": "26.9"}, {"type": "DN", "value": "20"}],
        [{"type": "SCHEDULE", "value": "SCH40S"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    assert MODULE.convert_output(source, plan)["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [
                {"type": "OD", "value": "26.9"},
                {"type": "DN", "value": "20"},
            ],
            "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40S"}],
        }
    ]


def test_outlet_single_metric_spec_belongs_to_branch():
    description = "1.材质:支管台 20# 2.规格: Ø45×3.5 3.连接方式:焊接"
    source = old_output(
        [{"type": "OD", "value": "45"}],
        [{"type": "MM", "value": "3.5"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    assert MODULE.convert_output(source, plan)["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "OD", "value": "45"}],
            "THICKNESS": [{"type": "MM", "value": "3.5"}],
        }
    ]


def test_outlet_with_one_thickness_defaults_to_branch():
    description = "对焊管接台 BW STD GB/T19326 DN125x32"
    source = old_output(
        [{"type": "DN", "value": "125"}, {"type": "DN", "value": "32"}],
        [{"type": "SCHEDULE", "value": "STD"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "125"}],
            "THICKNESS": [],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "DN", "value": "32"}],
            "THICKNESS": [{"type": "SCHEDULE", "value": "STD"}],
        },
    ]


def test_wol_hyphen_code_and_bw_olet_are_outlet_evidence():
    description = "NB/T 47008(II);GB/T 19326WOL-90 STD BWOlet DN150xDN50 20"
    source = old_output(
        [{"type": "DN", "value": "150"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "STD"}],
    )

    topology, _ = MODULE.topology_from_text(description)
    plan, _ = MODULE.deterministic_plan(topology, description, source)
    assert topology == "BRANCH"
    assert plan is not None
    assert MODULE.convert_output(source, plan)["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "150"}],
            "THICKNESS": [],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "DN", "value": "50"}],
            "THICKNESS": [{"type": "SCHEDULE", "value": "STD"}],
        },
    ]


def test_welding_outlet_maps_two_schedules_by_position():
    description = "WELDING OUTLET A105N BW MSS SP-97 DN100x25 S-40XS-160"
    source = old_output(
        [{"type": "DN", "value": "100"}, {"type": "DN", "value": "25"}],
        [
            {"type": "SCHEDULE", "value": "SCH40"},
            {"type": "SCHEDULE", "value": "SCH160"},
        ],
    )

    topology, _ = MODULE.topology_from_text(description)
    plan, _ = MODULE.deterministic_plan(topology, description, source)
    assert topology == "BRANCH"
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH40"}
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH160"}
    ]


def test_reinforcement_pad_opening_weld_is_an_attached_branch_structure():
    description = "补强板开口焊 DN200 x DN50 S-40 20 GB/T711"
    source = old_output(
        [{"type": "DN", "value": "200"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    topology, _ = MODULE.topology_from_text(description)
    plan, _ = MODULE.deterministic_plan(topology, description, source)
    assert topology == "BRANCH"
    assert plan is not None
    assert MODULE.convert_output(source, plan)["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "200"}],
            "THICKNESS": [],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "DN", "value": "50"}],
            "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
        },
    ]


def test_reducer_with_one_thickness_defaults_to_end_a():
    description = "CON.REDUCER C.S.ASTM A234 WPB SMLS BW ASME B16.9 XS DN20xDN15"
    source = old_output(
        [{"type": "DN", "value": "20"}, {"type": "DN", "value": "15"}],
        [{"type": "SCHEDULE", "value": "XS"}],
    )

    plan, _ = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["ROLE"] == "END_A"
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "XS"}
    ]
    assert converted["ITEMS"][1]["ROLE"] == "END_B"
    assert converted["ITEMS"][1]["THICKNESS"] == []


def test_half_coupling_with_run_and_outlet_sizes_is_branch_topology():
    description = "HALF COUPLING,SW,ASTM A105,ASME B16.11,3000LB DN80x25"
    source = old_output(
        [{"type": "DN", "value": "80"}, {"type": "DN", "value": "25"}],
        [],
    )

    topology, _ = MODULE.topology_from_text(description)
    plan, _ = MODULE.deterministic_plan(topology, description, source)
    assert topology == "BRANCH"
    assert plan is not None
    assert [item["ROLE"] for item in MODULE.convert_output(source, plan)["ITEMS"]] == [
        "MAIN",
        "BRANCH",
    ]


def test_glued_reducer_name_is_reducer_topology():
    description = "GB/T14976 SMLSReducer CON.273.1x4.19-219.1x3.76 BW S31603"
    source = old_output(
        [{"type": "OD", "value": "273.1"}, {"type": "OD", "value": "219.1"}],
        [{"type": "MM", "value": "4.19"}, {"type": "MM", "value": "3.76"}],
    )

    topology, _ = MODULE.topology_from_text(description)
    plan, _ = MODULE.deterministic_plan(topology, description, source)
    assert topology == "REDUCER"
    assert plan is not None
    assert [item["ROLE"] for item in MODULE.convert_output(source, plan)["ITEMS"]] == [
        "END_A",
        "END_B",
    ]


def test_glued_reducer_expands_explicit_equal_schedule_to_both_ends():
    description = "GB/T14976 SMLSReducer CON.88.9 SCH40S-60.3 SCH40S BW DN80x50"
    source = old_output(
        [{"type": "DN", "value": "80"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "SCH40S"}],
    )

    plan, _ = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert [item["ROLE"] for item in converted["ITEMS"]] == ["END_A", "END_B"]
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH40S"}
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH40S"}
    ]


def test_reducing_tee_uses_branch_topology_not_reducer_topology():
    topology, _ = MODULE.topology_from_text("REDUCINGTEE DN80x50 SCH40xSCH20")
    assert topology == "BRANCH"


def test_equal_branch_size_and_two_wall_thicknesses_follow_description_roles():
    description = "三通DN200x200 3.2x1.2"
    source = old_output(
        [{"type": "DN", "value": "200"}],
        [{"type": "MM", "value": "3.2"}, {"type": "MM", "value": "1.2"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "200"}],
            "THICKNESS": [{"type": "MM", "value": "3.2"}],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "DN", "value": "200"}],
            "THICKNESS": [{"type": "MM", "value": "1.2"}],
        },
    ]


def test_tee_with_one_explicit_size_keeps_it_on_main_only():
    description = "三通 DN200 3.2x1.2"
    source = old_output(
        [{"type": "DN", "value": "200"}],
        [{"type": "MM", "value": "3.2"}, {"type": "MM", "value": "1.2"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "200"}],
            "THICKNESS": [{"type": "MM", "value": "3.2"}],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [],
            "THICKNESS": [{"type": "MM", "value": "1.2"}],
        },
    ]


def test_equal_tee_with_equivalent_dn_keeps_dn_in_main_position():
    description = 'Tee 4"*4" SCH40*SCH40 SMLS,BW,A234 WPB,ASME B16.9 DN100'
    source = old_output(
        [{"type": "INCH", "value": "4"}, {"type": "DN", "value": "100"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)

    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["ROLE"] == "MAIN"
    assert converted["ITEMS"][0]["SIZE"] == [
        {"type": "INCH", "value": "4"},
        {"type": "DN", "value": "100"},
    ]


def test_reducer_expands_explicit_equal_thickness_to_both_ends():
    description = "偏心异径管 DN100x50 SCH40xSCH40 20 GB/T8163 BW"
    source = old_output(
        [{"type": "DN", "value": "100"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    plan, _ = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert [item["ROLE"] for item in converted["ITEMS"]] == ["END_A", "END_B"]
    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "SCHEDULE", "value": "SCH40"}]
    assert converted["ITEMS"][1]["THICKNESS"] == [{"type": "SCHEDULE", "value": "SCH40"}]


def test_single_position_keeps_all_explicit_equivalent_size_expressions_together():
    description = "管子;SMLS;BE;304;GB/T14976;HG/T20553(II);φ108X4mm DN100"
    source = old_output(
        [{"type": "OD", "value": "108"}, {"type": "DN", "value": "100"}],
        [{"type": "MM", "value": "4"}],
    )

    plan, _ = MODULE.deterministic_plan("SINGLE", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "SINGLE",
            "SIZE": [{"type": "OD", "value": "108"}, {"type": "DN", "value": "100"}],
            "THICKNESS": [{"type": "MM", "value": "4"}],
        }
    ]


def test_metric_component_specs_map_od_and_mm_to_both_branch_positions():
    description = "GB/T 14976;SH/T 3408 Red.Tee 48.3x2.77-26.7x2.87 BW S31603"
    source = old_output(
        [{"type": "OD", "value": "48.3"}, {"type": "OD", "value": "26.7"}],
        [{"type": "MM", "value": "2.77"}, {"type": "MM", "value": "2.87"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert [item["ROLE"] for item in converted["ITEMS"]] == ["MAIN", "BRANCH"]
    assert converted["ITEMS"][0]["SIZE"] == [{"type": "OD", "value": "48.3"}]
    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "MM", "value": "2.77"}]
    assert converted["ITEMS"][1]["SIZE"] == [{"type": "OD", "value": "26.7"}]
    assert converted["ITEMS"][1]["THICKNESS"] == [{"type": "MM", "value": "2.87"}]


def test_parallel_od_and_thickness_pairs_are_not_parsed_as_fractions():
    description = "异径三通 规格:Φ57×45/4.00×3.50"
    source = old_output(
        [{"type": "OD", "value": "57"}, {"type": "OD", "value": "45"}],
        [{"type": "MM", "value": "4"}, {"type": "MM", "value": "3.5"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"][0]["SIZE"] == [{"type": "OD", "value": "57"}]
    assert converted["ITEMS"][1]["SIZE"] == [{"type": "OD", "value": "45"}]
    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "MM", "value": "4"}]
    assert converted["ITEMS"][1]["THICKNESS"] == [{"type": "MM", "value": "3.5"}]


def test_branch_outlet_od_and_thickness_attach_to_branch():
    description = "对焊管接台;BW;F304;φ18X3mm;DN150x15"
    source = old_output(
        [
            {"type": "OD", "value": "18"},
            {"type": "DN", "value": "150"},
            {"type": "DN", "value": "15"},
        ],
        [{"type": "MM", "value": "3"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [{"type": "DN", "value": "150"}],
            "THICKNESS": [],
        },
        {
            "SCOPE": "BODY",
            "ROLE": "BRANCH",
            "SIZE": [{"type": "OD", "value": "18"}, {"type": "DN", "value": "15"}],
            "THICKNESS": [{"type": "MM", "value": "3"}],
        },
    ]


def test_mixed_mm_and_schedule_pair_maps_both_roles():
    description = "WELDOLET 18MMxSCH80 DN750X100"
    source = old_output(
        [{"type": "DN", "value": "750"}, {"type": "DN", "value": "100"}],
        [{"type": "MM", "value": "18"}, {"type": "SCHEDULE", "value": "SCH80"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "MM", "value": "18"}]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH80"}
    ]


def test_unanchored_old_size_item_does_not_auto_convert():
    description = "等径三通 DN50 THK=4mm"
    source = old_output(
        [{"type": "DN", "value": "50"}, {"type": "INCH", "value": "4"}],
        [{"type": "MM", "value": "4"}],
    )

    plan, reason = MODULE.deterministic_plan("BRANCH", description, source)

    assert plan is None
    assert "SIZE_ITEMS" in reason


def test_prefixed_hyphen_size_pair_maps_main_and_branch():
    description = 'Red.Tee DN25-DN20 CL3000 SW 1X0.75"'
    source = old_output(
        [
            {"type": "DN", "value": "25"},
            {"type": "DN", "value": "20"},
            {"type": "INCH", "value": "1"},
            {"type": "INCH", "value": "0.75"},
        ],
        [],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"][0]["SIZE"] == [
        {"type": "DN", "value": "25"},
        {"type": "INCH", "value": "1"},
    ]
    assert converted["ITEMS"][1]["SIZE"] == [
        {"type": "DN", "value": "20"},
        {"type": "INCH", "value": "0.75"},
    ]


def test_coupled_equal_tee_size_and_schedule_expand_both_roles():
    description = "EQUAL TEE DN200 S-10S X 200 S-10S"
    source = old_output(
        [{"type": "DN", "value": "200"}],
        [{"type": "SCHEDULE", "value": "SCH10S"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert [item["ROLE"] for item in converted["ITEMS"]] == ["MAIN", "BRANCH"]
    assert converted["ITEMS"][0]["SIZE"] == [{"type": "DN", "value": "200"}]
    assert converted["ITEMS"][1]["SIZE"] == [{"type": "DN", "value": "200"}]
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"}
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"}
    ]


def test_explicit_od_thickness_chain_maps_both_positions():
    description = "异径三通 Φ76X4MMXΦ45X3.5MM DN65x40"
    source = old_output(
        [
            {"type": "DN", "value": "65"},
            {"type": "DN", "value": "40"},
            {"type": "OD", "value": "76"},
            {"type": "OD", "value": "45"},
        ],
        [{"type": "MM", "value": "4"}, {"type": "MM", "value": "3.5"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)

    assert converted["ITEMS"][0]["SIZE"] == [
        {"type": "DN", "value": "65"},
        {"type": "OD", "value": "76"},
    ]
    assert converted["ITEMS"][1]["SIZE"] == [
        {"type": "DN", "value": "40"},
        {"type": "OD", "value": "45"},
    ]
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "MM", "value": "4"}
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "MM", "value": "3.5"}
    ]


def test_spaced_decimal_and_mixed_role_suffixes_are_normalized_for_mapping():
    description = "异径三通 THK=4. 5mmX3. 5mm DN65x40"
    source = old_output(
        [{"type": "DN", "value": "65"}, {"type": "DN", "value": "40"}],
        [{"type": "MM", "value": "4.5"}, {"type": "MM", "value": "3.5"}],
    )
    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None

    mixed_description = "异径三通 12mm(L)×SCH40(S) DN600X500"
    mixed_source = old_output(
        [{"type": "DN", "value": "600"}, {"type": "DN", "value": "500"}],
        [{"type": "MM", "value": "12"}, {"type": "SCHEDULE", "value": "SCH40"}],
    )
    mixed_plan, _ = MODULE.deterministic_plan("BRANCH", mixed_description, mixed_source)
    assert mixed_plan is not None


def test_schedule_pair_can_follow_a_glued_standard():
    description = "Reducer,ECC,ASME B16.9SCH40×SCH80 DN50X20"
    source = old_output(
        [{"type": "DN", "value": "50"}, {"type": "DN", "value": "20"}],
        [
            {"type": "SCHEDULE", "value": "SCH40"},
            {"type": "SCHEDULE", "value": "SCH80"},
        ],
    )

    plan, _ = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH40"}
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH80"}
    ]


def test_bare_dn_pair_is_accepted_only_when_it_matches_old_dn_labels():
    description = "异径三通TR PN16 50X40II;S30408;GB/T12459"
    source = old_output(
        [{"type": "DN", "value": "50"}, {"type": "DN", "value": "40"}],
        [],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert [item["ROLE"] for item in converted["ITEMS"]] == ["MAIN", "BRANCH"]


def test_product_and_glued_dn_pairs_map_without_word_boundaries():
    product = "钢制对焊支管座 WOL250x50-II-STD GB/T19326"
    source = old_output(
        [{"type": "DN", "value": "250"}, {"type": "DN", "value": "50"}],
        [{"type": "SCHEDULE", "value": "STD"}],
    )
    plan, _ = MODULE.deterministic_plan("BRANCH", product, source)
    assert plan is not None

    glued = "同心异径管,RC,BW,Sch.40DN20x10,GB/T12459"
    reducer_source = old_output(
        [{"type": "DN", "value": "20"}, {"type": "DN", "value": "10"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )
    reducer_plan, _ = MODULE.deterministic_plan("REDUCER", glued, reducer_source)
    assert reducer_plan is not None


def test_equal_tee_keeps_equivalent_size_expressions_in_position_roles():
    description = 'Tee 4"*4" SCH40*SCH40 BW,A234 WPB,ASME B16.9 DN100'
    source = old_output(
        [{"type": "INCH", "value": "4"}, {"type": "DN", "value": "100"}],
        [{"type": "SCHEDULE", "value": "SCH40"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0] == {
        "SCOPE": "BODY",
        "ROLE": "MAIN",
        "SIZE": [
            {"type": "INCH", "value": "4"},
            {"type": "DN", "value": "100"},
        ],
        "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
    }
    assert converted["ITEMS"][1] == {
        "SCOPE": "BODY",
        "ROLE": "BRANCH",
        "SIZE": [{"type": "INCH", "value": "4"}],
        "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
    }


def test_single_od_wall_spec_on_equal_tee_defaults_to_main():
    description = "GB/T14976;SH/T3408 Tee 33.4x2.77 BW S31603 DN25"
    source = old_output(
        [{"type": "OD", "value": "33.4"}, {"type": "DN", "value": "25"}],
        [{"type": "MM", "value": "2.77"}],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"] == [
        {
            "SCOPE": "BODY",
            "ROLE": "MAIN",
            "SIZE": [
                {"type": "OD", "value": "33.4"},
                {"type": "DN", "value": "25"},
            ],
            "THICKNESS": [{"type": "MM", "value": "2.77"}],
        },
    ]


def test_schedule_std_xs_and_abbreviated_pair_are_supported():
    description = "REDUCING TEE 3''x2'' SCH. STD x SCH. XS"
    source = old_output(
        [{"type": "INCH", "value": "3"}, {"type": "INCH", "value": "2"}],
        [
            {"type": "SCHEDULE", "value": "STD"},
            {"type": "SCHEDULE", "value": "XS"},
        ],
    )
    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None

    abbreviated = "三通 Sch80X80 THK=4.5X3.0mm DN25x25"
    abbreviated_source = old_output(
        [{"type": "DN", "value": "25"}],
        [
            {"type": "SCHEDULE", "value": "SCH80"},
            {"type": "MM", "value": "4.5"},
            {"type": "MM", "value": "3"},
        ],
    )
    abbreviated_plan, _ = MODULE.deterministic_plan(
        "BRANCH", abbreviated, abbreviated_source
    )
    assert abbreviated_plan is not None


def test_additional_explicit_pair_formats_map_without_guessing():
    inch_description = "REDUCER TEE STD SIZE36''X32''"
    inch_source = old_output(
        [{"type": "INCH", "value": "36"}, {"type": "INCH", "value": "32"}],
        [{"type": "SCHEDULE", "value": "STD"}],
    )
    inch_plan, _ = MODULE.deterministic_plan("BRANCH", inch_description, inch_source)
    assert inch_plan is not None

    glued_description = "异径三通 6.00mm×5.50mm100X50"
    glued_source = old_output(
        [{"type": "DN", "value": "100"}, {"type": "DN", "value": "50"}],
        [{"type": "MM", "value": "6"}, {"type": "MM", "value": "5.5"}],
    )
    glued_plan, _ = MODULE.deterministic_plan("BRANCH", glued_description, glued_source)
    assert glued_plan is not None


def test_multi_wall_reducer_keeps_all_explicit_end_a_walls():
    description = "Concentric Reducer 60.3x4.5/3.6 - 33.7x3.2 DN50X25"
    source = old_output(
        [
            {"type": "OD", "value": "60.3"},
            {"type": "OD", "value": "33.7"},
            {"type": "DN", "value": "50"},
            {"type": "DN", "value": "25"},
        ],
        [
            {"type": "MM", "value": "4.5"},
            {"type": "MM", "value": "3.6"},
            {"type": "MM", "value": "3.2"},
        ],
    )
    plan, _ = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "MM", "value": "4.5"},
        {"type": "MM", "value": "3.6"},
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "MM", "value": "3.2"}
    ]


def test_outlet_main_od_and_small_end_wall_suffix_are_mapped():
    description = "对焊支管座 外径219mm STD T=3.5mm(S) DN200X50"
    source = old_output(
        [
            {"type": "OD", "value": "219"},
            {"type": "DN", "value": "200"},
            {"type": "DN", "value": "50"},
        ],
        [{"type": "MM", "value": "3.5"}],
    )
    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["SIZE"] == [
        {"type": "OD", "value": "219"},
        {"type": "DN", "value": "200"},
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "MM", "value": "3.5"}
    ]


def test_duplicate_schedule_and_mm_pairs_map_to_both_positions():
    description = "异径三通 Sch10SX10S THK=3.6X2.9mm DN100×50"
    source = old_output(
        [{"type": "DN", "value": "100"}, {"type": "DN", "value": "50"}],
        [
            {"type": "SCHEDULE", "value": "SCH10S"},
            {"type": "SCHEDULE", "value": "SCH10S"},
            {"type": "MM", "value": "3.6"},
            {"type": "MM", "value": "2.9"},
        ],
    )

    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None
    converted = MODULE.convert_output(source, plan)
    assert converted["ITEMS"][0]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"},
        {"type": "MM", "value": "3.6"},
    ]
    assert converted["ITEMS"][1]["THICKNESS"] == [
        {"type": "SCHEDULE", "value": "SCH10S"},
        {"type": "MM", "value": "2.9"},
    ]


def test_glued_pressure_size_and_size_wall_pairs_use_trusted_old_values():
    pressure_source = old_output(
        [{"type": "DN", "value": "65"}, {"type": "DN", "value": "40"}],
        [],
    )
    plan, _ = MODULE.deterministic_plan(
        "BRANCH", "异径三通 SW CL300065x40", pressure_source
    )
    assert plan is not None

    wall_source = old_output(
        [{"type": "DN", "value": "300"}, {"type": "DN", "value": "150"}],
        [{"type": "MM", "value": "12.5"}, {"type": "MM", "value": "8.8"}],
    )
    wall_plan, _ = MODULE.deterministic_plan(
        "REDUCER", "DN300XDN15012.5mmX8.8mm 偏心异径管", wall_source
    )
    assert wall_plan is not None
    converted = MODULE.convert_output(wall_source, wall_plan)
    assert converted["ITEMS"][0]["THICKNESS"] == [{"type": "MM", "value": "12.5"}]
    assert converted["ITEMS"][1]["THICKNESS"] == [{"type": "MM", "value": "8.8"}]


def test_ocr_dn_pair_and_inherited_schedule_pair_are_normalized():
    description = "对焊管接台 DN8Ox20 S10S/S40S"
    source = old_output(
        [{"type": "DN", "value": "80"}, {"type": "DN", "value": "20"}],
        [
            {"type": "SCHEDULE", "value": "SCH10S"},
            {"type": "SCHEDULE", "value": "SCH40S"},
        ],
    )
    plan, _ = MODULE.deterministic_plan("BRANCH", description, source)
    assert plan is not None


def test_missing_explicit_role_wall_blocks_automatic_conversion():
    description = "偏心大小头 T=5.5mm(L)×T=4.5mm(S) Ф219×159×6.5/5.5 DN200X150"
    source = old_output(
        [
            {"type": "OD", "value": "219"},
            {"type": "OD", "value": "159"},
            {"type": "DN", "value": "200"},
            {"type": "DN", "value": "150"},
        ],
        [{"type": "MM", "value": "6.5"}, {"type": "MM", "value": "5.5"}],
    )
    plan, reason = MODULE.deterministic_plan("REDUCER", description, source)
    assert plan is None
    assert "4.5" in reason
