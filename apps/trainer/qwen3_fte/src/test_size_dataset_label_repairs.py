from __future__ import annotations

import unittest

from apps.trainer.qwen3_fte.src.apply_size_dataset_audit_repairs import (
    apply_selected_repairs,
)
from apps.trainer.qwen3_fte.src.audit_size_dataset_second_round import (
    audit,
    extract_explicit_dns,
    extract_explicit_inches,
    extract_explicit_mm_thicknesses,
    extract_explicit_ods,
    extract_schedules,
    extract_verified_bare_pipe_inches,
    filter_dirty_dn_evidence,
)
from apps.trainer.qwen3_fte.src.generate_size_error_contrast_datasets import (
    BUILDERS as SIZE_AUGMENTATION_BUILDERS,
    validate_datasets as validate_size_augmentation_datasets,
)
from apps.trainer.qwen3_fte.src.normalize_size_length_labels import (
    normalize_dataset,
    normalize_length,
)
from apps.trainer.qwen3_fte.src.repair_size_dataset_annotation_conflicts import (
    repair_dataset,
)


def _row(
    text: str,
    *,
    size_items: list[dict[str, str]] | None = None,
    thickness_items: list[dict[str, str]] | None = None,
    length: str = "",
    pressure: str = "",
) -> dict:
    return {
        "input": text,
        "output": {
            "SIZE_ITEMS": size_items or [],
            "LENGTH": length,
            "THICKNESS_ITEMS": thickness_items or [],
            "PRESSURE": pressure,
        },
    }


class NormalizeSizeLengthLabelsTest(unittest.TestCase):
    def test_normalizes_length_units_without_losing_decimals(self) -> None:
        self.assertEqual(normalize_length("328.061"), "328.061MM")
        self.assertEqual(normalize_length("32.8061CM"), "328.061MM")
        self.assertEqual(normalize_length("0.328061M"), "328.061MM")

    def test_extracts_strong_en_pipe_trailing_length(self) -> None:
        rows = [
            _row(
                "Pipe, X2CrNi19-11, SMLS, BE, 219.1x2.9, EN 10216-5 200mm",
                size_items=[{"type": "OD", "value": "219.1"}],
            )
        ]
        repaired, report = normalize_dataset(rows, fix_en_pipe_trailing_mm=True)
        self.assertEqual(repaired[0]["output"]["LENGTH"], "200MM")
        self.assertEqual(report["en_pipe_length_fixes"], 1)

    def test_extracts_flanged_pipe_trailing_length(self) -> None:
        rows = [
            _row(
                "法兰管, PTFE lined GB/T 8163-20, RF, CL 150, SH/T 3406, "
                "HG/T 20538, DN50, S-40 745.55mm",
                size_items=[{"type": "DN", "value": "50"}],
            )
        ]
        repaired, report = normalize_dataset(rows, fix_flanged_pipe_trailing_mm=True)
        self.assertEqual(repaired[0]["output"]["LENGTH"], "745.55MM")
        self.assertEqual(report["flanged_pipe_length_fixes"], 1)


class RepairSizeDatasetAnnotationConflictsTest(unittest.TestCase):
    def test_installation_mm_is_nominal_size_not_od(self) -> None:
        rows = [
            _row(
                "不锈钢管道安装 80mm,PIPE,SMLS,A312 TP304,ASME B36.19M,SCH10S",
                size_items=[{"type": "OD", "value": "80"}],
            )
        ]
        repaired, _ = repair_dataset(rows)
        self.assertEqual(repaired[0]["output"]["SIZE_ITEMS"], [{"type": "DN", "value": "80"}])

    def test_preserves_explicit_inch_and_dn_evidence(self) -> None:
        rows = [
            _row(
                'CS PIPE 2" SCH40 ASTM A106 Gr.B,ASME B36.10M DN50',
                size_items=[{"type": "INCH", "value": "2"}, {"type": "DN", "value": "50"}],
            )
        ]
        repaired, _ = repair_dataset(rows)
        self.assertEqual(
            repaired[0]["output"]["SIZE_ITEMS"],
            [{"type": "INCH", "value": "2"}, {"type": "DN", "value": "50"}],
        )

    def test_fills_unambiguous_pressure_anchors(self) -> None:
        rows = [
            _row("法兰管 DN50 CL 150", size_items=[{"type": "DN", "value": "50"}]),
            _row("搪玻璃管 DN25 PN10", size_items=[{"type": "DN", "value": "25"}]),
        ]
        repaired, _ = repair_dataset(rows)
        self.assertEqual(repaired[0]["output"]["PRESSURE"], "CL150")
        self.assertEqual(repaired[1]["output"]["PRESSURE"], "PN10")


class SizeDatasetSecondRoundAuditTest(unittest.TestCase):
    def test_inch_parser_handles_chains_fractions_and_dn_boundary(self) -> None:
        self.assertEqual(
            [value for _, value in extract_explicit_inches('Reducer 10X6"')],
            ["10", "6"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_inches('Reducer 2X0.75"')],
            ["2", "0.75"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_inches("NIPPLE 1-1/2''")],
            ["1-1/2"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_inches("A105 1/2 in")],
            ["1/2"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_inches('DN65*2"')],
            ["2"],
        )
        self.assertEqual(extract_explicit_inches('DN100*80 "Sch'), [])

    def test_dn_parser_excludes_wall_thickness_after_dn(self) -> None:
        self.assertEqual(
            [value for _, value in extract_explicit_dns("DN100x80")],
            ["100", "80"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_dns("DN1200x900x12/10")],
            ["1200", "900"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_dns("DN200x13")],
            ["200"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_dns("DN65*2\" SCH40")],
            ["65"],
        )
        self.assertEqual(extract_explicit_dns("R=1.5DN20#"), [])
        self.assertEqual(extract_explicit_dns("2.规格:DN803.连接方式"), [])
        self.assertEqual(extract_explicit_dns("DN403.5毫米"), [])
        self.assertEqual(extract_explicit_dns("DN808字盲板"), [])
        self.assertEqual(
            [value for _, value in extract_explicit_dns("DN500X20.0mm")],
            ["500"],
        )

    def test_dirty_dn_equal_to_explicit_od_is_not_proposed(self) -> None:
        dirty_text = "2.规格:DN57  Ф57×3.5"
        self.assertEqual(
            filter_dirty_dn_evidence(
                extract_explicit_dns(dirty_text),
                extract_explicit_ods(dirty_text),
            ),
            [],
        )

        valid_text = "2.规格:DN50  Ф57×3.5"
        self.assertEqual(
            [
                value
                for _, value in filter_dirty_dn_evidence(
                    extract_explicit_dns(valid_text),
                    extract_explicit_ods(valid_text),
                )
            ],
            ["50"],
        )

    def test_schedule_parser_preserves_s_suffix_and_inherited_values(self) -> None:
        self.assertEqual(
            [value for _, value in extract_schedules("Sch10SXSch40S")],
            ["10S", "40S"],
        )
        self.assertEqual(
            [value for _, value in extract_schedules("Sch10SX10S")],
            ["10S", "10S"],
        )
        self.assertEqual(
            [value for _, value in extract_schedules("SCH40SHG/T20553")],
            ["40S"],
        )
        self.assertEqual(extract_schedules("Sch THK=3.5mm"), [])

    def test_explicit_od_and_mm_anchors(self) -> None:
        self.assertEqual(
            [value for _, value in extract_explicit_ods("φ114.3x6.02 OD:60.3")],
            ["114.3", "60.3"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_ods("φ114. 3×3.2")],
            ["114.3"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_mm_thicknesses("THK=10.0X4.5mm T=3.2")],
            ["10", "4.5", "3.2"],
        )
        self.assertEqual(
            [value for _, value in extract_explicit_mm_thicknesses("THK=3. 5mm")],
            ["3.5"],
        )
        self.assertEqual(extract_explicit_mm_thicknesses("THK=3.54.0mm"), [])

    def test_applies_only_selected_audit_category(self) -> None:
        rows = [_row('PIPE 2" SCH40', size_items=[], thickness_items=[])]
        report = audit(rows)
        repaired, changes = apply_selected_repairs(
            rows,
            report,
            ["明确英制尺寸漏标"],
        )
        self.assertEqual(repaired[0]["output"]["SIZE_ITEMS"], [{"type": "INCH", "value": "2"}])
        self.assertEqual(repaired[0]["output"]["THICKNESS_ITEMS"], [])
        self.assertEqual(len(changes), 1)

    def test_size_augmentation_has_markers_and_excludes_dirty_dn57(self) -> None:
        datasets = {
            filename: builder()
            for filename, builder in SIZE_AUGMENTATION_BUILDERS.items()
        }
        report = validate_size_augmentation_datasets(datasets)
        self.assertEqual(report["总样本数"], 376)
        for rows in datasets.values():
            for row in rows:
                self.assertEqual(row["来源"], "数据增强")
                self.assertTrue(row["数据增强标识"])
                self.assertNotIn(
                    {"type": "DN", "value": "57"},
                    row["output"]["SIZE_ITEMS"],
                )
                length = row["output"]["LENGTH"]
                self.assertTrue(not length or length.endswith("MM"))

    def test_bare_pipe_inch_requires_matching_od_evidence(self) -> None:
        self.assertEqual(
            [value for _, value in extract_verified_bare_pipe_inches("碳素无缝钢管 3 89*5.5 PIPE")],
            ["3"],
        )
        self.assertEqual(
            [value for _, value in extract_verified_bare_pipe_inches("镀锌无缝钢管 0.5 22*3.5 PIPE")],
            ["0.5"],
        )
        self.assertEqual(
            extract_verified_bare_pipe_inches("碳素无缝钢管 15 89*5.5 PIPE"),
            [],
        )

    def test_size_proposal_keeps_mixed_items_in_source_order(self) -> None:
        rows = [
            _row(
                'GB/T 8163;SH/T 3408 Red.Tee 60.3x3.91-26.7x3.91 BW 20 2X0.75"',
                size_items=[
                    {"type": "OD", "value": "60.3"},
                    {"type": "OD", "value": "26.7"},
                ],
            )
        ]
        report = audit(rows)
        proposed = report["待确认修改"]["明确英制尺寸漏标"][0]["建议标签"]
        self.assertEqual(
            proposed,
            [
                {"type": "OD", "value": "60.3"},
                {"type": "OD", "value": "26.7"},
                {"type": "INCH", "value": "2"},
                {"type": "INCH", "value": "0.75"},
            ],
        )

    def test_thickness_proposals_keep_source_order(self) -> None:
        mm_first = audit(
            [
                _row(
                    "PIPE THK=3.5mm SCH40",
                    thickness_items=[{"type": "SCHEDULE", "value": "SCH40"}],
                )
            ]
        )
        self.assertEqual(
            mm_first["待确认修改"]["明确毫米壁厚漏标"][0]["建议标签"],
            [
                {"type": "MM", "value": "3.5"},
                {"type": "SCHEDULE", "value": "SCH40"},
            ],
        )

        schedule_first = audit(
            [
                _row(
                    "PIPE SCH40 THK=3.5mm",
                    thickness_items=[{"type": "MM", "value": "3.5"}],
                )
            ]
        )
        self.assertEqual(
            schedule_first["待确认修改"]["明确SCH壁厚漏标"][0]["建议标签"],
            [
                {"type": "SCHEDULE", "value": "SCH40"},
                {"type": "MM", "value": "3.5"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
