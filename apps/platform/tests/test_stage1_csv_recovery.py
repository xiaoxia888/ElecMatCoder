import csv
import json
import tempfile
import unittest
from pathlib import Path


class Stage1CsvRecoveryTest(unittest.TestCase):
    def test_parse_structural_fields_rebuilds_shared_items(self):
        from apps.platform.stage1_csv_recovery import parse_structural_fields

        recovered = parse_structural_fields(
            size_text="RUN: DN100 ; BRANCH: DN50 ; LENGTH: 6000",
            thickness_text="RUN: SCH40 ; BRANCH: 3.5MM",
            pressure_text="PN16",
        )
        self.assertEqual(
            recovered,
            {
                "ITEMS": [
                    {
                        "SCOPE": "BODY",
                        "ROLE": "RUN",
                        "SIZE": [{"type": "DN", "value": "100"}],
                        "THICKNESS": [{"type": "SCHEDULE", "value": "SCH40"}],
                    },
                    {
                        "SCOPE": "BODY",
                        "ROLE": "BRANCH",
                        "SIZE": [{"type": "DN", "value": "50"}],
                        "THICKNESS": [{"type": "MM", "value": "3.5"}],
                    },
                ],
                "LENGTH": "6000",
                "PRESSURE": "PN16",
            },
        )

    def test_parse_structural_fields_preserves_repeated_roles(self):
        from apps.platform.stage1_csv_recovery import parse_structural_fields

        recovered = parse_structural_fields(
            size_text="SINGLE: OD377 | DN450 ; SINGLE: DN350",
            thickness_text="",
            pressure_text="",
        )

        self.assertEqual(len(recovered["ITEMS"]), 2)
        self.assertEqual(
            recovered["ITEMS"][0]["SIZE"],
            [{"type": "OD", "value": "377"}, {"type": "DN", "value": "450"}],
        )
        self.assertEqual(
            recovered["ITEMS"][1]["SIZE"],
            [{"type": "DN", "value": "350"}],
        )

    def test_parse_flange_type_assigns_seal_without_swapping_body(self):
        from apps.platform.stage1_csv_recovery import parse_type_field

        self.assertEqual(
            parse_type_field("承插焊法兰;RF", category="法兰"),
            {"BODY": "承插焊法兰", "CONN": [], "SEAL": ["RF"]},
        )

    def test_parse_fitting_type_assigns_geometry_connection_and_manufacture(self):
        from apps.platform.stage1_csv_recovery import parse_type_field

        self.assertEqual(
            parse_type_field("弯头;90;LR;BW;SMLS", category="管件"),
            {
                "BODY": "弯头",
                "GEOMETRY": {"ANGLE": "90", "RADIUS": "LR"},
                "FLANGE_STYLE": "",
                "MANU": ["SMLS"],
                "CONN": ["BW"],
            },
        )

    def test_parse_fitting_type_keeps_unknown_connection_descriptor_after_body(self):
        from apps.platform.batch_export import _format_field_value
        from apps.platform.stage1_csv_recovery import parse_type_field

        source = "偏心异径短节;MNPT;BLE/TSE(MNPT)"
        recovered = parse_type_field(source, category="管件")

        self.assertEqual(recovered["FLANGE_STYLE"], "")
        self.assertEqual(recovered["CONN"], ["MNPT", "BLE/TSE(MNPT)"])
        self.assertEqual(_format_field_value("TYPE", recovered), source)

    def test_parse_fitting_type_supports_flange_style_before_body(self):
        from apps.platform.batch_export import _format_field_value
        from apps.platform.stage1_csv_recovery import parse_type_field

        source = "FLANGED;弯头;45;WELDED"
        recovered = parse_type_field(source, category="管件")

        self.assertEqual(recovered["FLANGE_STYLE"], "FLANGED")
        self.assertEqual(recovered["BODY"], "弯头")
        self.assertEqual(_format_field_value("TYPE", recovered), source)

    def test_parse_material_splits_special_requirement_suffix(self):
        from apps.platform.stage1_csv_recovery import parse_material_field

        self.assertEqual(
            parse_material_field("ASTM A105CE ; BOLT:ASTM A193 B8M"),
            [
                {"PART": "BODY", "VALUE": "ASTM A105", "SPECIAL_REQ": ["CE"]},
                {"PART": "BOLT", "VALUE": "ASTM A193 B8M", "SPECIAL_REQ": []},
            ],
        )

    def test_parse_standard_removes_display_only_categories(self):
        from apps.platform.stage1_csv_recovery import parse_standard_field

        self.assertEqual(
            parse_standard_field("GBT12459（生产） ; GBT8163（制造）"),
            [{"BODY": "GBT12459"}, {"BODY": "GBT8163"}],
        )

    def test_recover_row_prefers_unique_reference_labels(self):
        from apps.platform.stage1_csv_recovery import RecoveryReferences, recover_row

        expected_type = {"BODY": "参考标签", "CONN": [], "SEAL": ["RF"]}
        references = RecoveryReferences(
            type_by_signature={"显示文本": expected_type},
            material_by_signature={},
            standard_by_signature={},
            structure_by_signature={},
        )
        row = {
            "分类": "法兰",
            "原始描述": "原文",
            "原始总编码": "CODE",
            "TYPE_原始结果": "显示文本",
            "SIZE_原始结果": "SINGLE: DN50",
            "THICKNESS_原始结果": "",
            "PRESSURE_原始结果": "PN16",
            "MATERIAL_原始结果": "ASTM A105CE",
            "STANDARD_原始结果": "GBT12459（生产）",
        }

        recovered, methods = recover_row(
            row,
            references=references,
            preprocess=lambda text: f"格式化:{text}",
        )

        self.assertEqual(recovered["original_input"], "原文")
        self.assertEqual(recovered["input"], "格式化:原文")
        self.assertEqual(recovered["output"]["TYPE"], expected_type)
        self.assertEqual(methods["TYPE"], "reference")
        self.assertEqual(recovered["output"]["PRESSURE"], "PN16")

    def test_recover_csv_skips_rows_without_any_encoding_result_and_reports_them(self):
        from apps.platform.stage1_csv_recovery import RecoveryReferences, recover_csv

        headers = [
            "序号", "分类", "原始描述", "原始总编码", "TYPE_原始结果",
            "SIZE_原始结果", "THICKNESS_原始结果", "PRESSURE_原始结果",
            "MATERIAL_原始结果", "STANDARD_原始结果",
        ]
        complete = {
            "序号": "1", "分类": "法兰", "原始描述": "法兰;RF;DN50",
            "原始总编码": "F50", "TYPE_原始结果": "承插焊法兰;RF",
            "SIZE_原始结果": "SINGLE: DN50", "THICKNESS_原始结果": "",
            "PRESSURE_原始结果": "PN16", "MATERIAL_原始结果": "ASTM A105",
            "STANDARD_原始结果": "GBT12459（生产）",
        }
        missing = {header: "" for header in headers}
        missing.update({"序号": "2", "分类": "管件", "原始描述": "失败记录"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            dataset = root / "dataset.json"
            report = root / "report.json"
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows([complete, missing])

            summary = recover_csv(
                source,
                dataset,
                report,
                references=RecoveryReferences.empty(),
                preprocess=lambda text: text,
            )

            recovered = json.loads(dataset.read_text(encoding="utf-8"))
            written_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(len(recovered), 1)
            self.assertEqual(summary["source_rows"], 2)
            self.assertEqual(summary["recovered_rows"], 1)
            self.assertEqual(summary["skipped_rows"], 1)
            self.assertEqual(written_report["skipped"][0]["序号"], "2")


if __name__ == "__main__":
    unittest.main()
