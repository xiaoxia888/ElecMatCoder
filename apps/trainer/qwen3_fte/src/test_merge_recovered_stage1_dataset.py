import csv
import json
import tempfile
import unittest
from pathlib import Path

from apps.trainer.qwen3_fte.src.merge_recovered_stage1_dataset import (
    merge_recovered_stage1_dataset,
    merge_with_existing,
    prepare_recovered,
)


def _candidate(input_text: str, output: dict, sequence: str, category: str = "管件") -> dict:
    return {
        "record": {
            "instruction": "任务提示词",
            "input": input_text,
            "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        },
        "source_sequence": sequence,
        "category": category,
    }


class RecoveredDatasetMergeTest(unittest.TestCase):
    def test_prepare_recovered_treats_array_order_as_same_label(self) -> None:
        first = _candidate(
            "相同描述",
            {"MATERIAL": [], "STANDARD": [{"BODY": "A"}, {"BODY": "B"}]},
            "1",
        )
        reordered = _candidate(
            "相同描述",
            {"MATERIAL": [], "STANDARD": [{"BODY": "B"}, {"BODY": "A"}]},
            "2",
        )

        result = prepare_recovered([first, reordered], task="材质规范")

        self.assertEqual([item["source_sequence"] for item in result.accepted], ["1"])
        self.assertEqual(result.order_only_duplicate_groups, 1)
        self.assertEqual(result.conflicts, [])

    def test_prepare_recovered_excludes_every_row_from_true_conflict_group(self) -> None:
        with_size = _candidate(
            "冲突描述",
            {"ITEMS": [{"SCOPE": "BODY", "ROLE": "SINGLE", "SIZE": [{"type": "DN", "value": "50"}], "THICKNESS": []}], "LENGTH": "", "PRESSURE": ""},
            "10",
        )
        empty = _candidate(
            "冲突描述",
            {"ITEMS": [], "LENGTH": "", "PRESSURE": ""},
            "11",
        )

        result = prepare_recovered([with_size, empty], task="尺寸壁厚磅级")

        self.assertEqual(result.accepted, [])
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0]["材料描述"], "冲突描述")
        self.assertEqual(result.conflicts[0]["源序号"], ["10", "11"])

    def test_merge_with_existing_keeps_old_rows_and_only_appends_new_inputs(self) -> None:
        old = [
            {"instruction": "旧提示", "input": "已有描述", "output": '{"old":true}'},
        ]
        recovered = [
            _candidate("已有描述", {"new": "ignored"}, "20"),
            _candidate("新增描述", {"new": "added"}, "21"),
        ]

        merged, additions, skipped = merge_with_existing(old, recovered)

        self.assertEqual(merged[0], old[0])
        self.assertEqual([row["input"] for row in additions], ["新增描述"])
        self.assertEqual([row["input"] for row in merged], ["已有描述", "新增描述"])
        self.assertEqual(skipped, 1)

    def test_end_to_end_writes_five_splits_three_merges_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recovered_path = root / "recovered.json"
            csv_path = root / "results.csv"
            old_root = root / "old"
            output_dir = root / "merged"
            recovered_path.write_text(
                json.dumps(
                    [
                        {
                            "original_input": "管件原始描述",
                            "input": "管件格式描述",
                            "output": {
                                "TYPE": {"BODY": "弯头"},
                                "MATERIAL": [{"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": []}],
                                "STANDARD": [{"BODY": "GBT8163"}],
                                "ITEMS": [],
                                "LENGTH": "",
                                "PRESSURE": "",
                            },
                        },
                        {
                            "original_input": "法兰原始描述",
                            "input": "法兰格式描述",
                            "output": {
                                "TYPE": {"BODY": "盲板法兰", "CONN": [], "SEAL": ["RF"]},
                                "MATERIAL": [],
                                "STANDARD": [],
                                "ITEMS": [],
                                "LENGTH": "",
                                "PRESSURE": "CL150",
                            },
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            headers = ["序号", "分类", "原始描述", "TYPE_原始结果", "SIZE_原始结果", "THICKNESS_原始结果", "PRESSURE_原始结果", "MATERIAL_原始结果", "STANDARD_原始结果"]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerow({"序号": "1", "分类": "管件", "原始描述": "管件原始描述", "TYPE_原始结果": "弯头"})
                writer.writerow({"序号": "2", "分类": "法兰", "原始描述": "法兰原始描述", "TYPE_原始结果": "盲板法兰", "PRESSURE_原始结果": "CL150"})

            fixtures = {
                "种类/0806/种类_train.json": [{"instruction": "种类提示", "input": "已有种类", "output": '{"CATEGORY":"直管","TYPE":{"BODY":"直管"}}'}],
                "种类/0806/种类_val.json": [],
                "材质规范/0814/材质规范_train.json": [{"instruction": "材质提示", "input": "已有材质", "output": '{"MATERIAL":[],"STANDARD":[]}'}],
                "材质规范/0814/材质规范_val.json": [],
                "尺寸壁厚磅级/0817/尺寸壁厚磅级_train.json": [{"instruction": "尺寸提示", "input": "已有尺寸", "output": '{"ITEMS":[],"LENGTH":"","PRESSURE":""}'}],
                "尺寸壁厚磅级/0817/尺寸壁厚磅级_val.json": [],
            }
            for relative, data in fixtures.items():
                path = old_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            report = merge_recovered_stage1_dataset(
                recovered_path=recovered_path,
                source_csv_path=csv_path,
                old_root=old_root,
                output_dir=output_dir,
            )

            expected = {
                "恢复数据拆分/直管.json",
                "恢复数据拆分/法兰.json",
                "恢复数据拆分/管件.json",
                "恢复数据拆分/材质规范.json",
                "恢复数据拆分/尺寸壁厚磅级.json",
                "种类.json",
                "材质规范.json",
                "尺寸壁厚磅级.json",
                "标签冲突待复核.json",
                "合并报告.json",
            }
            self.assertEqual(
                {str(path.relative_to(output_dir)) for path in output_dir.rglob("*.json")},
                expected,
            )
            self.assertEqual(report["任务统计"]["种类"]["新增数量"], 2)
            self.assertEqual(report["任务统计"]["材质规范"]["新增数量"], 2)
            self.assertEqual(report["任务统计"]["尺寸壁厚磅级"]["新增数量"], 2)
            type_rows = json.loads((output_dir / "种类.json").read_text(encoding="utf-8"))
            self.assertEqual(len(type_rows), 3)
            self.assertEqual(json.loads(type_rows[1]["output"])["CATEGORY"], "管件")


if __name__ == "__main__":
    unittest.main()
