#!/usr/bin/env python3
"""Apply the reviewed fitting type label corrections with strict validation."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"
REVIEW_DIR = DATA_DIR / "管件标注审查_20260805"
DEFAULT_REVIEW = REVIEW_DIR / "管件_train_val_基础错误标签审核_20260805.json"
DEFAULT_REPORT = REVIEW_DIR / "管件_train_val_基础错误标签修复结果_20260805.json"
DATASETS = {
    "train": DATA_DIR / "管件_train.json",
    "val": DATA_DIR / "管件_val.json",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def resolve_fnpt_priority(current: dict[str, Any]) -> dict[str, Any]:
    """Keep the explicit design selection FNPT and discard process text SW."""
    proposed = copy.deepcopy(current)
    conn = proposed["TYPE"]["CONN"]
    if "FNPT" not in conn:
        raise ValueError("疑问项当前标签缺少FNPT，无法按审核结论处理")
    proposed["TYPE"]["CONN"] = [value for value in conn if value != "SW"]
    return proposed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = load_json(args.review)
    rows_by_dataset = {name: load_json(path) for name, path in DATASETS.items()}
    changes: list[dict[str, Any]] = []
    unchanged_manual: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    operations: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for item in review["明确建议修改"]:
        operations.append((item, item["建议修正标签"], "已审核明确修正"))

    for item in review["需要人工判断"]:
        proposed = resolve_fnpt_priority(item["当前标签"])
        if proposed == item["当前标签"]:
            unchanged_manual.append(
                {
                    "来源数据集": item["来源数据集"],
                    "source_index": item["source_index"],
                    "原始描述": item["原始描述"],
                    "标签": item["当前标签"],
                    "处理结论": "当前已仅保留FNPT，无需修改",
                }
            )
            continue
        operations.append((item, proposed, "设计选型FNPT优先，删除工艺描述SW"))

    for item, proposed, conclusion in operations:
        dataset = item["来源数据集"]
        index = item["source_index"]
        key = (dataset, index)
        if key in seen:
            raise ValueError(f"审核项重复: {dataset}[{index}]")
        seen.add(key)

        row = rows_by_dataset[dataset][index]
        expected_input = item["原始描述"]
        expected_before = item.get("修正前标签", item.get("当前标签"))
        if row.get("input") != expected_input:
            raise ValueError(f"原始描述已变化，停止写回: {dataset}[{index}]")
        if row.get("output") != expected_before:
            raise ValueError(f"当前标签已变化，停止写回: {dataset}[{index}]")

        before = copy.deepcopy(row["output"])
        row["output"] = copy.deepcopy(proposed)
        changes.append(
            {
                "来源数据集": dataset,
                "source_index": index,
                "原始描述": expected_input,
                "修正前标签": before,
                "修正后标签": proposed,
                "处理结论": conclusion,
            }
        )

    for dataset, path in DATASETS.items():
        write_json(path, rows_by_dataset[dataset])

    report = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "审核文件": str(args.review),
        "处理规则": "明确建议项按审核结果修正；FNPT与施工承插焊并存时，以设计选型FNPT为准并删除CONN中的SW。",
        "统计": {
            "实际修改条数": len(changes),
            "明确建议修改条数": sum(
                change["处理结论"] == "已审核明确修正" for change in changes
            ),
            "FNPT优先修正条数": sum(
                change["处理结论"] == "设计选型FNPT优先，删除工艺描述SW"
                for change in changes
            ),
            "疑问项无需修改条数": len(unchanged_manual),
            "训练集条数": len(rows_by_dataset["train"]),
            "验证集条数": len(rows_by_dataset["val"]),
        },
        "修改明细": changes,
        "疑问项无需修改": unchanged_manual,
    }
    write_json(args.report, report)
    print(json.dumps(report["统计"], ensure_ascii=False))
    print(args.report)


if __name__ == "__main__":
    main()
