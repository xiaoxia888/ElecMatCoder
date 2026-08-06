#!/usr/bin/env python3
"""Apply reviewed flange TYPE annotation proposals with strict source guards."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--change-log", type=Path, required=True)
    args = parser.parse_args()

    datasets = {"train": load_json(args.train), "val": load_json(args.val)}
    original_counts = {split: len(rows) for split, rows in datasets.items()}
    approved = set(args.category)
    report = load_json(args.report)
    proposals = report.get("去重后的建议记录", [])
    changes: list[dict[str, Any]] = []

    for proposal in proposals:
        issues = set(proposal.get("问题类别", []))
        if not issues or not issues.issubset(approved):
            continue

        split = str(proposal["来源数据集"])
        if split not in datasets:
            raise ValueError(f"未知来源数据集: {split}")
        index = int(proposal["source_index"])
        rows = datasets[split]
        if not 0 <= index < len(rows):
            raise IndexError(f"{split} source_index越界: {index}")

        row = rows[index]
        expected_input = proposal.get("原始描述", "")
        expected_output = proposal.get("修正前标签", {})
        if row.get("input", "") != expected_input:
            raise ValueError(f"{split}第{index}条原文已变化，停止写回")
        if row.get("output", {}) != expected_output:
            raise ValueError(f"{split}第{index}条标签已变化，停止写回")

        repaired_output = deepcopy(proposal["建议修正标签"])
        row["output"] = repaired_output
        changes.append(
            {
                "来源数据集": split,
                "source_index": index,
                "原始描述": expected_input,
                "问题类别": proposal["问题类别"],
                "修改前标签": expected_output,
                "修改后标签": repaired_output,
                "中文原因": proposal.get("中文原因", []),
            }
        )

    if not changes:
        raise ValueError("审核报告中没有匹配到可写回的已审核类别")
    if any(len(datasets[split]) != original_counts[split] for split in datasets):
        raise RuntimeError("写回过程中数据集行数发生变化")

    write_json(args.train, datasets["train"])
    write_json(args.val, datasets["val"])
    category_counts = {
        category: sum(category in change["问题类别"] for change in changes)
        for category in args.category
    }
    change_log = {
        "说明": "仅写回用户已审核通过的法兰TYPE问题类别，未删除或重排样本。",
        "已应用类别": args.category,
        "修改条数": len(changes),
        "来源统计": {
            split: sum(change["来源数据集"] == split for change in changes)
            for split in datasets
        },
        "分类统计": category_counts,
        "修改明细": changes,
    }
    write_json(args.change_log, change_log)
    print(
        json.dumps(
            {
                "修改条数": len(changes),
                "来源统计": change_log["来源统计"],
                "分类统计": category_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
