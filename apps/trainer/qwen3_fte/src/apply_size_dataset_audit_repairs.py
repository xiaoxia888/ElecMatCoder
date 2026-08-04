#!/usr/bin/env python3
"""Apply selected, reviewed size-dataset audit proposals with strict guards."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def apply_selected_repairs(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = deepcopy(rows)
    changes: list[dict[str, Any]] = []

    proposals_by_category = report.get("待确认修改", {})
    for category in categories:
        if category not in proposals_by_category:
            raise ValueError(f"审核报告中不存在问题类别: {category}")
        for proposal in proposals_by_category[category]:
            index = int(proposal["source_index"])
            if not 0 <= index < len(repaired):
                raise IndexError(f"source_index 越界: {index}")

            row = repaired[index]
            field = str(proposal["修改字段"])
            current = row.get("output", {}).get(field)
            if row.get("input", "") != proposal.get("原始描述", ""):
                raise ValueError(f"第 {index} 条原文已变更，停止写回")
            if current != proposal.get("当前标签"):
                raise ValueError(f"第 {index} 条 {field} 已变更，停止写回")

            proposed = deepcopy(proposal["建议标签"])
            row.setdefault("output", {})[field] = proposed
            changes.append(
                {
                    "source_index": index,
                    "问题类别": category,
                    "原始描述": row.get("input", ""),
                    "修改字段": field,
                    "修改前": current,
                    "修改后": proposed,
                    "修改原因": proposal.get("中文原因", ""),
                }
            )

    return repaired, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--change-log", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    repaired, changes = apply_selected_repairs(rows, report, args.category)

    args.dataset.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    change_log = {
        "说明": "仅写回用户指定且已确认的问题类别。",
        "已应用类别": args.category,
        "修改条数": len(changes),
        "分类统计": {
            category: sum(change["问题类别"] == category for change in changes)
            for category in args.category
        },
        "修改明细": changes,
    }
    args.change_log.write_text(
        json.dumps(change_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: change_log[key] for key in ("修改条数", "分类统计")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
