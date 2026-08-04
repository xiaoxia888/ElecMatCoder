#!/usr/bin/env python3
"""Repair high-confidence annotation conflicts in the size training dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


INSTALLATION_DN_RE = re.compile(
    r"^(?:不锈钢|碳钢)管道安装\s+(?:1\.规格材质：)?\s*(\d+(?:\.\d+)?)\s*mm\b",
    re.IGNORECASE,
)
CLASS_PRESSURE_RE = re.compile(
    r"(?<![A-Z])CL(?:ASS)?\s*[.:=-]?\s*(150|300|600|900|1500|2500|3000|6000)(?!\d)",
    re.IGNORECASE,
)
PN_PRESSURE_RE = re.compile(r"(?<![A-Z])PN\s*[.:=-]?\s*(\d+)(?![.\d])", re.IGNORECASE)


def _canonical_number(value: str) -> str:
    if "." not in value:
        return value
    return value.rstrip("0").rstrip(".")


def _conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("input") or "")].append((index, row))

    result: list[dict[str, Any]] = []
    for input_text, members in groups.items():
        outputs: dict[str, list[int]] = defaultdict(list)
        for index, row in members:
            signature = json.dumps(row.get("output"), ensure_ascii=False, sort_keys=True)
            outputs[signature].append(index)
        if len(outputs) <= 1:
            continue
        result.append(
            {
                "input": input_text,
                "variants": [
                    {"source_indexes": indexes, "output": json.loads(signature)}
                    for signature, indexes in outputs.items()
                ],
            }
        )
    return result


def repair_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired = deepcopy(rows)
    changes: list[dict[str, Any]] = []
    conflicts_before = _conflicts(repaired)

    for index, row in enumerate(repaired):
        input_text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            continue

        installation_match = INSTALLATION_DN_RE.search(input_text)
        size_items = output.get("SIZE_ITEMS")
        if installation_match and isinstance(size_items, list) and len(size_items) == 1:
            expected_value = _canonical_number(installation_match.group(1))
            item = size_items[0]
            if (
                isinstance(item, dict)
                and item.get("type") == "OD"
                and str(item.get("value") or "") == expected_value
            ):
                before = deepcopy(size_items)
                item["type"] = "DN"
                changes.append(
                    {
                        "source_index": index,
                        "category": "管道安装毫米公称尺寸误标为OD",
                        "input": input_text,
                        "before": before,
                        "after": deepcopy(size_items),
                    }
                )

        class_match = CLASS_PRESSURE_RE.search(input_text)
        if class_match and not str(output.get("PRESSURE") or "").strip():
            pressure = f"CL{class_match.group(1)}"
            output["PRESSURE"] = pressure
            changes.append(
                {
                    "source_index": index,
                    "category": "明确CL压力漏标",
                    "input": input_text,
                    "before": "",
                    "after": pressure,
                }
            )

        pn_match = PN_PRESSURE_RE.search(input_text)
        if pn_match and not str(output.get("PRESSURE") or "").strip():
            pressure = f"PN{pn_match.group(1)}"
            output["PRESSURE"] = pressure
            changes.append(
                {
                    "source_index": index,
                    "category": "明确整数PN压力漏标",
                    "input": input_text,
                    "before": "",
                    "after": pressure,
                }
            )

    conflicts_after = _conflicts(repaired)
    report = {
        "rows": len(rows),
        "changed_rows": len({item["source_index"] for item in changes}),
        "change_items": len(changes),
        "changes_by_category": {
            category: sum(item["category"] == category for item in changes)
            for category in sorted({item["category"] for item in changes})
        },
        "conflicting_inputs_before": len(conflicts_before),
        "conflicting_inputs_after": len(conflicts_after),
        "changes": changes,
        "unresolved_conflicts": conflicts_after,
    }
    return repaired, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset root must be a list")

    repaired, report = repair_dataset(rows)
    report_path = args.report or args.dataset.with_name(f"{args.dataset.stem}_标注冲突修复报告.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.execute:
        temporary_path = args.dataset.with_suffix(f"{args.dataset.suffix}.tmp")
        temporary_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(args.dataset)

    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "report": str(report_path),
                "execute": args.execute,
                "rows": report["rows"],
                "changed_rows": report["changed_rows"],
                "changes_by_category": report["changes_by_category"],
                "conflicting_inputs_before": report["conflicting_inputs_before"],
                "conflicting_inputs_after": report["conflicting_inputs_after"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
