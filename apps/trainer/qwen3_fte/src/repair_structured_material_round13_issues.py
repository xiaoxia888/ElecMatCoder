#!/usr/bin/env python3
"""Repair confirmed issues from the thirteenth structured-material audit."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from apps.trainer.qwen3_fte.src.repair_structured_material_round10_issues import (
    _deduplicate_standards,
    _remove_standard,
    _set_body_material,
)


AUDIT_ISSUES = {
    26: "原文无ASME B16.11却错误补入AB1611",
    142: "ASTM A335材料标准漏提",
    198: "ASTM A672材料标准漏提",
}


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(row)
    text = str(repaired.get("input", ""))
    output = repaired.get("output", {})
    changes: list[str] = []

    if (
        re.search(r"GB\s*/\s*T\s*14383", text, re.IGNORECASE)
        and not re.search(r"(?:ASME\s*)?B\s*16\.?11", text, re.IGNORECASE)
        and _remove_standard(output, "AB1611")
    ):
        changes.append("REMOVE_UNEVIDENCED_AB1611")

    if re.search(r"ASTM\s+A335\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"F11", "P11"},
            standard="ASTM A335",
        ):
            changes.append("ASTM_A335_MATERIAL_STANDARD")

    if re.search(r"(?:ASTM\s*)?A672[-\s]*C65\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"C65"},
            standard="ASTM A672",
        ):
            changes.append("ASTM_A672_C65_MATERIAL_STANDARD")

    _deduplicate_standards(output)
    return repaired, changes


def repair_dataset(path: Path, execute: bool) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = []
    details = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        repaired, rules = repair_row(row)
        result.append(repaired)
        if rules:
            counts.update(rules)
            details.append(
                {
                    "source_index": index,
                    "input": row.get("input", ""),
                    "rules": rules,
                    "before": row.get("output", {}),
                    "after": repaired.get("output", {}),
                }
            )
    if execute and details:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return {
        "dataset": str(path),
        "rows": len(rows),
        "changed_rows": len(details),
        "rule_counts": dict(sorted(counts.items())),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="修复第十三轮材质抽查问题")
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--audit-sample", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reports = [repair_dataset(path.resolve(), args.execute) for path in args.dataset]
    repair_report = {
        "executed": args.execute,
        "datasets": reports,
        "total_changed_rows": sum(item["changed_rows"] for item in reports),
        "total_rule_counts": dict(
            sorted(
                sum(
                    (Counter(item["rule_counts"]) for item in reports),
                    Counter(),
                ).items()
            )
        ),
    }
    args.report.write_text(
        json.dumps(repair_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    samples = json.loads(args.audit_sample.read_text(encoding="utf-8"))
    issues = [
        {
            "sample_number": number,
            "source_index": samples[number - 1].get("source_index"),
            "input": samples[number - 1].get("input", ""),
            "output": samples[number - 1].get("output", {}),
            "issue": issue,
        }
        for number, issue in AUDIT_ISSUES.items()
    ]
    audit_report = {
        "round": 13,
        "sampling": "排除前十二轮输入，并优先抽取未覆盖的输出签名",
        "checked_rows": len(samples),
        "definite_issue_rows": len(issues),
        "passed_rows": len(samples) - len(issues),
        "definite_issue_rate": round(len(issues) / len(samples), 6),
        "target_issue_rate": 0.02,
        "ready_for_training": len(issues) / len(samples) < 0.02,
        "issues": issues,
    }
    args.audit_report.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **repair_report,
                "datasets": [
                    {k: v for k, v in item.items() if k != "details"}
                    for item in reports
                ],
                "audit": {
                    k: v for k, v in audit_report.items() if k != "issues"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
