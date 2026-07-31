#!/usr/bin/env python3
"""Repair confirmed issues from the twelfth structured-material audit."""

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
    9: "ASTM A691材料标准漏提，1 1/4Cr未统一为1.25Cr",
    15: "ASTM A815材料标准漏提",
    33: "镀锌后缀被错误并入WPB牌号",
    37: "ASTM B622材料标准漏提",
    82: "ASTM A420材料标准被重复写入产品规范STANDARD",
}


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(row)
    text = str(repaired.get("input", ""))
    output = repaired.get("output", {})
    changes: list[str] = []

    if (
        re.search(
            r"1\s+1/4Cr\s*CL22\s*A691(?:/A691M)?",
            text,
            re.IGNORECASE,
        )
        and _set_body_material(
            output,
            current_grades={"1 1/4Cr", "1-1/4Cr"},
            standard="ASTM A691",
            grade="1.25Cr",
        )
    ):
        changes.append("ASTM_A691_1_25CR")

    if re.search(r"ASTM\s+A815\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"S32205", "WPS32205"},
            standard="ASTM A815",
        ):
            changes.append("ASTM_A815_MATERIAL_STANDARD")

    if (
        re.search(r"A234\s+WPB\s*Galv", text, re.IGNORECASE)
        and _set_body_material(
            output,
            current_grades={"WPBGalv"},
            standard="ASTM A234",
            grade="WPB",
        )
    ):
        changes.append("ASTM_A234_WPB_REMOVE_GALV_SUFFIX")

    if re.search(r"(?:ASTM\s*)?B622\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"N10276"},
            standard="ASTM B622",
        ):
            changes.append("ASTM_B622_MATERIAL_STANDARD")

    if (
        any(
            material.get("STANDARD") == "ASTM A420"
            for material in output.get("MATERIAL", [])
            if isinstance(material, dict)
        )
        and _remove_standard(output, "ASTM420")
    ):
        changes.append("REMOVE_DUPLICATE_ASTM420_PRODUCT_STANDARD")

    _deduplicate_standards(output)
    return repaired, changes


def repair_dataset(path: Path, *, execute: bool) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    repaired_rows = []
    details = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        repaired, rules = repair_row(row)
        repaired_rows.append(repaired)
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
            json.dumps(repaired_rows, ensure_ascii=False, indent=2) + "\n",
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


def write_audit_report(sample_path: Path, report_path: Path) -> None:
    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    issues = [
        {
            "sample_number": number,
            "source_index": rows[number - 1].get("source_index"),
            "input": rows[number - 1].get("input", ""),
            "output": rows[number - 1].get("output", {}),
            "issue": issue,
        }
        for number, issue in AUDIT_ISSUES.items()
    ]
    report = {
        "round": 12,
        "sample_source": str(sample_path),
        "sampling": "排除前十一轮输入，并优先抽取未覆盖的输出签名",
        "checked_rows": len(rows),
        "definite_issue_rows": len(issues),
        "passed_rows": len(rows) - len(issues),
        "definite_issue_rate": round(len(issues) / len(rows), 6),
        "target_issue_rate": 0.02,
        "ready_for_training": False,
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="修复第十二轮材质抽查问题")
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--audit-sample", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reports = [
        repair_dataset(path.expanduser().resolve(), execute=args.execute)
        for path in args.dataset
    ]
    report = {
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_sample and args.audit_report:
        write_audit_report(args.audit_sample, args.audit_report)
    print(
        json.dumps(
            {
                **report,
                "datasets": [
                    {k: v for k, v in item.items() if k != "details"}
                    for item in reports
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
