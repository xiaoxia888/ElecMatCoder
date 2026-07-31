#!/usr/bin/env python3
"""Repair confirmed issues from the eleventh structured-material audit."""

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
    _replace_standard,
    _set_body_material,
)


AUDIT_ISSUES = {
    3: "裸3410在锻制管件语境中被错误归为GB/T，应为SH/T 3410",
    14: "WP12CL.2被截断为WP12CL，CL.2不参与本任务牌号编码",
    43: "不规范牌号22Cr17Ni12Mo2未补全为022Cr17Ni12Mo2",
    79: "产品规范编码GB/T12459I未按统一格式输出为GBT12459I",
    98: "牌号310s大小写未统一为310S",
    135: "ASTM A240材料标准漏提",
    138: "ASTM B423材料标准漏提",
    163: "UNS N04400未统一为牌号N04400",
    171: "ASTM A240材料标准漏提",
}


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(row)
    text = str(repaired.get("input", ""))
    output = repaired.get("output", {})
    changes: list[str] = []

    bare_3410 = (
        re.search(r"(?<![A-Z0-9/])3410(?!\d)", text, re.IGNORECASE)
        and not re.search(
            r"(?:SH|GB|HG|NB)\s*/?\s*T?\s*3410",
            text,
            re.IGNORECASE,
        )
    )
    if bare_3410 and _replace_standard(output, "GBT3410", "SHT3410"):
        changes.append("BARE_3410_TO_SHT3410")

    if re.search(
        r"A\s*234\s*GR\.?\s*WP12\s*CL\.?\s*2\b",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"WP12CL", "WP12CL.2"},
            standard="ASTM A234",
            grade="WP12",
        ):
            changes.append("ASTM_A234_WP12_REMOVE_CL2")

    if re.search(
        r"(?<!0)22Cr17Ni12Mo2\s*[（(]\s*316L\s*[）)]",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"316L", "22Cr17Ni12Mo2"},
            grade="022Cr17Ni12Mo2",
        ):
            changes.append("NORMALIZE_22CR17NI12MO2")

    if _replace_standard(output, "GB/T12459I", "GBT12459I"):
        changes.append("CANONICALIZE_GBT12459I")

    if re.search(r"(?<![A-Z0-9])310s(?![A-Z0-9])", text):
        if _set_body_material(
            output,
            current_grades={"310s"},
            grade="310S",
        ):
            changes.append("NORMALIZE_310S_CASE")

    if re.search(
        r"(?<![A-Z0-9])(?:ASTM\s*)?A\s*240\s*(?:GR\.?|GRADE)?",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"TP304", "TP316", "S32205", "304H"},
            standard="ASTM A240",
        ):
            changes.append("ASTM_A240_MATERIAL_STANDARD")

    if re.search(r"ASTM\s+B\s*423\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"N08825"},
            standard="ASTM B423",
        ):
            changes.append("ASTM_B423_MATERIAL_STANDARD")

    for material in output.get("MATERIAL", []):
        if (
            isinstance(material, dict)
            and material.get("GRADE") == "UNS N04400"
            and material.get("STANDARD") in {"ASTM B165", "ASTM B564"}
        ):
            material["GRADE"] = "N04400"
            changes.append("NORMALIZE_UNS_N04400")

    _deduplicate_standards(output)
    return repaired, changes


def repair_dataset(path: Path, *, execute: bool) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    repaired_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        repaired, rules = repair_row(row)
        repaired_rows.append(repaired)
        if not rules:
            continue
        rule_counts.update(rules)
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
        "rule_counts": dict(sorted(rule_counts.items())),
        "details": details,
    }


def write_audit_report(sample_path: Path, report_path: Path) -> None:
    rows = json.loads(sample_path.read_text(encoding="utf-8"))
    issues = []
    for sample_number, row in enumerate(rows, start=1):
        if sample_number not in AUDIT_ISSUES:
            continue
        issues.append(
            {
                "sample_number": sample_number,
                "source_index": row.get("source_index"),
                "input": row.get("input", ""),
                "output": row.get("output", {}),
                "issue": AUDIT_ISSUES[sample_number],
            }
        )
    checked = len(rows)
    report = {
        "round": 11,
        "sample_source": str(sample_path),
        "sampling": "排除前十轮输入，并优先抽取未覆盖的输出签名",
        "checked_rows": checked,
        "definite_issue_rows": len(issues),
        "passed_rows": checked - len(issues),
        "definite_issue_rate": round(len(issues) / checked, 6) if checked else 0,
        "target_issue_rate": 0.02,
        "ready_for_training": bool(checked and len(issues) / checked < 0.02),
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复第十一轮结构化原始牌号抽查确认问题"
    )
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
        write_audit_report(
            args.audit_sample.expanduser().resolve(),
            args.audit_report.expanduser().resolve(),
        )

    summary = {
        **report,
        "datasets": [
            {key: value for key, value in item.items() if key != "details"}
            for item in reports
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
