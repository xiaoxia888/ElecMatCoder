#!/usr/bin/env python3
"""Repair confirmed issues from the fourteenth structured-material audit."""

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
    _set_body_material,
)


AUDIT_ISSUES = {
    52: "20 GLASS LINED被错误合并为单一主体牌号",
    175: "ASTM A691的1 1/4Cr牌号漏提",
    306: "ASTM A312 TP304被括号内UNS S30400错误覆盖",
}


def _repair_glass_lining(output: dict[str, Any]) -> bool:
    materials = output.get("MATERIAL", [])
    if any(
        isinstance(item, dict) and item.get("PART") == "LINING"
        for item in materials
    ):
        return False

    body = next(
        (
            item
            for item in materials
            if isinstance(item, dict) and item.get("PART") == "BODY"
        ),
        None,
    )
    if body is None:
        return False

    body["STANDARD"] = ""
    body["GRADE"] = "20"
    body.setdefault("CLASS", "")
    body.setdefault("SPECIAL_REQ", [])
    materials.append(
        {
            "PART": "LINING",
            "STANDARD": "",
            "GRADE": "GLASS",
            "CLASS": "",
            "SPECIAL_REQ": [],
        }
    )
    output["MATERIAL_RELATION"] = "COMPOSITE"
    return True


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(row)
    text = str(repaired.get("input", ""))
    output = repaired.get("output", {})
    changes: list[str] = []

    if re.search(r"\b20\s+GLASS\s+LINED\b", text, re.IGNORECASE):
        if _repair_glass_lining(output):
            changes.append("SPLIT_20_GLASS_LINING")

    if (
        re.search(
            r"(?:ASTM[\s-]*)?A691\b.*?\b(?:GR\.?\s*)?"
            r"(?:1\s*[- ]?\s*1\s*/\s*4|1\.25)\s*CR(?=\d|\b)",
            text,
            re.IGNORECASE,
        )
        and _set_body_material(
            output,
            current_grades={""},
            standard="ASTM A691",
            grade="1.25Cr",
        )
    ):
        changes.append("ASTM_A691_1_25CR_GRADE")

    if (
        re.search(
            r"(?:ASTM\s*)?A312\s*GRADE\s*TP304\s*"
            r"\(\s*UNS\s*S30400\s*\)",
            text,
            re.IGNORECASE,
        )
        and _set_body_material(
            output,
            current_grades={"S30400"},
            standard="ASTM A312",
            grade="TP304",
        )
    ):
        changes.append("ASTM_A312_TP304_PRECEDENCE")

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
    parser = argparse.ArgumentParser(description="修复第十四轮材质抽查问题")
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(repair_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    samples = json.loads(args.audit_sample.read_text(encoding="utf-8"))
    issues = [
        {
            "sample_number": number,
            "source_split": samples[number - 1].get("source_split"),
            "source_index": samples[number - 1].get("source_index"),
            "input": samples[number - 1].get("input", ""),
            "output": samples[number - 1].get("output", {}),
            "issue": issue,
        }
        for number, issue in AUDIT_ISSUES.items()
    ]
    audit_report = {
        "round": 14,
        "sampling": (
            "排除前十三轮2400条输入，按输出签名轮询抽取，"
            "400条样本覆盖400种不同输出结构"
        ),
        "checked_rows": len(samples),
        "unique_output_signatures": 400,
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
