#!/usr/bin/env python3
"""Repair confirmed issues from the tenth structured-material audit."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_ISSUES = {
    10: "核心中国牌号022Cr17Ni12Mo2被等价牌号316L覆盖",
    43: "ASTM A691材料等级1.25Cr漏提",
    83: "ASTM A182锻件牌号304漏掉F前缀",
    101: "产品规范GB/T 9711漏提",
    128: "产品规范DIN 11850漏提",
    141: "ASTM B564材料标准漏提且N04400发生OCR错误",
    149: "明确材质20被产品标识CF415覆盖",
    152: "原文不存在的GB/T 3087被错误补入",
    171: "GB/T 1340I中的OCR字母I未修复为数字1",
    174: "ASTM A182 Grade F304被错误标为S30400",
    177: "ASTM A240材料标准漏提",
    178: "核心中国牌号06Cr19Ni10被括号内等价牌号304覆盖",
    180: "尺寸系列Ⅱ被错误标为材料CLASS=Gr.II",
    189: "WPHC276(N10276)牌号缺少右括号",
}


def _body_materials(output: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        material
        for material in output.get("MATERIAL", [])
        if isinstance(material, dict) and material.get("PART") == "BODY"
    ]


def _standard_bodies(output: dict[str, Any]) -> list[str]:
    return [
        str(item.get("BODY", ""))
        for item in output.get("STANDARD", [])
        if isinstance(item, dict)
    ]


def _deduplicate_standards(output: dict[str, Any]) -> None:
    seen: set[str] = set()
    standards: list[dict[str, str]] = []
    for item in output.get("STANDARD", []):
        if not isinstance(item, dict):
            continue
        body = str(item.get("BODY", "")).strip()
        if body and body not in seen:
            standards.append({"BODY": body})
            seen.add(body)
    output["STANDARD"] = standards


def _append_standard(output: dict[str, Any], body: str) -> bool:
    if body in _standard_bodies(output):
        return False
    output.setdefault("STANDARD", []).append({"BODY": body})
    return True


def _replace_standard(output: dict[str, Any], old: str, new: str) -> bool:
    changed = False
    for item in output.get("STANDARD", []):
        if isinstance(item, dict) and item.get("BODY") == old:
            item["BODY"] = new
            changed = True
    if changed:
        _deduplicate_standards(output)
    return changed


def _remove_standard(output: dict[str, Any], body: str) -> bool:
    standards = output.get("STANDARD", [])
    kept = [
        item
        for item in standards
        if not (isinstance(item, dict) and item.get("BODY") == body)
    ]
    if len(kept) == len(standards):
        return False
    output["STANDARD"] = kept
    return True


def _set_body_material(
    output: dict[str, Any],
    *,
    current_grades: set[str],
    standard: str | None = None,
    grade: str | None = None,
    material_class: str | None = None,
) -> bool:
    changed = False
    for material in _body_materials(output):
        if str(material.get("GRADE", "")) not in current_grades:
            continue
        if standard is not None and material.get("STANDARD") != standard:
            material["STANDARD"] = standard
            changed = True
        if grade is not None and material.get("GRADE") != grade:
            material["GRADE"] = grade
            changed = True
        if material_class is not None and material.get("CLASS") != material_class:
            material["CLASS"] = material_class
            changed = True
    return changed


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repaired = copy.deepcopy(row)
    text = str(repaired.get("input", ""))
    output = repaired.get("output", {})
    changes: list[str] = []

    if (
        "022Cr17Ni12Mo2" in text
        and re.search(r"材料或性能等级\s*[：:]\s*316L\b", text, re.IGNORECASE)
        and _set_body_material(
            output,
            current_grades={"316L"},
            grade="022Cr17Ni12Mo2",
        )
    ):
        changes.append("PREFER_EXPLICIT_022CR17NI12MO2")

    if re.search(
        r"(?:ASTM\s*)?A\s*691\s*Gr\.?\s*1\.25Cr\b",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={""},
            standard="ASTM A691",
            grade="1.25Cr",
        ):
            changes.append("ASTM_A691_1_25CR_GRADE")

    if re.search(
        r"A[.\s]?182(?:\s*Grade)?[.\s]*F[.\s]*304\b",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"304", "S30400"},
            standard="ASTM A182",
            grade="F304",
        ):
            changes.append("ASTM_A182_F304")
    elif re.search(r"A[.\s]?182[.\s]+304\b", text, re.IGNORECASE):
        if _set_body_material(
            output,
            current_grades={"304"},
            standard="ASTM A182",
            grade="F304",
        ):
            changes.append("ASTM_A182_IMPLICIT_F304")

    if (
        re.search(r"GB\s*/\s*T\s*9711\b", text, re.IGNORECASE)
        and not any(
            body.startswith("GBT9711") for body in _standard_bodies(output)
        )
    ):
        if _append_standard(output, "GBT9711"):
            changes.append("GBT9711_MISSING")

    if re.search(r"DIN\s*17455\s*;\s*11850\b", text, re.IGNORECASE):
        if _append_standard(output, "DIN11850"):
            changes.append("DIN11850_MISSING")

    if (
        re.search(r"(?:ASTM[-\s]*)?B\s*564\b", text, re.IGNORECASE)
        and re.search(r"UNS\s*N[O0]4400\b", text, re.IGNORECASE)
    ):
        if _set_body_material(
            output,
            current_grades={"NO4400", "N04400"},
            standard="ASTM B564",
            grade="N04400",
        ):
            changes.append("ASTM_B564_N04400")

    if (
        "CF415" in text
        and re.search(
            r"L\s*=\s*\d+(?:\.\d+)?\s*mm\s+20\s+DN\d+",
            text,
            re.IGNORECASE,
        )
    ):
        if _set_body_material(
            output,
            current_grades={"CF415"},
            grade="20",
        ):
            changes.append("EXPLICIT_20_OVERRIDES_CF415")

    if "3087" not in text and _remove_standard(output, "GBT3087"):
        changes.append("REMOVE_UNEVIDENCED_GBT3087")

    if re.search(r"GB\s*/\s*T\s*1340I\b", text, re.IGNORECASE):
        if _replace_standard(output, "GBT1340I", "GBT13401"):
            changes.append("GBT13401_OCR_I_TO_ONE")

    if re.search(
        r"06Cr19Ni10\s*[（(]\s*304\s*[）)]",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"304"},
            grade="06Cr19Ni10",
        ):
            changes.append("PREFER_EXPLICIT_06CR19NI10")

    if (
        re.search(
            r"\d+\s*[x×*]\s*\d+\s*Ⅱ\s+304\b",
            text,
            re.IGNORECASE,
        )
        and _set_body_material(
            output,
            current_grades={"304"},
            material_class="",
        )
    ):
        changes.append("REMOVE_SIZE_SERIES_FROM_MATERIAL_CLASS")

    if _set_body_material(
        output,
        current_grades={"WPHC276(N10276"},
        grade="WPHC276(N10276)",
    ):
        changes.append("CLOSE_WPHC276_PARENTHESES")

    if re.search(
        r"ASTM\s+A240\s+(?:GRADE\s+)?S32205\b",
        text,
        re.IGNORECASE,
    ):
        if _set_body_material(
            output,
            current_grades={"S32205"},
            standard="ASTM A240",
        ):
            changes.append("ASTM_A240_S32205_STANDARD")

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
    passed = []
    for sample_number, row in enumerate(rows, start=1):
        item = {
            "sample_number": sample_number,
            "source_index": row.get("source_index"),
            "input": row.get("input", ""),
            "output": row.get("output", {}),
        }
        if sample_number in AUDIT_ISSUES:
            item["issue"] = AUDIT_ISSUES[sample_number]
            issues.append(item)
        else:
            passed.append(item)

    checked = len(rows)
    report = {
        "round": 10,
        "sample_source": str(sample_path),
        "sampling": "排除前九轮输入，并优先抽取未覆盖的输出签名",
        "checked_rows": checked,
        "definite_issue_rows": len(issues),
        "passed_rows": len(passed),
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
        description="修复第十轮结构化原始牌号抽查确认问题"
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
