#!/usr/bin/env python3
"""Apply audited OLET BODY fixes to the fitting type dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_type_olet_annotations import DEFAULT_DATASET, SCOPE_RE, audit_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复管件种类数据集中的 OLET BODY 标注")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--apply", action="store_true", help="实际写回数据集")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("管件数据集顶层必须是数组")

    audits = {
        index: audit_row(row, index)
        for index, row in enumerate(rows)
        if isinstance(row, dict) and SCOPE_RE.search(str(row.get("input") or ""))
    }
    mismatch_indexes = {
        index
        for index, item in audits.items()
        if item["status"] == "body_mismatch"
    }
    manual_indexes = {
        index
        for index, item in audits.items()
        if item["status"] == "manual_review"
    }
    if manual_indexes and len(manual_indexes) != 2:
        raise RuntimeError(f"预期删除 2 条冲突数据，实际发现 {len(manual_indexes)} 条")

    transitions: Counter[str] = Counter()
    repaired_rows: list[dict[str, Any]] = []
    deleted_inputs: list[str] = []
    for index, row in enumerate(rows):
        audit = audits.get(index)
        if index in manual_indexes:
            deleted_inputs.append(str(row.get("input") or ""))
            continue
        if index in mismatch_indexes and audit:
            type_value = row["output"]["TYPE"]
            current_body = str(type_value.get("BODY") or "")
            expected_body = str(audit["expected_body"])
            transitions[f"{current_body} -> {expected_body}"] += 1
            type_value["BODY"] = expected_body
        repaired_rows.append(row)

    summary = {
        "dataset": str(dataset_path),
        "before_rows": len(rows),
        "after_rows": len(repaired_rows),
        "updated_rows": sum(transitions.values()),
        "deleted_rows": len(deleted_inputs),
        "transitions": dict(transitions),
        "deleted_inputs": deleted_inputs,
        "applied": args.apply,
    }
    if args.apply and (transitions or deleted_inputs):
        dataset_path.write_text(
            json.dumps(repaired_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
