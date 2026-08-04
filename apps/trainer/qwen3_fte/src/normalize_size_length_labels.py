#!/usr/bin/env python3
"""Normalize non-empty LENGTH labels to canonical millimetre strings."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


LENGTH_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*(MM|CM|M)?$", re.IGNORECASE)
EN_PIPE_TRAILING_LENGTH_RE = re.compile(
    r"^Pipe,.*?,\s*\d+(?:\.\d+)?\s*[xX×*]\s*\d+(?:\.\d+)?,"
    r"\s*EN\s*1021[67]-\d+\s+(\d+(?:\.\d+)?)\s*mm\s*$",
    re.IGNORECASE,
)
FLANGED_PIPE_TRAILING_LENGTH_RE = re.compile(
    r"^法兰管\s*,\s*PTFE\s+lined\b.*?\bDN\s*\d+\s*,\s*S-40\s+"
    r"(\d+(?:\.\d+)?)\s*mm\s*$",
    re.IGNORECASE,
)


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def normalize_length(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    match = LENGTH_RE.fullmatch(text)
    if not match:
        raise ValueError(f"unsupported LENGTH label: {value!r}")

    try:
        number = Decimal(match.group(1))
    except InvalidOperation as exc:
        raise ValueError(f"invalid LENGTH number: {value!r}") from exc

    unit = (match.group(2) or "MM").upper()
    if unit == "CM":
        number *= Decimal("10")
    elif unit == "M":
        number *= Decimal("1000")

    return f"{_format_decimal(number)}MM"


def normalize_dataset(
    rows: list[dict[str, Any]],
    *,
    fix_en_pipe_trailing_mm: bool = False,
    fix_flanged_pipe_trailing_mm: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    non_empty = 0
    en_pipe_length_fixes = 0
    flanged_pipe_length_fixes = 0

    for index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(output, dict):
            errors.append({"source_index": index, "reason": "output不是对象"})
            continue

        before = output.get("LENGTH", "")
        if fix_en_pipe_trailing_mm and not str(before or "").strip():
            source_text = str(row.get("input") or "").strip()
            match = EN_PIPE_TRAILING_LENGTH_RE.fullmatch(source_text)
            if match:
                after = f"{match.group(1)}MM"
                changes.append(
                    {
                        "source_index": index,
                        "category": "EN管尾部毫米长度漏标",
                        "input": source_text,
                        "before": before,
                        "after": after,
                    }
                )
                output["LENGTH"] = after
                before = after
                en_pipe_length_fixes += 1

        if fix_flanged_pipe_trailing_mm and not str(before or "").strip():
            source_text = str(row.get("input") or "").strip()
            match = FLANGED_PIPE_TRAILING_LENGTH_RE.fullmatch(source_text)
            if match:
                after = f"{match.group(1)}MM"
                changes.append(
                    {
                        "source_index": index,
                        "category": "法兰管尾部毫米长度漏标",
                        "input": source_text,
                        "before": before,
                        "after": after,
                    }
                )
                output["LENGTH"] = after
                before = after
                flanged_pipe_length_fixes += 1

        if str(before or "").strip():
            non_empty += 1
        try:
            after = normalize_length(before)
        except ValueError as exc:
            errors.append(
                {
                    "source_index": index,
                    "category": "长度单位格式不统一",
                    "input": row.get("input", ""),
                    "before": before,
                    "reason": str(exc),
                }
            )
            continue

        if before != after:
            changes.append(
                {
                    "source_index": index,
                    "input": row.get("input", ""),
                    "before": before,
                    "after": after,
                }
            )
            output["LENGTH"] = after

    report = {
        "rows": len(rows),
        "non_empty_length_rows": non_empty,
        "changed_rows": len(changes),
        "en_pipe_length_fixes": en_pipe_length_fixes,
        "flanged_pipe_length_fixes": flanged_pipe_length_fixes,
        "error_rows": len(errors),
        "changes": changes,
        "errors": errors,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--fix-en-pipe-trailing-mm",
        action="store_true",
        help="将 Pipe ODxTHK, EN 10216/10217, 尾部数字mm 统一标为长度",
    )
    parser.add_argument(
        "--fix-flanged-pipe-trailing-mm",
        action="store_true",
        help="将法兰管 DN... S-40 尾部数字mm 统一标为长度",
    )
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset root must be a list")

    normalized_rows, report = normalize_dataset(
        rows,
        fix_en_pipe_trailing_mm=args.fix_en_pipe_trailing_mm,
        fix_flanged_pipe_trailing_mm=args.fix_flanged_pipe_trailing_mm,
    )
    report_path = args.report or args.dataset.with_name(f"{args.dataset.stem}_长度标签规范化报告.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if report["error_rows"]:
        raise ValueError(f"found {report['error_rows']} invalid LENGTH labels; see {report_path}")

    if args.execute:
        temporary_path = args.dataset.with_suffix(f"{args.dataset.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(normalized_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(args.dataset)

    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "report": str(report_path),
                "execute": args.execute,
                "rows": report["rows"],
                "non_empty_length_rows": report["non_empty_length_rows"],
                "changed_rows": report["changed_rows"],
                "en_pipe_length_fixes": report["en_pipe_length_fixes"],
                "flanged_pipe_length_fixes": report["flanged_pipe_length_fixes"],
                "error_rows": report["error_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
