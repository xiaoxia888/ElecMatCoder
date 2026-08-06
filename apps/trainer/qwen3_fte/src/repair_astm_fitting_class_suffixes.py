#!/usr/bin/env python3
"""Restore explicit ASTM A234/A403/A420 W/WX suffixes in v3 datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.trainer.qwen3_fte.src.convert_structured_material_v2_to_v3 import (
    compact_designation,
    extract_source_astm_fitting_class_designation,
    recover_source_astm_fitting_class_designation,
)


DATA_DIR = (
    ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范/结构化原始牌号"
)
DEFAULT_FILES = (
    DATA_DIR / "材质规范_结构化原始牌号_train.json",
    DATA_DIR / "材质规范_结构化原始牌号_val.json",
)
DEFAULT_REPORT = DATA_DIR / "材质规范_ASTM_W_WX后缀修复报告.json"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}根节点必须是数组")
    return rows


def dump_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def repair_rows(
    rows: list[dict[str, Any]],
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    candidates = 0
    already_correct = 0

    for index, row in enumerate(rows):
        source_text = str(row.get("input") or "")
        source_designation = extract_source_astm_fitting_class_designation(
            source_text
        )
        if not source_designation:
            continue
        candidates += 1

        materials = (row.get("output") or {}).get("MATERIAL") or []
        repaired_items: list[tuple[dict[str, Any], str, str]] = []
        for item in materials:
            before = str(item.get("VALUE") or "")
            after = recover_source_astm_fitting_class_designation(
                source_text,
                before,
            )
            if after and after != before:
                repaired_items.append((item, before, after))

        if len(repaired_items) == 1:
            item, before, after = repaired_items[0]
            item["VALUE"] = after
            changes.append(
                {
                    "source": source_name,
                    "source_index": index,
                    "input": source_text,
                    "before": before,
                    "after": after,
                    "evidence": source_designation,
                }
            )
            continue

        expected = compact_designation(source_designation)
        matching = [
            str(item.get("VALUE") or "")
            for item in materials
            if compact_designation(str(item.get("VALUE") or "")) == expected
        ]
        if not repaired_items and len(matching) == 1:
            already_correct += 1
            continue

        unresolved.append(
            {
                "source": source_name,
                "source_index": index,
                "input": source_text,
                "values": [str(item.get("VALUE") or "") for item in materials],
                "expected": source_designation,
                "reason": "无法唯一定位待修复的ASTM管件材质项",
            }
        )

    return rows, {
        "rows": len(rows),
        "candidates": candidates,
        "modified": len(changes),
        "already_correct": already_correct,
        "unresolved": unresolved,
        "changes": changes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", type=Path, default=list(DEFAULT_FILES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write", action="store_true", help="写回数据文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"write": args.write, "files": {}}
    total_candidates = 0
    total_modified = 0
    total_unresolved = 0

    for path in args.files:
        rows, file_report = repair_rows(load_rows(path), path.name)
        if args.write:
            dump_json(path, rows)
        report["files"][path.name] = file_report
        total_candidates += file_report["candidates"]
        total_modified += file_report["modified"]
        total_unresolved += len(file_report["unresolved"])

    report["summary"] = {
        "candidates": total_candidates,
        "modified": total_modified,
        "already_correct": sum(
            item["already_correct"] for item in report["files"].values()
        ),
        "unresolved": total_unresolved,
    }
    dump_json(args.report, report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    if total_unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
