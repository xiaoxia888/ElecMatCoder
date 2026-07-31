#!/usr/bin/env python3
"""Reduce size-only duplicates for a selected single-material label."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "output"
    / "按8类拆分数据集"
    / "材质规范"
    / "结构化原始牌号"
)
DEFAULT_INPUT = DEFAULT_DATA_DIR / "材质规范_结构化原始牌号_train.json"


SIZE_PATTERNS = [
    re.compile(
        r"\bDN\s*[IIL]?\s*\d+(?:\.\d+)?"
        r"(?:\s*[X×*]\s*(?:DN\s*)?[IIL]?\s*\d+(?:\.\d+)?){0,3}",
        re.IGNORECASE,
    ),
    re.compile(
        r'(?<![A-Z0-9])\d+(?:\.\d+)?\s*["”″]'
        r'(?:\s*[X×*]\s*\d+(?:\.\d+)?\s*["”″]){0,3}',
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Z0-9])(?:OD|ID|D|Φ|Ф)\s*[:=]?\s*\d+(?:\.\d+)?"
        r"(?:\s*[X×*]\s*\d+(?:\.\d+)?){0,3}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Z0-9/])(?:THK|WT|壁厚)\s*[:=]?\s*\d+(?:\.\d+)?"
        r"(?:\s*[X×*/]\s*\d+(?:\.\d+)?){0,3}\s*(?:MM)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Z0-9/])T\s*[:=]\s*\d+(?:\.\d+)?"
        r"(?:\s*[X×*/]\s*\d+(?:\.\d+)?){0,3}\s*(?:MM)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:SCH|S-)\s*\d+[A-Z]?"
        r"(?:\s*[X×*/]\s*(?:SCH|S-)?\s*\d+[A-Z]?){0,3}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:PN|CL|CLASS)\s*\.?\s*\d+(?:\.\d+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Z0-9/])(?:L|LENGTH|长度)\s*[:=]\s*\d+(?:\.\d+)?\s*(?:MM|M)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Z0-9])\d+(?:\.\d+)?"
        r"(?:\s*[X×*]\s*\d+(?:\.\d+)?){1,3}\s*MM\b",
        re.IGNORECASE,
    ),
]
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper()
    return re.sub(r"\s+", " ", value).strip()


def build_size_skeleton(value: str) -> str:
    skeleton = normalize_text(value)
    replacements = (
        " DN# ",
        " INCH# ",
        " SIZE# ",
        " THK# ",
        " THK# ",
        " SCH# ",
        " PRESSURE# ",
        " LENGTH# ",
        " SIZE# ",
    )
    for pattern, replacement in zip(SIZE_PATTERNS, replacements, strict=True):
        skeleton = pattern.sub(replacement, skeleton)
    return re.sub(r"\s+", " ", skeleton).strip(" ,;")


def size_sort_key(value: str) -> tuple[float, float, int, str]:
    numbers: list[float] = []
    normalized = normalize_text(value)
    for pattern in SIZE_PATTERNS:
        for match in pattern.finditer(normalized):
            numbers.extend(float(item) for item in NUMBER_PATTERN.findall(match.group()))
    if not numbers:
        return (-1.0, -1.0, 0, normalized)
    return (max(numbers), sum(numbers), len(numbers), normalized)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_target_single_material(row: dict[str, Any], grade: str) -> bool:
    output = row.get("output") or {}
    materials = output.get("MATERIAL") or []
    if output.get("MATERIAL_RELATION") != "SINGLE" or len(materials) != 1:
        return False
    material = materials[0]
    return (
        material.get("PART") == "BODY"
        and material.get("STANDARD", "") == ""
        and material.get("GRADE") == grade
        and material.get("CLASS", "") == ""
        and not material.get("SPECIAL_REQ")
    )


def evenly_spaced_representatives(
    indices: list[int],
    rows: list[dict[str, Any]],
    limit: int,
) -> set[int]:
    if len(indices) <= limit:
        return set(indices)
    ordered = sorted(indices, key=lambda index: (size_sort_key(rows[index]["input"]), index))
    if limit == 1:
        return {ordered[len(ordered) // 2]}

    positions = {
        min(len(ordered) - 1, math.floor(step * (len(ordered) - 1) / (limit - 1) + 0.5))
        for step in range(limit)
    }
    return {ordered[position] for position in positions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按非尺寸骨架降低单一材质样本的尺寸型重复。",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--removed-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--grade", default="20")
    parser.add_argument("--max-per-skeleton", type=int, default=3)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="使用降重结果覆盖输入文件；删除明细仍会单独保存。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_per_skeleton < 1:
        raise ValueError("--max-per-skeleton 必须大于等于 1")

    source = args.input.resolve()
    if args.in_place and args.output:
        raise ValueError("--in-place 与 --output 不能同时使用")

    output = (
        source
        if args.in_place
        else (args.output or source.with_name(f"{source.stem}_骨架降重.json")).resolve()
    )
    removed_output = (
        args.removed_output
        or source.with_name(f"{source.stem}_骨架降重删除项.json")
    ).resolve()
    report_output = (
        args.report_output
        or source.with_name(f"{source.stem}_骨架降重报告.json")
    ).resolve()

    rows: list[dict[str, Any]] = json.loads(source.read_text(encoding="utf-8"))
    target_indices = [
        index for index, row in enumerate(rows) if is_target_single_material(row, args.grade)
    ]

    removed: dict[int, dict[str, Any]] = {}

    exact_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in target_indices:
        row = rows[index]
        exact_groups[
            (normalize_text(row.get("input", "")), canonical_json(row.get("output")))
        ].append(index)
    for indices in exact_groups.values():
        for index in indices[1:]:
            removed[index] = {
                "source_index": index,
                "reason": "完全重复",
                "skeleton": build_size_skeleton(rows[index].get("input", "")),
                "row": rows[index],
            }

    skeleton_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in target_indices:
        if index in removed:
            continue
        row = rows[index]
        skeleton_groups[
            (build_size_skeleton(row.get("input", "")), canonical_json(row.get("output")))
        ].append(index)

    for (skeleton, _), indices in skeleton_groups.items():
        keep = evenly_spaced_representatives(
            indices,
            rows,
            args.max_per_skeleton,
        )
        for index in indices:
            if index not in keep:
                removed[index] = {
                    "source_index": index,
                    "reason": "仅尺寸壁厚压力变化的骨架重复",
                    "skeleton": skeleton,
                    "row": rows[index],
                }

    reduced_rows = [row for index, row in enumerate(rows) if index not in removed]
    target_after = sum(
        1 for row in reduced_rows if is_target_single_material(row, args.grade)
    )
    reason_counts: dict[str, int] = defaultdict(int)
    for item in removed.values():
        reason_counts[item["reason"]] += 1

    report = {
        "source": str(source),
        "output": str(output),
        "grade": args.grade,
        "max_per_skeleton": args.max_per_skeleton,
        "selection": "每组按显式尺寸数值排序，等距保留最小、中间、最大范围代表样本",
        "statistics": {
            "rows_before": len(rows),
            "rows_after": len(reduced_rows),
            "rows_removed": len(removed),
            "target_rows_before": len(target_indices),
            "target_rows_after": target_after,
            "target_ratio_before": round(len(target_indices) / len(rows), 6),
            "target_ratio_after": round(target_after / len(reduced_rows), 6),
            "target_unique_skeletons": len(skeleton_groups),
            "removed_by_reason": dict(sorted(reason_counts.items())),
        },
        "rules": [
            "仅处理单一BODY材质、STANDARD为空、CLASS为空、SPECIAL_REQ为空的指定GRADE",
            "分组键同时包含非尺寸骨架和完整output JSON",
            "保留产品名称、材料写法、制造方式、连接方式、标准及原始分隔格式",
            "仅屏蔽显式尺寸、壁厚、压力和长度字段",
            "验证集不参与处理",
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    removed_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.write_text(
        json.dumps(reduced_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    removed_output.write_text(
        json.dumps(
            [removed[index] for index in sorted(removed)],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
