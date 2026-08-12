#!/usr/bin/env python3
"""Repair high-confidence annotation conflicts in the size training dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (
    normalize_items,
    normalize_schedule_token,
    normalized_mapping_text,
    size_pair_mappings,
    thickness_pair_mappings,
    topology_from_text,
)


INSTALLATION_DN_RE = re.compile(
    r"^(?:不锈钢|碳钢)管道安装\s+(?:1\.规格材质：)?\s*(\d+(?:\.\d+)?)\s*mm\b",
    re.IGNORECASE,
)
CLASS_PRESSURE_RE = re.compile(
    r"(?<![A-Z])CL(?:ASS)?\s*[.:=-]?\s*(150|300|600|900|1500|2500|3000|6000)(?!\d)",
    re.IGNORECASE,
)
PN_PRESSURE_RE = re.compile(r"(?<![A-Z])PN\s*[.:=-]?\s*(\d+)(?![.\d])", re.IGNORECASE)
MPA_PRESSURE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*MPA\b", re.IGNORECASE)
EXPLICIT_SCHEDULE_PAIR_RE = re.compile(
    r"(?<![A-Z0-9])(?P<left>SCH(?:EDULE)?\s*\.?\s*-?\s*\d+\s*S?)\s*"
    r"[X×*]\s*(?P<right>SCH(?:EDULE)?\s*\.?\s*-?\s*\d+\s*S?)(?![A-Z0-9])",
    re.IGNORECASE,
)
COUPLED_OD_WALL_PAIR_RE = re.compile(
    r"(?<![A-Z0-9.])(?P<size_a>\d+(?:\.\d+)?)\s*[X×*]\s*"
    r"(?P<wall_a>\d+(?:\.\d+)?)\s*-\s*"
    r"(?P<size_b>\d+(?:\.\d+)?)\s*[X×*]\s*"
    r"(?P<wall_b>\d+(?:\.\d+)?)(?![A-Z0-9.])",
    re.IGNORECASE,
)


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


PairMappingBuilder = Callable[
    [str, list[dict[str, str]], str, str],
    list[dict[str, Any]],
]


def _reverse_explicit_pair_if_needed(
    input_text: str,
    output: dict[str, Any],
    *,
    field: str,
    left_role: str,
    right_role: str,
    mapping_builder: PairMappingBuilder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Reverse a two-item label only when the original text proves it is reversed."""
    items = normalize_items(output, field)
    if len(items) != 2:
        return None

    mappings = mapping_builder(input_text, items, left_role, right_role)
    role_sources: dict[str, set[int]] = defaultdict(set)
    for mapping in mappings:
        role_sources[str(mapping.get("ROLE") or "")].add(int(mapping["source_index"]))

    if role_sources.get(left_role) != {1} or role_sources.get(right_role) != {0}:
        return None

    before = deepcopy(output[field])
    output[field] = [deepcopy(before[1]), deepcopy(before[0])]
    return before, deepcopy(output[field])


def _explicit_schedule_pair(input_text: str) -> tuple[str, str] | None:
    pairs = {
        (
            normalize_schedule_token(match.group("left")),
            normalize_schedule_token(match.group("right")),
        )
        for match in EXPLICIT_SCHEDULE_PAIR_RE.finditer(
            normalized_mapping_text(input_text)
        )
    }
    pairs = {pair for pair in pairs if pair[0] and pair[1]}
    return next(iter(pairs)) if len(pairs) == 1 else None


def repair_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired = deepcopy(rows)
    changes: list[dict[str, Any]] = []
    conflicts_before = _conflicts(repaired)

    for index, row in enumerate(repaired):
        input_text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            continue

        topology, _ = topology_from_text(input_text)
        pair_rules: list[tuple[str, str, str, PairMappingBuilder, str]] = []
        if topology == "BRANCH":
            pair_rules = [
                ("SIZE_ITEMS", "MAIN", "BRANCH", size_pair_mappings, "明示尺寸对与SIZE_ITEMS顺序相反"),
                (
                    "THICKNESS_ITEMS",
                    "MAIN",
                    "BRANCH",
                    thickness_pair_mappings,
                    "明示壁厚对与THICKNESS_ITEMS顺序相反",
                ),
            ]
        elif topology == "REDUCER":
            pair_rules = [
                ("SIZE_ITEMS", "END_A", "END_B", size_pair_mappings, "明示尺寸对与SIZE_ITEMS顺序相反"),
                (
                    "THICKNESS_ITEMS",
                    "END_A",
                    "END_B",
                    thickness_pair_mappings,
                    "明示壁厚对与THICKNESS_ITEMS顺序相反",
                ),
            ]

        for field, left_role, right_role, mapping_builder, category in pair_rules:
            reversed_pair = _reverse_explicit_pair_if_needed(
                input_text,
                output,
                field=field,
                left_role=left_role,
                right_role=right_role,
                mapping_builder=mapping_builder,
            )
            if reversed_pair is None:
                continue
            before, after = reversed_pair
            changes.append(
                {
                    "source_index": index,
                    "category": category,
                    "input": input_text,
                    "before": before,
                    "after": after,
                }
            )

        explicit_schedule_pair = _explicit_schedule_pair(input_text)
        thickness_items = output.get("THICKNESS_ITEMS")
        if (
            explicit_schedule_pair is not None
            and isinstance(thickness_items, list)
            and len(thickness_items) == 2
            and all(
                isinstance(item, dict) and item.get("type") == "SCHEDULE"
                for item in thickness_items
            )
        ):
            expected = [
                {"type": "SCHEDULE", "value": explicit_schedule_pair[0]},
                {"type": "SCHEDULE", "value": explicit_schedule_pair[1]},
            ]
            if thickness_items != expected:
                before = deepcopy(thickness_items)
                output["THICKNESS_ITEMS"] = expected
                changes.append(
                    {
                        "source_index": index,
                        "category": "明示SCHEDULE壁厚对与旧标签不一致",
                        "input": input_text,
                        "before": before,
                        "after": deepcopy(expected),
                    }
                )

        coupled_specs = list(
            COUPLED_OD_WALL_PAIR_RE.finditer(normalized_mapping_text(input_text))
        )
        size_items = output.get("SIZE_ITEMS")
        thickness_items = output.get("THICKNESS_ITEMS")
        if (
            len(coupled_specs) == 1
            and isinstance(size_items, list)
            and isinstance(thickness_items, list)
            and len(thickness_items) == 2
            and all(
                isinstance(item, dict) and item.get("type") == "MM"
                for item in thickness_items
            )
        ):
            match = coupled_specs[0]
            expected_size = [
                {"type": "OD", "value": _canonical_number(match.group("size_a"))},
                {"type": "OD", "value": _canonical_number(match.group("size_b"))},
            ]
            expected_wall = [
                {"type": "MM", "value": _canonical_number(match.group("wall_a"))},
                {"type": "MM", "value": _canonical_number(match.group("wall_b"))},
            ]
            interleaved = [
                expected_size[0],
                {"type": "OD", "value": expected_wall[0]["value"]},
                expected_size[1],
                {"type": "OD", "value": expected_wall[1]["value"]},
            ]
            od_positions = [
                position
                for position, item in enumerate(size_items)
                if isinstance(item, dict) and item.get("type") == "OD"
            ]
            od_items = [size_items[position] for position in od_positions]
            if od_items == interleaved and thickness_items == expected_wall:
                before = deepcopy(size_items)
                mistaken_positions = {od_positions[1], od_positions[3]}
                output["SIZE_ITEMS"] = [
                    item
                    for position, item in enumerate(size_items)
                    if position not in mistaken_positions
                ]
                changes.append(
                    {
                        "source_index": index,
                        "category": "耦合外径壁厚中的壁厚误标为OD",
                        "input": input_text,
                        "before": before,
                        "after": deepcopy(output["SIZE_ITEMS"]),
                    }
                )

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

        mpa_matches = MPA_PRESSURE_RE.findall(input_text)
        current_pressure = str(output.get("PRESSURE") or "").strip()
        if (
            len(mpa_matches) == 1
            and current_pressure.upper().startswith("PN")
            and not PN_PRESSURE_RE.search(input_text)
            and not CLASS_PRESSURE_RE.search(input_text)
        ):
            pressure = f"{mpa_matches[0]}MPA"
            output["PRESSURE"] = pressure
            changes.append(
                {
                    "source_index": index,
                    "category": "MPa产品压力误换算为PN",
                    "input": input_text,
                    "before": current_pressure,
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
