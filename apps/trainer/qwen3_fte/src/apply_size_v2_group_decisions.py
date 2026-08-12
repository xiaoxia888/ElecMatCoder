#!/usr/bin/env python3
"""Apply reviewed V2 role decisions to the approved and unresolved datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (
    ROLE_ORDER,
    SCOPE_ORDER,
    branch_default_thickness_role,
    compact_json_text,
    convert_output,
    deterministic_plan,
    normalize_schedule_token,
    normalized_mapping_text,
    topology_from_text,
)


BASE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
)
DEFAULT_SOURCE = BASE_DIR / "尺寸壁厚磅级C1训练集.json"
DEFAULT_REVIEW_DIR = BASE_DIR / "V2转换审核"
DEFAULT_APPROVED = DEFAULT_REVIEW_DIR / "02_V2已审核通过数据.json"
DEFAULT_UNRESOLVED = DEFAULT_REVIEW_DIR / "04_复杂骨架_待模型判定_暂不人工审核.json"
DEFAULT_DECISIONS = DEFAULT_REVIEW_DIR / "06_主管支管结构_逐骨架判定.json"
DEFAULT_STATS = DEFAULT_REVIEW_DIR / "05_转换统计_无需审核.json"
DEFAULT_APPROVED_UNCERTAIN = DEFAULT_REVIEW_DIR / "07_已通过数据剩余疑问_人工审核.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED)
    parser.add_argument("--unresolved", type=Path, default=DEFAULT_UNRESOLVED)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--retry-deterministic",
        action="store_true",
        help="源标签修复后，重新尝试非夹套、非衬层骨架的确定性转换",
    )
    parser.add_argument(
        "--repair-approved-single-thickness",
        action="store_true",
        help="修复已通过BRANCH数据中错误落入SINGLE的唯一壁厚",
    )
    parser.add_argument(
        "--repair-approved-outlet-structure",
        action="store_true",
        help="修复已通过数据中被错误合并为SINGLE的明确支管台结构",
    )
    parser.add_argument(
        "--repair-approved-reducer-single-thickness",
        action="store_true",
        help="将已通过异径件中错误落入SINGLE的唯一壁厚移入END_A",
    )
    parser.add_argument(
        "--repair-approved-topology-structure",
        action="store_true",
        help="按原文明示拓扑修复已通过数据中的SINGLE多位置结构，并导出剩余疑问",
    )
    parser.add_argument(
        "--repair-approved-explicit-equivalent-specs",
        action="store_true",
        help="补齐原文明示但旧标签去重后丢失的等价尺寸和分位置规格",
    )
    parser.add_argument(
        "--approved-uncertain-output",
        type=Path,
        default=DEFAULT_APPROVED_UNCERTAIN,
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    text = compact_json_text(value) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def repair_approved_single_thickness_defaults(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("input") or "")
        if topology_from_text(text)[0] != "BRANCH":
            continue
        output = row.get("output")
        items = output.get("ITEMS") if isinstance(output, dict) else None
        if not isinstance(items, list):
            continue

        locations: list[tuple[dict[str, Any], dict[str, str]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for thickness in item.get("THICKNESS") or []:
                if isinstance(thickness, dict):
                    locations.append((item, thickness))
        if len(locations) != 1:
            continue

        source_item, thickness = locations[0]
        if source_item.get("SCOPE") != "BODY" or source_item.get("ROLE") != "SINGLE":
            continue
        target_role = branch_default_thickness_role(text)
        target_item = next(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("SCOPE") == "BODY"
                and item.get("ROLE") == target_role
            ),
            None,
        )
        if target_item is None:
            target_item = {
                "SCOPE": "BODY",
                "ROLE": target_role,
                "SIZE": [],
                "THICKNESS": [],
            }
            items.append(target_item)

        source_item["THICKNESS"] = []
        target_item.setdefault("THICKNESS", []).append(thickness)
        items[:] = [
            item
            for item in items
            if (item.get("SIZE") or []) or (item.get("THICKNESS") or [])
        ]
        items.sort(
            key=lambda item: (
                SCOPE_ORDER.get(str(item.get("SCOPE") or "UNKNOWN"), 99),
                ROLE_ORDER.get(str(item.get("ROLE") or "UNKNOWN"), 99),
            )
        )
        changes.append(
            {
                "approved_index": row_index,
                "target_role": target_role,
                "input": text,
                "thickness": thickness,
            }
        )
    return changes


def v2_single_to_v1_output(output: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the old flat shape only for one BODY/SINGLE item."""
    items = output.get("ITEMS")
    if not isinstance(items, list) or len(items) != 1:
        return None
    item = items[0]
    if not isinstance(item, dict):
        return None
    if item.get("SCOPE") != "BODY" or item.get("ROLE") != "SINGLE":
        return None
    sizes = item.get("SIZE") or []
    thicknesses = item.get("THICKNESS") or []
    if not isinstance(sizes, list) or not isinstance(thicknesses, list):
        return None
    return {
        "SIZE_ITEMS": sizes,
        "LENGTH": output.get("LENGTH", ""),
        "THICKNESS_ITEMS": thicknesses,
        "PRESSURE": output.get("PRESSURE", ""),
    }


def repair_approved_outlet_structures(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split explicit outlet run/branch pairs that were approved as SINGLE."""
    changes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("input") or "")
        topology, _ = topology_from_text(text)
        if topology != "BRANCH" or branch_default_thickness_role(text) != "BRANCH":
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        old_output = v2_single_to_v1_output(output)
        if old_output is None or len(old_output["SIZE_ITEMS"]) < 2:
            continue

        plan, reason = deterministic_plan("BRANCH", text, old_output)
        if plan is None:
            skipped.append({"approved_index": row_index, "input": text, "reason": reason})
            continue
        converted = convert_output(old_output, plan)
        roles = {item.get("ROLE") for item in converted.get("ITEMS") or []}
        if not {"MAIN", "BRANCH"}.issubset(roles):
            skipped.append(
                {
                    "approved_index": row_index,
                    "input": text,
                    "reason": "重新转换后未同时形成MAIN和BRANCH",
                }
            )
            continue
        row["output"] = converted
        changes.append(
            {
                "approved_index": row_index,
                "input": text,
                "before": output,
                "after": converted,
            }
        )
    return changes, skipped


def repair_approved_reducer_single_thickness(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Move one unqualified reducer wall from BODY/SINGLE to BODY/END_A."""
    changes: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("input") or "")
        if topology_from_text(text)[0] != "REDUCER":
            continue
        output = row.get("output")
        items = output.get("ITEMS") if isinstance(output, dict) else None
        if not isinstance(items, list):
            continue

        locations: list[tuple[dict[str, Any], dict[str, str]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for thickness in item.get("THICKNESS") or []:
                if isinstance(thickness, dict):
                    locations.append((item, thickness))
        if len(locations) != 1:
            continue

        source_item, thickness = locations[0]
        if source_item.get("SCOPE") != "BODY" or source_item.get("ROLE") != "SINGLE":
            continue
        target_item = next(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("SCOPE") == "BODY"
                and item.get("ROLE") == "END_A"
            ),
            None,
        )
        if target_item is None:
            continue

        source_item["THICKNESS"] = []
        target_item.setdefault("THICKNESS", []).append(thickness)
        items[:] = [
            item
            for item in items
            if (item.get("SIZE") or []) or (item.get("THICKNESS") or [])
        ]
        items.sort(
            key=lambda item: (
                SCOPE_ORDER.get(str(item.get("SCOPE") or "UNKNOWN"), 99),
                ROLE_ORDER.get(str(item.get("ROLE") or "UNKNOWN"), 99),
            )
        )
        changes.append(
            {
                "approved_index": row_index,
                "input": text,
                "thickness": thickness,
            }
        )
    return changes


def deduplicate_item_values(output: dict[str, Any]) -> int:
    """Remove exact duplicates inside one physical position."""
    removed = 0
    for item in output.get("ITEMS") or []:
        if not isinstance(item, dict):
            continue
        for field in ("SIZE", "THICKNESS"):
            values = item.get(field)
            if not isinstance(values, list):
                continue
            unique: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for value in values:
                if not isinstance(value, dict):
                    unique.append(value)
                    continue
                key = (
                    str(value.get("type") or "").upper(),
                    str(value.get("value") or ""),
                )
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                unique.append(value)
            item[field] = unique
    return removed


SCHEDULE_TEXT = r"(?:SCH(?:EDULE)?\s*\.?\s*-?\s*\d+\s*S?|S\s*-?\s*\d+\s*S?|STD|XXS|XS)"
COUPLED_OD_SCHEDULE_PATTERN = re.compile(
    rf"(?<![A-Z0-9.])"
    rf"(?P<left_od>\d+(?:\.\d+)?)\s*(?P<left_schedule>{SCHEDULE_TEXT})(?![A-Z0-9])"
    rf"\s*-\s*"
    rf"(?P<right_od>\d+(?:\.\d+)?)\s*(?P<right_schedule>{SCHEDULE_TEXT})(?![A-Z0-9])",
    re.IGNORECASE,
)
EXPLICIT_EQUAL_TEE_PATTERN = re.compile(
    rf"DN\s*(?P<main_dn>\d+(?:\.\d+)?)\s*"
    rf"(?P<main_schedule>{SCHEDULE_TEXT})\s*[X×*]\s*"
    rf"(?:DN\s*)?(?P<branch_dn>\d+(?:\.\d+)?)\s*"
    rf"(?P<branch_schedule>{SCHEDULE_TEXT})",
    re.IGNORECASE,
)
PIPE_OD_WALL_PATTERN = re.compile(
    r"(?<![A-Z0-9.])(?P<od>\d+(?:\.\d+)?)\s*[X×*]\s*"
    r"(?P<wall>\d+(?:\.\d+)?)(?!\s*[X×*]\s*\d)",
    re.IGNORECASE,
)
PIPE_PRODUCT_PATTERN = re.compile(r"\bPIPE\b|钢管|管子", re.IGNORECASE)
STRICT_SCHEDULE_TEXT = (
    r"(?:SCH(?:EDULE)?\s*\.?\s*-?\s*(?:\d+S?|STD|XXS|XS)|"
    r"S\s*-?\s*\d+S?|STD|XXS|XS)"
)
POSITIONAL_SCHEDULE_PAIR_X_PATTERN = re.compile(
    rf"(?<![A-Z0-9])(?P<left>{STRICT_SCHEDULE_TEXT})\s*[X×*/]\s*"
    rf"(?P<right>{STRICT_SCHEDULE_TEXT})(?![A-Z0-9])",
    re.IGNORECASE,
)
POSITIONAL_SCHEDULE_PAIR_SPACE_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"(?P<left>SCH(?:EDULE)?\s*\.?\s*-?\s*\d+S?)\s+"
    r"(?P<right>SCH(?:EDULE)?\s*\.?\s*-?\s*\d+S?)"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)
SINGLE_OD_SCHEDULE_PATTERN = re.compile(
    rf"(?<![A-Z0-9.])(?P<od>\d+\.\d+)\s*"
    rf"(?P<schedule>{STRICT_SCHEDULE_TEXT})(?![A-Z0-9])",
    re.IGNORECASE,
)
MPA_PRESSURE_PATTERN = re.compile(
    r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*MPA",
    re.IGNORECASE,
)
EXPLICIT_PN_CL_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:PN|CL(?:ASS)?)\s*[-.]?\s*\d",
    re.IGNORECASE,
)
NON_RATING_PRESSURE_CONTEXT_PATTERN = re.compile(
    r"设计压力|试验压力|工作压力|操作压力|压力试验|水压试验|气压试验|[≥≤<>]"
)


def canonical_number(value: Any) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else f"{number:.12g}"


def item_values(item: dict[str, Any], field: str, value_type: str) -> list[str]:
    return [
        str(value.get("value") or "")
        for value in item.get(field) or []
        if isinstance(value, dict)
        and str(value.get("type") or "").upper() == value_type
    ]


def append_unique_value(
    item: dict[str, Any],
    field: str,
    value_type: str,
    value: str,
    *,
    prepend: bool = False,
) -> bool:
    values = item.setdefault(field, [])
    target = (value_type, canonical_number(value) if value_type != "SCHEDULE" else value)
    for current in values:
        if not isinstance(current, dict):
            continue
        current_type = str(current.get("type") or "").upper()
        current_value = str(current.get("value") or "")
        normalized = (
            canonical_number(current_value)
            if current_type != "SCHEDULE"
            else current_value
        )
        if (current_type, normalized) == target:
            return False
    leaf = {"type": value_type, "value": value}
    if prepend:
        values.insert(0, leaf)
    else:
        values.append(leaf)
    return True


def body_role_items(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("ROLE") or ""): item
        for item in output.get("ITEMS") or []
        if isinstance(item, dict) and item.get("SCOPE") == "BODY"
    }


def schedule_is_compatible(item: dict[str, Any], expected: str) -> bool:
    schedules = item_values(item, "THICKNESS", "SCHEDULE")
    return not schedules or expected in schedules


def schedule_pair_text(value: str) -> str:
    """Normalize OCR-split schedules without loosening token boundaries."""
    text = normalized_mapping_text(value)
    text = re.sub(
        r"(SCH(?:EDULE)?\s*\.?\s*-?\s*\d+)\s+S\b",
        r"\1S",
        text,
        flags=re.IGNORECASE,
    )
    previous = ""
    while text != previous:
        previous = text
        text = re.sub(
            r"(SCH(?:EDULE)?\s*\.?\s*-?\s*\d)\s+(?=\d+S?\b)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
    return text


def explicit_positional_schedule_pair(value: str) -> tuple[str, str] | None:
    """Return one unambiguous two-position schedule pair from the source text."""
    text = schedule_pair_text(value)
    matches = list(POSITIONAL_SCHEDULE_PAIR_X_PATTERN.finditer(text))
    matches.extend(POSITIONAL_SCHEDULE_PAIR_SPACE_PATTERN.finditer(text))
    normalized_pairs = {
        (
            normalize_schedule_token(match.group("left")),
            normalize_schedule_token(match.group("right")),
        )
        for match in matches
    }
    normalized_pairs.discard(("", ""))
    return next(iter(normalized_pairs)) if len(normalized_pairs) == 1 else None


def known_equivalent_od_dn_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Collect OD/DN equivalences already evidenced by approved labels."""
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for item in output.get("ITEMS") or []:
            if not isinstance(item, dict):
                continue
            ods = {canonical_number(value) for value in item_values(item, "SIZE", "OD")}
            dns = {canonical_number(value) for value in item_values(item, "SIZE", "DN")}
            pairs.update((od, dn) for od in ods for dn in dns)
    return pairs


def repair_approved_explicit_equivalent_specs(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore explicit values lost when the old flat labels were deduplicated."""
    changes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    known_od_dn_pairs = known_equivalent_od_dn_pairs(rows)

    for row_index, row in enumerate(rows):
        text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        roles = body_role_items(output)
        categories: list[str] = []

        coupled_matches = list(COUPLED_OD_SCHEDULE_PATTERN.finditer(text))
        positional_roles: tuple[str, str] | None = None
        if {"MAIN", "BRANCH"}.issubset(roles):
            positional_roles = ("MAIN", "BRANCH")
        elif {"END_A", "END_B"}.issubset(roles):
            positional_roles = ("END_A", "END_B")
        if len(coupled_matches) == 1 and positional_roles is not None:
            match = coupled_matches[0]
            left_role, right_role = positional_roles
            left_item, right_item = roles[left_role], roles[right_role]
            left_schedule = normalize_schedule_token(match.group("left_schedule"))
            right_schedule = normalize_schedule_token(match.group("right_schedule"))
            if not left_schedule or not right_schedule:
                skipped.append(
                    {"approved_index": row_index, "input": text, "reason": "壁厚格式无法归一化"}
                )
            elif not schedule_is_compatible(left_item, left_schedule) or not schedule_is_compatible(
                right_item, right_schedule
            ):
                skipped.append(
                    {"approved_index": row_index, "input": text, "reason": "现有位置壁厚与原文明示壁厚冲突"}
                )
            else:
                changed = False
                changed |= append_unique_value(
                    left_item, "SIZE", "OD", canonical_number(match.group("left_od")), prepend=True
                )
                changed |= append_unique_value(
                    right_item, "SIZE", "OD", canonical_number(match.group("right_od")), prepend=True
                )
                changed |= append_unique_value(
                    left_item, "THICKNESS", "SCHEDULE", left_schedule
                )
                changed |= append_unique_value(
                    right_item, "THICKNESS", "SCHEDULE", right_schedule
                )
                if changed:
                    categories.append("两端外径与壁厚明示对补齐")

        schedule_pair = explicit_positional_schedule_pair(text)
        if schedule_pair is not None and positional_roles is not None:
            left_role, right_role = positional_roles
            left_item, right_item = roles[left_role], roles[right_role]
            left_schedule, right_schedule = schedule_pair
            if not left_schedule or not right_schedule:
                skipped.append(
                    {"approved_index": row_index, "input": text, "reason": "分位置壁厚格式无法归一化"}
                )
            elif not schedule_is_compatible(left_item, left_schedule) or not schedule_is_compatible(
                right_item, right_schedule
            ):
                skipped.append(
                    {"approved_index": row_index, "input": text, "reason": "现有位置壁厚与成对壁厚表达冲突"}
                )
            else:
                changed = False
                changed |= append_unique_value(
                    left_item, "THICKNESS", "SCHEDULE", left_schedule
                )
                changed |= append_unique_value(
                    right_item, "THICKNESS", "SCHEDULE", right_schedule
                )
                if changed:
                    categories.append("分位置明示壁厚补齐")

        equal_matches = list(EXPLICIT_EQUAL_TEE_PATTERN.finditer(text))
        if (
            len(equal_matches) == 1
            and ("EQUAL TEE" in text.upper() or "等径三通" in text)
            and "MAIN" in roles
            and "BRANCH" not in roles
        ):
            match = equal_matches[0]
            main_dn = canonical_number(match.group("main_dn"))
            branch_dn = canonical_number(match.group("branch_dn"))
            main_schedule = normalize_schedule_token(match.group("main_schedule"))
            branch_schedule = normalize_schedule_token(match.group("branch_schedule"))
            main_item = roles["MAIN"]
            if (
                main_dn == branch_dn
                and main_schedule == branch_schedule
                and main_dn in {canonical_number(value) for value in item_values(main_item, "SIZE", "DN")}
                and main_schedule in item_values(main_item, "THICKNESS", "SCHEDULE")
            ):
                output.setdefault("ITEMS", []).append(
                    {
                        "SCOPE": "BODY",
                        "ROLE": "BRANCH",
                        "SIZE": [{"type": "DN", "value": branch_dn}],
                        "THICKNESS": [{"type": "SCHEDULE", "value": branch_schedule}],
                    }
                )
                categories.append("等径三通明示支管规格补齐")

        if (
            PIPE_PRODUCT_PATTERN.search(text)
            and set(roles) == {"SINGLE"}
            and item_values(roles["SINGLE"], "SIZE", "DN")
            and not item_values(roles["SINGLE"], "SIZE", "OD")
        ):
            single_item = roles["SINGLE"]
            current_walls = {
                canonical_number(value)
                for value in item_values(single_item, "THICKNESS", "MM")
            }
            candidates = {
                (canonical_number(match.group("od")), canonical_number(match.group("wall")))
                for match in PIPE_OD_WALL_PATTERN.finditer(text)
                if canonical_number(match.group("wall")) in current_walls
                and float(match.group("od")) > float(match.group("wall"))
            }
            if len(candidates) == 1:
                od, _ = next(iter(candidates))
                if append_unique_value(single_item, "SIZE", "OD", od, prepend=True):
                    categories.append("直管等价外径补齐")
            elif len(candidates) > 1:
                skipped.append(
                    {"approved_index": row_index, "input": text, "reason": "存在多个可匹配的直管外径×壁厚表达"}
                )

        if set(roles) in ({"SINGLE"}, {"MAIN"}):
            single_item = roles.get("SINGLE") or roles.get("MAIN")
            assert single_item is not None
            dns = {canonical_number(value) for value in item_values(single_item, "SIZE", "DN")}
            current_schedules = set(item_values(single_item, "THICKNESS", "SCHEDULE"))
            if len(dns) == 1 and not item_values(single_item, "SIZE", "OD") and len(current_schedules) == 1:
                dn = next(iter(dns))
                candidates = {
                    (
                        canonical_number(match.group("od")),
                        normalize_schedule_token(match.group("schedule")),
                    )
                    for match in SINGLE_OD_SCHEDULE_PATTERN.finditer(schedule_pair_text(text))
                }
                candidates = {
                    (od, schedule)
                    for od, schedule in candidates
                    if schedule in current_schedules and (od, dn) in known_od_dn_pairs
                }
                if len(candidates) == 1:
                    od, _ = next(iter(candidates))
                    if append_unique_value(single_item, "SIZE", "OD", od, prepend=True):
                        categories.append("单位置等价外径补齐")
                elif len(candidates) > 1:
                    skipped.append(
                        {"approved_index": row_index, "input": text, "reason": "存在多个已有映射支持的等价外径"}
                    )

        current_pressure = str(output.get("PRESSURE") or "")
        mpa_matches = list(MPA_PRESSURE_PATTERN.finditer(text))
        if (
            current_pressure.upper().startswith("PN")
            and len(mpa_matches) == 1
            and not EXPLICIT_PN_CL_PATTERN.search(text)
            and not NON_RATING_PRESSURE_CONTEXT_PATTERN.search(text)
        ):
            mpa_pressure = f"{canonical_number(mpa_matches[0].group('value'))}MPA"
            if current_pressure != mpa_pressure:
                output["PRESSURE"] = mpa_pressure
                categories.append("MPA产品压力恢复原文单位")

        if categories:
            output["ITEMS"].sort(
                key=lambda item: (
                    SCOPE_ORDER.get(str(item.get("SCOPE") or "UNKNOWN"), 99),
                    ROLE_ORDER.get(str(item.get("ROLE") or "UNKNOWN"), 99),
                )
            )
            changes.append(
                {"approved_index": row_index, "input": text, "categories": categories}
            )

    return changes, skipped


def structural_issues(output: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    items = output.get("ITEMS")
    if not isinstance(items, list):
        return ["output缺少ITEMS数组"]

    roles_by_scope: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            issues.append("ITEMS包含非对象")
            continue
        scope = str(item.get("SCOPE") or "")
        role = str(item.get("ROLE") or "")
        roles_by_scope.setdefault(scope, set()).add(role)
        for field in ("SIZE", "THICKNESS"):
            grouped: dict[str, list[str]] = {}
            seen: set[tuple[str, str]] = set()
            for value in item.get(field) or []:
                if not isinstance(value, dict):
                    continue
                value_type = str(value.get("type") or "").upper()
                raw_value = str(value.get("value") or "")
                key = (value_type, raw_value)
                if key in seen:
                    issues.append(f"{scope}/{role}/{field}存在完全重复标签: {value_type}={raw_value}")
                seen.add(key)
                grouped.setdefault(value_type, []).append(raw_value)
            if role == "SINGLE":
                for value_type, values in grouped.items():
                    if len(set(values)) > 1:
                        issues.append(
                            f"{scope}/SINGLE/{field}包含同类型不同值: "
                            f"{value_type}={','.join(values)}"
                        )

    for scope, roles in roles_by_scope.items():
        positional = roles & {"MAIN", "BRANCH", "END_A", "END_B"}
        if "SINGLE" in roles and positional:
            issues.append(
                f"{scope}内SINGLE与位置角色并存: {','.join(sorted(positional))}"
            )
    return list(dict.fromkeys(issues))


def v2_to_deduplicated_v1(output: dict[str, Any]) -> dict[str, Any]:
    sizes: list[dict[str, Any]] = []
    thicknesses: list[dict[str, Any]] = []
    seen_size: set[tuple[str, str]] = set()
    seen_thickness: set[tuple[str, str]] = set()
    for item in output.get("ITEMS") or []:
        if not isinstance(item, dict):
            continue
        for field, target, seen in (
            ("SIZE", sizes, seen_size),
            ("THICKNESS", thicknesses, seen_thickness),
        ):
            for value in item.get(field) or []:
                if not isinstance(value, dict):
                    continue
                normalized = {
                    "type": str(value.get("type") or "").upper(),
                    "value": str(value.get("value") or ""),
                }
                key = (normalized["type"], normalized["value"])
                if key not in seen:
                    seen.add(key)
                    target.append(normalized)
    return {
        "SIZE_ITEMS": sizes,
        "LENGTH": output.get("LENGTH", ""),
        "THICKNESS_ITEMS": thicknesses,
        "PRESSURE": output.get("PRESSURE", ""),
    }


def repair_approved_topology_structures(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    changes: list[dict[str, Any]] = []
    failed_reasons: dict[int, str] = {}
    removed_duplicates = 0

    for row_index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        removed_duplicates += deduplicate_item_values(output)
        issues = structural_issues(output)
        if not issues:
            continue

        text = str(row.get("input") or "")
        topology, evidence = topology_from_text(text)
        if topology not in {"BRANCH", "REDUCER"}:
            failed_reasons[row_index] = "原文未提供可确定的分支或变径拓扑证据"
            continue

        old_output = v2_to_deduplicated_v1(output)
        plan, reason = deterministic_plan(topology, text, old_output)
        if plan is None:
            failed_reasons[row_index] = reason
            continue
        converted = convert_output(old_output, plan)
        roles = {
            str(item.get("ROLE") or "")
            for item in converted.get("ITEMS") or []
            if isinstance(item, dict)
        }
        required = {"MAIN", "BRANCH"} if topology == "BRANCH" else {"END_A", "END_B"}
        has_multi_position_evidence = any(
            "同类型不同值" in issue for issue in issues
        )
        if has_multi_position_evidence and not required.issubset(roles):
            failed_reasons[row_index] = f"转换后未形成完整{sorted(required)}结构"
            continue
        remaining = structural_issues(converted)
        if remaining:
            failed_reasons[row_index] = "转换后仍存在结构问题: " + "；".join(remaining)
            continue

        before = output
        row["output"] = converted
        changes.append(
            {
                "approved_index": row_index,
                "topology": topology,
                "evidence": evidence,
                "issues": issues,
                "input": text,
                "before": before,
                "after": converted,
            }
        )

    uncertain: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        issues = structural_issues(output)
        if not issues:
            continue
        text = str(row.get("input") or "")
        topology, evidence = topology_from_text(text)
        uncertain.append(
            {
                "approved_index": row_index,
                "原始描述": text,
                "拓扑判定": topology,
                "拓扑证据": evidence,
                "待审核问题": issues,
                "未自动修改原因": failed_reasons.get(
                    row_index, "自动转换后仍未通过结构校验"
                ),
                "当前标签": output,
            }
        )
    return changes, uncertain, removed_duplicates


def main() -> int:
    args = parse_args()
    source = load_json(args.source)
    approved = load_json(args.approved)
    unresolved = load_json(args.unresolved)
    decision_document = load_json(args.decisions)
    stats = load_json(args.stats)

    if not isinstance(source, list) or not isinstance(approved, list):
        raise ValueError("源数据和通过数据必须是JSON数组")
    groups = unresolved.get("groups")
    if not isinstance(groups, list):
        raise ValueError("待处理文件缺少groups数组")
    decisions = decision_document.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("判定文件缺少decisions数组")

    applied_ids = set(stats.get("applied_group_ids") or [])
    group_lookup = {group["group_id"]: group for group in groups}
    newly_applied: list[dict[str, Any]] = []
    deterministically_retried: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    resolved_indexes: set[int] = set()
    approved_role_repairs = (
        repair_approved_single_thickness_defaults(approved)
        if args.repair_approved_single_thickness
        else []
    )
    approved_structure_repairs, approved_structure_skips = (
        repair_approved_outlet_structures(approved)
        if args.repair_approved_outlet_structure
        else ([], [])
    )
    approved_reducer_repairs = (
        repair_approved_reducer_single_thickness(approved)
        if args.repair_approved_reducer_single_thickness
        else []
    )
    approved_topology_repairs, approved_uncertain, removed_exact_duplicates = (
        repair_approved_topology_structures(approved)
        if args.repair_approved_topology_structure
        else ([], [], 0)
    )
    approved_explicit_repairs, approved_explicit_skips = (
        repair_approved_explicit_equivalent_specs(approved)
        if args.repair_approved_explicit_equivalent_specs
        else ([], [])
    )

    for decision in decisions:
        if decision.get("status") != "CONFIRMED":
            continue
        current_id = str(decision.get("group_id") or "")
        if current_id in applied_ids:
            continue
        group = group_lookup.get(current_id)
        if group is None:
            raise ValueError(f"待处理文件中找不到骨架: {current_id}")
        if group.get("topology_hint") != "BRANCH":
            raise ValueError(f"{current_id}不是主管/支管结构")

        plan = {
            "size": decision.get("size") or [],
            "thickness": decision.get("thickness") or [],
        }
        indexes = group.get("source_indexes") or []
        if len(indexes) != group.get("row_count"):
            raise ValueError(f"{current_id}的row_count与source_indexes不一致")
        row_overrides = {
            int(item["source_index"]): item
            for item in decision.get("row_overrides") or []
        }
        unknown_override_indexes = set(row_overrides) - set(indexes)
        if unknown_override_indexes:
            raise ValueError(f"{current_id}包含不属于该骨架的单行覆盖: {sorted(unknown_override_indexes)}")
        for source_index in indexes:
            row = source[source_index]
            output = row.get("output")
            if not isinstance(output, dict):
                raise ValueError(f"源数据{source_index}缺少output")
            override = row_overrides.get(source_index) or {}
            row_plan = {
                "size": override.get("size", plan["size"]),
                "thickness": override.get("thickness", plan["thickness"]),
            }
            converted = convert_output(output, row_plan)
            new_rows.append({"input": row.get("input", ""), "output": converted})
            resolved_indexes.add(source_index)

        newly_applied.append(
            {
                "group_id": current_id,
                "row_count": len(indexes),
                "reason": decision.get("reason", ""),
            }
        )
        decision["status"] = "APPLIED"
        decision["applied_rows"] = len(indexes)

    manually_resolved_ids = {item["group_id"] for item in newly_applied}
    if args.retry_deterministic:
        for group in groups:
            current_id = str(group.get("group_id") or "")
            if current_id in manually_resolved_ids:
                continue
            topology = str(group.get("topology_hint") or "")
            if topology == "COMPLEX_SCOPE":
                continue

            indexes = group.get("source_indexes") or []
            if any(
                topology_from_text(str(source[index].get("input") or ""))[0]
                == "COMPLEX_SCOPE"
                for index in indexes
            ):
                continue
            plans: list[dict[str, Any]] = []
            reasons: list[str] = []
            for source_index in indexes:
                row = source[source_index]
                output = row.get("output")
                if not isinstance(output, dict):
                    plans = []
                    break
                plan, reason = deterministic_plan(topology, str(row.get("input") or ""), output)
                if plan is None:
                    plans = []
                    break
                plans.append(plan)
                reasons.append(reason)

            if len(plans) != len(indexes):
                continue

            for source_index, plan in zip(indexes, plans):
                row = source[source_index]
                converted = convert_output(row["output"], plan)
                new_rows.append({"input": row.get("input", ""), "output": converted})
                resolved_indexes.add(source_index)
            deterministically_retried.append(
                {
                    "group_id": current_id,
                    "row_count": len(indexes),
                    "reason": reasons[0] if reasons else "",
                }
            )

    deterministically_resolved_ids = {
        item["group_id"] for item in deterministically_retried
    }
    resolved_group_ids = manually_resolved_ids | deterministically_resolved_ids
    remaining_groups = [group for group in groups if group["group_id"] not in resolved_group_ids]
    unresolved["groups"] = remaining_groups
    unresolved["source_indexes"] = [
        index for index in unresolved.get("source_indexes", []) if index not in resolved_indexes
    ]
    unresolved["group_count"] = len(remaining_groups)
    unresolved["row_count"] = len(unresolved["source_indexes"])

    projected_approved = len(approved) + len(new_rows)
    projected_applied_ids = sorted(applied_ids | {item["group_id"] for item in newly_applied})
    previous_retry_ids = set(stats.get("deterministically_retried_group_ids") or [])
    projected_retry_ids = sorted(previous_retry_ids | deterministically_resolved_ids)
    stats["approved_rows"] = projected_approved
    stats["unresolved_rows"] = unresolved["row_count"]
    stats["model_task_groups"] = unresolved["group_count"]
    stats["manually_resolved_groups"] = len(projected_applied_ids)
    stats["manually_resolved_rows"] = int(
        stats.get("manually_resolved_rows") or 0
    ) + sum(item["row_count"] for item in newly_applied)
    stats["applied_group_ids"] = projected_applied_ids
    stats["deterministically_retried_groups"] = len(projected_retry_ids)
    stats["deterministically_retried_rows"] = int(
        stats.get("deterministically_retried_rows") or 0
    ) + sum(item["row_count"] for item in deterministically_retried)
    stats["deterministically_retried_group_ids"] = projected_retry_ids
    status_groups = stats.setdefault("status_groups", {})
    status_groups["NEEDS_MODEL"] = unresolved["group_count"]
    status_groups["MANUALLY_RESOLVED"] = len(projected_applied_ids)
    status_groups["DETERMINISTIC_RETRY"] = len(projected_retry_ids)

    report = {
        "execute": args.execute,
        "approved_single_thickness_repairs": len(approved_role_repairs),
        "approved_outlet_structure_repairs": len(approved_structure_repairs),
        "approved_outlet_structure_skips": len(approved_structure_skips),
        "approved_reducer_single_thickness_repairs": len(approved_reducer_repairs),
        "approved_topology_structure_repairs": len(approved_topology_repairs),
        "approved_explicit_equivalent_spec_repairs": len(approved_explicit_repairs),
        "approved_explicit_equivalent_spec_skips": len(approved_explicit_skips),
        "approved_exact_duplicate_values_removed": removed_exact_duplicates,
        "approved_uncertain_rows": len(approved_uncertain),
        "approved_repairs_by_target_role": {
            role: sum(item["target_role"] == role for item in approved_role_repairs)
            for role in sorted({item["target_role"] for item in approved_role_repairs})
        },
        "approved_repair_examples": approved_role_repairs[:20],
        "approved_outlet_structure_examples": approved_structure_repairs[:20],
        "approved_outlet_structure_skip_examples": approved_structure_skips[:20],
        "approved_reducer_repair_examples": approved_reducer_repairs[:20],
        "approved_topology_repair_examples": approved_topology_repairs[:20],
        "approved_explicit_repair_examples": approved_explicit_repairs[:20],
        "approved_explicit_skip_examples": approved_explicit_skips[:20],
        "newly_applied_groups": newly_applied,
        "deterministically_retried_groups": deterministically_retried,
        "newly_approved_rows": len(new_rows),
        "approved_rows_after": projected_approved,
        "unresolved_groups_after": unresolved["group_count"],
        "unresolved_rows_after": unresolved["row_count"],
    }
    if args.execute:
        approved.extend(new_rows)
        approved_changed = bool(
            new_rows
            or approved_role_repairs
            or approved_structure_repairs
            or approved_reducer_repairs
            or approved_topology_repairs
            or approved_explicit_repairs
            or removed_exact_duplicates
        )
        if approved_changed:
            write_json_atomic(args.approved, approved)
        if args.repair_approved_topology_structure:
            write_json_atomic(
                args.approved_uncertain_output,
                {
                    "说明": "仅包含已通过数据中仍无法按原文可靠确定位置归属的样本；这些样本未被自动修改。",
                    "样本数": len(approved_uncertain),
                    "样本": approved_uncertain,
                },
            )
        if newly_applied or deterministically_retried:
            write_json_atomic(args.unresolved, unresolved)
            write_json_atomic(args.decisions, decision_document)
            write_json_atomic(args.stats, stats)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
