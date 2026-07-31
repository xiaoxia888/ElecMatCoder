#!/usr/bin/env python3
"""Merge and stratify structured material datasets without skeleton leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deduplicate_single_material_skeletons import (
    build_size_skeleton,
    canonical_json,
    normalize_text,
)


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
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "材质规范_结构化原始牌号_train.json"
DEFAULT_VAL = DEFAULT_DATA_DIR / "材质规范_结构化原始牌号_val.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "重新划分_v2"

PART_ORDER = {
    "BODY": 0,
    "INNER_PIPE": 1,
    "OUTER_PIPE": 2,
    "FLANGE": 3,
    "LINING": 4,
}
OBJECTIVE_WEIGHTS = {
    "row_count": 0.1,
    "output_signature": 10.0,
    "grade": 1.0,
    "relation": 1.0,
    "validation_coverage": 0.05,
}


@dataclass
class SkeletonGroup:
    key: str
    row_indices: list[int]
    signature_counts: Counter[str]
    grade_counts: Counter[str]
    relation_counts: Counter[str]
    score: float

    @property
    def size(self) -> int:
        return len(self.row_indices)


@dataclass
class SplitState:
    val_rows: int
    signature_counts: Counter[str]
    grade_counts: Counter[str]
    relation_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并结构化材质训练集和验证集，按非尺寸骨架分层重新划分。",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--min-groups-for-val",
        type=int,
        default=2,
        help="参与验证覆盖优化的完整输出标签至少需要多少个独立骨架组。",
    )
    parser.add_argument(
        "--min-rows-for-val-coverage",
        type=int,
        help="参与验证覆盖优化的完整输出标签最少行数；默认 ceil(1/val_ratio)。",
    )
    parser.add_argument(
        "--optimization-passes",
        type=int,
        default=6,
        help="单组移动优化轮数。",
    )
    parser.add_argument(
        "--swap-passes",
        type=int,
        default=3,
        help="相同行数骨架组交换优化轮数。",
    )
    parser.add_argument(
        "--swap-candidates",
        type=int,
        default=24,
        help="每个验证组最多测试多少个同尺寸训练组。",
    )
    parser.add_argument(
        "--keep-exact-duplicates",
        action="store_true",
        help="保留输入和输出均完全相同的重复记录。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_score(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def signature_id(signature: str) -> str:
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def semantic_output(output: dict[str, Any] | None) -> dict[str, Any]:
    """Canonicalize only semantically order-insensitive fields for splitting."""
    output = output or {}
    materials: list[dict[str, Any]] = []
    for item in output.get("MATERIAL") or []:
        if not isinstance(item, dict):
            continue
        materials.append(
            {
                "PART": str(item.get("PART", "")),
                "STANDARD": str(item.get("STANDARD", "")),
                "GRADE": str(item.get("GRADE", "")),
                "CLASS": str(item.get("CLASS", "")),
                "SPECIAL_REQ": sorted(
                    {str(value) for value in item.get("SPECIAL_REQ") or [] if value}
                ),
            }
        )
    materials.sort(
        key=lambda item: (
            PART_ORDER.get(item["PART"], 99),
            item["PART"],
            item["STANDARD"],
            item["GRADE"],
            item["CLASS"],
            canonical_json(item["SPECIAL_REQ"]),
        )
    )

    standards: list[dict[str, str]] = []
    for item in output.get("STANDARD") or []:
        if not isinstance(item, dict):
            continue
        normalized = {
            str(key): str(value)
            for key, value in sorted(item.items())
            if value not in (None, "")
        }
        if normalized:
            standards.append(normalized)
    standards.sort(key=canonical_json)
    return {
        "MATERIAL": materials,
        "STANDARD": standards,
        "MATERIAL_RELATION": str(output.get("MATERIAL_RELATION", "")),
    }


def semantic_signature(output: dict[str, Any] | None) -> str:
    return canonical_json(semantic_output(output))


def summarize_output(signature: str) -> dict[str, Any]:
    output = json.loads(signature)
    standards: list[str] = []
    for item in output.get("STANDARD") or []:
        standards.extend(str(value) for value in item.values() if value)
    return {
        "relation": output.get("MATERIAL_RELATION", ""),
        "materials": [
            {
                "part": item.get("PART", ""),
                "standard": item.get("STANDARD", ""),
                "grade": item.get("GRADE", ""),
                "class": item.get("CLASS", ""),
                "special_req": item.get("SPECIAL_REQ") or [],
            }
            for item in output.get("MATERIAL") or []
        ],
        "product_standards": standards,
    }


def deduplicate_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[tuple[str, str], int] = {}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for merged_index, row in enumerate(rows):
        key = (
            normalize_text(str(row.get("input", ""))),
            canonical_json(row.get("output")),
        )
        if key in seen:
            removed.append(
                {
                    "merged_index": merged_index,
                    "kept_merged_index": seen[key],
                    "input": row.get("input", ""),
                    "output": row.get("output"),
                }
            )
            continue
        seen[key] = merged_index
        kept.append(row)
    return kept, removed


def row_grade_counts(row: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(item.get("GRADE", ""))
        for item in (row.get("output") or {}).get("MATERIAL") or []
        if isinstance(item, dict)
    )


def row_relation_counts(row: dict[str, Any]) -> Counter[str]:
    return Counter([str((row.get("output") or {}).get("MATERIAL_RELATION", ""))])


def build_groups(
    rows: list[dict[str, Any]],
    seed: int,
) -> tuple[
    list[SkeletonGroup],
    dict[str, set[int]],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    row_signatures: list[str] = []
    row_grades: list[Counter[str]] = []
    row_relations: list[Counter[str]] = []
    for index, row in enumerate(rows):
        skeleton = build_size_skeleton(str(row.get("input", "")))
        grouped_indices[skeleton].append(index)
        row_signatures.append(semantic_signature(row.get("output")))
        row_grades.append(row_grade_counts(row))
        row_relations.append(row_relation_counts(row))

    groups: list[SkeletonGroup] = []
    signature_groups: dict[str, set[int]] = defaultdict(set)
    total_signatures: Counter[str] = Counter()
    total_grades: Counter[str] = Counter()
    total_relations: Counter[str] = Counter()
    for group_index, (key, indices) in enumerate(sorted(grouped_indices.items())):
        signature_counts: Counter[str] = Counter()
        grade_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        for index in indices:
            signature_counts[row_signatures[index]] += 1
            grade_counts.update(row_grades[index])
            relation_counts.update(row_relations[index])
        groups.append(
            SkeletonGroup(
                key=key,
                row_indices=indices,
                signature_counts=signature_counts,
                grade_counts=grade_counts,
                relation_counts=relation_counts,
                score=stable_score(seed, key),
            )
        )
        for signature in signature_counts:
            signature_groups[signature].add(group_index)
        total_signatures.update(signature_counts)
        total_grades.update(grade_counts)
        total_relations.update(relation_counts)
    return (
        groups,
        signature_groups,
        total_signatures,
        total_grades,
        total_relations,
    )


def counter_for_groups(
    groups: list[SkeletonGroup],
    group_ids: Iterable[int],
    attribute: str,
) -> Counter[str]:
    result: Counter[str] = Counter()
    for group_id in group_ids:
        result.update(getattr(groups[group_id], attribute))
    return result


def build_state(groups: list[SkeletonGroup], val_groups: set[int]) -> SplitState:
    return SplitState(
        val_rows=sum(groups[group_id].size for group_id in val_groups),
        signature_counts=counter_for_groups(
            groups, val_groups, "signature_counts"
        ),
        grade_counts=counter_for_groups(groups, val_groups, "grade_counts"),
        relation_counts=counter_for_groups(
            groups, val_groups, "relation_counts"
        ),
    )


def abs_error_delta(
    current: Counter[str],
    changes: Counter[str],
    targets: dict[str, float],
) -> float:
    return sum(
        abs(current[key] + delta - targets.get(key, 0.0))
        - abs(current[key] - targets.get(key, 0.0))
        for key, delta in changes.items()
    )


def coverage_delta(
    current: Counter[str],
    changes: Counter[str],
    eligible: set[str],
) -> int:
    delta = 0
    for signature, change in changes.items():
        if signature not in eligible:
            continue
        was_missing = current[signature] <= 0
        is_missing = current[signature] + change <= 0
        delta += int(is_missing) - int(was_missing)
    return delta


def scaled_counter(counter: Counter[str], factor: int) -> Counter[str]:
    return Counter({key: value * factor for key, value in counter.items()})


def group_changes(
    group: SkeletonGroup,
    direction: int,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    return (
        scaled_counter(group.signature_counts, direction),
        scaled_counter(group.grade_counts, direction),
        scaled_counter(group.relation_counts, direction),
    )


def objective_delta(
    state: SplitState,
    row_delta: int,
    signature_changes: Counter[str],
    grade_changes: Counter[str],
    relation_changes: Counter[str],
    *,
    target_val_rows: int,
    total_rows: int,
    signature_targets: dict[str, float],
    grade_targets: dict[str, float],
    relation_targets: dict[str, float],
    total_grade_items: int,
    eligible_coverage: set[str],
) -> float:
    row_error = (
        abs(state.val_rows + row_delta - target_val_rows)
        - abs(state.val_rows - target_val_rows)
    ) / max(total_rows, 1)
    signature_error = abs_error_delta(
        state.signature_counts, signature_changes, signature_targets
    ) / max(total_rows, 1)
    grade_error = abs_error_delta(
        state.grade_counts, grade_changes, grade_targets
    ) / max(total_grade_items, 1)
    relation_error = abs_error_delta(
        state.relation_counts, relation_changes, relation_targets
    ) / max(total_rows, 1)
    missing_coverage = coverage_delta(
        state.signature_counts, signature_changes, eligible_coverage
    ) / max(len(eligible_coverage), 1)
    return (
        OBJECTIVE_WEIGHTS["row_count"] * row_error
        + OBJECTIVE_WEIGHTS["output_signature"] * signature_error
        + OBJECTIVE_WEIGHTS["grade"] * grade_error
        + OBJECTIVE_WEIGHTS["relation"] * relation_error
        + OBJECTIVE_WEIGHTS["validation_coverage"] * missing_coverage
    )


def can_move_to_val(
    group: SkeletonGroup,
    state: SplitState,
    total_signatures: Counter[str],
) -> bool:
    return all(
        total_signatures[signature]
        - state.signature_counts[signature]
        - count
        > 0
        for signature, count in group.signature_counts.items()
    )


def respects_signature_error_caps(
    state: SplitState,
    changes: Counter[str],
    total_signatures: Counter[str],
    signature_targets: dict[str, float],
) -> bool:
    """Prevent average-error optimization from sacrificing individual labels."""
    for signature, change in changes.items():
        total = total_signatures[signature]
        if total < 10:
            continue
        target = signature_targets[signature]
        before_error = abs(state.signature_counts[signature] - target)
        after_error = abs(state.signature_counts[signature] + change - target)
        if total >= 100:
            allowance = max(2.0, total * 0.05)
        elif total >= 20:
            allowance = max(2.0, total * 0.15)
        else:
            allowance = max(2.0, total * 0.20)
        if after_error > allowance and after_error > before_error + 1e-12:
            return False
    return True


def apply_changes(
    state: SplitState,
    row_delta: int,
    signature_changes: Counter[str],
    grade_changes: Counter[str],
    relation_changes: Counter[str],
) -> None:
    state.val_rows += row_delta
    state.signature_counts.update(signature_changes)
    state.grade_counts.update(grade_changes)
    state.relation_counts.update(relation_changes)


def repair_training_coverage(
    groups: list[SkeletonGroup],
    val_groups: set[int],
    total_signatures: Counter[str],
) -> int:
    state = build_state(groups, val_groups)
    moves = 0
    for signature in sorted(total_signatures):
        if state.signature_counts[signature] < total_signatures[signature]:
            continue
        candidates = [
            group_id
            for group_id in val_groups
            if signature in groups[group_id].signature_counts
        ]
        if not candidates:
            continue
        group_id = min(
            candidates,
            key=lambda item: (groups[item].size, -groups[item].score),
        )
        group = groups[group_id]
        changes = group_changes(group, -1)
        val_groups.remove(group_id)
        apply_changes(state, -group.size, *changes)
        moves += 1
    return moves


def optimize_single_group_moves(
    groups: list[SkeletonGroup],
    train_groups: set[int],
    val_groups: set[int],
    state: SplitState,
    *,
    passes: int,
    objective_kwargs: dict[str, Any],
    total_signatures: Counter[str],
) -> int:
    moves = 0
    ordered_ids = sorted(
        range(len(groups)),
        key=lambda group_id: (groups[group_id].score, groups[group_id].key),
    )
    for pass_index in range(passes):
        improved = 0
        pass_ids = ordered_ids if pass_index % 2 == 0 else list(reversed(ordered_ids))
        for group_id in pass_ids:
            group = groups[group_id]
            if group_id in train_groups:
                direction = 1
                if not can_move_to_val(group, state, total_signatures):
                    continue
            else:
                direction = -1
            changes = group_changes(group, direction)
            if not respects_signature_error_caps(
                state,
                changes[0],
                total_signatures,
                objective_kwargs["signature_targets"],
            ):
                continue
            delta = objective_delta(
                state,
                direction * group.size,
                *changes,
                **objective_kwargs,
            )
            if delta >= -1e-12:
                continue
            if direction > 0:
                train_groups.remove(group_id)
                val_groups.add(group_id)
            else:
                val_groups.remove(group_id)
                train_groups.add(group_id)
            apply_changes(state, direction * group.size, *changes)
            moves += 1
            improved += 1
        if not improved:
            break
    return moves


def rebalance_signature_counts(
    groups: list[SkeletonGroup],
    signature_groups: dict[str, set[int]],
    train_groups: set[int],
    val_groups: set[int],
    state: SplitState,
    *,
    passes: int,
    objective_kwargs: dict[str, Any],
    total_signatures: Counter[str],
    minimum_rows: int = 10,
) -> int:
    """Directly replace oversized signature groups with better group combinations."""
    moves = 0
    signatures = sorted(
        (
            signature
            for signature, total in total_signatures.items()
            if total >= minimum_rows and len(signature_groups[signature]) >= 2
        ),
        key=lambda signature: (-total_signatures[signature], signature),
    )
    targets: dict[str, float] = objective_kwargs["signature_targets"]
    for _ in range(passes):
        improved = 0
        for signature in signatures:
            current_error = abs(state.signature_counts[signature] - targets[signature])
            best: tuple[
                float,
                float,
                int,
                int,
                tuple[Counter[str], Counter[str], Counter[str]],
            ] | None = None
            for group_id in signature_groups[signature]:
                group = groups[group_id]
                direction = 1 if group_id in train_groups else -1
                if direction > 0 and not can_move_to_val(
                    group, state, total_signatures
                ):
                    continue
                changes = group_changes(group, direction)
                new_error = abs(
                    state.signature_counts[signature]
                    + changes[0][signature]
                    - targets[signature]
                )
                if new_error >= current_error - 1e-12:
                    continue
                if not respects_signature_error_caps(
                    state,
                    changes[0],
                    total_signatures,
                    targets,
                ):
                    continue
                delta = objective_delta(
                    state,
                    direction * group.size,
                    *changes,
                    **objective_kwargs,
                )
                candidate = (
                    new_error,
                    delta,
                    group.size,
                    group_id,
                    changes,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
            if best is None:
                continue
            _, _, _, group_id, changes = best
            group = groups[group_id]
            direction = 1 if group_id in train_groups else -1
            if direction > 0:
                train_groups.remove(group_id)
                val_groups.add(group_id)
            else:
                val_groups.remove(group_id)
                train_groups.add(group_id)
            apply_changes(state, direction * group.size, *changes)
            moves += 1
            improved += 1
        if not improved:
            break
    return moves


def adjust_val_size(
    groups: list[SkeletonGroup],
    train_groups: set[int],
    val_groups: set[int],
    state: SplitState,
    *,
    objective_kwargs: dict[str, Any],
    total_signatures: Counter[str],
) -> int:
    target_val_rows = int(objective_kwargs["target_val_rows"])
    moves = 0
    while state.val_rows != target_val_rows:
        current_error = abs(state.val_rows - target_val_rows)
        direction = 1 if state.val_rows < target_val_rows else -1
        source = train_groups if direction > 0 else val_groups
        candidates: list[tuple[float, int, tuple[Counter[str], Counter[str], Counter[str]]]] = []
        for group_id in source:
            group = groups[group_id]
            new_error = abs(
                state.val_rows + direction * group.size - target_val_rows
            )
            if new_error >= current_error:
                continue
            if direction > 0 and not can_move_to_val(
                group, state, total_signatures
            ):
                continue
            changes = group_changes(group, direction)
            if not respects_signature_error_caps(
                state,
                changes[0],
                total_signatures,
                objective_kwargs["signature_targets"],
            ):
                continue
            delta = objective_delta(
                state,
                direction * group.size,
                *changes,
                **objective_kwargs,
            )
            candidates.append((delta, group_id, changes))
        if not candidates:
            break
        _, group_id, changes = min(
            candidates,
            key=lambda item: (
                abs(
                    state.val_rows
                    + direction * groups[item[1]].size
                    - target_val_rows
                ),
                item[0],
                groups[item[1]].score,
            ),
        )
        if direction > 0:
            train_groups.remove(group_id)
            val_groups.add(group_id)
        else:
            val_groups.remove(group_id)
            train_groups.add(group_id)
        apply_changes(
            state,
            direction * groups[group_id].size,
            *changes,
        )
        moves += 1
    return moves


def combined_changes(
    positive: Counter[str],
    negative: Counter[str],
) -> Counter[str]:
    result = positive.copy()
    result.subtract(negative)
    return Counter({key: value for key, value in result.items() if value})


def can_swap(
    train_group: SkeletonGroup,
    val_group: SkeletonGroup,
    state: SplitState,
    total_signatures: Counter[str],
) -> bool:
    changes = combined_changes(
        train_group.signature_counts,
        val_group.signature_counts,
    )
    return all(
        total_signatures[signature]
        - state.signature_counts[signature]
        - change
        > 0
        for signature, change in changes.items()
        if change > 0
    )


def optimize_equal_size_swaps(
    groups: list[SkeletonGroup],
    train_groups: set[int],
    val_groups: set[int],
    state: SplitState,
    *,
    passes: int,
    candidate_limit: int,
    objective_kwargs: dict[str, Any],
    total_signatures: Counter[str],
) -> int:
    swaps = 0
    for pass_index in range(passes):
        train_by_size: dict[int, list[int]] = defaultdict(list)
        for group_id in train_groups:
            train_by_size[groups[group_id].size].append(group_id)
        for group_ids in train_by_size.values():
            group_ids.sort(
                key=lambda group_id: (groups[group_id].score, groups[group_id].key)
            )

        improved = 0
        val_order = sorted(
            val_groups,
            key=lambda group_id: (
                stable_score(pass_index + 1, groups[group_id].key),
                groups[group_id].key,
            ),
        )
        for val_group_id in val_order:
            val_group = groups[val_group_id]
            candidates = train_by_size.get(val_group.size, [])
            if not candidates:
                continue
            if len(candidates) > candidate_limit:
                offset = int(
                    stable_score(pass_index + 17, val_group.key) * len(candidates)
                )
                step = max(1, len(candidates) // candidate_limit)
                candidate_ids = [
                    candidates[(offset + index * step) % len(candidates)]
                    for index in range(candidate_limit)
                ]
            else:
                candidate_ids = candidates

            best: tuple[float, int, tuple[Counter[str], Counter[str], Counter[str]]] | None = None
            for train_group_id in candidate_ids:
                train_group = groups[train_group_id]
                if not can_swap(
                    train_group, val_group, state, total_signatures
                ):
                    continue
                changes = (
                    combined_changes(
                        train_group.signature_counts,
                        val_group.signature_counts,
                    ),
                    combined_changes(
                        train_group.grade_counts,
                        val_group.grade_counts,
                    ),
                    combined_changes(
                        train_group.relation_counts,
                        val_group.relation_counts,
                    ),
                )
                if not respects_signature_error_caps(
                    state,
                    changes[0],
                    total_signatures,
                    objective_kwargs["signature_targets"],
                ):
                    continue
                delta = objective_delta(
                    state,
                    0,
                    *changes,
                    **objective_kwargs,
                )
                if best is None or delta < best[0]:
                    best = (delta, train_group_id, changes)
            if best is None or best[0] >= -1e-12:
                continue

            _, train_group_id, changes = best
            train_groups.remove(train_group_id)
            val_groups.add(train_group_id)
            val_groups.remove(val_group_id)
            train_groups.add(val_group_id)
            apply_changes(state, 0, *changes)
            candidates.remove(train_group_id)
            candidates.append(val_group_id)
            candidates.sort(
                key=lambda group_id: (groups[group_id].score, groups[group_id].key)
            )
            swaps += 1
            improved += 1
        if not improved:
            break
    return swaps


def material_grade_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row_grade_counts(row))
    return counts


def relation_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row_relation_counts(row))
    return counts


def semantic_signature_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(semantic_signature(row.get("output")) for row in rows)


def distribution_rows(
    total_counts: Counter[str],
    train_counts: Counter[str],
    val_counts: Counter[str],
    total_items: int,
    train_items: int,
    val_items: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label, count in total_counts.most_common(limit):
        result.append(
            {
                "label": label,
                "total": count,
                "train": train_counts[label],
                "val": val_counts[label],
                "total_ratio": round(count / max(total_items, 1), 6),
                "train_ratio": round(train_counts[label] / max(train_items, 1), 6),
                "val_ratio": round(val_counts[label] / max(val_items, 1), 6),
            }
        )
    return result


def distribution_quality(
    total_counts: Counter[str],
    val_counts: Counter[str],
    val_ratio: float,
    *,
    high_frequency_threshold: int = 100,
) -> dict[str, Any]:
    total_items = sum(total_counts.values())
    weighted_absolute_error = sum(
        abs(val_counts[label] - count * val_ratio)
        for label, count in total_counts.items()
    ) / max(total_items, 1)
    high_frequency = {
        label: count
        for label, count in total_counts.items()
        if count >= high_frequency_threshold
    }
    deviations = [
        {
            "label": label,
            "total": count,
            "val": val_counts[label],
            "val_ratio": round(val_counts[label] / count, 6),
            "target_ratio": val_ratio,
            "absolute_deviation": round(
                abs(val_counts[label] / count - val_ratio), 6
            ),
        }
        for label, count in total_counts.items()
    ]
    deviations.sort(
        key=lambda item: (-item["absolute_deviation"], -item["total"], item["label"])
    )
    high_frequency_deviations = [
        item for item in deviations if item["label"] in high_frequency
    ]
    return {
        "weighted_mean_absolute_error": round(weighted_absolute_error, 8),
        "high_frequency_threshold": high_frequency_threshold,
        "high_frequency_label_count": len(high_frequency),
        "high_frequency_max_ratio_deviation": round(
            max(
                (
                    item["absolute_deviation"]
                    for item in high_frequency_deviations
                ),
                default=0.0,
            ),
            6,
        ),
        "worst_high_frequency_deviations_top20": high_frequency_deviations[:20],
        "worst_deviations_top20": deviations[:20],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 0 和 1 之间")
    if args.min_groups_for_val < 2:
        raise ValueError("--min-groups-for-val 必须大于等于 2")
    if args.optimization_passes < 0 or args.swap_passes < 0:
        raise ValueError("优化轮数不能小于 0")
    if args.swap_candidates < 1:
        raise ValueError("--swap-candidates 必须大于等于 1")

    train_source = json.loads(args.train.read_text(encoding="utf-8"))
    val_source = json.loads(args.val.read_text(encoding="utf-8"))
    merged = train_source + val_source
    if args.keep_exact_duplicates:
        rows = merged
        removed_duplicates: list[dict[str, Any]] = []
    else:
        rows, removed_duplicates = deduplicate_rows(merged)

    (
        groups,
        signature_groups,
        total_signature_counts,
        total_grade_counts,
        total_relation_counts,
    ) = build_groups(rows, args.seed)
    target_val_rows = round(len(rows) * args.val_ratio)
    min_rows_for_coverage = (
        args.min_rows_for_val_coverage
        if args.min_rows_for_val_coverage is not None
        else math.ceil(1 / args.val_ratio)
    )
    eligible_coverage = {
        signature
        for signature, group_ids in signature_groups.items()
        if len(group_ids) >= args.min_groups_for_val
        and total_signature_counts[signature] >= min_rows_for_coverage
    }

    val_group_ids = {
        index for index, group in enumerate(groups) if group.score < args.val_ratio
    }
    training_coverage_repairs = repair_training_coverage(
        groups,
        val_group_ids,
        total_signature_counts,
    )
    train_group_ids = set(range(len(groups))) - val_group_ids
    state = build_state(groups, val_group_ids)

    objective_kwargs = {
        "target_val_rows": target_val_rows,
        "total_rows": len(rows),
        "signature_targets": {
            key: count * args.val_ratio
            for key, count in total_signature_counts.items()
        },
        "grade_targets": {
            key: count * args.val_ratio for key, count in total_grade_counts.items()
        },
        "relation_targets": {
            key: count * args.val_ratio
            for key, count in total_relation_counts.items()
        },
        "total_grade_items": sum(total_grade_counts.values()),
        "eligible_coverage": eligible_coverage,
    }

    single_moves = optimize_single_group_moves(
        groups,
        train_group_ids,
        val_group_ids,
        state,
        passes=args.optimization_passes,
        objective_kwargs=objective_kwargs,
        total_signatures=total_signature_counts,
    )
    signature_rebalance_moves = rebalance_signature_counts(
        groups,
        signature_groups,
        train_group_ids,
        val_group_ids,
        state,
        passes=3,
        objective_kwargs=objective_kwargs,
        total_signatures=total_signature_counts,
    )
    ratio_moves = adjust_val_size(
        groups,
        train_group_ids,
        val_group_ids,
        state,
        objective_kwargs=objective_kwargs,
        total_signatures=total_signature_counts,
    )
    equal_size_swaps = optimize_equal_size_swaps(
        groups,
        train_group_ids,
        val_group_ids,
        state,
        passes=args.swap_passes,
        candidate_limit=args.swap_candidates,
        objective_kwargs=objective_kwargs,
        total_signatures=total_signature_counts,
    )

    train_indices = {
        row_index
        for group_id in train_group_ids
        for row_index in groups[group_id].row_indices
    }
    val_indices = {
        row_index
        for group_id in val_group_ids
        for row_index in groups[group_id].row_indices
    }
    if train_indices & val_indices or train_indices | val_indices != set(range(len(rows))):
        raise RuntimeError("划分后的记录集合不完整或存在交集")

    train_rows = [row for index, row in enumerate(rows) if index in train_indices]
    val_rows = [row for index, row in enumerate(rows) if index in val_indices]
    train_skeletons = {groups[group_id].key for group_id in train_group_ids}
    val_skeletons = {groups[group_id].key for group_id in val_group_ids}
    train_pairs = {
        (normalize_text(str(row.get("input", ""))), canonical_json(row.get("output")))
        for row in train_rows
    }
    val_pairs = {
        (normalize_text(str(row.get("input", ""))), canonical_json(row.get("output")))
        for row in val_rows
    }
    train_signature_counts = semantic_signature_counts(train_rows)
    val_signature_counts = semantic_signature_counts(val_rows)
    missing_train_signatures = set(val_signature_counts) - set(train_signature_counts)

    if train_skeletons & val_skeletons:
        raise RuntimeError("训练集和验证集存在非尺寸骨架泄漏")
    if train_pairs & val_pairs:
        raise RuntimeError("训练集和验证集存在完全相同的输入输出对")
    if missing_train_signatures:
        raise RuntimeError("验证集中存在训练集未覆盖的语义输出标签")

    train_grade_counts = material_grade_counts(train_rows)
    val_grade_counts = material_grade_counts(val_rows)
    train_relation_counts = relation_counts(train_rows)
    val_relation_counts = relation_counts(val_rows)
    covered_eligible = {
        signature
        for signature in eligible_coverage
        if val_signature_counts[signature] > 0
    }

    signature_distribution: list[dict[str, Any]] = []
    for signature, total_count in total_signature_counts.most_common(100):
        val_count = val_signature_counts[signature]
        signature_distribution.append(
            {
                "signature_id": signature_id(signature),
                "summary": summarize_output(signature),
                "total": total_count,
                "train": train_signature_counts[signature],
                "val": val_count,
                "val_ratio": round(val_count / total_count, 6),
                "ratio_deviation": round(
                    abs(val_count / total_count - args.val_ratio), 6
                ),
                "skeleton_groups": len(signature_groups[signature]),
            }
        )

    output_dir = args.output_dir.resolve()
    train_output = output_dir / "材质规范_结构化原始牌号_train.json"
    val_output = output_dir / "材质规范_结构化原始牌号_val.json"
    report_output = output_dir / "材质规范_结构化原始牌号_划分报告.json"
    duplicate_output = output_dir / "材质规范_结构化原始牌号_完全重复删除项.json"

    report = {
        "source": {
            "train": str(args.train.resolve()),
            "val": str(args.val.resolve()),
            "train_rows": len(train_source),
            "val_rows": len(val_source),
        },
        "parameters": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "min_groups_for_val": args.min_groups_for_val,
            "min_rows_for_val_coverage": min_rows_for_coverage,
            "optimization_passes": args.optimization_passes,
            "swap_passes": args.swap_passes,
            "swap_candidates": args.swap_candidates,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "remove_exact_duplicates": not args.keep_exact_duplicates,
            "grouping": "屏蔽显式尺寸、壁厚、压力和长度后的输入骨架",
            "stratification": "完整语义输出签名 + GRADE + MATERIAL_RELATION 的加权分层",
        },
        "output": {
            "train": str(train_output),
            "val": str(val_output),
            "report": str(report_output),
            "removed_duplicates": str(duplicate_output),
        },
        "statistics": {
            "merged_rows_before_deduplication": len(merged),
            "exact_duplicates_removed": len(removed_duplicates),
            "rows_after_deduplication": len(rows),
            "skeleton_groups": len(groups),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "target_val_rows": target_val_rows,
            "actual_val_ratio": round(len(val_rows) / len(rows), 6),
            "train_skeleton_groups": len(train_group_ids),
            "val_skeleton_groups": len(val_group_ids),
            "skeleton_overlap": 0,
            "exact_pair_overlap": 0,
            "semantic_output_signatures": len(total_signature_counts),
            "train_semantic_output_signatures": len(train_signature_counts),
            "val_semantic_output_signatures": len(val_signature_counts),
            "val_signatures_missing_from_train": [],
            "eligible_val_coverage_signatures": len(eligible_coverage),
            "covered_eligible_val_signatures": len(covered_eligible),
            "eligible_val_signature_coverage_ratio": round(
                len(covered_eligible) / max(len(eligible_coverage), 1), 6
            ),
            "training_coverage_repairs": training_coverage_repairs,
            "single_group_optimization_moves": single_moves,
            "signature_rebalance_moves": signature_rebalance_moves,
            "ratio_adjustment_group_moves": ratio_moves,
            "equal_size_group_swaps": equal_size_swaps,
        },
        "quality": {
            "output_signature": distribution_quality(
                total_signature_counts,
                val_signature_counts,
                args.val_ratio,
            ),
            "grade": distribution_quality(
                total_grade_counts,
                val_grade_counts,
                args.val_ratio,
            ),
            "relation": distribution_quality(
                total_relation_counts,
                val_relation_counts,
                args.val_ratio,
                high_frequency_threshold=1,
            ),
        },
        "grade_distribution_top100": distribution_rows(
            total_grade_counts,
            train_grade_counts,
            val_grade_counts,
            sum(total_grade_counts.values()),
            sum(train_grade_counts.values()),
            sum(val_grade_counts.values()),
        ),
        "relation_distribution": distribution_rows(
            total_relation_counts,
            train_relation_counts,
            val_relation_counts,
            sum(total_relation_counts.values()),
            sum(train_relation_counts.values()),
            sum(val_relation_counts.values()),
            limit=len(total_relation_counts),
        ),
        "output_signature_distribution_top100": signature_distribution,
    }

    print(
        json.dumps(
            {
                "parameters": report["parameters"],
                "statistics": report["statistics"],
                "quality": report["quality"],
                "output": report["output"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    write_json(train_output, train_rows)
    write_json(val_output, val_rows)
    write_json(report_output, report)
    write_json(duplicate_output, removed_duplicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
