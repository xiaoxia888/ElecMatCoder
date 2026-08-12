#!/usr/bin/env python3
"""Apply confirmed semantic review decisions to the approved V2 dataset.

The review file is authoritative for topology. This script does not infer
BRANCH or REDUCER from the description; it only validates source identity,
performs the reviewed role allocation, and removes all non-confirmed rows from
the approved dataset.
"""

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

from apps.trainer.qwen3_fte.src.apply_size_v2_group_decisions import (
    structural_issues,
    v2_to_deduplicated_v1,
    write_json_atomic,
)
from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (
    assignment,
    branch_default_thickness_role,
    convert_output,
    deterministic_plan,
)


BASE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "V2转换审核"
)
DEFAULT_APPROVED = BASE_DIR / "02_V2已审核通过数据.json"
DEFAULT_REVIEW = BASE_DIR / "08_剩余疑问逐条语义审核_待确认.json"
DEFAULT_REPORT = BASE_DIR / "09_181条写回及121条删除日志.json"
DEFAULT_STATS = BASE_DIR / "05_转换统计_无需审核.json"

REVIEW_TO_TOPOLOGY = {
    "可确定-分支结构": "BRANCH",
    "可确定-变径结构": "REDUCER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def role_set(output: dict[str, Any]) -> set[str]:
    return {
        str(item.get("ROLE") or "")
        for item in output.get("ITEMS") or []
        if isinstance(item, dict)
    }


def compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def explicitly_repeated_in_text(text: str, item: dict[str, Any]) -> bool:
    """Check whether one deduplicated label was explicitly written twice."""
    if str(item.get("type") or "").upper() != "SCHEDULE":
        return False
    token = compact_token(str(item.get("value") or ""))
    return bool(token) and compact_token(text).count(token) >= 2


def fallback_confirmed_plan(
    topology: str,
    text: str,
    flattened: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Allocate fields only after topology was fixed by semantic review.

    This fallback never determines product topology. It handles paired values
    whose textual separators are not covered by the general converter.
    """
    sizes = flattened.get("SIZE_ITEMS") or []
    thicknesses = flattened.get("THICKNESS_ITEMS") or []

    if topology == "REDUCER":
        if len(sizes) != 2 or sizes[0].get("type") != sizes[1].get("type"):
            return None, "已确认变径，但不是两个同类型尺寸，不能按顺序安全分配"
        size_plan = [
            assignment(0, "BODY", "END_A"),
            assignment(1, "BODY", "END_B"),
        ]
        if len(thicknesses) == 0:
            thickness_plan: list[dict[str, Any]] = []
        elif len(thicknesses) == 1:
            if explicitly_repeated_in_text(text, thicknesses[0]):
                thickness_plan = [
                    assignment(0, "BODY", "END_A", explicit_repeat=True),
                    assignment(0, "BODY", "END_B", explicit_repeat=True),
                ]
            else:
                thickness_plan = [assignment(0, "BODY", "END_A")]
        elif len(thicknesses) == 2:
            thickness_plan = [
                assignment(0, "BODY", "END_A"),
                assignment(1, "BODY", "END_B"),
            ]
        else:
            return None, "已确认变径，但壁厚数量超过两个，不能按顺序安全分配"
        return {
            "size": size_plan,
            "thickness": thickness_plan,
        }, "审核已确认变径；两个同类型尺寸按原文顺序分配至END_A/END_B"

    if topology == "BRANCH":
        if len(sizes) == 1 and re.search(r"(?:Y[- ]?TEE|TEE|三通)", text, re.IGNORECASE):
            size_plan = [assignment(0, "BODY", "MAIN")]
        elif len(sizes) == 2 and sizes[0].get("type") == sizes[1].get("type"):
            size_plan = [
                assignment(0, "BODY", "MAIN"),
                assignment(1, "BODY", "BRANCH"),
            ]
        else:
            return None, "已确认分支，但尺寸组不能按主管/支管安全分配"

        if len(thicknesses) == 0:
            thickness_plan = []
        elif len(thicknesses) == 1:
            thickness_plan = [
                assignment(0, "BODY", branch_default_thickness_role(text))
            ]
        elif len(thicknesses) == 2:
            thickness_plan = [
                assignment(0, "BODY", "MAIN"),
                assignment(1, "BODY", "BRANCH"),
            ]
        else:
            return None, "已确认分支，但壁厚数量超过两个，不能按顺序安全分配"
        return {
            "size": size_plan,
            "thickness": thickness_plan,
        }, "审核已确认分支；明示规格按原文顺序分配至MAIN/BRANCH"

    return None, f"不支持的已审核拓扑: {topology}"


def validate_converted(output: dict[str, Any], flattened: dict[str, Any], topology: str) -> None:
    required = {"MAIN", "BRANCH"} if topology == "BRANCH" else {"END_A", "END_B"}
    roles = role_set(output)
    current_sizes = flattened.get("SIZE_ITEMS") or []
    same_type_counts: dict[str, int] = {}
    for value in current_sizes:
        value_type = str(value.get("type") or "")
        same_type_counts[value_type] = same_type_counts.get(value_type, 0) + 1
    has_two_position_size_evidence = any(count >= 2 for count in same_type_counts.values())
    if has_two_position_size_evidence and not required.issubset(roles):
        raise ValueError(f"转换后未形成完整角色: expected={sorted(required)}, actual={sorted(roles)}")

    issues = structural_issues(output)
    if issues:
        raise ValueError("转换后仍存在结构问题: " + "；".join(issues))
    if "SINGLE" in roles and roles & {"MAIN", "BRANCH", "END_A", "END_B"}:
        raise ValueError("转换后SINGLE与位置角色并存")


def transform_reviewed_row(
    row: dict[str, Any],
    review: dict[str, Any],
    topology: str,
) -> tuple[dict[str, Any], str]:
    current = row.get("output")
    if not isinstance(current, dict):
        raise ValueError("已通过样本缺少V2 output")
    flattened = v2_to_deduplicated_v1(current)
    text = str(row.get("input") or "")
    plan, reason = deterministic_plan(topology, text, flattened)
    primary_error = ""
    if plan is not None:
        try:
            converted = convert_output(flattened, plan)
            validate_converted(converted, flattened, topology)
            return converted, reason
        except ValueError as exc:
            primary_error = str(exc)
    else:
        primary_error = reason

    fallback_plan, fallback_reason = fallback_confirmed_plan(topology, text, flattened)
    if fallback_plan is None:
        raise ValueError(
            f"已确认{topology}，常规分配失败: {primary_error}；补充分配失败: {fallback_reason}"
        )
    converted = convert_output(flattened, fallback_plan)
    validate_converted(converted, flattened, topology)
    return converted, f"{fallback_reason}（常规分配失败: {primary_error}）"


def main() -> int:
    args = parse_args()
    approved = load_json(args.approved)
    review_document = load_json(args.review)
    reviews = review_document.get("样本") if isinstance(review_document, dict) else None
    if not isinstance(approved, list) or not isinstance(reviews, list):
        raise ValueError("approved必须是数组，review必须包含样本数组")

    review_by_index: dict[int, dict[str, Any]] = {}
    for item in reviews:
        approved_index = int(item["approved_index"])
        if approved_index in review_by_index:
            raise ValueError(f"审核文件包含重复approved_index: {approved_index}")
        if not 0 <= approved_index < len(approved):
            raise ValueError(f"approved_index越界: {approved_index}")
        actual_text = str(approved[approved_index].get("input") or "")
        reviewed_text = str(item.get("原始描述") or "")
        if actual_text != reviewed_text:
            raise ValueError(f"approved_index={approved_index}原文不一致，拒绝写回")
        review_by_index[approved_index] = item

    transformed: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    projected: list[dict[str, Any]] = []

    for approved_index, row in enumerate(approved):
        review = review_by_index.get(approved_index)
        if review is None:
            projected.append(row)
            continue
        review_decision = str(review.get("模型逐条判定") or "")
        topology = REVIEW_TO_TOPOLOGY.get(review_decision)
        if topology is None:
            removed.append(
                {
                    "approved_index": approved_index,
                    "原始描述": row.get("input", ""),
                    "审核结论": review_decision,
                    "删除原因": review.get("模型分析原因", ""),
                }
            )
            continue
        try:
            converted, allocation_reason = transform_reviewed_row(row, review, topology)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "approved_index": approved_index,
                    "原始描述": row.get("input", ""),
                    "topology": topology,
                    "error": str(exc),
                }
            )
            projected.append(row)
            continue
        projected.append({"input": row.get("input", ""), "output": converted})
        transformed.append(
            {
                "approved_index": approved_index,
                "原始描述": row.get("input", ""),
                "topology": topology,
                "语义审核原因": review.get("模型分析原因", ""),
                "位置分配原因": allocation_reason,
                "修改前": row.get("output"),
                "修改后": converted,
            }
        )

    expected_transformed = sum(
        str(item.get("模型逐条判定") or "") in REVIEW_TO_TOPOLOGY for item in reviews
    )
    expected_removed = len(reviews) - expected_transformed
    report = {
        "execute": args.execute,
        "approved_rows_before": len(approved),
        "review_rows": len(reviews),
        "expected_transformed": expected_transformed,
        "actual_transformed": len(transformed),
        "expected_removed": expected_removed,
        "actual_removed": len(removed),
        "failures": failures,
        "approved_rows_after": len(projected),
        "transformed_by_topology": {
            topology: sum(item["topology"] == topology for item in transformed)
            for topology in ("BRANCH", "REDUCER")
        },
        "removed_by_review_decision": {
            decision: sum(item["审核结论"] == decision for item in removed)
            for decision in sorted({item["审核结论"] for item in removed})
        },
        "transformed": transformed,
        "removed": removed,
    }
    if failures or len(transformed) != expected_transformed or len(removed) != expected_removed:
        failure_summary = {
            key: value
            for key, value in report.items()
            if key not in {"transformed", "removed"}
        }
        print(json.dumps(failure_summary, ensure_ascii=False, indent=2))
        raise RuntimeError("审核写回数量或结构校验失败，未写入任何文件")

    if args.execute:
        write_json_atomic(args.approved, projected)
        write_json_atomic(args.report, report)
        if args.stats.expanduser().exists():
            stats = load_json(args.stats)
            if isinstance(stats, dict):
                stats["approved_rows"] = len(projected)
                stats["semantic_review_applied_rows"] = len(transformed)
                stats["semantic_review_removed_rows"] = len(removed)
                write_json_atomic(args.stats, stats)

    summary = {key: value for key, value in report.items() if key not in {"transformed", "removed"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
