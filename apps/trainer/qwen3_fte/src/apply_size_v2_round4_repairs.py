#!/usr/bin/env python3
"""Apply the fourth manual-audit fixes and their confirmed sibling patterns."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.apply_size_v2_group_decisions import (
    PIPE_OD_WALL_PATTERN,
    PIPE_PRODUCT_PATTERN,
    append_unique_value,
    body_role_items,
    canonical_number,
    item_values,
    structural_issues,
    write_json_atomic,
)
from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (
    ROLE_ORDER,
    SCOPE_ORDER,
    description_skeleton,
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
DEFAULT_DATASET = BASE_DIR / "02_V2已审核通过数据.json"
DEFAULT_REVIEW = BASE_DIR / "13_人工抽查200条第四轮审核结果.json"
DEFAULT_LOG = BASE_DIR / "14_第四轮13条及同源问题修复日志.json"

OUTLET_PRODUCT_PATTERN = re.compile(
    r"支管台|支管座|管接台|OLET|管嘴|补强板",
    re.IGNORECASE,
)
TEE_PATTERN = re.compile(r"三通|\bTEE\b", re.IGNORECASE)
Y_TEE_OD_PATTERN = re.compile(
    r"Y[- ]?TEE\s*(?P<od>\d+(?:\.\d+)?)\s*[X×*]",
    re.IGNORECASE,
)

# These audited descriptions explicitly contain two positional wall values.
PAIRED_WALL_SOURCE_INDEXES = {53745, 50131, 41359, 41366, 47472, 47466}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def sort_items(output: dict[str, Any]) -> None:
    output["ITEMS"].sort(
        key=lambda item: (
            SCOPE_ORDER.get(str(item.get("SCOPE") or "UNKNOWN"), 99),
            ROLE_ORDER.get(str(item.get("ROLE") or "UNKNOWN"), 99),
        )
    )


def remove_empty_items(output: dict[str, Any]) -> None:
    output["ITEMS"] = [
        item
        for item in output.get("ITEMS") or []
        if (item.get("SIZE") or []) or (item.get("THICKNESS") or [])
    ]
    sort_items(output)


def is_outlet_product(text: str) -> bool:
    return bool(OUTLET_PRODUCT_PATTERN.search(text)) and not TEE_PATTERN.search(text)


def repair_single_position_outlet(text: str, output: dict[str, Any]) -> bool:
    """One stated outlet size/wall belongs to BRANCH, not MAIN."""
    if not is_outlet_product(text):
        return False
    roles = body_role_items(output)

    if set(roles) == {"MAIN"}:
        main = roles["MAIN"]
        if len(main.get("SIZE") or []) != 1:
            return False
        main["ROLE"] = "BRANCH"
        sort_items(output)
        return True

    if set(roles) != {"MAIN", "BRANCH"}:
        return False
    main = roles["MAIN"]
    branch = roles["BRANCH"]
    if len(main.get("SIZE") or []) != 1 or branch.get("SIZE"):
        return False
    if not branch.get("THICKNESS"):
        return False

    branch["SIZE"] = main["SIZE"]
    main["SIZE"] = []
    remove_empty_items(output)
    return True


def repair_paired_wall(
    text: str,
    output: dict[str, Any],
    paired_wall_skeletons: set[str],
) -> bool:
    if description_skeleton(text) not in paired_wall_skeletons:
        return False
    roles = body_role_items(output)
    if {"MAIN", "BRANCH"}.issubset(roles):
        first, second = roles["MAIN"], roles["BRANCH"]
    elif {"END_A", "END_B"}.issubset(roles):
        first, second = roles["END_A"], roles["END_B"]
    else:
        return False
    walls = first.get("THICKNESS") or []
    if len(walls) != 1 or second.get("THICKNESS"):
        return False
    second["THICKNESS"] = deepcopy(walls)
    return True


def repair_y_tee_od(text: str, output: dict[str, Any]) -> bool:
    match = Y_TEE_OD_PATTERN.search(text)
    roles = body_role_items(output)
    if match is None or "MAIN" not in roles:
        return False
    return append_unique_value(
        roles["MAIN"],
        "SIZE",
        "OD",
        canonical_number(match.group("od")),
        prepend=True,
    )


def repair_pipe_mm_wall(text: str, output: dict[str, Any]) -> bool:
    roles = body_role_items(output)
    if not PIPE_PRODUCT_PATTERN.search(text) or set(roles) != {"SINGLE"}:
        return False
    item = roles["SINGLE"]
    if item_values(item, "THICKNESS", "MM"):
        return False
    ods = {canonical_number(value) for value in item_values(item, "SIZE", "OD")}
    candidates = {
        (canonical_number(match.group("od")), canonical_number(match.group("wall")))
        for match in PIPE_OD_WALL_PATTERN.finditer(text)
        if canonical_number(match.group("od")) in ods
        and float(match.group("od")) > float(match.group("wall"))
    }
    if len(candidates) != 1:
        return False
    _, wall = next(iter(candidates))
    return append_unique_value(item, "THICKNESS", "MM", wall, prepend=True)


def repair_duplicate_branch_inch(text: str, output: dict[str, Any]) -> bool:
    if not is_outlet_product(text):
        return False
    roles = body_role_items(output)
    if not {"MAIN", "BRANCH"}.issubset(roles):
        return False
    main, branch = roles["MAIN"], roles["BRANCH"]
    main_inches = item_values(main, "SIZE", "INCH")
    branch_inches = set(item_values(branch, "SIZE", "INCH"))
    duplicates = set(main_inches) & branch_inches
    if len(main_inches) < 2 or len(duplicates) != 1:
        return False
    duplicate = next(iter(duplicates))
    main["SIZE"] = [
        value
        for value in main.get("SIZE") or []
        if not (
            str(value.get("type") or "").upper() == "INCH"
            and str(value.get("value") or "") == duplicate
        )
    ]
    return True


def validate_review_state(
    rows: list[dict[str, Any]],
    review: dict[str, Any],
    *,
    field: str,
) -> None:
    for item in review["确认有误样本"]:
        index = int(item["source_index"])
        row = rows[index]
        if row.get("input") != item.get("原始描述"):
            raise ValueError(f"source_index={index}原文已变化")
        if row.get("output") != item[field]:
            raise ValueError(f"source_index={index}不等于{field}，拒绝继续")


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"source_index={index}缺少output")
        issues = structural_issues(output)
        if issues:
            raise ValueError(f"source_index={index}结构异常: {'；'.join(issues)}")
        for item in output.get("ITEMS") or []:
            if not (item.get("SIZE") or []) and not (item.get("THICKNESS") or []):
                raise ValueError(f"source_index={index}存在空ITEM")


def main() -> int:
    args = parse_args()
    source_rows = load_json(args.dataset)
    review = load_json(args.review)
    if not isinstance(source_rows, list) or not isinstance(review, dict):
        raise ValueError("dataset必须是数组，review必须是对象")
    validate_review_state(source_rows, review, field="当前标签")

    paired_wall_skeletons = {
        description_skeleton(source_rows[index]["input"])
        for index in PAIRED_WALL_SOURCE_INDEXES
    }
    repaired = deepcopy(source_rows)
    changes_by_index: dict[int, dict[str, Any]] = {}

    repair_steps = (
        ("支管附件单位置规格归入BRANCH", repair_single_position_outlet),
        (
            "原文明示双位置壁厚补齐",
            lambda text, output: repair_paired_wall(
                text, output, paired_wall_skeletons
            ),
        ),
        ("Y-Tee明示等价OD补齐", repair_y_tee_od),
        ("直管OD乘壁厚中的MM壁厚补齐", repair_pipe_mm_wall),
        ("支管附件英制尺寸位置去重", repair_duplicate_branch_inch),
    )

    for index, row in enumerate(repaired):
        text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        before = deepcopy(output)
        categories: list[str] = []
        for category, repair in repair_steps:
            if repair(text, output):
                categories.append(category)
        if categories:
            changes_by_index[index] = {
                "source_index": index,
                "原始描述": text,
                "修复类别": categories,
                "修改前": before,
                "修改后": deepcopy(output),
            }

    if len(repaired) != len(source_rows):
        raise ValueError("修复前后数据行数变化")
    validate_review_state(repaired, review, field="建议标签")
    validate_dataset(repaired)

    category_counts = Counter(
        category
        for change in changes_by_index.values()
        for category in change["修复类别"]
    )
    audited_indexes = {
        int(item["source_index"]) for item in review["确认有误样本"]
    }
    changed_indexes = set(changes_by_index)
    log = {
        "说明": "第四轮13条审核错误全部写回；同源修复仅使用明确产品结构和原文位置证据。",
        "execute": args.execute,
        "dataset_rows_before": len(source_rows),
        "dataset_rows_after": len(repaired),
        "第四轮确认错误数": len(audited_indexes),
        "第四轮确认错误已修复数": len(audited_indexes & changed_indexes),
        "同源额外修复数": len(changed_indexes - audited_indexes),
        "总修复行数": len(changed_indexes),
        "分类统计": dict(sorted(category_counts.items())),
        "修改明细": list(changes_by_index.values()),
    }

    if args.execute:
        write_json_atomic(args.dataset, repaired)
        write_json_atomic(args.log, log)

    print(
        json.dumps(
            {key: value for key, value in log.items() if key != "修改明细"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
