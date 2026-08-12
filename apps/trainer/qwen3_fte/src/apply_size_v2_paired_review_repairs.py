#!/usr/bin/env python3
"""Apply a reviewed round's paired-thickness fixes and same-skeleton fixes."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.apply_size_v2_group_decisions import (  # noqa: E402
    body_role_items,
    structural_issues,
    write_json_atomic,
)
from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (  # noqa: E402
    description_skeleton,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核/02_V2已审核通过数据.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--error", action="append", required=True, help="source_index=错误说明")
    parser.add_argument("--delete", action="append", default=[], help="source_index=删除原因")
    parser.add_argument("--consecutive", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def parse_errors(values: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        index_text, separator, message = value.partition("=")
        if not separator or not index_text.isdigit() or not message:
            raise ValueError(f"--error格式错误: {value}")
        result[int(index_text)] = message
    return result


def second_position(roles: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if {"MAIN", "BRANCH"}.issubset(roles):
        return roles["MAIN"], roles["BRANCH"]
    if {"END_A", "END_B"}.issubset(roles):
        return roles["END_A"], roles["END_B"]
    return None


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"source_index={index}缺少output")
        issues = structural_issues(output)
        if issues:
            raise ValueError(f"source_index={index}结构异常: {'；'.join(issues)}")


def main() -> int:
    args = parse_args()
    errors = parse_errors(args.error)
    deletes = parse_errors(args.delete)
    if set(errors) & set(deletes):
        raise ValueError("同一source_index不能同时修复和删除")
    source_rows = load_json(args.dataset)
    sample = load_json(args.sample)
    samples = sample.get("样本") if isinstance(sample, dict) else None
    if not isinstance(source_rows, list) or not isinstance(samples, list):
        raise ValueError("dataset或sample格式错误")
    sample_by_index = {int(item["source_index"]): item for item in samples}
    reviewed_indexes = set(errors) | set(deletes)
    if not reviewed_indexes.issubset(sample_by_index):
        raise ValueError("审核错误索引不属于本轮样本")
    for index in reviewed_indexes:
        if sample_by_index[index]["input"] != source_rows[index]["input"]:
            raise ValueError(f"source_index={index}原文变化")
        if sample_by_index[index]["output"] != source_rows[index]["output"]:
            raise ValueError(f"source_index={index}标签变化")

    target_skeletons = {description_skeleton(source_rows[index]["input"]) for index in errors}
    repaired = deepcopy(source_rows)
    changes: list[dict[str, Any]] = []
    for index, row in enumerate(repaired):
        text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict) or description_skeleton(text) not in target_skeletons:
            continue
        positions = second_position(body_role_items(output))
        if positions is None:
            continue
        first, second = positions
        first_thickness = first.get("THICKNESS") or []
        second_thickness = second.get("THICKNESS") or []
        if len(first_thickness) == 1 and not second_thickness:
            source, target = first, second
        elif not first_thickness and len(second_thickness) == 1:
            source, target = second, first
        else:
            continue
        before = deepcopy(output)
        target["THICKNESS"] = deepcopy(source["THICKNESS"])
        changes.append(
            {
                "source_index": index,
                "原始描述": text,
                "修复类别": "成对表达缺失位置壁厚补齐",
                "修改前": before,
                "修改后": deepcopy(output),
            }
        )

    changed_indexes = {item["source_index"] for item in changes}
    missing = sorted(set(errors) - changed_indexes)
    if missing:
        raise ValueError(f"审核错误未全部修复: {missing}")
    deleted_details = []
    delete_inputs = {index: source_rows[index]["input"] for index in deletes}
    for index, reason in deletes.items():
        matches = [
            row_index
            for row_index, row in enumerate(repaired)
            if row.get("input") == delete_inputs[index]
        ]
        if matches != [index]:
            raise ValueError(f"source_index={index}待删除原文匹配异常: {matches}")
        deleted_details.append(
            {
                "source_index": index,
                "原始描述": delete_inputs[index],
                "删除原因": reason,
                "原标签": deepcopy(source_rows[index]["output"]),
            }
        )
    delete_input_values = set(delete_inputs.values())
    repaired = [row for row in repaired if row.get("input") not in delete_input_values]
    validate_dataset(repaired)
    change_by_index = {item["source_index"]: item for item in changes}
    audited_details = []
    for index, error_type in errors.items():
        sample_item = sample_by_index[index]
        change = change_by_index[index]
        audited_details.append(
            {
                "抽查序号": sample_item["抽查序号"],
                "source_index": index,
                "原始描述": sample_item["input"],
                "错误类型": error_type,
                "当前标签": change["修改前"],
                "建议标签": change["修改后"],
            }
        )
    for index, reason in deletes.items():
        sample_item = sample_by_index[index]
        audited_details.append(
            {
                "抽查序号": sample_item["抽查序号"],
                "source_index": index,
                "原始描述": sample_item["input"],
                "错误类型": reason,
                "当前标签": sample_item["output"],
                "建议标签": "删除样本",
            }
        )

    total = len(samples)
    error_count = len(errors) + len(deletes)
    correct = total - error_count
    report = {
        "说明": f"第{args.round}轮样本均逐条人工审核；确认错误及同骨架问题按原文明示的成对壁厚修复。",
        "execute": args.execute,
        "抽查总数": total,
        "未发现问题": correct,
        "确认有误": error_count,
        "待确认": 0,
        "抽查准确率": f"{correct / total:.2%}",
        "连续达标轮次": args.consecutive,
        "确认有误样本": sorted(audited_details, key=lambda item: item["抽查序号"]),
        "修复统计": {
            "审核错误已修复": len(set(errors) & changed_indexes),
            "同源额外修复": len(changed_indexes - set(errors)),
            "总修复行数": len(changed_indexes),
            "删除脏数据": len(deleted_details),
        },
        "修改明细": changes,
        "删除明细": deleted_details,
    }
    if args.execute:
        write_json_atomic(args.dataset, repaired)
        write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"确认有误样本", "修改明细", "删除明细"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
