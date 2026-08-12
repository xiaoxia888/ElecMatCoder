#!/usr/bin/env python3
"""Apply manually confirmed round-8 paired-thickness fixes and siblings."""

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


BASE_DIR = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核"
DEFAULT_DATASET = BASE_DIR / "02_V2已审核通过数据.json"
DEFAULT_SAMPLE = Path("/private/tmp/v2_size_manual_sample_200_round8.json")
DEFAULT_REPORT = BASE_DIR / "18_人工抽查200条第八轮审核与修复结果.json"
AUDITED_ERRORS = {
    35193: "异径三通支管明示SCH40漏标",
    50040: "偏心大小头第二端明示10mm漏标",
    47457: "异径三通支管明示3.5mm漏标",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


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
    source_rows = load_json(args.dataset)
    sample = load_json(args.sample)
    samples = sample.get("样本") if isinstance(sample, dict) else None
    if not isinstance(source_rows, list) or not isinstance(samples, list):
        raise ValueError("dataset或sample格式错误")
    sample_by_index = {int(item["source_index"]): item for item in samples}
    if not set(AUDITED_ERRORS).issubset(sample_by_index):
        raise ValueError("第八轮错误索引不完整")
    for index in AUDITED_ERRORS:
        if sample_by_index[index]["input"] != source_rows[index]["input"]:
            raise ValueError(f"source_index={index}原文变化")
        if sample_by_index[index]["output"] != source_rows[index]["output"]:
            raise ValueError(f"source_index={index}标签变化")

    target_skeletons = {description_skeleton(source_rows[index]["input"]) for index in AUDITED_ERRORS}
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
        if len(first.get("THICKNESS") or []) != 1 or second.get("THICKNESS"):
            continue
        before = deepcopy(output)
        second["THICKNESS"] = deepcopy(first["THICKNESS"])
        changes.append(
            {
                "source_index": index,
                "原始描述": text,
                "修复类别": "第二位置明示壁厚补齐",
                "修改前": before,
                "修改后": deepcopy(output),
            }
        )

    changed_indexes = {item["source_index"] for item in changes}
    missing = sorted(set(AUDITED_ERRORS) - changed_indexes)
    if missing:
        raise ValueError(f"第八轮错误未全部修复: {missing}")
    validate_dataset(repaired)
    change_by_index = {item["source_index"]: item for item in changes}
    audited_details = []
    for index, error_type in AUDITED_ERRORS.items():
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
    report = {
        "说明": "第八轮200条均逐条人工审核；确认错误及同骨架问题按原文明示的成对壁厚修复。",
        "execute": args.execute,
        "抽查总数": 200,
        "未发现问题": 197,
        "确认有误": 3,
        "待确认": 0,
        "抽查准确率": "98.50%",
        "连续达标轮次": 1,
        "确认有误样本": sorted(audited_details, key=lambda item: item["抽查序号"]),
        "修复统计": {
            "审核错误已修复": len(set(AUDITED_ERRORS) & changed_indexes),
            "同源额外修复": len(changed_indexes - set(AUDITED_ERRORS)),
            "总修复行数": len(changed_indexes),
        },
        "修改明细": changes,
    }
    if args.execute:
        write_json_atomic(args.dataset, repaired)
        write_json_atomic(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"确认有误样本", "修改明细"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
