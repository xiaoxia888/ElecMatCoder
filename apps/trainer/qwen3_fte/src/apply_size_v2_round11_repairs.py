#!/usr/bin/env python3
"""Apply reviewed round-11 V2 size repairs and remove one dirty row."""

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
DEFAULT_SAMPLE = Path("/private/tmp/v2_size_manual_sample_200_round11.json")
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核/21_人工抽查200条第十一轮审核与修复结果.json"
)

TEE_SOURCE_INDEX = 27012
DIRTY_SOURCE_INDEX = 50448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        issues = structural_issues(row.get("output") or {})
        if issues:
            raise ValueError(f"source_index={index}结构异常: {'；'.join(issues)}")


def main() -> int:
    args = parse_args()
    rows = load_json(args.dataset)
    sample = load_json(args.sample)
    samples = sample.get("样本") if isinstance(sample, dict) else None
    if not isinstance(rows, list) or not isinstance(samples, list):
        raise ValueError("dataset或sample格式错误")

    sample_by_index = {int(item["source_index"]): item for item in samples}
    for index in (TEE_SOURCE_INDEX, DIRTY_SOURCE_INDEX):
        if index not in sample_by_index:
            raise ValueError(f"source_index={index}不属于第11轮样本")
        if sample_by_index[index]["input"] != rows[index]["input"]:
            raise ValueError(f"source_index={index}原文变化")
        if sample_by_index[index]["output"] != rows[index]["output"]:
            raise ValueError(f"source_index={index}标签变化")

    tee_skeleton = description_skeleton(rows[TEE_SOURCE_INDEX]["input"])
    dirty_input = rows[DIRTY_SOURCE_INDEX]["input"]
    repaired = deepcopy(rows)
    changes: list[dict[str, Any]] = []
    for index, row in enumerate(repaired):
        if description_skeleton(str(row.get("input") or "")) != tee_skeleton:
            continue
        output = row.get("output") or {}
        items = output.get("ITEMS") or []
        if len(items) != 1 or items[0].get("ROLE") != "MAIN":
            raise ValueError(f"source_index={index}同源三通结构超出预期")
        main_thickness = items[0].get("THICKNESS") or []
        if len(main_thickness) != 1:
            raise ValueError(f"source_index={index}同源三通壁厚结构超出预期")
        before = deepcopy(output)
        items.append(
            {
                "SCOPE": "BODY",
                "ROLE": "BRANCH",
                "SIZE": [],
                "THICKNESS": deepcopy(main_thickness),
            }
        )
        changes.append(
            {
                "source_index": index,
                "原始描述": row["input"],
                "修复类别": "等径三通成对壁厚补齐",
                "修改前": before,
                "修改后": deepcopy(output),
            }
        )

    deleted = [
        {
            "source_index": index,
            "原始描述": row["input"],
            "删除原因": "DN600×DN500与24×0.75英寸端部尺寸冲突，属于脏描述",
            "原标签": deepcopy(row["output"]),
        }
        for index, row in enumerate(repaired)
        if row.get("input") == dirty_input
    ]
    if len(deleted) != 1:
        raise ValueError(f"脏描述匹配数量异常: {len(deleted)}")
    repaired = [row for row in repaired if row.get("input") != dirty_input]
    validate_dataset(repaired)

    audited = []
    change_by_index = {item["source_index"]: item for item in changes}
    tee_sample = sample_by_index[TEE_SOURCE_INDEX]
    audited.append(
        {
            "抽查序号": tee_sample["抽查序号"],
            "source_index": TEE_SOURCE_INDEX,
            "原始描述": tee_sample["input"],
            "错误类型": "明示S-40 X S-40，支管壁厚漏标",
            "当前标签": change_by_index[TEE_SOURCE_INDEX]["修改前"],
            "建议标签": change_by_index[TEE_SOURCE_INDEX]["修改后"],
        }
    )
    dirty_sample = sample_by_index[DIRTY_SOURCE_INDEX]
    audited.append(
        {
            "抽查序号": dirty_sample["抽查序号"],
            "source_index": DIRTY_SOURCE_INDEX,
            "原始描述": dirty_sample["input"],
            "错误类型": "端部DN与英寸尺寸冲突，删除脏描述",
            "当前标签": dirty_sample["output"],
            "建议标签": "删除样本",
        }
    )

    report = {
        "说明": "第11轮200条均逐条人工审核；修复等径三通成对壁厚，并删除尺寸冲突脏描述。",
        "execute": args.execute,
        "抽查总数": 200,
        "未发现问题": 198,
        "确认有误": 2,
        "待确认": 0,
        "抽查准确率": "99.00%",
        "连续达标轮次": 1,
        "确认有误样本": sorted(audited, key=lambda item: item["抽查序号"]),
        "修复统计": {
            "审核错误已处理": 2,
            "同源额外修复": len(changes) - 1,
            "总修复行数": len(changes),
            "删除脏数据": len(deleted),
        },
        "修改明细": changes,
        "删除明细": deleted,
    }
    if args.execute:
        write_json_atomic(args.dataset, repaired)
        write_json_atomic(args.report, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"确认有误样本", "修改明细", "删除明细"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
