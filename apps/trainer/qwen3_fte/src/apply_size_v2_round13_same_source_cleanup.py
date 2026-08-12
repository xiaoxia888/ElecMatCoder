#!/usr/bin/env python3
"""Remove the remaining round-13 same-source conflicting-size row."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.apply_size_v2_group_decisions import write_json_atomic  # noqa: E402


DATASET = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核/02_V2已审核通过数据.json"
)
REPORT = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核/23_人工抽查200条第十三轮审核与修复结果.json"
)
TARGET_INPUT = (
    "异径三通,GB/T13401-CF415,BE,GB/T12459,SeriesI,SMLS,"
    'DN100xDN65,S-40xS-40,_GB/T_13401-CF415;40x2.5"'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    matches = [(index, row) for index, row in enumerate(rows) if row.get("input") == TARGET_INPUT]
    if len(matches) != 1:
        raise ValueError(f"同源脏描述匹配数量异常: {len(matches)}")
    index, row = matches[0]
    detail = {
        "source_index_at_cleanup": index,
        "原始描述": row["input"],
        "删除原因": "同源尺寸冲突：DN100×DN65与40×2.5英寸不等价",
        "原标签": deepcopy(row["output"]),
    }
    repaired = [item for item in rows if item.get("input") != TARGET_INPUT]
    result = {"execute": args.execute, "删除行数": 1, "删除后行数": len(repaired), "删除明细": detail}
    if args.execute:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report.setdefault("删除明细", []).append(detail)
        report["修复统计"]["同源额外删除"] = report["修复统计"].get("同源额外删除", 0) + 1
        report["修复统计"]["删除脏数据"] = report["修复统计"].get("删除脏数据", 0) + 1
        write_json_atomic(DATASET, repaired)
        write_json_atomic(REPORT, report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
