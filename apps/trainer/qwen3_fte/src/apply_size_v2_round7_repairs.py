#!/usr/bin/env python3
"""Apply round-7 paired-thickness fixes and remove confirmed dirty siblings."""

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


BASE_DIR = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2转换审核"
)
DEFAULT_DATASET = BASE_DIR / "02_V2已审核通过数据.json"
DEFAULT_SAMPLE = Path("/private/tmp/v2_size_manual_sample_200_round7.json")
DEFAULT_REPORT = BASE_DIR / "17_人工抽查200条第七轮审核与修复结果.json"

PAIRED_WALL_ERRORS = {
    47467: "异径三通支管明示3.5mm漏标",
    49800: "等径三通支管明示4.0mm漏标",
    50403: "异径管第二端明示8.0mm漏标",
    50373: "异径管第二端明示SCH80漏标",
    49834: "等径三通支管明示3.5mm漏标",
}
AUDITED_DIRTY = {
    1225: "管子OD32与DN150明显冲突",
    12457: "同一弯头同时明示6.50mm与4.00mm，壁厚冲突",
}
# Same-skeleton rows were individually inspected; only these records have
# physically incompatible OD/DN pairs or mutually contradictory wall values.
DIRTY_SIBLINGS = {
    1153,
    1157,
    1170,
    1181,
    1195,
    1206,
    1207,
    1223,
    1224,
    12456,
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

    audited_errors = {**PAIRED_WALL_ERRORS, **AUDITED_DIRTY}
    sample_by_index = {int(item["source_index"]): item for item in samples}
    if not set(audited_errors).issubset(sample_by_index):
        raise ValueError("第七轮错误索引不完整")
    for index in audited_errors:
        if sample_by_index[index]["input"] != source_rows[index]["input"]:
            raise ValueError(f"source_index={index}原文变化")
        if sample_by_index[index]["output"] != source_rows[index]["output"]:
            raise ValueError(f"source_index={index}标签变化")

    paired_skeletons = {
        description_skeleton(source_rows[index]["input"])
        for index in PAIRED_WALL_ERRORS
    }
    dirty_indexes = set(AUDITED_DIRTY) | DIRTY_SIBLINGS
    repaired_rows = deepcopy(source_rows)
    changes: list[dict[str, Any]] = []

    for index, row in enumerate(repaired_rows):
        text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        if description_skeleton(text) not in paired_skeletons:
            continue
        roles = body_role_items(output)
        positions = second_position(roles)
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
                "处理方式": "修复第二位置明示壁厚",
                "修改前": before,
                "修改后": deepcopy(output),
            }
        )

    changed_indexes = {item["source_index"] for item in changes}
    missing = sorted(set(PAIRED_WALL_ERRORS) - changed_indexes)
    if missing:
        raise ValueError(f"第七轮壁厚错误未全部修复: {missing}")

    deleted = [
        {
            "source_index": index,
            "原始描述": source_rows[index]["input"],
            "处理方式": "删除原文冲突脏数据",
        }
        for index in sorted(dirty_indexes)
    ]
    final_rows = [row for index, row in enumerate(repaired_rows) if index not in dirty_indexes]
    if len(final_rows) != len(source_rows) - len(dirty_indexes):
        raise ValueError("删除行数校验失败")
    validate_dataset(final_rows)

    change_by_index = {item["source_index"]: item for item in changes}
    audited_details = []
    for index, error_type in audited_errors.items():
        sample_item = sample_by_index[index]
        if index in change_by_index:
            suggestion: Any = change_by_index[index]["修改后"]
        else:
            suggestion = "删除该条脏数据"
        audited_details.append(
            {
                "抽查序号": sample_item["抽查序号"],
                "source_index": index,
                "原始描述": sample_item["input"],
                "错误类型": error_type,
                "当前标签": sample_item["output"],
                "建议处理": suggestion,
            }
        )

    report = {
        "说明": "第七轮200条均逐条人工审核；成对壁厚漏标已修复，原文冲突脏数据已删除。",
        "execute": args.execute,
        "抽查总数": 200,
        "未发现问题": 193,
        "确认有误": 7,
        "待确认": 0,
        "抽查准确率": "96.50%",
        "连续达标轮次": 0,
        "确认有误样本": sorted(audited_details, key=lambda item: item["抽查序号"]),
        "修复统计": {
            "审核壁厚错误已修复": len(set(PAIRED_WALL_ERRORS) & changed_indexes),
            "同源额外壁厚修复": len(changed_indexes - set(PAIRED_WALL_ERRORS)),
            "壁厚修复总数": len(changed_indexes),
            "审核脏数据删除": len(set(AUDITED_DIRTY) & dirty_indexes),
            "同源脏数据删除": len(DIRTY_SIBLINGS),
            "脏数据删除总数": len(dirty_indexes),
        },
        "修改明细": changes,
        "删除明细": deleted,
    }

    if args.execute:
        write_json_atomic(args.dataset, final_rows)
        write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"确认有误样本", "修改明细", "删除明细"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
