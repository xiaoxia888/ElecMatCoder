#!/usr/bin/env python3
"""Export original-grade material datasets as an Excel-friendly audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    QWEN_ROOT
    / "output"
    / "按8类拆分数据集"
    / "材质规范"
    / "原始牌号"
)
DEFAULT_TRAIN = SOURCE_DIR / "材质规范_原始牌号_train.json"
DEFAULT_VAL = SOURCE_DIR / "材质规范_原始牌号_val.json"
DEFAULT_OUTPUT = SOURCE_DIR / "材质规范_原始牌号_标注审计.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出原始牌号材质标注审计 CSV")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} 顶层必须是数组")
    return value


def standard_text(output: dict[str, Any]) -> str:
    return " | ".join(
        str(item.get("BODY") or "").strip()
        for item in output.get("STANDARD") or []
        if str(item.get("BODY") or "").strip()
    )


def special_text(item: dict[str, Any]) -> str:
    return " | ".join(
        str(value).strip()
        for value in item.get("SPECIAL_REQ") or []
        if str(value).strip()
    )


def annotation_key(material: dict[str, Any]) -> str:
    parts = []
    for item in material.get("ITEMS") or []:
        parts.append(
            "/".join(
                [
                    str(item.get("ROLE") or ""),
                    str(item.get("MATERIAL_STANDARD") or ""),
                    str(item.get("GRADE") or ""),
                    str(item.get("QUALITY_LEVEL") or ""),
                    special_text(item),
                ]
            )
        )
    return f"{material.get('RELATION', '')}::{' || '.join(parts)}"


def collect_records(
    datasets: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_order = 0
    for split, rows in datasets:
        for split_index, row in enumerate(rows, start=1):
            source_order += 1
            output = row.get("output") or {}
            material = output.get("MATERIAL") or {}
            items = material.get("ITEMS") or []
            complete_key = annotation_key(material)
            for item_index, item in enumerate(items, start=1):
                records.append(
                    {
                        "_source_order": source_order,
                        "_grade_key": (
                            str(item.get("MATERIAL_STANDARD") or ""),
                            str(item.get("GRADE") or ""),
                            str(item.get("QUALITY_LEVEL") or ""),
                        ),
                        "_annotation_key": complete_key,
                        "数据集": split,
                        "数据集行号": split_index,
                        "原始描述": str(row.get("input") or ""),
                        "材质关系": str(material.get("RELATION") or ""),
                        "材质项序号": item_index,
                        "材质项总数": len(items),
                        "角色": str(item.get("ROLE") or ""),
                        "材料标准": str(item.get("MATERIAL_STANDARD") or ""),
                        "原始牌号": str(item.get("GRADE") or ""),
                        "材料等级": str(item.get("QUALITY_LEVEL") or ""),
                        "特殊要求": special_text(item),
                        "顶层规范": standard_text(output),
                        "完整标注组合": complete_key,
                    }
                )
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    grade_counts = Counter(record["_grade_key"] for record in records)
    annotation_counts = Counter(record["_annotation_key"] for record in records)
    records.sort(
        key=lambda record: (
            record["材料标准"],
            record["原始牌号"],
            record["材料等级"],
            record["材质关系"],
            record["角色"],
            record["_source_order"],
            record["材质项序号"],
        )
    )

    fieldnames = [
        "序号",
        "数据集",
        "数据集行号",
        "原始描述",
        "材质关系",
        "材质项序号",
        "材质项总数",
        "角色",
        "材料标准",
        "原始牌号",
        "材料等级",
        "特殊要求",
        "顶层规范",
        "牌号标注出现次数",
        "完整标注组合出现次数",
        "完整标注组合",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            writer.writerow(
                {
                    "序号": index,
                    **{
                        key: record[key]
                        for key in fieldnames
                        if key
                        not in {
                            "序号",
                            "牌号标注出现次数",
                            "完整标注组合出现次数",
                        }
                    },
                    "牌号标注出现次数": grade_counts[record["_grade_key"]],
                    "完整标注组合出现次数": annotation_counts[
                        record["_annotation_key"]
                    ],
                }
            )


def main() -> int:
    args = parse_args()
    train = read_rows(args.train)
    val = read_rows(args.val)
    records = collect_records([("train", train), ("val", val)])
    write_csv(args.output, records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_rows": len(train) + len(val),
                "material_item_rows": len(records),
                "train_rows": len(train),
                "val_rows": len(val),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
