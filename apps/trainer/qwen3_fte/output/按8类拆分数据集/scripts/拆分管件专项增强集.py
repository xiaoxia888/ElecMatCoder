#!/usr/bin/env python3
"""Split reviewed fitting augmentations by complete contrast groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
TYPE_DIR = REPO_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"
DEFAULT_INPUT = (
    TYPE_DIR
    / "专项增强_20260806_V31"
    / "管件专项对比训练集_审核版_带来源.json"
)
DEFAULT_OUTPUT_DIR = TYPE_DIR / "专项增强_20260806_V31"
DEFAULT_SEED = "20260806-fitting-augmentation-split-v1"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"输入文件顶层必须是数组: {path}")
    return rows


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def stable_score(seed: str, family: str, group_id: str) -> str:
    raw = f"{seed}\0{family}\0{group_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def row_key(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise ValueError("增强集不能为空")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_inputs: set[str] = set()
    for index, row in enumerate(rows):
        family = str(row.get("增强类型", "")).strip()
        group_id = str(row.get("增强组ID", "")).strip()
        text = str(row.get("input", "")).strip()
        if not family or not group_id or not text:
            raise ValueError(f"缺少增强类型、增强组ID或input: index={index}")
        if text in seen_inputs:
            raise ValueError(f"输入重复: index={index}, input={text}")
        seen_inputs.add(text)
        grouped[family][group_id].append(row)

    val_groups: set[str] = set()
    family_selection: dict[str, Any] = {}
    for family, groups in sorted(grouped.items()):
        if len(groups) < 2:
            raise ValueError(f"增强类型至少需要2个组才能拆分: {family}")
        selected = min(groups, key=lambda group: stable_score(args.seed, family, group))
        val_groups.add(selected)
        family_selection[family] = {
            "总组数": len(groups),
            "验证组": selected,
            "验证组样本数": len(groups[selected]),
        }

    raw_train_rows = [row for row in rows if row["增强组ID"] not in val_groups]
    raw_val_rows = [row for row in rows if row["增强组ID"] in val_groups]

    train_inputs = {row["input"] for row in raw_train_rows}
    val_inputs = {row["input"] for row in raw_val_rows}
    train_groups = {row["增强组ID"] for row in raw_train_rows}
    actual_val_groups = {row["增强组ID"] for row in raw_val_rows}
    if train_inputs & val_inputs:
        raise ValueError("train/val存在input交集")
    if train_groups & actual_val_groups:
        raise ValueError("train/val存在增强组交集")
    if Counter(map(row_key, rows)) != Counter(map(row_key, raw_train_rows + raw_val_rows)):
        raise ValueError("拆分后的样本并集与原集合不一致")

    train_families = Counter(row["增强类型"] for row in raw_train_rows)
    val_families = Counter(row["增强类型"] for row in raw_val_rows)
    missing_train = set(grouped) - set(train_families)
    missing_val = set(grouped) - set(val_families)
    if missing_train or missing_val:
        raise ValueError(
            f"增强类型覆盖不完整: train={sorted(missing_train)}, val={sorted(missing_val)}"
        )

    train_rows = [
        {**row, "拆分用途": "train", "建议用途": "仅并入训练集"}
        for row in raw_train_rows
    ]
    val_rows = [
        {
            **row,
            "拆分用途": "val",
            "建议用途": "仅并入验证集，作为合成专项回归样本",
        }
        for row in raw_val_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "管件专项增强_train_待合并_带来源.json"
    val_path = args.output_dir / "管件专项增强_val_待合并_带来源.json"
    report_path = args.output_dir / "管件专项增强_train_val拆分报告.json"
    dump_json(train_path, train_rows)
    dump_json(val_path, val_rows)

    report = {
        "输入文件": str(args.input),
        "随机种子": args.seed,
        "拆分策略": "每个增强类型稳定选取1个完整对比组进入val，同一增强组不跨train/val。",
        "重要说明": "val为合成增强验证数据，可用于专项回归，不能替代真实项目独立验证集。",
        "输出文件": {"train": str(train_path), "val": str(val_path)},
        "统计": {
            "总样本数": len(rows),
            "train样本数": len(train_rows),
            "val样本数": len(val_rows),
            "train占比": round(len(train_rows) / len(rows), 6),
            "val占比": round(len(val_rows) / len(rows), 6),
            "train增强组数": len(train_groups),
            "val增强组数": len(actual_val_groups),
            "train按增强类型": dict(train_families),
            "val按增强类型": dict(val_families),
        },
        "各类型验证组选择": family_selection,
        "质量检查": {
            "train_val_input交集数": 0,
            "train_val增强组交集数": 0,
            "拆分后并集与原数据一致": True,
            "train_val均覆盖全部增强类型": True,
        },
    }
    dump_json(report_path, report)
    print(json.dumps(report["统计"], ensure_ascii=False))
    print(train_path)
    print(val_path)
    print(report_path)


if __name__ == "__main__":
    main()
