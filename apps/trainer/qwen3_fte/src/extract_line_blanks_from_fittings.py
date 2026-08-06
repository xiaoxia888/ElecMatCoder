#!/usr/bin/env python3
"""Extract mislabeled line blanks from fitting datasets for manual review."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


LINE_BLANK_PATTERN = re.compile(
    r"插板|垫环|八字盲板|8字盲板|"
    r"SPECTACLE\s*(?:BLIND|BLANK)|FIGURE\s*-?\s*8\s*(?:BLIND|BLANK)?|"
    r"PADDLE\s*(?:BLANK|BLIND|SPACER)|\bSPADE\b|"
    r"\bBLANK\b|\bSPACER\b",
    re.IGNORECASE,
)

SPECTACLE_PATTERN = re.compile(
    r"八字盲板|8字盲板|SPECTACLE\s*(?:BLIND|BLANK)|"
    r"FIGURE\s*-?\s*8\s*(?:BLIND|BLANK)?",
    re.IGNORECASE,
)

COMPONENT_PATTERN = re.compile(
    r"成套供货|"
    r"(?:PADDLE\s*)?BLANK\s*(?:AND|&|\+)\s*SPACER|"
    r"SPACER\s*(?:AND|&|\+)\s*BLANKS?|"
    r"插板\s*(?:[/+、和及&]|与)\s*垫环|"
    r"盲板\s*(?:[,，/+、和及&]|与)\s*垫环",
    re.IGNORECASE,
)

SPACER_PATTERN = re.compile(r"PADDLE\s*SPACER|(?:^|[,，;；:/])\s*垫环(?:\s|[,，;；:/]|$)", re.IGNORECASE)
BLANK_PATTERN = re.compile(r"插板|PADDLE\s*(?:BLANK|BLIND)|\bSPADE\b|\bBLANK\b", re.IGNORECASE)


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return data


def classify_body(text: str) -> tuple[str, str]:
    if SPECTACLE_PATTERN.search(text):
        return "8字盲板", "原文明示8字盲板、SPECTACLE BLIND或FIGURE 8结构"
    if COMPONENT_PATTERN.search(text) or (
        re.search(r"插板|\bSPADE\b|\bBLANK\b", text, re.IGNORECASE)
        and re.search(r"垫环|\bSPACER\b", text, re.IGNORECASE)
    ):
        return "插板垫环组件", "原文明示插板与垫环同时配套或成套供货"
    if SPACER_PATTERN.search(text):
        return "垫环", "原文明示PADDLE SPACER或单垫环"
    if BLANK_PATTERN.search(text):
        return "插板", "原文明示PADDLE BLANK、SPADE、BLANK或插板"
    raise ValueError(f"无法确定管道盲板主体: {text}")


def extract_seals(text: str) -> list[str]:
    upper = text.upper()

    specific_ring: tuple[str, re.Pattern[str]] | None = None
    female = re.search(r"FEMALE\s*(?:RING[ -]?JOINT|RJ)", upper)
    male = re.search(r"MALE\s*(?:RING[ -]?JOINT|RJ)", upper)
    if female:
        specific_ring = ("FRJ", re.compile(re.escape(female.group(0))))
    elif male:
        specific_ring = ("MRJ", re.compile(re.escape(male.group(0))))

    candidates: list[tuple[int, int, str]] = []
    definitions: list[tuple[str, str]] = [
        ("SERRATED", r"\bSERRATED(?:\s+FINISH)?\b"),
        ("MFM", r"(?<![A-Z0-9])MFM(?![A-Z0-9])"),
        ("LM", r"(?<![A-Z0-9])LMFE(?![A-Z0-9])|(?<![A-Z0-9])LM(?![A-Z0-9])"),
        ("LF", r"(?<![A-Z0-9])LF(?![A-Z0-9])"),
        ("MF", r"(?<![A-Z0-9])MF(?![A-Z0-9])"),
        ("FM", r"(?<![A-Z0-9])FM(?![A-Z0-9])"),
        ("TG", r"(?<![A-Z0-9])TG(?![A-Z0-9])"),
        ("T", r"\bTONGUE(?:\s+FACE)?\b|(?<![A-Z0-9/])T(?![A-Z0-9])"),
        ("G", r"\bGROOVE(?:\s+FACE)?\b|(?<![A-Z0-9./])G(?![A-Z0-9])"),
        ("FF", r"WFR\s*-\s*FF|(?<![A-Z0-9])FF(?![A-Z0-9])"),
        ("RF", r"(?<![A-Z0-9])RF(?![A-Z0-9])"),
    ]

    if specific_ring:
        label, pattern = specific_ring
        match = pattern.search(upper)
        if match:
            candidates.append((match.start(), match.end(), label))
    else:
        definitions.extend(
            [
                ("RJ", r"(?<![A-Z0-9])FLRJ(?![A-Z0-9])"),
                ("RTJ", r"(?<![A-Z0-9])RTJ(?![A-Z0-9])"),
                ("RJ", r"(?<![A-Z0-9])RJ(?![A-Z0-9])"),
            ]
        )

    for label, pattern in definitions:
        for match in re.finditer(pattern, upper):
            candidates.append((match.start(), match.end(), label))

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    seals: list[str] = []
    occupied: list[tuple[int, int]] = []
    for start, end, label in candidates:
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        if label not in seals:
            seals.append(label)
    return seals


def corrected_output(text: str) -> tuple[dict[str, Any], str]:
    body, reason = classify_body(text)
    return {
        "CATEGORY": "法兰",
        "TYPE": {
            "BODY": body,
            "CONN": [],
            "SEAL": extract_seals(text),
        },
    }, reason


def process_split(
    path: Path, split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    corrected: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for index, row in enumerate(load_json(path)):
        text = str(row.get("input", ""))
        if not LINE_BLANK_PATTERN.search(text):
            remaining.append(row)
            continue
        output, reason = corrected_output(text)
        corrected.append({"input": text, "output": output})
        review.append(
            {
                "来源数据集": split,
                "source_index": index,
                "原始描述": text,
                "修正前标签": row.get("output", {}),
                "修正后标签": output,
                "中文原因": reason,
            }
        )
    return corrected, review, remaining


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="管件训练集")
    parser.add_argument("--val", type=Path, required=True, help="管件验证集")
    parser.add_argument("--output-dir", type=Path, required=True, help="待审核输出目录")
    parser.add_argument(
        "--remove-from-source",
        action="store_true",
        help="从管件训练集和验证集中移除已提取样本，并在输出目录保留原文件备份",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_review, train_remaining = process_split(args.train, "train")
    val_rows, val_review, val_remaining = process_split(args.val, "val")
    review = train_review + val_review

    if args.remove_from_source and not review:
        raise RuntimeError("源文件中没有找到可提取的管道盲板样本，拒绝覆盖审核文件")

    train_path = args.output_dir / "管件_train_迁移法兰_修正后.json"
    val_path = args.output_dir / "管件_val_迁移法兰_修正后.json"
    review_path = args.output_dir / "管件盲板类迁移_修正前后审核.json"
    report_path = args.output_dir / "管件盲板类迁移_统计.json"

    write_json(train_path, train_rows)
    write_json(val_path, val_rows)
    write_json(review_path, review)

    backup_paths: dict[str, str] = {}
    if args.remove_from_source:
        train_backup = args.output_dir / "管件_train_抽取前备份.json"
        val_backup = args.output_dir / "管件_val_抽取前备份.json"
        shutil.copy2(args.train, train_backup)
        shutil.copy2(args.val, val_backup)
        write_json(args.train, train_remaining)
        write_json(args.val, val_remaining)
        backup_paths = {"train": str(train_backup), "val": str(val_backup)}

    body_counts = Counter(
        item["修正后标签"]["TYPE"]["BODY"]
        for item in review
    )
    seal_counts = Counter(
        seal
        for item in review
        for seal in item["修正后标签"]["TYPE"]["SEAL"]
    )
    report = {
        "source": {"train": str(args.train), "val": str(args.val)},
        "output": {
            "train": str(train_path),
            "val": str(val_path),
            "review": str(review_path),
            "source_backups": backup_paths,
        },
        "statistics": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "total_rows": len(review),
            "train_remaining_rows": len(train_remaining),
            "val_remaining_rows": len(val_remaining),
            "body_counts": dict(body_counts),
            "seal_counts": dict(seal_counts),
        },
        "source_datasets_modified": args.remove_from_source,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
