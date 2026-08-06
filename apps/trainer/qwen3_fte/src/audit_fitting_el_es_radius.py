#!/usr/bin/env python3
"""Audit fitting EL/ES product codes whose radius label is missing or conflicting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"
DEFAULT_OUTPUT = (
    DATA_DIR
    / "管件标注审查_20260806"
    / "管件_train_val_EL_ES半径审核_20260806.json"
)
DATASETS = {
    "train": DATA_DIR / "管件_train.json",
    "val": DATA_DIR / "管件_val.json",
}
LONG_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?P<code>(?:W|S)?(?:30|45|60|90)EL)(?![A-Za-z0-9])",
    re.I,
)
SHORT_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?P<code>S(?:30|45|60|90)ES|(?:30|45|60|90)ESS|(?:30|45|60|90)ES)(?![A-Za-z0-9])",
    re.I,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_code(text: str) -> tuple[str, str] | None:
    long_match = LONG_CODE.search(text)
    short_match = SHORT_CODE.search(text)
    matches = [
        (match.start(), match.group("code").upper(), radius)
        for match, radius in ((long_match, "LR"), (short_match, "SR"))
        if match is not None
    ]
    if not matches:
        return None
    _, code, radius = min(matches)
    return code, radius


def build_report() -> dict[str, Any]:
    definite: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}
    hashes: dict[str, str] = {}

    for dataset, path in DATASETS.items():
        rows = load_rows(path)
        row_counts[dataset] = len(rows)
        hashes[dataset] = sha256(path)
        for index, row in enumerate(rows):
            text = str(row.get("input", ""))
            detected = detect_code(text)
            if detected is None:
                continue
            code, expected_radius = detected
            current = row.get("output", {})
            radius = str(current.get("TYPE", {}).get("GEOMETRY", {}).get("RADIUS", ""))

            if not radius:
                proposed = copy.deepcopy(current)
                proposed["TYPE"]["GEOMETRY"]["RADIUS"] = expected_radius
                definite.append(
                    {
                        "来源数据集": dataset,
                        "source_index": index,
                        "问题类别": "EL/ES产品代号半径漏标",
                        "原始描述": text,
                        "命中代号": code,
                        "修正前标签": current,
                        "建议修正标签": proposed,
                        "中文原因": (
                            f"原文独立产品代号{code}中"
                            f"{'EL表示长半径' if expected_radius == 'LR' else 'ES表示短半径'}，"
                            f"当前RADIUS为空，建议补为{expected_radius}。"
                        ),
                        "审核状态": "待审核",
                    }
                )
            elif (expected_radius == "LR" and radius == "SR") or (
                expected_radius == "SR" and radius == "LR"
            ):
                manual.append(
                    {
                        "来源数据集": dataset,
                        "source_index": index,
                        "问题类别": "EL/ES代号与现有半径冲突",
                        "原始描述": text,
                        "命中代号": code,
                        "当前标签": current,
                        "中文原因": (
                            f"代号建议{expected_radius}，但当前标注为{radius}，"
                            "必须回看原文是否存在更精确的半径证据。"
                        ),
                        "审核状态": "待人工判断",
                    }
                )

    definite_counts = Counter(item["来源数据集"] for item in definite)
    code_counts = Counter(item["命中代号"] for item in definite)
    return {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "说明": "本文件仅提供EL/ES半径审核建议，未修改管件train/val。",
        "规则": [
            "独立产品代号EL表示LR，ES表示SR。",
            "ELBOW单词不触发EL代号规则。",
            "原文已明示1D/1.5D/3D或绝对半径时，保留更精确值，不改写为LR/SR。",
        ],
        "数据源": {name: str(path) for name, path in DATASETS.items()},
        "源文件SHA256": hashes,
        "统计": {
            "训练集条数": row_counts.get("train", 0),
            "验证集条数": row_counts.get("val", 0),
            "明确建议修改条数": len(definite),
            "需要人工判断条数": len(manual),
            "明确建议按数据集": dict(sorted(definite_counts.items())),
            "按产品代号": dict(code_counts.most_common()),
        },
        "明确建议修改": definite,
        "需要人工判断": manual,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report["统计"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
