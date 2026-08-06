#!/usr/bin/env python3
"""Build a review report for fitting BODY/CONN migration to annotation V3.1."""

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
    / "管件_train_val_V3.1旧标签迁移审核_20260806.json"
)
DATASETS = {
    "train": DATA_DIR / "管件_train.json",
    "val": DATA_DIR / "管件_val.json",
}

BODY_MIGRATIONS = {
    "对焊支管台": "支管台",
    "承插焊支管台": "支管台",
    "螺纹支管台": "支管台",
    "螺纹管帽": "管帽",
}
THREAD_VALUES = {"THD", "SCRD", "NPT", "NPTF", "FNPT", "MNPT", "FTE", "MTE"}
SPECIFIC_THREAD_PATTERNS = (
    ("FNPT", re.compile(r"(?<![A-Za-z0-9])(?:FNPT|FEMALE\s+NPT)(?![A-Za-z0-9])", re.I)),
    ("MNPT", re.compile(r"(?<![A-Za-z0-9])(?:MNPT|MALE\s+NPT)(?![A-Za-z0-9])", re.I)),
    ("NPTF", re.compile(r"(?<![A-Za-z0-9])NPTF(?![A-Za-z0-9])", re.I)),
    ("FTE", re.compile(r"(?<![A-Za-z0-9])FTE(?![A-Za-z0-9])", re.I)),
    ("MTE", re.compile(r"(?<![A-Za-z0-9])MTE(?![A-Za-z0-9])", re.I)),
    ("SCRD", re.compile(r"(?<![A-Za-z0-9])SCRD(?![A-Za-z0-9])", re.I)),
    ("NPT", re.compile(r"(?<![A-Za-z0-9])NPT(?![A-Za-z0-9])", re.I)),
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


def detect_explicit_thread(text: str) -> str | None:
    for value, pattern in SPECIFIC_THREAD_PATTERNS:
        if pattern.search(text):
            return value
    return None


def normalized_thread_conn(text: str, conn: list[str]) -> list[str]:
    """Keep explicit specific evidence; otherwise use generic THD."""
    explicit = detect_explicit_thread(text)
    if explicit:
        return [explicit]

    specific_current = [value for value in conn if value in THREAD_VALUES and value != "THD"]
    if specific_current:
        return [specific_current[0]]
    return ["THD"]


def build_definite_item(
    dataset: str,
    index: int,
    text: str,
    current: dict[str, Any],
    old_body: str,
) -> dict[str, Any]:
    proposed = copy.deepcopy(current)
    type_data = proposed["TYPE"]
    current_conn = list(type_data.get("CONN", []))
    type_data["BODY"] = BODY_MIGRATIONS[old_body]

    if old_body == "承插焊支管台":
        type_data["CONN"] = ["SW"]
        conn_reason = "原文产品名称明确为承插焊支管台/SOCKOLET，连接形式统一保留为SW。"
    elif old_body in {"螺纹支管台", "螺纹管帽"}:
        type_data["CONN"] = normalized_thread_conn(text, current_conn)
        if type_data["CONN"] == ["THD"]:
            conn_reason = "原文只明确泛螺纹语义，没有具体制式，按固定规则标为THD。"
        else:
            conn_reason = f"保留原文明示的具体螺纹制式{type_data['CONN'][0]}。"
    else:
        type_data["CONN"] = current_conn
        conn_reason = "对焊/BW/BE属于默认端部，不进入CONN，现有连接标签保持不变。"

    proposed["TYPE"] = type_data
    return {
        "来源数据集": dataset,
        "source_index": index,
        "问题类别": "V3.1历史BODY拆分迁移",
        "原始描述": text,
        "修正前标签": current,
        "建议修正标签": proposed,
        "修改摘要": {
            "BODY": f"{old_body} -> {type_data['BODY']}",
            "CONN": f"{current_conn} -> {type_data['CONN']}",
        },
        "中文原因": [
            "V3.1要求管件BODY只保留核心产品结构，普通连接形式写入CONN。",
            conn_reason,
        ],
        "审核状态": "待审核",
    }


def build_manual_item(
    dataset: str,
    index: int,
    text: str,
    current: dict[str, Any],
    old_body: str,
) -> dict[str, Any]:
    base = copy.deepcopy(current)
    base["TYPE"]["BODY"] = BODY_MIGRATIONS[old_body]
    sw_candidate = copy.deepcopy(base)
    sw_candidate["TYPE"]["CONN"] = ["SW"]
    thread_candidate = copy.deepcopy(base)
    thread_candidate["TYPE"]["CONN"] = ["THD"]
    return {
        "来源数据集": dataset,
        "source_index": index,
        "问题类别": "承插焊与泛螺纹互斥连接同时出现",
        "原始描述": text,
        "当前标签": current,
        "固定迁移部分": f"BODY: {old_body} -> {BODY_MIGRATIONS[old_body]}",
        "候选修正标签": [sw_candidate, thread_candidate],
        "中文原因": (
            "原文同时出现SOCKOLET/SW和THR，无法仅凭当前描述确认最终设计端部。"
            "BODY可迁移为支管台，但CONN必须人工选择SW或THD。"
        ),
        "审核状态": "待人工判断",
    }


def build_report() -> dict[str, Any]:
    definite: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    dataset_rows: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    selected_counts: Counter[str] = Counter()
    definite_by_dataset: Counter[str] = Counter()
    manual_by_dataset: Counter[str] = Counter()

    for dataset, path in DATASETS.items():
        rows = load_rows(path)
        dataset_rows[dataset] = len(rows)
        source_hashes[dataset] = sha256(path)
        for index, row in enumerate(rows):
            current = row.get("output", {})
            type_data = current.get("TYPE", {})
            old_body = str(type_data.get("BODY", ""))
            if old_body not in BODY_MIGRATIONS:
                continue

            selected_counts[old_body] += 1
            text = str(row.get("input", ""))
            conn = list(type_data.get("CONN", []))
            has_sw = "SW" in conn
            has_thread = any(value in THREAD_VALUES for value in conn)

            if has_sw and has_thread:
                manual.append(build_manual_item(dataset, index, text, current, old_body))
                manual_by_dataset[dataset] += 1
                continue

            definite.append(build_definite_item(dataset, index, text, current, old_body))
            definite_by_dataset[dataset] += 1

    selected_total = sum(selected_counts.values())
    if selected_total != len(definite) + len(manual):
        raise AssertionError("迁移候选没有完整进入明确建议或人工判断分组")

    return {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "规则版本": "管件种类数据集标注规范V3.1-20260806",
        "说明": (
            "本文件仅提供审核建议，未修改管件_train.json或管件_val.json。"
            "人工确认后才能使用独立写回脚本修改源数据。"
        ),
        "迁移原则": [
            "管件BODY只保留核心产品结构，普通连接形式写入CONN。",
            "泛螺纹且没有具体制式时统一标为THD，不根据产品标准在描述阶段推导NPT。",
            "NPT/FNPT/MNPT/NPTF/SCRD/FTE/MTE等原文明示制式保留最具体值。",
            "SW与螺纹制式同时出现时不自动迁移，必须人工判断。",
        ],
        "数据源": {name: str(path) for name, path in DATASETS.items()},
        "源文件SHA256": source_hashes,
        "统计": {
            "训练集条数": dataset_rows.get("train", 0),
            "验证集条数": dataset_rows.get("val", 0),
            "迁移候选总数": selected_total,
            "明确建议修改条数": len(definite),
            "需要人工判断条数": len(manual),
            "按历史BODY": dict(selected_counts.most_common()),
            "明确建议按数据集": dict(sorted(definite_by_dataset.items())),
            "人工判断按数据集": dict(sorted(manual_by_dataset.items())),
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
