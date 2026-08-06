#!/usr/bin/env python3
"""Apply the reviewed fitting annotation V3.1 BODY/CONN migration."""

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
REVIEW_DIR = DATA_DIR / "管件标注审查_20260806"
DEFAULT_REVIEW = REVIEW_DIR / "管件_train_val_V3.1旧标签迁移审核_20260806.json"
DEFAULT_REPORT = REVIEW_DIR / "管件_train_val_V3.1旧标签迁移修复结果_20260806.json"
DATASETS = {
    "train": DATA_DIR / "管件_train.json",
    "val": DATA_DIR / "管件_val.json",
}
DEPRECATED_BODIES = {"对焊支管台", "承插焊支管台", "螺纹支管台", "螺纹管帽"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.v31.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sockolet_sw(item: dict[str, Any]) -> dict[str, Any]:
    text = item["原始描述"]
    current = item["当前标签"]
    conn = current["TYPE"]["CONN"]
    if not re.search(r"SOCKOLET", text, re.I):
        raise ValueError("人工判断项缺少SOCKOLET完整产品名称，停止写回")
    if not re.search(r"(?<![A-Za-z0-9])SW(?![A-Za-z0-9])", text, re.I):
        raise ValueError("人工判断项缺少明确SW证据，停止写回")
    if "SW" not in conn or "THD" not in conn:
        raise ValueError("人工判断项当前标签不再是SW与THD冲突，停止写回")

    proposed = copy.deepcopy(current)
    proposed["TYPE"]["BODY"] = "支管台"
    proposed["TYPE"]["CONN"] = ["SW"]
    return proposed


def validate_source_hashes(review: dict[str, Any]) -> dict[str, str]:
    expected = review.get("源文件SHA256", {})
    actual = {dataset: sha256(path) for dataset, path in DATASETS.items()}
    for dataset in DATASETS:
        if expected.get(dataset) != actual[dataset]:
            raise ValueError(
                f"{dataset}源文件已在审核后发生变化，停止写回: "
                f"expected={expected.get(dataset)}, actual={actual[dataset]}"
            )
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = load_json(args.review)
    before_hashes = validate_source_hashes(review)
    rows_by_dataset = {name: load_json(path) for name, path in DATASETS.items()}
    original_row_counts = {name: len(rows) for name, rows in rows_by_dataset.items()}
    operations: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    for item in review["明确建议修改"]:
        operations.append((item, item["建议修正标签"], "按V3.1明确迁移建议修正"))
    for item in review["需要人工判断"]:
        operations.append(
            (
                item,
                resolve_sockolet_sw(item),
                "完整产品名称SOCKOLET与明确SW双重证据优先，CONN保留SW",
            )
        )

    expected_total = review["统计"]["迁移候选总数"]
    if len(operations) != expected_total:
        raise ValueError(
            f"写回操作数与审核统计不一致: operations={len(operations)}, expected={expected_total}"
        )

    changes: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item, proposed, conclusion in operations:
        dataset = item["来源数据集"]
        index = item["source_index"]
        key = (dataset, index)
        if key in seen:
            raise ValueError(f"审核项重复: {dataset}[{index}]")
        seen.add(key)

        row = rows_by_dataset[dataset][index]
        expected_input = item["原始描述"]
        expected_before = item.get("修正前标签", item.get("当前标签"))
        if row.get("input") != expected_input:
            raise ValueError(f"原始描述已变化，停止写回: {dataset}[{index}]")
        if row.get("output") != expected_before:
            raise ValueError(f"当前标签已变化，停止写回: {dataset}[{index}]")
        if proposed == expected_before:
            raise ValueError(f"建议标签没有产生实际变化: {dataset}[{index}]")

        before = copy.deepcopy(row["output"])
        row["output"] = copy.deepcopy(proposed)
        changes.append(
            {
                "来源数据集": dataset,
                "source_index": index,
                "原始描述": expected_input,
                "修正前标签": before,
                "修正后标签": proposed,
                "处理结论": conclusion,
            }
        )

    remaining = []
    invalid_schema = []
    for dataset, rows in rows_by_dataset.items():
        if len(rows) != original_row_counts[dataset]:
            raise AssertionError(f"{dataset}数据条数发生变化")
        for index, row in enumerate(rows):
            output = row.get("output", {})
            type_data = output.get("TYPE", {})
            if type_data.get("BODY") in DEPRECATED_BODIES:
                remaining.append((dataset, index, type_data.get("BODY")))
            if output.get("CATEGORY") != "管件" or not isinstance(type_data.get("CONN"), list):
                invalid_schema.append((dataset, index))

    if remaining:
        raise AssertionError(f"迁移后仍有历史BODY: {remaining[:10]}")
    if invalid_schema:
        raise AssertionError(f"迁移后结构异常: {invalid_schema[:10]}")

    for dataset, path in DATASETS.items():
        write_json(path, rows_by_dataset[dataset])

    after_hashes = {dataset: sha256(path) for dataset, path in DATASETS.items()}
    body_transitions = Counter(
        f"{item['修正前标签']['TYPE']['BODY']} -> {item['修正后标签']['TYPE']['BODY']}"
        for item in changes
    )
    conn_transitions = Counter(
        f"{item['修正前标签']['TYPE']['CONN']} -> {item['修正后标签']['TYPE']['CONN']}"
        for item in changes
    )
    dataset_counts = Counter(item["来源数据集"] for item in changes)
    report = {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "审核文件": str(args.review),
        "处理规则": (
            "明确建议项按V3.1审核结果写回；两条SOCKOLET SW/THR冲突项按人工结论"
            "保留CONN=SW；未修改数据条数和原始描述。"
        ),
        "统计": {
            "实际修改条数": len(changes),
            "按数据集": dict(sorted(dataset_counts.items())),
            "按BODY迁移": dict(body_transitions.most_common()),
            "按CONN迁移": dict(conn_transitions.most_common()),
            "训练集条数": len(rows_by_dataset["train"]),
            "验证集条数": len(rows_by_dataset["val"]),
            "剩余历史BODY条数": len(remaining),
        },
        "源文件SHA256_修复前": before_hashes,
        "源文件SHA256_修复后": after_hashes,
        "修改明细": changes,
    }
    write_json(args.report, report)
    print(json.dumps(report["统计"], ensure_ascii=False, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
