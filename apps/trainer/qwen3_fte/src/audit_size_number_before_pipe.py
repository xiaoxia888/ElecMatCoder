#!/usr/bin/env python3
"""Export every `number + PIPE` match for manual review without changing labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from apps.trainer.qwen3_fte.src.audit_size_dataset_second_round import (
    canonical_number,
    insert_size_item_in_source_order,
)


NUMBER_BEFORE_PIPE_RE = re.compile(
    r"(?<![A-Z0-9])(?P<value>\d+(?:\.\d+)?)(?P<space>\s*)PIPE\b",
    re.IGNORECASE,
)
STANDARD_BEFORE_PIPE_RE = re.compile(
    r"(?:SH/T|GB/T|HG/T|NB/T|ASME|ASTM|DIN|EN)"
    r"(?:\s*[-/]?\s*[A-Z]+)?\s*[-/]?\s*\d+(?:\.\d+)?\s*$",
    re.IGNORECASE,
)
WALL_BEFORE_PIPE_RE = re.compile(r"\d+(?:\.\d+)?\s*[xX×*]\s*$")


def classify_match(row: dict[str, Any], match: re.Match[str]) -> tuple[str, str, list[dict[str, str]]]:
    text = str(row.get("input") or "")
    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    current_items = output.get("SIZE_ITEMS") if isinstance(output.get("SIZE_ITEMS"), list) else []
    value = canonical_number(match.group("value"))
    same_value_types = [
        str(item.get("type") or "")
        for item in current_items
        if canonical_number(item.get("value")) == value
    ]

    if not match.group("space"):
        if "INCH" in same_value_types:
            return (
                "紧粘连英制尺寸已标注",
                "数字与PIPE无空格粘连，当前已保留为INCH，无需修改。",
                deepcopy(current_items),
            )
        proposed = deepcopy(current_items)
        insert_size_item_in_source_order(
            text,
            proposed,
            {"type": "INCH", "value": value},
            match.start("value"),
        )
        return (
            "紧粘连英制尺寸待补标",
            "数字与PIPE无空格粘连，在该批管道骨架中表示寸径；建议补充INCH并保留已有DN/OD证据。",
            proposed,
        )

    left_with_value = text[max(0, match.start("value") - 32) : match.end("value")]
    if STANDARD_BEFORE_PIPE_RE.search(left_with_value):
        return (
            "规范号并非尺寸",
            "命中数字属于PIPE前的规范编号，例如SH/T 3405、GB/T 9711或ASME B36.10，不能标为尺寸。",
            deepcopy(current_items),
        )

    left = text[max(0, match.start("value") - 24) : match.start("value")]
    if WALL_BEFORE_PIPE_RE.search(left):
        return (
            "壁厚值并非尺寸",
            "命中数字是外径×壁厚结构中紧邻PIPE的右侧壁厚，不应写入SIZE_ITEMS。",
            deepcopy(current_items),
        )

    if value in {"304", "316", "316L", "321"}:
        return (
            "材质牌号并非尺寸",
            "命中数字是PIPE前的材质牌号，不应写入SIZE_ITEMS。",
            deepcopy(current_items),
        )

    if "DN" in same_value_types:
        return (
            "空格分隔DN已标注",
            "该裸数字在原文其他位置有同值DN证据，当前已保留DN，无需重复增加。",
            deepcopy(current_items),
        )

    return (
        "空格分隔语义待人工确认",
        "数字与PIPE之间存在空格，且未命中规范、壁厚、材质或已有DN规则，不自动推断。",
        deepcopy(current_items),
    )


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        text = str(row.get("input") or "")
        match = NUMBER_BEFORE_PIPE_RE.search(text)
        if not match:
            continue
        category, reason, proposed = classify_match(row, match)
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        current = output.get("SIZE_ITEMS") if isinstance(output.get("SIZE_ITEMS"), list) else []
        value = canonical_number(match.group("value"))
        entry = {
            "source_index": index,
            "问题类别": category,
            "原始描述": text,
            "命中片段": match.group(0),
            "候选数字": value,
            "连接形式": "紧粘连" if not match.group("space") else "空格分隔",
            "当前SIZE_ITEMS": deepcopy(current),
            "建议SIZE_ITEMS": proposed,
            "是否建议修改": proposed != current,
            "中文原因": reason,
        }
        groups.setdefault(category, []).append(entry)

    ordered_groups = {
        category: groups.get(category, [])
        for category in (
            "紧粘连英制尺寸待补标",
            "紧粘连英制尺寸已标注",
            "规范号并非尺寸",
            "壁厚值并非尺寸",
            "材质牌号并非尺寸",
            "空格分隔DN已标注",
            "空格分隔语义待人工确认",
        )
    }
    count = sum(len(items) for items in ordered_groups.values())
    if count != 329:
        raise ValueError(f"预期329条，实际命中{count}条")
    return {
        "说明": "本文件仅为数字+PIPE命中项审核清单，未写回训练集。",
        "训练集总条数": len(rows),
        "命中总条数": count,
        "分类统计": {category: len(items) for category, items in ordered_groups.items()},
        "建议修改条数": sum(
            entry["是否建议修改"]
            for items in ordered_groups.values()
            for entry in items
        ),
        "审核清单": ordered_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = audit(rows)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "输出文件": str(args.output),
                "命中总条数": report["命中总条数"],
                "分类统计": report["分类统计"],
                "建议修改条数": report["建议修改条数"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
