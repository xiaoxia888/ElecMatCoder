#!/usr/bin/env python3
"""Create a reproducible, skeleton-diverse sample for manual label review."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "尺寸壁厚磅级C1训练集.json"
)
DEFAULT_OUTPUT = DEFAULT_DATASET.with_name("尺寸壁厚磅级C1训练集_人工抽查200条.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-per-skeleton", type=int, default=1)
    return parser.parse_args()


def description_shape(text: str) -> str:
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_english = bool(re.search(r"[A-Za-z]", text))
    if has_chinese and has_english:
        language = "mixed"
    elif has_chinese:
        language = "zh"
    else:
        language = "en"
    layout = "multiline" if "\n" in text else "singleline"
    compact = "compact" if len(re.findall(r"\s", text)) <= max(1, len(text) // 40) else "spaced"
    return f"{language}:{layout}:{compact}"


def skeleton(text: str) -> str:
    normalized = text.upper()
    normalized = re.sub(r"\d+(?:\.\d+)?(?:/\d+)?", "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[，,;；:：/\\()（）\[\]{}_-]+", "|", normalized)
    return normalized.strip(" |").strip()


def item_types(output: dict[str, Any], field: str) -> tuple[str, ...]:
    items = output.get(field)
    if not isinstance(items, list):
        return ()
    return tuple(str(item.get("type") or "EMPTY").upper() for item in items)


def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    return (
        item_types(output, "SIZE_ITEMS"),
        item_types(output, "THICKNESS_ITEMS"),
        bool(output.get("LENGTH")),
        bool(output.get("PRESSURE")),
        description_shape(str(row.get("input") or "")),
    )


def main() -> int:
    args = parse_args()
    rows = json.loads(args.dataset.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("训练集顶层必须为JSON数组")

    rng = random.Random(args.seed)
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[stratum(row)].append(index)
    for indexes in buckets.values():
        rng.shuffle(indexes)

    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    selected: list[int] = []
    selected_set: set[int] = set()
    skeleton_counts: dict[str, int] = defaultdict(int)
    cursors = {key: 0 for key in bucket_keys}

    while len(selected) < args.count:
        progressed = False
        for key in bucket_keys:
            indexes = buckets[key]
            while cursors[key] < len(indexes):
                index = indexes[cursors[key]]
                cursors[key] += 1
                row_skeleton = skeleton(str(rows[index].get("input") or ""))
                if skeleton_counts[row_skeleton] >= args.max_per_skeleton:
                    continue
                selected.append(index)
                selected_set.add(index)
                skeleton_counts[row_skeleton] += 1
                progressed = True
                break
            if len(selected) >= args.count:
                break
        if not progressed:
            break

    if len(selected) < args.count:
        remaining = [index for index in range(len(rows)) if index not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.count - len(selected)])

    samples = [
        {
            "抽查序号": sample_number,
            "source_index": source_index,
            "input": rows[source_index].get("input", ""),
            "output": rows[source_index].get("output", {}),
        }
        for sample_number, source_index in enumerate(selected, start=1)
    ]
    result = {
        "说明": "本文件仅由程序分层抽样，不包含任何自动标签判断。",
        "源数据集": str(args.dataset.expanduser().resolve()),
        "随机种子": args.seed,
        "抽查条数": len(samples),
        "相同骨架最大抽取数": args.max_per_skeleton,
        "样本": samples,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "count": len(samples)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
