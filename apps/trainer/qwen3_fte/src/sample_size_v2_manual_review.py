#!/usr/bin/env python3
"""Sample skeleton-diverse V2 size labels while excluding earlier rounds."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_fte.src.prepare_size_dataset_v2_conversion import (
    description_skeleton,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "V2转换审核"
    / "02_V2已审核通过数据.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-per-skeleton", type=int, default=1)
    return parser.parse_args()


def description_shape(text: str) -> str:
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_english = bool(re.search(r"[A-Za-z]", text))
    language = "mixed" if has_chinese and has_english else "zh" if has_chinese else "en"
    layout = "multiline" if "\n" in text else "singleline"
    compact = "compact" if len(re.findall(r"\s", text)) <= max(1, len(text) // 40) else "spaced"
    return f"{language}:{layout}:{compact}"


def leaf_types(item: dict[str, Any], field: str) -> tuple[str, ...]:
    return tuple(
        str(value.get("type") or "EMPTY").upper()
        for value in item.get(field) or []
        if isinstance(value, dict)
    )


def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    items = output.get("ITEMS") if isinstance(output.get("ITEMS"), list) else []
    item_shapes = tuple(
        (
            str(item.get("SCOPE") or ""),
            str(item.get("ROLE") or ""),
            leaf_types(item, "SIZE"),
            leaf_types(item, "THICKNESS"),
        )
        for item in items
        if isinstance(item, dict)
    )
    return (
        item_shapes,
        bool(output.get("LENGTH")),
        bool(output.get("PRESSURE")),
        description_shape(str(row.get("input") or "")),
    )


def row_fingerprint(row: dict[str, Any]) -> str:
    # Labels can change after a reviewed repair, while the source description
    # remains the stable identity needed to avoid resampling the same record.
    return str(row.get("input") or "")


def load_excluded(paths: list[Path]) -> tuple[set[int], set[str]]:
    excluded_indexes: set[int] = set()
    excluded_fingerprints: set[str] = set()
    for path in paths:
        document = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        samples = document.get("样本") if isinstance(document, dict) else document
        if not isinstance(samples, list):
            raise ValueError(f"排除文件格式不支持: {path}")
        for sample in samples:
            if isinstance(sample, dict) and "source_index" in sample:
                excluded_indexes.add(int(sample["source_index"]))
            if isinstance(sample, dict) and "input" in sample and "output" in sample:
                excluded_fingerprints.add(row_fingerprint(sample))
    return excluded_indexes, excluded_fingerprints


def main() -> int:
    args = parse_args()
    rows = json.loads(args.dataset.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset顶层必须是数组")
    excluded_indexes, excluded_fingerprints = load_excluded(args.exclude)
    rng = random.Random(args.seed)

    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row_fingerprint(row) in excluded_fingerprints:
            continue
        # Index fallback keeps compatibility with older exclusion files that do
        # not contain complete input/output snapshots.
        if not excluded_fingerprints and index in excluded_indexes:
            continue
        buckets[stratum(row)].append(index)
    for indexes in buckets.values():
        rng.shuffle(indexes)
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)

    selected: list[int] = []
    skeleton_counts: dict[str, int] = defaultdict(int)
    cursors = {key: 0 for key in bucket_keys}
    while len(selected) < args.count:
        progressed = False
        for key in bucket_keys:
            indexes = buckets[key]
            while cursors[key] < len(indexes):
                index = indexes[cursors[key]]
                cursors[key] += 1
                skeleton = description_skeleton(rows[index].get("input", ""))
                if skeleton_counts[skeleton] >= args.max_per_skeleton:
                    continue
                skeleton_counts[skeleton] += 1
                selected.append(index)
                progressed = True
                break
            if len(selected) >= args.count:
                break
        if not progressed:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"只能抽取{len(selected)}条，目标为{args.count}条")

    samples = [
        {
            "抽查序号": number,
            "source_index": index,
            "input": rows[index].get("input", ""),
            "output": rows[index].get("output", {}),
        }
        for number, index in enumerate(selected, start=1)
    ]
    result = {
        "说明": "仅使用代码分层抽样，不包含自动审核结论。",
        "随机种子": args.seed,
        "排除历史索引数": len(excluded_indexes),
        "排除历史样本指纹数": len(excluded_fingerprints),
        "抽查条数": len(samples),
        "相同骨架最大抽取数": args.max_per_skeleton,
        "样本": samples,
    }
    args.output.expanduser().resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "count": len(samples),
                "excluded": len(excluded_fingerprints) or len(excluded_indexes),
                "unique_skeletons": len(skeleton_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
