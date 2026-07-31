#!/usr/bin/env python3
"""Sample structured-material rows for a non-repeating manual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} 顶层必须是对象数组")
    return rows


def output_signature(row: dict[str, Any]) -> str:
    payload = json.dumps(
        row.get("output", {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成不重复的结构化材质抽查样本")
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--exclude-sample", action="append", default=[], type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count 必须大于 0")

    excluded_inputs: set[str] = set()
    seen_signatures: Counter[str] = Counter()
    for path in args.exclude_sample:
        if not path.exists():
            continue
        for row in load_rows(path):
            input_text = str(row.get("input", "")).strip()
            if input_text:
                excluded_inputs.add(input_text)
            seen_signatures[output_signature(row)] += 1

    candidates: list[dict[str, Any]] = []
    candidate_inputs: set[str] = set()
    for dataset_path in args.dataset:
        split = "val" if "_val" in dataset_path.stem else "train"
        for source_index, row in enumerate(load_rows(dataset_path)):
            input_text = str(row.get("input", "")).strip()
            if (
                not input_text
                or input_text in excluded_inputs
                or input_text in candidate_inputs
            ):
                continue
            candidate_inputs.add(input_text)
            candidates.append(
                {
                    "source_split": split,
                    "source_index": source_index,
                    "input": input_text,
                    "output": row.get("output", {}),
                    **(
                        {"来源": row["来源"]}
                        if isinstance(row.get("来源"), str)
                        else {}
                    ),
                }
            )

    if len(candidates) < args.count:
        raise ValueError(
            f"排除历史样本后仅剩 {len(candidates)} 条，无法抽取 {args.count} 条"
        )

    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[output_signature(row)].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    signatures = list(buckets)
    rng.shuffle(signatures)
    signatures.sort(
        key=lambda signature: (
            seen_signatures[signature] > 0,
            seen_signatures[signature],
        )
    )

    sampled: list[dict[str, Any]] = []
    while len(sampled) < args.count:
        added = False
        for signature in signatures:
            if buckets[signature]:
                sampled.append(buckets[signature].pop())
                added = True
                if len(sampled) == args.count:
                    break
        if not added:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sampled, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(args.output.resolve()),
        "sampled_rows": len(sampled),
        "excluded_unique_inputs": len(excluded_inputs),
        "candidate_rows": len(candidates),
        "unique_output_signatures": len(
            {output_signature(row) for row in sampled}
        ),
        "previously_unseen_output_signatures": sum(
            seen_signatures[output_signature(row)] == 0 for row in sampled
        ),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
