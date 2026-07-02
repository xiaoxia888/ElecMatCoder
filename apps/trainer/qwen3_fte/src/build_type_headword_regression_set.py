#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集"
DEFAULT_DATASET = BASE / "种类.json"
DEFAULT_ANCHOR_MAP = BASE / "type_headword_anchor_map.json"
DEFAULT_AUG = BASE / "管件_主词优先专项增强.json"
DEFAULT_OUTPUT = BASE / "种类_主词专项回归集.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成种类主词专项回归集")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--anchor-map", type=Path, default=DEFAULT_ANCHOR_MAP)
    parser.add_argument("--augmentation", type=Path, default=DEFAULT_AUG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-anchor", type=int, default=40, help="每个主词最多保留多少条原始样本")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_anchor_map(anchor_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for item in anchor_map:
        compiled.append(
            {
                "name": item["name"],
                "expected_body": item.get("expected_body", ""),
                "patterns": [re.compile(p, re.I) for p in item.get("patterns", [])],
            }
        )
    return compiled


def match_anchor(text: str, compiled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for item in compiled:
        if any(pattern.search(text) for pattern in item["patterns"]):
            hits.append(item)
    return hits


def sort_key(row: dict[str, Any]) -> tuple[int, str]:
    output = row.get("output", {}) or {}
    body = (output.get("TYPE", {}) or {}).get("BODY", "")
    text = row.get("input", "")
    # 让“和主词预期完全一致”的样本优先靠前。
    return (0 if body else 1, text)


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset)
    anchor_map = load_json(args.anchor_map)
    augmentation = load_json(args.augmentation) if args.augmentation.exists() else []
    compiled = compile_anchor_map(anchor_map)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        text = row.get("input", "")
        for hit in match_anchor(text, compiled):
            enriched = {
                "input": row["input"],
                "output": row["output"],
                "_anchor": hit["name"],
                "_expected_body": hit["expected_body"],
                "_source": "dataset",
            }
            grouped[hit["name"]].append(enriched)

    for row in augmentation:
        text = row.get("input", "")
        for hit in match_anchor(text, compiled):
            enriched = {
                "input": row["input"],
                "output": row["output"],
                "_anchor": hit["name"],
                "_expected_body": hit["expected_body"],
                "_source": row.get("_source", "augmentation"),
                "_family": row.get("_family", ""),
                "_reason": row.get("_reason", ""),
            }
            grouped[hit["name"]].append(enriched)

    final_rows: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for item in compiled:
        name = item["name"]
        rows = grouped.get(name, [])
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in sorted(rows, key=sort_key):
            key = row["input"].strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        kept = deduped[: args.per_anchor]
        summary[name] = len(kept)
        final_rows.extend(kept)

    args.output.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成: {args.output}")
    print(f"总条数: {len(final_rows)}")
    for name, count in summary.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
