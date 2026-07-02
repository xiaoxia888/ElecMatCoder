#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类.json"
DEFAULT_ANCHOR_MAP = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/type_headword_anchor_map.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析种类训练集中的主词歧义情况")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--anchor-map", type=Path, default=DEFAULT_ANCHOR_MAP)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def compile_patterns(anchor_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compiled = []
    for item in anchor_map:
        compiled.append(
            {
                "name": item["name"],
                "expected_category": item.get("expected_category", ""),
                "expected_body": item.get("expected_body", ""),
                "patterns": [re.compile(p, re.I) for p in item.get("patterns", [])],
            }
        )
    return compiled


def match_anchor(text: str, compiled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for item in compiled:
        if any(p.search(text) for p in item["patterns"]):
            hits.append(item)
    return hits


def main() -> None:
    args = parse_args()
    rows = load_json(args.dataset)
    anchor_map = load_json(args.anchor_map)
    compiled = compile_patterns(anchor_map)

    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "total_rows": len(rows),
        "anchors": {},
        "ambiguous_anchors": [],
    }

    anchor_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "category_counter": Counter(),
            "body_counter": Counter(),
            "examples": [],
            "expected_category": "",
            "expected_body": "",
        }
    )

    for row in rows:
        text = row.get("input", "")
        output = row.get("output", {}) or {}
        category = output.get("CATEGORY", "")
        body = (output.get("TYPE", {}) or {}).get("BODY", "")
        for hit in match_anchor(text, compiled):
            stat = anchor_stats[hit["name"]]
            stat["count"] += 1
            stat["expected_category"] = hit["expected_category"]
            stat["expected_body"] = hit["expected_body"]
            stat["category_counter"][category] += 1
            stat["body_counter"][body] += 1
            if len(stat["examples"]) < 8:
                stat["examples"].append(
                    {
                        "input": text,
                        "category": category,
                        "body": body,
                    }
                )

    for name, stat in anchor_stats.items():
        entry = {
            "count": stat["count"],
            "expected_category": stat["expected_category"],
            "expected_body": stat["expected_body"],
            "category_counter": dict(stat["category_counter"]),
            "body_counter": dict(stat["body_counter"]),
            "examples": stat["examples"],
        }
        report["anchors"][name] = entry
        if len(stat["body_counter"]) > 1 or len(stat["category_counter"]) > 1:
            report["ambiguous_anchors"].append(
                {
                    "name": name,
                    "expected_category": stat["expected_category"],
                    "expected_body": stat["expected_body"],
                    "count": stat["count"],
                    "category_counter": dict(stat["category_counter"]),
                    "body_counter": dict(stat["body_counter"]),
                }
            )

    report["ambiguous_anchors"].sort(key=lambda x: (-x["count"], x["name"]))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"已生成: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
