# -*- coding: utf-8 -*-
"""Build standard code map config from material-standard dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


PREFIX_RE = re.compile(r"([A-Z]+)")


def load_counter(dataset_path: Path) -> Counter[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    counter: Counter[str] = Counter()
    for item in data:
        for std in item.get("output", {}).get("STANDARD", []) or []:
            body = (std or {}).get("BODY", "").strip()
            if body:
                counter[body] += 1
    return counter


def build_mapping(counter: Counter[str]) -> dict:
    grouped: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for code, count in counter.most_common():
        m = PREFIX_RE.match(code)
        prefix = m.group(1) if m else "OTHER"
        grouped[prefix].append({"code": code, "count": count})

    return {
        "meta": {
            "total_unique_standards": len(counter),
            "group_count": len(grouped),
        },
        "standard_groups": dict(sorted(grouped.items(), key=lambda x: x[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard code map config")
    parser.add_argument(
        "--dataset",
        default="apps/trainer/qwen3_fte/output/按8类拆分数据集/output.json",
        help="Path to material-standard dataset json",
    )
    parser.add_argument(
        "--output",
        default="src/material_description_splitter/config/standard_code_map.yaml",
        help="Output yaml path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter = load_counter(dataset_path)
    mapping = build_mapping(counter)
    output_path.write_text(
        yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(f"dataset: {dataset_path}")
    print(f"unique standards: {len(counter)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
