#!/usr/bin/env python3
"""Freeze reviewed material proposals for reproducible dataset generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from apps.trainer.qwen3_fte.src.convert_material_to_part_structure import (
    DEFAULT_CONFIRMED_PROPOSALS,
    DEFAULT_TRAIN_SOURCE,
    DEFAULT_VAL_SOURCE,
    resolve_material_relation,
    validate_material_items,
    write_json,
)
from apps.trainer.qwen3_fte.src.propose_unresolved_material_aliases import (
    DEFAULT_PROPOSALS,
)


DEFAULT_CONFIDENCES = ("可自动确认", "建议确认")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIRMED_PROPOSALS)
    parser.add_argument(
        "--confidence",
        action="append",
        dest="confidences",
        help="需要固化的确认等级，可重复传入；默认包含可自动确认和建议确认",
    )
    return parser.parse_args()


def validate_confirmed_row(
    row: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
) -> None:
    split = row.get("split")
    source_index = row.get("source_index")
    if split not in sources or not isinstance(source_index, int):
        raise ValueError(f"建议缺少有效 split/source_index: {row}")
    if not 0 <= source_index < len(sources[split]):
        raise ValueError(f"建议索引越界: {split}[{source_index}]")

    source_row = sources[split][source_index]
    if source_row.get("input") != row.get("input"):
        raise ValueError(f"建议原文快照已失效: {split}[{source_index}]")
    if (
        source_row.get("output", {}).get("MATERIAL", [])
        != row.get("current_source_material")
    ):
        raise ValueError(f"建议材质快照已失效: {split}[{source_index}]")

    proposal = row.get("proposal")
    if not isinstance(proposal, dict):
        raise ValueError(f"建议缺少 proposal: {split}[{source_index}]")
    material = proposal.get("MATERIAL")
    if not isinstance(material, list) or not material:
        raise ValueError(f"已确认建议没有 MATERIAL: {split}[{source_index}]")
    errors = validate_material_items(material)
    _, relation_errors = resolve_material_relation(
        material,
        str(proposal.get("MATERIAL_RELATION", "")),
    )
    errors.extend(relation_errors)
    if errors:
        raise ValueError(f"建议结构无效 {split}[{source_index}]: {errors}")


def main() -> int:
    args = parse_args()
    confidences = tuple(args.confidences or DEFAULT_CONFIDENCES)
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    sources = {
        "train": json.loads(args.train_source.read_text(encoding="utf-8")),
        "val": json.loads(args.val_source.read_text(encoding="utf-8")),
    }

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in proposals:
        proposal = row.get("proposal", {})
        if proposal.get("confidence") not in confidences:
            continue
        validate_confirmed_row(row, sources)
        key = (row["split"], row["source_index"])
        if key in seen:
            raise ValueError(f"确认建议包含重复索引: {key}")
        seen.add(key)
        selected.append(row)

    payload = {
        "source": str(args.proposals),
        "confirmed_confidences": list(confidences),
        "source_files_modified": False,
        "statistics": {
            "rows": len(selected),
            "split_counts": dict(Counter(row["split"] for row in selected)),
            "confidence_counts": dict(
                Counter(row["proposal"]["confidence"] for row in selected)
            ),
        },
        "rows": selected,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                **payload["statistics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
