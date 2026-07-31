#!/usr/bin/env python3
"""Delete source rows whose material proposal remains unresolvable."""

from __future__ import annotations

import argparse
import bisect
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from apps.trainer.qwen3_fte.src.convert_material_to_part_structure import (
    DEFAULT_CONFIRMED_PROPOSALS,
    DEFAULT_TRAIN_SOURCE,
    DEFAULT_VAL_SOURCE,
    OUTPUT_DIR,
    write_json,
)
from apps.trainer.qwen3_fte.src.propose_unresolved_material_aliases import (
    DEFAULT_PROPOSALS,
)


UNRESOLVED_CONFIDENCE = "无法确认"
DEFAULT_REPORT = OUTPUT_DIR / "材质规范_无法确认记录删除报告.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument(
        "--confirmed-proposals",
        type=Path,
        default=DEFAULT_CONFIRMED_PROPOSALS,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际写回源数据和已确认建议清单；默认只进行校验预览",
    )
    return parser.parse_args()


def source_material(row: dict[str, Any]) -> Any:
    return row.get("output", {}).get("MATERIAL", [])


def validate_source_snapshot(
    proposal_row: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
) -> tuple[str, int]:
    split = proposal_row.get("split")
    source_index = proposal_row.get("source_index")
    if split not in sources or not isinstance(source_index, int):
        raise ValueError(f"无法确认记录缺少有效 split/source_index: {proposal_row}")
    if not 0 <= source_index < len(sources[split]):
        raise ValueError(f"无法确认记录索引越界: {split}[{source_index}]")

    source_row = sources[split][source_index]
    if source_row.get("input") != proposal_row.get("input"):
        raise ValueError(f"无法确认记录原文快照已失效: {split}[{source_index}]")
    if source_material(source_row) != proposal_row.get("current_source_material"):
        raise ValueError(f"无法确认记录材质快照已失效: {split}[{source_index}]")
    return split, source_index


def validate_confirmed_snapshot(
    confirmed_row: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
) -> tuple[str, int]:
    split = confirmed_row.get("split")
    source_index = confirmed_row.get("source_index")
    if split not in sources or not isinstance(source_index, int):
        raise ValueError(f"已确认记录缺少有效 split/source_index: {confirmed_row}")
    if not 0 <= source_index < len(sources[split]):
        raise ValueError(f"已确认记录索引越界: {split}[{source_index}]")

    source_row = sources[split][source_index]
    if source_row.get("input") != confirmed_row.get("input"):
        raise ValueError(f"已确认记录原文快照已失效: {split}[{source_index}]")
    if source_material(source_row) != confirmed_row.get("current_source_material"):
        raise ValueError(f"已确认记录材质快照已失效: {split}[{source_index}]")
    return split, source_index


def remap_confirmed_rows(
    payload: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
    deleted_indices: dict[str, list[int]],
) -> dict[str, Any]:
    remapped = copy.deepcopy(payload)
    for row in remapped.get("rows", []):
        split, old_index = validate_confirmed_snapshot(row, sources)
        if old_index in deleted_indices[split]:
            raise ValueError(f"已确认记录与待删除记录冲突: {split}[{old_index}]")
        row["source_index"] = old_index - bisect.bisect_left(
            deleted_indices[split],
            old_index,
        )
    remapped["source_rows_deleted"] = {
        split: len(indices) for split, indices in deleted_indices.items()
    }
    remapped["source_indices_remapped"] = True
    return remapped


def validate_remapped_confirmed_rows(
    payload: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
) -> None:
    for row in payload.get("rows", []):
        validate_confirmed_snapshot(row, sources)


def main() -> int:
    args = parse_args()
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    sources = {
        "train": json.loads(args.train_source.read_text(encoding="utf-8")),
        "val": json.loads(args.val_source.read_text(encoding="utf-8")),
    }
    confirmed_payload = json.loads(
        args.confirmed_proposals.read_text(encoding="utf-8")
    )

    selected = [
        row
        for row in proposals
        if row.get("proposal", {}).get("confidence") == UNRESOLVED_CONFIDENCE
    ]
    delete_keys: set[tuple[str, int]] = set()
    deleted_indices = {"train": [], "val": []}
    for row in selected:
        key = validate_source_snapshot(row, sources)
        if key in delete_keys:
            raise ValueError(f"无法确认记录包含重复索引: {key}")
        delete_keys.add(key)
        deleted_indices[key[0]].append(key[1])
    for indices in deleted_indices.values():
        indices.sort()

    remapped_confirmed = remap_confirmed_rows(
        confirmed_payload,
        sources,
        deleted_indices,
    )
    cleaned_sources = {
        split: [
            row
            for index, row in enumerate(rows)
            if (split, index) not in delete_keys
        ]
        for split, rows in sources.items()
    }
    validate_remapped_confirmed_rows(remapped_confirmed, cleaned_sources)

    report = {
        "executed": args.execute,
        "selection": {
            "proposal_confidence": UNRESOLVED_CONFIDENCE,
            "proposal_source": str(args.proposals),
        },
        "statistics": {
            "deleted_rows": len(selected),
            "deleted_split_counts": dict(Counter(row["split"] for row in selected)),
            "source_rows_before": {
                split: len(rows) for split, rows in sources.items()
            },
            "source_rows_after": {
                split: len(rows) for split, rows in cleaned_sources.items()
            },
            "confirmed_rows_remapped": len(remapped_confirmed.get("rows", [])),
        },
        "deleted_rows": [
            {
                "split": row["split"],
                "source_index": row["source_index"],
                "input": row["input"],
                "current_source_material": row.get("current_source_material", []),
                "rule": row.get("proposal", {}).get("rule", ""),
            }
            for row in selected
        ],
    }

    if args.execute:
        write_json(args.train_source, cleaned_sources["train"])
        write_json(args.val_source, cleaned_sources["val"])
        write_json(args.confirmed_proposals, remapped_confirmed)
    write_json(args.report, report)
    print(json.dumps(report["statistics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
