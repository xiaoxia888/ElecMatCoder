#!/usr/bin/env python3
"""Generate a standalone SWEEPOLET contrast dataset for type extraction."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类" / "管件.json"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE.parent
DATASET_NAME = "管件_SWEEPOLET专项对比增强"
SOURCE_LABEL = "数据增强-SWEEPOLET专项对比"
SWEEPOLET_RE = re.compile(r"\bSWEEP\s*OLET\b", re.IGNORECASE)


@dataclass(frozen=True)
class Signal:
    text: str
    body: str
    pattern: str
    conn: tuple[str, ...] = ()


POSITIVE_SIGNALS = (
    Signal("SWEEPOLET", "SWEEPOLET", "SWEEPOLET标准写法"),
    Signal("Sweepolet", "SWEEPOLET", "SWEEPOLET大小写"),
    Signal("SWEEP OLET", "SWEEPOLET", "SWEEP-OLET分词"),
    Signal("Sweep Olet", "SWEEPOLET", "SWEEP-OLET大小写"),
    Signal("SWEEPOLET B.W.", "SWEEPOLET", "SWEEPOLET优先于B.W."),
    Signal("B.W. SWEEPOLET", "SWEEPOLET", "SWEEPOLET优先于前置B.W."),
    Signal("SWEEP OLET BW", "SWEEPOLET", "SWEEP-OLET优先于BW"),
    Signal("SWEEPOLET BUTT WELD", "SWEEPOLET", "SWEEPOLET优先于BUTT-WELD"),
)

CONTRAST_SIGNALS = (
    Signal("WELDOLET", "对焊支管台", "对比-WELDOLET"),
    Signal("WELD OLET", "对焊支管台", "对比-WELD-OLET"),
    Signal("OLET B.W.", "对焊支管台", "对比-裸OLET-B.W."),
    Signal("OLET BUTT WELD", "对焊支管台", "对比-裸OLET-BUTT-WELD"),
    Signal("SOCKOLET", "承插焊支管台", "对比-SOCKOLET", ("SW",)),
    Signal("SOCKET OLET", "承插焊支管台", "对比-SOCKET-OLET", ("SW",)),
    Signal("OLET S.W.", "承插焊支管台", "对比-裸OLET-S.W.", ("SW",)),
    Signal("THREDOLET", "螺纹支管台", "对比-THREDOLET"),
    Signal("OLET FNPT", "螺纹支管台", "对比-裸OLET-FNPT", ("FNPT",)),
    Signal("LATROLET", "斜支管台", "对比-LATROLET"),
    Signal("OLET", "支管台", "对比-裸OLET"),
)

CONTEXTS = (
    '{signal}, 18"x4", SCH40xSCH80, ASTM A105, MSS SP-97',
    '22"x8" C.S. ASTM A234 WPB {signal} ASME B16.9 SCH. STD x SCH. 80',
    "NAME:{signal};DN650x200;SCH80xSCH40;A350 LF2;MSS SP-97",
    "支管件 | {signal} | DN750x250 | STDxXS | S30408 | GB/T 19326",
    '{signal} MSS SP-97 A182 F316L BW STDxSCH80 26"x6"',
    '30"x10" C.S. ASTM A105 {signal} ASME B16.9 B.W. SCH. XS x SCH. 80',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 SWEEPOLET 种类专项对比增强训练集")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def make_output(signal: Signal) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": signal.body,
            "GEOMETRY": {"ANGLE": "", "RADIUS": ""},
            "FLANGE_STYLE": "",
            "MANU": [],
            "CONN": list(signal.conn),
        },
    }


def make_rows(
    signals: tuple[Signal, ...],
    context_count: int,
    context_offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_index, signal in enumerate(signals):
        for repeat in range(context_count):
            context = CONTEXTS[(signal_index + repeat + context_offset) % len(CONTEXTS)]
            rows.append(
                {
                    "input": context.format(signal=signal.text),
                    "output": make_output(signal),
                    "来源": SOURCE_LABEL,
                    "增强模式": signal.pattern,
                }
            )
    return rows


def description_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def load_inputs(path: Path) -> tuple[set[str], int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    inputs = {
        description_key(row.get("input", ""))
        for row in rows
        if isinstance(row, dict) and str(row.get("input") or "").strip()
    }
    sweep_rows = sum(
        1
        for row in rows
        if isinstance(row, dict) and SWEEPOLET_RE.search(str(row.get("input") or ""))
    )
    return inputs, sweep_rows


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for row in rows:
        key = description_key(row["input"])
        signature = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            if seen[key] != signature:
                raise ValueError(f"同一描述存在冲突标签: {row['input']}")
            duplicate_count += 1
            continue
        seen[key] = signature
        result.append(row)
    return result, duplicate_count


def validate(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        text = str(row.get("input") or "")
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        type_value = output.get("TYPE") if isinstance(output.get("TYPE"), dict) else {}
        body = str(type_value.get("BODY") or "")
        has_sweepolet = bool(SWEEPOLET_RE.search(text))
        if output.get("CATEGORY") != "管件" or not body:
            raise ValueError(f"row={index}: 输出结构不完整")
        if has_sweepolet and body != "SWEEPOLET":
            raise ValueError(f"row={index}: SWEEPOLET 标签错误: {body} | {text}")
        if not has_sweepolet and body == "SWEEPOLET":
            raise ValueError(f"row={index}: 对比样本误标为 SWEEPOLET: {text}")


def main() -> int:
    args = parse_args()
    source_path = args.source_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = make_rows(POSITIVE_SIGNALS, context_count=6, context_offset=0)
    rows.extend(make_rows(CONTRAST_SIGNALS, context_count=4, context_offset=2))
    rows, duplicates_removed = deduplicate(rows)
    validate(rows)

    existing_inputs, source_sweepolet_rows = load_inputs(source_path)
    overlap_count = sum(description_key(row["input"]) in existing_inputs for row in rows)
    random.Random(args.seed).shuffle(rows)

    output_path = output_dir / f"{DATASET_NAME}.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "source_dataset": str(source_path),
        "output": str(output_path),
        "seed": args.seed,
        "rows": len(rows),
        "positive_rows": sum(row["output"]["TYPE"]["BODY"] == "SWEEPOLET" for row in rows),
        "contrast_rows": sum(row["output"]["TYPE"]["BODY"] != "SWEEPOLET" for row in rows),
        "source_sweepolet_rows": source_sweepolet_rows,
        "duplicates_removed": duplicates_removed,
        "existing_input_overlaps": overlap_count,
        "body_counts": dict(Counter(row["output"]["TYPE"]["BODY"] for row in rows)),
        "pattern_counts": dict(Counter(row["增强模式"] for row in rows)),
        "validation_errors": 0,
    }
    report_path = output_dir / f"{DATASET_NAME}_报告.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
