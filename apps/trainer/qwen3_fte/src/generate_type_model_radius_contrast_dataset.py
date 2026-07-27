#!/usr/bin/env python3
"""Generate a standalone Model nD radius contrast dataset for type extraction."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类" / "管件.json"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE.parent
DATASET_NAME = "管件_Model数字D半径专项对比增强"
SOURCE_LABEL = "数据增强-Model数字D半径专项对比"
MODEL_RADIUS_RE = re.compile(
    r"\bMODEL\s*[:：=\-]?\s*(\d+(?:\.\d+)?)\s*D\b",
    re.IGNORECASE,
)

RADII = ("1D", "1.5D", "2.5D", "3D", "4D", "5D", "10D")
ANGLES = ("90", "45")
MANU_VALUES = ("SMLS", "WELDED")

POSITIVE_TEMPLATES = (
    "{angle} Elbow (Model {radius}), {standard}, Type B, BE, {material}, {manu} DN{dn} {thickness} mm",
    "{angle} Elbow, Model {radius}, {standard}, Type B, BW, {material}, {manu}, DN{dn}, SCH {schedule}",
    "Model {radius} {angle} DEG ELBOW | {standard} | {material} | {manu} | DN{dn} | SCH{schedule}",
    "{angle} DEG. ELBOW - MODEL:{radius} - {standard} - {material} - {manu} - DN{dn}",
    "{angle}度弯头（Model {radius}），{standard}，{material}，{manu}，DN{dn}，SCH{schedule}",
    "{angle}度弯头;MODEL-{radius};{standard};{material};{manu};DN{dn};{thickness}mm",
    "ELBOW {angle}° / Model={radius} / {standard} / {material} / {manu} / DN{dn}",
    "{standard};{angle} ELBOW;model {radius};{manu};BE;{material};DN{dn}",
)

NEGATIVE_CASES = (
    ("{angle} Elbow, {standard}, Type B, BE, {material}, {manu} DN{dn} {thickness} mm", "对比-Type B不是半径"),
    ("{angle} DEG ELBOW;TYPE 3;BW;{material};{manu};DN{dn};SCH{schedule}", "对比-Type 3不是半径"),
    ("{angle} Elbow (Model B), {standard}, BE, {material}, {manu}, DN{dn}", "对比-Model B不是半径"),
    ("{angle} Elbow (Model 3), {standard}, BE, {material}, {manu}, DN{dn}", "对比-Model 3缺少D"),
    ("{angle} Elbow, Design A, {standard}, BE, {material}, {manu}, DN{dn}", "对比-Design A不是半径"),
    ("{angle}度弯头;型号3;{standard};{material};{manu};DN{dn};SCH{schedule}", "对比-型号3不是半径"),
)

STANDARDS = ("EN10253-4", "EN10253-2", "ASME B16.9", "GB/T 12459", "SH/T 3408")
MATERIALS = ("X5CrNi18-10", "P235GH", "ASTM A234 WPB", "S31603", "20")
DNS = ("50", "80", "100", "150", "200", "300")
THICKNESSES = ("3.0", "4.0", "4.5", "5.6", "6.3")
SCHEDULES = ("10S", "20", "40", "80", "STD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Model 数字D 弯头半径专项对比增强训练集")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def make_output(angle: str, radius: str, manu: str) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": "弯头",
            "GEOMETRY": {
                "ANGLE": angle,
                "RADIUS": radius,
            },
            "FLANGE_STYLE": "",
            "MANU": [manu],
            "CONN": [],
        },
    }


def context_values(index: int) -> dict[str, str]:
    return {
        "standard": STANDARDS[index % len(STANDARDS)],
        "material": MATERIALS[(index * 2 + 1) % len(MATERIALS)],
        "dn": DNS[(index * 3 + 2) % len(DNS)],
        "thickness": THICKNESSES[(index * 5 + 1) % len(THICKNESSES)],
        "schedule": SCHEDULES[(index * 7 + 3) % len(SCHEDULES)],
    }


def make_row(
    description: str,
    *,
    angle: str,
    radius: str,
    manu: str,
    pattern: str,
) -> dict[str, Any]:
    return {
        "input": description,
        "output": make_output(angle, radius, manu),
        "来源": SOURCE_LABEL,
        "增强模式": pattern,
    }


def build_positive_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0
    for radius_index, radius in enumerate(RADII):
        repeat_count = 4 if radius == "3D" else 2
        for repeat in range(repeat_count):
            for angle_index, angle in enumerate(ANGLES):
                template_index = (
                    radius_index * 3 + repeat * 2 + angle_index
                ) % len(POSITIVE_TEMPLATES)
                manu = MANU_VALUES[(radius_index + repeat + angle_index) % len(MANU_VALUES)]
                values = context_values(row_index)
                description = POSITIVE_TEMPLATES[template_index].format(
                    angle=angle,
                    radius=radius,
                    manu=manu,
                    **values,
                )
                rows.append(
                    make_row(
                        description,
                        angle=angle,
                        radius=radius,
                        manu=manu,
                        pattern=f"正例-Model数字D-{radius}",
                    )
                )
                row_index += 1
    return rows


def build_contrast_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0
    for case_index, (template, pattern) in enumerate(NEGATIVE_CASES):
        for repeat in range(4):
            angle = ANGLES[(case_index + repeat) % len(ANGLES)]
            manu = MANU_VALUES[(case_index + repeat + 1) % len(MANU_VALUES)]
            values = context_values(row_index + 100)
            description = template.format(
                angle=angle,
                manu=manu,
                **values,
            )
            rows.append(
                make_row(
                    description,
                    angle=angle,
                    radius="",
                    manu=manu,
                    pattern=pattern,
                )
            )
            row_index += 1
    return rows


def description_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def load_source_inputs(path: Path) -> tuple[set[str], int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    inputs = {
        description_key(row.get("input", ""))
        for row in rows
        if isinstance(row, dict) and str(row.get("input") or "").strip()
    }
    model_radius_rows = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and MODEL_RADIUS_RE.search(str(row.get("input") or ""))
    )
    return inputs, model_radius_rows


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
        description = str(row.get("input") or "")
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        type_value = output.get("TYPE") if isinstance(output.get("TYPE"), dict) else {}
        geometry = type_value.get("GEOMETRY") if isinstance(type_value.get("GEOMETRY"), dict) else {}
        radius = str(geometry.get("RADIUS") or "")
        match = MODEL_RADIUS_RE.search(description)

        if output.get("CATEGORY") != "管件" or type_value.get("BODY") != "弯头":
            raise ValueError(f"row={index}: 输出结构错误")
        if match:
            expected_radius = f"{match.group(1)}D"
            if radius != expected_radius:
                raise ValueError(
                    f"row={index}: Model数字D标注错误，expected={expected_radius}, actual={radius}",
                )
        elif radius:
            raise ValueError(f"row={index}: 对比样本不应标注半径: {description}")


def main() -> int:
    args = parse_args()
    source_path = args.source_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_positive_rows()
    rows.extend(build_contrast_rows())
    rows, duplicates_removed = deduplicate(rows)
    validate(rows)

    existing_inputs, source_model_radius_rows = load_source_inputs(source_path)
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
        "positive_rows": sum(
            bool(row["output"]["TYPE"]["GEOMETRY"]["RADIUS"])
            for row in rows
        ),
        "contrast_rows": sum(
            not row["output"]["TYPE"]["GEOMETRY"]["RADIUS"]
            for row in rows
        ),
        "source_model_radius_rows": source_model_radius_rows,
        "duplicates_removed": duplicates_removed,
        "existing_input_overlaps": overlap_count,
        "radius_counts": dict(
            Counter(row["output"]["TYPE"]["GEOMETRY"]["RADIUS"] or "EMPTY" for row in rows)
        ),
        "angle_counts": dict(
            Counter(row["output"]["TYPE"]["GEOMETRY"]["ANGLE"] for row in rows)
        ),
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
