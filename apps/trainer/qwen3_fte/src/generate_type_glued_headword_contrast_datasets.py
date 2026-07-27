#!/usr/bin/env python3
"""Generate category-separated contrast datasets for glued type headwords."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类"
SOURCE_LABEL = "数据增强-制造工艺与主体词粘连专项"
DATASET_BASENAME = "制造工艺与主体词粘连专项对比增强"


@dataclass(frozen=True)
class FittingSpec:
    headword: str
    body: str
    family: str
    angle: str = ""
    radius: str = ""
    conn: tuple[str, ...] = ()
    allow_welded_variants: bool = False


FITTING_SPECS = (
    FittingSpec("Tee", "三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("EqualTee", "等径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("StraightTee", "等径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("Red.Tee", "异径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("ReducingTee", "异径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("LateralTee", "斜三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("EqualLateralTee", "等径斜三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("ReducingLateralTee", "异径斜三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("Y-Tee", "Y型三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("Y-EqualTee", "Y型等径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("Y-ReducingTee", "Y型异径三通", "butt_weld", allow_welded_variants=True),
    FittingSpec("Elbow", "弯头", "butt_weld", allow_welded_variants=True),
    FittingSpec("90Elbow", "弯头", "butt_weld", angle="90", allow_welded_variants=True),
    FittingSpec("45Elbow", "弯头", "butt_weld", angle="45", allow_welded_variants=True),
    FittingSpec("Con.Reducer", "同心异径管", "butt_weld", allow_welded_variants=True),
    FittingSpec("Ecc.Reducer", "偏心异径管", "butt_weld", allow_welded_variants=True),
    FittingSpec("Reducer", "异径管", "butt_weld", allow_welded_variants=True),
    FittingSpec("Cap", "管帽", "butt_weld", allow_welded_variants=True),
    FittingSpec("Olet", "支管台", "branch"),
    FittingSpec("Weldolet", "对焊支管台", "branch"),
    FittingSpec("Sockolet", "承插焊支管台", "branch", conn=("SW",)),
    FittingSpec("Nipple", "短节", "forged"),
    FittingSpec("HalfCoupling", "半管接头", "forged"),
    FittingSpec("Coupling", "管箍", "forged"),
)

FLANGE_SPECS = (
    ("WeldNeckFlange", "带颈对焊法兰"),
    ("WNFlange", "带颈对焊法兰"),
    ("BlindFlange", "盲板法兰"),
    ("BLFlange", "盲板法兰"),
    ("SocketWeldFlange", "承插焊法兰"),
    ("SWFlange", "承插焊法兰"),
    ("ThreadedFlange", "螺纹法兰"),
    ("THDFlange", "螺纹法兰"),
    ("LapJointFlange", "松套法兰"),
    ("LJFlange", "松套法兰"),
    ("PlateFlange", "板式平焊法兰"),
    ("OrificeFlange", "孔板法兰"),
)

PIPE_MANU = {
    "SMLS": "SMLS",
    "SEAMLESS": "SMLS",
    "WELDED": "WELDED",
    "ERW": "ERW",
    "EFW": "EFW",
    "HFW": "HFW",
    "SAW": "SAW",
    "DSAW": "DSAW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成按直管、法兰、管件拆分的种类主体词粘连专项对比训练集",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def normalize_description(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def split_camel_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)


def load_existing_inputs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_description(row["input"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("input"), str)
    }


def make_row(description: str, output: dict[str, Any], pattern: str) -> dict[str, Any]:
    return {
        "input": description.strip(),
        "output": output,
        "来源": SOURCE_LABEL,
        "增强模式": pattern,
    }


def pipe_output(manu: str, flange_style: str = "") -> dict[str, Any]:
    return {
        "CATEGORY": "直管",
        "TYPE": {
            "BODY": "直管",
            "FLANGE_STYLE": flange_style,
            "MANU": [manu],
        },
    }


def flange_output(body: str) -> dict[str, Any]:
    return {
        "CATEGORY": "法兰",
        "TYPE": {
            "BODY": body,
            "CONN": [],
            "SEAL": ["RF"],
        },
    }


def fitting_output(
    spec: FittingSpec,
    manu: str,
    *,
    flange_style: str = "",
) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": spec.body,
            "GEOMETRY": {
                "ANGLE": spec.angle,
                "RADIUS": spec.radius,
            },
            "FLANGE_STYLE": flange_style,
            "MANU": [manu],
            "CONN": list(spec.conn),
        },
    }


def fitting_context(token: str, family: str, index: int) -> str:
    if family == "butt_weld":
        templates = (
            "GB/T 14976;SH/T 3408 {token} 33.4x2.77BW S31603",
            "GB/T 8163;GB/T 12459 {token} DN100 SCH40 BW 20",
            "ASTM A403 WP316L;ASME B16.9 {token} 4in SCH10S BW",
            "NAME:{token};DN80;SCH40S;BW;S30403;GB/T 13401",
        )
    elif family == "branch":
        templates = (
            "MSS SP-97 {token} DN100x25 CL3000 ASTM A105",
            "NAME:{token};DN80x20;MSS SP-97;A105",
            "{token} 4x1in CL3000 BW ASTM A105",
            "SH/T 3410;{token};DN50x15;20",
        )
    else:
        templates = (
            "ASME B16.11 {token} DN25 CL3000 ASTM A105",
            "NAME:{token};DN40;CL3000;NB/T 47008 20",
            "{token} 1in SCH80 ASTM A182 F304",
            "SH/T 3410;{token};DN20;S30408",
        )
    return templates[index % len(templates)].format(token=token)


def build_pipe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = (
        "GB/T 14976 {token} 33.4x2.77 S31603",
        "{token};DN100;SCH40;GB/T 8163;20",
        "ASTM A312 TP316L {token} 4in SCH10S",
        "NAME:{token};OD60.3x3.91;S30403;HG/T 20553",
    )
    for index, (source_manu, normalized_manu) in enumerate(PIPE_MANU.items()):
        variants = (
            (f"{source_manu}PIPE", "前缀粘连", 0),
            (f"{source_manu} PIPE", "正常空格对照", 0),
            (f"{source_manu}Pipe", "前缀粘连-大小写", 1),
            (f"{source_manu}TUBE", "前缀粘连-TUBE", 2),
            (f"PIPE{source_manu}", "后缀粘连", 3),
        )
        for token, pattern, context_offset in variants:
            description = templates[(index + context_offset) % len(templates)].format(token=token)
            rows.append(make_row(description, pipe_output(normalized_manu), pattern))

    flanged_variants = (
        ("SMLSFlangedPipe", "前缀与法兰管主体粘连"),
        ("SMLSFLANGEDPIPE", "前缀与法兰管主体粘连-大写"),
        ("SMLS Flanged Pipe", "法兰管正常空格对照"),
        ("ERWFlangedPipe", "焊接工艺与法兰管主体粘连"),
    )
    for token, pattern in flanged_variants:
        rows.append(
            make_row(
                f"{token};DN100;CL150;RF;ASTM A312 TP304",
                pipe_output("ERW" if token.startswith("ERW") else "SMLS", "FLANGED"),
                pattern,
            )
        )
    return rows


def build_flange_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    templates = (
        "HG/T 20592 {token} DN100 PN16 RF 20",
        "ASME B16.5;{token};4in;CL300;RF;ASTM A105",
        "NAME:{token};DN50;PN25;RF;NB/T 47010 S30408",
        "{token} DN80 CL150 RF ASTM A350 LF2",
    )
    for index, (headword, body) in enumerate(FLANGE_SPECS):
        spaced = split_camel_case(headword)
        variants = (
            (headword, "法兰主体内部粘连", 0),
            (spaced, "法兰主体正常空格对照", 0),
            (headword.upper(), "法兰主体内部粘连-大写", 1),
        )
        for token, pattern, context_offset in variants:
            description = templates[(index + context_offset) % len(templates)].format(token=token)
            rows.append(make_row(description, flange_output(body), pattern))
    return rows


def build_fitting_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(FITTING_SPECS):
        manu_variants = ("SMLS",)
        if spec.allow_welded_variants:
            manu_variants += ("WELDED", "EFW")

        for manu_index, manu in enumerate(manu_variants):
            variants = (
                (f"{manu}{spec.headword}", "前缀粘连", 0),
                (f"{manu} {spec.headword}", "正常空格对照", 0),
                (f"{manu.title()}{spec.headword}", "前缀粘连-大小写", 1),
                (f"{spec.headword}{manu}", "后缀粘连", 2),
            )
            for token, pattern, context_offset in variants:
                description = fitting_context(
                    token,
                    spec.family,
                    index + manu_index + context_offset,
                )
                rows.append(make_row(description, fitting_output(spec, manu), pattern))

    flanged_specs = (
        FittingSpec("FlangedElbow", "弯头", "butt_weld"),
        FittingSpec("FlangedTee", "三通", "butt_weld"),
        FittingSpec("FlangedReducer", "异径管", "butt_weld"),
    )
    for index, spec in enumerate(flanged_specs):
        for token, pattern in (
            (f"SMLS{spec.headword}", "前缀与法兰管件主体粘连"),
            (f"SMLS {spec.headword}", "法兰管件正常空格对照"),
        ):
            rows.append(
                make_row(
                    fitting_context(token, spec.family, index),
                    fitting_output(spec, "SMLS", flange_style="FLANGED"),
                    pattern,
                )
            )
    return rows


def deduplicate_rows(
    rows: list[dict[str, Any]],
    existing_inputs: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    seen: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    duplicate_count = 0
    overlap_count = 0
    for row in rows:
        normalized = normalize_description(row["input"])
        output_signature = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)
        previous = seen.get(normalized)
        if previous is not None:
            if previous != output_signature:
                raise ValueError(f"同一描述存在冲突标签: {row['input']}")
            duplicate_count += 1
            continue
        seen[normalized] = output_signature
        if normalized in existing_inputs:
            overlap_count += 1
        result.append(row)
    return result, duplicate_count, overlap_count


def validate_rows(rows: list[dict[str, Any]], expected_category: str) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        output = row.get("output")
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            errors.append(f"row={index}: input 为空")
            continue
        if not isinstance(output, dict) or output.get("CATEGORY") != expected_category:
            errors.append(f"row={index}: CATEGORY 不等于 {expected_category}")
            continue
        type_value = output.get("TYPE")
        if not isinstance(type_value, dict) or not str(type_value.get("BODY") or "").strip():
            errors.append(f"row={index}: TYPE.BODY 为空")
    return errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "category_counts": dict(Counter(row["output"]["CATEGORY"] for row in rows)),
        "body_counts": dict(Counter(row["output"]["TYPE"]["BODY"] for row in rows)),
        "pattern_counts": dict(Counter(row["增强模式"] for row in rows)),
        "manu_counts": dict(
            Counter(
                manu
                for row in rows
                for manu in row["output"]["TYPE"].get("MANU", [])
            )
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    builders = {
        "直管": build_pipe_rows,
        "法兰": build_flange_rows,
        "管件": build_fitting_rows,
    }
    report: dict[str, Any] = {
        "seed": args.seed,
        "source_label": SOURCE_LABEL,
        "output_dir": str(output_dir),
        "datasets": {},
    }

    for category, builder in builders.items():
        rows = builder()
        existing_inputs = load_existing_inputs(output_dir / f"{category}.json")
        rows, duplicate_count, overlap_count = deduplicate_rows(rows, existing_inputs)
        errors = validate_rows(rows, category)
        if errors:
            raise ValueError("\n".join(errors[:20]))
        rng.shuffle(rows)

        output_path = output_dir / f"{category}_{DATASET_BASENAME}.json"
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["datasets"][category] = {
            "path": str(output_path),
            "duplicates_removed": duplicate_count,
            "existing_input_overlaps": overlap_count,
            "validation_errors": 0,
            **summarize_rows(rows),
        }

    report_path = output_dir / f"{DATASET_BASENAME}_报告.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
