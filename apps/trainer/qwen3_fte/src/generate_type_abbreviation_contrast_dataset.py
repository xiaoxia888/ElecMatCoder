#!/usr/bin/env python3
"""Generate a standalone fitting-abbreviation contrast dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类" / "管件.json"
DEFAULT_PROJECT_CSV = ROOT / "materials_export0701.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE.parent
DATASET_NAME = "管件_历史简写专项对比增强"
SOURCE_LABEL = "数据增强-历史简写专项对比"


@dataclass(frozen=True)
class AbbreviationSpec:
    name: str
    token: str
    pattern: re.Pattern[str]
    body: str
    angle: str = ""
    radius: str = ""
    manu: tuple[str, ...] = ()
    conn: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()
    synthetic: bool = True


def compile_token(token_pattern: str) -> re.Pattern[str]:
    return re.compile(token_pattern, re.IGNORECASE)


# Long and specific patterns must come first.
SPECS = (
    AbbreviationSpec(
        "WOL-90",
        "WOL-90",
        compile_token(r"(?<![A-Z])WOL\s*-\s*90(?![A-Z])"),
        "对焊支管台",
        angle="90",
        expected_codes=("O",),
    ),
    AbbreviationSpec(
        "WOL-45",
        "WOL-45",
        compile_token(r"(?<![A-Z])WOL\s*-\s*45(?![A-Z])"),
        "对焊支管台",
        angle="45",
        expected_codes=("O",),
    ),
    AbbreviationSpec(
        "WELD-BS90",
        "WELD BS90",
        compile_token(r"(?<![A-Z])WELD\s+BS\s*90(?![A-Z])"),
        "弯头",
        angle="90",
        manu=("WELDED",),
        expected_codes=("90ELW",),
    ),
    AbbreviationSpec(
        "WELD-BS45",
        "WELD BS45",
        compile_token(r"(?<![A-Z])WELD\s+BS\s*45(?![A-Z])"),
        "弯头",
        angle="45",
        manu=("WELDED",),
        expected_codes=("45ELW",),
    ),
    AbbreviationSpec(
        "90E(L)",
        "90E(L)",
        compile_token(r"(?<![A-Z0-9])90E\s*\(\s*L\s*\)(?![A-Z])"),
        "弯头",
        angle="90",
        radius="LR",
        expected_codes=("90EL",),
    ),
    AbbreviationSpec(
        "90E(S)",
        "90E(S)",
        compile_token(r"(?<![A-Z0-9])90E\s*\(\s*S\s*\)(?![A-Z])"),
        "弯头",
        angle="90",
        radius="SR",
        expected_codes=("90ES",),
    ),
    AbbreviationSpec(
        "W90ES",
        "W90ES",
        compile_token(r"(?<![A-Z])W90ES(?![A-Z])"),
        "弯头",
        angle="90",
        radius="SR",
        manu=("WELDED",),
        expected_codes=("90ESW",),
    ),
    AbbreviationSpec(
        "W45ES",
        "W45ES",
        compile_token(r"(?<![A-Z])W45ES(?![A-Z])"),
        "弯头",
        angle="45",
        radius="SR",
        manu=("WELDED",),
        expected_codes=("45ESW",),
    ),
    AbbreviationSpec(
        "W90EL",
        "W90EL",
        compile_token(r"(?<![A-Z])W90EL(?![A-Z])"),
        "弯头",
        angle="90",
        radius="LR",
        manu=("WELDED",),
        expected_codes=("90ELW",),
    ),
    AbbreviationSpec(
        "W45EL",
        "W45EL",
        compile_token(r"(?<![A-Z])W45EL(?![A-Z])"),
        "弯头",
        angle="45",
        radius="LR",
        manu=("WELDED",),
        expected_codes=("45ELW",),
    ),
    AbbreviationSpec(
        "E90SR",
        "E90SR",
        compile_token(r"(?<![A-Z])E90SR(?![A-Z])"),
        "弯头",
        angle="90",
        radius="SR",
        expected_codes=("90ES",),
    ),
    AbbreviationSpec(
        "E45SR",
        "E45SR",
        compile_token(r"(?<![A-Z])E45SR(?![A-Z])"),
        "弯头",
        angle="45",
        radius="SR",
        expected_codes=("45ES",),
    ),
    AbbreviationSpec(
        "E90LR",
        "E90LR",
        compile_token(r"(?<![A-Z])E90LR(?![A-Z])"),
        "弯头",
        angle="90",
        radius="LR",
        expected_codes=("90EL",),
    ),
    AbbreviationSpec(
        "E45LR",
        "E45LR",
        compile_token(r"(?<![A-Z])E45LR(?![A-Z])"),
        "弯头",
        angle="45",
        radius="LR",
        expected_codes=("45EL",),
    ),
    AbbreviationSpec(
        "S90E",
        "S90E",
        compile_token(r"(?<![A-Z])S90E(?![A-Z])"),
        "弯头",
        angle="90",
        conn=("SW",),
        expected_codes=("90ELS",),
    ),
    AbbreviationSpec(
        "W9ES-OCR",
        "W9ES",
        compile_token(r"(?<![A-Z])W9ES(?![A-Z])"),
        "弯头",
        angle="90",
        radius="SR",
        manu=("WELDED",),
        expected_codes=("90ESW",),
        synthetic=False,
    ),
    AbbreviationSpec(
        "WRE",
        "WRE",
        compile_token(r"(?<![A-Z])WRE(?![A-Z])"),
        "偏心异径管",
        manu=("WELDED",),
        expected_codes=("REW",),
    ),
    AbbreviationSpec(
        "WTR",
        "WTR",
        compile_token(r"(?<![A-Z])WTR(?![A-Z])"),
        "异径三通",
        manu=("WELDED",),
        expected_codes=("RTW",),
    ),
    AbbreviationSpec(
        "BW-OLET",
        "BW Olet",
        compile_token(r"(?<![A-Z])BW\s+OLET(?![A-Z])"),
        "对焊支管台",
        expected_codes=("O",),
    ),
    AbbreviationSpec(
        "RTS",
        "RTS",
        compile_token(r"(?<![A-Z])RTS(?![A-Z])"),
        "异径三通",
        expected_codes=("RT", "RTS"),
    ),
    AbbreviationSpec(
        "WOL",
        "WOL",
        compile_token(r"(?<![A-Z])WOL(?![A-Z])"),
        "对焊支管台",
        expected_codes=("O",),
    ),
    AbbreviationSpec(
        "SOL",
        "SOL",
        compile_token(r"(?<![A-Z])SOL(?![A-Z])"),
        "承插焊支管台",
        conn=("SW",),
        expected_codes=("OS",),
    ),
    AbbreviationSpec(
        "SWEEPOLET",
        "SWEEPOLET",
        compile_token(r"(?<![A-Z])SWEEP\s*OLET(?![A-Z])"),
        "SWEEPOLET",
        expected_codes=("SOL",),
    ),
    AbbreviationSpec(
        "RK",
        "RK",
        compile_token(r"(?<![A-Z])RK(?![A-Z])"),
        "同心异径管",
        expected_codes=("RC",),
    ),
    AbbreviationSpec(
        "RC",
        "RC",
        compile_token(r"(?<![A-Z])RC(?![A-Z])"),
        "同心异径管",
        expected_codes=("RC", "RCW"),
    ),
    AbbreviationSpec(
        "RE",
        "RE",
        compile_token(r"(?<![A-Z])RE(?![A-Z])"),
        "偏心异径管",
        expected_codes=("RE", "REW"),
    ),
    AbbreviationSpec(
        "TS",
        "TS",
        compile_token(r"(?<![A-Z])TS(?![A-Z])"),
        "等径三通",
        expected_codes=("T", "TW"),
    ),
    AbbreviationSpec(
        "有同头",
        "有同头",
        compile_token(r"有同头"),
        "同心异径管",
        manu=("WELDED",),
        expected_codes=("RCW",),
    ),
)

SPECS_BY_NAME = {spec.name: spec for spec in SPECS}

CONTRAST_ONLY_SPECS = (
    AbbreviationSpec(
        "90EL",
        "90EL",
        compile_token(r"(?<![A-Z])90EL(?![A-Z])"),
        "弯头",
        angle="90",
        radius="LR",
        expected_codes=("90EL",),
    ),
    AbbreviationSpec(
        "45EL",
        "45EL",
        compile_token(r"(?<![A-Z])45EL(?![A-Z])"),
        "弯头",
        angle="45",
        radius="LR",
        expected_codes=("45EL",),
    ),
    AbbreviationSpec(
        "90ES",
        "90ES",
        compile_token(r"(?<![A-Z])90ES(?![A-Z])"),
        "弯头",
        angle="90",
        radius="SR",
        expected_codes=("90ES",),
    ),
    AbbreviationSpec(
        "45ES",
        "45ES",
        compile_token(r"(?<![A-Z])45ES(?![A-Z])"),
        "弯头",
        angle="45",
        radius="SR",
        expected_codes=("45ES",),
    ),
)

ALL_SYNTHETIC_SPECS = tuple(spec for spec in SPECS if spec.synthetic) + CONTRAST_ONLY_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成管件历史简写专项对比训练集")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--project-csv", type=Path, default=DEFAULT_PROJECT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-real-per-signal", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def normalize_description(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def normalize_radius_value(value: str) -> str:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?D", text)
    if not match:
        return text
    fraction = (match.group(2) or "").rstrip("0")
    return f"{match.group(1)}{'.' + fraction if fraction else ''}D"


def explicit_radius(description: str) -> str:
    text = unicodedata.normalize("NFKC", str(description or ""))
    # Accept the known OCR confusion l/I -> 1 only inside an explicit R=...D block.
    match = re.search(
        r"(?i)\bR\s*=\s*([1lI])\s*\.\s*(\d+)\s*D\b",
        text,
    )
    if match:
        return normalize_radius_value(f"1.{match.group(2)}D")
    match = re.search(r"(?i)\bR\s*=\s*(\d+(?:\.\d+)?)\s*D\b", text)
    if match:
        return normalize_radius_value(f"{match.group(1)}D")
    return ""


def has_explicit_welded(description: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?<![A-Z])(?:WELD(?:ED)?|ERW|EFW|HFW|SAW|DSAW|LSAW|HSAW)(?![A-Z])|焊接|有缝",
            description,
        )
    )


def has_explicit_seamless(description: str) -> bool:
    return bool(re.search(r"(?i)(?<![A-Z])(?:SMLS|SEAMLESS)(?![A-Z])|无缝", description))


def make_output(
    *,
    body: str,
    angle: str = "",
    radius: str = "",
    manu: tuple[str, ...] = (),
    conn: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": body,
            "GEOMETRY": {
                "ANGLE": angle,
                "RADIUS": radius,
            },
            "FLANGE_STYLE": "",
            "MANU": list(manu),
            "CONN": list(conn),
        },
    }


def resolve_output(
    spec: AbbreviationSpec,
    description: str,
    correct_code: str = "",
) -> dict[str, Any]:
    radius = explicit_radius(description) or spec.radius
    manu = spec.manu

    if correct_code.endswith("W"):
        manu = ("WELDED",)
    elif has_explicit_seamless(description):
        manu = ("SMLS",)
    elif has_explicit_welded(description):
        manu = ("WELDED",)

    # Product words WOL/BW Olet denote the outlet type, not its manufacturing process.
    if spec.body in {"对焊支管台", "承插焊支管台", "SWEEPOLET"}:
        manu = ()

    return make_output(
        body=spec.body,
        angle=spec.angle,
        radius=radius,
        manu=manu,
        conn=spec.conn,
    )


def find_spec(description: str) -> AbbreviationSpec | None:
    for spec in SPECS:
        if spec.pattern.search(description):
            return spec
    return None


def load_project_rows(
    path: Path,
    *,
    max_per_signal: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        machine_headers = next(reader, [])
        display_headers = next(reader, [])
        if not machine_headers or not display_headers:
            raise ValueError(f"项目 CSV 缺少双行表头: {path}")

        index = {name.strip(): position for position, name in enumerate(display_headers)}
        required = ("材料描述", "项目简称", "分类", "是否一致", "C1-名称简写")
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"项目 CSV 缺少列: {missing}")

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scanned = 0
        matched = 0
        skipped_code_conflicts = 0
        for values in reader:
            scanned += 1
            if len(values) < len(display_headers):
                values.extend([""] * (len(display_headers) - len(values)))
            if values[index["分类"]].strip() != "管件":
                continue

            description = values[index["材料描述"]].strip()
            if not description:
                continue
            spec = find_spec(description)
            if spec is None:
                continue

            correct_code = values[index["C1-名称简写"]].strip().upper()
            if correct_code and spec.expected_codes and correct_code not in spec.expected_codes:
                skipped_code_conflicts += 1
                continue

            matched += 1
            consistency = values[index["是否一致"]].strip()
            grouped[spec.name].append(
                {
                    "input": description,
                    "output": resolve_output(spec, description, correct_code),
                    "来源": f"{SOURCE_LABEL}-真实项目",
                    "增强模式": f"真实项目-{spec.name}",
                    "项目": values[index["项目简称"]].strip(),
                    "正确编码": correct_code,
                    "_is_mismatch": "不一致" in consistency,
                }
            )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for signal, rows in grouped.items():
        available_counts[signal] = len(rows)
        mismatch_rows = [row for row in rows if row["_is_mismatch"]]
        other_rows = [row for row in rows if not row["_is_mismatch"]]
        rng.shuffle(mismatch_rows)
        rng.shuffle(other_rows)
        chosen = (mismatch_rows + other_rows)[:max_per_signal]
        selected_counts[signal] = len(chosen)
        for row in chosen:
            row.pop("_is_mismatch", None)
        selected.extend(chosen)

    report = {
        "csv_rows_scanned": scanned,
        "matched_rows_before_cap": matched,
        "skipped_code_conflicts": skipped_code_conflicts,
        "available_counts": dict(sorted(available_counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
    }
    return selected, report


def synthetic_context(spec: AbbreviationSpec, index: int) -> str:
    if spec.body in {"对焊支管台", "承插焊支管台", "SWEEPOLET"}:
        standards = ("GB/T 19326II", "MSS SP-97", "SH/T 3410", "MSS SP-97", "GB/T 19326")
    else:
        standards = ("GB/T 12459II", "GB/T 13401", "SH/T 3408", "ASME B16.9", "EN 10253-4")
    materials = ("S30408", "S31603", "Q235B", "ASTM A234 WPB", "20")
    dns = ("DN50", "DN100", "DN200", "DN400", "DN800")
    thicknesses = ("SCH10S", "SCH40", "STD", "4.5mm", "8mm")
    standard = standards[index % len(standards)]
    material = materials[(index * 2 + 1) % len(materials)]
    size = dns[(index * 3 + 2) % len(dns)]
    thickness = thicknesses[(index * 5 + 3) % len(thicknesses)]
    token = spec.token

    if spec.body == "弯头":
        templates = (
            "{token}, BE, {standard}, {material}, {size}, {thickness}",
            "NAME:{token};{size};{thickness};{material};{standard}",
            "弯头 {standard}{token} Φ219.1X4.00 {material} {size}",
            "{token}弯头（R={radius_hint}）{standard} {size}×{thickness} {material}",
            "{size}-{thickness}-{token}-BW-{standard} {material}",
        )
        radius_hint = "1.0D" if spec.radius == "SR" else "1.5D"
    elif "异径管" in spec.body:
        templates = (
            "{token}, BE, {standard}, 4x3mm, {material}, {size}XDN25",
            "REDUCER({token});{size}xDN25;{thickness};{material};{standard}",
            "{standard}{token} {size}xDN25 {thickness} {material}",
            "{token}{size}xDN25-{thickness}-{material}-{standard}",
            "NAME:{token};{size}xDN25;{material};{standard}",
        )
        radius_hint = ""
    elif "三通" in spec.body:
        templates = (
            "{token}, {standard}, {material}, {size}XDN25, {thickness}",
            "TEE({token});{size}xDN25;{thickness};{material};{standard}",
            "{standard}{token} {size}xDN25 {thickness} {material}",
            "{token}{size}xDN25-{thickness}-{material}-{standard}",
            "NAME:{token};{size}xDN25;{material};{standard}",
        )
        radius_hint = ""
    else:
        templates = (
            "{token}, CL3000, {standard}, {material}, {size}XDN25",
            "支管座 {token}-CL3000-{size}xDN25 {material} {standard}",
            "{standard}{token} {size}xDN25 CL3000 {material}",
            "NAME:{token};{size}xDN25;CL3000;{material};{standard}",
            "{token}{size}xDN25-{material}-{standard}",
        )
        radius_hint = ""

    return templates[index % len(templates)].format(
        token=token,
        standard=standard,
        material=material,
        size=size,
        thickness=thickness,
        radius_hint=radius_hint,
    )


def build_synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(ALL_SYNTHETIC_SPECS):
        for repeat in range(5):
            description = synthetic_context(spec, spec_index * 5 + repeat)
            rows.append(
                {
                    "input": description,
                    "output": resolve_output(spec, description),
                    "来源": f"{SOURCE_LABEL}-组合对比",
                    "增强模式": f"组合对比-{spec.name}",
                    "正确编码": spec.expected_codes[0] if spec.expected_codes else "",
                }
            )
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for row in rows:
        key = normalize_description(row["input"])
        signature = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            if seen[key] != signature:
                raise ValueError(f"同一描述存在冲突标签: {row['input']}")
            duplicate_count += 1
            continue
        seen[key] = signature
        result.append(row)
    return result, duplicate_count


def load_source_signatures(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("input") or "").strip():
            continue
        result[normalize_description(row["input"])] = row
    return result


def validate(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        type_value = output.get("TYPE") if isinstance(output.get("TYPE"), dict) else {}
        geometry = type_value.get("GEOMETRY") if isinstance(type_value.get("GEOMETRY"), dict) else {}
        if output.get("CATEGORY") != "管件":
            raise ValueError(f"row={index}: CATEGORY 必须为管件")
        if not str(type_value.get("BODY") or "").strip():
            raise ValueError(f"row={index}: BODY 不能为空")
        if not isinstance(geometry, dict) or "ANGLE" not in geometry or "RADIUS" not in geometry:
            raise ValueError(f"row={index}: GEOMETRY 结构不完整")
        if not isinstance(type_value.get("MANU"), list) or not isinstance(type_value.get("CONN"), list):
            raise ValueError(f"row={index}: MANU/CONN 必须为数组")

        text = str(row.get("input") or "")
        if re.search(r"(?<![A-Z])SOL(?![A-Z])", text, re.IGNORECASE):
            if type_value.get("BODY") != "承插焊支管台" or "SW" not in type_value.get("CONN", []):
                raise ValueError(f"row={index}: 原文 SOL 必须标注为承插焊支管台+SW")
        if re.search(r"SWEEP\s*OLET", text, re.IGNORECASE):
            if type_value.get("BODY") != "SWEEPOLET":
                raise ValueError(f"row={index}: SWEEPOLET 对比标签错误")


def main() -> int:
    args = parse_args()
    source_path = args.source_dataset.expanduser().resolve()
    project_csv = args.project_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_rows, real_report = load_project_rows(
        project_csv,
        max_per_signal=args.max_real_per_signal,
        seed=args.seed,
    )
    synthetic_rows = build_synthetic_rows()
    rows, duplicates_removed = deduplicate(real_rows + synthetic_rows)
    validate(rows)

    source_rows = load_source_signatures(source_path)
    overlap_count = 0
    conflicting_overlap_count = 0
    conflicting_overlap_details: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        key = normalize_description(row["input"])
        source_row = source_rows.get(key)
        if source_row is None:
            filtered_rows.append(row)
            continue
        overlap_count += 1
        source_signature = json.dumps(
            source_row.get("output"),
            ensure_ascii=False,
            sort_keys=True,
        )
        signature = json.dumps(row["output"], ensure_ascii=False, sort_keys=True)
        if source_signature != signature:
            conflicting_overlap_count += 1
            conflicting_overlap_details.append(
                {
                    "input": row["input"],
                    "source_output": source_row.get("output"),
                    "generated_output": row["output"],
                    "signal": row["增强模式"].split("-", 1)[-1],
                    "correct_code": row.get("正确编码") or "",
                }
            )

    rows = filtered_rows
    random.Random(args.seed).shuffle(rows)
    output_path = output_dir / f"{DATASET_NAME}.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "source_dataset": str(source_path),
        "project_csv": str(project_csv),
        "output": str(output_path),
        "seed": args.seed,
        "max_real_per_signal": args.max_real_per_signal,
        "rows": len(rows),
        "real_project_rows": sum("真实项目" in row["来源"] for row in rows),
        "synthetic_contrast_rows": sum("组合对比" in row["来源"] for row in rows),
        "duplicates_removed": duplicates_removed,
        "existing_input_overlaps_excluded": overlap_count,
        "conflicting_existing_overlaps_excluded": conflicting_overlap_count,
        "conflicting_existing_overlap_details": conflicting_overlap_details,
        "body_counts": dict(Counter(row["output"]["TYPE"]["BODY"] for row in rows)),
        "signal_counts": dict(Counter(row["增强模式"].split("-", 1)[-1] for row in rows)),
        "correct_code_counts": dict(Counter(row.get("正确编码") or "EMPTY" for row in rows)),
        "project_extraction": real_report,
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
