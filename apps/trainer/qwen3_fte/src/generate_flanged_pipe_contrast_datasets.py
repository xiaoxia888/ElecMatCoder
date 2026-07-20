#!/usr/bin/env python3
"""Generate category-separated contrast datasets for flanged-pipe recognition."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import openpyxl


SYNTHETIC_SOURCE_LABEL = "数据增强"
SPECIALTY_NAME = "法兰管主体识别专项对比增强"

PIPE_CODES = {"FP", "LFP"}
FLANGE_CODES = {"LF", "LFPL", "FJSO", "FPL"}
FITTING_CODES = {
    "90ELFN",
    "90ELFT",
    "LF45EL",
    "LF90EL",
    "LFRT",
    "LFT",
    "F45EL",
    "F90EL",
    "FRT",
    "FT",
    "FRC",
    "FRE",
    "F45EL10D",
    "F90EL10D",
    "F90EL5D",
    "F45LT",
}


def normalize_description(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def description_skeleton(value: str) -> str:
    value = value.upper()
    value = re.sub(r"(?<![A-Z])\d+(?:\.\d+)?", "#", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def flatten_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(flatten_rows(item))
        return rows
    return [value] if isinstance(value, dict) else []


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return flatten_rows(json.load(handle))


def collect_existing_inputs(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_json_rows(path):
            description = row.get("input")
            if isinstance(description, str) and description.strip():
                result.add(normalize_description(description))
    return result


def pipe_output(style: str, manu: list[str]) -> dict[str, Any]:
    return {
        "CATEGORY": "直管",
        "TYPE": {"BODY": "直管", "FLANGE_STYLE": style, "MANU": manu},
    }


def flange_output(body: str, conn: list[str], seals: list[str]) -> dict[str, Any]:
    return {
        "CATEGORY": "法兰",
        "TYPE": {"BODY": body, "CONN": conn, "SEAL": seals},
    }


def fitting_output(
    body: str,
    flange_style: str,
    manu: list[str],
    angle: str = "",
    radius: str = "",
    conn: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "CATEGORY": "管件",
        "TYPE": {
            "BODY": body,
            "GEOMETRY": {"ANGLE": angle, "RADIUS": radius},
            "FLANGE_STYLE": flange_style,
            "MANU": manu,
            "CONN": conn or [],
        },
    }


def make_row(
    description: str,
    output: dict[str, Any],
    source_label: str = SYNTHETIC_SOURCE_LABEL,
) -> dict[str, Any]:
    return {"input": description.strip(), "output": output, "来源": source_label}


def clean_synthetic_description(description: str) -> str:
    description = re.sub(r";\s*;", ";", description)
    description = re.sub(r",\s*,", ",", description)
    description = re.sub(r"\|\s*\|", "|", description)
    return description.strip(" ,;|")


def detect_manu(description: str) -> list[str]:
    if re.search(r"(?:SMLS|SEAMLESS|无缝)", description, re.IGNORECASE):
        return ["SMLS"]
    if re.search(r"(?:\bWELDED\b|\bERW\b|\bEFW\b|焊接管)", description, re.IGNORECASE):
        return ["WELDED"]
    return []


def detect_conn(description: str) -> list[str]:
    tokens = ["FNPT", "NPTF", "MNPT", "NPT", "FTE", "MTE", "TBE", "TSE", "SCRD", "THD", "SW"]
    upper = description.upper()
    for token in tokens:
        if re.search(rf"(?<![A-Z]){re.escape(token)}(?![A-Z])", upper):
            return [token]
    return []


def detect_seals(description: str) -> list[str]:
    upper = description.upper()
    for token in ["FLRF", "FLRJ", "RTJ", "MFM", "FM", "MF", "LM", "LF", "RJ", "RF", "FF"]:
        if re.search(rf"(?<![A-Z]){re.escape(token)}(?![A-Z])", upper):
            return [token]
    return []


def detect_flange_style(description: str, default: str) -> str:
    if re.search(
        r"LAP\s*JOINT|LJ\s*FLG|LOOSE\s*FLANGE|活套法兰|松套法兰",
        description,
        re.IGNORECASE,
    ):
        return "LAP_JOINT_FLANGED"
    if re.search(r"FIX(?:ED)?\s*FLG|FIX(?:ED)?\s*FLANGE|固定法兰", description, re.IGNORECASE):
        return "固定法兰"
    return default


def detect_radius(description: str) -> str:
    match = re.search(r"(?<![A-Z0-9.])(\d+(?:\.\d+)?)\s*D(?![A-Z0-9])", description.upper())
    return f"{match.group(1)}D" if match else ""


def category_for_code(type_code: str) -> str | None:
    if type_code in PIPE_CODES:
        return "直管"
    if type_code in FLANGE_CODES:
        return "法兰"
    if type_code in FITTING_CODES:
        return "管件"
    return None


def output_for_real_row(type_code: str, description: str) -> dict[str, Any]:
    category = category_for_code(type_code)
    if category == "直管":
        default_style = "LAP_JOINT_FLANGED" if type_code == "LFP" else "FLANGED"
        return pipe_output(detect_flange_style(description, default_style), detect_manu(description))

    if category == "法兰":
        body_by_code = {
            "LF": "松套法兰",
            "LFPL": "板式平焊松套法兰",
            "FJSO": "带颈平焊夹套法兰",
            "FPL": "板式平焊法兰",
        }
        return flange_output(body_by_code[type_code], detect_conn(description), detect_seals(description))

    if category != "管件":
        raise ValueError(f"不支持的标准化种类: {type_code}")

    angle = ""
    radius = ""
    body = ""
    default_style = "FLANGED"
    if type_code in {"90ELFN", "90ELFT"}:
        body, angle, default_style = "弯头", "90", ""
    elif type_code in {"LF45EL", "F45EL"}:
        body, angle = "弯头", "45"
    elif type_code in {"LF90EL", "F90EL"}:
        body, angle = "弯头", "90"
    elif type_code in {"LFRT", "FRT"}:
        body = "异径三通"
    elif type_code in {"LFT", "FT"}:
        body = "等径三通" if re.search(r"等径|STRAIGHT|EQUAL", description, re.IGNORECASE) else "三通"
    elif type_code == "FRC":
        body = "同心异径管"
    elif type_code == "FRE":
        body = "偏心异径管"
    elif type_code in {"F45EL10D", "F90EL10D", "F90EL5D"}:
        body = "BEND"
        angle = "45" if type_code.startswith("F45") else "90"
        radius = detect_radius(description)
        default_style = "固定法兰"
    elif type_code == "F45LT":
        body, angle, default_style = "斜三通", "45", "固定法兰"

    if type_code.startswith("LF"):
        default_style = "LAP_JOINT_FLANGED"
    flange_style = detect_flange_style(description, default_style)
    return fitting_output(
        body,
        flange_style,
        detect_manu(description),
        angle=angle,
        radius=radius,
        conn=detect_conn(description),
    )


def read_real_rows(
    workbook_paths: list[Path],
    existing_inputs: set[str],
    seed: int,
    max_per_skeleton: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[str, dict[str, Any], str]]] = defaultdict(list)
    seen: dict[str, tuple[dict[str, Any], str]] = {}
    workbook_reports: dict[str, Any] = {}
    conflict_rows: list[dict[str, Any]] = []
    overlap_count = 0

    for workbook_path in workbook_paths:
        workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
        worksheet = workbook.active
        values = worksheet.iter_rows(values_only=True)
        headers = next(values)
        index = {str(value): idx for idx, value in enumerate(headers) if value is not None}
        required = {"子表.描述", "子表.标准化种类"}
        missing = required - index.keys()
        if missing:
            raise ValueError(f"{workbook_path} 缺少字段: {sorted(missing)}")

        source_label = f"真实项目数据-{workbook_path.stem}"
        loaded_rows = 0
        supported_rows = 0
        duplicate_rows = 0
        unsupported_codes: Counter[str] = Counter()
        type_distribution: Counter[str] = Counter()
        category_distribution: Counter[str] = Counter()
        category_corrections: Counter[str] = Counter()

        for row in values:
            raw_description = row[index["子表.描述"]]
            if raw_description is None or not str(raw_description).strip():
                continue
            loaded_rows += 1
            description = str(raw_description).strip()
            standardized = str(row[index["子表.标准化种类"]] or "").strip().upper()
            corrected = (
                str(row[index["子表.修正种类"]] or "").strip().upper()
                if "子表.修正种类" in index
                else ""
            )
            type_code = corrected if category_for_code(corrected) else standardized
            category = category_for_code(type_code)
            if category is None:
                unsupported_codes[type_code or "<EMPTY>"] += 1
                continue
            supported_rows += 1
            type_distribution[type_code] += 1
            category_distribution[category] += 1
            excel_category = (
                str(row[index["子表.分类"]] or "").strip()
                if "子表.分类" in index
                else ""
            )
            if excel_category and excel_category != category:
                category_corrections[f"{excel_category}->{category}"] += 1

            normalized = normalize_description(description)
            output = output_for_real_row(type_code, description)
            if normalized in existing_inputs:
                overlap_count += 1
                continue
            if normalized in seen:
                duplicate_rows += 1
                previous_output, previous_source = seen[normalized]
                if previous_output != output:
                    conflict_rows.append(
                        {
                            "input": description,
                            "first_source": previous_source,
                            "first_output": previous_output,
                            "second_source": source_label,
                            "second_output": output,
                        }
                    )
                continue
            seen[normalized] = (output, source_label)
            body = str(output["TYPE"].get("BODY", ""))
            grouped[(category, type_code + ":" + body, description_skeleton(description))].append(
                (description, output, source_label)
            )

        workbook_reports[workbook_path.name] = {
            "loaded_rows": loaded_rows,
            "supported_rows": supported_rows,
            "duplicate_rows_in_or_across_workbooks": duplicate_rows,
            "unsupported_type_codes": dict(sorted(unsupported_codes.items())),
            "type_code_distribution": dict(sorted(type_distribution.items())),
            "derived_category_distribution": dict(sorted(category_distribution.items())),
            "excel_category_corrections": dict(sorted(category_corrections.items())),
        }

    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {"直管": [], "法兰": [], "管件": []}
    selected_group_counts: Counter[str] = Counter()
    for (category, type_body, _), candidates in sorted(grouped.items(), key=lambda item: item[0]):
        rng.shuffle(candidates)
        for description, output, source_label in candidates[:max_per_skeleton]:
            selected[category].append(make_row(description, output, source_label))
            selected_group_counts[f"{category}:{type_body}"] += 1
    for rows in selected.values():
        rng.shuffle(rows)

    report = {
        "workbooks": workbook_reports,
        "overlap_with_existing_training_rows_skipped": overlap_count,
        "conflict_occurrences_ignored": conflict_rows,
        "real_rows_selected": {category: len(rows) for category, rows in selected.items()},
        "selected_type_body_distribution": dict(sorted(selected_group_counts.items())),
        "real_skeleton_groups": len(grouped),
        "max_rows_per_skeleton": max_per_skeleton,
    }
    return selected, report


def pipe_context_rows() -> list[dict[str, str]]:
    sizes = ["DN25", "DN40", "DN50", "DN80", "DN100", "DN150", "DN200", '2"', '4"', '6"']
    materials = [
        "20/PTFE LINED",
        "ASTM A106 Gr.B + PTFE LINED",
        "20/衬胶",
        "06Cr19Ni10/PTFE",
    ]
    ratings = [
        ("PN10", "RF", "HG/T 20592;HG/T 20538"),
        ("PN16", "RF", "HG/T 20592;HG/T 20538"),
        ("CL150", "RF", "SH/T 3406;HG/T 20538"),
        ("CL300", "FF", "ASME B16.5;ASME B36.10M"),
    ]
    thicknesses = ["3.0mm", "4.0mm", "SCH40", "SCH80"]
    rows = []
    for size, material, (pressure, seal, standard), thickness in product(
        sizes, materials, ratings, thicknesses
    ):
        rows.append(
            {
                "size": size,
                "material": material,
                "pressure": pressure,
                "seal": seal,
                "standard": standard,
                "thickness": thickness,
            }
        )
    return rows


def flange_context_rows() -> list[dict[str, str]]:
    sizes = ["DN25", "DN40", "DN50", "DN80", "DN100", "DN150", "DN200", '2"', '4"', '6"']
    ratings = [
        ("PN10", "RF", "HG/T 20592"),
        ("PN16", "RF", "HG/T 20592"),
        ("CL150", "RF", "ASME B16.5"),
        ("CL300", "FF", "ASME B16.5"),
    ]
    return [
        {"size": size, "pressure": pressure, "seal": seal, "standard": standard}
        for size, (pressure, seal, standard) in product(sizes, ratings)
    ]


def fitting_context_rows() -> list[dict[str, str]]:
    sizes = [
        ("DN25", "DN25x15"),
        ("DN40", "DN40x25"),
        ("DN50", "DN50x25"),
        ("DN80", "DN80x50"),
        ("DN100", "DN100x50"),
        ("DN150", "DN150x100"),
        ("DN200", "DN200x100"),
        ('2"', '2"x1"'),
        ('4"', '4"x2"'),
        ('6"', '6"x4"'),
    ]
    materials = [
        ("ASTM A234 Gr.WPB", "SMLS"),
        ("ASTM A403 Gr.WP304L", "SMLS"),
        ("20", "SMLS"),
        ("20/PTFE LINED", ""),
        ("FRP DARAKANE 470", ""),
    ]
    ratings = [
        ("PN10", "RF", "HG/T 20592", "HG/T 20538"),
        ("PN16", "RF", "HG/T 20592", "HG/T 20538"),
        ("CL150", "RF", "ASME B16.5", "ASME B16.9"),
        ("CL300", "FF", "ASME B16.5", "ASME B16.9"),
    ]
    thicknesses = ["3.0mm", "4.0mm", "SCH40", "SCH80"]
    return [
        {
            "single_size": single_size,
            "reducing_size": reducing_size,
            "material": material,
            "manu": manu,
            "pressure": pressure,
            "seal": seal,
            "flange_standard": flange_standard,
            "fitting_standard": fitting_standard,
            "thickness": thickness,
        }
        for (single_size, reducing_size), (material, manu),
        (pressure, seal, flange_standard, fitting_standard), thickness in product(
            sizes, materials, ratings, thicknesses
        )
    ]


def generate_synthetic_pipe_rows(
    existing_inputs: set[str], seed: int, flanged_target: int, lap_joint_target: int
) -> list[dict[str, Any]]:
    flanged_templates = [
        "法兰管, {material}, {seal}, {pressure}, {standard}, SMLS, {size}, {thickness}",
        "{material};{pressure};{seal};{size};法兰管;SMLS;{standard};{thickness}",
        "SMLS {material} {size} {thickness}，管道端部形式：两端法兰，{pressure} {seal}，{standard}",
        "FLANGED PIPE, {material}, SMLS, {pressure}, {seal}, {standard}, {size}, {thickness}",
        "{material} | {size} | {thickness} | PIPE FLGD | SMLS | {pressure} | {seal} | {standard}",
        "1.Name: PIPE 2.Ends: FLANGED 3.Material: {material} 4.Size: {size} {thickness} 5.Rating: {pressure} {seal} 6.Std: {standard} 7.Mfg: SMLS",
        "直管 {size} {thickness} {material}，两端法兰连接，{pressure} {seal}，{standard}，无缝",
        "{pressure};{standard};{material};{size};两端带法兰的直管;{seal};SMLS;{thickness}",
    ]
    lap_templates = [
        "LAP JOINT FLANGED PIPE, {material}, {pressure}, {seal}, {standard}, {size}, {thickness}",
        "PIPE;SMLS;{material};{size};{thickness};LJ FLANGE X FLANGE;{pressure};{seal};{standard}",
        "直管 {size} {thickness} {material}，两端活套法兰，{pressure} {seal}，{standard}，SMLS",
        "{material} | PIPE WITH LOOSE FLANGE ENDS | {size} | {pressure} | {seal} | {standard} | {thickness}",
    ]
    contexts = pipe_context_rows()
    rng = random.Random(seed + 11)

    def build(templates: list[str], style: str, target: int) -> list[dict[str, Any]]:
        candidates = [template.format(**ctx) for ctx in contexts for template in templates]
        rng.shuffle(candidates)
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for description in candidates:
            normalized = normalize_description(description)
            if normalized in seen or normalized in existing_inputs:
                continue
            seen.add(normalized)
            description = clean_synthetic_description(description)
            result.append(make_row(description, pipe_output(style, detect_manu(description))))
            if len(result) >= target:
                break
        if len(result) != target:
            raise RuntimeError(f"{style} 仅生成 {len(result)} 条，目标 {target} 条")
        return result

    rows = build(flanged_templates, "FLANGED", flanged_target)
    rows.extend(build(lap_templates, "LAP_JOINT_FLANGED", lap_joint_target))
    rng.shuffle(rows)
    return rows


def generate_flange_rows(existing_inputs: set[str], seed: int, target: int) -> list[dict[str, Any]]:
    families = [
        ("WN FLANGE", "带颈对焊法兰", []),
        ("WELD NECK FLANGE", "带颈对焊法兰", []),
        ("SO FLANGE", "带颈平焊法兰", []),
        ("SLIP-ON FLANGE", "带颈平焊法兰", []),
        ("SW FLANGE", "承插焊法兰", []),
        ("SOCKET WELD FLANGE", "承插焊法兰", []),
        ("BLIND FLANGE", "盲板法兰", []),
        ("盲法兰", "盲板法兰", []),
        ("THREADED FLANGE FNPT", "螺纹法兰", ["FNPT"]),
        ("衬里法兰", "衬里法兰", []),
        ("LAP JOINT FLANGE", "松套法兰", []),
        ("FLANGE", "法兰", []),
    ]
    base_materials = ["ASTM A105", "ASTM A182 F304L", "ASTM A350 LF2", "20", "06Cr19Ni10"]
    lined_materials = ["20/PTFE LINED", "20/衬胶", "ASTM A105 + PTFE LINED"]
    contexts = flange_context_rows()
    rng = random.Random(seed + 23)
    rng.shuffle(contexts)
    candidates: list[dict[str, Any]] = []
    per_family = (target + len(families) - 1) // len(families)
    for family_idx, (name, body, conn) in enumerate(families):
        generated = 0
        for idx, ctx in enumerate(contexts):
            materials = lined_materials if body == "衬里法兰" else base_materials
            material = materials[(idx + family_idx) % len(materials)]
            data = {**ctx, "name": name, "material": material}
            template_idx = (idx + family_idx) % 4
            templates = [
                "{name};{pressure};{seal};{material};{standard};{size}",
                "{size} | {material} | {pressure} | {name} | {seal} | {standard}",
                "1.材质:{material} 2.规格:{size} {pressure} {seal} 3.名称:{name} 4.标准:{standard}",
                "{material},{standard},{size},{pressure},{seal},NAME:{name}",
            ]
            description = clean_synthetic_description(templates[template_idx].format(**data))
            normalized = normalize_description(description)
            if normalized in existing_inputs:
                continue
            candidates.append(make_row(description, flange_output(body, conn, [ctx["seal"]])))
            generated += 1
            if generated >= per_family:
                break
    rng.shuffle(candidates)
    return candidates[:target]


def generate_fitting_rows(existing_inputs: set[str], seed: int, target: int) -> list[dict[str, Any]]:
    # The last four families are hard negatives: RF or flange standards are mentioned,
    # but the fitting itself is not explicitly described as flanged.
    families = [
        ("法兰弯头", "弯头", "FLANGED", "explicit"),
        ("FLANGED ELBOW", "弯头", "FLANGED", "explicit"),
        ("法兰等径三通", "等径三通", "FLANGED", "explicit"),
        ("FLANGED REDUCING TEE", "异径三通", "FLANGED", "explicit"),
        ("两端法兰同心异径管", "同心异径管", "FLANGED", "explicit"),
        ("FLANGED ECCENTRIC REDUCER", "偏心异径管", "FLANGED", "explicit"),
        ("法兰管帽", "管帽", "FLANGED", "explicit"),
        ("LAP JOINT FLANGED ELBOW", "弯头", "LAP_JOINT_FLANGED", "explicit"),
        ("INSTRUMENT TEE", "仪表三通", "", "matching_flange"),
        ("ELBOW", "弯头", "", "matching_flange"),
        ("REDUCING TEE", "异径三通", "", "matching_flange"),
        ("CONCENTRIC REDUCER", "同心异径管", "", "matching_flange"),
    ]
    contexts = fitting_context_rows()
    rng = random.Random(seed + 37)
    rng.shuffle(contexts)
    candidates: list[dict[str, Any]] = []
    per_family = (target + len(families) - 1) // len(families)
    for family_idx, (name, body, flange_style, evidence) in enumerate(families):
        generated = 0
        for idx, ctx in enumerate(contexts):
            manu_text = ctx["manu"]
            manu = [manu_text] if manu_text else []
            reducing_bodies = {"偏心异径管", "同心异径管", "异径三通"}
            size = ctx["reducing_size"] if body in reducing_bodies else ctx["single_size"]
            if evidence == "matching_flange":
                suffix = (
                    f"管件标准{ctx['fitting_standard']}，配套法兰密封面{ctx['seal']}，"
                    f"法兰标准{ctx['flange_standard']}"
                )
            else:
                suffix = (
                    f"FLANGED ENDS {ctx['seal']};{ctx['flange_standard']};"
                    f"BODY STD {ctx['fitting_standard']}"
                )
            data = {**ctx, "size": size, "name": name, "manu": manu_text, "suffix": suffix}
            templates = [
                "{name};{material};{pressure};{suffix};{manu};{size};{thickness}",
                "{size} | {pressure} | {material} | {name} | {suffix} | {manu} | {thickness}",
                "1.名称:{name} 2.规格:{size} {thickness} 3.材质:{material} 4.压力:{pressure} 5.{suffix} 6.{manu}",
                "{material},{size},{thickness},{pressure},{suffix},{manu},NAME:{name}",
            ]
            description = clean_synthetic_description(
                templates[(idx + family_idx) % len(templates)].format(**data)
            )
            normalized = normalize_description(description)
            if normalized in existing_inputs:
                continue
            candidates.append(make_row(description, fitting_output(body, flange_style, manu)))
            generated += 1
            if generated >= per_family:
                break
    rng.shuffle(candidates)
    return candidates[:target]


def validate_rows(
    category: str,
    rows: list[dict[str, Any]],
    existing_inputs: set[str],
) -> dict[str, Any]:
    normalized = [normalize_description(row["input"]) for row in rows]
    duplicates = len(normalized) - len(set(normalized))
    overlap = len(set(normalized) & existing_inputs)
    wrong_category = sum(row["output"].get("CATEGORY") != category for row in rows)
    invalid_source = sum(
        not (
            str(row.get("来源", "")).startswith("数据增强")
            or str(row.get("来源", "")).startswith("真实项目数据-")
        )
        for row in rows
    )
    if duplicates or overlap or wrong_category or invalid_source:
        raise ValueError(
            f"{category} 校验失败: duplicates={duplicates}, overlap={overlap}, "
            f"wrong_category={wrong_category}, invalid_source={invalid_source}"
        )
    return {
        "rows": len(rows),
        "duplicate_descriptions": duplicates,
        "overlap_with_existing_descriptions": overlap,
        "wrong_category_rows": wrong_category,
        "invalid_source_rows": invalid_source,
        "source_distribution": dict(sorted(Counter(str(row["来源"]) for row in rows).items())),
    }


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: Counter[str] = Counter()
    for row in rows:
        value = row["output"]["TYPE"].get(field)
        if isinstance(value, list):
            values.update(value or ["<EMPTY>"])
        else:
            values[str(value or "<EMPTY>")] += 1
    return dict(sorted(values.items()))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel",
        type=Path,
        action="append",
        help="可重复指定；不指定时默认读取 /Users/guoxi/Downloads/0720.xlsx 和 0721.xlsx",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-real-rows-per-skeleton", type=int, default=24)
    parser.add_argument("--synthetic-flanged-pipe-rows", type=int, default=180)
    parser.add_argument("--synthetic-lap-joint-pipe-rows", type=int, default=60)
    parser.add_argument("--flange-rows", type=int, default=240)
    parser.add_argument("--fitting-rows", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbook_paths = args.excel or [
        Path("/Users/guoxi/Downloads/0720.xlsx"),
        Path("/Users/guoxi/Downloads/0721.xlsx"),
    ]
    source_paths = {
        "直管": args.source_dir / "直管.json",
        "法兰": args.source_dir / "法兰.json",
        "管件": args.source_dir / "管件_清洗后.json",
    }
    all_existing = collect_existing_inputs(list(source_paths.values()))
    real_rows, real_report = read_real_rows(
        workbook_paths,
        all_existing,
        args.seed,
        args.max_real_rows_per_skeleton,
    )
    synthetic_pipe_rows = generate_synthetic_pipe_rows(
        all_existing
        | {
            normalize_description(row["input"])
            for category_rows in real_rows.values()
            for row in category_rows
        },
        args.seed,
        args.synthetic_flanged_pipe_rows,
        args.synthetic_lap_joint_pipe_rows,
    )
    pipe_rows = real_rows["直管"] + synthetic_pipe_rows
    random.Random(args.seed + 41).shuffle(pipe_rows)

    occupied = (
        all_existing
        | {normalize_description(row["input"]) for row in pipe_rows}
        | {normalize_description(row["input"]) for row in real_rows["法兰"]}
        | {normalize_description(row["input"]) for row in real_rows["管件"]}
    )
    synthetic_flange_rows = generate_flange_rows(occupied, args.seed, args.flange_rows)
    flange_rows = real_rows["法兰"] + synthetic_flange_rows
    random.Random(args.seed + 43).shuffle(flange_rows)
    occupied.update(normalize_description(row["input"]) for row in flange_rows)
    synthetic_fitting_rows = generate_fitting_rows(occupied, args.seed, args.fitting_rows)
    fitting_rows = real_rows["管件"] + synthetic_fitting_rows
    random.Random(args.seed + 47).shuffle(fitting_rows)

    output_paths = {
        "直管": args.source_dir / f"直管_{SPECIALTY_NAME}.json",
        "法兰": args.source_dir / f"法兰_{SPECIALTY_NAME}.json",
        "管件": args.source_dir / f"管件_{SPECIALTY_NAME}.json",
    }
    report_path = args.source_dir / f"{SPECIALTY_NAME}_报告.json"
    for category, rows in (("直管", pipe_rows), ("法兰", flange_rows), ("管件", fitting_rows)):
        write_json(output_paths[category], rows)

    report = {
        "specialty": SPECIALTY_NAME,
        "source_labels": [
            *(f"真实项目数据-{path.stem}" for path in workbook_paths),
            SYNTHETIC_SOURCE_LABEL,
        ],
        "source_excels": [str(path.resolve()) for path in workbook_paths],
        "seed": args.seed,
        "real_project_selection": real_report,
        "output_counts": {
            "直管": len(pipe_rows),
            "直管_真实项目": len(real_rows["直管"]),
            "直管_合成骨架": len(synthetic_pipe_rows),
            "法兰": len(flange_rows),
            "法兰_真实项目": len(real_rows["法兰"]),
            "法兰_合成骨架": len(synthetic_flange_rows),
            "管件": len(fitting_rows),
            "管件_真实项目": len(real_rows["管件"]),
            "管件_合成骨架": len(synthetic_fitting_rows),
        },
        "distributions": {
            "直管_FLANGE_STYLE": distribution(pipe_rows, "FLANGE_STYLE"),
            "直管_MANU": distribution(pipe_rows, "MANU"),
            "法兰_BODY": distribution(flange_rows, "BODY"),
            "法兰_CONN": distribution(flange_rows, "CONN"),
            "法兰_SEAL": distribution(flange_rows, "SEAL"),
            "管件_BODY": distribution(fitting_rows, "BODY"),
            "管件_FLANGE_STYLE": distribution(fitting_rows, "FLANGE_STYLE"),
            "管件_MANU": distribution(fitting_rows, "MANU"),
        },
        "validation": {
            "直管": validate_rows("直管", pipe_rows, all_existing),
            "法兰": validate_rows("法兰", flange_rows, all_existing),
            "管件": validate_rows("管件", fitting_rows, all_existing),
            "cross_file_description_overlap": len(
                ({normalize_description(row["input"]) for row in pipe_rows}
                 & {normalize_description(row["input"]) for row in flange_rows})
                | ({normalize_description(row["input"]) for row in pipe_rows}
                   & {normalize_description(row["input"]) for row in fitting_rows})
                | ({normalize_description(row["input"]) for row in flange_rows}
                   & {normalize_description(row["input"]) for row in fitting_rows})
            ),
        },
        "outputs": {category: str(path.resolve()) for category, path in output_paths.items()},
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
