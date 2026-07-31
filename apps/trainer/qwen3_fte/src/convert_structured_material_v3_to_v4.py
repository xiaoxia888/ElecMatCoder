#!/usr/bin/env python3
"""Convert v3 material labels to the minimal coding-oriented v4 values.

V4 keeps the v3 JSON schema but changes VALUE semantics:

* keep the shortest source-backed material expression that can determine code;
* remove a material-standard prefix when the remaining grade is self-contained;
* retain the standard for context-dependent grades such as Gr.B and Gr.6;
* never synthesize an equivalent grade that is absent from the source.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = (
    ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范"
    / "结构化原始牌号/重新划分_v3"
)
DEFAULT_OUTPUT_DIR = SOURCE_DIR.parent / "重新划分_v4"
TRAIN_NAME = "材质规范_结构化原始牌号_train.json"
VAL_NAME = "材质规范_结构化原始牌号_val.json"

ALLOWED_PARTS = {"BODY", "LINING", "INNER_PIPE", "OUTER_PIPE", "FLANGE"}
ASTM_DESIGNATION_RE = re.compile(
    r"^(?P<astm>ASTM\s+)?"
    r"(?P<standard>[AB]\d+[A-Z]?(?:/[AB]?\d+[A-Z]?)?)"
    r"(?:\s+|-(?=[A-Za-z]))?"
    r"(?P<grade>.*)$",
    re.IGNORECASE,
)
CONTEXT_DEPENDENT_GRADE_RE = re.compile(
    r"^(?:"
    r"Gr(?:ade)?\.?\s*"
    r"|(?:CC|C|B)\d+(?:\b|$)"
    r"|\d+(?:\.\d+)?\s*Cr(?:\b|$)"
    r")",
    re.IGNORECASE,
)
STANDARD_IN_PARENTHESES_RE = re.compile(
    r"^(?P<outer>\d{3,5})\s*[（(]\s*"
    r"(?:ASTM\s+)?A\d+[A-Z]?\s+(?P<grade>[A-Za-z]+\d+[A-Za-z0-9/-]*)"
    r"\s*[)）]$",
    re.IGNORECASE,
)
B_STANDARD_ALIAS_RE = re.compile(
    r"^(?:ASTM\s+)?B\d+[A-Z]?\s*[（(]\s*(?:UNS\s+)?"
    r"(?P<grade>[A-Z]\d{4,6})\s*[)）]$",
    re.IGNORECASE,
)
B_STANDARD_GRADE_ALIAS_RE = re.compile(
    r"^(?:ASTM\s+)?B\d+[A-Z]?[-\s]+"
    r"(?P<grade>[A-Za-z][A-Za-z0-9/-]+)"
    r"\s*[（(]\s*(?:UNS\s+)?[A-Z]\d{4,6}\s*[)）]$",
    re.IGNORECASE,
)
STRONG_4PE_RE = re.compile(
    r"(?:"
    r"(?:外防腐|外涂|外覆|涂层|涂覆|加强级).{0,30}"
    r"(?<![A-Za-z0-9])4PE(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])4PE(?![A-Za-z0-9]).{0,30}"
    r"(?:外防腐|外涂|外覆|涂层|涂覆|加强级)"
    r")",
    re.IGNORECASE,
)
MS97_SOURCE_RE = re.compile(
    r"(?:MSS\s*SP|MS|SP)\s*[-–—]?\s*97\b",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _minimize_single_expression(value: str) -> tuple[str, str]:
    value = clean(value)
    if not value:
        return value, "unchanged"

    match = STANDARD_IN_PARENTHESES_RE.fullmatch(value)
    if match:
        return clean(match.group("grade")), "prefer_explicit_international_grade"

    match = B_STANDARD_GRADE_ALIAS_RE.fullmatch(value)
    if match:
        return clean(match.group("grade")), "remove_material_standard"

    match = B_STANDARD_ALIAS_RE.fullmatch(value)
    if match:
        return clean(match.group("grade")), "remove_material_standard"

    # Preserve an explicitly composite parenthetical expression, but minimize
    # each material designation independently.
    composite = re.fullmatch(
        r"(?P<outer>.+?)\s+\((?P<inner>(?:ASTM\s+)?[AB]\d+.+)\)",
        value,
        re.IGNORECASE,
    )
    if composite:
        outer, outer_reason = minimize_material_value(composite.group("outer"))
        inner, inner_reason = minimize_material_value(composite.group("inner"))
        reason = (
            "remove_material_standard"
            if "remove_material_standard" in {outer_reason, inner_reason}
            else "unchanged"
        )
        return f"{outer} ({inner})", reason

    match = ASTM_DESIGNATION_RE.fullmatch(value)
    if match:
        astm = clean(match.group("astm"))
        standard = clean(match.group("standard"))
        grade = clean(match.group("grade"))
        original_standard = f"{astm} {standard}".strip()
        if not grade:
            # A105/A106 itself is the shortest source-backed identifier.
            return standard, "remove_astm_prefix" if astm else "unchanged"
        if CONTEXT_DEPENDENT_GRADE_RE.match(grade):
            return f"{original_standard} {grade}", "retain_context_standard"
        return grade, "remove_material_standard"

    api_match = re.fullmatch(
        r"(?P<standard>API\s*5L)\s+(?P<grade>.+)",
        value,
        re.IGNORECASE,
    )
    if api_match:
        standard = clean(api_match.group("standard"))
        grade = clean(api_match.group("grade"))
        if CONTEXT_DEPENDENT_GRADE_RE.match(grade):
            return f"{standard} {grade}", "retain_context_standard"
        return grade, "remove_material_standard"

    return value, "unchanged"


def minimize_material_value(value: str) -> tuple[str, str]:
    """Return a minimal designation without inventing aliases.

    Alternatives are minimized branch-by-branch. Slash expressions are kept
    intact because they can represent dual certification or a real material
    combination rather than two independent alternatives.
    """
    value = clean(value)
    alternatives = re.split(r"\s+or\s+", value, flags=re.IGNORECASE)
    if len(alternatives) > 1:
        converted = [_minimize_single_expression(item)[0] for item in alternatives]
        result = " or ".join(converted)
        return result, "minimize_alternatives" if result != value else "unchanged"
    return _minimize_single_expression(value)


def repair_minimal_value_from_source(
    source_text: str,
    old_value: str,
    minimized_value: str,
) -> tuple[str, str]:
    """Repair known source-backed losses without creating an equivalent alias."""
    if (
        minimized_value.upper().endswith("CL")
        and re.search(
            rf"{re.escape(minimized_value)}\s*[.]?\s*\d+\b",
            source_text,
            re.IGNORECASE,
        )
    ):
        return minimized_value[:-2], "remove_class_fragment"

    if minimized_value.upper() == "HDPE" and re.search(
        r"HDPE\s*[（(]\s*PE100\s*[)）]",
        source_text,
        re.IGNORECASE,
    ):
        return "HDPE(PE100)", "recover_explicit_material_level"

    if minimized_value.upper() == "S316L" and re.search(
        r"(?<![A-Za-z0-9])SS316L(?![A-Za-z0-9])",
        source_text,
        re.IGNORECASE,
    ):
        return "SS316L", "recover_explicit_grade_characters"

    dual = re.fullmatch(
        r"(?P<prefix>WPS|WP|TP|F)"
        r"(?P<first>[A-Za-z0-9]+)"
        r"/(?P<second>[A-Za-z0-9]+)",
        minimized_value,
        re.IGNORECASE,
    )
    if dual:
        full_value = (
            f"{dual.group('prefix')}{dual.group('first')}/"
            f"{dual.group('prefix')}{dual.group('second')}"
        )
        source_compact = re.sub(r"[\s._-]+", "", source_text).upper()
        if full_value.upper() in source_compact:
            return full_value, "recover_explicit_dual_prefix"

    return minimized_value, "unchanged"


def add_source_backed_special_requirements(
    source_text: str,
    materials: list[dict[str, Any]],
) -> bool:
    if not STRONG_4PE_RE.search(source_text):
        return False

    target = next(
        (item for item in materials if clean(item.get("PART")).upper() == "BODY"),
        materials[0] if materials else None,
    )
    if target is None:
        return False

    requirements = unique(
        [clean(item) for item in target.get("SPECIAL_REQ", [])] + ["4PE"]
    )
    changed = requirements != target.get("SPECIAL_REQ", [])
    target["SPECIAL_REQ"] = requirements
    return changed


def normalize_product_standards(
    source_text: str,
    standards: Any,
) -> tuple[list[dict[str, str]], list[str]]:
    result: list[dict[str, str]] = []
    changes: list[str] = []
    for item in standards if isinstance(standards, list) else []:
        if not isinstance(item, dict):
            continue
        body = clean(item.get("BODY"))
        if not body:
            continue
        if body == "MS97" and not MS97_SOURCE_RE.search(source_text):
            changes.append("remove_unsupported_ms97")
            continue
        if body not in {value["BODY"] for value in result}:
            result.append({"BODY": body})

    if re.search(r"(?<![A-Za-z0-9])02S403(?![A-Za-z0-9])", source_text, re.I):
        if not any(item["BODY"] == "02S403" for item in result):
            result.append({"BODY": "02S403"})
            changes.append("add_explicit_02s403")
    return result, changes


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_text = clean(row.get("input"))
    output = row.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be an object")

    source_materials = output.get("MATERIAL")
    if not isinstance(source_materials, list) or not source_materials:
        raise ValueError("MATERIAL must be a non-empty list")

    materials: list[dict[str, Any]] = []
    changes: list[dict[str, str]] = []
    for item in source_materials:
        if not isinstance(item, dict):
            raise ValueError("MATERIAL item must be an object")
        old_value = clean(item.get("VALUE"))
        new_value, reason = minimize_material_value(old_value)
        repaired_value, repair_reason = repair_minimal_value_from_source(
            source_text,
            old_value,
            new_value,
        )
        if repaired_value != new_value:
            new_value = repaired_value
            reason = repair_reason
        converted_item = {
            "PART": clean(item.get("PART")).upper(),
            "VALUE": new_value,
            "SPECIAL_REQ": unique(
                [clean(value) for value in item.get("SPECIAL_REQ", [])]
            ),
        }
        materials.append(converted_item)
        if old_value != new_value:
            changes.append(
                {
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": reason,
                }
            )

    special_req_added = add_source_backed_special_requirements(
        source_text,
        materials,
    )
    standards, standard_changes = normalize_product_standards(
        source_text,
        output.get("STANDARD", []),
    )
    converted = {
        "input": source_text,
        "output": {
            "MATERIAL": materials,
            "STANDARD": standards,
        },
    }
    return converted, {
        "value_changes": changes,
        "special_req_4pe_added": special_req_added,
        "standard_changes": standard_changes,
    }


def validate_row(row: dict[str, Any], source: str, index: int) -> list[str]:
    errors: list[str] = []
    if not row.get("input"):
        errors.append("input为空")

    output = row.get("output")
    if not isinstance(output, dict) or set(output) != {"MATERIAL", "STANDARD"}:
        return [f"{source}第{index + 1}条: output字段结构无效"]

    materials = output.get("MATERIAL")
    if not isinstance(materials, list) or not materials:
        errors.append("MATERIAL必须是非空数组")
    else:
        for material_index, item in enumerate(materials):
            prefix = f"MATERIAL[{material_index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix}不是对象")
                continue
            if set(item) != {"PART", "VALUE", "SPECIAL_REQ"}:
                errors.append(f"{prefix}字段不符合v4结构")
            if item.get("PART") not in ALLOWED_PARTS:
                errors.append(f"{prefix}.PART无效: {item.get('PART')!r}")
            if not clean(item.get("VALUE")):
                errors.append(f"{prefix}.VALUE为空")
            if not isinstance(item.get("SPECIAL_REQ"), list):
                errors.append(f"{prefix}.SPECIAL_REQ不是数组")

    standards = output.get("STANDARD")
    if not isinstance(standards, list):
        errors.append("STANDARD不是数组")
    else:
        for standard_index, item in enumerate(standards):
            if (
                not isinstance(item, dict)
                or set(item) != {"BODY"}
                or not clean(item.get("BODY"))
            ):
                errors.append(f"STANDARD[{standard_index}]结构无效")

    return [f"{source}第{index + 1}条: {error}" for error in errors]


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} root must be a list")
    return data


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_dataset(
    rows: list[dict[str, Any]],
    source_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], Counter[tuple[str, str]]]:
    converted: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    mappings: Counter[tuple[str, str]] = Counter()

    for index, source_row in enumerate(rows):
        try:
            row, audit = convert_row(source_row)
        except (TypeError, ValueError) as exc:
            reviews.append(
                {
                    "source": source_name,
                    "source_index": index,
                    "input": clean(source_row.get("input")),
                    "reason": f"转换失败：{exc}",
                    "source_output": deepcopy(source_row.get("output")),
                }
            )
            continue

        errors = validate_row(row, source_name, index)
        if errors:
            reviews.append(
                {
                    "source": source_name,
                    "source_index": index,
                    "input": row["input"],
                    "reason": "；".join(errors),
                    "converted_output": deepcopy(row.get("output")),
                }
            )
            continue

        converted.append(row)
        if audit["value_changes"]:
            stats["VALUE已最小化的行"] += 1
        for change in audit["value_changes"]:
            stats[f"转换原因_{change['reason']}"] += 1
            mappings[(change["old_value"], change["new_value"])] += 1
        if audit["special_req_4pe_added"]:
            stats["补充4PE的行"] += 1
        for reason in audit["standard_changes"]:
            stats[f"产品规范修复_{reason}"] += 1

    return converted, reviews, stats, mappings


def build_report(
    train_source: Path,
    val_source: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_converted: list[dict[str, Any]],
    val_converted: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    stats: Counter[str],
    mappings: Counter[tuple[str, str]],
) -> dict[str, Any]:
    values: Counter[str] = Counter(
        item["VALUE"]
        for row in train_converted + val_converted
        for item in row["output"]["MATERIAL"]
    )
    retained_context = {
        value: count
        for value, count in values.items()
        if (
            re.match(r"^(?:ASTM\s+)?[AB]\d+", value, re.IGNORECASE)
            or re.match(r"^API\s*5L\s+Gr", value, re.IGNORECASE)
        )
    }
    return {
        "schema_version": "v4",
        "label_semantics": "能够唯一确定材质编码的最小原文材料表达",
        "source": {
            "train": str(train_source),
            "val": str(val_source),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "output": {
            "train_rows": len(train_converted),
            "val_rows": len(val_converted),
            "review_rows": len(reviews),
            "rows_preserved": len(train_converted) + len(val_converted),
        },
        "conversion_statistics": dict(sorted(stats.items())),
        "value_mapping_count": len(mappings),
        "value_mappings": [
            {"old": old, "new": new, "count": count}
            for (old, new), count in mappings.most_common()
        ],
        "retained_context_dependent_values": [
            {"value": value, "count": count}
            for value, count in sorted(
                retained_context.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "top_values": [
            {"value": value, "count": count}
            for value, count in values.most_common(100)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将v3材质数据转换为最小可编码VALUE的v4结构"
    )
    parser.add_argument("--train", type=Path, default=SOURCE_DIR / TRAIN_NAME)
    parser.add_argument("--val", type=Path, default=SOURCE_DIR / VAL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)

    train_converted, train_reviews, train_stats, train_mappings = convert_dataset(
        train_rows,
        "train",
    )
    val_converted, val_reviews, val_stats, val_mappings = convert_dataset(
        val_rows,
        "val",
    )
    reviews = train_reviews + val_reviews
    stats = train_stats + val_stats
    mappings = train_mappings + val_mappings
    report = build_report(
        args.train,
        args.val,
        train_rows,
        val_rows,
        train_converted,
        val_converted,
        reviews,
        stats,
        mappings,
    )

    dump_json(args.output_dir / TRAIN_NAME, train_converted)
    dump_json(args.output_dir / VAL_NAME, val_converted)
    dump_json(
        args.output_dir / "材质规范_结构化原始牌号_v4_转换报告.json",
        report,
    )
    dump_json(
        args.output_dir / "材质规范_结构化原始牌号_v4_待复核.json",
        reviews,
    )
    print(json.dumps(report["output"], ensure_ascii=False, indent=2))
    return 0 if not reviews else 1


if __name__ == "__main__":
    raise SystemExit(main())
