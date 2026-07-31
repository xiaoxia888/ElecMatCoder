#!/usr/bin/env python3
"""Convert the reviewed v2 material dataset to the compact v3 schema.

V3 treats a material designation as one complete VALUE per physical part:

    {"PART": "BODY", "VALUE": "ASTM A403 WP304/304L", "SPECIAL_REQ": []}

The old STANDARD/GRADE/CLASS split and MATERIAL_RELATION are intentionally not
carried into the output. Product standards remain in the root STANDARD array.
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
    / "结构化原始牌号/重新划分_v2"
)
DEFAULT_OUTPUT_DIR = SOURCE_DIR.parent / "重新划分_v3"
TRAIN_NAME = "材质规范_结构化原始牌号_train.json"
VAL_NAME = "材质规范_结构化原始牌号_val.json"

ALLOWED_PARTS = {"BODY", "LINING", "INNER_PIPE", "OUTER_PIPE", "FLANGE"}
OLD_RELATIONS = {
    "SINGLE",
    "DUAL_CERTIFIED",
    "EQUIVALENT",
    "ALTERNATIVE",
    "COMPOSITE",
}
SHARED_GRADE_PREFIXES = ("WPS", "WP", "TP", "F")
MS97_SOURCE_RE = re.compile(
    r"(?:MSS\s*SP|MS|SP)\s*[-–—]?\s*97\b",
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
SUS_DESIGNATION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"SUS"
    r"(?:[\s._-]*(?P<form>F))?"
    r"[\s._-]*(?P<grade>S?\d[A-Za-z0-9]*)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
SS_DESIGNATION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<designation>SS\d[A-Za-z0-9]*)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def compact_designation(value: str) -> str:
    value = re.sub(r"(?i)\bASTM\b", "", value)
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", value).upper()


def material_atom(item: dict[str, Any]) -> str:
    """Build one complete designation without guessing a final code."""
    standard = clean(item.get("STANDARD"))
    grade = clean(item.get("GRADE"))
    material_class = clean(item.get("CLASS"))
    parts: list[str] = []

    if standard:
        parts.append(standard)
    if grade and grade.casefold() != standard.casefold():
        parts.append(grade)
    if material_class:
        parts.append(material_class)

    return " ".join(parts)


def recover_complete_parenthesized_value(
    source_text: str,
    computed_value: str,
) -> str | None:
    """Recover source forms such as 2205(A182 F51) and B564(N10276).

    The source phrase is accepted only when its inner designation, or the
    outer+inner combination, accounts for the already-reviewed v2 value. This
    prevents restoring discarded adjuncts such as CF415 from 20(CF415).
    """
    computed = compact_designation(computed_value)
    if not computed:
        return None

    pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?P<outer>[A-Za-z0-9][A-Za-z0-9._+\-/]{0,30})"
        r"\s*[（(]\s*"
        r"(?P<inner>[^()（）,，;；]{2,50}?)"
        r"\s*[)）]"
    )
    for match in pattern.finditer(source_text):
        outer = clean(match.group("outer")).rstrip("._+-/")
        inner = clean(match.group("inner"))
        inner_compact = compact_designation(inner)
        combined_compact = compact_designation(outer + inner)
        inner_only_match = computed == inner_compact and bool(re.search(r"\d", outer))
        if computed == combined_compact or inner_only_match:
            return f"{outer}({inner})"
    return None


def recover_source_sus_designation(
    source_text: str,
    computed_value: str,
) -> str | None:
    """Restore an explicit JIS SUS prefix when only that prefix was lost."""
    computed = compact_designation(computed_value)
    if not computed or computed.startswith("SUS"):
        return None

    for match in SUS_DESIGNATION_RE.finditer(source_text):
        form = (match.group("form") or "").upper()
        grade = match.group("grade").upper()
        designation_without_sus = f"{form}{grade}"
        if computed == compact_designation(designation_without_sus):
            return f"SUS {designation_without_sus}"
    return None


def recover_source_ss_designation(
    source_text: str,
    computed_value: str,
) -> str | None:
    """Restore SS316L-like source designations truncated to S316L."""
    computed = compact_designation(computed_value)
    if not computed or computed.startswith("SS"):
        return None

    for match in SS_DESIGNATION_RE.finditer(source_text):
        source_designation = match.group("designation").upper()
        if source_designation == f"S{computed}":
            return source_designation
    return None


def split_grade_prefix(value: str) -> tuple[str, str]:
    compact = re.sub(r"\s+", "", value)
    upper = compact.upper()
    for prefix in SHARED_GRADE_PREFIXES:
        if upper.startswith(prefix) and len(compact) > len(prefix):
            return compact[: len(prefix)], compact[len(prefix) :]
    return "", compact


def merge_same_standard_items(items: list[dict[str, Any]]) -> str | None:
    """Merge dual designations while retaining one shared ASTM standard."""
    standards = unique([clean(item.get("STANDARD")) for item in items])
    classes = unique([clean(item.get("CLASS")) for item in items])
    grades = unique([clean(item.get("GRADE")) for item in items])

    if len(standards) > 1 or len(classes) > 1 or not grades:
        return None

    standard = standards[0] if standards else ""
    grade_value: str
    if len(grades) == 1:
        grade_value = grades[0]
    else:
        prefixed = [split_grade_prefix(grade) for grade in grades]
        prefixes = {prefix.upper() for prefix, _ in prefixed if prefix}
        if (
            len(prefixes) == 1
            and all(prefix for prefix, _ in prefixed)
            and len(prefixed) == len(grades)
        ):
            prefix = prefixed[0][0]
            grade_value = prefix + "/".join(suffix for _, suffix in prefixed)
        else:
            grade_value = "/".join(grades)

    components = [value for value in (standard, grade_value) if value]
    if classes:
        components.append(classes[0])
    return " ".join(components)


def combine_same_part(
    items: list[dict[str, Any]],
    relation: str,
) -> dict[str, Any]:
    part = clean(items[0].get("PART")).upper() or "BODY"
    special_req = unique(
        [
            clean(requirement)
            for item in items
            for requirement in (item.get("SPECIAL_REQ") or [])
        ]
    )

    if len(items) == 1:
        value = material_atom(items[0])
    elif relation == "ALTERNATIVE":
        value = " or ".join(unique([material_atom(item) for item in items]))
    elif relation in {"DUAL_CERTIFIED", "EQUIVALENT"}:
        value = merge_same_standard_items(items) or "/".join(
            unique([material_atom(item) for item in items])
        )
    elif relation == "COMPOSITE":
        # Same physical part with multiple source designations is one complete
        # material expression, not multiple fake BODY components.
        atoms = unique([material_atom(item) for item in items])
        value = atoms[0] if len(atoms) == 1 else f"{atoms[0]} ({'; '.join(atoms[1:])})"
    else:
        value = "/".join(unique([material_atom(item) for item in items]))

    return {
        "PART": part,
        "VALUE": value,
        "SPECIAL_REQ": special_req,
    }


def convert_materials(
    materials: list[dict[str, Any]],
    relation: str,
) -> list[dict[str, Any]]:
    if relation not in OLD_RELATIONS:
        raise ValueError(f"unsupported MATERIAL_RELATION: {relation!r}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in materials:
        part = clean(item.get("PART")).upper() or "BODY"
        if part not in grouped:
            grouped[part] = []
            order.append(part)
        grouped[part].append(item)

    return [combine_same_part(grouped[part], relation) for part in order]


def normalize_root_standards(
    values: Any,
    source_text: str,
) -> tuple[list[dict[str, str]], list[str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    changes: list[str] = []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        body = clean(item.get("BODY"))
        if body == "MS97" and not MS97_SOURCE_RE.search(source_text):
            changes.append("删除原文无证据的MS97")
            continue
        if body and body not in seen:
            result.append({"BODY": body})
            seen.add(body)

    if re.search(r"(?<![A-Za-z0-9])02S403(?![A-Za-z0-9])", source_text, re.I):
        if "02S403" not in seen:
            result.append({"BODY": "02S403"})
            changes.append("补充原文明示的02S403")
    return result, changes


def repair_materials_from_source(
    source_text: str,
    materials: list[dict[str, Any]],
) -> list[str]:
    """Repair only values that are explicitly supported by the source text."""
    changes: list[str] = []
    source_compact = re.sub(r"[\s._-]+", "", source_text).upper()

    for item in materials:
        value = clean(item.get("VALUE"))
        repaired = value

        if (
            repaired.upper().endswith("CL")
            and re.search(
                rf"{re.escape(repaired)}\s*[.]?\s*\d+\b",
                source_text,
                re.IGNORECASE,
            )
        ):
            repaired = repaired[:-2]
            changes.append("移除牌号后的Class残片")

        if repaired.upper() == "HDPE" and re.search(
            r"HDPE\s*[（(]\s*PE100\s*[)）]",
            source_text,
            re.IGNORECASE,
        ):
            repaired = "HDPE(PE100)"
            changes.append("恢复原文明示的PE100等级")

        recovered_ss = recover_source_ss_designation(source_text, repaired)
        if recovered_ss:
            repaired = recovered_ss
            changes.append("恢复原文明示的SS牌号前缀")

        recovered_sus = recover_source_sus_designation(source_text, repaired)
        if recovered_sus:
            repaired = recovered_sus
            changes.append("恢复原文明示的JIS SUS牌号前缀")

        dual = re.fullmatch(
            r"(?P<head>.*?\s)?"
            r"(?P<prefix>WPS|WP|TP|F)"
            r"(?P<first>[A-Za-z0-9]+)"
            r"/(?P<second>[A-Za-z0-9]+)",
            repaired,
            re.IGNORECASE,
        )
        if dual:
            grade = (
                f"{dual.group('prefix')}{dual.group('first')}/"
                f"{dual.group('prefix')}{dual.group('second')}"
            )
            if grade.upper() in source_compact:
                repaired = f"{dual.group('head') or ''}{grade}".strip()
                changes.append("恢复双牌号第二项前缀")

        if repaired != value:
            item["VALUE"] = repaired

    target = next(
        (item for item in materials if item.get("PART") == "BODY"),
        materials[0] if materials else None,
    )
    if target is not None and STRONG_4PE_RE.search(source_text):
        requirements = unique(
            [clean(value) for value in target.get("SPECIAL_REQ", [])] + ["4PE"]
        )
        if requirements != target.get("SPECIAL_REQ", []):
            target["SPECIAL_REQ"] = requirements
            changes.append("补充4PE加强级外防腐")
    return changes


def source_row_removal_reason(source_text: str) -> str | None:
    """Remove rare same-part rows whose competing material standards are unclear."""
    has_a269_tp316 = re.search(
        r"(?<![A-Za-z0-9])(?:ASTM\s*)?A269\s*[-, ]\s*TP316\b",
        source_text,
        re.IGNORECASE,
    )
    has_a312_tp316 = re.search(
        r"(?<![A-Za-z0-9])ASTM\s*A312\s*TP316\b",
        source_text,
        re.IGNORECASE,
    )
    if has_a269_tp316 and has_a312_tp316:
        return "同一管材同时出现A269 TP316与ASTM A312 TP316，无法确定主材料标准"
    return None


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output = row.get("output") or {}
    materials = output.get("MATERIAL")
    relation = clean(output.get("MATERIAL_RELATION")).upper()
    if not isinstance(materials, list) or not materials:
        raise ValueError("MATERIAL must be a non-empty list")

    source_text = clean(row.get("input"))
    converted_materials = convert_materials(materials, relation)
    material_repairs = repair_materials_from_source(
        source_text,
        converted_materials,
    )
    source_value_recovered = False
    if len(converted_materials) == 1:
        recovered = recover_complete_parenthesized_value(
            source_text,
            converted_materials[0]["VALUE"],
        )
        if recovered and recovered != converted_materials[0]["VALUE"]:
            converted_materials[0]["VALUE"] = recovered
            source_value_recovered = True

    root_standards, standard_repairs = normalize_root_standards(
        output.get("STANDARD"),
        source_text,
    )
    converted = {
        "input": source_text,
        "output": {
            "MATERIAL": converted_materials,
            "STANDARD": root_standards,
        },
    }
    audit = {
        "old_relation": relation,
        "old_material_count": len(materials),
        "new_material_count": len(converted["output"]["MATERIAL"]),
        "parts": [item["PART"] for item in converted["output"]["MATERIAL"]],
        "source_value_recovered": source_value_recovered,
        "material_repairs": material_repairs,
        "standard_repairs": standard_repairs,
    }
    return converted, audit


def validate_row(row: dict[str, Any], source: str, index: int) -> list[str]:
    errors: list[str] = []
    if not row.get("input"):
        errors.append("input为空")

    output = row.get("output")
    if not isinstance(output, dict) or set(output) != {"MATERIAL", "STANDARD"}:
        return ["output字段必须且只能包含MATERIAL、STANDARD"]

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
                errors.append(f"{prefix}字段不符合v3结构")
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

    if errors:
        return [f"{source}第{index + 1}条: {error}" for error in errors]
    return []


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deduplicate_converted_rows(
    rows: list[dict[str, Any]],
    standard_frequency: Counter[str],
    source_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove duplicate inputs and resolve conflicting legacy normalizations."""
    selected: dict[str, tuple[int, dict[str, Any]]] = {}
    order: list[str] = []
    resolutions: list[dict[str, Any]] = []

    def score(row: dict[str, Any]) -> int:
        return sum(
            standard_frequency[clean(item.get("BODY"))]
            for item in row["output"]["STANDARD"]
        )

    for index, row in enumerate(rows):
        input_text = row["input"]
        if input_text not in selected:
            selected[input_text] = (index, row)
            order.append(input_text)
            continue

        previous_index, previous = selected[input_text]
        if canonical_json(previous["output"]) == canonical_json(row["output"]):
            chosen_index, chosen = previous_index, previous
            reason = "完全重复，保留首次出现"
        elif score(row) > score(previous):
            chosen_index, chosen = index, row
            selected[input_text] = (index, row)
            reason = "同输入标签冲突，选择全数据中更一致的规范编码"
        else:
            chosen_index, chosen = previous_index, previous
            reason = "同输入标签冲突，选择全数据中更一致的规范编码"

        resolutions.append(
            {
                "source": source_name,
                "input": input_text,
                "candidate_indices": [previous_index, index],
                "chosen_index": chosen_index,
                "reason": reason,
                "chosen_output": deepcopy(chosen["output"]),
                "discarded_output": deepcopy(
                    row["output"] if chosen_index == previous_index else previous["output"]
                ),
            }
        )

    return [selected[input_text][1] for input_text in order], resolutions


def convert_dataset(
    rows: list[dict[str, Any]],
    source_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Counter[str],
]:
    converted: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for index, source_row in enumerate(rows):
        input_text = clean(source_row.get("input"))
        removal_reason = source_row_removal_reason(input_text)
        if removal_reason:
            removed.append(
                {
                    "source": source_name,
                    "source_index": index,
                    "input": input_text,
                    "reason": removal_reason,
                    "source_output": deepcopy(source_row.get("output")),
                }
            )
            stats["规则删除项"] += 1
            continue

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
                    "source_output": deepcopy(source_row.get("output")),
                    "converted_output": deepcopy(row.get("output")),
                }
            )
            continue

        converted.append(row)
        stats[f"原关系_{audit['old_relation']}"] += 1
        stats[f"新部件数_{audit['new_material_count']}"] += 1
        if audit["new_material_count"] < audit["old_material_count"]:
            stats["同部件多标签已合并"] += 1
        if audit["source_value_recovered"]:
            stats["从原文恢复完整VALUE"] += 1
        for reason in audit["material_repairs"]:
            stats[f"材质修复_{reason}"] += 1
        for reason in audit["standard_repairs"]:
            stats[f"规范修复_{reason}"] += 1
        for part in audit["parts"]:
            stats[f"部件_{part}"] += 1

    return converted, reviews, removed, stats


def build_report(
    train_source: Path,
    val_source: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_converted: list[dict[str, Any]],
    val_converted: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    duplicate_resolutions: list[dict[str, Any]],
    stats: Counter[str],
) -> dict[str, Any]:
    value_counts: Counter[str] = Counter()
    part_counts: Counter[str] = Counter()
    for row in train_converted + val_converted:
        for item in row["output"]["MATERIAL"]:
            value_counts[item["VALUE"]] += 1
            part_counts[item["PART"]] += 1

    return {
        "schema_version": "v3",
        "schema": {
            "MATERIAL": [
                {
                    "PART": "物理部件",
                    "VALUE": "原文语义完整的材质表达",
                    "SPECIAL_REQ": ["参与材质编码的附加要求"],
                }
            ],
            "STANDARD": [{"BODY": "产品/制造/尺寸/检验规范编码"}],
            "removed_fields": [
                "MATERIAL[].STANDARD",
                "MATERIAL[].GRADE",
                "MATERIAL[].CLASS",
                "MATERIAL_RELATION",
            ],
        },
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
            "removed_rows": len(removed),
            "rows_preserved": len(train_converted) + len(val_converted),
            "duplicate_inputs_removed": len(duplicate_resolutions),
        },
        "conversion_statistics": dict(sorted(stats.items())),
        "part_distribution": dict(part_counts.most_common()),
        "top_values": [
            {"value": value, "count": count}
            for value, count in value_counts.most_common(100)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将已复核v2材质数据转换为PART+VALUE的v3结构"
    )
    parser.add_argument("--train", type=Path, default=SOURCE_DIR / TRAIN_NAME)
    parser.add_argument("--val", type=Path, default=SOURCE_DIR / VAL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)

    train_converted, train_reviews, train_removed, train_stats = convert_dataset(
        train_rows, "train"
    )
    val_converted, val_reviews, val_removed, val_stats = convert_dataset(
        val_rows,
        "val",
    )
    stats = train_stats + val_stats
    reviews = train_reviews + val_reviews
    removed = train_removed + val_removed

    standard_frequency: Counter[str] = Counter(
        clean(item.get("BODY"))
        for row in train_converted + val_converted
        for item in row["output"]["STANDARD"]
    )
    train_converted, train_duplicate_resolutions = deduplicate_converted_rows(
        train_converted,
        standard_frequency,
        "train",
    )
    val_converted, val_duplicate_resolutions = deduplicate_converted_rows(
        val_converted,
        standard_frequency,
        "val",
    )
    duplicate_resolutions = (
        train_duplicate_resolutions + val_duplicate_resolutions
    )
    stats["重复输入已删除"] = len(duplicate_resolutions)

    report = build_report(
        args.train,
        args.val,
        train_rows,
        val_rows,
        train_converted,
        val_converted,
        reviews,
        removed,
        duplicate_resolutions,
        stats,
    )

    dump_json(args.output_dir / TRAIN_NAME, train_converted)
    dump_json(args.output_dir / VAL_NAME, val_converted)
    dump_json(args.output_dir / "材质规范_结构化原始牌号_v3_转换报告.json", report)
    dump_json(args.output_dir / "材质规范_结构化原始牌号_v3_待复核.json", reviews)
    dump_json(args.output_dir / "材质规范_结构化原始牌号_v3_规则删除项.json", removed)
    dump_json(
        args.output_dir / "材质规范_结构化原始牌号_v3_重复输入消解报告.json",
        duplicate_resolutions,
    )

    print(json.dumps(report["output"], ensure_ascii=False, indent=2))
    return 0 if not reviews else 1


if __name__ == "__main__":
    raise SystemExit(main())
