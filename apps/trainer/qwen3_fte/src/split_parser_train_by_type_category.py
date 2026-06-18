#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("apps/trainer/qwen3_fte/output/parser_train_new_schema.json")
DEFAULT_MAPPING = Path("docs/模型分类总预览.json")
DEFAULT_OUTPUT_DIR = Path("apps/trainer/qwen3_fte/output/按8类拆分数据集")

WELD_MANU_VALUES = {"WELDED", "ERW", "EFW", "SAWL", "SAWH", "SAW"}
PRIMARY_MANU_VALUES = {"SMLS", "FORGED"} | WELD_MANU_VALUES
THREAD_CONN_VALUES = {"THD", "NPT"}
THREAD_END_VALUES = {"MNPT", "FNPT", "NPTF", "FTE", "MTE"}
SW_CONN_VALUES = {"SW"}

PIPE_LIKE_BODIES = {"直管", "法兰管", "夹套钢管"}
TEE_LIKE_BODIES = {"三通", "等径三通", "异径三通", "斜三通", "异径斜三通"}

CATEGORY_TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "直管": ("BODY", "MANU", "CONN", "ENDS"),
    "法兰": ("BODY", "CONN", "SEAL", "ENDS"),
    "弯头": ("BODY", "GEOMETRY", "MANU", "CONN", "ENDS"),
    "三通": ("BODY", "GEOMETRY", "MANU", "CONN", "ENDS"),
    "变径件": ("BODY", "MANU", "CONN", "ENDS"),
    "支管台": ("BODY", "MANU", "CONN", "ENDS"),
    "管帽": ("BODY", "MANU", "CONN", "ENDS"),
    "其他管件": ("BODY", "MANU", "CONN", "ENDS"),
}

EXPLICIT_BODY_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "法兰": (
        ("Orifice Flange Welding Neck Type", "孔板对焊法兰"),
        ("Orifice Flange Welding Neck", "孔板对焊法兰"),
        ("PAD FLANGE", "板式平焊法兰"),
        ("PadFlange", "板式平焊法兰"),
        ("Pad Flange", "板式平焊法兰"),
        ("REDUCING FLANGE", "异径法兰"),
        ("THREADED FLANGE", "螺纹法兰"),
        ("THREADFLANGE", "螺纹法兰"),
        ("Screwed Flange", "螺纹法兰"),
        ("Flange Threaded", "螺纹法兰"),
        ("SOCKET WELD FLANGE", "承插焊法兰"),
        ("SOCKET FLANGE", "承插焊法兰"),
        ("Socket Welding Flange", "承插焊法兰"),
        ("带颈对焊法兰", "带颈对焊法兰"),
        ("带颈对焊钢制管法兰", "带颈对焊法兰"),
        ("带颈平焊法兰", "带颈平焊法兰"),
        ("Flange Slip On", "带颈平焊法兰"),
        ("板式平焊法兰", "板式平焊法兰"),
        ("平焊法兰", "平焊法兰"),
        ("对焊法兰", "对焊法兰"),
        ("螺纹法兰", "螺纹法兰"),
        ("承插焊法兰", "承插焊法兰"),
        ("法兰盖", "法兰盖"),
        ("BF", "盲法兰"),
        ("盲法兰", "盲法兰"),
        ("8字盲板", "8字盲板"),
        ("八字盲板", "8字盲板"),
        ("Figure 8 Blanks", "8字盲板"),
        ("Figure-8 Blanks", "8字盲板"),
        ("阀门插板", "插板"),
        ("插板", "插板"),
        ("插板+垫环", "插板"),
        ("插板和垫环", "插板"),
        ("插板&垫环", "插板"),
        ("BLANK+SPACER", "插板"),
        ("BLANK & SPACER", "插板"),
        ("SPACER & BLANKS", "插板"),
        ("夹套法兰", "夹套法兰"),
        ("松套法兰", "松套法兰"),
        ("FLANGE LAPPED", "松套法兰"),
        ("活套法兰", "松套法兰"),
        ("Flange Lapped", "松套法兰"),
    ),
    "支管台": (
        ("SOCKOLET", "承插焊支管台"),
        ("SOCKET OLET", "承插焊支管台"),
        ("承插焊支管台", "承插焊支管台"),
        ("承插支管台", "承插焊支管台"),
        ("承插焊管接台", "承插焊支管台"),
        ("承插焊支管座", "承插焊支管台"),
        ("THREDOLET", "螺纹支管台"),
        ("THREAD OLET", "螺纹支管台"),
        ("丝扣支管台", "螺纹支管台"),
        ("螺纹管接台", "螺纹支管台"),
    ),
}


def listify_str(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def has_any(values: list[str], candidates: set[str]) -> bool:
    return any(v in candidates for v in values)


def text_has_explicit_weld_manu_hint(text: str) -> bool:
    text = text.upper()
    return any(token in text for token in ("WELDED", "ERW", "EFW", "SAWL", "SAWH", "SAW"))


def text_has_generic_weld_hint(text: str) -> bool:
    return "焊接" in text


def text_has_smls_hint(text: str) -> bool:
    text = text.upper()
    return any(token in text for token in ("无缝", "SMLS", "SEAMLESS"))


def text_has_forged_hint(text: str) -> bool:
    text = text.upper()
    return any(token in text for token in ("锻制", "FORGED"))


def text_has_named_welded_body_hint(text: str) -> bool:
    return any(
        token in text for token in (
            "焊接钢管",
            "焊接不锈钢管",
            "焊接碳钢管",
            "焊接合金钢管",
            "焊接三通",
        )
    )


def text_has_thread_hint(text: str) -> bool:
    text = text.upper()
    return any(token in text for token in ("螺纹", "THREADED", "THREAD"))


def text_has_sw_hint(text: str) -> bool:
    text = text.upper()
    return any(token in text for token in ("承插焊", "SOCKET WELD", "SOCKOLET", "SW "))


def normalize_entry(
    row: dict[str, Any],
    category: str,
    normalized_body: str,
    raw_body: str,
) -> tuple[dict[str, Any], list[str]]:
    row_copy = copy.deepcopy(row)
    actions: list[str] = []

    output = row_copy.setdefault("output", {})
    normalized_body = apply_explicit_body_override(
        category,
        row_copy.get("input", ""),
        normalized_body,
    )
    type_obj = output.setdefault("TYPE", {})
    type_obj["BODY"] = normalized_body
    actions.append(f"BODY->{normalized_body}")
    source_text = f"{raw_body} || {row_copy.get('input', '')}"

    manu = listify_str(type_obj.get("MANU", []))
    conn = listify_str(type_obj.get("CONN", []))
    ends = listify_str(type_obj.get("ENDS", []))
    explicit_weld = (
        text_has_explicit_weld_manu_hint(source_text)
        or text_has_named_welded_body_hint(source_text)
    )
    generic_weld = text_has_generic_weld_hint(source_text)
    has_smls = text_has_smls_hint(source_text)
    has_forged = text_has_forged_hint(source_text)

    if (has_smls or has_forged) and not explicit_weld:
        filtered = [m for m in manu if m not in WELD_MANU_VALUES]
        if filtered != manu:
            manu = filtered
            actions.append("MANU-=WELDED_FAMILY")

    if has_smls and not has_any(manu, PRIMARY_MANU_VALUES):
        manu.append("SMLS")
        actions.append("MANU+=SMLS")

    if has_forged and not has_any(manu, PRIMARY_MANU_VALUES):
        manu.append("FORGED")
        actions.append("MANU+=FORGED")

    if normalized_body in PIPE_LIKE_BODIES | TEE_LIKE_BODIES:
        if (explicit_weld or generic_weld) and not has_any(manu, PRIMARY_MANU_VALUES):
            manu.append("WELDED")
            actions.append("MANU+=WELDED")

    if text_has_thread_hint(source_text):
        if not has_any(conn, THREAD_CONN_VALUES) and not has_any(ends, THREAD_END_VALUES):
            conn.append("THD")
            actions.append("CONN+=THD")

    if text_has_sw_hint(source_text):
        if not has_any(conn, SW_CONN_VALUES):
            conn.append("SW")
            actions.append("CONN+=SW")

    type_obj["MANU"] = manu
    type_obj["CONN"] = conn
    type_obj["ENDS"] = ends
    type_obj = reshape_type_by_category(type_obj, category)
    output["TYPE"] = type_obj

    return row_copy, actions


def apply_explicit_body_override(category: str, input_text: str, normalized_body: str) -> str:
    input_text_upper = str(input_text).upper()
    for phrase, target_body in EXPLICIT_BODY_OVERRIDES.get(category, ()):
        if str(phrase).upper() in input_text_upper:
            return target_body
    return normalized_body


def reshape_type_by_category(type_obj: dict[str, Any], category: str) -> dict[str, Any]:
    allowed = CATEGORY_TYPE_FIELDS.get(category, tuple(type_obj.keys()))
    reshaped: dict[str, Any] = {}

    for field in allowed:
        if field == "BODY":
            reshaped["BODY"] = type_obj.get("BODY", "")
        elif field == "GEOMETRY":
            geometry = type_obj.get("GEOMETRY", {}) if isinstance(type_obj.get("GEOMETRY", {}), dict) else {}
            if category == "三通":
                reshaped["GEOMETRY"] = {
                    "ANGLE": geometry.get("ANGLE", "")
                }
            else:
                reshaped["GEOMETRY"] = {
                    "ANGLE": geometry.get("ANGLE", ""),
                    "RADIUS": geometry.get("RADIUS", ""),
                }
        else:
            reshaped[field] = listify_str(type_obj.get(field, []))

    return reshaped


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def build_alias_to_category(mapping: dict[str, dict[str, list[str]]]) -> tuple[dict[str, str], dict[str, str]]:
    alias_to_category: dict[str, str] = {}
    alias_to_normalized_body: dict[str, str] = {}

    for category, body_mapping in mapping.items():
        for normalized_body, aliases in body_mapping.items():
            all_names = [normalized_body, *aliases]
            for name in all_names:
                alias = normalize_text(name)
                if alias in alias_to_category and alias_to_category[alias] != category:
                    raise ValueError(
                        f"别名冲突: {alias!r} 同时映射到 {alias_to_category[alias]!r} 和 {category!r}"
                    )
                alias_to_category[alias] = category
                alias_to_normalized_body[alias] = normalized_body
    return alias_to_category, alias_to_normalized_body


def classify_entries(
    rows: list[dict[str, Any]],
    alias_to_category: dict[str, str],
    alias_to_normalized_body: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], Counter[str], list[dict[str, Any]], Counter[str]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncategorized: list[dict[str, Any]] = []
    uncategorized_bodies: Counter[str] = Counter()
    normalization_report: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        body = (
            row.get("output", {})
            .get("TYPE", {})
            .get("BODY", "")
        )
        body_key = normalize_text(body)

        if body_key in alias_to_category:
            category = alias_to_category[body_key]
            normalized_body = alias_to_normalized_body[body_key]
            row_copy, actions = normalize_entry(row, category, normalized_body, body)
            buckets[category].append(row_copy)
            normalization_report.append(
                {
                    "index": idx,
                    "category": category,
                    "raw_body": body,
                    "normalized_body": normalized_body,
                    "actions": actions,
                }
            )
            for action in actions:
                action_counter[action] += 1
            continue

        uncategorized_bodies[body_key or "<EMPTY>"] += 1
        uncategorized.append(
            {
                "index": idx,
                "body": body,
                "input": row.get("input", ""),
                "row": row,
            }
        )

    return buckets, uncategorized, uncategorized_bodies, normalization_report, action_counter


def write_outputs(
    output_dir: Path,
    mapping: dict[str, dict[str, list[str]]],
    buckets: dict[str, list[dict[str, Any]]],
    uncategorized: list[dict[str, Any]],
    uncategorized_bodies: Counter[str],
    normalization_report: list[dict[str, Any]],
    action_counter: Counter[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "分类数": len(mapping),
        "分类统计": {},
        "未分类条数": len(uncategorized),
        "未分类BODY统计": dict(uncategorized_bodies.most_common()),
        "归一化动作统计": dict(action_counter.most_common()),
    }

    for category in mapping:
        rows = buckets.get(category, [])
        dump_json(output_dir / f"{category}.json", rows)
        summary["分类统计"][category] = len(rows)

    dump_json(output_dir / "未分类.json", uncategorized)
    dump_json(output_dir / "归一化报告.json", normalization_report)
    dump_json(output_dir / "拆分统计.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 8 类 TYPE 分类拆分 parser_train_new_schema.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入数据集路径")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="分类映射 JSON 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_json(args.input)
    mapping = load_json(args.mapping)

    if not isinstance(rows, list):
        raise TypeError(f"输入数据集不是数组: {args.input}")
    if not isinstance(mapping, dict):
        raise TypeError(f"分类映射不是对象: {args.mapping}")

    alias_to_category, alias_to_normalized_body = build_alias_to_category(mapping)
    buckets, uncategorized, uncategorized_bodies, normalization_report, action_counter = classify_entries(
        rows,
        alias_to_category,
        alias_to_normalized_body,
    )
    write_outputs(
        args.output_dir,
        mapping,
        buckets,
        uncategorized,
        uncategorized_bodies,
        normalization_report,
        action_counter,
    )

    print(f"输入总条数: {len(rows)}")
    for category in mapping:
        print(f"{category}: {len(buckets.get(category, []))}")
    print(f"未分类: {len(uncategorized)}")
    if uncategorized_bodies:
        print("未分类 BODY TOP20:")
        for body, count in uncategorized_bodies.most_common(20):
            print(f"  {body}: {count}")


if __name__ == "__main__":
    main()
