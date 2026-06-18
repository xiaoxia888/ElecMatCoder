#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from split_parser_train_by_type_category import (
    DEFAULT_INPUT,
    DEFAULT_MAPPING,
    apply_explicit_body_override,
    build_alias_to_category,
    normalize_text,
)


DEFAULT_OUTPUT_DIR = Path("apps/trainer/qwen3_fte/output/body_raw_expression_distribution")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_alias_candidates(mapping: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for category, body_mapping in mapping.items():
        cat_out: dict[str, list[str]] = {}
        for normalized_body, aliases in body_mapping.items():
            seen: set[str] = set()
            values: list[str] = []
            for item in [normalized_body, *aliases]:
                text = str(item).strip()
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                values.append(text)
            values.sort(key=lambda x: (-len(x), x.casefold()))
            cat_out[normalized_body] = values
        out[category] = cat_out
    return out


def build_alias_pattern(alias: str) -> re.Pattern[str]:
    parts: list[str] = []
    for ch in alias:
        if ch.isspace():
            parts.append(r"\s+")
        elif ch == "+":
            parts.append(r"\s*\+\s*")
        elif ch == "&":
            parts.append(r"\s*&\s*")
        elif ch == "/":
            parts.append(r"\s*/\s*")
        elif ch == "-":
            parts.append(r"\s*-\s*")
        elif ch == ";":
            parts.append(r"\s*;\s*")
        elif ch == ",":
            parts.append(r"\s*,\s*")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts), flags=re.IGNORECASE)


SEGMENT_DELIMS = ",;，；|:："
RIGHT_STOP_RE = re.compile(
    r"""
    \s+(
        GB/T|HG/T|SH/T|NB/T|ASME|ASTM|MSS|DIN|EN|ISO|JIS|API|AISI|
        DN\d|NPS\d|SCH|S-\d|S\d|CL\d|PN\d|
        RF|FF|RJ|RTJ|MFM|BW|SW|THD|NPT|FNPT|MNPT|
        SMLS|WELDED|ERW|EFW|SAWL|SAWH|FORGED|
        0?6Cr|0Cr|1Cr|12Cr|15Cr|20G|20#|Q235|Q245|Q345|
        S\d{5}|TP\d|WP\d|CF\d|LF\d|F\d{2,4}|
        20"|16"|14"|12"|10"|8"|6"|4"|3"|2"|1"
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

FIELD_LABEL_RE = re.compile(
    r"""
    \b\d+\s*[.、]\s*(名称|规格|型号|规格型号|材质|标准|压力等级|连接形式|安装部位|设计要求|备注)\b|
    \b(名称|规格|型号|规格型号|材质|标准|压力等级|连接形式|安装部位|设计要求|备注)\s*[:：]
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

NOISE_PREFIX_RE = re.compile(
    r"""
    ^\s*(?:不锈钢管件|碳钢管件|合金钢管件|合金管件|锻钢制|国标锻钢制|钢制管件|管件|不锈钢|碳钢)\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

TAIL_CUT_RE = re.compile(
    r"""
    \s+(?:
        GB/T|HG/T|SH/T|NB/T|ASME|ASTM|MSS|DIN|EN|ISO|JIS|API|AISI|
        DN\d|NPS\d|OD\d|SCH|S-\d|S\d|CL\d|PN\d|
        RF|FF|RJ|RTJ|MFM|BW|SW|THD|NPT|FNPT|MNPT|
        WELDED|ERW|EFW|SAWL|SAWH|FORGED|
        S\d{5}|TP\d|WP\d|CF\d|LF\d|F\d{2,4}|
        0?6Cr|0Cr|1Cr|12Cr|15Cr|20G|20#|Q235|Q245|Q345|
        [\d.]+(?:mm|MM)|[\d.]+(?:x|X)[\d.]+
    ).*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

TRAILING_TOKEN_PATTERNS = [
    re.compile(r"\s*\(?\s*SMLS\s*$", flags=re.IGNORECASE),
    re.compile(r"\s*\(?\s*SEAMLESS\s*$", flags=re.IGNORECASE),
    re.compile(r"\s*\(?\s*WELDED\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+(?:SCH\s*\d+\w*|S-?\d+\w*|\d+S)\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+(?:DN|NPS)\s*\d+\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+(?:CL|PN)\s*\.?\s*\d+\s*$", flags=re.IGNORECASE),
    re.compile(r"\s+(?:304L?|316L?|321|347H?|310S?|2205|2507|20G|20#|20)\s*$", flags=re.IGNORECASE),
]


def expand_type_fragment(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and text[left - 1] not in SEGMENT_DELIMS:
        left -= 1

    right = end
    while right < len(text) and text[right] not in SEGMENT_DELIMS:
        right += 1

    fragment = text[left:right].strip()
    if not fragment:
        return fragment

    stop_match = RIGHT_STOP_RE.search(fragment, pos=max(0, start - left))
    if stop_match:
        fragment = fragment[:stop_match.start()].strip()

    fragment = re.sub(r"^\d+\s*[.、]\s*", "", fragment)
    fragment = re.sub(r"^(名称|规格|型号|材质|标准|形式)\s*[:：]\s*", "", fragment, flags=re.IGNORECASE)
    return fragment.strip() or text[start:end].strip()


def clean_raw_type_phrase(raw_type_phrase: str, normalized_body: str) -> str:
    cleaned = str(raw_type_phrase or "").strip()
    if not cleaned:
        return cleaned

    cleaned = cleaned.replace("（", "(").replace("）", ")")
    cleaned = cleaned.replace("【", "(").replace("】", ")")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;，；|:/")
    cleaned = re.sub(r"^\d+\s*[.、]\s*", "", cleaned)
    cleaned = re.sub(r"^(名称|规格|型号|材质|标准|形式)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)

    label_match = FIELD_LABEL_RE.search(cleaned)
    if label_match:
        cleaned = cleaned[:label_match.start()].rstrip(" ,;，；|:/")

    tail_match = TAIL_CUT_RE.search(cleaned)
    if tail_match:
        cleaned = cleaned[:tail_match.start()].rstrip(" ,;，；|:/")

    cleaned = re.sub(r"\s*\([^)]*$", "", cleaned)
    cleaned = re.sub(r"\s*[\\/]\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;，；|:/")

    if normalized_body in {"弯头", "等径三通", "异径三通", "斜三通", "异径斜三通"}:
        cleaned = NOISE_PREFIX_RE.sub("", cleaned).strip()

    for pattern in TRAILING_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip(" ,;，；|:/")

    cleaned = re.sub(r"\s*R\s*=\s*[\d.]+\s*(?:D|DN)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+\s*[°º度])\s+弯头", r"\1弯头", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+\s*[°º度])\s+长半径弯头", r"\1长半径弯头", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+\s*[°º度])\s+短半径弯头", r"\1短半径弯头", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(弯头)\s*DN\d+$", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(ELBOW)\s*DN\d+$", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(弯头)\s*\d+\s*/\s*(?:衬胶|PTFE|衬PTFE)\s*$", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(弯头)\s*/\s*(?:衬胶|PTFE|衬PTFE)\s*$", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"弯头\s+(?=\d)", "弯头", cleaned)
    cleaned = re.sub(r"([^\s])\s*DN\d+$", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;，；|:/")

    return cleaned or str(raw_type_phrase or "").strip()


def extract_raw_type_phrase(
    input_text: str,
    raw_body: str,
    aliases: list[str],
) -> tuple[str, str]:
    for alias in aliases:
        pattern = build_alias_pattern(alias)
        match = pattern.search(input_text)
        if match:
            fragment = expand_type_fragment(input_text, match.start(), match.end())
            return fragment, alias

    raw_body_text = str(raw_body or "").strip()
    if raw_body_text:
        return raw_body_text, "__raw_body_fallback__"

    return "<未命中原始种类短语>", "__unmatched__"


def analyze_rows(
    rows: list[dict[str, Any]],
    mapping: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    alias_to_category, alias_to_normalized_body = build_alias_to_category(mapping)
    alias_candidates = build_alias_candidates(mapping)

    nested_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    uncategorized_rows: list[dict[str, Any]] = []
    unresolved_expression_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        input_text = str(row.get("input", "") or "")
        raw_body = (
            row.get("output", {})
            .get("TYPE", {})
            .get("BODY", "")
        )
        body_key = normalize_text(raw_body)

        if body_key not in alias_to_category:
            uncategorized_rows.append(
                {
                    "index": idx,
                    "raw_body": raw_body,
                    "input": input_text,
                }
            )
            continue

        category = alias_to_category[body_key]
        normalized_body = alias_to_normalized_body[body_key]
        normalized_body = apply_explicit_body_override(category, input_text, normalized_body)
        candidates = alias_candidates.get(category, {}).get(normalized_body, [normalized_body])

        raw_type_phrase, match_source = extract_raw_type_phrase(input_text, raw_body, candidates)

        body_bucket = nested_stats[category].setdefault(
            normalized_body,
            {
                "total_rows": 0,
                "raw_type_phrase_counter": Counter(),
                "raw_type_phrase_examples": defaultdict(list),
                "cleaned_type_phrase_counter": Counter(),
                "cleaned_type_phrase_examples": defaultdict(list),
                "cleaned_type_phrase_raw_variants": defaultdict(Counter),
                "raw_body_counter": Counter(),
                "raw_body_examples": defaultdict(list),
            },
        )
        body_bucket["total_rows"] += 1
        body_bucket["raw_type_phrase_counter"][raw_type_phrase] += 1
        cleaned_type_phrase = clean_raw_type_phrase(raw_type_phrase, normalized_body)
        body_bucket["cleaned_type_phrase_counter"][cleaned_type_phrase] += 1
        body_bucket["cleaned_type_phrase_raw_variants"][cleaned_type_phrase][raw_type_phrase] += 1
        body_bucket["raw_body_counter"][str(raw_body or "").strip() or "<EMPTY>"] += 1

        if len(body_bucket["raw_type_phrase_examples"][raw_type_phrase]) < 3:
            body_bucket["raw_type_phrase_examples"][raw_type_phrase].append(input_text)
        if len(body_bucket["cleaned_type_phrase_examples"][cleaned_type_phrase]) < 3:
            body_bucket["cleaned_type_phrase_examples"][cleaned_type_phrase].append(input_text)
        raw_body_key = str(raw_body or "").strip() or "<EMPTY>"
        if len(body_bucket["raw_body_examples"][raw_body_key]) < 3:
            body_bucket["raw_body_examples"][raw_body_key].append(input_text)

        if match_source in {"__raw_body_fallback__", "__unmatched__"}:
            unresolved_expression_rows.append(
                {
                    "index": idx,
                    "category": category,
                    "normalized_body": normalized_body,
                    "raw_body": raw_body,
                    "raw_type_phrase": raw_type_phrase,
                    "match_source": match_source,
                    "input": input_text,
                }
            )

    finalized: dict[str, Any] = {"categories": {}}
    flat_rows: list[dict[str, Any]] = []

    for category, body_mapping in nested_stats.items():
        cat_out: dict[str, Any] = {}
        for normalized_body, info in body_mapping.items():
            total_rows = info["total_rows"]
            expressions = []
            for cleaned_type_phrase, count in info["cleaned_type_phrase_counter"].most_common():
                raw_variants = [
                    {"raw_type_phrase": raw, "count": raw_count}
                    for raw, raw_count in info["cleaned_type_phrase_raw_variants"][cleaned_type_phrase].most_common()
                ]
                expressions.append(
                    {
                        "raw_type_phrase_cleaned": cleaned_type_phrase,
                        "count": count,
                        "ratio_in_body": round(count / total_rows, 4) if total_rows else 0.0,
                        "raw_type_phrase_variants": raw_variants,
                        "examples": info["cleaned_type_phrase_examples"][cleaned_type_phrase],
                    }
                )
                flat_rows.append(
                    {
                        "category": category,
                        "normalized_body": normalized_body,
                        "raw_type_phrase": raw_variants[0]["raw_type_phrase"] if raw_variants else cleaned_type_phrase,
                        "raw_type_phrase_cleaned": cleaned_type_phrase,
                        "raw_type_phrase_variants": " | ".join(
                            f"{item['raw_type_phrase']} ({item['count']})" for item in raw_variants[:8]
                        ),
                        "count": count,
                        "ratio_in_body": round(count / total_rows, 4) if total_rows else 0.0,
                        "full_sample_1": info["cleaned_type_phrase_examples"][cleaned_type_phrase][0]
                        if len(info["cleaned_type_phrase_examples"][cleaned_type_phrase]) > 0 else "",
                        "full_sample_2": info["cleaned_type_phrase_examples"][cleaned_type_phrase][1]
                        if len(info["cleaned_type_phrase_examples"][cleaned_type_phrase]) > 1 else "",
                        "full_sample_3": info["cleaned_type_phrase_examples"][cleaned_type_phrase][2]
                        if len(info["cleaned_type_phrase_examples"][cleaned_type_phrase]) > 2 else "",
                    }
                )

            raw_bodies = [
                {
                    "raw_body": raw_body,
                    "count": count,
                    "examples": info["raw_body_examples"][raw_body],
                }
                for raw_body, count in info["raw_body_counter"].most_common()
            ]

            cat_out[normalized_body] = {
                "total_rows": total_rows,
                "distinct_raw_type_phrases": len(info["raw_type_phrase_counter"]),
                "distinct_cleaned_type_phrases": len(info["cleaned_type_phrase_counter"]),
                "distinct_raw_bodies": len(info["raw_body_counter"]),
                "raw_type_phrases": expressions,
                "raw_bodies": raw_bodies,
            }
        finalized["categories"][category] = cat_out

    flat_rows.sort(key=lambda x: (x["category"], x["normalized_body"], -x["count"], x["raw_type_phrase_cleaned"]))
    return finalized, flat_rows, uncategorized_rows + unresolved_expression_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "normalized_body",
        "raw_type_phrase",
        "raw_type_phrase_cleaned",
        "raw_type_phrase_variants",
        "count",
        "ratio_in_body",
        "full_sample_1",
        "full_sample_2",
        "full_sample_3",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计标准 BODY 对应的原始表达分布")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入数据集 JSON")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="分类映射 JSON")
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

    nested, flat_rows, unresolved = analyze_rows(rows, mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dump_json(args.output_dir / "body_raw_expression_distribution.json", nested)
    write_csv(args.output_dir / "body_raw_expression_distribution.csv", flat_rows)
    dump_json(args.output_dir / "body_raw_expression_unresolved.json", unresolved)

    print(f"输入总条数: {len(rows)}")
    print(f"分类数: {len(nested['categories'])}")
    print(f"平铺统计行数: {len(flat_rows)}")
    print(f"未分类或原始表达未命中条数: {len(unresolved)}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
