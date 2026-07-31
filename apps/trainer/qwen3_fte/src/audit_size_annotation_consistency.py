#!/usr/bin/env python3
"""Audit size annotations without changing the source dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    QWEN_ROOT
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "尺寸壁厚磅级C1训练集.json"
)
DEFAULT_JSON_REPORT = DEFAULT_DATASET.with_name("尺寸标注一致性审计报告.json")
DEFAULT_MD_REPORT = DEFAULT_DATASET.with_name("尺寸标注一致性审计报告.md")

NUMBER = r"\d+(?:\.\s*\d+)?"
OUTPUT_NUMBER = r"\d+(?:\.\d+)?"
DN_VALUE_RE = re.compile(rf"(?i)DN\s*(?P<value>{NUMBER})")
DN_TEXT_RE = re.compile(
    r"公称(?:直径|通径|口径)|NOMINAL\s+(?:DIAMETER|BORE)",
)
RADIUS_DN_RE = re.compile(rf"(?i)R\s*=\s*{NUMBER}\s*DN")
RADIUS_DN_MATERIAL_RE = re.compile(
    rf"(?i)R\s*=\s*{NUMBER}\s*DN\s*(?P<material>\d+)\s*#",
)
EXPLICIT_OD_RE = re.compile(
    rf"(?i)(?:"
    rf"(?<![A-Z])OD\s*[:：=]?\s*(?P<od>{NUMBER})|"
    rf"(?:外径|管外径)\s*[:：=]?\s*(?P<outer>{NUMBER})|"
    rf"[ΦφØ]\s*(?P<phi>{NUMBER})"
    rf")"
)
EXPLICIT_INCH_RE = re.compile(
    rf"(?<![\d.]){NUMBER}\s*(?:\"|″|”|“|英寸)|"
    rf"(?<![A-Z0-9])NPS\s*{NUMBER}",
    re.IGNORECASE,
)
INCH_VALUE_RE = re.compile(
    rf"(?i)(?<![\d.])(?P<quote>{NUMBER})\s*(?:\"|″|”|“|英寸)|"
    rf"(?<![A-Z0-9])NPS\s*(?P<nps>{NUMBER})",
)
EXPLICIT_D_VALUE_RE = re.compile(rf"(?i)(?<![A-Z0-9])D\s*(?P<value>{NUMBER})")
D_COMPOSITE_RE = re.compile(
    r"(?i)(?<![A-Z0-9])D\s*"
    r"(?P<values>\d+(?:\s*[xX×*]\s*\d+){1,3})",
)
OCR_DN_COMPOSITE_RE = re.compile(
    r"(?i)DN(?P<values>[0-9ILO]+(?:\s*[xX×*]\s*[0-9ILO]+){1,3})",
)
EMBEDDED_ELBOW_DN_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(?:W|S)?\d+(?:\.\d+)?(?:EL|ES|E)"
    r"(?P<value>\d+)(?=\s*[-_–—/])",
)
OCR_DK_VALUE_RE = re.compile(
    r"(?i)(?<![A-Z0-9])DK\s*(?P<value>\d+)(?![\d.])",
)
BARE_COMMON_DN_COMPOSITE_RE = re.compile(
    r"(?i)(?<![\d.])"
    r"(?P<values>\d+(?:\s*[xX×*]\s*\d+){1,3})"
    r"(?![\d.])",
)
COMMON_INTEGER_RE = re.compile(
    r"(?<![\d.])(?P<value>\d+)(?![\d.])",
)
PIPE_METRIC_OD_RE = re.compile(
    rf"(?i)(?<![A-Z])(?:SMLS\s*)?PIPE\s*"
    rf"(?P<od>{NUMBER})"
    rf"(?=\s*(?:"
    rf"SCH|S\s*[-_]?\s*\d|STD\b|XS\b|XXS\b|"
    rf"[xX×*]\s*{NUMBER}(?:\s*mm\b)?"
    rf"))"
)
SH_T_3405_RE = re.compile(r"(?i)SH\s*/\s*T\s*3405")
MIN_STRONG_METRIC_OD_MM = 13.0
COMMON_DN_VALUES = {
    6,
    8,
    10,
    15,
    20,
    25,
    32,
    40,
    50,
    65,
    80,
    100,
    125,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
    1000,
    1050,
    1100,
    1150,
    1200,
    1300,
    1400,
    1500,
    1600,
    1800,
    2000,
}
OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "I": "1",
        "L": "1",
        "O": "0",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计尺寸壁厚磅级数据集的尺寸标注")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--md-report", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument(
        "--fix-pipe-od-missing-with-explicit-dn",
        action="store_true",
        help="仅为明确 PIPE 外径和 DN 并存但漏标 OD 的样本补充 OD",
    )
    parser.add_argument(
        "--fix-pipe-od-missing-with-explicit-inch",
        action="store_true",
        help="仅为明确 PIPE 外径和英寸并存但漏标 OD 的样本补充 OD",
    )
    return parser.parse_args()


def normalize_number(value: str) -> str:
    numeric = float(re.sub(r"\s+", "", value))
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def output_size_items(row: dict[str, Any]) -> list[dict[str, str]]:
    output = row.get("output")
    if not isinstance(output, dict):
        return []
    items = output.get("SIZE_ITEMS")
    if not isinstance(items, list):
        return []
    return [
        {
            "type": str(item.get("type") or "").upper(),
            "value": str(item.get("value") or ""),
        }
        for item in items
        if isinstance(item, dict)
    ]


def values_by_type(items: list[dict[str, str]], item_type: str) -> set[str]:
    values: set[str] = set()
    for item in items:
        if item["type"] != item_type:
            continue
        try:
            values.add(normalize_number(item["value"]))
        except ValueError:
            values.add(item["value"])
    return values


def explicit_od_values(text: str) -> list[str]:
    values: list[str] = []
    for match in EXPLICIT_OD_RE.finditer(text):
        raw = match.group("od") or match.group("outer") or match.group("phi")
        values.append(normalize_number(raw))
    return values


def strong_pipe_metric_od_values(text: str) -> list[str]:
    if not SH_T_3405_RE.search(text):
        return []
    values: list[str] = []
    for match in PIPE_METRIC_OD_RE.finditer(text):
        value = normalize_number(match.group("od"))
        if float(value) >= MIN_STRONG_METRIC_OD_MM:
            values.append(value)
    return values


def explicit_inch_values(text: str) -> list[str]:
    values: list[str] = []
    for match in INCH_VALUE_RE.finditer(text):
        raw = match.group("quote") or match.group("nps")
        values.append(normalize_number(raw))
    return values


def normalize_ocr_integer(value: str) -> str | None:
    normalized = value.upper().translate(OCR_DIGIT_TRANSLATION)
    if not normalized.isdigit():
        return None
    numeric = int(normalized)
    if numeric not in COMMON_DN_VALUES:
        return None
    return str(numeric)


def implied_dn_values(text: str) -> set[str]:
    values: set[str] = set()

    # D100 and D100x50 both use D as the nominal-size prefix. For a composite
    # size, require every segment to be a known DN to avoid treating thickness
    # in D114.3x6.02 as another nominal diameter.
    for match in EXPLICIT_D_VALUE_RE.finditer(text):
        values.add(normalize_number(match.group("value")))
    for match in D_COMPOSITE_RE.finditer(text):
        parts = re.split(r"\s*[xX×*]\s*", match.group("values"))
        numeric_parts = [int(part) for part in parts]
        if all(part in COMMON_DN_VALUES for part in numeric_parts):
            values.update(str(part) for part in numeric_parts)

    # OCR frequently turns DN150x100 into DNI50xl00.
    for match in OCR_DN_COMPOSITE_RE.finditer(text):
        parts = re.split(r"\s*[xX×*]\s*", match.group("values"))
        normalized_parts = [normalize_ocr_integer(part) for part in parts]
        if all(part is not None for part in normalized_parts):
            values.update(part for part in normalized_parts if part is not None)

    # Product codes such as 90EL20-II-2.5 carry DN20 after the elbow code.
    for match in EMBEDDED_ELBOW_DN_RE.finditer(text):
        numeric = int(match.group("value"))
        if numeric in COMMON_DN_VALUES:
            values.add(str(numeric))

    # DK500 is a recurring OCR corruption of DN500 in project descriptions.
    for match in OCR_DK_VALUE_RE.finditer(text):
        numeric = int(match.group("value"))
        if numeric in COMMON_DN_VALUES:
            values.add(str(numeric))

    # Pure numeric composites are also used as nominal fitting sizes, including
    # embedded forms such as WOL250x50 and terminal forms such as 80x65.
    for match in BARE_COMMON_DN_COMPOSITE_RE.finditer(text):
        parts = re.split(r"\s*[xX×*]\s*", match.group("values"))
        numeric_parts = [int(part) for part in parts]
        if all(part in COMMON_DN_VALUES for part in numeric_parts):
            values.update(str(part) for part in numeric_parts)

    # A common DN may be written as an unprefixed integer or attached to a
    # product abbreviation, such as TS80, WOL250, SW40, or a terminal 150.
    common_integer_text = INCH_VALUE_RE.sub(" ", text)
    common_integer_text = RADIUS_DN_MATERIAL_RE.sub(" ", common_integer_text)
    for match in COMMON_INTEGER_RE.finditer(common_integer_text):
        numeric = int(match.group("value"))
        if numeric in COMMON_DN_VALUES:
            values.add(str(numeric))
    return values


def has_dn_semantics(text: str) -> bool:
    # Remove radius multipliers such as R=1.5DN before looking for a DN size.
    without_radius_dn = RADIUS_DN_RE.sub("", text)
    return bool(DN_VALUE_RE.search(without_radius_dn) or DN_TEXT_RE.search(text))


def make_issue(
    category: str,
    source_index: int,
    text: str,
    size_items: list[dict[str, str]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": category,
        "source_index": source_index,
        "input": text,
        "current_size_items": size_items,
        "evidence": evidence,
    }


def audit_rows(rows: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("input") or "")
        items = output_size_items(row)
        od_labels = values_by_type(items, "OD")
        dn_labels = values_by_type(items, "DN")
        anchored_od = explicit_od_values(text)
        pipe_od = strong_pipe_metric_od_values(text)

        missing_anchored_od = [value for value in anchored_od if value not in od_labels]
        if missing_anchored_od:
            issues.append(
                make_issue(
                    "explicit_od_missing",
                    source_index,
                    text,
                    items,
                    {"missing_od_values": missing_anchored_od},
                ),
            )

        missing_pipe_od = [value for value in pipe_od if value not in od_labels]
        if missing_pipe_od:
            has_explicit_dn = has_dn_semantics(text)
            has_explicit_inch = bool(EXPLICIT_INCH_RE.search(text))
            if has_explicit_dn:
                category = "pipe_od_missing_with_explicit_dn"
            elif has_explicit_inch:
                category = "pipe_od_missing_with_explicit_inch"
            elif dn_labels:
                category = "pipe_od_mislabeled_as_dn"
            else:
                category = "pipe_od_missing"
            issues.append(
                make_issue(
                    category,
                    source_index,
                    text,
                    items,
                    {
                        "missing_od_values": missing_pipe_od,
                        "has_explicit_dn_semantics": has_explicit_dn,
                        "has_explicit_inch_semantics": has_explicit_inch,
                    },
                ),
            )

        fractional_dn = [
            value
            for value in dn_labels
            if re.fullmatch(OUTPUT_NUMBER, value) and not float(value).is_integer()
        ]
        if fractional_dn:
            issues.append(
                make_issue(
                    "fractional_dn_label",
                    source_index,
                    text,
                    items,
                    {"fractional_dn_values": fractional_dn},
                ),
            )

        if dn_labels and not has_dn_semantics(text):
            implicit_dn = implied_dn_values(text)
            unexplained_dn = [
                value
                for value in sorted(dn_labels, key=lambda item: float(item))
                if value not in implicit_dn
            ]
            if unexplained_dn:
                evidence_kinds: list[str] = []
                if anchored_od:
                    evidence_kinds.append("explicit_od")
                if pipe_od:
                    evidence_kinds.append("pipe_metric_od")
                if EXPLICIT_INCH_RE.search(text):
                    evidence_kinds.append("explicit_inch")
                inch_values = explicit_inch_values(text)
                radius_material_values = [
                    normalize_number(match.group("material"))
                    for match in RADIUS_DN_MATERIAL_RE.finditer(text)
                ]
                if any(value in radius_material_values for value in unexplained_dn):
                    category = "radius_dn_material_mislabeled_as_dn"
                elif any(value in inch_values for value in unexplained_dn):
                    category = "inch_mislabeled_as_dn"
                elif pipe_od:
                    # This direct PIPE OD -> DN conflict is already reported above.
                    continue
                else:
                    category = "dn_without_dn_semantics"
                issues.append(
                    make_issue(
                        category,
                        source_index,
                        text,
                        items,
                        {
                            "unexplained_dn_values": unexplained_dn,
                            "conflicting_size_evidence": evidence_kinds,
                            "explicit_inch_values": inch_values,
                            "radius_material_values": radius_material_values,
                        },
                    ),
                )
    return issues


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fix_pipe_od_missing_with_explicit_dn(
    rows: list[Any],
    issues: list[dict[str, Any]],
) -> int:
    fixed_rows = 0
    for issue in issues:
        if issue["category"] != "pipe_od_missing_with_explicit_dn":
            continue
        source_index = issue["source_index"]
        missing_values = issue["evidence"]["missing_od_values"]
        if len(missing_values) != 1:
            raise ValueError(
                f"数据索引 {source_index} 的待补 OD 数量不是 1：{missing_values}",
            )
        row = rows[source_index]
        size_items = row["output"]["SIZE_ITEMS"]
        existing_od = values_by_type(output_size_items(row), "OD")
        additions = [
            {"type": "OD", "value": value}
            for value in missing_values
            if value not in existing_od
        ]
        if not additions:
            continue

        text = str(row.get("input") or "")
        missing_value = missing_values[0]
        od_positions = [
            match.start()
            for match in PIPE_METRIC_OD_RE.finditer(text)
            if normalize_number(match.group("od")) == missing_value
        ]
        dn_values = values_by_type(output_size_items(row), "DN")
        dn_positions = [
            match.start()
            for match in DN_VALUE_RE.finditer(RADIUS_DN_RE.sub("", text))
            if normalize_number(match.group("value")) in dn_values
        ]
        dn_item_indexes = [
            index
            for index, item in enumerate(size_items)
            if str(item.get("type") or "").upper() == "DN"
        ]
        if not od_positions or not dn_positions or not dn_item_indexes:
            raise ValueError(
                f"数据索引 {source_index} 无法按原文确定 OD 与 DN 顺序",
            )
        if min(od_positions) < min(dn_positions):
            insert_at = min(dn_item_indexes)
        else:
            insert_at = max(dn_item_indexes) + 1
        size_items[insert_at:insert_at] = additions
        fixed_rows += 1
    return fixed_rows


def fix_pipe_od_missing_with_explicit_inch(
    rows: list[Any],
    issues: list[dict[str, Any]],
) -> int:
    fixed_rows = 0
    for issue in issues:
        if issue["category"] != "pipe_od_missing_with_explicit_inch":
            continue
        source_index = issue["source_index"]
        missing_values = issue["evidence"]["missing_od_values"]
        if len(missing_values) != 1:
            raise ValueError(
                f"数据索引 {source_index} 的待补 OD 数量不是 1：{missing_values}",
            )
        row = rows[source_index]
        size_items = row["output"]["SIZE_ITEMS"]
        existing_od = values_by_type(output_size_items(row), "OD")
        additions = [
            {"type": "OD", "value": value}
            for value in missing_values
            if value not in existing_od
        ]
        if not additions:
            continue

        text = str(row.get("input") or "")
        missing_value = missing_values[0]
        od_positions = [
            match.start()
            for match in PIPE_METRIC_OD_RE.finditer(text)
            if normalize_number(match.group("od")) == missing_value
        ]
        inch_values = values_by_type(output_size_items(row), "INCH")
        inch_positions = []
        for match in INCH_VALUE_RE.finditer(text):
            raw = match.group("quote") or match.group("nps")
            if normalize_number(raw) in inch_values:
                inch_positions.append(match.start())
        inch_item_indexes = [
            index
            for index, item in enumerate(size_items)
            if str(item.get("type") or "").upper() == "INCH"
        ]
        if not od_positions or not inch_positions or not inch_item_indexes:
            raise ValueError(
                f"数据索引 {source_index} 无法按原文确定 OD 与 INCH 顺序",
            )
        if min(od_positions) < min(inch_positions):
            insert_at = min(inch_item_indexes)
        else:
            insert_at = max(inch_item_indexes) + 1
        size_items[insert_at:insert_at] = additions
        fixed_rows += 1
    return fixed_rows


def render_markdown(
    dataset_path: Path,
    rows_count: int,
    issues: list[dict[str, Any]],
) -> str:
    counts = Counter(issue["category"] for issue in issues)
    labels = {
        "explicit_od_missing": "带 OD/Φ/外径锚点但 OD 漏标",
        "pipe_od_missing_with_explicit_dn": "PIPE 公制外径漏标（同时存在明确 DN）",
        "pipe_od_missing_with_explicit_inch": "PIPE 公制外径漏标（同时存在明确英寸）",
        "pipe_od_mislabeled_as_dn": "PIPE 公制外径被直接标成 DN",
        "pipe_od_missing": "PIPE 公制外径漏标且未输出 DN",
        "fractional_dn_label": "小数被标为 DN",
        "radius_dn_material_mislabeled_as_dn": "R=倍数DN 后的材质数字被标成 DN",
        "inch_mislabeled_as_dn": "明确英寸值被标成同值 DN",
        "dn_without_dn_semantics": "没有 DN 语义却输出 DN（疑似）",
    }
    lines = [
        "# 尺寸标注一致性审计报告",
        "",
        f"- 数据集：`{dataset_path}`",
        f"- 总样本：{rows_count}",
        f"- 问题记录：{len(issues)}（同一描述可命中多类）",
        "",
        "## 分类统计",
        "",
        "| 分类 | 数量 |",
        "|---|---:|",
    ]
    for category, label in labels.items():
        lines.append(f"| {label} | {counts.get(category, 0)} |")

    for category, label in labels.items():
        category_issues = [issue for issue in issues if issue["category"] == category]
        lines.extend(["", f"## {label}", ""])
        if not category_issues:
            lines.append("无。")
            continue
        for index, issue in enumerate(category_issues, 1):
            lines.extend(
                [
                    f"### {index}. 数据索引 {issue['source_index']}",
                    "",
                    f"- 描述：`{issue['input']}`",
                    f"- 当前标注：`{json.dumps(issue['current_size_items'], ensure_ascii=False)}`",
                    f"- 证据：`{json.dumps(issue['evidence'], ensure_ascii=False)}`",
                    "",
                ],
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    json_report_path = args.json_report.expanduser().resolve()
    md_report_path = args.md_report.expanduser().resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("尺寸壁厚磅级数据集顶层必须是数组")

    issues = audit_rows(rows)
    fixed_dn_rows = 0
    fixed_inch_rows = 0
    if args.fix_pipe_od_missing_with_explicit_dn:
        fixed_dn_rows = fix_pipe_od_missing_with_explicit_dn(rows, issues)
    if args.fix_pipe_od_missing_with_explicit_inch:
        fixed_inch_rows = fix_pipe_od_missing_with_explicit_inch(rows, issues)
    if fixed_dn_rows or fixed_inch_rows:
        write_json(dataset_path, rows)
    if (
        args.fix_pipe_od_missing_with_explicit_dn
        or args.fix_pipe_od_missing_with_explicit_inch
    ):
        issues = audit_rows(rows)
    counts = Counter(issue["category"] for issue in issues)
    issues_by_category: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        issues_by_category.setdefault(issue["category"], []).append(issue)
    report = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "issue_records": len(issues),
        "unique_issue_rows": len({issue["source_index"] for issue in issues}),
        "category_counts": dict(counts),
        "issues_by_category": issues_by_category,
    }
    if args.fix_pipe_od_missing_with_explicit_dn:
        report["fixed_pipe_od_missing_with_explicit_dn_rows"] = fixed_dn_rows
    if args.fix_pipe_od_missing_with_explicit_inch:
        report["fixed_pipe_od_missing_with_explicit_inch_rows"] = fixed_inch_rows
    write_json(json_report_path, report)
    md_report_path.write_text(
        render_markdown(dataset_path, len(rows), issues),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "issues_by_category"
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
