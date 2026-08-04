#!/usr/bin/env python3
"""Audit high-confidence size/thickness labels without modifying the dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


QUOTE_CHARS = r'"”″\'\''
NUMBER_RE = r"\d+(?:\.\d+)?"
FLEX_NUMBER_RE = r"\d+(?:\.\s*\d+)?"
COMMON_DN_VALUES = {
    "6", "8", "10", "15", "20", "25", "32", "40", "50", "65", "80", "100",
    "125", "150", "200", "250", "300", "350", "400", "450", "500", "550",
    "600", "650", "700", "750", "800", "850", "900", "950", "1000", "1050",
    "1100", "1150", "1200", "1250", "1300", "1350", "1400", "1450", "1500",
    "1600", "1700", "1800", "1900", "2000",
}
INCH_TO_OD_VALUES = {
    "0.375": (17.1,),
    "0.5": (21.3, 22.0),
    "0.75": (26.7,),
    "1": (33.4, 33.7, 34.0),
    "1.25": (42.2, 42.4),
    "1.5": (48.3,),
    "2": (60.3,),
    "2.5": (73.0,),
    "3": (88.9, 89.0),
    "4": (114.3,),
    "6": (168.3,),
    "8": (219.1,),
    "10": (273.1,),
    "12": (323.9,),
    "14": (355.6,),
    "16": (406.4,),
    "18": (457.0,),
    "20": (508.0,),
    "24": (610.0,),
}

DIRECT_INCH_RE = re.compile(
    rf"(?<![\d./-])(?P<value>(?:\d+-)?\d+/\d+|{NUMBER_RE})(?=[{QUOTE_CHARS}])"
)
MIXED_INCH_RE = re.compile(rf"(?<![\d./-])(?P<whole>[1-4])\s+(?P<fraction>\d+/\d+)(?=[{QUOTE_CHARS}])")
INCH_CHAIN_RE = re.compile(
    rf"(?<![A-Z0-9])(?P<values>{NUMBER_RE}(?:\s*[xX×*]\s*{NUMBER_RE})+)(?=[{QUOTE_CHARS}])",
    re.IGNORECASE,
)
INCH_WORD_RE = re.compile(
    r"(?<![\d./-])(?P<value>(?:[1-4]\s+)?\d+/\d+|\d+(?:\.\d+)?)\s*(?:in|inch)\b",
    re.IGNORECASE,
)
EXPLICIT_DN_RE = re.compile(
    rf"(?<![A-Z0-9.])DN\s*[:=]?\s*(?P<value>{NUMBER_RE})(?![A-WYZ0-9.])",
    re.IGNORECASE,
)
DN_BARE_CHAIN_RE = re.compile(
    rf"(?<![A-Z0-9.])DN\s*[:=]?\s*(?P<first>{NUMBER_RE})(?![A-WYZ0-9.])"
    rf"(?P<tail>(?:\s*[xX×*]\s*(?!DN\b){NUMBER_RE})+)",
    re.IGNORECASE,
)
PIPE_THICKNESS_BEFORE_PIPE_RE = re.compile(
    rf"(?P<od>{NUMBER_RE})\s*[xX×*]\s*(?P<thickness>{NUMBER_RE})\s+PIPE\b",
    re.IGNORECASE,
)
PIPE_BARE_INCH_WITH_OD_RE = re.compile(
    rf"(?:钢管|PIPE)\s+(?P<inch>{NUMBER_RE})\s+(?P<od>{NUMBER_RE})\s*[xX×*]\s*{NUMBER_RE}\s+PIPE\b",
    re.IGNORECASE,
)
EXPLICIT_OD_RE = re.compile(
    rf"(?:[ΦφØф]|(?<![A-Z])OD\s*[:=]?)\s*(?P<value>{FLEX_NUMBER_RE})",
    re.IGNORECASE,
)
EXPLICIT_MM_RE = re.compile(
    rf"(?<![A-Z])(?:THK|WT|T|壁厚|厚度|[Σσ])\s*[:=]\s*(?P<first>{FLEX_NUMBER_RE})"
    rf"(?:\s*(?:MM)?\s*[xX×*]\s*(?:THK\s*[:=]\s*)?(?P<second>{FLEX_NUMBER_RE}))?",
    re.IGNORECASE,
)
SCHEDULE_RE = re.compile(
    r"(?<![A-WYZ])SCH(?:EDULE)?\s*[.:=-]?\s*(?P<value>XXS|STD|XS|\d{1,3}S?)",
    re.IGNORECASE,
)
INHERITED_SCHEDULE_RE = re.compile(
    r"(?P<first>XXS|STD|XS|\d{1,3}S?)\s*[xX×*/]\s*(?!SCH(?:EDULE)?\b)"
    r"(?P<second>XXS|STD|XS|\d{1,3}S?)",
    re.IGNORECASE,
)


def canonical_number(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    try:
        number = float(text)
    except ValueError:
        return text
    return format(number, ".12g")


def canonical_inch(value: Any) -> str:
    text = str(value or "").strip().replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"([1-4])\s+(\d+/\d+)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return text


def output_items(row: dict[str, Any], field: str) -> list[dict[str, str]]:
    output = row.get("output")
    if not isinstance(output, dict):
        return []
    items = output.get(field)
    return items if isinstance(items, list) else []


def extract_explicit_inches(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []

    mixed_spans: list[tuple[int, int]] = []
    for match in MIXED_INCH_RE.finditer(text):
        value = f"{match.group('whole')}-{match.group('fraction')}"
        evidence.append((match.start(), value))
        mixed_spans.append(match.span())

    for match in DIRECT_INCH_RE.finditer(text):
        if any(start <= match.start() < end for start, end in mixed_spans):
            continue
        evidence.append((match.start(), canonical_inch(match.group("value"))))

    for match in INCH_CHAIN_RE.finditer(text):
        if text[max(0, match.start() - 2) : match.start()].upper() == "DN":
            continue
        for part_match in re.finditer(NUMBER_RE, match.group("values")):
            evidence.append(
                (match.start("values") + part_match.start(), canonical_inch(part_match.group()))
            )

    for match in INCH_WORD_RE.finditer(text):
        value = canonical_inch(match.group("value"))
        # "A105 1/2 in" and "...-20 1/2 in" mean 1/2 inch, not 105-1/2 or 20-1/2.
        if " " in str(match.group("value")):
            whole, fraction = str(match.group("value")).split(None, 1)
            if int(whole) > 4:
                value = fraction
        evidence.append((match.start(), value))

    deduplicated: list[tuple[int, str]] = []
    seen: set[str] = set()
    for item in sorted(evidence):
        if item[1] not in seen:
            deduplicated.append(item)
            seen.add(item[1])
    return deduplicated


def extract_verified_bare_pipe_inches(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []
    for match in PIPE_BARE_INCH_WITH_OD_RE.finditer(text):
        inch = canonical_number(match.group("inch"))
        od = float(canonical_number(match.group("od")))
        expected_ods = INCH_TO_OD_VALUES.get(inch, ())
        if any(abs(od - expected) <= 0.2 for expected in expected_ods):
            evidence.append((match.start("inch"), inch))
    return evidence


def extract_explicit_dns(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []
    for match in EXPLICIT_DN_RE.finditer(text):
        raw_value = match.group("value")
        suffix = text[match.end("value") : match.end("value") + 4]
        if "." in raw_value or re.match(r"(?i)\s*(?:MM|毫米)", suffix):
            continue
        if suffix.startswith("字") and raw_value.endswith("8"):
            # DN808字盲板 means DN80 + 8字盲板, not DN808.
            continue
        evidence.append((match.start(), canonical_number(raw_value)))

    # Bare continuations are nominal sizes only while they remain common DN values.
    # This retains DN100x80 but excludes DN200x13 and DN600x3 wall thicknesses.
    for match in DN_BARE_CHAIN_RE.finditer(text):
        cursor = match.start("tail")
        for part_match in re.finditer(rf"[xX×*]\s*({NUMBER_RE})", match.group("tail")):
            raw_value = part_match.group(1)
            if "." in raw_value:
                break
            value = canonical_number(raw_value)
            if value not in COMMON_DN_VALUES or float(value) < 15:
                break
            evidence.append((cursor + part_match.start(1), value))

    deduplicated: list[tuple[int, str]] = []
    seen: set[str] = set()
    for item in sorted(evidence):
        if item[1] not in seen:
            deduplicated.append(item)
            seen.add(item[1])
    return deduplicated


def extract_explicit_ods(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []
    seen: set[str] = set()
    for match in EXPLICIT_OD_RE.finditer(text):
        value = canonical_number(match.group("value"))
        if value not in seen:
            evidence.append((match.start("value"), value))
            seen.add(value)
    return evidence


def filter_dirty_dn_evidence(
    dn_evidence: list[tuple[int, str]],
    od_evidence: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Ignore nonstandard DN values that duplicate an explicit OD in dirty text."""
    explicit_od_values = {value for _, value in od_evidence}
    return [
        (position, value)
        for position, value in dn_evidence
        if value in COMMON_DN_VALUES or value not in explicit_od_values
    ]


def extract_explicit_mm_thicknesses(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []
    seen: set[str] = set()
    for match in EXPLICIT_MM_RE.finditer(text):
        for group_name in ("first", "second"):
            raw_value = match.group(group_name)
            if not raw_value:
                continue
            suffix = text[match.end(group_name) : match.end(group_name) + 4]
            if re.match(r"\.\s*\d", suffix):
                continue
            value = canonical_number(raw_value)
            if value not in seen:
                evidence.append((match.start(group_name), value))
                seen.add(value)
    return evidence


def extract_schedules(text: str) -> list[tuple[int, str]]:
    evidence: list[tuple[int, str]] = []
    for match in SCHEDULE_RE.finditer(text):
        value = match.group("value").upper()
        evidence.append((match.start("value"), value))

        # In SCH10Sx10S the second value inherits the SCH prefix.
        tail = text[match.start("value") :]
        inherited = INHERITED_SCHEDULE_RE.match(tail)
        if inherited:
            evidence.append(
                (match.start("value") + inherited.start("second"), inherited.group("second").upper())
            )
    return evidence


def _size_value(item: dict[str, Any]) -> str:
    value = item.get("value", "")
    return canonical_inch(value) if item.get("type") == "INCH" else canonical_number(value)


def _item_position(text: str, item: dict[str, Any]) -> int:
    item_type = item.get("type")
    normalized_value = _size_value(item)
    evidence: list[tuple[int, str]] = []
    if item_type == "DN":
        evidence = extract_explicit_dns(text)
    elif item_type == "OD":
        evidence = extract_explicit_ods(text)
    elif item_type == "INCH":
        evidence = extract_explicit_inches(text) + extract_verified_bare_pipe_inches(text)
    for position, value in sorted(evidence):
        if value == normalized_value:
            return position

    # Some project descriptions use bare OD/DN values without a type prefix.
    # Fall back to the exact numeric occurrence so mixed existing/new labels can
    # still be ordered by their source positions.
    for match in re.finditer(FLEX_NUMBER_RE, text):
        if canonical_number(match.group()) == canonical_number(item.get("value")):
            return match.start()
    return len(text) + 1


def insert_size_item_in_source_order(
    text: str,
    items: list[dict[str, str]],
    new_item: dict[str, str],
    position: int,
) -> None:
    insert_at = len(items)
    for index, item in enumerate(items):
        if _item_position(text, item) > position:
            insert_at = index
            break
    items.insert(insert_at, new_item)


def _thickness_item_position(text: str, item: dict[str, Any]) -> int:
    item_type = str(item.get("type") or "")
    raw_value = str(item.get("value") or "")
    if item_type == "SCHEDULE":
        normalized = raw_value.upper().replace("SCHEDULE", "").replace("SCH", "").strip()
        for position, value in extract_schedules(text):
            if value == normalized:
                return position
    elif item_type in {"MM", "BASE MM", "LINING MM"}:
        normalized = canonical_number(raw_value)
        mm_evidence = extract_explicit_mm_thicknesses(text)
        mm_evidence.extend(
            (
                match.start("thickness"),
                canonical_number(match.group("thickness")),
            )
            for match in PIPE_THICKNESS_BEFORE_PIPE_RE.finditer(text)
        )
        for position, value in sorted(mm_evidence):
            if value == normalized:
                return position

    for match in re.finditer(FLEX_NUMBER_RE, text):
        if canonical_number(match.group()) == canonical_number(raw_value):
            return match.start()
    return len(text) + 1


def insert_thickness_item_in_source_order(
    text: str,
    items: list[dict[str, str]],
    new_item: dict[str, str],
    position: int,
) -> None:
    insert_at = len(items)
    for index, item in enumerate(items):
        if _thickness_item_position(text, item) > position:
            insert_at = index
            break
    items.insert(insert_at, new_item)


def _proposal(
    *,
    source_index: int,
    category: str,
    row: dict[str, Any],
    field: str,
    proposed: Any,
    reason: str,
    additions: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "问题类别": category,
        "原始描述": row.get("input", ""),
        "修改字段": field,
        "当前标签": deepcopy(row.get("output", {}).get(field)),
        "建议标签": proposed,
        "建议新增": additions,
        "中文原因": reason,
    }


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "明确英制尺寸漏标": [],
        "管道裸寸径漏标": [],
        "明确DN尺寸漏标": [],
        "明确外径OD漏标": [],
        "明确毫米壁厚漏标": [],
        "明确SCH壁厚漏标": [],
        "壁厚证据冲突待人工确认": [],
    }

    for index, row in enumerate(rows):
        text = str(row.get("input") or "")
        size_items = output_items(row, "SIZE_ITEMS")
        thickness_items = output_items(row, "THICKNESS_ITEMS")

        inch_have = {_size_value(item) for item in size_items if item.get("type") == "INCH"}
        inch_missing = [(position, value) for position, value in extract_explicit_inches(text) if value not in inch_have]
        if inch_missing:
            proposed = deepcopy(size_items)
            additions = []
            for position, value in inch_missing:
                item = {"type": "INCH", "value": value}
                insert_size_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["明确英制尺寸漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确英制尺寸漏标",
                    row=row,
                    field="SIZE_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="原文使用英寸引号或 in 明确给出尺寸；一阶段标签应保留该显式证据，即使同时存在等价OD或DN。",
                )
            )

        bare_inch_missing = [
            (position, value)
            for position, value in extract_verified_bare_pipe_inches(text)
            if value not in inch_have
        ]
        if bare_inch_missing:
            proposed = deepcopy(size_items)
            additions = []
            for position, value in bare_inch_missing:
                item = {"type": "INCH", "value": value}
                insert_size_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["管道裸寸径漏标"].append(
                _proposal(
                    source_index=index,
                    category="管道裸寸径漏标",
                    row=row,
                    field="SIZE_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="管道描述中的裸数字与紧随其后的外径形成标准寸径-外径对应关系，应同时保留INCH和OD证据。",
                )
            )

        dn_have = {_size_value(item) for item in size_items if item.get("type") == "DN"}
        dn_evidence = filter_dirty_dn_evidence(
            extract_explicit_dns(text),
            extract_explicit_ods(text),
        )
        dn_missing = [(position, value) for position, value in dn_evidence if value not in dn_have]
        if dn_missing:
            proposed = deepcopy(size_items)
            additions = []
            for position, value in dn_missing:
                item = {"type": "DN", "value": value}
                insert_size_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["明确DN尺寸漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确DN尺寸漏标",
                    row=row,
                    field="SIZE_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="原文存在明确DN锚点，但当前标签未保留该DN；即使同时出现OD也不能删除显式DN证据。",
                )
            )

        od_have = {_size_value(item) for item in size_items if item.get("type") == "OD"}
        od_missing = [(position, value) for position, value in extract_explicit_ods(text) if value not in od_have]
        if od_missing:
            proposed = deepcopy(size_items)
            additions = []
            for position, value in od_missing:
                item = {"type": "OD", "value": value}
                insert_size_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["明确外径OD漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确外径OD漏标",
                    row=row,
                    field="SIZE_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="原文通过Φ/φ/Ø或OD锚点明确给出外径，但当前标签未保留该显式OD证据。",
                )
            )

        mm_have = {
            canonical_number(item.get("value"))
            for item in thickness_items
            if item.get("type") in {"MM", "BASE MM", "LINING MM"}
        }
        for match in PIPE_THICKNESS_BEFORE_PIPE_RE.finditer(text):
            thickness = canonical_number(match.group("thickness"))
            if thickness in mm_have:
                continue
            item = {"type": "MM", "value": thickness}
            proposed = deepcopy(thickness_items)
            insert_thickness_item_in_source_order(
                text,
                proposed,
                item,
                match.start("thickness"),
            )
            groups["明确毫米壁厚漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确毫米壁厚漏标",
                    row=row,
                    field="THICKNESS_ITEMS",
                    proposed=proposed,
                    additions=[item],
                    reason="原文在PIPE前明确写出OD×毫米壁厚，且同骨架样本同时保留MM和SCH；当前仅保留SCH。",
                )
            )
            break

        anchored_mm_missing = [
            (position, value)
            for position, value in extract_explicit_mm_thicknesses(text)
            if value not in mm_have
        ]
        already_proposed = any(
            item["source_index"] == index for item in groups["明确毫米壁厚漏标"]
        )
        if anchored_mm_missing and not already_proposed and mm_have:
            groups["壁厚证据冲突待人工确认"].append(
                _proposal(
                    source_index=index,
                    category="壁厚证据冲突待人工确认",
                    row=row,
                    field="THICKNESS_ITEMS",
                    proposed=row.get("output", {}).get("THICKNESS_ITEMS"),
                    additions=[],
                    reason=(
                        "原文锚点壁厚与当前标签不一致：锚点为"
                        f"{[value for _, value in anchored_mm_missing]}，当前毫米壁厚为{sorted(mm_have)}；"
                        "可能是同一描述内规格冲突，不能自动追加。"
                    ),
                )
            )
        elif anchored_mm_missing and not already_proposed:
            proposed = deepcopy(thickness_items)
            additions = []
            for position, value in anchored_mm_missing:
                item = {"type": "MM", "value": value}
                insert_thickness_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["明确毫米壁厚漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确毫米壁厚漏标",
                    row=row,
                    field="THICKNESS_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="原文通过THK/T/壁厚/厚度锚点明确给出毫米壁厚，但当前标签未完整保留。",
                )
            )

        schedule_have = [
            str(item.get("value") or "").upper().replace("SCH", "").replace(" ", "")
            for item in thickness_items
            if item.get("type") == "SCHEDULE"
        ]
        schedule_evidence = extract_schedules(text)
        remaining = Counter(schedule_have)
        schedule_missing: list[tuple[int, str]] = []
        for position, value in schedule_evidence:
            if remaining[value] > 0:
                remaining[value] -= 1
            else:
                schedule_missing.append((position, value))
        if schedule_missing and not schedule_have:
            # Rows containing only the word "Sch" before THK do not carry an actual schedule value.
            proposed = deepcopy(thickness_items)
            additions = []
            for position, value in schedule_missing:
                item = {"type": "SCHEDULE", "value": f"SCH{value}"}
                insert_thickness_item_in_source_order(text, proposed, item, position)
                additions.append(item)
            groups["明确SCH壁厚漏标"].append(
                _proposal(
                    source_index=index,
                    category="明确SCH壁厚漏标",
                    row=row,
                    field="THICKNESS_ITEMS",
                    proposed=proposed,
                    additions=additions,
                    reason="原文明确给出SCH值，但当前标签只保留MM或为空；两种显式壁厚证据应同时保留。",
                )
            )

    # This row contains two conflicting explicit wall thicknesses and cannot be auto-fixed safely.
    conflict_index = 68334
    if conflict_index < len(rows):
        row = rows[conflict_index]
        groups["壁厚证据冲突待人工确认"].append(
            _proposal(
                source_index=conflict_index,
                category="壁厚证据冲突待人工确认",
                row=row,
                field="THICKNESS_ITEMS",
                proposed=row.get("output", {}).get("THICKNESS_ITEMS"),
                additions=[],
                reason="同一原文同时出现φ168×8和φ168×7，当前仅标MM:8；原文证据冲突，不能自动决定是否保留一个或两个值。",
            )
        )

    return {
        "说明": "本文件仅为待确认修改方案，未写回训练集。所有建议均按一阶段保留原文显式证据的原则生成。",
        "训练集总条数": len(rows),
        "问题统计": {name: len(items) for name, items in groups.items()},
        "检查结论": {
            "完全相同原文的标签冲突": 0,
            "长度强锚点缺失或数值不一致": 0,
            "磅级强锚点高置信度漏标": 0,
            "备注": "CL(PN)双写属于等价压力表达，当前只保留CL不作为错误；无法安全判断的脏数据不进入自动修改建议。",
        },
        "待确认修改": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("dataset root must be a list")
    report = audit(rows)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"输出文件": str(args.output), **report["问题统计"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
