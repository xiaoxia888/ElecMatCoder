#!/usr/bin/env python3
"""Propose source-grounded fixes for unresolved material aliases.

This script audits only the review group named
``原文中未找到可确认的材质别名``. It never modifies the source datasets.
The old normalized material code is retained for comparison, but is not used
as proof of a material designation.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.trainer.qwen3_fte.src.convert_material_to_part_structure import (
    DEFAULT_GROUPED_REVIEW_OUTPUT,
    DEFAULT_TRAIN_SOURCE,
    DEFAULT_VAL_SOURCE,
    canonicalize_material_class,
    canonicalize_material_grade,
    extract_grade_from_standard_context,
    extract_special_requirements,
    find_material_standards,
)


REVIEW_REASON = "原文中未找到可确认的材质别名"
DEFAULT_OUTPUT_DIR = DEFAULT_GROUPED_REVIEW_OUTPUT.parent / "待复核分析"
DEFAULT_PROPOSALS = DEFAULT_OUTPUT_DIR / "原文未找到材质别名_逐条处理建议.json"
DEFAULT_GROUPED = DEFAULT_OUTPUT_DIR / "原文未找到材质别名_处理建议_按结论分组.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "原文未找到材质别名_处理建议报告.json"

STRONG_OCR_CORRECTIONS: tuple[tuple[str, str], ...] = (
    ("22Cr17Ni12Mo2", "022Cr17Ni12Mo2"),
    ("022Cr17Ni12M", "022Cr17Ni12Mo2"),
    ("06Crl9Nil0", "06Cr19Ni10"),
    ("0Crl8Ni9", "0Cr18Ni9"),
    ("06Cr17Nil2Mo2", "06Cr17Ni12Mo2"),
    ("S304O8", "S30408"),
    ("2OG", "20G"),
    ("15CrMoGB/T", "15CrMo"),
    ("12CrMoVG", "12Cr1MoVG"),
)

INCOMPLETE_OCR_CORRECTIONS: tuple[tuple[str, str], ...] = (
    ("022Cr17Ni12", "022Cr17Ni12Mo2"),
)

DIRECT_GRADE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("JIS牌号", re.compile(r"(?i)\bSUS[\s._-]*F[\s._-]*(?:304L?|316L?|321H?)\b")),
    ("UNS牌号", re.compile(r"(?i)\b(?:UNS[\s._-]*)?[NS]\d{5}\b")),
    (
        "国标不锈钢牌号",
        re.compile(
            r"(?i)(?<![A-Z0-9])(?:"
            r"S\d{5}|"
            r"\d{1,3}Cr[0-9A-Za-z-]+|"
            r"X\d+Cr[0-9A-Za-z-]+"
            r")(?![A-Z0-9])"
        ),
    ),
    (
        "锻件牌号",
        re.compile(
            r"(?i)(?<![A-Z0-9])SF(?:304L?|316L?|321H?|310|2205)(?![A-Z0-9])"
        ),
    ),
    (
        "常用合金牌号",
        re.compile(
            r"(?i)(?<![A-Z0-9])(?:"
            r"15CrMoG?(?!B/T)|12Cr1MoVG?|16MnD?|LF485K2|CF415K?|"
            r"Q(?:235|245|345)[A-Z]?|L245M?|X65Q?|TA10|"
            r"F321H|SUS[\s._-]*F[\s._-]*316"
            r")(?![A-Z0-9])"
        ),
    ),
    (
        "欧洲数字牌号",
        re.compile(r"(?<![0-9])1\.(?:4301|4307|4401|4404|4541|4571)(?![0-9])"),
    ),
)

ASTM_GRADELESS_STANDARDS = {
    "ASTM A53",
    "ASTM A105",
    "ASTM A106",
    "ASTM A234",
    "ASTM A312",
}

SPECIAL_GRADE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ASTM A790",
        re.compile(r"(?i)(?:ASTM[\s._/-]*)?A[\s._/-]*790[\s,._/-]*(S\d{5})"),
    ),
    (
        "ASTM B444",
        re.compile(
            r"(?i)(?:ASTM[\s._/-]*)?B[\s._/-]*444"
            r"(?:[\s,._/-]*(?:GRADE|GR)\.?)?[\s,._/-]*(N\d{5})"
        ),
    ),
    (
        "ASTM A358",
        re.compile(
            r"(?i)(?:ASTM[\s._/-]*)?A[\s._/-]*358"
            r"(?:[\s,._/-]*(?:GRADE|GR)\.?)?[\s,._/-]*"
            r"((?:TP)?(?:304L?|316L?|316TI|310H))"
        ),
    ),
)


@dataclass(frozen=True)
class Proposal:
    confidence: str
    rule: str
    reason: str
    material: list[dict[str, Any]] | None
    relation: str
    evidence: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_GROUPED_REVIEW_OUTPUT)
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--grouped", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return text.replace("Ǫ", "Q").replace("Ｏ", "O")


def clean_grade(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,;._-/")
    value = re.sub(r"(?i)^SUS[\s._-]*F[\s._-]*", "SUS F", value)
    return value


def material_item(
    grade: str,
    *,
    standard: str = "",
    material_class: str = "",
    special: list[str] | None = None,
    part: str = "BODY",
) -> dict[str, Any]:
    canonical_class = canonicalize_material_class(material_class)
    return {
        "PART": part,
        "STANDARD": standard,
        "GRADE": canonicalize_material_grade(
            standard,
            clean_grade(grade),
            canonical_class,
        ),
        "CLASS": canonical_class,
        "SPECIAL_REQ": list(dict.fromkeys(special or [])),
    }


def extract_material_class(text: str, start: int, end: int) -> str:
    del start
    tail = text[end : min(len(text), end + 24)]
    match = re.match(
        r"(?i)\s*(?:\bGR(?:ADE)?\.?\s*)?"
        r"(III|II|I|Ⅲ|Ⅱ|Ⅰ)(?![A-Z0-9])",
        tail,
    )
    if not match:
        return ""
    return {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}.get(
        match.group(1),
        match.group(1).upper(),
    )


def explicit_astm_proposal(text: str) -> Proposal | None:
    standards = find_material_standards(text)
    if re.search(r"(?i)STD[\s._/-]*A[\s._/-]*53\b", text):
        standards = ["ASTM A53", *standards]
    if not standards:
        return None

    items: list[dict[str, Any]] = []
    evidence: list[str] = []
    specials = extract_special_requirements(text)
    for standard in standards:
        grade = ""
        for pattern_standard, pattern in SPECIAL_GRADE_PATTERNS:
            if pattern_standard != standard:
                continue
            match = pattern.search(text)
            if match:
                grade = match.group(1).upper()
                break
        if not grade:
            grade = extract_grade_from_standard_context(text, standard)
        if standard == "ASTM A53" and not grade:
            match = re.search(
                r"(?i)STD[\s._/-]*A[\s._/-]*53"
                r"[\s,._/-]*(?:GR(?:ADE)?\.?\s*)?([AB])\b",
                text,
            )
            if match:
                grade = match.group(1).upper()
        if grade:
            grade = re.sub(r"(?i)(?:SMLS|EFW|ERW)$", "", grade)
            grade = re.sub(r"(?i)(?:[-\s]+)(?:WX|WU|W|S)$", "", grade)
            grade = re.sub(r"(?i)\bCL(?:ASS)?\.?\s*\d+\b", "", grade)
            grade = clean_grade(grade)
            items.append(material_item(grade, standard=standard, special=specials))
            evidence.append(f"{standard} {grade}".strip())
        else:
            items.append(material_item("", standard=standard, special=specials))
            evidence.append(standard)

    if not items:
        return None

    unique: list[dict[str, Any]] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    items = unique

    if len(items) > 1:
        relation = "ALTERNATIVE" if re.search(r"(?i)\bOR\b|或", text) else ""
        if not relation:
            return Proposal(
                "建议确认",
                "多个ASTM材料标准",
                "原文出现多个可作为材料标准的 ASTM 编号，但未明确其相互关系。",
                items,
                "",
                evidence,
            )
    else:
        relation = "SINGLE"

    if any(item["GRADE"] for item in items):
        return Proposal(
            "可自动确认",
            "ASTM材料标准及牌号",
            "原文明示 ASTM 材料标准及其牌号，按原始标准和牌号拆分。",
            items,
            relation,
            evidence,
        )
    return Proposal(
        "建议确认",
        "仅识别到ASTM材料标准",
        "原文只明确材料标准，未找到可确认的具体牌号。",
        items,
        relation,
        evidence,
    )


def api_5l_proposal(text: str) -> Proposal | None:
    api = re.search(r"(?i)\bAPI\s*5L\b", text)
    if not api:
        return None
    grade = ""
    grade_match = re.search(
        r"(?i)(?<![A-Z0-9])(?:GR(?:ADE)?\.?\s*)?"
        r"(L\d{3}[MNQ]?|X\d{2}[MNQ]?)(?![A-Z0-9])",
        text,
    )
    if grade_match:
        grade = grade_match.group(1).upper()
    item = material_item(
        grade,
        standard="API 5L",
        special=extract_special_requirements(text),
    )
    if grade:
        return Proposal(
            "可自动确认",
            "API 5L材料标准及牌号",
            "原文明示 API 5L 及具体钢级。",
            [item],
            "SINGLE",
            [f"API 5L {grade}"],
        )
    return Proposal(
        "建议确认",
        "仅识别到API 5L材料标准",
        "原文只出现 API 5L，未找到可确认的具体钢级。",
        [item],
        "SINGLE",
        ["API 5L"],
    )


def ocr_grade_proposal(text: str) -> Proposal | None:
    for raw, corrected in STRONG_OCR_CORRECTIONS:
        match = re.search(re.escape(raw), text, re.IGNORECASE)
        if not match:
            continue
        item = material_item(
            corrected,
            material_class=extract_material_class(text, match.start(), match.end()),
            special=extract_special_requirements(text),
        )
        return Proposal(
            "可自动确认",
            "强上下文OCR牌号修复",
            f"原文牌号“{match.group(0)}”属于可确认的 OCR/缺字符形式，补全为“{corrected}”。",
            [item],
            "SINGLE",
            [match.group(0), corrected],
        )
    for raw, corrected in INCOMPLETE_OCR_CORRECTIONS:
        match = re.search(re.escape(raw), text, re.IGNORECASE)
        if not match:
            continue
        item = material_item(
            corrected,
            material_class=extract_material_class(text, match.start(), match.end()),
            special=extract_special_requirements(text),
        )
        return Proposal(
            "建议确认",
            "不完整牌号补全",
            f"原文“{match.group(0)}”疑似缺少 Mo2，建议补全为“{corrected}”。",
            [item],
            "SINGLE",
            [match.group(0), corrected],
        )
    return None


def direct_grade_proposal(text: str) -> Proposal | None:
    for rule, pattern in DIRECT_GRADE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        grade = clean_grade(match.group(0))
        material_class = extract_material_class(text, match.start(), match.end())
        return Proposal(
            "可自动确认",
            rule,
            f"原文明示材质牌号“{grade}”，不使用旧编码反推。",
            [
                material_item(
                    grade,
                    material_class=material_class,
                    special=extract_special_requirements(text),
                )
            ],
            "SINGLE",
            [match.group(0)],
        )

    if re.search(r"玻璃钢|(?<![A-Z])FRP(?![A-Z])", text, re.IGNORECASE):
        return Proposal(
            "可自动确认",
            "非金属材质明示",
            "原文明示玻璃钢/FRP。",
            [material_item("FRP")],
            "SINGLE",
            ["FRP"],
        )
    if re.search(r"(?i)(?<![A-Z])PP[\s-]*R(?![A-Z])", text):
        return Proposal(
            "可自动确认",
            "非金属材质明示",
            "原文明示 PP-R。",
            [material_item("PP-R")],
            "SINGLE",
            ["PP-R"],
        )
    return None


def shorthand_grade_proposal(text: str) -> Proposal | None:
    shorthand_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "WP304L",
            re.compile(r"(?i)WP304L(?![A-Z0-9])"),
        ),
        (
            "WP304",
            re.compile(r"(?i)WP304(?![A-Z0-9])"),
        ),
        (
            "SF304L",
            re.compile(r"(?i)SF304L(?=$|[^A-Z0-9]|ASME)"),
        ),
        (
            "SF304",
            re.compile(r"(?i)SF304(?=$|[^A-Z0-9]|GB|HG|SH|NB)"),
        ),
        (
            "SS304",
            re.compile(r"(?i)SS304(?![A-Z0-9])"),
        ),
        (
            "S316L",
            re.compile(r"(?i)S316L(?![A-Z0-9])"),
        ),
        (
            "SS316L",
            re.compile(r"(?i)SS316L(?![A-Z0-9])"),
        ),
        (
            "SS316",
            re.compile(r"(?i)SS316(?![A-Z0-9])"),
        ),
    )
    for grade, pattern in shorthand_patterns:
        match = pattern.search(text)
        if not match:
            continue
        return Proposal(
            "可自动确认",
            "原文明示材质简写",
            f"原文明示材质简写“{match.group(0)}”，保留其原始牌号语义。",
            [material_item(grade, special=extract_special_requirements(text))],
            "SINGLE",
            [match.group(0)],
        )

    sulfur = re.search(r"(?i)(?<![A-Z0-9])(16Mn)H2S(?![A-Z0-9])", text)
    if sulfur:
        return Proposal(
            "可自动确认",
            "牌号与耐硫要求粘连",
            "原文“16MnH2S”拆分为主体牌号16Mn和ANTI-H2S要求。",
            [material_item("16Mn", special=["ANTI-H2S"])],
            "SINGLE",
            [sulfur.group(0)],
        )

    anti_hic = re.search(r"(?i)(?<![A-Z0-9])(20)S(?=GB/T\s*9948)", text)
    if anti_hic:
        return Proposal(
            "可自动确认",
            "牌号与材料标准粘连",
            "原文“20SGB/T9948”中的主体牌号为20，耐蚀要求按原文ANTI-HIC单独保留。",
            [material_item("20", special=extract_special_requirements(text))],
            "SINGLE",
            [anti_hic.group(0)],
        )

    uncertain_20s = re.search(
        r"(?i)(?<![A-Z0-9])(20)S(?=\s*GB/T\s*9948)",
        text,
    )
    if uncertain_20s:
        return Proposal(
            "建议确认",
            "项目简写20S",
            "原文出现项目简写20S，可确认主体牌号为20，但S的专项要求需结合项目定义确认。",
            [material_item("20", special=extract_special_requirements(text))],
            "SINGLE",
            [uncertain_20s.group(0)],
        )

    code_suffix = re.search(
        r"(?i)(?:^|[^A-Z0-9])"
        r"(?:PE|BE|RE|RC|BW|RF|RJ|FNPT|NPT|SW)\s*(20)"
        r"(?:(III|II|I|Ⅲ|Ⅱ|Ⅰ))?(?=$|[^0-9A-Z])",
        text,
    )
    if code_suffix:
        specials = extract_special_requirements(text)
        material_class = ""
        if code_suffix.group(2):
            material_class = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}.get(
                code_suffix.group(2),
                code_suffix.group(2).upper(),
            )
        return Proposal(
            "可自动确认",
            "部件或连接简写后粘连20钢",
            "原文部件/连接简写后的20为材质牌号，按20钢拆分。",
            [
                material_item(
                    "20",
                    material_class=material_class,
                    special=specials,
                )
            ],
            "SINGLE",
            [code_suffix.group(0).strip()],
        )

    pressure_suffix = re.search(
        r"(?i)CL\s*\d+\s*(20)"
        r"(?:(III|II|I|Ⅲ|Ⅱ|Ⅰ))?"
        r"(?=\s*(?:GB/T|NB/T|HG/T|SH/T|RF|$))",
        text,
    )
    if pressure_suffix:
        material_class = ""
        if pressure_suffix.group(2):
            material_class = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}.get(
                pressure_suffix.group(2),
                pressure_suffix.group(2).upper(),
            )
        return Proposal(
            "可自动确认",
            "压力等级后粘连20钢",
            "原文压力等级后的20为材质牌号，按20钢拆分。",
            [
                material_item(
                    "20",
                    material_class=material_class,
                    special=extract_special_requirements(text),
                )
            ],
            "SINGLE",
            [pressure_suffix.group(0).strip()],
        )
    return None


def high_silicon_composite_proposal(text: str) -> Proposal | None:
    if re.search(r"焊环高硅不锈钢\s*/\s*法兰\s*304", text):
        return Proposal(
            "可自动确认",
            "高硅不锈钢焊环与304法兰",
            "原文明示焊环为高硅不锈钢、松套法兰为304；沿用现有约定，将非FLANGE的焊环归入BODY。",
            [
                material_item("高硅不锈钢"),
                material_item("304", part="FLANGE"),
            ],
            "COMPOSITE",
            ["焊环高硅不锈钢", "法兰304"],
        )
    if re.search(r"304\s*衬\s*高硅不锈钢", text):
        return Proposal(
            "建议确认",
            "304主体衬高硅不锈钢",
            "原文明示304主体及高硅不锈钢衬层。",
            [
                material_item("304"),
                material_item("高硅不锈钢", part="LINING"),
            ],
            "COMPOSITE",
            ["304", "衬高硅不锈钢"],
        )
    return None


def explicit_20_proposal(text: str) -> Proposal | None:
    special = extract_special_requirements(text)

    if re.search(r"(?i)\b20\s*(?:#)?\s*(?:\+|/)\s*PTFE\b|20\s*\([^)]*\)\s*\+\s*PTFE", text):
        return Proposal(
            "可自动确认",
            "主体加衬里材质",
            "原文明示 20 钢主体与 PTFE 衬里。",
            [material_item("20"), material_item("PTFE", part="LINING")],
            "COMPOSITE",
            ["20", "PTFE"],
        )

    coating_match = re.search(r"(?i)\b(4PE|3PE)\b", text)
    if coating_match and re.search(r"(?<!\d)20(?:#)?(?!\d)", text):
        coating = coating_match.group(1).upper()
        return Proposal(
            "可自动确认",
            "主体加涂层材质",
            f"原文明示 20 钢主体与 {coating} 外防腐层。",
            [material_item("20"), material_item(coating, part="COATING")],
            "COMPOSITE",
            ["20", coating],
        )

    class_match = re.search(
        r"(?<![0-9A-Z])20\s*(III|II|I|Ⅲ|Ⅱ|Ⅰ)(?![A-Z0-9])",
        text,
        re.IGNORECASE,
    )
    if class_match:
        material_class = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}.get(
            class_match.group(1),
            class_match.group(1).upper(),
        )
        return Proposal(
            "可自动确认",
            "20钢及材料等级明示",
            f"原文明示20钢，材料等级为{material_class}。",
            [
                material_item(
                    "20",
                    material_class=material_class,
                    special=special,
                )
            ],
            "SINGLE",
            [class_match.group(0)],
        )

    patterns: tuple[tuple[str, str], ...] = (
        (r"(?i)(?<![0-9A-Z])20#?(?![0-9A-Z])", "原文明示20钢"),
        (
            r"(?i)(?:GB/?T|NB/?T)?\s*(?:8163|47008|9948|5310)\s*[-_/]?\s*(20)(?!\d)",
            "材料标准后粘连20钢",
        ),
        (
            r"(?i)(?:BW|RF|FORGED|SMLS|EFW|ERW)\s*(20)(?!\d)",
            "工艺或连接标识后粘连20钢",
        ),
        (
            r"(?i)(?:MM|毫米|[x×*]\s*\d+(?:\.\d+)?)\s*(20)(?=\s*(?:DN|$|[,;]))",
            "尺寸字段后粘连20钢",
        ),
    )
    for pattern, rule in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if "GALVANIZED" in special or re.search(r"(?i)镀锌|\bGALV", text):
            special = list(dict.fromkeys([*special, "GALVANIZED"]))
        return Proposal(
            "可自动确认",
            rule,
            "原文可定位到 20 钢牌号；粘连形式仅拆分字符，不依据旧编码补造牌号。",
            [material_item("20", special=special)],
            "SINGLE",
            [match.group(0)],
        )
    return None


def simple_numeric_grade_proposal(text: str) -> Proposal | None:
    candidates = (
        (
            "304",
            r"(?<![A-Z0-9.])304(?![A-Z0-9.])|(?<=[ⅠⅡⅢI])304(?![A-Z0-9.])",
        ),
        ("316", r"(?<![A-Z0-9.])316(?![A-Z0-9.])"),
        ("316L", r"(?<![A-Z0-9.])316L(?![A-Z0-9.])"),
        ("304L", r"(?<![A-Z0-9.])304L(?![A-Z0-9.])"),
        ("2205", r"(?<![A-Z0-9.])2205(?![A-Z0-9.])"),
    )
    for grade, pattern in candidates:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        return Proposal(
            "可自动确认",
            "独立牌号明示",
            f"原文明示独立材质牌号“{match.group(0)}”。",
            [material_item(grade, special=extract_special_requirements(text))],
            "SINGLE",
            [match.group(0)],
        )
    return None


def coating_or_galvanized_proposal(text: str) -> Proposal | None:
    q_grade = re.search(
        r"(?i)(?<![A-Z0-9])(Q(?:235|245|345)[A-Z]?)(CE)?(?![A-Z0-9])",
        text,
    )
    if q_grade:
        grade = q_grade.group(1).upper()
        specials = extract_special_requirements(text)
        if q_grade.group(2):
            specials = list(dict.fromkeys([*specials, "NACE"]))
        items = [material_item(grade, special=specials)]
        outer = re.search(r"(?i)外(?:加强级)?\s*(PE|EP)(?![A-Z])", text)
        inner = re.search(r"(?i)内(?:涂)?\s*(PE|EP)(?![A-Z])", text)
        if outer:
            items.append(material_item(outer.group(1).upper(), part="COATING"))
        if inner:
            items.append(material_item(inner.group(1).upper(), part="LINING"))
        return Proposal(
            "可自动确认",
            "碳钢牌号及表面处理",
            f"原文明示主体牌号“{grade}”及可识别的表面处理。",
            items,
            "COMPOSITE" if len(items) > 1 else "SINGLE",
            [grade],
        )
    return None


def unresolved_evidence_proposal(text: str) -> Proposal:
    if re.search(r"材料不明|材质不明|\bUNDEFINED\b", text, re.IGNORECASE):
        return Proposal(
            "无法确认",
            "原文明示材料不明",
            "原文明确写明材料不明，不能补造具体牌号。",
            None,
            "",
            [],
        )

    incomplete_astm = re.search(
        r"(?i)\bASTM\s+(?:GR(?:ADE)?\.?\s*[A-Z0-9.-]+)\b",
        text,
    )
    if incomplete_astm:
        return Proposal(
            "无法确认",
            "ASTM材料标准编号缺失",
            "原文只出现 ASTM 与等级文字，但缺少 A/B 系列标准编号，无法确定完整材料标准。",
            None,
            "",
            [incomplete_astm.group(0)],
        )

    family = re.search(
        r"(?i)(?<![A-Z])(?:"
        r"碳钢|不锈钢|双相不锈钢|钛(?:材|合金)?|"
        r"CARBON\s+STEEL|STAINLESS\s+STEEL|DUPLEX\s+SS|CS"
        r")(?![A-Z])",
        text,
    )
    if family:
        return Proposal(
            "无法确认",
            "仅有材料族或通称",
            f"原文只有材料族/通称“{family.group(0)}”，不足以唯一确定具体牌号。",
            None,
            "",
            [family.group(0)],
        )

    standard = re.search(
        r"(?i)(?:"
        r"GB/?T?\s*(?:3087|3624|713|8163|9948|14976|21833)|"
        r"NB/?T\s*470(?:08|10)|"
        r"EN\s*10305|ASTM\s*A182"
        r")",
        text,
    )
    if standard:
        return Proposal(
            "无法确认",
            "仅有相关标准但缺少牌号",
            f"原文出现“{standard.group(0)}”，但该标准不能唯一推出当前材料牌号。",
            None,
            "",
            [standard.group(0)],
        )

    return Proposal(
        "无法确认",
        "原文无可确认材质信息",
        "原文未提供可独立确认的材料标准或具体牌号，不能依据历史编码反推。",
        None,
        "",
        [],
    )


def propose(text: str) -> Proposal:
    normalized = normalize_text(text)
    for resolver in (
        explicit_astm_proposal,
        api_5l_proposal,
        ocr_grade_proposal,
        high_silicon_composite_proposal,
        coating_or_galvanized_proposal,
        shorthand_grade_proposal,
        direct_grade_proposal,
        explicit_20_proposal,
        simple_numeric_grade_proposal,
    ):
        proposal = resolver(normalized)
        if proposal is not None:
            return proposal
    return unresolved_evidence_proposal(normalized)


def current_source_row(
    review_row: dict[str, Any],
    sources: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    split = review_row["split"]
    index = review_row["source_index"]
    return sources[split][index]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    grouped_review = json.loads(args.review.read_text(encoding="utf-8"))
    review_rows = grouped_review.get(REVIEW_REASON, [])
    sources = {
        "train": json.loads(args.train_source.read_text(encoding="utf-8")),
        "val": json.loads(args.val_source.read_text(encoding="utf-8")),
    }

    proposals: list[dict[str, Any]] = []
    confidence_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    confidence_current_material_counts: dict[str, Counter[str]] = defaultdict(Counter)
    changed_old_target = 0
    changed_source_text = 0

    for review_row in review_rows:
        source_row = current_source_row(review_row, sources)
        if source_row.get("input") != review_row.get("input"):
            changed_source_text += 1
        source_output = source_row.get("output", {})
        current_material = source_output.get("MATERIAL", [])
        stale_material = review_row.get("current_material", [])
        if current_material != stale_material:
            changed_old_target += 1

        result = propose(source_row.get("input", ""))
        confidence_counts[result.confidence] += 1
        rule_counts[result.rule] += 1
        current_values = "/".join(
            str(item.get("VALUE", ""))
            for item in current_material
            if item.get("VALUE", "") != ""
        )
        confidence_current_material_counts[result.confidence][
            current_values or "空"
        ] += 1
        proposals.append(
            {
                "split": review_row["split"],
                "source_index": review_row["source_index"],
                "input": source_row.get("input", ""),
                "review_snapshot_input": review_row.get("input", ""),
                "current_source_material": current_material,
                "review_snapshot_material": stale_material,
                "current_standard": source_output.get("STANDARD", []),
                "proposal": {
                    "confidence": result.confidence,
                    "rule": result.rule,
                    "reason": result.reason,
                    "MATERIAL_RELATION": result.relation,
                    "MATERIAL": result.material,
                    "source_evidence": result.evidence,
                },
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        key = row["proposal"]["confidence"]
        grouped[key].append(row)

    report = {
        "review_reason": REVIEW_REASON,
        "source_rows": len(review_rows),
        "proposal_rows": len(proposals),
        "source_files_modified": False,
        "review_snapshot_input_changed_since_generation": changed_source_text,
        "review_snapshot_material_changed_since_generation": changed_old_target,
        "confidence_counts": dict(confidence_counts),
        "confidence_current_material_counts": {
            confidence: dict(counts.most_common())
            for confidence, counts in confidence_current_material_counts.items()
        },
        "rule_counts": dict(rule_counts.most_common()),
        "policy": {
            "source_of_truth": "原始描述",
            "old_material_usage": "仅用于展示差异，不作为牌号证据",
            "ocr_completion": "仅对规则中列明的强上下文错误自动补全",
            "unresolved": "原文证据不足时保持无法确认，不编造材质",
        },
        "outputs": {
            "proposals": str(args.proposals),
            "grouped": str(args.grouped),
        },
    }
    write_json(args.proposals, proposals)
    write_json(args.grouped, dict(grouped))
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
