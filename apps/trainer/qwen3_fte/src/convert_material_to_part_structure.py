#!/usr/bin/env python3
"""Convert normalized material labels into source-grounded material structures.

The source datasets store final business codes in ``MATERIAL[].VALUE``.  Those
codes are useful for validation, but they are not reliable enough to reconstruct
source grades.  This converter therefore emits a row only when the material
expression can be located in the original input text.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import yaml


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "材质规范"
OUTPUT_DIR = SOURCE_DIR / "结构化原始牌号"
DEFAULT_TRAIN_SOURCE = SOURCE_DIR / "材质规范_train.json"
DEFAULT_VAL_SOURCE = SOURCE_DIR / "材质规范_val.json"
DEFAULT_TRAIN_OUTPUT = OUTPUT_DIR / "材质规范_结构化原始牌号_train.json"
DEFAULT_VAL_OUTPUT = OUTPUT_DIR / "材质规范_结构化原始牌号_val.json"
DEFAULT_REVIEW_OUTPUT = OUTPUT_DIR / "材质规范_结构化原始牌号_待复核.json"
DEFAULT_GROUPED_REVIEW_OUTPUT = (
    OUTPUT_DIR / "材质规范_结构化原始牌号_待复核_按原因分组.json"
)
DEFAULT_REPORT_OUTPUT = OUTPUT_DIR / "材质规范_结构化原始牌号_转换报告.json"
DEFAULT_CONFIRMED_PROPOSALS = OUTPUT_DIR / "材质规范_结构化原始牌号_已确认建议.json"
DEFAULT_MAPPING = REPO_ROOT / "src" / "encoder" / "config" / "material_mapping.yaml"


ROMAN_MAP = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}
BARE_GRADE_STANDARDS = {
    "API 5L",
    "ASTM A53",
    "ASTM A106",
    "ASTM A333",
    "ASTM A387",
    "ASTM A516",
}
ALLOWED_PARTS = {
    "BODY",
    "LINING",
    "FLANGE",
    "INNER_PIPE",
    "OUTER_PIPE",
    "CLADDING",
    "COATING",
}
ALLOWED_RELATIONS = {
    "SINGLE",
    "DUAL_CERTIFIED",
    "ALTERNATIVE",
    "COMPOSITE",
    "EQUIVALENT",
}

STATUS_ZH = {
    "alias_not_found": "原文中未找到可确认的材质别名",
    "alternative_relation_not_explicit": "原文未明确表达备选材料标准关系",
    "alternative_standard_missing": "原文缺少备选材料标准",
    "atomic_failed": "单一材质解析失败",
    "converted_alternative_standard": "备选材料标准转换成功",
    "converted_atomic": "单一材质转换成功",
    "converted_ceramic_fixed_flange": "陶瓷衬里固定法兰复合材质转换成功",
    "converted_coated_material": "涂层复合材质转换成功",
    "converted_coating_requirement": "涂层要求转换成功",
    "converted_dual_grade": "双牌号材质转换成功",
    "converted_equivalent_designation": "等效牌号表达转换成功",
    "converted_fractional_alloy": "分数形式合金牌号转换成功",
    "converted_fractional_alloy_lining": "分数形式合金衬里材质转换成功",
    "converted_frp_galvanized_flange": "玻璃钢与镀锌法兰复合材质转换成功",
    "converted_glass_lined_loose_flange": "搪玻璃松套法兰复合材质转换成功",
    "converted_inner_outer": "内外管复合材质转换成功",
    "converted_hdpp_lining": "HDPP衬里复合材质转换成功",
    "converted_lining_composite": "衬里复合材质转换成功",
    "converted_loose_flange_pair": "松套法兰复合材质转换成功",
    "converted_material_validation_failed": "转换结果未通过结构校验",
    "invalid_material_relation": "材质关系无效",
    "converted_stainless_lined_pipe": "内衬不锈钢复合管转换成功",
    "converted_alternative_material": "备选材质转换成功",
    "converted_welded_fitting_plate_composite": "焊制管件与板材复合材料转换成功",
    "direct_alias": "原文直接命中材质别名",
    "dual_grade_alias_not_found": "原文中未找到双牌号表达",
    "dual_grade_empty": "双牌号解析结果为空",
    "dual_grade_expression": "原文命中双牌号表达",
    "dual_grade_source_expression_missing": "原文缺少双牌号分隔表达",
    "empty_atomic_material": "单一材质解析结果为空",
    "empty_normalized_material": "原归一化材质为空",
    "excluded_weld_ring_loose_flange": "焊环松套法兰暂不纳入初版训练集",
    "explicit_inner_outer_pair": "原文明示内外管材质关系",
    "explicit_lining_pair": "原文明示主体与衬里材质关系",
    "fractional_alloy_failed": "分数形式合金牌号解析失败",
    "glued_material_after_standard": "原文命中与材料标准粘连的牌号",
    "glued_safe_alias": "原文命中可安全确认的粘连材质别名",
    "merged_confirmed_proposal": "人工确认建议合并成功",
    "inner_outer_component_not_resolved": "内外管材质组成未能完整解析",
    "inner_outer_relation_not_explicit": "原文未明确表达内外管关系",
    "invalid_output": "原数据输出结构无效",
    "known_structure_source_evidence_missing": "原文缺少已知复合结构所需证据",
    "lining_body_not_resolved": "衬里结构的主体材质未能解析",
    "lining_component_not_found": "归一化结果中未找到衬里材质",
    "lining_relation_not_explicit": "原文未明确表达衬里关系",
    "lining_source_token_not_found": "原文中未找到衬里材质",
    "loose_flange_component_not_resolved": "松套法兰材质组成未能完整解析",
    "loose_flange_relation_not_explicit": "原文未明确表达松套法兰关系",
    "material_cardinality_not_one": "原数据材质项数量不是一项",
    "normalized_ce_without_source_requirement": "归一化结果含耐腐蚀后缀但原文无对应要求",
    "normalized_zn_without_source_requirement": "归一化结果含镀锌后缀但原文无镀锌依据",
    "not_inner_outer_target": "归一化结果不属于内外管结构",
    "not_loose_flange_pair": "归一化结果不属于松套法兰结构",
    "unresolved_composite_material": "复合材质结构无法可靠确定",
}

MATERIAL_STANDARD_NUMBERS = {
    "A53",
    "A105",
    "A106",
    "A182",
    "A193",
    "A194",
    "A216",
    "A217",
    "A234",
    "A240",
    "A269",
    "A270",
    "A312",
    "A320",
    "A333",
    "A335",
    "A350",
    "A351",
    "A352",
    "A358",
    "A387",
    "A403",
    "A420",
    "A516",
    "A536",
    "A564",
    "A671",
    "A672",
    "A691",
    "A694",
    "A790",
    "A815",
    "A860",
    "B165",
    "B167",
    "B366",
    "B444",
    "B564",
}

DUAL_GRADE_CODES = {
    "304/304L",
    "304L/304",
    "316/316L",
    "304/304LIII",
    "304/304LII",
    "304/304LI",
    "316/316LIII",
    "316/316LII",
    "316/316LI",
}

SLASH_ATOMIC_CODES = {
    "11/4Cr",
    "1-1/4Cr-1/2Mo",
}

SUPPLEMENTAL_ALIASES: dict[str, tuple[str, ...]] = {
    "20": ("20号钢", "20#"),
    "20ZN": ("20镀锌", "20 GALV", "20 GALVANIZED"),
    "304": (
        "S30408",
        "06Cr19Ni10",
        "06Cr18Ni9",
        "0Cr18Ni9",
        "TP304",
        "WP304",
        "F304",
        "SF304",
        "SUS 304TP",
        "SUS304TP",
    ),
    "304L": (
        "S30403",
        "022Cr19Ni10",
        "00Cr19Ni10",
        "TP304L",
        "WP304L",
        "F304L",
        "SF304L",
    ),
    "304H": (
        "S30409",
        "07Cr19Ni10",
        "WP304H",
        "F304H",
        "TP304H",
    ),
    "316": (
        "S31608",
        "06Cr17Ni12Mo2",
        "TP316",
        "WP316",
        "F316",
        "SUS 316TP",
        "SUS316TP",
        "S31608TP",
    ),
    "316L": (
        "S31603",
        "022Cr17Ni12Mo2",
        "X2CrNiMo17-12-2",
        "TP316L",
        "WP316L",
        "F316L",
        "SF316L",
        "SUS 316LTP",
        "SUS316LTP",
        "SS316L-BA",
    ),
    "321": (
        "S32168",
        "06Cr18Ni11Ti",
        "0Cr18Ni10Ti",
        "TP321",
        "WP321",
        "F321",
        "SF321",
    ),
    "347H": ("TP347H", "WP347H", "F347H"),
    "2205": (
        "S32205",
        "S22053",
        "F60",
        "F2205",
        "WPS32205",
        "ASTM A182 GRADE F60",
        "ASTM A182 GRADEF60",
    ),
    "32205": ("S32205", "WPS32205"),
    "2200": ("N02200",),
    "2201": ("N02201",),
    "2507": ("S25073",),
    "6600": ("N06600", "WPNCI"),
    "8825": ("N08825",),
    "A105": ("ASTM A105", "A105"),
    "A333": ("ASTM A333M", "ASTM A333", "A333M", "A333"),
    "DIZN": ("DI GALV", "DUCTILE IRON GALVANIZED"),
    "F11": ("F11",),
    "F12": ("F12",),
    "WP11": ("WP11",),
    "347": ("TP347", "WP347", "F347"),
    "310S": ("06Cr25Ni20", "S31008"),
    "316H": ("TP316H", "WP316H", "F316H", "S31609"),
    "CF8": ("CF8",),
    "CF415": ("CF415",),
    "FRPP": ("FRPP",),
    "L245": ("L245",),
    "L245N": ("L245N",),
    "LF2": ("LF2",),
    "M400": ("N04400", "NO4400", "N0 4400"),
    "Q245R": ("Q245R",),
    "TA2": ("TA2",),
    "WPB": ("WPB", "WPBS", "WPBW"),
    "WPL6": ("WPL6", "WPL6W"),
    "321H": ("SF321H", "TP321H", "WP321H", "F321H"),
    "12Cr5MoNT": ("12Cr5MoNT",),
    "15CrMoG": ("I5CrMoG", "l5CrMoG"),
    "11/4Cr": ("1 1/4Cr", "1-1/4Cr", "1.25Cr"),
    "GL": ("搪玻璃管", "搪玻璃"),
    "PE": ("STEEL REINFORCED POLYETHYLENE", "钢丝网骨架聚乙烯"),
    "15CrMo": ("l5CrMo",),
}

GLUED_SAFE_ALIASES: dict[str, tuple[str, ...]] = {
    "20": ("20#",),
    "20ZN": ("20镀锌",),
    "304": ("S30408", "06Cr19Ni10", "06Cr18Ni9", "0Cr18Ni9"),
    "304L": ("S30403", "022Cr19Ni10", "00Cr19Ni10", "SF304L"),
    "304H": ("S30409", "07Cr19Ni10"),
    "316": ("S31608", "06Cr17Ni12Mo2"),
    "316L": ("S31603", "022Cr17Ni12Mo2", "X2CrNiMo17-12-2"),
    "321": ("S32168", "06Cr18Ni11Ti", "0Cr18Ni10Ti", "WP321"),
    "321H": ("SF321H", "TP321H", "WP321H", "F321H"),
    "347H": ("TP347H", "WP347H", "F347H"),
    "2205": ("S32205", "S22053", "WPS32205", "F2205"),
    "32205": ("S32205", "WPS32205"),
    "2200": ("N02200",),
    "2201": ("N02201",),
    "2507": ("S25073",),
    "6600": ("N06600", "WPNCI"),
    "8825": ("N08825",),
    "A105": ("A105",),
    "A333": ("A333M", "A333"),
    "DIZN": ("DI GALV",),
    "F11": ("F11",),
    "F12": ("F12",),
    "WP11": ("WP11",),
    "347": ("TP347", "WP347", "F347"),
    "310S": ("06Cr25Ni20", "S31008"),
    "316H": ("TP316H", "WP316H", "F316H", "S31609"),
    "CF8": ("CF8",),
    "CF415": ("CF415",),
    "FRPP": ("FRPP",),
    "FRP": ("FRP",),
    "L245": ("L245",),
    "L245N": ("L245N",),
    "LF2": ("LF2",),
    "M400": ("N04400", "NO4400"),
    "Q245R": ("Q245R",),
    "TA2": ("TA2",),
    "WPB": ("WPBS", "WPBW", "WPB"),
    "WPL6": ("WPL6W", "WPL6"),
    "12Cr5MoNT": ("12Cr5MoNT",),
    "15CrMoG": ("I5CrMoG", "l5CrMoG"),
    "11/4Cr": ("1 1/4Cr", "1-1/4Cr", "1.25Cr"),
}

LINING_CODE_TOKENS = {
    "PTFE",
    "RPTFE",
    "MPTFE",
    "EPTFE",
    "PP",
    "PE",
    "EAA",
    "PVC",
    "CPVC",
    "衬胶",
    "CERAMIC",
    "CEM",
    "GL",
}

LINING_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    ("RPTFE", re.compile(r"(?i)(?<![A-Z0-9])RPTFE(?![A-Z0-9])")),
    ("MPTFE", re.compile(r"(?i)(?<![A-Z0-9])MPTFE(?![A-Z0-9])")),
    ("EPTFE", re.compile(r"(?i)(?<![A-Z0-9])E[\s-]*PTFE(?![A-Z0-9])")),
    (
        "PTFE",
        re.compile(
            r"(?i)(?<![A-Z0-9])PTFE(?=$|[^A-Z0-9]|LIN(?:ED|ING))|"
            r"聚四氟乙烯|四氟"
        ),
    ),
    ("CERAMIC", re.compile(r"(?i)CERAMIC|陶瓷")),
    ("GLASS", re.compile(r"(?i)GLASS[\s-]*LINED|搪玻璃|玻璃衬里")),
    ("CEMENT", re.compile(r"(?i)CEMENT[\s-]*LINED|水泥衬里")),
    ("衬胶", re.compile(r"(?i)RUBBER[\s-]*LINED|衬胶")),
    ("CPVC", re.compile(r"(?i)(?<![A-Z0-9])CPVC(?![A-Z0-9])")),
    ("PVC", re.compile(r"(?i)(?<![A-Z0-9])(?:PVC[\s-]*U|U[\s-]*PVC|PVC)(?![A-Z0-9])")),
    ("EAA", re.compile(r"(?i)(?<![A-Z0-9])EAA(?![A-Z0-9])")),
    ("PP", re.compile(r"(?i)(?<![A-Z0-9])PP(?![A-Z0-9])|聚丙烯")),
    ("PE", re.compile(r"(?i)(?<![A-Z0-9])(?:HDPE|PE)(?![A-Z0-9])|聚乙烯")),
)

SPECIAL_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "NACE",
        re.compile(r"(?i)(?<![A-Z0-9])NACE(?:\s*MR[\s-]*0?1(?:03|75))?(?![A-Z0-9])"),
    ),
    (
        "ANTI-H2S",
        re.compile(r"(?i)ANTI[\s_-]*H2S|抗硫化氢|H2S"),
    ),
    (
        "ANTI-HIC",
        re.compile(r"(?i)(?<![A-Z0-9])(?:ANTI[\s_-]*)?HIC(?![A-Z0-9])"),
    ),
    ("ANTI-SCC", re.compile(r"(?i)ANTI[\s_-]*SCC|ANTT[\s_-]*SCC")),
    (
        "GALVANIZED",
        re.compile(
            r"(?i)GALV(?:ANIZED)?|HOT[\s-]*DIP(?:PED)?|镀锌|"
            r"(?:^|[/+;,\s])ZN(?=$|[/+;,\s]|GB|HG|SH|NB|EN|DIN)"
        ),
    ),
    ("CE", re.compile(r"(?i)(?:^|[-/+;,\s])CE(?=$|[-/+;,\s]|GB|HG|SH|NB)")),
    (
        "3PE",
        re.compile(
            r"(?i)(?<![A-Z0-9])3PE(?![A-Z0-9])|三层\s*(?:PE|聚乙烯)"
        ),
    ),
)

MATERIAL_STANDARD_RE = re.compile(
    r"(?i)(?:"
    r"(?<![A-Z0-9])ASTM[\s._/-]*(?P<astm_letter>[AB])?"
    r"[\s._/-]*(?P<astm_number>\d{2,4})(?:M)?"
    r"|"
    r"(?<![A-Z0-9])(?P<plain_letter>[AB])[\s._/-]*"
    r"(?P<plain_number>\d{2,4})(?:M)?"
    r")(?!\d)"
)
API_5L_RE = re.compile(
    r"(?i)(?<![A-Z0-9])API[\s._/-]*5L"
    r"(?:[\s,;:/_-]*(?:PSL\s*[12][\s,;:/_-]*)?"
    r"(?:GR(?:ADE)?\.?\s*)?"
    r"(?P<grade>L\d{3}[A-Z]?|X\d{2,3}[A-Z]?|B[N]?|A))?"
)
API_5L_GRADE_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(?P<grade>L\d{3}[A-Z]?|X\d{2,3}[A-Z]?)"
    r"(?![A-Z0-9])"
)


@dataclass(frozen=True)
class AliasRule:
    target: str
    alias: str
    pattern: Pattern[str]


@dataclass(frozen=True)
class AliasHit:
    target: str
    alias: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class ConversionResult:
    material: list[dict[str, Any]] | None
    status: str
    evidence: dict[str, Any]
    relation: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建按部件标注的原始牌号材质数据集")
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument(
        "--grouped-review-output",
        type=Path,
        default=DEFAULT_GROUPED_REVIEW_OUTPUT,
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--confirmed-proposals",
        type=Path,
        default=DEFAULT_CONFIRMED_PROPOSALS,
        help="已人工确认的结构化材质建议；文件不存在时不合并",
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_material_standard_boundaries(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(
        r"(?i)(?<![A-Z0-9])XS(?=A[\s._/-]*\d{2,4})",
        "XS ",
        text,
    )


def compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]+", "", normalize_text(value).upper())


def clean_fragment(value: Any) -> str:
    return normalize_text(value).strip(" \t\r\n,，;；:+")


def split_chunks(value: str) -> list[str]:
    normalized = normalize_text(value)
    return re.findall(
        r"[A-Za-z]+|[0-9]+|[\u4e00-\u9fff]+",
        normalized,
    )


def loose_alias_pattern(value: str) -> Pattern[str]:
    chunks = split_chunks(value)
    if not chunks:
        return re.compile(r"(?!x)x")
    separator = r"[\s._,/+()#\-]*"
    body = separator.join(re.escape(chunk) for chunk in chunks)
    return re.compile(
        rf"(?<![A-Za-z0-9\u4e00-\u9fff])(?P<raw>{body})"
        rf"(?=$|[^A-Za-z0-9\u4e00-\u9fff]|"
        rf"GB|HG|SH|NB|SY|EN|DIN|ASTM|ASME|API|GR)",
        re.IGNORECASE,
    )


def load_alias_rules(path: Path) -> dict[str, list[AliasRule]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value_mapping = payload.get("value_mapping") or {}
    result: dict[str, list[AliasRule]] = {}
    targets = list(
        dict.fromkeys(
            [str(target) for target in value_mapping]
            + list(SUPPLEMENTAL_ALIASES)
        )
    )
    for target in targets:
        aliases = value_mapping.get(target)
        candidates = list(aliases) if isinstance(aliases, list) else []
        candidates.extend(SUPPLEMENTAL_ALIASES.get(str(target), ()))
        candidates.append(str(target))
        deduped: dict[str, AliasRule] = {}
        for alias in candidates:
            alias_text = clean_fragment(alias)
            if not alias_text:
                continue
            deduped.setdefault(
                normalize_text(alias_text).casefold(),
                AliasRule(
                target=str(target),
                alias=alias_text,
                pattern=loose_alias_pattern(alias_text),
                ),
            )
        result[str(target)] = sorted(
            deduped.values(),
            key=lambda rule: (-len(compact(rule.alias)), -len(rule.alias)),
        )
    return result


def find_alias_hits(text: str, rules: list[AliasRule]) -> list[AliasHit]:
    hits: list[AliasHit] = []
    normalized = normalize_text(text)
    for rule in rules:
        for match in rule.pattern.finditer(normalized):
            hits.append(
                AliasHit(
                    target=rule.target,
                    alias=rule.alias,
                    raw=clean_fragment(match.group("raw")),
                    start=match.start(),
                    end=match.end(),
                )
            )
    hits.sort(
        key=lambda hit: (
            -len(compact(hit.raw)),
            hit.start,
            hit.alias.casefold(),
        )
    )
    return hits


def target_base_and_modifier(
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> tuple[str, str]:
    base, modifiers = target_root_and_modifiers(target, alias_rules)
    return base, modifiers[0] if modifiers else ""


def target_root_and_modifiers(
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> tuple[str, list[str]]:
    """Peel coding modifiers while retaining only known material roots."""

    current = target
    modifiers: list[str] = []
    for _ in range(4):
        upper = current.upper()
        matched = False
        for suffix in ("CE", "ZN", "3PE", "III", "II", "I"):
            if not upper.endswith(suffix) or len(current) <= len(suffix):
                continue
            candidate = current[: -len(suffix)]
            if candidate not in alias_rules:
                continue
            current = candidate
            modifiers.append(suffix)
            matched = True
            break
        if not matched:
            break
    return current, modifiers


def resolve_atomic_alias(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> tuple[AliasHit | None, str]:
    rules = alias_rules.get(target)
    if rules is None:
        rules = [
            AliasRule(
                target=target,
                alias=target,
                pattern=loose_alias_pattern(target),
            )
        ]
    direct = find_alias_hits(text, rules)
    if direct:
        return direct[0], "direct_alias"

    base, modifiers = target_root_and_modifiers(target, alias_rules)
    if base != target:
        base_hits = find_alias_hits(text, alias_rules.get(base, []))
        if base_hits:
            return base_hits[0], f"base_alias:{'+'.join(modifiers)}"

    normalized = normalize_text(text)
    if (
        base == "20"
        and "CE" in modifiers
        and re.search(
            r"(?i)(?:47008|8163|9948|5310)\s*20(?:#)?(?=$|[^0-9])",
            normalized,
        )
    ):
        match = re.search(r"20(?:#)?(?=$|[^0-9])", normalized, re.IGNORECASE)
        if match:
            return (
                AliasHit(
                    target=target,
                    alias="20",
                    raw=clean_fragment(match.group(0)),
                    start=match.start(),
                    end=match.end(),
                ),
                "glued_material_after_standard",
            )
    if base == "20" and any(
        modifier in {"I", "II", "III"} for modifier in modifiers
    ):
        match = re.search(
            r"(?i)20(?:#)?(?=[,;\s]*(?:NB/T)?47008.*GR(?:ADE)?\.?\s*[ⅠⅡⅢI]{1,3})",
            normalized,
        )
        if match:
            return (
                AliasHit(
                    target=target,
                    alias="20",
                    raw=clean_fragment(match.group(0)),
                    start=match.start(),
                    end=match.end(),
                ),
                "class_qualified_material_alias",
            )
    for alias in GLUED_SAFE_ALIASES.get(base, ()):
        match = re.search(re.escape(alias), normalized, re.IGNORECASE)
        if match:
            return (
                AliasHit(
                    target=target,
                    alias=alias,
                    raw=clean_fragment(match.group(0)),
                    start=match.start(),
                    end=match.end(),
                ),
                "glued_safe_alias",
            )
    return None, "alias_not_found"


def extract_special_requirements(text: str) -> list[str]:
    requirements: list[str] = []
    for label, pattern in SPECIAL_PATTERNS:
        if pattern.search(text):
            requirements.append(label)
    requirements.extend(extract_coating_requirements(text))
    if "3PE" in requirements:
        requirements = [item for item in requirements if item != "PE"]
    return list(dict.fromkeys(requirements))


def extract_coating_requirements(text: str) -> list[str]:
    """Extract coating codes only when the source states coating semantics.

    ``PE`` is also the common abbreviation for plain end, so a standalone
    ``PE`` token is intentionally insufficient evidence.
    """

    normalized = normalize_text(text)
    candidates: list[tuple[int, str]] = []

    coating_patterns: tuple[tuple[str, Pattern[str]], ...] = (
        (
            "PE",
            re.compile(
                r"(?i)(?:外|内)[^,;，；]{0,12}(?<![A-Z0-9])PE(?![A-Z0-9])|"
                r"(?<![A-Z0-9])PE(?![A-Z0-9])\s*(?:涂层|防腐|包覆)"
            ),
        ),
        (
            "EP",
            re.compile(
                r"(?i)(?:外|内)[^,;，；]{0,12}(?<![A-Z0-9])EP(?![A-Z0-9])|"
                r"(?<![A-Z0-9])EP(?![A-Z0-9])\s*(?:涂层|防腐|包覆)"
            ),
        ),
    )
    for label, pattern in coating_patterns:
        match = pattern.search(normalized)
        if match and "衬" not in match.group(0):
            token_offset = match.group(0).upper().rfind(label)
            candidates.append((match.start() + max(token_offset, 0), label))

    return [
        label
        for _, label in sorted(candidates)
    ]


def strip_special_tokens(value: str) -> str:
    result = value
    for _, pattern in SPECIAL_PATTERNS:
        result = pattern.sub(" ", result)
    return clean_fragment(result)


def find_material_standards(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    normalized = normalize_material_standard_boundaries(text)
    for match in MATERIAL_STANDARD_RE.finditer(normalized):
        letter = (
            match.group("astm_letter")
            or match.group("plain_letter")
            or "A"
        ).upper()
        number = match.group("astm_number") or match.group("plain_number")
        code = f"{letter}{number}"
        if code in MATERIAL_STANDARD_NUMBERS:
            found.append(
                (
                    match.start(),
                    canonical_material_standard(letter, number),
                )
            )
    unique: list[str] = []
    for _, standard in sorted(found):
        if standard not in unique:
            unique.append(standard)
    return unique


def infer_material_standard_from_context(text: str, grade: str) -> str:
    standards = find_material_standards(text)
    if not standards:
        return ""

    upper = compact(grade)
    compatibility: tuple[tuple[Pattern[str], set[str]], ...] = (
        (re.compile(r"^F\d"), {"ASTM A182", "ASTM A350", "ASTM A694"}),
        (re.compile(r"^LF"), {"ASTM A350"}),
        (re.compile(r"^WP"), {"ASTM A234", "ASTM A403", "ASTM A420", "ASTM A815", "ASTM A860", "ASTM B366"}),
        (re.compile(r"^TP"), {"ASTM A312", "ASTM A358"}),
        (re.compile(r"^(?:CF|WCB|WC)"), {"ASTM A216", "ASTM A217", "ASTM A351"}),
        (re.compile(r"^P\d"), {"ASTM A333", "ASTM A335"}),
        (re.compile(r"^CC\d"), {"ASTM A671"}),
        (re.compile(r"^S\d{5}$"), {"ASTM A790"}),
        (
            re.compile(r"^(?:N|UNS(?:N)?)0"),
            {"ASTM B165", "ASTM B167", "ASTM B366", "ASTM B444", "ASTM B564"},
        ),
        (
            re.compile(r"^(?:S|SF)?(?:304|304L|304H|316|316L|316H|321|321H|347|347H|310H?)$"),
            {
                "ASTM A182",
                "ASTM A240",
                "ASTM A269",
                "ASTM A270",
                "ASTM A312",
                "ASTM A358",
                "ASTM A403",
            },
        ),
    )
    for pattern, allowed in compatibility:
        if not pattern.search(upper):
            continue
        matches = [standard for standard in standards if standard in allowed]
        if len(matches) == 1:
            return matches[0]
    return ""


def extract_class(value: str, target: str) -> tuple[str, str]:
    text = clean_fragment(value)
    explicit = re.search(
        r"(?i)(?:\bGR(?:ADE)?\.?\s*)(III|II|I|Ⅲ|Ⅱ|Ⅰ)\s*$",
        text,
    )
    if explicit:
        material_class = ROMAN_MAP.get(explicit.group(1), explicit.group(1).upper())
        return text[: explicit.start()].strip(" .,_/-"), material_class

    unicode_suffix = re.search(r"(Ⅲ|Ⅱ|Ⅰ)\s*$", text)
    if unicode_suffix:
        material_class = ROMAN_MAP[unicode_suffix.group(1)]
        return text[: unicode_suffix.start()].strip(" .,_/-"), material_class

    _, modifier = target_base_and_modifier(target, {})
    del modifier
    target_upper = target.upper()
    for suffix in ("III", "II", "I"):
        if not target_upper.endswith(suffix):
            continue
        spaced = re.search(rf"(?i)\s+{suffix}\s*$", text)
        if spaced:
            return text[: spaced.start()].strip(" .,_/-"), suffix
    return text, ""


def strip_ignored_astm_suffixes(value: str) -> str:
    text = clean_fragment(value)
    text = re.sub(r"(?i)\(\s*UNS[^)]*\)", "", text)
    text = re.sub(r"(?i)(?:[-\s]+)(?:WX|WU|W|S)\s*$", "", text)
    for suffix in ("WX", "WU", "W", "S"):
        if not text.upper().startswith("WP") or not text.upper().endswith(suffix):
            continue
        candidate = text[: -len(suffix)]
        if len(candidate) > 2:
            text = candidate
            break
    text = re.sub(r"(?i)\bCL(?:ASS)?\.?\s*\d+[A-Z]?\b", "", text)
    # Pressure class is frequently glued directly to the material grade.
    text = re.sub(r"(?i)CL(?:ASS)?\.?\s*\d+[A-Z]?\s*$", "", text)
    text = re.sub(r"(?i)\bPSL\s*[12]\b", "", text)
    return clean_fragment(text).strip(" .,_/-")


def sanitize_grade_boundary(value: str) -> str:
    text = clean_fragment(value)
    text = re.sub(r"(?i)\(\s*UNS[^)]*$", "", text)
    text = re.sub(r"(?i)[\s_/-]+DN\s*\d+\b.*$", "", text)
    text = re.sub(
        r"(?i)/\s*(?:"
        r"BE|BOE(?:-TOE)?|TOE(?:-TOE)?|RF|FF|FNPT|NPT|SW|BW|"
        r"SCH(?:EDULE)?\s*\d*[A-Z]*|POLISHING|DN\s*\d*"
        r")\b.*$",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^(WP(?:304L?|304H|316L?|316H|321H?|347H?|310H?))SMLS?$",
        r"\1",
        text,
    )
    text = re.sub(
        r"(?i)^((?:F|LF)\d+[A-Z]?)(?:BW|SW)$",
        r"\1",
        text,
    )
    return clean_fragment(text).strip("() .,_/-")


def extract_source_preserved_grade(
    full_text: str,
    hit: AliasHit,
) -> str:
    """Preserve source designations whose meaningful punctuation exceeds an alias."""

    normalized = normalize_text(full_text)
    for match in re.finditer(
        r"(?i)(?<![A-Z0-9])SUS[\s._/-]*F[\s._/-]*"
        r"(?P<alloy>304L?|304H|316L?|316H|321H?|347H?)"
        r"(?![A-Z0-9])",
        normalized,
    ):
        if match.start() <= hit.start and match.end() >= hit.end:
            return f"SUS F{match.group('alloy').upper()}"

    for match in re.finditer(
        r"(?i)(?<![A-Z0-9])"
        r"(?P<grade>X\d+[A-Z][A-Z0-9-]*)"
        r"\s*\(\s*(?P<number>\d+\.\d+)\s*\)",
        normalized,
    ):
        if match.start() <= hit.start and match.end() >= hit.end:
            grade = re.sub(r"\s+", "", match.group("grade"))
            return f"{grade}({match.group('number')})"
    return ""


def canonical_material_standard(letter: str, number: str) -> str:
    return f"ASTM {letter.upper()}{number}"


def strip_material_class_from_grade(grade: str, material_class: str) -> str:
    text = clean_fragment(grade)
    canonical_class = canonicalize_material_class(material_class)
    if not canonical_class:
        return text
    roman = canonical_class.removeprefix("Gr.")
    text = re.sub(
        rf"(?i)[\s,._/-]*GR(?:ADE)?\.?\s*{roman}\s*$",
        "",
        text,
    )
    text = re.sub(rf"(?i)[\s,._/-]+{roman}\s*$", "", text)
    if text.endswith(roman):
        text = text[: -len(roman)]
    return clean_fragment(text).strip(" .,_/-")


def canonicalize_material_grade(
    standard: str,
    grade: str,
    material_class: str = "",
) -> str:
    """Preserve Gr. only when the ASTM grade has no designation-family prefix."""
    canonical_standard = clean_fragment(standard).upper()
    text = strip_material_class_from_grade(grade, material_class)
    # Strong OCR correction for the known 15CrMo/15CrMoG designation family.
    text = re.sub(r"^[lI](?=5(?i:CrMoG?)$)", "1", text)
    text = re.sub(r"(?i)^GR(?:ADE)?\.?\s*", "", text).strip()
    text = re.sub(
        r"(?i)^(WPHY|WPS|WPL|WP|TP|LF|CF|F|P)[\s._-]+(?=[A-Z0-9])",
        lambda match: match.group(1).upper(),
        text,
    )
    if canonical_standard == "API 5L" and text.upper() == "BN":
        text = "B"
    if canonical_standard not in BARE_GRADE_STANDARDS:
        return text
    if re.fullmatch(r"(?i)[A-Z]|\d+(?:\.\d+)?[A-Z]?", text):
        return f"Gr.{text.upper()}"
    return text


def canonicalize_material_class(material_class: str) -> str:
    text = clean_fragment(material_class)
    text = re.sub(r"(?i)^GR(?:ADE)?\.?\s*", "", text).strip()
    roman = ROMAN_MAP.get(text, text.upper())
    if roman in {"I", "II", "III"}:
        return f"Gr.{roman}"
    return clean_fragment(material_class)


def extract_grade_from_standard_context(
    full_text: str,
    standard: str,
) -> str:
    match = re.fullmatch(r"ASTM ([AB])(\d+)", standard)
    if not match:
        return ""
    letter, number = match.groups()
    standard_token = (
        rf"(?:ASTM[\s._/-]*)?{letter}[\s._/-]*{number}(?:M)?"
    )
    if standard == "ASTM A403":
        class_then_alloy = re.search(
            rf"(?i){standard_token}[\s,._/-]*"
            r"WP(?:[\s._/-]*(?:WX|WU|W|S))?\s*OR\s*"
            r"WP[\s._/-]*(?:WX|WU|W|S)[\s._/-]*"
            r"(?P<alloy>304L?|304H|316L?|316H|321H?|347H?|310H?)",
            normalize_text(full_text),
        )
        if class_then_alloy:
            return f"WP{class_then_alloy.group('alloy').upper()}"

    grade_patterns: dict[str, str] = {
        "A53": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>[AB])",
        "A106": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>[ABC])",
        "A182": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>F[A-Z0-9/-]+)",
        "A234": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WP[A-Z0-9/-]+)",
        "A240": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>(?:S)?(?:304L?|304H|316L?|316H|321H?|347H?|310H?))",
        "A312": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>TP[A-Z0-9/-]+)",
        "A333": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>(?:P\d+[A-Z]?|\d+[A-Z]?))",
        "A335": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>P\d+[A-Z]?)",
        "A350": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>LF\d+)",
        "A351": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>CF[A-Z0-9]+)",
        "A358": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>(?:TP)?[A-Z]?\d+[A-Z/-]*)",
        "A387": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>\d+[A-Z]?)",
        "A403": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WP[A-Z0-9/-]+)",
        "A420": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WPL\d+[A-Z]?)",
        "A516": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>\d+[A-Z]?)",
        "A671": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>CC\d+[A-Z]?)",
        "A672": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>[A-Z]?\d+[A-Z]?)",
        "A694": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>F\d+[A-Z]?)",
        "A790": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>S\d{5})",
        "A815": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WPS[A-Z0-9]+)",
        "A860": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WPHY\d+[A-Z]?)",
        "B366": r"(?:GR(?:ADE)?\.?\s*)?(?P<grade>WP[A-Z0-9]+)",
    }
    grade_pattern = grade_patterns.get(f"{letter}{number}")
    if not grade_pattern:
        return ""
    context = re.search(
        rf"(?i){standard_token}[\s,._/-]*{grade_pattern}",
        normalize_text(full_text),
    )
    if not context:
        return ""
    return sanitize_grade_boundary(
        strip_ignored_astm_suffixes(context.group("grade"))
    )


def parse_api_expression(raw: str, full_text: str) -> tuple[str, str] | None:
    raw_matches = list(API_5L_RE.finditer(raw))
    full_matches = list(API_5L_RE.finditer(full_text))
    raw_match = next(
        (match for match in raw_matches if match.group("grade")),
        raw_matches[0] if raw_matches else None,
    )
    full_match = next(
        (match for match in full_matches if match.group("grade")),
        full_matches[0] if full_matches else None,
    )
    match = (
        raw_match
        if raw_match and raw_match.group("grade")
        else full_match or raw_match
    )
    if not match:
        return None
    grade = (match.group("grade") or "").upper()
    if not grade:
        candidates = {
            candidate.group("grade").upper()
            for candidate in API_5L_GRADE_RE.finditer(full_text)
        }
        if len(candidates) == 1:
            grade = candidates.pop()
    return "API 5L", grade


def parse_standard_expression(
    raw: str,
    full_text: str,
    target: str,
) -> tuple[str, str, str] | None:
    api = parse_api_expression(raw, full_text)
    if api:
        return api[0], api[1], ""

    normalized = normalize_material_standard_boundaries(raw)
    match = MATERIAL_STANDARD_RE.search(normalized)
    if not match:
        return None

    letter = (
        match.group("astm_letter")
        or match.group("plain_letter")
        or "A"
    ).upper()
    number = match.group("astm_number") or match.group("plain_number")
    standard_code = f"{letter}{number}"
    if standard_code not in MATERIAL_STANDARD_NUMBERS:
        return None

    standard = canonical_material_standard(letter, number)
    remainder = normalized[match.end() :]
    remainder = strip_special_tokens(remainder)
    remainder = re.sub(r"(?i)^\s*GR(?:ADE)?\.?\s*", "", remainder)
    remainder = strip_ignored_astm_suffixes(remainder)
    grade, material_class = extract_class(remainder, target)
    if not grade:
        grade = extract_grade_from_standard_context(full_text, standard)
    grade = sanitize_grade_boundary(grade)

    if standard == "ASTM A105" and grade.upper() == "N":
        grade = ""
    return standard, grade, material_class


def normalize_standalone_grade(raw: str, target: str) -> tuple[str, str]:
    grade = strip_special_tokens(raw)
    grade = re.sub(r"(?i)\(\s*(?:NACE|HIC|ANTI[^)]*)\s*\)", "", grade)
    grade = grade.rstrip("#").strip()
    grade, material_class = extract_class(grade, target)
    return strip_ignored_astm_suffixes(grade), material_class


def canonicalize_dual_grade(grade: str, standard: str) -> str:
    parts = [re.sub(r"\s+", "", part) for part in grade.split("/")]
    if len(parts) != 2 or not all(parts):
        return re.sub(r"\s+", "", grade)

    standard_prefix = {
        "ASTM A182": "F",
        "ASTM A312": "TP",
        "ASTM A403": "WP",
    }.get(standard, "")
    if standard_prefix:
        parts = [
            part
            if part.upper().startswith(standard_prefix)
            else f"{standard_prefix}{part}"
            for part in parts
        ]
    return "/".join(parts)


def split_dual_grade_items(
    grade: str,
    standard: str,
    material_class: str,
    special: list[str],
) -> list[dict[str, Any]]:
    canonical = canonicalize_dual_grade(grade, standard)
    return [
        make_item("BODY", standard, part, material_class, special)
        for part in canonical.split("/")
        if part
    ]


def make_item(
    part: str,
    standard: str = "",
    grade: str = "",
    material_class: str = "",
    special: list[str] | None = None,
) -> dict[str, Any]:
    if part not in ALLOWED_PARTS:
        raise ValueError(f"unsupported material PART: {part}")
    canonical_standard = clean_fragment(standard)
    canonical_class = canonicalize_material_class(material_class)
    return {
        "PART": part,
        "STANDARD": canonical_standard,
        "GRADE": canonicalize_material_grade(
            canonical_standard,
            grade,
            canonical_class,
        ),
        "CLASS": canonical_class,
        "SPECIAL_REQ": canonicalize_special_requirements(special or []),
    }


def canonicalize_special_requirements(requirements: list[str]) -> list[str]:
    canonical: list[str] = []
    for requirement in requirements:
        normalized = clean_fragment(requirement).upper()
        if normalized == "PLASTIC_DIPPING":
            continue
        if normalized == "HIC":
            normalized = "ANTI-HIC"
        if normalized and normalized not in canonical:
            canonical.append(normalized)
    return canonical


def parse_atomic_item(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
    part: str = "BODY",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    special = extract_special_requirements(text)
    hit, status = resolve_atomic_alias(text, target, alias_rules)
    if not hit:
        context_candidates: list[tuple[str, str]] = []
        for candidate_standard in find_material_standards(text):
            candidate_grade = extract_grade_from_standard_context(
                text,
                candidate_standard,
            )
            if candidate_grade:
                context_candidates.append(
                    (candidate_standard, candidate_grade)
                )
        if len(context_candidates) == 1:
            candidate_standard, candidate_grade = context_candidates[0]
            target_root, _ = target_root_and_modifiers(target, alias_rules)
            grade_root = re.sub(
                r"(?i)^(?:WPHY|WPS|WPL|WP|TP|LF|CF|CC|F|P)",
                "",
                candidate_grade,
            )
            grade_code_alias = re.sub(
                r"(?i)^CC(?=\d)",
                "C",
                candidate_grade,
            )
            if compact(target_root) in {
                compact(candidate_grade),
                compact(grade_root),
                compact(grade_code_alias),
            }:
                return (
                    make_item(
                        part,
                        candidate_standard,
                        candidate_grade,
                        special=special,
                    ),
                    {
                        "reason": "unique_material_standard_context",
                        "target": target,
                        "standard": candidate_standard,
                        "grade": candidate_grade,
                    },
                )
        return None, {"reason": status, "target": target}

    standard_expression = parse_standard_expression(hit.raw, text, target)
    if standard_expression:
        standard, grade, material_class = standard_expression
    else:
        standard = ""
        grade, material_class = normalize_standalone_grade(hit.raw, target)
        standard = infer_material_standard_from_context(text, grade)
        context_grade = extract_grade_from_standard_context(text, standard)
        if context_grade:
            grade = context_grade

    preserved_grade = extract_source_preserved_grade(text, hit)
    if preserved_grade:
        standard = ""
        grade = preserved_grade
        material_class = ""

    if grade and not material_class:
        roman_suffix = re.search(
            rf"(?i)(?<![A-Z0-9]){re.escape(grade)}\s*(Ⅱ|Ⅲ|II|III)(?![A-Z0-9])",
            normalize_text(text),
        )
        if roman_suffix:
            material_class = ROMAN_MAP.get(
                roman_suffix.group(1),
                roman_suffix.group(1).upper(),
            )

    _, modifiers = target_root_and_modifiers(target, alias_rules)
    class_modifiers = [
        modifier for modifier in modifiers if modifier in {"I", "II", "III"}
    ]
    if class_modifiers and not material_class:
        material_class = class_modifiers[0]

    if "CE" in modifiers and not any(
        req in special
        for req in ("NACE", "ANTI-H2S", "ANTI-HIC", "ANTI-SCC", "CE")
    ):
        return None, {
            "reason": "normalized_ce_without_source_requirement",
            "target": target,
            "raw": hit.raw,
        }
    if "ZN" in modifiers and "GALVANIZED" not in special:
        return None, {
            "reason": "normalized_zn_without_source_requirement",
            "target": target,
            "raw": hit.raw,
        }

    if not standard and not grade:
        return None, {
            "reason": "empty_atomic_material",
            "target": target,
            "raw": hit.raw,
        }

    return (
        make_item(part, standard, grade, material_class, special),
        {
            "reason": status,
            "target": target,
            "alias": hit.alias,
            "raw": hit.raw,
        },
    )


def parse_dual_grade_expression(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult:
    hit, status = resolve_atomic_alias(text, target, alias_rules)
    raw = hit.raw if hit else ""
    if not raw:
        pair = re.search(
            r"(?i)(?:DUAL\s*)?"
            r"(?P<first>(?:WP|TP|F|S)?(?:304L?|316L?))"
            r"\s*/\s*"
            r"(?P<second>(?:WP|TP|F|S)?(?:304L?|316L?))",
            normalize_text(text),
        )
        if pair:
            raw = pair.group(0)
            status = "dual_grade_expression"
    if not raw:
        return ConversionResult(None, "dual_grade_alias_not_found", {"target": target})

    special = extract_special_requirements(text)
    standard_expression = parse_standard_expression(raw, text, target)
    if standard_expression:
        standard, grade, material_class = standard_expression
    else:
        standard = ""
        grade, material_class = normalize_standalone_grade(raw, target)
        standard = infer_material_standard_from_context(text, grade)

    if "/" not in raw and "/" not in grade:
        return ConversionResult(
            None,
            "dual_grade_source_expression_missing",
            {"target": target, "raw": raw},
        )
    grade = re.sub(r"(?i)^DUAL\s*", "", grade).strip()
    if not grade:
        return ConversionResult(None, "dual_grade_empty", {"target": target, "raw": raw})

    standards = find_material_standards(text)
    if len(standards) >= 2:
        alternative_items = []
        for candidate_standard in standards:
            candidate_grade = extract_grade_from_standard_context(
                text,
                candidate_standard,
            )
            if candidate_grade:
                alternative_items.append(
                    make_item(
                        "BODY",
                        candidate_standard,
                        candidate_grade,
                        material_class,
                        special,
                    )
                )
        if len(alternative_items) >= 2:
            return ConversionResult(
                alternative_items,
                "converted_alternative_material",
                {
                    "target": target,
                    "raw": raw,
                    "standards": standards,
                    "alias_status": status,
                },
                "ALTERNATIVE",
            )

    items = split_dual_grade_items(
        grade,
        standard,
        material_class,
        special,
    )
    if len(items) != 2:
        return ConversionResult(
            None,
            "dual_grade_empty",
            {"target": target, "raw": raw, "grade": grade},
        )

    return ConversionResult(
        items,
        "converted_dual_grade",
        {"target": target, "raw": raw, "alias_status": status},
        "DUAL_CERTIFIED",
    )


def find_lining(text: str, preferred: str = "") -> tuple[str, str] | None:
    preferred_upper = preferred.upper()
    candidates: list[tuple[int, str, str]] = []
    for canonical, pattern in LINING_PATTERNS:
        for match in pattern.finditer(normalize_text(text)):
            score = 0 if canonical == preferred_upper else 1
            candidates.append((score * 100000 + match.start(), canonical, match.group(0)))
    if not candidates:
        return None
    _, canonical, raw = min(candidates)
    return canonical, clean_fragment(raw)


def is_explicit_lining_context(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)衬|LIN(?:ED|ING)|内衬|INSIDE|INNER|CERAMIC|搪玻璃|CEMENT",
            text,
        )
    )


def has_explicit_lining_separator(text: str) -> bool:
    lining_token = (
        r"(?:R?M?E?PTFE|PP|PE|EAA|CPVC|PVC|RUBBER|CERAMIC|CEMENT|"
        r"GLASS|衬胶|陶瓷|搪玻璃|聚四氟乙烯|四氟)"
    )
    return bool(
        re.search(
            rf"(?i)(?:{lining_token}\s*[/+]|[/+]\s*{lining_token})",
            normalize_text(text),
        )
    )


def parse_lining_composite(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult:
    components = [part for part in target.split("/") if part]
    lining_component = next(
        (part for part in reversed(components) if part.upper() in LINING_CODE_TOKENS),
        "",
    )
    if not lining_component:
        return ConversionResult(None, "lining_component_not_found", {"target": target})
    if not (
        is_explicit_lining_context(text)
        or has_explicit_lining_separator(text)
    ):
        return ConversionResult(
            None,
            "lining_relation_not_explicit",
            {"target": target},
        )

    body_target = target[: target.rfind("/")]
    body_item, body_evidence = parse_atomic_item(text, body_target, alias_rules)
    if not body_item and body_target == "20" and re.search(
        r"(?i)20#?\s*[/+]\s*(?:R?M?E?PTFE|PP|PE|EAA|CPVC|PVC)",
        normalize_text(text),
    ):
        body_item = make_item("BODY", grade="20")
        body_evidence = {
            "reason": "explicit_lining_pair",
            "target": body_target,
            "raw": "20",
        }
    if not body_item:
        return ConversionResult(
            None,
            "lining_body_not_resolved",
            {"target": target, "body": body_evidence},
        )

    lining = find_lining(text, lining_component)
    if not lining:
        return ConversionResult(
            None,
            "lining_source_token_not_found",
            {"target": target},
        )
    canonical, raw = lining
    lining_grade = raw if raw.upper() not in {"四氟", "聚四氟乙烯"} else "PTFE"
    if canonical in {"GLASS", "CEMENT", "衬胶"}:
        lining_grade = canonical
    return ConversionResult(
        [
            body_item,
            make_item("LINING", grade=lining_grade),
        ],
        "converted_lining_composite",
        {"target": target, "body": body_evidence, "lining_raw": raw},
    )


def parse_a106_a53_alternative(text: str) -> ConversionResult:
    normalized = normalize_text(text)
    if not re.search(r"(?i)\bOR\b|或", normalized):
        return ConversionResult(
            None,
            "alternative_relation_not_explicit",
            {"target": "A106/A53"},
        )
    if not re.search(r"(?i)A[\s.]*106", normalized) or not re.search(
        r"(?i)A[\s.]*53",
        normalized,
    ):
        return ConversionResult(
            None,
            "alternative_standard_missing",
            {"target": "A106/A53"},
        )
    grades = re.findall(
        r"(?i)A[\s.]*(?:106|53)[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?([ABC])",
        normalized,
    )
    grade = grades[0].upper() if grades and len(set(x.upper() for x in grades)) == 1 else ""
    return ConversionResult(
        [
            make_item("BODY", standard="ASTM A106", grade=grade),
            make_item("BODY", standard="ASTM A53", grade=grade),
        ],
        "converted_alternative_standard",
        {"target": "A106/A53", "grades": grades},
        "ALTERNATIVE",
    )


def parse_inner_outer_composite(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult:
    if target not in {"304/20", "304/2200", "304/N02200"}:
        return ConversionResult(None, "not_inner_outer_target", {"target": target})
    has_explicit_inner_outer = (
        re.search(r"(?i)\bIN(?:NER)?\s*[:：]|内管|内层", text)
        and re.search(r"(?i)\bOUT(?:ER)?\s*[:：]|外管|外层", text)
    )
    if not has_explicit_inner_outer and "夹套管" not in text:
        return ConversionResult(
            None,
            "inner_outer_relation_not_explicit",
            {"target": target},
        )

    components = target.split("/")
    items: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        part = "INNER_PIPE" if index == 0 else "OUTER_PIPE"
        item, item_evidence = parse_atomic_item(text, component, alias_rules, part)
        if (
            not item
            and target == "304/20"
            and component == "20"
            and re.search(
                r"(?i)(?:S?3040?8?|06Cr19Ni10)\s*/\s*20",
                normalize_text(text),
            )
        ):
            item = make_item(part, grade="20")
            item_evidence = {
                "reason": "explicit_inner_outer_pair",
                "target": component,
                "raw": "20",
            }
        if not item:
            return ConversionResult(
                None,
                "inner_outer_component_not_resolved",
                {"target": target, "component": component, "detail": item_evidence},
            )
        items.append(item)
        evidence.append(item_evidence)
    return ConversionResult(
        items,
        "converted_inner_outer",
        {"target": target, "components": evidence},
    )


def parse_loose_flange_pair(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult:
    if target not in {
        "20/TA2",
        "20/304",
        "304/N02200",
        "304/2200",
    }:
        return ConversionResult(None, "not_loose_flange_pair", {"target": target})
    if not re.search(r"(?i)PJ\s*/\s*SE|LAP\s*JOINT|LOOSE\s*FLANGE|松套法兰", text):
        return ConversionResult(
            None,
            "loose_flange_relation_not_explicit",
            {"target": target},
        )

    flange_target, body_target = target.split("/", 1)
    flange_item, flange_evidence = parse_atomic_item(
        text,
        flange_target,
        alias_rules,
        "FLANGE",
    )
    body_item, body_evidence = parse_atomic_item(
        text,
        body_target,
        alias_rules,
        "BODY",
    )
    if not flange_item or not body_item:
        return ConversionResult(
            None,
            "loose_flange_component_not_resolved",
            {
                "target": target,
                "flange": flange_evidence,
                "body": body_evidence,
            },
        )
    return ConversionResult(
        [flange_item, body_item],
        "converted_loose_flange_pair",
        {
            "target": target,
            "flange": flange_evidence,
            "body": body_evidence,
        },
    )


def detect_weld_ring_loose_flange(text: str) -> str:
    """Return source evidence when the item itself is a weld-ring loose flange."""

    chinese = re.search(
        r"焊环[^,;]{0,24}松套法兰|松套法兰[^,;]{0,24}焊环",
        text,
        re.IGNORECASE,
    )
    if chinese:
        return chinese.group(0)

    type_marker = re.search(
        r"(?i)(?<![A-Z0-9])(?:PJ\s*/\s*SE|PL\s*/\s*C)(?![A-Z0-9])",
        text,
    )
    lap_joint = re.search(r"(?i)\bLAP\s*JOINT\s*FLANG(?:E)?\b", text)
    if type_marker and lap_joint:
        return f"{type_marker.group(0)} + {lap_joint.group(0)}"
    return ""


def parse_known_multi_part_material(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult | None:
    """Parse rare structures only when their part relation is explicit."""

    normalized = normalize_text(text)
    special = extract_special_requirements(normalized)

    a234_plate = re.search(
        r"(?i)(?:ASTM\s*)?A[\s._/-]*234"
        r"[\s._/-]*(?:GR(?:ADE)?\.?\s*)?WPB(?:[\s._/-]*W)?",
        normalized,
    )
    a516_plate = re.search(
        r"(?i)\(\s*(?:ASTM\s*)?A[\s._/-]*516"
        r"[\s._/-]*(?:GR(?:ADE)?\.?\s*)?65\s*\)",
        normalized,
    )
    if compact(target) == "A516" and a234_plate and a516_plate:
        return ConversionResult(
            [
                make_item("BODY", standard="ASTM A234", grade="WPB"),
                make_item("BODY", standard="ASTM A516", grade="65"),
            ],
            "converted_welded_fitting_plate_composite",
            {
                "target": target,
                "fitting_material_raw": a234_plate.group(0),
                "plate_material_raw": a516_plate.group(0),
            },
            "COMPOSITE",
        )

    a403_primary = re.search(
        r"(?i)(?:ASTM\s*)?A[\s._/-]*403"
        r"[\s._/-]*(?:GR(?:ADE)?\.?\s*)?"
        r"(?P<grade>WP[\s._/-]*304)"
        r"\s*\(\s*UNS\s*S30400\s*\)",
        normalized,
    )
    if a403_primary and compact(target) in {"304", "S30400"}:
        return ConversionResult(
            [make_item("BODY", standard="ASTM A403", grade="WP304")],
            "converted_atomic",
            {
                "target": target,
                "formal_grade_raw": a403_primary.group("grade"),
                "ignored_parenthetical_alias": "UNS S30400",
            },
            "SINGLE",
        )

    if compact(target) == "MR0103" and "NACE" in special:
        candidates: list[tuple[str, str]] = []
        for standard in find_material_standards(normalized):
            grade = extract_grade_from_standard_context(normalized, standard)
            if grade:
                candidates.append((standard, grade))
        if len(candidates) == 1:
            standard, grade = candidates[0]
            return ConversionResult(
                [
                    make_item(
                        "BODY",
                        standard=standard,
                        grade=grade,
                        special=special,
                    )
                ],
                "converted_nace_mr_material_context",
                {
                    "target": target,
                    "standard": standard,
                    "grade": grade,
                },
                "SINGLE",
            )

    if compact(target) == "20PTFE":
        pair = re.search(
            r"(?i)(?<![A-Z0-9])20#?\s*[/+]\s*PTFE(?![A-Z0-9])",
            normalized,
        )
        if pair:
            return ConversionResult(
                [
                    make_item("BODY", grade="20"),
                    make_item("LINING", grade="PTFE"),
                ],
                "converted_lining_composite",
                {
                    "target": target,
                    "raw": pair.group(0),
                    "lining_raw": "PTFE",
                },
                "COMPOSITE",
            )

    coating_requirements = [
        requirement for requirement in special if requirement in {"PE", "EP"}
    ]
    if coating_requirements and target in {"20", "Q235B"}:
        material_pattern = (
            r"(?<![A-Z0-9])20#?(?![A-Z0-9])"
            if target == "20"
            else r"(?<![A-Z0-9])Q235B(?![A-Z0-9])"
        )
        material_match = re.search(material_pattern, normalized, re.IGNORECASE)
        if material_match:
            return ConversionResult(
                [
                    make_item(
                        "BODY",
                        grade=target,
                        special=coating_requirements,
                    )
                ],
                "converted_coating_requirement",
                {
                    "target": target,
                    "raw": material_match.group(0),
                    "coating_requirements": coating_requirements,
                },
                "SINGLE",
            )

    pipeline_grade_aliases = {
        "L245": "B",
        "L360M": "X52M",
    }
    if target in pipeline_grade_aliases:
        alias = pipeline_grade_aliases[target]
        pair = re.search(
            rf"(?i)(?<![A-Z0-9]){re.escape(target)}\s*/\s*"
            rf"{re.escape(alias)}(?![A-Z0-9])",
            normalized,
        )
        if pair:
            return ConversionResult(
                [make_item("BODY", grade=target, special=special)],
                "converted_pipeline_grade_alias",
                {"target": target, "raw": pair.group(0)},
                "SINGLE",
            )

    if target == "Q235B/304":
        pair = re.search(
            r"(?i)(?P<body>Q235B)\s*[+/]\s*(?P<lining>S30408|304)",
            normalized,
        )
        if not pair or not re.search(r"(?i)CJ\s*/?\s*T\s*192|内衬不锈钢", normalized):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", grade=pair.group("body").upper()),
                make_item("LINING", grade=pair.group("lining").upper()),
            ],
            "converted_stainless_lined_pipe",
            {"target": target, "raw": pair.group(0)},
            "COMPOSITE",
        )

    if target == "20/304" and re.search(r"(?i)衬里|LIN(?:ED|ING)", normalized):
        pair = re.search(
            r"(?i)(?P<body>20#?)\s*[+/]\s*(?P<lining>S30408|304)",
            normalized,
        )
        if not pair:
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", grade="20"),
                make_item("LINING", grade=pair.group("lining").upper()),
            ],
            "converted_stainless_lined_pipe",
            {"target": target, "raw": pair.group(0)},
            "COMPOSITE",
        )

    if target == "WPB/304":
        wpb = re.search(
            r"(?i)A[\s._/-]*234[\s._/-]*WPB(?:[\s._/-]*S)?",
            normalized,
        )
        stainless = re.search(
            r"(?i)(?:WP[\s._/-]*(?:WX|WU|W|S)[\s._/-]*)?304",
            normalized,
        )
        if not wpb or not stainless or not re.search(r"(?i)\bOR\b|或", normalized):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", standard="ASTM A234", grade="WPB"),
                make_item("BODY", grade="304"),
            ],
            "converted_alternative_material",
            {
                "target": target,
                "wpb_raw": wpb.group(0),
                "stainless_raw": stainless.group(0),
            },
            "ALTERNATIVE",
        )

    if target == "20GLA105":
        if not (
            re.search(r"(?i)(?<!\d)20\W+GLASS\W*LINED", normalized)
            and re.search(r"(?i)LOOSE\W*FLANGE\W*A[\s.-]*105", normalized)
        ):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", grade="20"),
                make_item("LINING", grade="GLASS"),
                make_item("FLANGE", standard="ASTM A105"),
            ],
            "converted_glass_lined_loose_flange",
            {"target": target},
        )

    if target in {"A106A105/C", "WPB/A105"}:
        body_match = re.search(
            r"(?i)(?P<standard>A[\s.-]*(?:106|234))"
            r"[\s.-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(?P<grade>B|WPB)\s*\+\s*"
            r"A[\s.-]*105\s*/\s*CERAMIC",
            normalized,
        )
        if not body_match or not re.search(
            r"(?i)(?:FIX(?:ED)?\s*FLG|FIX(?:ED)?\s*FLANGE|固定法兰)",
            normalized,
        ):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        standard_number = re.sub(
            r"[^0-9]",
            "",
            body_match.group("standard"),
        )
        return ConversionResult(
            [
                make_item(
                    "BODY",
                    standard=f"ASTM A{standard_number}",
                    grade=body_match.group("grade").upper(),
                ),
                make_item("LINING", grade="CERAMIC"),
                make_item("FLANGE", standard="ASTM A105"),
            ],
            "converted_ceramic_fixed_flange",
            {"target": target, "raw": body_match.group(0)},
        )

    if target == "FRP/A105ZN":
        if not (
            re.search(r"(?i)(?<![A-Z0-9])FRP(?![A-Z0-9])", normalized)
            and re.search(
                r"(?i)(?:LAP\s*JOINT\s*FLANGE|LOOSE\s*FLANGE|松套法兰)"
                r"[^;]*A[\s.-]*105[^;]*(?:GALV|ZN|镀锌)",
                normalized,
            )
        ):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", grade="FRP"),
                make_item(
                    "FLANGE",
                    standard="ASTM A105",
                    special=["GALVANIZED"],
                ),
            ],
            "converted_frp_galvanized_flange",
            {"target": target},
        )

    if target == "1-1/4Cr-1/2Mo/410":
        alloy = re.search(
            r"(?i)(?P<body>1[\s-]*1/4Cr[\s-]*1/2Mo)\s*[/+]\s*(?P<lining>410)",
            normalized,
        )
        if not alloy or not is_explicit_lining_context(normalized):
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        body_grade = re.sub(r"\s+", "-", alloy.group("body"))
        body_grade = re.sub(r"-+", "-", body_grade)
        return ConversionResult(
            [
                make_item("BODY", grade=body_grade),
                make_item("LINING", grade=alloy.group("lining")),
            ],
            "converted_fractional_alloy_lining",
            {"target": target, "raw": alloy.group(0)},
        )

    cf_identifier = re.search(
        r"(?i)(?<![A-Z0-9])CF(?:370|415|418|420)(?![A-Z0-9])",
        normalized,
    )
    core_20_matches = list(
        re.finditer(r"(?i)(?<![A-Z0-9])20#?(?![A-Z0-9])", normalized)
    )
    core_q235b = re.search(
        r"(?i)(?<![A-Z0-9])Q235B(?![A-Z0-9])", normalized
    )
    target_token = compact(target)

    # CF370/CF415/CF418/CF420 are supplementary identifiers when an explicit
    # core grade is present. Keep standalone CF grades unchanged.
    if cf_identifier and core_20_matches and target_token in {
        "20",
        "20CF415",
        "CF415",
        "CF418",
        "CF420",
    }:
        if "夹套" in normalized and len(core_20_matches) >= 2:
            return ConversionResult(
                [
                    make_item("INNER_PIPE", grade="20"),
                    make_item("OUTER_PIPE", grade="20"),
                ],
                "converted_jacket_core_materials",
                {
                    "target": target,
                    "ignored_cf_identifiers": re.findall(
                        r"(?i)(?<![A-Z0-9])CF(?:370|415|418|420)(?![A-Z0-9])",
                        normalized,
                    ),
                },
                "COMPOSITE",
            )
        return ConversionResult(
            [make_item("BODY", grade="20")],
            "converted_core_material_ignoring_cf_identifier",
            {"target": target, "ignored_cf_identifier": cf_identifier.group(0)},
            "SINGLE",
        )

    if cf_identifier and core_q235b and target_token in {
        "Q235B",
        "Q235BCF370",
        "CF370",
    }:
        return ConversionResult(
            [make_item("BODY", grade="Q235B")],
            "converted_core_material_ignoring_cf_identifier",
            {"target": target, "ignored_cf_identifier": cf_identifier.group(0)},
            "SINGLE",
        )

    if target == "WPBHDPP":
        match = re.search(
            r"(?i)(?:ASTM\s*)?A[\s.-]*234"
            r"[\s.-]*(?:GR(?:ADE)?\.?\s*)?WPB\s*\+\s*HDPP",
            normalized,
        )
        if not match:
            return ConversionResult(
                None,
                "known_structure_source_evidence_missing",
                {"target": target},
            )
        return ConversionResult(
            [
                make_item("BODY", standard="ASTM A234", grade="WPB"),
                make_item("LINING", grade="HDPP"),
            ],
            "converted_hdpp_lining",
            {"target": target, "raw": match.group(0)},
        )

    return None


def parse_generic_composite(
    text: str,
    target: str,
    alias_rules: dict[str, list[AliasRule]],
) -> ConversionResult:
    known_structure = parse_known_multi_part_material(
        text,
        target,
        alias_rules,
    )
    if known_structure is not None:
        return known_structure

    if target in SLASH_ATOMIC_CODES or re.fullmatch(
        r"\d+(?:-\d+)?/\d+Cr(?:-\d+/\d+Mo)?",
        target,
        re.IGNORECASE,
    ):
        item, evidence = parse_atomic_item(text, target, alias_rules)
        return ConversionResult(
            [item] if item else None,
            "converted_fractional_alloy" if item else evidence.get("reason", "fractional_alloy_failed"),
            evidence,
        )
    if target in DUAL_GRADE_CODES:
        return parse_dual_grade_expression(text, target, alias_rules)
    if target == "A106/A53":
        return parse_a106_a53_alternative(text)

    lining_result = parse_lining_composite(text, target, alias_rules)
    if lining_result.material:
        return lining_result

    inner_outer = parse_inner_outer_composite(text, target, alias_rules)
    if inner_outer.material:
        return inner_outer

    loose_flange = parse_loose_flange_pair(text, target, alias_rules)
    if loose_flange.material:
        return loose_flange

    return ConversionResult(
        None,
        "unresolved_composite_material",
        {
            "target": target,
            "lining": lining_result.status,
            "inner_outer": inner_outer.status,
            "loose_flange": loose_flange.status,
        },
    )


def validate_material_items(items: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_keys = {"PART", "STANDARD", "GRADE", "CLASS", "SPECIAL_REQ"}
    for index, item in enumerate(items):
        if set(item) != expected_keys:
            errors.append(f"item[{index}] keys={sorted(item)}")
        if item.get("PART") not in ALLOWED_PARTS:
            errors.append(f"item[{index}] invalid PART={item.get('PART')}")
        if not item.get("STANDARD") and not item.get("GRADE"):
            errors.append(f"item[{index}] empty material")
        if not isinstance(item.get("SPECIAL_REQ"), list):
            errors.append(f"item[{index}] SPECIAL_REQ is not list")
        grade = clean_fragment(item.get("GRADE"))
        if "/" in grade and not re.fullmatch(
            r"(?i)(?:\d+(?:\s+|[-]))?\d+/\d+Cr(?:-\d+/\d+Mo)?",
            grade,
        ):
            errors.append(
                f"item[{index}] GRADE contains multiple designations={grade}"
            )
    return errors


def resolve_material_relation(
    items: list[dict[str, Any]],
    explicit_relation: str,
) -> tuple[str, list[str]]:
    relation = clean_fragment(explicit_relation).upper()
    if not relation:
        relation = (
            "SINGLE"
            if len(items) == 1
            else "COMPOSITE"
            if len({item.get("PART") for item in items}) > 1
            else ""
        )

    errors: list[str] = []
    if relation not in ALLOWED_RELATIONS:
        errors.append(f"invalid MATERIAL_RELATION={relation or '<empty>'}")
        return relation, errors

    parts = [item.get("PART") for item in items]
    if relation == "SINGLE" and len(items) != 1:
        errors.append("SINGLE relation requires exactly one material item")
    elif relation == "COMPOSITE":
        if len(items) < 2:
            errors.append(
                "COMPOSITE relation requires at least two material items"
            )
        same_body_product_and_plate = (
            set(parts) == {"BODY"}
            and {item.get("STANDARD") for item in items}
            == {"ASTM A234", "ASTM A516"}
        )
        if len(set(parts)) < 2 and not same_body_product_and_plate:
            errors.append(
                "COMPOSITE relation requires different physical PART values"
            )
    elif relation in {"DUAL_CERTIFIED", "ALTERNATIVE", "EQUIVALENT"}:
        if len(items) < 2:
            errors.append(
                f"{relation} relation requires at least two material items"
            )
        if len(set(parts)) != 1:
            errors.append(
                f"{relation} relation requires the same physical PART"
            )
    return relation, errors


def load_confirmed_proposals(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"已确认建议文件格式错误: {path}")

    confirmed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"已确认建议包含非对象记录: {path}")
        split = str(row.get("split", ""))
        source_index = row.get("source_index")
        if split not in {"train", "val"} or not isinstance(source_index, int):
            raise ValueError(f"已确认建议缺少有效 split/source_index: {row}")
        key = confirmed_proposal_key(
            split,
            row.get("input"),
            row.get("current_source_material"),
        )
        if key in confirmed:
            raise ValueError(f"已确认建议存在重复内容快照: {key}")
        confirmed[key] = row
    return confirmed


def confirmed_proposal_key(
    split: str,
    input_text: Any,
    source_material: Any,
) -> tuple[str, str, str]:
    material_snapshot = json.dumps(
        source_material if isinstance(source_material, list) else [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return split, normalize_text(input_text), material_snapshot


def confirmed_proposal_result(
    source_row: dict[str, Any],
    confirmed_row: dict[str, Any],
) -> ConversionResult:
    if source_row.get("input") != confirmed_row.get("input"):
        raise ValueError(
            "已确认建议原文快照与当前源数据不一致: "
            f"{confirmed_row.get('split')}[{confirmed_row.get('source_index')}]"
        )

    source_material = source_row.get("output", {}).get("MATERIAL", [])
    if source_material != confirmed_row.get("current_source_material"):
        raise ValueError(
            "已确认建议材质快照与当前源数据不一致: "
            f"{confirmed_row.get('split')}[{confirmed_row.get('source_index')}]"
        )

    proposal = confirmed_row.get("proposal")
    if not isinstance(proposal, dict):
        raise ValueError("已确认建议缺少 proposal")
    material = proposal.get("MATERIAL")
    relation = clean_fragment(proposal.get("MATERIAL_RELATION")).upper()
    if not isinstance(material, list) or not material:
        raise ValueError("已确认建议缺少 MATERIAL")

    normalized_material = [
        make_item(
            clean_fragment(item.get("PART")),
            clean_fragment(item.get("STANDARD")),
            clean_fragment(item.get("GRADE")),
            clean_fragment(item.get("CLASS")),
            item.get("SPECIAL_REQ") if isinstance(item.get("SPECIAL_REQ"), list) else [],
        )
        for item in material
    ]
    errors = validate_material_items(normalized_material)
    resolved_relation, relation_errors = resolve_material_relation(
        normalized_material,
        relation,
    )
    errors.extend(relation_errors)
    if errors:
        raise ValueError(f"已确认建议结构无效: {errors}")
    return ConversionResult(
        normalized_material,
        "merged_confirmed_proposal",
        {
            "confidence": proposal.get("confidence", ""),
            "rule": proposal.get("rule", ""),
        },
        resolved_relation,
    )


class MaterialConverter:
    def __init__(self, mapping_path: Path) -> None:
        self.alias_rules = load_alias_rules(mapping_path)

    def convert(self, row: dict[str, Any]) -> ConversionResult:
        text = normalize_text(row.get("input"))
        weld_ring_marker = detect_weld_ring_loose_flange(text)
        if weld_ring_marker:
            return ConversionResult(
                None,
                "excluded_weld_ring_loose_flange",
                {
                    "matched_marker": weld_ring_marker,
                    "policy": "初版训练集暂不处理焊环与松套法兰盘的材质归属及编码顺序",
                },
            )

        output = row.get("output")
        if not isinstance(output, dict):
            return ConversionResult(None, "invalid_output", {})
        materials = output.get("MATERIAL")
        if not isinstance(materials, list) or len(materials) != 1:
            return ConversionResult(
                None,
                "material_cardinality_not_one",
                {"materials": materials},
            )

        source_material = materials[0]
        target = clean_fragment(source_material.get("VALUE"))
        if not target:
            return ConversionResult(None, "empty_normalized_material", {})

        known_structure = parse_known_multi_part_material(
            text,
            target,
            self.alias_rules,
        )
        if known_structure is not None:
            result = known_structure
        elif "/" in target:
            result = parse_generic_composite(text, target, self.alias_rules)
        else:
            item, evidence = parse_atomic_item(text, target, self.alias_rules)
            result = ConversionResult(
                [item] if item else None,
                "converted_atomic" if item else evidence.get("reason", "atomic_failed"),
                evidence,
            )

        if result.material:
            errors = validate_material_items(result.material)
            relation, relation_errors = resolve_material_relation(
                result.material,
                result.relation,
            )
            errors.extend(relation_errors)
            if errors:
                return ConversionResult(
                    None,
                    "converted_material_validation_failed",
                    {"errors": errors, "previous": result.evidence},
                )
            result = ConversionResult(
                result.material,
                result.status,
                result.evidence,
                relation,
            )
        return result


def convert_split(
    split: str,
    source_path: Path,
    converter: MaterialConverter,
    confirmed_proposals: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    converted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    confirmed_proposals = confirmed_proposals or {}

    for index, source_row in enumerate(rows):
        result = converter.convert(source_row)
        confirmed_row = confirmed_proposals.get(
            confirmed_proposal_key(
                split,
                source_row.get("input"),
                source_row.get("output", {}).get("MATERIAL", []),
            )
        )
        if result.status == "alias_not_found" and confirmed_row is not None:
            result = confirmed_proposal_result(source_row, confirmed_row)
        status_counts[result.status] += 1
        if result.material:
            row = copy.deepcopy(source_row)
            row["output"]["MATERIAL"] = result.material
            row["output"]["MATERIAL_RELATION"] = result.relation
            converted.append(row)
            continue

        old_materials = source_row.get("output", {}).get("MATERIAL", [])
        review.append(
            {
                "split": split,
                "source_index": index,
                "input": source_row.get("input", ""),
                "current_material": old_materials,
                "current_standard": source_row.get("output", {}).get("STANDARD", []),
                "reason": localize_status(result.status),
                "evidence": localize_evidence(result.evidence),
            }
        )
    return converted, review, status_counts


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def localize_status(status: str) -> str:
    return STATUS_ZH.get(status, status)


def localize_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                localize_status(item)
                if key == "reason" and isinstance(item, str)
                else localize_evidence(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [localize_evidence(item) for item in value]
    return value


def localize_counter(counter: Counter[str]) -> dict[str, int]:
    localized: Counter[str] = Counter()
    for status, count in counter.items():
        localized[localize_status(status)] += count
    return dict(localized.most_common())


def main() -> int:
    args = parse_args()
    converter = MaterialConverter(args.mapping)
    confirmed_proposals = load_confirmed_proposals(args.confirmed_proposals)

    train, train_review, train_status = convert_split(
        "train",
        args.train_source,
        converter,
        confirmed_proposals,
    )
    val, val_review, val_status = convert_split(
        "val",
        args.val_source,
        converter,
        confirmed_proposals,
    )
    review = train_review + val_review

    write_json(args.train_output, train)
    write_json(args.val_output, val)
    write_json(args.review_output, review)
    grouped_review: dict[str, list[dict[str, Any]]] = {}
    for row in review:
        grouped_review.setdefault(row["reason"], []).append(row)
    write_json(args.grouped_review_output, grouped_review)

    report = {
        "sources": {
            "train": str(args.train_source),
            "val": str(args.val_source),
            "mapping_dictionary": str(args.mapping),
            "confirmed_proposals": (
                str(args.confirmed_proposals)
                if args.confirmed_proposals.exists()
                else ""
            ),
        },
        "outputs": {
            "train": str(args.train_output),
            "val": str(args.val_output),
            "review": str(args.review_output),
            "grouped_review": str(args.grouped_review_output),
        },
        "policy": {
            "source_of_truth": "input",
            "old_value_usage": "candidate alias selection and validation only",
            "uncertain_rows": "excluded from converted train/val and written to review",
            "weld_ring_loose_flange": (
                "excluded from the initial train/val because component material "
                "ownership and final code ordering are not yet unified"
            ),
            "confirmed_proposals": (
                "human-confirmed proposals are merged only when input and "
                "source-material snapshots still match"
            ),
            "source_files_modified": False,
            "grade_cardinality": "each MATERIAL item contains one official designation",
            "relation_usage": {
                "SINGLE": "one material designation",
                "DUAL_CERTIFIED": "multiple certified grades for the same physical part",
                "ALTERNATIVE": "alternative grades/specifications for the same physical part",
                "EQUIVALENT": "equivalent designations for the same physical part",
                "COMPOSITE": "materials assigned to different physical parts",
            },
        },
        "schema": {
            "MATERIAL": [
                {
                    "PART": sorted(ALLOWED_PARTS),
                    "STANDARD": "ASTM/API material specification or empty",
                    "GRADE": "one source-grounded official designation; no combined grades",
                    "CLASS": "coding-relevant Gr.I/Gr.II/Gr.III or empty",
                    "SPECIAL_REQ": "source-grounded coding requirements",
                }
            ],
            "MATERIAL_RELATION": sorted(ALLOWED_RELATIONS),
            "STANDARD": "preserved from source row",
        },
        "statistics": {
            "source_train_rows": len(train) + len(train_review),
            "source_val_rows": len(val) + len(val_review),
            "converted_train_rows": len(train),
            "converted_val_rows": len(val),
            "review_rows": len(review),
            "confirmed_proposal_rows": (
                train_status.get("merged_confirmed_proposal", 0)
                + val_status.get("merged_confirmed_proposal", 0)
            ),
            "train_conversion_rate": round(
                len(train) / max(1, len(train) + len(train_review)),
                6,
            ),
            "val_conversion_rate": round(
                len(val) / max(1, len(val) + len(val_review)),
                6,
            ),
        },
        "train_status_counts": localize_counter(train_status),
        "val_status_counts": localize_counter(val_status),
        "train_relation_counts": dict(
            Counter(
                row["output"]["MATERIAL_RELATION"] for row in train
            ).most_common()
        ),
        "val_relation_counts": dict(
            Counter(
                row["output"]["MATERIAL_RELATION"] for row in val
            ).most_common()
        ),
        "review_reason_counts": dict(
            Counter(row["reason"] for row in review).most_common()
        ),
        "review_material_counts": dict(
            Counter(
                str(item.get("VALUE", ""))
                for row in review
                for item in row.get("current_material", [])
            ).most_common()
        ),
    }
    write_json(args.report_output, report)

    print(
        json.dumps(
            {
                "train": len(train),
                "val": len(val),
                "review": len(review),
                "report": str(args.report_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
