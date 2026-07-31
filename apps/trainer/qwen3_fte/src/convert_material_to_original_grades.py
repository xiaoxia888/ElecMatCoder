#!/usr/bin/env python3
"""Build material datasets that retain source standards and source grades.

The existing material datasets use normalized business codes in ``VALUE``.
This converter only emits rows whose source text contains enough evidence to
recover the original material expression. Uncertain rows are quarantined in a
separate audit file instead of teaching the model a guessed label.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import yaml


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = QWEN_ROOT / "output" / "按8类拆分数据集" / "材质规范"
OUTPUT_DIR = SOURCE_DIR / "原始牌号"
DEFAULT_TRAIN_SOURCE = SOURCE_DIR / "材质规范_train.json"
DEFAULT_VAL_SOURCE = SOURCE_DIR / "材质规范_val.json"
DEFAULT_TRAIN_OUTPUT = OUTPUT_DIR / "材质规范_原始牌号_train.json"
DEFAULT_VAL_OUTPUT = OUTPUT_DIR / "材质规范_原始牌号_val.json"
DEFAULT_AUDIT_OUTPUT = OUTPUT_DIR / "材质规范_原始牌号_待审计.json"
DEFAULT_REPORT_OUTPUT = OUTPUT_DIR / "材质规范_原始牌号_构建报告.json"
DEFAULT_MAPPING = REPO_ROOT / "src" / "encoder" / "config" / "material_mapping.yaml"
DEFAULT_RELATION_REPORT = QWEN_ROOT / "output" / "material_relation_split_full_report.json"


QUALITY_RE = re.compile(
    r"(?i)(?:\bGR(?:ADE)?\.?\s*)(III|II|I|Ⅲ|Ⅱ|Ⅰ)(?![A-Za-z0-9])"
)
TRAILING_QUALITY_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(III|II|I|Ⅲ|Ⅱ|Ⅰ)(?![A-Za-z0-9])"
)
ASTM_STANDARD_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:ASTM[\s.\-/]*)?A[\s.\-/]*(\d{2,4})(?:M)?"
)
API_5L_GRADE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])API[\s.\-/]*5L"
    r"[\s,;:/-]*(?:GR(?:ADE)?\.?\s*)?"
    r"(L\d{3}[A-Z]?|X\d{2,3}[A-Z]?|[AB])"
    r"(?![A-Za-z0-9])"
)

ROMAN_MAP = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III"}
ASTM_CLASS_SUFFIX_RE = re.compile(r"(?i)-(?:WX|WU|W|S)$")

LINING_TOKENS = {
    "PTFE",
    "RPTFE",
    "PP",
    "PE",
    "EAA",
    "PVC",
    "CPVC",
    "衬胶",
    "RUBBER",
    "GLASS LINED",
    "GL",
    "CERAMIC",
}
DUAL_GRADE_PAIRS = {
    "304/304L",
    "304L/304",
    "316/316L",
    "304/304LIII",
}
EXCLUDED_DIRTY_MATERIALS = {
    "20/WC",
    "WPB/304",
}
PENDING_COMBINATION_ORDER_MATERIALS = {
    "20/CF415",
    "20/TA2",
}


@dataclass(frozen=True)
class AliasRule:
    target: str
    alias: str
    pattern: Pattern[str]


@dataclass(frozen=True)
class RawResolution:
    raw: str
    status: str
    evidence: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建原始牌号材质训练集和验证集")
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--val-source", type=Path, default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--relation-report", type=Path, default=DEFAULT_RELATION_REPORT)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", normalize_text(value).casefold())


def normalized_code(value: str) -> str:
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]+", "", normalize_text(value).upper())


def clean_fragment(value: str) -> str:
    return normalize_text(value).strip(" \t\r\n,，;；:+")


def flexible_pattern(value: str) -> Pattern[str]:
    parts: list[str] = []
    for char in normalize_text(value):
        if char.isspace():
            parts.append(r"\s*")
        elif char in "-./":
            parts.append(rf"\s*{re.escape(char)}\s*")
        else:
            parts.append(re.escape(char))
    body = "".join(parts)
    return re.compile(
        rf"(?<![A-Za-z0-9\u4e00-\u9fff])({body})"
        rf"(?![A-Za-z0-9\u4e00-\u9fff])",
        re.IGNORECASE,
    )


def load_alias_rules(path: Path) -> dict[str, list[AliasRule]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = raw.get("value_mapping", {}) or {}
    result: dict[str, list[AliasRule]] = {}
    for target, aliases in mapping.items():
        rules: list[AliasRule] = []
        for alias in aliases if isinstance(aliases, list) else []:
            alias_text = clean_fragment(str(alias))
            if alias_text:
                rules.append(
                    AliasRule(
                        target=str(target),
                        alias=alias_text,
                        pattern=flexible_pattern(alias_text),
                    )
                )
        rules.sort(
            key=lambda item: (-len(normalized_code(item.alias)), -len(item.alias))
        )
        result[str(target)] = rules
    return result


def find_first_alias(text: str, rules: list[AliasRule]) -> tuple[AliasRule, str] | None:
    matches: list[tuple[int, int, str, AliasRule, str]] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            raw = clean_fragment(match.group(1))
            matches.append(
                (
                    -len(normalized_code(raw)),
                    match.start(),
                    rule.alias.casefold(),
                    rule,
                    raw,
                )
            )
    if not matches:
        return None
    _, _, _, rule, raw = min(matches)
    return rule, raw


def split_normalized_modifier(value: str) -> tuple[str, str]:
    upper = value.upper()
    for suffix in ("III", "II", "I", "CE", "ZN", "3PE"):
        if upper.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)], suffix
    return value, ""


class RawAliasResolver:
    def __init__(self, mapping_path: Path) -> None:
        self.rules = load_alias_rules(mapping_path)

    def resolve(self, text: str, current_value: str) -> RawResolution | None:
        direct = find_first_alias(text, self.rules.get(current_value, []))
        if direct:
            rule, raw = direct
            return RawResolution(
                raw=raw,
                status="direct_alias",
                evidence={"target": current_value, "alias": rule.alias},
            )

        base, modifier = split_normalized_modifier(current_value)
        if modifier:
            base_hit = find_first_alias(text, self.rules.get(base, []))
            if base_hit:
                _, raw = base_hit
                quality = extract_quality(text)
                if modifier in {"I", "II", "III"} and quality:
                    raw = f"{raw} Gr.{quality}"
                return RawResolution(
                    raw=raw,
                    status="base_alias_with_modifier",
                    evidence={"target": current_value, "base": base, "modifier": modifier},
                )

        literal = flexible_pattern(current_value).search(text)
        if literal:
            return RawResolution(
                raw=clean_fragment(literal.group(1)),
                status="literal",
                evidence={"target": current_value},
            )
        return None

    def resolve_atomic(self, text: str, target: str) -> str | None:
        direct = find_first_alias(text, self.rules.get(target, []))
        if direct:
            return direct[1]
        literal = flexible_pattern(target).search(text)
        return clean_fragment(literal.group(1)) if literal else None


def extract_quality(text: str) -> str:
    match = QUALITY_RE.search(normalize_text(text))
    if not match:
        return ""
    return ROMAN_MAP.get(match.group(1).upper(), match.group(1).upper())


def extract_special_requirements(text: str, modifier: str = "") -> list[str]:
    source = normalize_text(text)
    result: list[str] = []
    patterns = (
        ("NACE", r"(?i)(?<![A-Za-z0-9])NACE(?![A-Za-z0-9])"),
        ("ANTI-H2S", r"(?i)ANTI[\s-]*H2S"),
        ("HIC", r"(?i)(?<![A-Za-z0-9])HIC(?![A-Za-z0-9])"),
        ("RT100%", r"(?i)RT\s*100\s*%"),
        ("GALVANIZED", r"(?i)GALVANIZED|镀锌"),
        ("3PE", r"(?i)(?<![A-Za-z0-9])3PE(?![A-Za-z0-9])"),
    )
    allowed = {
        "CE": {"NACE", "ANTI-H2S", "HIC"},
        "ZN": {"GALVANIZED"},
        "3PE": {"3PE"},
    }.get(modifier)
    for label, pattern in patterns:
        if (allowed is None or label in allowed) and re.search(pattern, source):
            result.append(label)
    return result


def canonical_standard_body(body: str) -> str:
    code = normalized_code(body)
    substitutions = (
        ("GBT", "GB/T "),
        ("NBT", "NB/T "),
        ("SHT", "SH/T "),
        ("HGT", "HG/T "),
        ("SYT", "SY/T "),
    )
    for prefix, display in substitutions:
        if code.startswith(prefix):
            tail = code[len(prefix) :]
            tail = re.sub(r"(?:IA|IB|IIA|IIB|III|II|I|A|B)$", "", tail)
            return f"{display}{tail}"
    if code.startswith("DIN"):
        return f"DIN {code[3:]}"
    if code.startswith("EN"):
        return f"EN {code[2:]}"
    if code.startswith("JISG"):
        return f"JIS G {code[4:]}"
    if code.startswith("API5L"):
        return "API 5L"
    return clean_fragment(body)


def select_material_standard(
    row: dict[str, Any],
    grade: str,
) -> tuple[str, str]:
    # Dataset contract: standalone GB/T, NB/T, EN, DIN and similar standards
    # remain in top-level STANDARD. MATERIAL_STANDARD is reserved for a
    # standard that is part of an ASTM material expression, e.g. A106 + B.
    del row, grade
    return "", ""


def canonical_astm_standard(number: str) -> str:
    return f"ASTM A{number}"


def strip_astm_class_suffix(grade: str, standard: str) -> str:
    cleaned = clean_fragment(grade)
    cleaned = re.sub(r"(?i)\(\s*UNS[^)]*\)", "", cleaned).strip()
    cleaned = re.sub(
        r"(?i)\(\s*(?:NACE|ANTI[\s-]*H2S|HIC|CE|GALVANIZED)\s*\)",
        "",
        cleaned,
    ).strip()
    if standard in {"ASTM A403", "ASTM A234"} or cleaned.upper().startswith("WP"):
        cleaned = ASTM_CLASS_SUFFIX_RE.sub("", cleaned)
    return cleaned


def split_quality_from_grade(raw_grade: str, current_value: str) -> tuple[str, str]:
    quality = extract_quality(raw_grade)
    grade = QUALITY_RE.sub("", raw_grade).strip(" .,-")
    _, modifier = split_normalized_modifier(current_value)
    if not quality and modifier in {"I", "II", "III"}:
        trailing = TRAILING_QUALITY_RE.search(grade)
        if trailing:
            quality = ROMAN_MAP.get(trailing.group(1), trailing.group(1).upper())
            grade = grade[: trailing.start()].strip(" .,-")
    _, modifier = split_normalized_modifier(current_value)
    if modifier in {"CE", "ZN", "3PE"} and grade.upper().endswith(modifier):
        grade = grade[: -len(modifier)].strip(" .,-+")
    return grade, quality


def parse_astm_expression(raw: str, current_value: str) -> tuple[str, str, str] | None:
    text = normalize_text(raw)
    match = ASTM_STANDARD_RE.search(text)
    if not match:
        return None
    standard = canonical_astm_standard(match.group(1))
    remainder = text[match.end() :].strip(" .,-")
    remainder = re.sub(r"(?i)^GR(?:ADE)?\.?\s*", "", remainder).strip()
    if not remainder:
        remainder = f"A{match.group(1)}"
    elif standard == "ASTM A105" and remainder.upper() == "N":
        remainder = "A105N"
    grade, quality = split_quality_from_grade(remainder, current_value)
    grade = strip_astm_class_suffix(grade, standard)
    return standard, grade, quality


ASTM_FAMILY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "ASTM A403",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*403[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(?:WP[\s.-]*)?([0-9]{3}[A-Z]?)(?:\s*-\s*(?:WX|WU|W|S))?"
        ),
    ),
    (
        "ASTM A234",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*234[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(WP(?:B|[0-9]{1,3}[A-Z]?))(?:\s*-\s*(?:WX|WU|W|S))?"
        ),
    ),
    (
        "ASTM A182",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*182[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(F[0-9]{1,3}[A-Z]?)"
        ),
    ),
    (
        "ASTM A312",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*312[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(TP[0-9]{3}[A-Z]?)"
        ),
    ),
    (
        "ASTM A420",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*420[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(WPL[0-9A-Z]+)"
        ),
    ),
    (
        "ASTM A350",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*350[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?"
            r"(LF[0-9A-Z]+)"
        ),
    ),
    (
        "ASTM A106",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*106[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?([ABC])"
        ),
    ),
    (
        "ASTM A53",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*53[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?([ABC])"
        ),
    ),
    (
        "ASTM A333",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*333[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?([0-9]+)"
        ),
    ),
    (
        "ASTM A516",
        re.compile(
            r"(?i)(?:ASTM[\s.]*)?A[\s.]*516[\s.,/-]*(?:GR(?:ADE)?\.?\s*)?([0-9]+)"
        ),
    ),
)


def extract_astm_items(text: str) -> list[dict[str, str]]:
    found: list[tuple[int, dict[str, str]]] = []
    for standard, pattern in ASTM_FAMILY_PATTERNS:
        for match in pattern.finditer(normalize_text(text)):
            grade = match.group(1).upper()
            if standard == "ASTM A403" and not grade.startswith("WP"):
                grade = f"WP{grade}"
            found.append(
                (
                    match.start(),
                    {
                        "MATERIAL_STANDARD": standard,
                        "GRADE": grade,
                        "QUALITY_LEVEL": "",
                    },
                )
            )
    found.sort(key=lambda item: item[0])
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, item in found:
        key = (item["MATERIAL_STANDARD"], item["GRADE"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def extract_material_items(text: str) -> list[dict[str, str]]:
    items = extract_astm_items(text)
    for match in API_5L_GRADE_RE.finditer(normalize_text(text)):
        items.append(
            {
                "MATERIAL_STANDARD": "API 5L",
                "GRADE": match.group(1).upper(),
                "QUALITY_LEVEL": "",
            }
        )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            item["MATERIAL_STANDARD"],
            item["GRADE"],
            item["QUALITY_LEVEL"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def select_source_material_item(
    text: str,
    current_value: str,
    raw: str,
) -> dict[str, str] | None:
    items = extract_material_items(text)
    if not items:
        return None

    base_value, _ = split_normalized_modifier(current_value)
    target_codes = {
        normalized_code(base_value),
        normalized_code(current_value),
        normalized_code(raw),
    }
    candidates = []
    for item in items:
        standard_code = normalized_code(item["MATERIAL_STANDARD"])
        grade_code = normalized_code(item["GRADE"])
        if any(
            code
            and (
                code == grade_code
                or code in standard_code
                or standard_code.endswith(code)
            )
            for code in target_codes
        ):
            candidates.append(item)

    if len(candidates) == 1:
        return candidates[0]
    if len(items) == 1:
        return items[0]
    return None


def make_item(
    role: str,
    standard: str,
    grade: str,
    quality: str = "",
    special: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ROLE": role,
        "MATERIAL_STANDARD": clean_fragment(standard),
        "GRADE": clean_fragment(grade),
        "QUALITY_LEVEL": quality,
        "SPECIAL_REQ": list(dict.fromkeys(special or [])),
    }


def is_nested_material_standard(standard: str) -> bool:
    upper = clean_fragment(standard).upper()
    return upper.startswith("ASTM ") or upper == "API 5L"


def enforce_material_standard_contract(material: dict[str, Any]) -> dict[str, Any]:
    """Keep only explicitly supported material specifications."""
    for item in material.get("ITEMS") or []:
        standard = clean_fragment(item.get("MATERIAL_STANDARD", ""))
        item["MATERIAL_STANDARD"] = standard if is_nested_material_standard(standard) else ""
    return material


def standard_was_used_as_grade(item: dict[str, Any]) -> bool:
    standard_code = normalized_code(item.get("MATERIAL_STANDARD", ""))
    grade_code = normalized_code(item.get("GRADE", ""))
    if not standard_code or not grade_code:
        return False
    if standard_code == "ASTMA105" and grade_code in {"A105", "A105N"}:
        return False
    standard_tail = re.sub(r"^(?:ASTM|API)", "", standard_code)
    return grade_code in {standard_code, standard_tail}


def parse_atomic_item(
    row: dict[str, Any],
    current_value: str,
    raw: str,
    role: str = "MAIN",
) -> tuple[dict[str, Any], str]:
    source_item = select_source_material_item(
        row.get("input", ""), current_value, raw
    )
    astm = parse_astm_expression(raw, current_value)
    used_standard_body = ""
    if source_item:
        standard = source_item["MATERIAL_STANDARD"]
        grade = source_item["GRADE"]
        quality = source_item["QUALITY_LEVEL"]
    elif astm:
        standard, grade, quality = astm
    else:
        grade, quality = split_quality_from_grade(raw, current_value)
        standard, used_standard_body = select_material_standard(row, grade)
    _, modifier = split_normalized_modifier(current_value)
    special = extract_special_requirements(row.get("input", ""), modifier)
    return make_item(role, standard, grade, quality, special), used_standard_body


def expand_shared_grade_prefix(grades: list[str]) -> list[str]:
    if len(grades) < 2:
        return grades
    prefix = re.match(r"(?i)^(WP|TP|F)", grades[0])
    if not prefix:
        return grades
    result = [grades[0]]
    for grade in grades[1:]:
        result.append(grade if re.match(r"(?i)^(WP|TP|F)", grade) else prefix.group(1) + grade)
    return result


def build_dual_grade(
    row: dict[str, Any],
    current_value: str,
    raw: str,
) -> tuple[dict[str, Any], set[str]] | None:
    standard = ""
    expression = normalize_text(raw)
    astm_match = ASTM_STANDARD_RE.search(expression)
    if astm_match:
        standard = canonical_astm_standard(astm_match.group(1))
        expression = expression[astm_match.end() :]
    expression = re.sub(r"(?i)\bDUAL\b", "", expression)
    expression = re.sub(r"(?i)\bGR(?:ADE)?\.?", "", expression)
    grades = [clean_fragment(part) for part in expression.split("/") if clean_fragment(part)]
    grades = expand_shared_grade_prefix(grades)
    if len(grades) != 2:
        return None

    used: set[str] = set()
    items: list[dict[str, Any]] = []
    for grade in grades:
        quality = extract_quality(row.get("input", ""))
        grade = QUALITY_RE.sub("", grade).strip(" .,-")
        if not standard:
            standard, body = select_material_standard(row, grade)
            if body:
                used.add(body)
        items.append(make_item("MAIN", standard, strip_astm_class_suffix(grade, standard), quality))
    return {"RELATION": "dual_grade", "ITEMS": items}, used


def component_role(component: str, index: int, current_value: str, text: str) -> str:
    upper = component.upper()
    if upper in LINING_TOKENS or "衬" in component:
        return "LINING"
    if current_value == "304/20":
        return "INNER" if index == 0 else "OUTER"
    if current_value in {"20/304", "Q235B/304", "304/2200", "304/N02200"}:
        return "MAIN" if index == 0 else "CLADDING"
    if "A105" in upper and ("FLANGE" in text.upper() or "法兰" in text):
        return "FLANGE"
    return "MAIN" if index == 0 else "SECONDARY"


def normalize_component_grade(component: str, raw: str | None) -> str:
    if raw:
        return raw
    aliases = {
        "304": "304",
        "304L": "304L",
        "316": "316",
        "316L": "316L",
        "20": "20",
        "CS": "CS",
        "2200": "N02200",
    }
    return aliases.get(component, component)


def build_composite(
    row: dict[str, Any],
    current_value: str,
    resolver: RawAliasResolver,
) -> tuple[dict[str, Any], set[str]] | None:
    text = str(row.get("input") or "")
    upper = current_value.upper()

    if upper == "A106A105/C":
        components = ["A106", "A105", "CERAMIC"]
    elif upper == "20GLA105":
        components = ["20", "GLASS LINED", "A105"]
    elif upper == "FRP/A105ZN":
        components = ["FRP", "A105"]
    elif upper == "WPB/A105":
        components = ["WPB", "A105", "CERAMIC"]
    elif upper == "P11/410":
        components = ["P11", "410"]
    else:
        components = [clean_fragment(part) for part in current_value.split("/") if clean_fragment(part)]
    if len(components) < 2:
        return None

    global_astm = extract_astm_items(text)
    used: set[str] = set()
    items: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        role = component_role(component, index, current_value, text)
        token = component.upper()
        if token in LINING_TOKENS or component == "GLASS LINED":
            grade = {
                "GL": "GLASS LINED",
                "RPTFE": "RPTFE",
            }.get(component, component)
            items.append(make_item(role, "", grade))
            continue

        matching_astm = next(
            (
                item
                for item in global_astm
                if normalized_code(component) in normalized_code(item["GRADE"])
                or normalized_code(component) in normalized_code(item["MATERIAL_STANDARD"])
            ),
            None,
        )
        if matching_astm:
            items.append(
                make_item(
                    role,
                    matching_astm["MATERIAL_STANDARD"],
                    matching_astm["GRADE"],
                )
            )
            continue

        raw = resolver.resolve_atomic(text, component)
        raw = normalize_component_grade(component, raw)
        item, body = parse_atomic_item(row, component, raw, role)
        if body:
            used.add(body)
        items.append(item)

    if current_value in {"A106/PTFE", "WPB/PTFE"} and global_astm:
        items[0] = make_item(
            "MAIN",
            global_astm[0]["MATERIAL_STANDARD"],
            global_astm[0]["GRADE"],
        )
    return {"RELATION": "composite", "ITEMS": items}, used


def build_alternative_from_text(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    text = str(row.get("input") or "")
    if not re.search(r"(?i)\bOR\b|或", text):
        return None
    items = extract_astm_items(text)
    if len(items) < 2:
        return None
    material_items = [
        make_item(
            "MAIN",
            item["MATERIAL_STANDARD"],
            item["GRADE"],
            item["QUALITY_LEVEL"],
        )
        for item in items
    ]
    unique = {
        (item["MATERIAL_STANDARD"], item["GRADE"])
        for item in material_items
    }
    if len(unique) < 2:
        return None
    return {"RELATION": "alternative", "ITEMS": material_items}


def convert_historical_material(material: dict[str, Any]) -> dict[str, Any]:
    relation = str(material.get("RELATION") or "single")
    items: list[dict[str, Any]] = []
    for index, source in enumerate(material.get("ITEMS") or []):
        standard = str(
            source.get("MATERIAL_STANDARD")
            or source.get("EXEC_STANDARD")
            or ""
        )
        grade = str(source.get("GRADE") or "")
        quality = str(source.get("QUALITY_LEVEL") or "")
        grade, parsed_quality = split_quality_from_grade(grade, grade)
        quality = quality or parsed_quality
        role = str(source.get("ROLE") or ("MAIN" if index == 0 else "SECONDARY"))
        if relation == "composite" and index > 0 and (
            normalized_code(grade) in {normalized_code(x) for x in LINING_TOKENS}
            or "LINED" in grade.upper()
        ):
            role = "LINING"
        items.append(
            make_item(
                role,
                standard,
                strip_astm_class_suffix(grade, standard),
                quality,
                list(source.get("SPECIAL_REQ") or []),
            )
        )
    return {"RELATION": relation, "ITEMS": items}


def load_relation_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        material = row.get("new_material")
        if material:
            result[normalized_key(row.get("input", ""))] = convert_historical_material(material)
    return result


def remove_consumed_standards(
    standards: list[dict[str, Any]],
    consumed_bodies: set[str],
) -> list[dict[str, Any]]:
    consumed_codes = {normalized_code(body) for body in consumed_bodies}
    return [
        copy.deepcopy(item)
        for item in standards
        if normalized_code(item.get("BODY", "")) not in consumed_codes
    ]


class DatasetConverter:
    def __init__(self, mapping: Path, relation_report: Path) -> None:
        self.resolver = RawAliasResolver(mapping)
        self.relation_overrides = load_relation_overrides(relation_report)

    def convert(self, source_row: dict[str, Any]) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
        row = copy.deepcopy(source_row)
        text = str(row.get("input") or "")
        output = row.get("output")
        if not isinstance(output, dict):
            return None, "invalid_output", {}
        materials = output.get("MATERIAL")
        if not isinstance(materials, list) or len(materials) != 1:
            return None, "invalid_material_structure", {"materials": materials}
        current_value = str(materials[0].get("VALUE") or "").strip()
        if not current_value:
            return None, "empty_material", {}

        current_value_upper = current_value.upper()
        if current_value_upper in EXCLUDED_DIRTY_MATERIALS:
            return None, "explicitly_excluded_dirty_material", {
                "current_value": current_value,
            }

        if current_value_upper in PENDING_COMBINATION_ORDER_MATERIALS:
            return None, "pending_material_combination_order", {
                "current_value": current_value,
            }

        override = self.relation_overrides.get(normalized_key(text))
        if override:
            output["MATERIAL"] = enforce_material_standard_contract(
                copy.deepcopy(override)
            )
            return row, "relation_override", {}

        alternative = build_alternative_from_text(row)
        if alternative:
            output["MATERIAL"] = enforce_material_standard_contract(alternative)
            return row, "alternative", {}

        if current_value in DUAL_GRADE_PAIRS:
            resolution = self.resolver.resolve(text, current_value)
            if resolution:
                dual = build_dual_grade(row, current_value, resolution.raw)
                if dual:
                    material, used = dual
                    output["MATERIAL"] = enforce_material_standard_contract(material)
                    output["STANDARD"] = remove_consumed_standards(
                        output.get("STANDARD", []) or [], used
                    )
                    return row, "dual_grade", resolution.evidence

        is_composite = (
            "/" in current_value
            and current_value not in DUAL_GRADE_PAIRS
        ) or current_value in {"20GLA105", "FRP/A105ZN", "A106A105/C"}
        if is_composite:
            composite = build_composite(row, current_value, self.resolver)
            if composite:
                material, used = composite
                output["MATERIAL"] = enforce_material_standard_contract(material)
                output["STANDARD"] = remove_consumed_standards(
                    output.get("STANDARD", []) or [], used
                )
                return row, "composite", {"current_value": current_value}

        resolution = self.resolver.resolve(text, current_value)
        if not resolution:
            return None, "unresolved", {"current_value": current_value}
        item, used_body = parse_atomic_item(
            row, current_value, resolution.raw
        )
        if not item["GRADE"]:
            return None, "empty_grade_after_parse", {
                "current_value": current_value,
                "raw": resolution.raw,
            }
        if standard_was_used_as_grade(item):
            return None, "material_standard_without_grade", {
                "current_value": current_value,
                "raw": resolution.raw,
                "material_standard": item["MATERIAL_STANDARD"],
            }
        output["MATERIAL"] = enforce_material_standard_contract(
            {"RELATION": "single", "ITEMS": [item]}
        )
        if used_body:
            output["STANDARD"] = remove_consumed_standards(
                output.get("STANDARD", []) or [], {used_body}
            )
        return row, resolution.status, resolution.evidence


def convert_split(
    path: Path,
    converter: DatasetConverter,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    grades: Counter[str] = Counter()

    for index, source_row in enumerate(rows):
        converted, status, evidence = converter.convert(source_row)
        statuses[status] += 1
        if converted is None:
            rejected.append(
                {
                    "split": split,
                    "source_index": index,
                    "category": status,
                    "input": source_row.get("input", ""),
                    "current_output": source_row.get("output", {}),
                    "evidence": evidence,
                }
            )
            continue
        accepted.append(converted)
        material = converted["output"]["MATERIAL"]
        relations[material["RELATION"]] += 1
        for item in material["ITEMS"]:
            grades[item["GRADE"]] += 1

    report = {
        "source": str(path),
        "source_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "acceptance_rate": round(len(accepted) / len(rows), 6) if rows else 0,
        "status_counts": dict(statuses.most_common()),
        "relation_counts": dict(relations.most_common()),
        "unique_grades": len(grades),
        "top_grades": dict(grades.most_common(50)),
    }
    return accepted, rejected, report


def validate_rows(rows: list[dict[str, Any]], split: str) -> None:
    for index, row in enumerate(rows):
        material = row.get("output", {}).get("MATERIAL")
        if not isinstance(material, dict):
            raise ValueError(f"{split}[{index}] MATERIAL 不是对象")
        if material.get("RELATION") not in {
            "single",
            "composite",
            "alternative",
            "dual_grade",
        }:
            raise ValueError(f"{split}[{index}] RELATION 非法")
        items = material.get("ITEMS")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{split}[{index}] ITEMS 为空")
        for item in items:
            expected = {
                "ROLE",
                "MATERIAL_STANDARD",
                "GRADE",
                "QUALITY_LEVEL",
                "SPECIAL_REQ",
            }
            if set(item) != expected:
                raise ValueError(f"{split}[{index}] 材质字段不完整: {item}")
            if not item["GRADE"]:
                raise ValueError(f"{split}[{index}] GRADE 为空")
            standard = item["MATERIAL_STANDARD"]
            if standard and not is_nested_material_standard(standard):
                raise ValueError(
                    f"{split}[{index}] 非材料规范进入 MATERIAL_STANDARD: {standard}"
                )
            if standard_was_used_as_grade(item):
                raise ValueError(
                    f"{split}[{index}] 规范编号被误作 GRADE: {item}"
                )


def exclude_train_val_overlap(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    val_rejected: list[dict[str, Any]],
    val_report: dict[str, Any],
) -> list[dict[str, Any]]:
    train_keys = {normalized_key(row["input"]) for row in train}
    retained: list[dict[str, Any]] = []
    removed = 0
    for index, row in enumerate(val):
        if normalized_key(row["input"]) not in train_keys:
            retained.append(row)
            continue
        removed += 1
        val_rejected.append(
            {
                "split": "val",
                "source_index": index,
                "category": "train_val_overlap",
                "input": row["input"],
                "current_output": row["output"],
                "evidence": {},
            }
        )
    if removed:
        val_report["accepted_rows"] = len(retained)
        val_report["rejected_rows"] += removed
        val_report["acceptance_rate"] = round(
            len(retained) / val_report["source_rows"], 6
        )
        statuses = Counter(val_report["status_counts"])
        statuses["train_val_overlap_excluded"] += removed
        val_report["status_counts"] = dict(statuses.most_common())
    return retained


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    converter = DatasetConverter(args.mapping, args.relation_report)
    train, train_rejected, train_report = convert_split(
        args.train_source, converter, "train"
    )
    val, val_rejected, val_report = convert_split(
        args.val_source, converter, "val"
    )
    val = exclude_train_val_overlap(
        train, val, val_rejected, val_report
    )
    validate_rows(train, "train")
    validate_rows(val, "val")

    train_keys = {normalized_key(row["input"]) for row in train}
    val_keys = {normalized_key(row["input"]) for row in val}
    overlap = sorted(train_keys & val_keys)

    write_json(args.train_output, train)
    write_json(args.val_output, val)
    write_json(
        args.audit_output,
        {
            "policy": "无法从原文可靠恢复原始牌号的样本不进入训练集",
            "grouped_counts": dict(
                Counter(item["category"] for item in train_rejected + val_rejected)
            ),
            "items": train_rejected + val_rejected,
        },
    )
    report = {
        "schema": {
            "MATERIAL": {
                "RELATION": "single|composite|alternative|dual_grade",
                "ITEMS": [
                    {
                        "ROLE": "MAIN|LINING|INNER|OUTER|CLADDING|FLANGE|SECONDARY",
                        "MATERIAL_STANDARD": "ASTM/API 材料标准或空",
                        "GRADE": "原始牌号",
                        "QUALITY_LEVEL": "I|II|III 或空",
                        "SPECIAL_REQ": [],
                    }
                ],
            }
        },
        "train_output": str(args.train_output),
        "val_output": str(args.val_output),
        "audit_output": str(args.audit_output),
        "train_val_description_overlap": len(overlap),
        "train": train_report,
        "val": val_report,
    }
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
