#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "apps/trainer/qwen3_fte/output/pipe_project_sampling_full/直管语义字段补充候选_新架构.xlsx"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/pipe_project_sampling_full"

SEMANTIC_DIMS = {"MANU_SIG", "CONN_SIG", "MAT_EXPR_SIG", "STD_SIG", "MAT_FAMILY"}
ALLOWED_CONN = ("MNPT", "FNPT", "NPT", "THD", "SW", "FTE", "MTE", "SF")

STANDARD_MAP = {
    "GBT8163": "GB/T8163",
    "GBT14976": "GB/T14976",
    "GBT12771": "GB/T12771",
    "GBT9711": "GB/T9711",
    "GBT9948": "GB/T9948",
    "GBT6479": "GB/T6479",
    "GBT5310": "GB/T5310",
    "GBT3087": "GB/T3087",
    "GBT3091": "GB/T3091",
    "GBT17395": "GB/T17395",
    "GBT218332": "GB/T21833.2",
    "GBT218322": "GB/T21832.2",
    "GBT3621": "GB/T3621",
    "HGT20553": "HG/T20553",
    "HGT20538": "HG/T20538",
    "HGT2130": "HG/T2130",
    "HGT3731": "HG/T3731",
    "SHT3405": "SH/T3405",
    "SHT3406": "SH/T3406",
    "SYT5037": "SY/T5037",
    "SYT5257": "SY/T5257",
    "EN102165": "EN 10216-5",
    "EN1127": "EN ISO 1127",
    "ENI1127": "EN ISO 1127",
    "AB3610": "ASME B36.10",
    "AB3619": "ASME B36.19",
    "ASTMA312": "ASTM A312",
    "ASTMA333": "ASTM A333",
    "ASTMA358": "ASTM A358",
    "API5L": "API 5L",
    "CJT120": "CJ/T120",
}


def text(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_num(value: str) -> str:
    value = text(value)
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def infer_body(type_code: str, desc: str, material_code: str = "") -> str:
    code = text(type_code).upper()
    desc_u = desc.upper()
    mat_u = text(material_code).upper()
    if (
        code in {"FP", "LFP"}
        or "FLANGED PIPE" in desc_u
        or re.search(r"LJ\s*FLANGE\s*[X×]\s*FLANGE|FLANGE\s*[X×]\s*FLANGE", desc_u)
        or "法兰管" in desc
        or "两端活套法兰" in desc
        or "活套法兰" in desc
    ):
        return "法兰管"
    if (
        code in {"LP", "PSF"}
        or any(token in desc for token in ["衬", "搪玻璃", "涂塑", "复合管"])
        or any(token in desc_u for token in ["LINED", "PTFE", "RPTFE", "FRP/PVC", "FRP/CPVC", "GLASS LINED"])
        or any(token in mat_u for token in ["PTFE", "RPTFE", "FRP/PVC", "FRP/CPVC", "EAA"])
    ):
        return "衬里复合管"
    return "直管"


def infer_manu(desc: str, type_code: str = "") -> list[str]:
    desc_u = desc.upper()
    specific_patterns = [
        ("DSAWL", r"(?<![A-Z0-9])DSAWL(?![A-Z0-9])|直缝双面埋弧|直.*双面埋弧|LONGI\\s*TUDE\\s+DOUBLE|LONGITUDE\\s+DOUBLE"),
        ("DSAWH", r"(?<![A-Z0-9])DSAWH(?![A-Z0-9])|螺旋缝双面埋弧|螺旋.*双面埋弧|(?=.*(?<![A-Z0-9])SAWH(?![A-Z0-9]))(?=.*双面埋弧)"),
        ("SAWL", r"(?<![A-Z0-9])SAWL(?![A-Z0-9])|(?<![A-Z0-9])LSAW(?![A-Z0-9])|直缝埋弧|直.*埋弧|直焊缝.*埋弧"),
        ("SAWH", r"(?<![A-Z0-9])SAWH(?![A-Z0-9])|(?<![A-Z0-9])HSAW(?![A-Z0-9])|螺旋缝埋弧|螺旋埋弧|螺旋焊接|螺旋焊管|螺旋焊"),
        ("DSAW", r"(?<![A-Z0-9])DSAW(?![A-Z0-9])|(?<![A-Z0-9])DASW(?![A-Z0-9])|DOUBLE\s+SUBMERGED|双面埋弧"),
        ("ERW", r"(?<![A-Z0-9])ERW(?![A-Z0-9])|直缝电阻焊|高频电阻焊"),
        ("HFW", r"(?<![A-Z0-9])HFW(?![A-Z0-9])|高频焊"),
        ("EFW", r"(?<![A-Z0-9])EFW(?![A-Z0-9])|E\.F\.W|电熔焊|电熔化焊"),
        ("SAW", r"(?<![A-Z0-9])SAW(?![A-Z0-9])|埋弧焊"),
    ]
    for token, pattern in specific_patterns:
        if re.search(pattern, desc_u):
            return [token]

    if re.search(r"(?<![A-Z0-9])SMLS(?![A-Z0-9])|SEAMLESS", desc_u) or "无缝" in desc:
        return ["SMLS"]

    if text(type_code).upper() in {"PW", "PWM"}:
        return ["WELDED"]
    if any(token in desc for token in ["焊接钢管", "焊接不锈钢管", "焊接管", "焊缝钢管", "焊管", "有缝钢管", "直缝焊接"]):
        return ["WELDED"]
    if re.search(r"(?<![A-Z0-9])WELDED(?![A-Z0-9])|(?<![A-Z0-9])WELD(?![A-Z0-9])", desc_u):
        return ["WELDED"]
    return []


def infer_conn(desc: str) -> list[str]:
    desc_u = desc.upper()
    values: list[str] = []
    for token in ALLOWED_CONN:
        if token == "THD":
            if "THD" in desc_u or "THREADED" in desc_u or "螺纹连接" in desc:
                values.append(token)
            continue
        if token == "NPT":
            if re.search(r"(?<![A-Z0-9])NPT(?![A-Z0-9])|TE\(NPT\)|TH\(NPT\)", desc_u):
                if not any(x in desc_u for x in ("MNPT", "FNPT")):
                    values.append(token)
            continue
        if token == "SW":
            if re.search(r"(?<![A-Z0-9])SW(?![A-Z0-9])", desc_u) or "承插" in desc:
                values.append(token)
            continue
        if token == "SF":
            if re.search(r"(?<![A-Z0-9])SF(?![A-Z0-9])|承插粘接", desc_u):
                values.append(token)
            continue
        if re.search(rf"(?<![A-Z0-9]){token}(?![A-Z0-9])", desc_u):
            values.append(token)
    return uniq(values)


def _extract_size_thickness_from_desc(desc: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    desc_u = desc.upper()
    dn: list[str] = []
    od: list[str] = []
    inch: list[str] = []
    mm: list[str] = []
    schedule: list[str] = []
    series: list[str] = []
    bwg: list[str] = []

    def raw_num(value: str) -> str:
        return value.strip().replace("O", "0")

    def dn_value(value: str) -> str:
        return normalize_num(raw_num(value))

    for match in re.finditer(r"DN\s*(\d+(?:\.\d+)?)", desc_u):
        dn.append(f"DN{dn_value(match.group(1))}")
    for match in re.finditer(r"DN\s*(\d+(?:\.\d+)?)\s*[X×]\s*(?:DN\s*)?(\d+(?:\.\d+)?)(?!\s*(?:MM|SCH))", desc_u):
        second = match.group(2)
        if float(second) >= 15 and "." not in second:
            dn.append(f"DN{dn_value(second)}")

    for match in re.finditer(r"(?:外径\s*)?[ΦФφØ]\s*(\d+(?:\.\d+)?)\s*(?:[X×*]\s*(\d+(?:\.\d+)?))?", desc, re.IGNORECASE):
        od.append(raw_num(match.group(1)))
        if match.group(2):
            mm.append(raw_num(match.group(2)))
    for match in re.finditer(r"(?<![A-Z0-9])OD\s*(\d+(?:\.\d+)?)\s*(?:[X×*]\s*(\d+(?:\.\d+)?))?", desc_u):
        od.append(raw_num(match.group(1)))
        if match.group(2):
            mm.append(raw_num(match.group(2)))
    for match in re.finditer(r"(?<![A-Z0-9])D\s*(\d+(?:\.\d+)?)\s*[X×*]\s*(\d+(?:\.\d+)?)", desc_u):
        od.append(raw_num(match.group(1)))
        mm.append(raw_num(match.group(2)))
    for match in re.finditer(r"(?<![A-Z0-9])([1-9]\d{1,3}(?:\.\d+)?)\s*[X×*]\s*(\d+(?:\.\d+)?)(?![\d.])", desc, re.IGNORECASE):
        tail = desc[match.end(2) : match.end(2) + 4]
        if re.match(r"\s*(?:mm|m\b)", tail, re.IGNORECASE):
            continue
        context = desc[max(0, match.start() - 20) : match.end() + 20]
        if (
            re.search(r"[ΦФφØ]|\bOD\b|\bNPS|\"|管径|外径", context, re.IGNORECASE)
            or (float(match.group(1)) >= 50 and float(match.group(2)) <= 80 and not re.search(r"DN\s*$", desc[max(0, match.start() - 5) : match.start()], re.IGNORECASE))
        ):
            od.append(raw_num(match.group(1)))
            mm.append(raw_num(match.group(2)))

    for match in re.finditer(r"\bNPS\s*(\d+(?:\s+\d+/\d+|-\d+/\d+|/\d+|\.\d+)?)(?:\s*\")?", desc, re.IGNORECASE):
        inch.append("NPS" + re.sub(r"\s+", " ", match.group(1).strip()))
    for match in re.finditer(r'(?<![A-Z0-9])((?:\d+\s+\d+/\d+)|(?:\d+-\d+/\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*"', desc):
        if re.search(r"NPS\s*$", desc[max(0, match.start() - 5) : match.start()], re.IGNORECASE):
            continue
        inch.append(re.sub(r"\s+", " ", match.group(1).strip()) + '"')
    for match in re.finditer(r"\bSIZE\s*((?:\d+\s+\d+/\d+)|(?:\d+-\d+/\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*\"", desc, re.IGNORECASE):
        inch.append(re.sub(r"\s+", " ", match.group(1).strip()) + '"')

    for match in re.finditer(r"DN\s*\d+(?:\.\d+)?\s*[X×*]\s*(\d+(?:\.\d+)?)\s*MM\b", desc_u):
        mm.append(raw_num(match.group(1)))
    thk_pattern = r"\b(?:THK|WT|T)\s*[=:：]?\s*(\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)*)\s*MM\b|壁厚\s*[=:：]?\s*(\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)*)\s*MM\b"
    for match in re.finditer(thk_pattern, desc, re.IGNORECASE):
        value = match.group(1) or match.group(2)
        mm.extend(raw_num(num) for num in re.findall(r"\d+(?:\.\d+)?", value))

    for match in re.finditer(r"SCH\s*\.?\s*-?\s*(\d{1,3}S?|STD|XS|XXS)(?=$|[^A-Z0-9]|DN)", desc_u):
        token = match.group(1)
        if token in {"STD", "XS", "XXS"}:
            series.append(token)
        else:
            schedule.append("SCH" + token)
    for match in re.finditer(r"(?<![A-Z0-9])S\s*-\s*(\d{1,3}S?|STD|XS|XXS)(?=$|[^A-Z0-9]|DN)", desc_u):
        token = match.group(1)
        if token in {"STD", "XS", "XXS"}:
            series.append(token)
        else:
            schedule.append("SCH" + token)
    for match in re.finditer(r"(?<![A-Z0-9])S(\d{1,3}S?)(?=$|[^A-Z0-9]|DN)", desc_u):
        schedule.append("SCH" + match.group(1))
    for match in re.finditer(r"B36\.19M?\s+(\d{1,3}S)(?=$|[^A-Z0-9]|DN)", desc_u):
        schedule.append("SCH" + match.group(1))
    for token in ("XXS", "STD", "XS"):
        if re.search(rf"(?<![A-Z0-9]){token}(?:\s*WT)?(?=$|[^A-Z0-9]|DN)", desc_u):
            series.append(token)
    for match in re.finditer(r"(?<![A-Z0-9])BWG\s*(\d+)(?![A-Z0-9])", desc_u):
        bwg.append("BWG" + match.group(1))
    for match in re.finditer(r"系列\s*([12])(?=\D|$)", desc_u):
        series.append("Series " + match.group(1))

    size = {"DN": uniq(dn), "OD": uniq(od), "INCH": uniq(inch), "LENGTH": []}
    thickness = {"MM": uniq(mm), "SCHEDULE": uniq(schedule), "SERIES": uniq(series), "BWG": uniq(bwg), "INCH": []}
    return size, thickness


def build_size(size_code: str, desc: str = "") -> dict[str, list[str]]:
    actual_size, _ = _extract_size_thickness_from_desc(desc)
    if actual_size["DN"] or actual_size["OD"] or actual_size["INCH"]:
        return actual_size
    code = text(size_code)
    dn = [f"DN{normalize_num(x)}" for x in re.findall(r"\d+(?:\.\d+)?", code)]
    return {"DN": uniq(dn), "OD": [], "INCH": [], "LENGTH": []}


def build_thickness(thk_code: str, desc: str = "") -> dict[str, list[str]]:
    _, actual_thickness = _extract_size_thickness_from_desc(desc)
    if actual_thickness["MM"] or actual_thickness["SCHEDULE"] or actual_thickness["SERIES"] or actual_thickness["BWG"]:
        return actual_thickness
    code = text(thk_code).upper().replace(" ", "")
    mm: list[str] = []
    schedule: list[str] = []
    series: list[str] = []
    for part in re.split(r"[xX×]", code):
        part = part.strip()
        if not part:
            continue
        if part.endswith("MM"):
            mm.append(normalize_num(part[:-2]))
        elif part in {"STD", "XS", "XXS"}:
            series.append(part)
        elif part.startswith("S") and re.fullmatch(r"S\d+S?", part):
            schedule.append("SCH" + part[1:])
        elif part.startswith("SCH"):
            token = part[3:]
            if token in {"STD", "XS", "XXS"}:
                series.append(token)
            else:
                schedule.append("SCH" + token)
    return {"MM": uniq(mm), "SCHEDULE": uniq(schedule), "SERIES": uniq(series), "BWG": [], "INCH": []}


def build_pressure(desc: str) -> str:
    desc_u = desc.upper()
    if re.search(r"≥\s*10\s*MPA", desc_u):
        return ""
    # A672/A671 Cxx CLxx 中的 CLxx 是材料等级/类别，按当前标注规则不进 PRESSURE。
    masked = re.sub(
        r"(?:ASTM\s*)?A67[12]\s+C\d+\s+CL\d+",
        lambda m: m.group(0).replace("CL", "MATERIAL_CLASS_"),
        desc_u,
    )
    match = re.search(r"\bPN\s*(\d+(?:\.\d+)?)\b", masked)
    if match:
        return "PN" + normalize_num(match.group(1))
    match = re.search(r"\bCLASS\s*(\d+)\b|\bCL\s*(\d+)\b", masked)
    if match:
        return "CL" + (match.group(1) or match.group(2))
    match = re.search(r"\b(\d+)\s*(?:LB|#)\b", masked)
    if match:
        return match.group(1) + "LB"
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*MPA\b", masked)
    if match:
        return normalize_num(match.group(1)) + "MPa"
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*BAR\b", masked)
    if match:
        return normalize_num(match.group(1)) + "bar"
    return ""


def base_material_from_desc_or_code(desc: str, material_code: str) -> str:
    desc_u = desc.upper()
    code = text(material_code)
    if code.upper().endswith("PEEP") and ("外加强级PE" in desc or "内EP" in desc):
        code = code[:-4]
    paren_grade = re.search(r"(?<![A-Z0-9])(\d{1,3}[A-Z]?|[A-Z]\d{3,5}|S\d{5}|Q\d{3}[A-Z]?)\s*[（(]\s*([A-Z0-9.+-]+)\s*[）)]", desc_u)
    if paren_grade:
        return f"{paren_grade.group(1)}({paren_grade.group(2)})"

    a67x = re.search(r"(?:ASTM\s*)?(A67[12])\s+C\s*(\d+)(?:\s+CL\s*\d+)?", desc_u)
    if a67x:
        return f"ASTM {a67x.group(1)} C{a67x.group(2)}"

    a312_gr_tp = re.search(r"(?:ASTM\s*)?A312\s*(?:GRADE\.?|GR\.?)\s*TP\s*([A-Z0-9/]+)", desc_u)
    if a312_gr_tp:
        return "ASTM A312 Gr.TP" + a312_gr_tp.group(1).removeprefix("TP")

    a312_gr = re.search(r"(?:ASTM\s*)?A312\s*(?:GRADE\.?|GR\.?)\s*([A-Z0-9/]+)", desc_u)
    if a312_gr:
        grade = a312_gr.group(1).removeprefix("GR.").removeprefix("GRADE")
        return "ASTM A312 Gr." + grade

    a312_tp = re.search(r"(?:ASTM\s*)?A312\s*TP\s*([A-Z0-9/]+)", desc_u)
    if a312_tp:
        grade = a312_tp.group(1)
        if grade in {"304/304L", "316/316L"}:
            parts = [f"TP{p}" for p in grade.split("/")]
            return "ASTM A312 " + "/".join(parts)
        if "/" in grade and not grade.startswith("TP"):
            parts = [p if p.startswith("TP") else f"TP{p}" for p in grade.split("/")]
            return "ASTM A312 " + "/".join(parts)
        return "ASTM A312 TP" + grade.removeprefix("TP")

    if "A106" in desc_u:
        return "ASTM A106 Gr.B" if re.search(r"A106[-\s.]?B|A106\s+GR\.?\s*B", desc_u) else "ASTM A106"
    if "A312" in desc_u:
        if "TP304/304L" in desc_u:
            return "ASTM A312 TP304/TP304L"
        if "TP316/316L" in desc_u:
            return "ASTM A312 TP316/TP316L"
        m = re.search(r"A312\s+TP\s*([A-Z0-9/]+)", desc_u)
        if m:
            return "ASTM A312 TP" + m.group(1)
    for suffix in ("ZN", "3PE"):
        if code.upper().endswith(suffix) and len(code) > len(suffix):
            return code[:-len(suffix)]
    return code


def build_material(desc: str, material_code: str) -> list[dict[str, Any]]:
    code = text(material_code)
    if not code:
        return []
    desc_u = desc.upper()
    value = base_material_from_desc_or_code(desc, code)
    inner: list[str] = []
    outer: list[str] = []
    special: list[str] = []
    lining_tokens: list[str] = []

    for token in ("PTFE", "RPTFE", "EAA", "GLASS LINED", "搪玻璃"):
        if token in desc_u or token in desc:
            lining_tokens.append("GLASS LINED" if token == "搪玻璃" else token)
    if "GALV" in desc_u or "镀锌" in desc:
        outer.append("Galvanized")
    if "外加强级PE" in desc:
        outer.append("加强级PE")
    elif "3PE" in desc_u:
        outer.append("PE")
    if "内EP" in desc:
        inner.append("EP")
    if "NACE" in desc_u:
        special.append("NACE")

    if lining_tokens:
        merged = [value]
        for token in uniq(lining_tokens):
            if token and token not in merged:
                merged.append(token)
        value = "/".join(merged)

    return [{
        "ROLE": "MAIN",
        "VALUE": value,
        "COATING": {"INNER": uniq(inner), "OUTER": uniq(outer)},
        "SPECIAL_REQ": uniq(special),
    }]


def parse_standard_token(token: str) -> dict[str, str] | None:
    raw = text(token).upper().replace(" ", "")
    if not raw:
        return None

    grade = ""
    bracket = re.search(r"\[([^\]]+)\]$", raw)
    if bracket:
        grade = bracket.group(1)
        raw = raw[: bracket.start()]
    elif raw.endswith("IA") and raw[:-2] in STANDARD_MAP:
        grade = "Ia"
        raw = raw[:-2]
    elif raw.endswith("II") and raw[:-2] in STANDARD_MAP:
        grade = "II"
        raw = raw[:-2]
    elif raw.endswith("I") and raw[:-1] in STANDARD_MAP:
        grade = "I"
        raw = raw[:-1]

    body = STANDARD_MAP.get(raw)
    if not body:
        for prefix, mapped in sorted(STANDARD_MAP.items(), key=lambda x: -len(x[0])):
            if raw.startswith(prefix):
                body = mapped
                rest = raw[len(prefix):]
                if not grade and rest in {"I", "II", "III", "IV", "A", "B", "IA"}:
                    grade = "Ia" if rest == "IA" else rest
                break
    if not body:
        return None
    if grade.upper() == "IA":
        grade = "Ia"
    return {"BODY": body, "GRADE": grade, "METHOD": "", "APPENDIX": ""}


def build_standard(split_value: str, code_value: str = "", desc: str = "") -> list[dict[str, str]]:
    source = text(split_value) or text(code_value)
    desc_u = text(desc).upper()
    if "ASME B36.19M/B36.10M" in desc_u:
        return [
            {"BODY": "ASME B36.19M", "GRADE": "", "METHOD": "", "APPENDIX": ""},
            {"BODY": "ASME B36.10M", "GRADE": "", "METHOD": "", "APPENDIX": ""},
        ]
    if "ASME B36.19/B36.10" in desc_u:
        return [
            {"BODY": "ASME B36.19", "GRADE": "", "METHOD": "", "APPENDIX": ""},
            {"BODY": "ASME B36.10", "GRADE": "", "METHOD": "", "APPENDIX": ""},
        ]
    if not source:
        return []
    source = re.sub(r"(AB3619|AB3610|ASME\s*B36\.19M?|ASME\s*B36\.10M?)\s*/\s*(AB3619|AB3610|B36\.19M?|B36\.10M?)", r"\1+\2", source, flags=re.IGNORECASE)
    tokens = re.split(r"[+;；,，]", source)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        token = token.strip()
        if re.fullmatch(r"B36\.10M", token, flags=re.IGNORECASE):
            item = {"BODY": "ASME B36.10M", "GRADE": "", "METHOD": "", "APPENDIX": ""}
        elif re.fullmatch(r"B36\.19M", token, flags=re.IGNORECASE):
            item = {"BODY": "ASME B36.19M", "GRADE": "", "METHOD": "", "APPENDIX": ""}
        elif re.fullmatch(r"B36\.10", token, flags=re.IGNORECASE):
            item = {"BODY": "ASME B36.10", "GRADE": "", "METHOD": "", "APPENDIX": ""}
        elif re.fullmatch(r"B36\.19", token, flags=re.IGNORECASE):
            item = {"BODY": "ASME B36.19", "GRADE": "", "METHOD": "", "APPENDIX": ""}
        else:
            item = parse_standard_token(token)
        if not item:
            continue
        key = (item["BODY"], item["GRADE"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_output_from_candidate(row: dict[str, Any]) -> OrderedDict[str, Any]:
    desc = text(row.get("材料描述"))
    type_code = text(row.get("BODY/种类编码") or row.get("类型编码"))
    material_code = text(row.get("材质编码"))
    output: OrderedDict[str, Any] = OrderedDict()
    output["TYPE"] = {
        "BODY": infer_body(type_code, desc, material_code),
        "MANU": infer_manu(desc, type_code),
        "CONN": infer_conn(desc),
    }
    output["SIZE"] = build_size(text(row.get("尺寸编码")), desc)
    output["THICKNESS"] = build_thickness(text(row.get("壁厚编码")), desc)
    pressure = build_pressure(desc)
    if pressure:
        output["PRESSURE"] = pressure
    material = build_material(desc, material_code)
    if material:
        output["MATERIAL"] = material
    standard = build_standard(text(row.get("规范拆分")), text(row.get("规范编码")), desc)
    if standard:
        output["STANDARD"] = standard
    return output


def load_real_supplement(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name="语义候选总表")
    df = df[df["命中维度"].astype(str).isin(SEMANTIC_DIMS)].copy()
    # Manufacturing combinations are often source conflicts; keep them in Excel, not training JSON.
    bad_manu_combo = df["命中维度"].eq("MANU_SIG") & df["缺口签名"].astype(str).str.contains("+", regex=False)
    df = df[~bad_manu_combo]
    df = df.drop_duplicates("材料描述")
    if max_rows:
        df = df.head(max_rows)

    dataset: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = row.to_dict()
        desc = text(record.get("材料描述"))
        if not desc:
            continue
        dataset.append({
            "input": desc,
            "output": build_output_from_candidate(record),
            "_source": "real_semantic_supplement",
            "_reason": text(record.get("命中维度")),
            "_gap": text(record.get("缺口签名")),
        })
    return dataset


def make_base_output(
    *,
    body: str = "直管",
    manu: list[str] | None = None,
    conn: list[str] | None = None,
    material: str = "20",
    standard_body: str = "GB/T8163",
    standard_grade: str = "",
) -> OrderedDict[str, Any]:
    output: OrderedDict[str, Any] = OrderedDict()
    output["TYPE"] = {"BODY": body, "MANU": manu or [], "CONN": conn or []}
    output["SIZE"] = {"DN": ["DN100"], "OD": [], "INCH": [], "LENGTH": []}
    output["THICKNESS"] = {"MM": [], "SCHEDULE": [], "SERIES": ["STD"], "BWG": [], "INCH": []}
    output["MATERIAL"] = [{
        "ROLE": "MAIN",
        "VALUE": material,
        "COATING": {"INNER": [], "OUTER": []},
        "SPECIAL_REQ": [],
    }]
    output["STANDARD"] = [{"BODY": standard_body, "GRADE": standard_grade, "METHOD": "", "APPENDIX": ""}]
    return output


def build_augmented_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    manu_cases = [
        ("无缝钢管 20 GB/T8163 SMLS BE DN100 STD", ["SMLS"]),
        ("PIPE 20 GB/T8163 SEAMLESS BE DN100 STD", ["SMLS"]),
        ("焊接钢管 Q235B GB/T3091 DN100 4.0mm", ["WELDED"]),
        ("PIPE WELDED Q235B GB/T3091 DN100 4.0mm", ["WELDED"]),
        ("直缝埋弧焊接钢管 L245 GB/T9711 SAWL DN100 STD", ["SAWL"]),
        ("螺旋缝埋弧焊接钢管 L245 GB/T9711 SAWH DN100 STD", ["SAWH"]),
        ("双面埋弧焊钢管 API 5L Gr.B DSAW DN100 STD", ["DSAW"]),
        ("直缝双面埋弧焊钢管 API 5L Gr.B DSAWL DN100 STD", ["DSAWL"]),
        ("高频电阻焊钢管 Q235B ERW GB/T3091 DN100 4.0mm", ["ERW"]),
        ("高频焊管 Q235B HFW GB/T3091 DN100 4.0mm", ["HFW"]),
        ("电熔焊管 S30408 EFW GB/T12771 Type I DN100 SCH10S", ["EFW"]),
        ("碳钢管道 DN100 GB/T8163 20 BE STD SH/T3405 连接形式:焊接", []),
    ]
    for desc, manu in manu_cases:
        out = make_base_output(manu=manu)
        if "GB/T3091" in desc:
            out["MATERIAL"][0]["VALUE"] = "Q235B"
            out["STANDARD"] = [{"BODY": "GB/T3091", "GRADE": "", "METHOD": "", "APPENDIX": ""}]
            out["THICKNESS"]["SERIES"] = []
            out["THICKNESS"]["MM"] = ["4.0"]
        elif "GB/T9711" in desc:
            out["MATERIAL"][0]["VALUE"] = "L245" if "L245" in desc else "API 5L Gr.B"
            out["STANDARD"] = [{"BODY": "GB/T9711", "GRADE": "", "METHOD": "", "APPENDIX": ""}]
        elif "GB/T12771" in desc:
            out["MATERIAL"][0]["VALUE"] = "S30408"
            out["STANDARD"] = [{"BODY": "GB/T12771", "GRADE": "Type I", "METHOD": "", "APPENDIX": ""}]
            out["THICKNESS"]["SERIES"] = []
            out["THICKNESS"]["SCHEDULE"] = ["SCH10S"]
        samples.append({"input": desc, "output": out, "_source": "semantic_augmentation", "_reason": "MANU"})

    conn_cases = [
        ("管子 20 GB/T8163 TE(NPT) SH/T3405 DN100 STD", ["NPT"]),
        ("管子 20 GB/T8163 MNPT End SH/T3405 DN100 STD", ["MNPT"]),
        ("管子 20 GB/T8163 FNPT End SH/T3405 DN100 STD", ["FNPT"]),
        ("管子 20 GB/T8163 THD SH/T3405 DN100 STD", ["THD"]),
        ("管子 20 GB/T8163 螺纹连接 SH/T3405 DN100 STD", ["THD"]),
        ("管子 20 GB/T8163 SW SH/T3405 DN100 STD", ["SW"]),
        ("管子 20 GB/T8163 FTE SH/T3405 DN100 STD", ["FTE"]),
        ("管子 20 GB/T8163 MTE SH/T3405 DN100 STD", ["MTE"]),
        ("FRP/CPVC管 FRP/PVC SF HG/T3731 DN100 THK=6.0mm", ["SF"]),
    ]
    for desc, conn in conn_cases:
        if "FRP" in desc:
            out = make_base_output(body="衬里复合管", conn=conn, material="FRP/PVC", standard_body="HG/T3731")
            out["THICKNESS"]["SERIES"] = []
            out["THICKNESS"]["MM"] = ["6.0"]
        else:
            out = make_base_output(conn=conn)
            out["STANDARD"].append({"BODY": "SH/T3405", "GRADE": "", "METHOD": "", "APPENDIX": ""})
        samples.append({"input": desc, "output": out, "_source": "semantic_augmentation", "_reason": "CONN"})

    material_cases = [
        ("钢衬四氟管道 20/PTFE DN100 GB/T8163 HG/T20538", "20/PTFE", [], []),
        ("碳钢管道 20#+EAA DN100 GB/T8163 HG/T20538", "20/EAA", [], []),
        ("搪玻璃管 DN100 HG/T2130 搪玻璃", "搪玻璃/GLASS LINED", [], []),
        ("涂塑复合钢管 DN100 Q235B外加强级PE内EP CJ/T120", "Q235B", ["EP"], ["PE"]),
        ("镀锌钢管 Q235B Galv. GB/T3091 ERW DN100 4.0mm", "Q235B", [], ["Galvanized"]),
    ]
    for desc, value, inner, outer in material_cases:
        body = "衬里复合管" if "/" in value or inner or "搪玻璃" in desc or "复合钢管" in desc else "直管"
        standard_body = "HG/T2130" if "HG/T2130" in desc else ("CJ/T120" if "CJ/T120" in desc else ("GB/T3091" if "GB/T3091" in desc else "GB/T8163"))
        out = make_base_output(body=body, material=value, standard_body=standard_body)
        out["MATERIAL"][0]["COATING"] = {"INNER": inner, "OUTER": outer}
        if "ERW" in desc:
            out["TYPE"]["MANU"] = ["ERW"]
        samples.append({"input": desc, "output": out, "_source": "semantic_augmentation", "_reason": "MATERIAL_STRUCTURE"})

    standard_cases = [
        ("管子 20 GB/T8163 HG/T20553(Ia) SMLS DN100 STD", "HG/T20553", "Ia"),
        ("管子 20 GB/T8163 HG/T20553 Ia系列 SMLS DN100 STD", "HG/T20553", "Ia"),
        ("管子 20 GB/T8163 HG/T20553 II系列 SMLS DN100 STD", "HG/T20553", "II"),
        ("PIPE S30408 GB/T12771 TYPE I EFW DN100 SCH10S", "GB/T12771", "Type I"),
        ("管子 S30408 GB/T12771 I类 EFW DN100 SCH10S", "GB/T12771", "I类"),
        ("不锈钢无缝钢管PN16-400Ⅱ-0;304;GB/T 14976-2012", "GB/T14976", "II"),
    ]
    for desc, body, grade in standard_cases:
        out = make_base_output(
            manu=["EFW"] if "EFW" in desc else (["SMLS"] if "SMLS" in desc or "无缝" in desc else []),
            material="S30408" if "S30408" in desc else ("304" if "304" in desc else "20"),
            standard_body=body,
            standard_grade=grade,
        )
        if "GB/T8163" in desc:
            out["STANDARD"].insert(0, {"BODY": "GB/T8163", "GRADE": "", "METHOD": "", "APPENDIX": ""})
        samples.append({"input": desc, "output": out, "_source": "semantic_augmentation", "_reason": "STANDARD_GRADE"})

    return samples


def strip_meta(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"input": item["input"], "output": item["output"]} for item in dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建直管语义字段补样与增强样本")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-real", type=int, default=0, help="真实补样最大条数；0 表示不限制")
    args = parser.parse_args()

    real = load_real_supplement(args.candidate, max_rows=args.max_real or None)
    aug = build_augmented_samples()
    combined = strip_meta(real) + strip_meta(aug)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_path = args.output_dir / "直管语义真实补样草稿.json"
    aug_path = args.output_dir / "直管语义增强草稿.json"
    combined_path = args.output_dir / "直管语义补样增强合并草稿.json"
    review_path = args.output_dir / "直管语义补样增强_来源说明.xlsx"

    real_path.write_text(json.dumps(strip_meta(real), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aug_path.write_text(json.dumps(strip_meta(aug), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    review_rows = []
    for item in real + aug:
        review_rows.append({
            "来源": item.get("_source", ""),
            "原因": item.get("_reason", ""),
            "缺口": item.get("_gap", ""),
            "材料描述": item["input"],
            "TYPE": json.dumps(item["output"].get("TYPE", {}), ensure_ascii=False),
            "MATERIAL": json.dumps(item["output"].get("MATERIAL", []), ensure_ascii=False),
            "STANDARD": json.dumps(item["output"].get("STANDARD", []), ensure_ascii=False),
        })
    pd.DataFrame(review_rows).to_excel(review_path, index=False)

    print(f"真实补样: {len(real)} -> {real_path}")
    print(f"增强样本: {len(aug)} -> {aug_path}")
    print(f"合并草稿: {len(combined)} -> {combined_path}")
    print(f"来源说明: {review_path}")


if __name__ == "__main__":
    main()
