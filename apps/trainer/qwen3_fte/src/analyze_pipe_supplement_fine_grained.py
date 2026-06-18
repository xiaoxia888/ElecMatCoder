#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "apps/trainer/qwen3_fte/output/pipe_project_sampling_full"
DRAFT_PATH = BASE / "直管训练草稿.json"
TOTAL_XLSX = Path("/Users/guoxi/Desktop/总数据集.xlsx")
OUT_XLSX = BASE / "直管细粒度补充候选分析.xlsx"
OUT_JSON = BASE / "直管细粒度补充候选.json"

PIPE_CODES = {"P", "PW", "LP", "FP", "LFP", "IP", "PN", "PWM", "PSF", "TW"}
CONN_CODES = ("MNPT", "FNPT", "NPT", "THD", "FTE", "MTE", "SW", "SF")

STD_TOKEN_RE = re.compile(
    r"(ASME(?:B)?\d+(?:\d+)?(?:M)?|"
    r"AB\d+(?:M)?|"
    r"ASTM[A-Z]?\d+[A-Z]?|"
    r"A\d+[A-Z]?|"
    r"GB/T\d+(?:\.\d+)?|GBT\d+(?:\.\d+)?|"
    r"HG/T\d+|HGT\d+|"
    r"SH/T\d+|SHT\d+|"
    r"NB/T\d+|NBT\d+|"
    r"SY/T\d+|SYT\d+|"
    r"CJ/T\d+|CJT\d+|"
    r"ENISO\d+|EN\d+(?:-\d+)?|DINEN\d+(?:-\d+)?|ISO\d+|"
    r"API\d*L?|MSSSP\d+|MSSSP-\d+)"
    r"(TYPEIV|TYPEIII|TYPEII|TYPEI|IV|III|II|IA|I|A|B|C|D)?",
    re.I,
)


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def compact(value: str) -> str:
    value = unicodedata.normalize("NFKC", text(value)).upper()
    value = value.replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV")
    return re.sub(r"[\s,;:()（）\[\]{}_\-/\.]+", "", value)


def norm_std_body(raw: str) -> str:
    raw = compact(raw)
    pairs = [
        ("GBT", "GBT"),
        ("HGT", "HGT"),
        ("SHT", "SHT"),
        ("NBT", "NBT"),
        ("SYT", "SYT"),
        ("CJT", "CJT"),
    ]
    for prefix, normalized in pairs:
        if raw.startswith(prefix):
            return normalized + raw[len(prefix) :]
    return raw


def norm_std_suffix(raw: str) -> str:
    raw = compact(raw)
    mapping = {
        "IA": "Ia",
        "TYPEI": "Type I",
        "TYPEII": "Type II",
        "TYPEIII": "Type III",
        "TYPEIV": "Type IV",
    }
    return mapping.get(raw, raw)


def parse_standard_code(value: str) -> tuple[str, str]:
    raw = compact(value)
    if not raw:
        return "EMPTY", "EMPTY"
    parts: list[str] = []
    for match in STD_TOKEN_RE.finditer(raw):
        body = norm_std_body(match.group(1))
        suffix = norm_std_suffix(match.group(2) or "")
        parts.append(f"{body}[{suffix}]" if suffix else body)
    if not parts:
        return "UNPARSED", raw
    return "+".join(parts), "+".join(sorted(set(parts)))


def first_nonempty(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = text(row.get(name))
        if value:
            return value
    return ""


def infer_type_code(row: dict[str, Any]) -> str:
    code = first_nonempty(row, "修正种类", "标准化种类", "原始种类")
    if code:
        return code
    full = first_nonempty(row, "修正后编码", "编码")
    match = re.match(r"[A-Z]+", compact(full))
    return match.group(0) if match else ""


def in_pipe_scope(row: dict[str, Any]) -> bool:
    category = text(row.get("分类"))
    type_code = infer_type_code(row)
    return category in {"直管", "管子"} and type_code in PIPE_CODES


def listify(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [text(x) for x in value if text(x)]
    return [text(value)]


def item_standard_code(item: dict[str, Any]) -> str:
    standards = item.get("output", {}).get("STANDARD") or []
    parts = []
    for std in standards:
        if not isinstance(std, dict):
            continue
        body = compact(std.get("BODY", ""))
        grade = compact(std.get("GRADE", ""))
        appendix = compact(std.get("APPENDIX", ""))
        part = body + (grade if grade else "") + (appendix if appendix else "")
        if part:
            parts.append(part)
    return "+".join(parts)


def detect_manu(text_value: str) -> str:
    hits: list[str] = []
    for token, pattern in [
        ("DSAWH", r"DSAWH|双面.*螺旋|螺旋.*双面"),
        ("DSAWL", r"DSAWL|双面.*直缝|直缝.*双面"),
        ("SAWH", r"SAWH|螺旋缝埋弧|螺旋焊|螺旋缝"),
        ("SAWL", r"SAWL|直缝埋弧"),
        ("DSAW", r"\bDSAW\b|双面埋弧"),
        ("SAW", r"\bSAW\b|埋弧焊"),
        ("ERW", r"\bERW\b|电阻焊"),
        ("HFW", r"\bHFW\b|高频焊"),
        ("EFW", r"\bEFW\b|电熔焊"),
        ("SMLS", r"SMLS|SEAMLESS|无缝"),
        ("WELDED", r"WELDED(?:\s*PIPE)?|PIPE\s*WELD(?:ED)?|焊接钢管|焊管|有缝"),
    ]:
        if re.search(pattern, text_value, re.I):
            hits.append(token)
    hits = list(dict.fromkeys(hits))
    for child, parents in {
        "DSAWH": {"DSAW", "SAW", "WELDED"},
        "DSAWL": {"DSAW", "SAW", "WELDED"},
        "SAWH": {"SAW", "WELDED"},
        "SAWL": {"SAW", "WELDED"},
        "DSAW": {"SAW", "WELDED"},
        "ERW": {"WELDED"},
        "HFW": {"WELDED"},
        "EFW": {"WELDED"},
    }.items():
        if child in hits:
            hits = [h for h in hits if h not in parents]
    return "+".join(hits) or "EMPTY"


def detect_conn(text_value: str) -> str:
    s = unicodedata.normalize("NFKC", text_value).upper()
    hits = []
    for token in CONN_CODES:
        if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", s):
            hits.append(token)
    if re.search(r"承插|SOCKET", s) and "SW" not in hits:
        hits.append("SW")
    if re.search(r"螺纹|THREAD|THRD|SCRD", s) and "THD" not in hits:
        hits.append("THD")
    return "+".join(dict.fromkeys(hits)) or "EMPTY"


def detect_size_format(text_value: str, code_value: str = "") -> str:
    s = unicodedata.normalize("NFKC", text_value).upper()
    formats = []
    if re.search(r"\bDN\s*\d+\s*[X×*]\s*DN\s*\d+\b", s):
        formats.append("DNxDN_EXPLICIT")
    elif re.search(r"\bDN\s*\d+\s*[X×*]\s*\d+\b", s):
        formats.append("DNxNUM")
    if re.search(r"\bNPS\s*\d+(?:-\d+/\d+|\s+\d+/\d+|\.\d+)?\b", s):
        formats.append("NPS")
    if re.search(r"\b\d+\s*-\s*\d+/\d+\s*\"", s):
        formats.append("INCH_FRAC_DASH")
    if re.search(r"\b\d+\s+\d+/\d+\s*\"", s):
        formats.append("INCH_FRAC_SPACE")
    if re.search(r"\b\d+\.\d+\s*\"", s):
        formats.append("INCH_DECIMAL_QUOTE")
    if re.search(r"\b\d+\s*\"", s):
        formats.append("INCH_INT_QUOTE")
    if re.search(r"[Φφ]\s*\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?", s):
        formats.append("ODxTHK_TEXT")
    elif re.search(r"\b\d{2,4}\.\d+\s*[X×*]\s*\d+(?:\.\d+)?", s):
        formats.append("ODxTHK_TEXT")
    elif re.search(r"\bD\s*\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?", s):
        formats.append("ODxTHK_D_PREFIX")
    if re.search(r"\bDN\s*\d+\b", s):
        formats.append("DN")
    if not formats:
        code = compact(code_value)
        if "X" in code and code:
            formats.append("CODE_COMBO")
        elif code:
            formats.append("CODE_SINGLE")
    return "+".join(dict.fromkeys(formats)) or "EMPTY"


def detect_thk_format(text_value: str, code_value: str = "") -> str:
    s = unicodedata.normalize("NFKC", text_value).upper()
    formats = []
    if re.search(r"SCH\s*\w+\s*[X×*]\s*\w+", s):
        formats.append("SCHxSCH")
    elif re.search(r"SCH\.?\s*\w+", s):
        formats.append("SCH")
    if re.search(r"\bS-\s*(STD|XS|XXS|\d+\w*)", s):
        formats.append("S_PREFIX")
    if re.search(r"\b(STD|XXS|XS)\b", s):
        formats.append("SERIES")
    if re.search(r"THK\s*=\s*\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?\s*MM", s):
        formats.append("THK_EQ_MMxMM")
    elif re.search(r"THK\s*=?\s*\d+(?:\.\d+)?\s*MM", s):
        formats.append("THK_EQ_MM")
    if re.search(r"\b\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?\s*MM", s):
        formats.append("MMxMM")
    elif re.search(r"\b\d+(?:\.\d+)?\s*MM\b", s):
        formats.append("MM")
    if re.search(r"\b\d+(?:\.\d+)?\s*IN\b|\b0?\.\d+\s*\"", s):
        formats.append("INCH_THK")
    if re.search(r"[ΦφD]?\s*\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?", s):
        formats.append("ODxTHK_CONTEXT")
    if not formats:
        code = compact(code_value)
        if "X" in code and code:
            formats.append("CODE_COMBO")
        elif code:
            formats.append("CODE_SINGLE")
    return "+".join(dict.fromkeys(formats)) or "EMPTY"


def material_family(value: str) -> str:
    raw = compact(value)
    if not raw:
        return "EMPTY"
    if any(x in raw for x in ["PTFE", "RPTFE", "FRP", "PVC", "CPVC", "EAA", "GLASS", "搪玻璃", "PEEP"]):
        return "衬里复合"
    if any(x in raw for x in ["TA2", "TA10", "TITANIUM", "钛"]):
        return "钛"
    if any(x in raw for x in ["PVCU", "PVC", "CPVC", "FRP"]):
        return "塑料/非金属"
    if any(x in raw for x in ["304", "316", "321", "2205", "2507", "S304", "S316", "06CR", "022CR", "N066"]):
        return "不锈钢/镍基/双相"
    if any(x in raw for x in ["15CR", "12CR", "A335", "P11", "P22"]):
        return "合金钢"
    return "碳钢/低合金"


def material_expression_sig(desc: str, code_value: str) -> str:
    s = unicodedata.normalize("NFKC", desc).upper()
    code = compact(code_value) or "EMPTY"
    exprs = []
    if re.search(r"ASTM\s*A312|A312", s):
        exprs.append("ASTM_A312_GRADE")
    if re.search(r"ASTM\s*A358|A358", s):
        exprs.append("ASTM_A358_GRADE")
    if re.search(r"ASTM\s*A106|A106", s):
        exprs.append("ASTM_A106_GRADE")
    if re.search(r"ASTM\s*A672|A672", s):
        exprs.append("ASTM_A672_GRADE")
    if re.search(r"ASTM\s*A671|A671", s):
        exprs.append("ASTM_A671_GRADE")
    if re.search(r"GB/T\s*\d+[^,;，；]{0,18}(S\d{5}|06CR\d+|022CR\d+|20#?|Q235B|L245)", s):
        exprs.append("GB_STANDARD_GRADE")
    if re.search(r"\bS\d{5}\b", s):
        exprs.append("CHINA_UNS")
    if re.search(r"\b0?6CR\d+", s):
        exprs.append("CHINA_ELEMENT")
    if re.search(r"\b022CR\d+", s):
        exprs.append("CHINA_LOW_C")
    if re.search(r"\bSF\d{3,4}\b", s):
        exprs.append("SF_PREFIX")
    if re.search(r"\bTP\d{3,4}L?\b", s):
        exprs.append("TP_PREFIX")
    if re.search(r"\bGR(?:ADE)?\.?\s*[A-Z0-9.]+", s):
        exprs.append("GRADE_WORD")
    if re.search(r"\b\d{3,4}L?\b", s):
        exprs.append("BARE_NUMERIC")
    if re.search(r"FRP|PVC|CPVC|PTFE|RPTFE|EAA|搪玻璃", s):
        exprs.append("COMPOSITE_TEXT")
    return f"{code}||{'+'.join(dict.fromkeys(exprs)) or 'UNKNOWN_EXPR'}"


def standard_signature_from_output(item: dict[str, Any]) -> str:
    return parse_standard_code(item_standard_code(item))[1]


def field_for_total(row: dict[str, Any], *cols: str) -> str:
    return first_nonempty(row, *cols)


def summarize_counter(counter: Counter[str]) -> pd.DataFrame:
    total = sum(counter.values())
    return pd.DataFrame(
        [{"签名": k, "数量": v, "占比": round(v / total, 4) if total else 0} for k, v in counter.most_common()]
    )


def add_examples(df: pd.DataFrame, key_col: str, limit_per_key: int = 8) -> pd.DataFrame:
    rows = []
    seen: Counter[str] = Counter()
    for _, row in df.iterrows():
        key = row[key_col]
        if seen[key] >= limit_per_key:
            continue
        seen[key] += 1
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
    total_df = pd.read_excel(TOTAL_XLSX, sheet_name="Sheet1")
    total_rows = total_df.fillna("").to_dict("records")
    pipe_rows = [row for row in total_rows if in_pipe_scope(row)]

    current_records = []
    for idx, item in enumerate(draft):
        out = item.get("output", {})
        text_value = text(item.get("input"))
        current_records.append(
            {
                "来源": "当前草稿",
                "行号": idx,
                "材料描述": text_value,
                "BODY": text(out.get("TYPE", {}).get("BODY")),
                "MANU_SIG": "+".join(listify(out.get("TYPE", {}).get("MANU"))) or "EMPTY",
                "CONN_SIG": "+".join(listify(out.get("TYPE", {}).get("CONN"))) or "EMPTY",
                "SIZE_SIG": detect_size_format(text_value),
                "THK_SIG": detect_thk_format(text_value),
                "MAT_FAMILY": material_family(json.dumps(out.get("MATERIAL", []), ensure_ascii=False)),
                "MAT_EXPR_SIG": material_expression_sig(
                    text_value,
                    "+".join(
                        text(m.get("VALUE", ""))
                        for m in (out.get("MATERIAL") or [])
                        if isinstance(m, dict)
                    ),
                ),
                "STD_SIG": standard_signature_from_output(item),
                "STD_CODE": item_standard_code(item),
            }
        )

    candidate_records = []
    for _, row in enumerate(pipe_rows):
        desc = text(row.get("材料描述"))
        std_code = field_for_total(row, "修正规范", "标准化规范", "原始规范")
        mat_code = field_for_total(row, "修正材质", "标准化材质", "原始材质")
        std_combo, std_sig = parse_standard_code(std_code)
        candidate_records.append(
            {
                "来源": "总表候选",
                "Excel行号": int(row.get("__row__", 0)) if row.get("__row__") else "",
                "材料描述": desc,
                "类型编码": infer_type_code(row),
                "BODY": field_for_total(row, "修正种类", "标准化种类", "原始种类"),
                "MANU_SIG": detect_manu(desc),
                "CONN_SIG": detect_conn(desc),
                "SIZE_SIG": detect_size_format(desc, field_for_total(row, "修正尺寸", "标准化尺寸", "原始尺寸")),
                "THK_SIG": detect_thk_format(desc, field_for_total(row, "修正壁厚", "标准化壁厚", "原始壁厚")),
                "MAT_FAMILY": material_family(mat_code),
                "MAT_EXPR_SIG": material_expression_sig(desc, mat_code),
                "STD_SIG": std_sig,
                "STD_CODE": std_code,
                "STD_COMBO": std_combo,
                "尺寸编码": field_for_total(row, "修正尺寸", "标准化尺寸", "原始尺寸"),
                "壁厚编码": field_for_total(row, "修正壁厚", "标准化壁厚", "原始壁厚"),
                "材质编码": mat_code,
                "总编码": field_for_total(row, "修正后编码", "编码"),
            }
        )

    current_df = pd.DataFrame(current_records)
    candidate_df = pd.DataFrame(candidate_records)

    dimensions = ["MANU_SIG", "CONN_SIG", "SIZE_SIG", "THK_SIG", "MAT_FAMILY", "MAT_EXPR_SIG", "STD_SIG"]
    gap_rows = []
    for dim in dimensions:
        current_counter = Counter(current_df[dim])
        candidate_counter = Counter(candidate_df[dim])
        for sig, cand_count in candidate_counter.most_common():
            cur_count = current_counter.get(sig, 0)
            if cur_count < 30 or sig not in current_counter:
                gap_rows.append(
                    {
                        "维度": dim,
                        "签名": sig,
                        "当前数量": cur_count,
                        "总表候选全量": cand_count,
                        "建议": "优先补" if cur_count < 10 else "可补",
                    }
                )

    joint_dims = [
        ("STD_SIG", "THK_SIG"),
        ("MANU_SIG", "STD_SIG"),
        ("SIZE_SIG", "THK_SIG"),
        ("CONN_SIG", "SIZE_SIG"),
        ("MAT_EXPR_SIG", "STD_SIG"),
    ]
    joint_gap_rows = []
    for a, b in joint_dims:
        cur_counter = Counter(zip(current_df[a], current_df[b]))
        cand_counter = Counter(zip(candidate_df[a], candidate_df[b]))
        for (va, vb), cand_count in cand_counter.most_common():
            cur_count = cur_counter.get((va, vb), 0)
            if cur_count < 10 and cand_count >= 3:
                joint_gap_rows.append(
                    {
                        "维度组合": f"{a}+{b}",
                        a: va,
                        b: vb,
                        "当前数量": cur_count,
                        "总表候选全量": cand_count,
                    }
                )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(gap_rows).to_excel(writer, sheet_name="缺口汇总", index=False)
        pd.DataFrame(joint_gap_rows).to_excel(writer, sheet_name="联合缺口", index=False)
        for dim in dimensions:
            summarize_counter(Counter(current_df[dim])).to_excel(writer, sheet_name=f"当前_{dim}"[:31], index=False)
            summarize_counter(Counter(candidate_df[dim])).to_excel(writer, sheet_name=f"总表_{dim}"[:31], index=False)
            cols = [
                "材料描述",
                "类型编码",
                "BODY",
                dim,
                "STD_CODE",
                "STD_COMBO",
                "尺寸编码",
                "壁厚编码",
                "材质编码",
                "总编码",
            ]
            add_examples(candidate_df.sort_values(dim), dim)[cols].to_excel(
                writer, sheet_name=f"候选_{dim}"[:31], index=False
            )

    summary = {
        "draft_count": len(current_df),
        "candidate_count": len(candidate_df),
        "gap_rows": len(gap_rows),
        "joint_gap_rows": len(joint_gap_rows),
        "top_gaps": gap_rows[:50],
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUT_XLSX)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
