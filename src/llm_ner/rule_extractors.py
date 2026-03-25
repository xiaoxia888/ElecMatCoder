# -*- coding: utf-8 -*-
"""
规则抽取器（Hybrid 后处理）

规则配置统一来源于 ontology.yaml（配置驱动）。
"""

import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml


logger = logging.getLogger(__name__)


def canonicalize_size_token(token: str, subtype: str = "") -> str:
    """
    尺寸 canonical 归一（仅用于规则-模型比对）。
    """
    if not token:
        return ""
    s = str(token).strip().upper()
    subtype = str(subtype or "").strip().upper()
    s = s.replace("”", "\"").replace("″", "\"").replace("“", "\"")

    if subtype == "DN":
        m = re.search(r"DN\s*-?\s*(\d+(?:\.\d+)?)", s)
        if m:
            return f"DN{_normalize_number_text(m.group(1))}"
        return s.replace(" ", "")

    if subtype == "OD":
        # OD=89 / Φ89 / φ89 / OD89MM -> OD89 / OD89MM
        m = re.search(r"(?:OD\s*=?\s*|[Φφ])(\d+(?:\.\d+)?)(\s*MM)?", s)
        if m:
            num = _normalize_number_text(m.group(1))
            suffix = "MM" if (m.group(2) or "").strip() else ""
            return f"OD{num}{suffix}"
        m = re.search(r"[Φφ]\s*(\d+(?:\.\d+)?)(\s*MM)?", s)
        if m:
            num = _normalize_number_text(m.group(1))
            suffix = "MM" if (m.group(2) or "").strip() else ""
            return f"OD{num}{suffix}"
        m = re.match(r"^(\d+(?:\.\d+)?)(\s*MM)?$", s)
        if m:
            num = _normalize_number_text(m.group(1))
            suffix = "MM" if (m.group(2) or "").strip() else ""
            return f"OD{num}{suffix}"
        return s.replace(" ", "")

    if subtype == "INCH":
        return re.sub(r"\s+", "", s)

    if subtype == "LENGTH":
        m = re.search(r"(?:L|LEN|LENGTH)\s*=?\s*(\d+(?:\.\d+)?)(\s*(?:MM|CM|M))?", s)
        if m:
            num = _normalize_number_text(m.group(1))
            unit = (m.group(2) or "").strip()
            return f"L={num}{unit}" if unit else f"L={num}"
        return s.replace(" ", "")

    return re.sub(r"\s+", "", s)


def _normalize_number_text(num: str) -> str:
    try:
        v = float(str(num))
    except Exception:
        return str(num).strip()
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def _canonicalize_thickness_single(token: str, subtype: str = "") -> str:
    if not token:
        return ""

    t = str(token).strip().upper()
    t = t.replace(" ", "")
    t = t.replace("”", "\"").replace("″", "\"").replace("“", "\"")

    subtype = str(subtype or "").strip().upper()

    # MM 子类型：数字统一补 MM
    if subtype == "MM":
        m = re.match(r"^(?:T|THK)=?(\d+(?:\.\d+)?)(?:MM)?$", t)
        if m:
            return f"{_normalize_number_text(m.group(1))}MM"
        m = re.match(r"^(\d+(?:\.\d+)?)(?:MM)?$", t)
        if m:
            return f"{_normalize_number_text(m.group(1))}MM"

    if subtype == "INCH":
        return t.replace(" ", "")

    if subtype == "BWG":
        m = re.match(r"^(\d+)\s*BWG$", t)
        if m:
            return f"{m.group(1)}BWG"
        return t

    series_tokens = _get_thickness_series_tokens()
    if t in series_tokens:
        return t

    m = re.match(r"^SCH[.\s]*(XXS|XS|STD|\d+(?:\.\d+)?S?)$", t)
    if m:
        core = m.group(1)
        return core if core in series_tokens else f"SCH{core}"

    m = re.match(r"^S-(XXS|XS|STD|\d+(?:\.\d+)?S?)$", t)
    if m:
        core = m.group(1)
        return core if core in series_tokens else f"SCH{core}"

    m = re.match(r"^S(\d+(?:\.\d+)?S?)$", t)
    if m:
        return f"SCH{m.group(1)}"

    # 通用 mm 兜底
    m = re.match(r"^(?:T|THK)=?(\d+(?:\.\d+)?)(?:MM)?$", t)
    if m:
        return f"{_normalize_number_text(m.group(1))}MM"
    m = re.match(r"^(\d+(?:\.\d+)?)MM$", t)
    if m:
        return f"{_normalize_number_text(m.group(1))}MM"

    return t


def canonicalize_thickness_token(token: str, subtype: str = "") -> str:
    """
    将壁厚 token 归一为 canonical 形式。

    例如:
      - SCH80 / SCH 80 / S-80 / S80 -> S80
      - SCH40S / S-40S / S40S -> S40S
      - STD / XS / XXS -> 原样
    """
    if not token:
        return ""

    raw = str(token).strip()
    if not raw:
        return ""

    subtype_u = str(subtype or "").strip().upper()
    if subtype_u in {"INCH", "BWG"}:
        return _canonicalize_thickness_single(raw, subtype=subtype_u)

    # 先处理系列常量，避免 XXS 被 X 分隔逻辑误拆
    maybe_series = _canonicalize_thickness_single(raw, subtype=subtype)
    if maybe_series in _get_thickness_series_tokens():
        return maybe_series

    # 多段壁厚统一处理：X/*/×/, 均视作分隔符
    split_by_x = bool(re.search(r"[xX]", raw) and re.search(r"\d", raw))
    has_other_sep = bool(re.search(r"[×*/,]", raw))
    if split_by_x or has_other_sep:
        sep_pat = r"\s*[×*/,]\s*" if not split_by_x else r"\s*[xX×*/,]\s*"
        parts = [p for p in re.split(sep_pat, raw) if p and p.strip()]
        cparts = [_canonicalize_thickness_single(p, subtype=subtype) for p in parts]
        cparts = [p for p in cparts if p]
        if not cparts:
            return ""
        # 如果 MM 复合表达里只有部分带单位，统一补齐
        if subtype.upper() == "MM" and any(p.endswith("MM") for p in cparts):
            cparts = [p if p.endswith("MM") else f"{p}MM" for p in cparts]
        return "X".join(cparts)

    return _canonicalize_thickness_single(raw, subtype=subtype)


def canonical_to_structured_thickness(canonical: str, subtype: str = "") -> Dict[str, str]:
    """
    将 canonical 壁厚值映射为结构化 THICKNESS 子类型和值。
    """
    c = canonicalize_thickness_token(canonical, subtype=subtype)
    if not c:
        return {"subtype": "", "value": ""}

    subtype = str(subtype or "").strip().upper()
    if subtype in {"MM", "INCH", "BWG"}:
        return {"subtype": subtype, "value": c}

    if c in _get_thickness_series_tokens():
        return {"subtype": "SERIES", "value": c}

    m = re.match(r"^S(?:CH)?(\d+(?:\.\d+)?S?)$", c)
    if m:
        return {"subtype": "SCHEDULE", "value": f"SCH{m.group(1)}"}

    return {"subtype": "SCHEDULE", "value": c}


@lru_cache(maxsize=1)
def _load_ontology_config() -> Dict:
    """
    加载 ontology 配置（YAML）。
    配置文件路径: src/llm_ner/config/ontology.yaml
    """
    path = Path(__file__).parent / "config" / "ontology.yaml"
    if not path.exists():
        logger.warning(f"ontology配置不存在: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception as e:
        logger.warning(f"加载ontology配置失败: {e}")
        return {}


def _get_thickness_series_tokens() -> set:
    cfg = _load_ontology_config()
    thk = (cfg.get("fields") or {}).get("THICKNESS") or {}
    norm = thk.get("normalization") or {}
    tokens = norm.get("series_tokens") or []
    if not isinstance(tokens, list):
        return {"STD", "XS", "XXS"}
    clean = {str(t).strip().upper() for t in tokens if str(t).strip()}
    return clean or {"STD", "XS", "XXS"}


def extract_material_special_req_rules(text: str) -> List[Dict[str, str]]:
    """
    从原文提取材质特殊要求（当前聚焦抗硫语义）。
    返回统一语义值 SPECIAL_REQ='抗硫'。
    """
    if not text:
        return []

    cfg = _load_ontology_config()
    material = (cfg.get("fields") or {}).get("MATERIAL") or {}
    special = material.get("special_requirements") or {}
    rule_items = special.get("rules") or []
    if not isinstance(rule_items, list):
        return []

    hits: List[Dict[str, str]] = []
    seen = set()

    for item in rule_items:
        if not isinstance(item, dict):
            continue
        semantic = str(item.get("semantic") or "").strip()
        special_req = str(item.get("value") or "").strip()
        patterns = item.get("patterns") or []
        if not special_req or not isinstance(patterns, list):
            continue

        for pattern in patterns:
            try:
                pat = re.compile(str(pattern), re.IGNORECASE)
            except re.error:
                logger.warning(f"无效语义规则正则: {pattern}")
                continue

            for m in pat.finditer(text):
                start = m.start()
                end = m.end()
                key = (start, end, special_req)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "raw": m.group(0),
                        "semantic": semantic,
                        "special_req": special_req,
                        "start": str(start),
                        "_end": str(end),
                    }
                )

    # 去重策略：同一 special_req 下，优先保留更长命中并移除被包含的短命中。
    hits.sort(key=lambda x: (int(x["start"]), -len(x["raw"])))
    selected: List[Dict[str, str]] = []
    for h in hits:
        h_start = int(h["start"])
        h_end = int(h.get("_end", h_start + len(h["raw"])))
        h_req = h.get("special_req", "")

        covered = False
        for s in selected:
            s_start = int(s["start"])
            s_end = int(s.get("_end", s_start + len(s["raw"])))
            if s.get("special_req", "") == h_req and s_start <= h_start and s_end >= h_end:
                covered = True
                break
        if not covered:
            selected.append(h)

    for h in selected:
        h.pop("_end", None)
    selected.sort(key=lambda x: int(x["start"]))
    return selected


def extract_thickness_rules(text: str) -> List[Dict[str, str]]:
    """
    从原文高精度提取壁厚 token，并给出 canonical/subtype 建议。
    规则来源：ontology.yaml.thickness.extraction_rules
    """
    if not text:
        return []

    cfg = _load_ontology_config()
    thk = (cfg.get("fields") or {}).get("THICKNESS") or {}
    subtypes = thk.get("subtypes") or {}
    if not isinstance(subtypes, dict):
        return []

    src = str(text).upper()
    hits: List[Dict[str, str]] = []
    seen = set()

    length_spans: List[tuple] = []
    for lm in re.finditer(r"\b(?:L|LEN|LENGTH)\s*=\s*\d+(?:\.\d+)?\s*(?:MM|CM|M)\b", src, flags=re.IGNORECASE):
        length_spans.append((lm.start(), lm.end()))

    def _in_length_span(s: int, e: int) -> bool:
        for ls, le in length_spans:
            if ls <= s and e <= le:
                return True
        return False

    for subtype, scfg in subtypes.items():
        subtype = str(subtype).strip().upper()
        rules = (scfg or {}).get("rules") if isinstance(scfg, dict) else None
        if not subtype or not isinstance(rules, list):
            continue

        for item in rules:
            if not isinstance(item, dict):
                continue
            pattern = str(item.get("pattern") or "").strip()
            canonical_from_match = bool(item.get("canonical_from_match"))
            if not pattern:
                continue
            try:
                pat = re.compile(pattern, flags=re.IGNORECASE)
            except re.error:
                logger.warning(f"无效壁厚规则正则: {pattern}")
                continue

            for m in pat.finditer(src):
                raw = m.group(0).strip()
                if canonical_from_match and m.lastindex:
                    src_text = str(m.group(1)).strip()
                else:
                    src_text = raw
                if subtype == "MM" and _in_length_span(m.start(), m.end()):
                    continue
                canonical = canonicalize_thickness_token(src_text, subtype=subtype)
                if not canonical:
                    continue
                final_subtype = subtype
                if canonical in _get_thickness_series_tokens():
                    final_subtype = "SERIES"
                start = m.start()
                end = m.end()
                key = (canonical, start, final_subtype)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "raw": raw,
                        "canonical": canonical,
                        "subtype": final_subtype,
                        "start": str(start),
                        "_end": str(end),
                    }
                )

    # 去重策略：同一 subtype 下，优先保留更长命中并移除被包含短命中（避免 T=4x3.5mm 内再命中 3.5mm/5mm）。
    hits.sort(key=lambda x: (int(x["start"]), -len(x["raw"])))
    selected: List[Dict[str, str]] = []
    for h in hits:
        h_start = int(h["start"])
        h_end = int(h.get("_end", h_start + len(h["raw"])))
        h_sub = h.get("subtype", "")

        covered = False
        for s in selected:
            s_start = int(s["start"])
            s_end = int(s.get("_end", s_start + len(s["raw"])))
            if s.get("subtype", "") == h_sub and s_start <= h_start and s_end >= h_end:
                covered = True
                break
        if not covered:
            selected.append(h)

    for h in selected:
        h.pop("_end", None)
    selected.sort(key=lambda x: int(x["start"]))
    return selected


def extract_size_rules(text: str) -> List[Dict[str, str]]:
    """
    从原文提取尺寸 token（SIZE），规则来源：ontology.yaml.size.extraction_rules
    """
    if not text:
        return []

    cfg = _load_ontology_config()
    size_cfg = (cfg.get("fields") or {}).get("SIZE") or {}
    subtypes = size_cfg.get("subtypes") or {}
    if not isinstance(subtypes, dict):
        return []

    src = str(text)
    hits: List[Dict[str, str]] = []
    seen = set()
    for subtype, scfg in subtypes.items():
        subtype = str(subtype).strip().upper()
        rules = (scfg or {}).get("rules") if isinstance(scfg, dict) else None
        if not subtype or not isinstance(rules, list):
            continue
        for item in rules:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("id") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            canonical_from_match = bool(item.get("canonical_from_match"))
            mapped_value = str(item.get("canonical") or "").strip()
            if not pattern:
                continue
            try:
                pat = re.compile(pattern, flags=re.IGNORECASE)
            except re.error:
                logger.warning(f"无效尺寸规则正则: {pattern}")
                continue

            for m in pat.finditer(src):
                raw = m.group(0).strip()
                if canonical_from_match and m.lastindex:
                    src_text = str(m.group(1)).strip()
                elif mapped_value:
                    src_text = mapped_value
                else:
                    src_text = raw
                canonical = canonicalize_size_token(src_text, subtype=subtype)
                if not canonical:
                    continue
                start = m.start()
                end = m.end()
                key = (subtype, canonical, start)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "raw": raw,
                        "canonical": canonical,
                        "subtype": subtype,
                        "rule_id": rule_id,
                        "source": "explicit_pattern",
                        "confidence": "high",
                        "start": str(start),
                        "_end": str(end),
                    }
                )

    # DNxN 补提：对 DN150x100 这类表达，补提右值为第二个 DN（DN100）。
    # 注意：仅在右值是整数时补提；小整数歧义（如 DN30x10）仍由歧义规则告警。
    amb = size_cfg.get("ambiguity") or {}
    dnxn = amb.get("dnxn") or {}
    dnxn_pattern = str(dnxn.get("pattern") or "").strip()
    if dnxn_pattern:
        try:
            dnxn_pat = re.compile(dnxn_pattern, flags=re.IGNORECASE)
            for m in dnxn_pat.finditer(src):
                rhs = str(m.group(2) or "").strip()
                if not rhs or "." in rhs:
                    continue
                right_dn = f"DN{_normalize_number_text(rhs)}"
                start = m.start(2)
                end = m.end(2)
                key = ("DN", right_dn, start)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    {
                        "raw": rhs,
                        "canonical": right_dn,
                        "subtype": "DN",
                        "rule_id": "dnxn_rhs_completion",
                        "source": "dnxn_rhs_inferred",
                        "confidence": "low",
                        "start": str(start),
                        "_end": str(end),
                    }
                )
        except re.error:
            logger.warning(f"无效DNxN补提正则: {dnxn_pattern}")

    # 同 subtype 下按包含关系去重，优先长串
    hits.sort(key=lambda x: (int(x["start"]), -len(x["raw"])))
    selected: List[Dict[str, str]] = []
    for h in hits:
        hs, he = int(h["start"]), int(h["_end"])
        st = h["subtype"]
        covered = False
        for s in selected:
            ss, se = int(s["start"]), int(s["_end"])
            if s["subtype"] == st and ss <= hs and se >= he:
                covered = True
                break
        if not covered:
            selected.append(h)

    for h in selected:
        h.pop("_end", None)
    selected.sort(key=lambda x: int(x["start"]))
    return selected


def extract_dnxn_ambiguity_rules(text: str) -> List[Dict[str, str]]:
    """
    提取 DNx数字 歧义模式。
    仅用于告警，不直接落字段。

    规则：
    - DN50X3.2  -> 倾向 THICKNESS.MM
    - DN50X32   -> 倾向 SIZE(异径/第二尺寸)
    """
    if not text:
        return []

    cfg = _load_ontology_config()
    size_cfg = (cfg.get("fields") or {}).get("SIZE") or {}
    amb = size_cfg.get("ambiguity") or {}
    dnxn = amb.get("dnxn") or {}
    pattern = str(dnxn.get("pattern") or "").strip()
    if not pattern:
        return []
    threshold = dnxn.get("small_integer_threshold", 20)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 20.0

    src = str(text).upper()
    # 允许 X/x/×/*
    pat = re.compile(pattern, re.IGNORECASE)
    hits: List[Dict[str, str]] = []
    for m in pat.finditer(src):
        dn_num = m.group(1)
        rhs = m.group(2)
        rhs_val = float(rhs)

        if "." in rhs:
            suggest_field = "THICKNESS"
            suggest_subtype = "MM"
            reason = "dn_x_decimal"
        elif rhs_val <= threshold:
            # 小整数也可能是壁厚，属于高歧义区
            suggest_field = "THICKNESS"
            suggest_subtype = "MM"
            reason = "dn_x_small_integer"
        else:
            suggest_field = "SIZE"
            suggest_subtype = "DN"
            reason = "dn_x_large_integer"

        hits.append(
            {
                "raw": m.group(0),
                "dn_left": f"DN{_normalize_number_text(dn_num)}",
                "right_value": _normalize_number_text(rhs),
                "suggest_field": suggest_field,
                "suggest_subtype": suggest_subtype,
                "reason": reason,
                "start": str(m.start()),
            }
        )

    return hits


def extract_type_rules(text: str) -> List[Dict[str, str]]:
    """
    提取 TYPE 可规则化子字段（MANU/ENDS/SEAL/CONN），不处理 BODY。
    规则来源：ontology.yaml.fields.TYPE.subtypes
    """
    if not text:
        return []

    cfg = _load_ontology_config()
    type_cfg = (cfg.get("fields") or {}).get("TYPE") or {}
    subtypes_cfg = type_cfg.get("subtypes") or {}
    if not isinstance(subtypes_cfg, dict):
        return []

    src = str(text)
    hits: List[Dict[str, str]] = []
    seen = set()

    def _add_hit(subtype: str, canonical: str, raw: str, start: int, end: int, rule_id: str = ""):
        subtype = str(subtype or "").strip().upper()
        canonical = str(canonical or "").strip().upper()
        raw = str(raw or "").strip()
        if not subtype or not canonical or not raw:
            return
        key = (subtype, canonical, start, end)
        if key in seen:
            return
        seen.add(key)
        hits.append(
            {
                "raw": raw,
                "subtype": subtype,
                "canonical": canonical,
                "source": "explicit_pattern",
                "confidence": "high",
                "rule_id": str(rule_id or "").strip(),
                "start": str(start),
                "_end": str(end),
            }
        )

    for subtype, scfg in subtypes_cfg.items():
        if not isinstance(scfg, dict):
            continue
        subtype = str(subtype).strip().upper()
        if not subtype:
            continue

        rules = scfg.get("rules") or []
        if not isinstance(rules, list):
            continue
        for p in rules:
            if not isinstance(p, dict):
                continue
            rule_id = str(p.get("id") or "").strip()
            pattern = str(p.get("pattern") or "").strip()
            mapped_value = str(p.get("canonical") or "").strip()
            canonical_from_match = bool(p.get("canonical_from_match"))
            if not pattern:
                continue
            try:
                pat = re.compile(pattern, flags=re.IGNORECASE)
            except re.error:
                logger.warning(f"无效TYPE规则正则: {pattern}")
                continue
            for m in pat.finditer(src):
                if canonical_from_match:
                    canonical = str(m.group(1) if m.lastindex else m.group(0)).strip().upper()
                else:
                    canonical = mapped_value.upper() if mapped_value else str(m.group(0)).strip().upper()
                _add_hit(subtype, canonical, m.group(0), m.start(), m.end(), rule_id=rule_id)

    # 同 subtype+canonical 去重：保留更长覆盖
    hits.sort(key=lambda x: (x["subtype"], x["canonical"], int(x["start"]), -len(x["raw"])))
    selected: List[Dict[str, str]] = []
    for h in hits:
        hs, he = int(h["start"]), int(h["_end"])
        covered = False
        for s in selected:
            if s["subtype"] != h["subtype"] or s["canonical"] != h["canonical"]:
                continue
            ss, se = int(s["start"]), int(s["_end"])
            if ss <= hs and se >= he:
                covered = True
                break
        if not covered:
            selected.append(h)

    for h in selected:
        h.pop("_end", None)
    selected.sort(key=lambda x: int(x["start"]))
    return selected
