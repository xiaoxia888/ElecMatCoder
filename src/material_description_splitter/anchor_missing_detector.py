# -*- coding: utf-8 -*-
"""Detector for missing-anchor naked numbers."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import DifficultyFeature, GlueHit


NAKED_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")
NAKED_SPEC_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"\d+(?:\.\d+)?\s*(?:[xX×*/-]|/)\s*\d+(?:\.\d+)?(?:\s*mm)?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
NAKED_MM_RE = re.compile(
    r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?\s*mm(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_STANDARD_FAMILY_PATTERN = r"(?:GB|NB|SH|HG|DL|TB)(?:\s*/?\s*T)?|ASTM|ASME|API|EN|DIN|ISO|MSS"
STANDARD_FAMILY_RE = re.compile(_STANDARD_FAMILY_PATTERN, re.IGNORECASE)
STANDARD_PREFIX_TOKEN_RE = re.compile(
    r"(?:GB|NB|SH|HG|DL|TB|ASTM|ASME|API|EN|DIN|ISO|MSS|ANSI|JIS|AB|B)",
    re.IGNORECASE,
)
STRICT_STANDARD_BODY_TAIL_RE = re.compile(
    rf"(?:{_STANDARD_FAMILY_PATTERN}|(?:ASME\s*-?\s*)?B|ANSI|JIS|AB)"
    r"[\s/\-_.]*[A-Z]?\d+(?:\.\d+)?[\s/\-_.]*$",
    re.IGNORECASE,
)
MALFORMED_STANDARD_BODY_TAIL_RE = re.compile(
    r"(?:GB|NB|SH|HG|DL|TB)(?:\s*/?\s*T)?[\s/\-_.]*[A-Z][A-Z0-9]*(?:\.[A-Z0-9]*)?[\s/\-_.]*$",
    re.IGNORECASE,
)


class AnchorMissingDetector:
    """Detect naked numeric values that lack clear field anchors."""

    TAG_NAME = "anchor_missing"

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config" / "common_code_mapping.yaml"
        )
        self._load_config()

    def analyze(self, text: str) -> DifficultyFeature:
        hits: list[GlueHit] = []
        covered_spans: list[tuple[int, int]] = []

        for match in NAKED_SPEC_RE.finditer(text):
            if self._has_explicit_anchor(text, match.start(), match.end()):
                continue
            standard_context = self._classify_standard_body_context(text, match.start())
            if standard_context in {"strict", "malformed"}:
                continue
            hits.append(
                GlueHit(
                    tag=self.TAG_NAME,
                    code_group="naked_spec",
                    code=match.group(0),
                    token=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    note=f"裸规格 {match.group(0)} 左右缺少明确字段锚点，可能需要靠语义判断其含义",
                )
            )
            covered_spans.append((match.start(), match.end()))

        for match in NAKED_MM_RE.finditer(text):
            value = match.group(0)
            if any(start <= match.start() and match.end() <= end for start, end in covered_spans):
                continue
            if self._has_explicit_anchor(text, match.start(), match.end()):
                continue
            if self._has_length_context_on_right(text, match.end()):
                continue
            standard_context = self._classify_standard_body_context(text, match.start())
            if standard_context in {"strict", "malformed"}:
                continue
            hits.append(
                GlueHit(
                    tag=self.TAG_NAME,
                    code_group="naked_metric_value",
                    code=value,
                    token=value,
                    start=match.start(),
                    end=match.end(),
                    note=f"公制值 {value} 左右缺少明确字段锚点，可能需要靠语义判断其含义",
                )
            )
            covered_spans.append((match.start(), match.end()))

        for match in NAKED_NUMBER_RE.finditer(text):
            value = match.group(0)
            if not self._is_candidate_number(value):
                continue
            if value in self.common_numeric_material_codes:
                continue
            if any(start <= match.start() and match.end() <= end for start, end in covered_spans):
                continue
            if self._is_percentage_value(text, match.start(), match.end()):
                continue
            if self._has_explicit_anchor(text, match.start(), match.end()):
                continue
            standard_context = self._classify_standard_body_context(text, match.start())
            if standard_context in {"strict", "malformed"}:
                continue
            if self._looks_like_standard_suffix(text, match.start(), match.end()):
                continue

            hits.append(
                GlueHit(
                    tag=self.TAG_NAME,
                    code_group="naked_numbers",
                    code=value,
                    token=value,
                    start=match.start(),
                    end=match.end(),
                    note=f"数字 {value} 左右缺少明确字段锚点，可能需要靠语义判断其含义",
                )
            )

        reason = ""
        if hits:
            groups = {hit.code_group for hit in hits}
            if groups == {"naked_spec"}:
                reason = "存在缺少字段锚点的裸规格"
            elif groups == {"naked_metric_value"}:
                reason = "存在缺少字段锚点的裸公制值"
            elif "naked_spec" in groups:
                if "naked_metric_value" in groups:
                    reason = "存在缺少字段锚点的裸规格、裸公制值，以及缺少字段锚点的裸数字"
                else:
                    reason = "存在缺少字段锚点的裸规格，以及缺少字段锚点的裸数字"
            elif "naked_metric_value" in groups:
                reason = "存在缺少字段锚点的裸公制值，以及缺少字段锚点的裸数字"
            else:
                reason = "存在缺少字段锚点的裸数字"

        return DifficultyFeature(name=self.TAG_NAME, matched=bool(hits), reason=reason, hits=hits)

    def _load_config(self) -> None:
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 这里只排除“纯数字型常见材质”，例如 20 / 304 / 316 / 2205。
        # 304L、316L 这类本来就不会被裸数字规则命中，不需要额外处理。
        self.common_numeric_material_codes = {
            text
            for raw in data.get("common_material_codes", [])
            if (text := str(raw).strip()) and re.fullmatch(r"\d+(?:\.\d+)?", text)
        }

    @staticmethod
    def _is_candidate_number(value: str) -> bool:
        # First pass is intentionally conservative:
        # - decimals are suspicious enough
        # - integers only when length >= 3
        return "." in value or len(value) >= 3

    @staticmethod
    def _is_percentage_value(text: str, start: int, end: int) -> bool:
        # 像 100%RT、2.5% 这类“数字+百分号”属于百分比表达，不按裸数字处理。
        right = text[end : min(len(text), end + 2)]
        return right.lstrip().startswith("%")

    @staticmethod
    def _has_length_context_on_right(text: str, end: int) -> bool:
        right = text[end : min(len(text), end + 16)]
        return bool(re.match(r"(?i)^\s*(?:length|len|long|长度)\b", right))


    @staticmethod
    def _has_explicit_anchor(text: str, start: int, end: int) -> bool:
        left = text[max(0, start - 12) : start].upper()
        right = text[end : min(len(text), end + 4)].upper()
        left_raw = text[max(0, start - 16) : start]

        left_compact = re.sub(r"[\s:=\-_/.]", "", left)

        if any(left_compact.endswith(anchor) for anchor in ("DN", "OD", "NPS", "PN", "CL", "CLASS", "THK", "SCH")):
            return True
        if any(left.rstrip().endswith(anchor) for anchor in ("φ", "Φ")):
            return True
        if any(left.endswith(anchor) for anchor in ("STD ", "XS ", "XXS ")):
            return True
        if left.rstrip().endswith(("L=", "S=", "R=")):
            return True
        # 通用字母锚点：如 E=0.85、C=1.0、K=0.7。
        # 这里不区分具体语义，只要是局部“字母=数字”结构，就不应再当作裸数字。
        if re.search(r"(?i)(?:^|[^A-Za-z0-9])(?:[A-Za-z]{1,6})\s*=\s*$", left_raw):
            return True
        if right.startswith('"'):
            return True
        # 组合尺寸里的锚点传递：DN350*150、DN80/40、OD48x2.8 等，
        # 第二个数字虽然本身没有再次写出锚点，但前一个数字已被同一组合里的
        # 显式锚点约束，此时不应再当作“裸数字缺锚点”。
        if re.search(r"(?i)(?:DN|OD|NPS|PN|CL|CLASS)\s*\d+(?:\.\d+)?\s*[*xX×/]\s*$", left_raw):
            return True
        if re.search(r"(?:φ|Φ)\s*\d+(?:\.\d+)?\s*[*xX×/]\s*$", left_raw):
            return True
        return False

    @staticmethod
    def _looks_like_standard_suffix(text: str, start: int, end: int) -> bool:
        prev_non_space = ""
        i = start - 1
        while i >= 0:
            if not text[i].isspace():
                prev_non_space = text[i]
                break
            i -= 1

        if prev_non_space not in {"-", "("}:
            return False

        left_window = text[max(0, start - 24) : start]
        return bool(STANDARD_FAMILY_RE.search(left_window))

    @staticmethod
    def _classify_standard_body_context(text: str, start: int) -> str | None:
        left_window = text[max(0, start - 24) : start]
        right_window = text[start : min(len(text), start + 16)]

        # 先认严格标准族前缀，如 HG/T、SH/T、ASTM、ASME。
        matches = list(STANDARD_FAMILY_RE.finditer(left_window))
        if matches:
            last = matches[-1]
            family_text = last.group(0)
            tail = left_window[last.end() :]
            if re.fullmatch(r"[\s\-_/\.]*", tail):
                # HG/120592、GB/1234 这类“前缀后直接斜杠接主体”属于前缀异常，
                # 不能因为斜杠本身是分隔符就当成严格标准号。
                if "/" in tail and "T" not in family_text.upper():
                    return "malformed"
                return "strict"
            # 统一支持带 T 和不带 T 的标准族前缀：
            # - GB/T13401、HGT20553
            # - GB6479、SH3408
            # - ASME B16.10、ASME B16.25
            #
            # 只要左侧尾巴已经形成“标准族前缀 + 合法主体 + 分隔”，
            # 当前数字就按严格标准号上下文处理，不再误判为裸数字。
            if STRICT_STANDARD_BODY_TAIL_RE.search(left_window):
                return "strict"
            # 如果已经命中了严格标准族前缀，但尾巴长得像小数主体左半段且混入了异常字符，
            # 则按“标准前缀异常”处理，而不是落回普通裸数字。
            if MALFORMED_STANDARD_BODY_TAIL_RE.search(left_window):
                return "malformed"

        # 再认“疑似标准号前缀异常”形态：
        # - 数字左侧是更像标准族的短前缀，并且带有 /、-、.、空格等分隔
        # - 或左侧是更像标准族的紧凑前缀（如 HG1 20592）
        # - 或数字右侧还带常见标准尾缀，如 -09B、-2017、(II)
        #
        # 这里不再接受任意短大写 token，避免把 BE 17.48、BW 20 这类普通业务 token
        # 误判成“标准号前缀异常”。
        left_shape_sep = re.search(r"(?:^|[^A-Za-z0-9])([A-Z]{1,6})[\s/\-_.]+$", left_window)
        left_shape_compact = re.search(r"(?:^|[^A-Za-z0-9])([A-Z]{1,6}\d{1,2})\s+$", left_window)

        sep_prefix = left_shape_sep.group(1) if left_shape_sep else ""
        compact_prefix = left_shape_compact.group(1) if left_shape_compact else ""
        compact_prefix_alpha = re.match(r"[A-Z]+", compact_prefix)
        compact_prefix_alpha = compact_prefix_alpha.group(0) if compact_prefix_alpha else ""

        if (
            (sep_prefix and STANDARD_PREFIX_TOKEN_RE.fullmatch(sep_prefix))
            or (compact_prefix_alpha and STANDARD_PREFIX_TOKEN_RE.fullmatch(compact_prefix_alpha))
        ):
            return "malformed"

        if re.match(r"^(?:\s*[-/_.]\s*[A-Za-z0-9]{1,6}|\s*\([A-Za-z0-9ⅠⅡⅢⅣ]{1,8}\))", right_window):
            if sep_prefix or compact_prefix:
                return "malformed"
            if re.search(r"(?:^|[^A-Za-z0-9])([A-Z]{1,6})[\s/\-_.]*$", left_window):
                return "malformed"

        return None
