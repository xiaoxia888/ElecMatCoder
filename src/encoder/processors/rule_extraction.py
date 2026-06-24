from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List
from typing import Optional
import yaml

from .pressure_processor import PressureProcessor, RulePressureExtraction
from .size_processor import RuleSizeExtraction, SizeProcessor
from .thickness_processor import RuleThicknessExtraction, ThicknessProcessor
from .weak_fallback_processor import WeakFallbackProcessor


_OD_NUMERIC_PAIR_RE = re.compile(
    r'(?i)(?:\bOD|[φΦФф]|\bD)\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)(\s*MM)?(?!\s*[xX×*]\s*\d)'
)


@dataclass
class RuleExtractionResult:
    size: RuleSizeExtraction
    thickness: RuleThicknessExtraction
    pressure: RulePressureExtraction


@dataclass
class OdPairDecision:
    pair_span: tuple[int, int]
    second_value_span: tuple[int, int]
    action: str  # keep_as_thickness / treat_as_size_pair / ambiguous_clear
    second_value: str


_UNRESOLVED_SPEC_PATTERNS = (
    # 裸组合数字：30x25 / 24x23mm / -2x2
    re.compile(r'(?<![A-Za-z0-9./])\d{1,3}(?:\.\d+)?\s*[xX×*]\s*\d{1,3}(?:\.\d+)?\s*(?:MM\b|毫米\b)?', re.IGNORECASE),
    # 裸斜杠双值：23/23
    re.compile(r'(?<![A-Za-z0-9./])\d{1,3}(?:\.\d+)?\s*/\s*\d{1,3}(?:\.\d+)?(?![A-Za-z0-9./])'),
    # 裸 mm：24mm
    re.compile(r'(?<![A-Za-z0-9./])\d+(?:\.\d+)?\s*(?:MM\b|毫米\b)', re.IGNORECASE),
    # 裸整数：位数不限，但必须处在干净分隔边界上
    re.compile(r'(?:(?<=^)|(?<=[,，;；、\s()\-]))\d+(?=(?:$|[,，;；、\s()\-]))'),
)

_COMMON_DN_VALUES: Optional[set[int]] = None


def _get_common_dn_values() -> set[int]:
    global _COMMON_DN_VALUES
    if _COMMON_DN_VALUES is not None:
        return _COMMON_DN_VALUES
    config_path = Path(__file__).parent.parent / "config" / "encoder_config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        size_config = config.get("size_processing", {}) or {}
        _COMMON_DN_VALUES = {int(v) for v in size_config.get("common_dn_values", [])}
    except Exception:
        _COMMON_DN_VALUES = set()
    return _COMMON_DN_VALUES


def _is_valid_single_residual_integer(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or not raw.isdigit():
        return False
    if raw == "20":
        return False
    try:
        return int(raw) in _get_common_dn_values()
    except Exception:
        return False


def _is_ignored_residual_slash_pair(text: str) -> bool:
    raw = re.sub(r"\s+", "", str(text or "").strip())
    return raw == "20/20"


_ANGLE_CN_SUFFIX_RE = re.compile(r"^\s*(?:°|度)")
_ANGLE_EN_SUFFIX_RE = re.compile(r"^\s*DEG(?:REE)?S?\b", re.IGNORECASE)
_ANGLE_PREFIX_RE = re.compile(r"(?:角度|度数)\s*$", re.IGNORECASE)
_LEADING_NOISE_RE = re.compile(r"^[\s,，;；、()\[\]{}:：\-_/]+$")


def _is_angle_residual_context(source: str, span: tuple[int, int]) -> bool:
    left = source[max(0, span[0] - 8):span[0]]
    right = source[span[1]:min(len(source), span[1] + 12)]
    return bool(
        _ANGLE_CN_SUFFIX_RE.match(right)
        or _ANGLE_EN_SUFFIX_RE.match(right)
        or _ANGLE_PREFIX_RE.search(left)
    )


def _is_leading_angle_integer(source: str, span: tuple[int, int], text: str) -> bool:
    raw = str(text or "").strip()
    if raw not in {"45", "90"}:
        return False
    left = str(source or "")[:span[0]]
    if not left:
        return True
    return bool(_LEADING_NOISE_RE.fullmatch(left))


def _should_ignore_unresolved_match(
    pattern_index: int,
    match_text: str,
    source: str = "",
    span: Optional[tuple[int, int]] = None,
) -> bool:
    text = str(match_text or "").strip()
    # 20/20 在当前业务里按材质表达处理，不参与规格残留清空。
    if pattern_index == 1 and _is_ignored_residual_slash_pair(text):
        return True
    # 单裸整数只在“像公称直径”的情况下参与残留校验：
    # 1. 纯整数
    # 2. 在 common_dn_values 中
    # 3. 排除常见裸材质 20
    if pattern_index == 3:
        if span is not None and source and _is_angle_residual_context(source, span):
            return True
        if span is not None and source and _is_leading_angle_integer(source, span, text):
            return True
        return not _is_valid_single_residual_integer(text)
    return False


def _has_any_size(size_result: RuleSizeExtraction) -> bool:
    return bool(size_result.dn or size_result.od or size_result.inch or size_result.length)


def _has_any_thickness(thickness_result: RuleThicknessExtraction) -> bool:
    return bool(thickness_result.mm or thickness_result.schedule)


def _has_any_pressure(pressure_result: RulePressureExtraction) -> bool:
    return bool(pressure_result.pressure_code)


def _finalize_pressure_result(
    pressure_result: RulePressureExtraction,
    *,
    clear_conflicted_multi_pressure: bool,
) -> RulePressureExtraction:
    if not clear_conflicted_multi_pressure:
        return pressure_result
    if not getattr(pressure_result, "conflicted", False):
        return pressure_result
    return RulePressureExtraction(
        values=[],
        pressure_code="",
        matched_texts=list(getattr(pressure_result, "matched_texts", []) or []),
        matched_spans=list(getattr(pressure_result, "matched_spans", []) or []),
        consumed_spans=[],
        ordered_items=[],
        conflicted=True,
        conflict_reason=str(getattr(pressure_result, "conflict_reason", "") or "multiple_pressure_without_slash"),
        cleared=True,
        clear_reason=str(getattr(pressure_result, "conflict_reason", "") or "multiple_pressure_without_slash"),
    )


def _has_unconsumed_residual_spec(
    text: str,
    size_result: RuleSizeExtraction,
    thickness_result: RuleThicknessExtraction,
    pressure_result: RulePressureExtraction,
) -> bool:
    consumed_spans = list(getattr(size_result, "consumed_spans", []) or size_result.matched_spans)
    consumed_spans += list(getattr(thickness_result, "consumed_spans", []) or thickness_result.matched_spans)
    consumed_spans += list(getattr(pressure_result, "consumed_spans", []) or pressure_result.matched_spans)
    source = str(text or "")
    for pattern_index, pattern in enumerate(_UNRESOLVED_SPEC_PATTERNS):
        for m in pattern.finditer(source):
            span = m.span(0)
            if any(start < span[1] and span[0] < end for start, end in consumed_spans):
                continue
            if _should_ignore_unresolved_match(pattern_index, m.group(0), source, span):
                continue
            return True
    return False


def _should_clear_due_to_residual_spec(
    text: str,
    size_result: RuleSizeExtraction,
    thickness_result: RuleThicknessExtraction,
    pressure_result: RulePressureExtraction,
) -> bool:
    if not _has_any_size(size_result):
        return False
    if not (_has_any_thickness(thickness_result) or _has_any_pressure(pressure_result)):
        return False
    return _has_unconsumed_residual_spec(str(text or ""), size_result, thickness_result, pressure_result)


def _normalize_int_text(raw: str) -> str:
    try:
        return str(int(float(str(raw).strip())))
    except (TypeError, ValueError):
        return str(raw).strip()


def _match_value_to_dn(raw_value: str, dn_value: str, size_processor: SizeProcessor) -> bool:
    raw_text = str(raw_value or "").strip()
    dn_text = _normalize_int_text(dn_value)
    if not raw_text or not dn_text:
        return False

    if raw_text.isdigit() and _normalize_int_text(raw_text) == dn_text:
        return True

    try:
        converted_dn = size_processor._od_to_dn(float(raw_text))
    except (TypeError, ValueError):
        converted_dn = None

    return converted_dn is not None and str(int(converted_dn)) == dn_text


def _classify_od_pair_decisions(text: str, size_result: RuleSizeExtraction, size_processor: SizeProcessor) -> List[OdPairDecision]:
    source = str(text or "")
    dn_values = [str(v) for v in (getattr(size_result, "dn", []) or [])]
    decisions: List[OdPairDecision] = []

    has_dn_pair = len(dn_values) >= 2
    has_different_dn_pair = has_dn_pair and _normalize_int_text(dn_values[0]) != _normalize_int_text(dn_values[1])

    for match in _OD_NUMERIC_PAIR_RE.finditer(source):
        first_value = str(match.group(1) or "").strip()
        second_value = str(match.group(2) or "").strip()
        explicit_mm = bool(match.group(3) and str(match.group(3)).strip())
        is_decimal = "." in second_value

        if explicit_mm:
            action = "keep_as_thickness"
        elif has_different_dn_pair:
            first_matches = _match_value_to_dn(first_value, dn_values[0], size_processor)
            second_matches = _match_value_to_dn(second_value, dn_values[1], size_processor)
            action = "treat_as_size_pair" if (first_matches and second_matches) else "keep_as_thickness"
        elif is_decimal:
            action = "keep_as_thickness"
        else:
            action = "ambiguous_clear"

        decisions.append(
            OdPairDecision(
                pair_span=match.span(),
                second_value_span=match.span(2),
                action=action,
                second_value=second_value,
            )
        )

    return decisions


def _augment_size_result_with_size_pair_echo(size_result: RuleSizeExtraction, text: str, decisions: List[OdPairDecision]) -> RuleSizeExtraction:
    od_values = list(size_result.od)
    matched_spans = list(size_result.matched_spans)
    consumed_spans = list(getattr(size_result, "consumed_spans", []) or [])
    ordered_items = list(size_result.ordered_items)

    for decision in decisions:
        if decision.action != "treat_as_size_pair":
            continue
        second_od = str(decision.second_value).strip()
        if second_od and second_od not in od_values:
            od_values.append(second_od)
        if decision.pair_span not in matched_spans:
            matched_spans.append(decision.pair_span)
        if decision.second_value_span not in consumed_spans:
            consumed_spans.append(decision.second_value_span)
        candidate = {"type": "OD", "value": second_od}
        if second_od and candidate not in ordered_items:
            ordered_items.append(candidate)

    return RuleSizeExtraction(
        dn=list(size_result.dn),
        od=od_values,
        inch=list(size_result.inch),
        length=list(size_result.length),
        size_code=size_result.size_code,
        matched_texts=list(size_result.matched_texts),
        matched_spans=matched_spans,
        consumed_spans=consumed_spans,
        ordered_items=ordered_items,
    )


def build_structured_size_field(result: RuleSizeExtraction, original_text: str = "") -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "DN": list(result.dn),
        "OD": list(result.od),
        "INCH": list(result.inch),
        "LENGTH": list(result.length),
    }
    items: List[Dict[str, str]] = []
    if result.ordered_items:
        items = [{"type": str(item["type"]), "value": str(item["value"])} for item in result.ordered_items]
    else:
        for value in result.dn:
            items.append({"type": "DN", "value": str(value)})
        for value in result.od:
            items.append({"type": "OD", "value": str(value)})
        for value in result.inch:
            items.append({"type": "INCH", "value": str(value)})
        for value in result.length:
            items.append({"type": "LENGTH", "value": str(value)})
    if items:
        field["_ITEMS"] = items
    return field


def build_structured_thickness_field(result: RuleThicknessExtraction, original_text: str = "") -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "MM": [v.replace("MM", "") for v in result.mm],
        "SCHEDULE": list(result.schedule),
        "INCH": [],
        "SERIES": [],
        "BWG": [],
    }
    items: List[Dict[str, str]] = []
    if result.ordered_items:
        items = [{"type": str(item["type"]), "value": str(item["value"])} for item in result.ordered_items]
    else:
        for value in result.mm:
            items.append({"type": "MM", "value": value.replace("MM", "")})
        for value in result.schedule:
            items.append({"type": "SCHEDULE", "value": str(value)})
    if items:
        field["_ITEMS"] = items
    return field


def build_structured_rule_entities(result: RuleExtractionResult, original_text: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "SIZE": build_structured_size_field(result.size, original_text=original_text),
        "THICKNESS": build_structured_thickness_field(result.thickness, original_text=original_text),
        "PRESSURE": result.pressure.pressure_code,
    }
    pressure_flags: Dict[str, Any] = {}
    if getattr(result.pressure, "conflicted", False):
        pressure_flags["conflicted"] = True
        pressure_flags["conflict_reason"] = str(getattr(result.pressure, "conflict_reason", "") or "")
    if getattr(result.pressure, "cleared", False):
        pressure_flags["cleared"] = True
        pressure_flags["clear_reason"] = str(getattr(result.pressure, "clear_reason", "") or "")
    if pressure_flags:
        payload["_rule_flags"] = {
            "PRESSURE": pressure_flags,
        }
    return payload


def extract_size_and_thickness_by_rules(
    text: str,
    size_processor: Optional[SizeProcessor] = None,
    thickness_processor: Optional[ThicknessProcessor] = None,
    pressure_processor: Optional[PressureProcessor] = None,
    weak_fallback_processor: Optional[WeakFallbackProcessor] = None,
    *,
    apply_residual_guard: bool = True,
) -> RuleExtractionResult:
    """
    规则抽取统一入口。

    顺序固定为：
    1. 先处理尺寸
    2. 再处理壁厚

    当前壁厚规则优先依赖自身的显式锚点和稳定结构模板，
    这里先保留顺序化调用，不额外做文本替换，避免破坏原文结构。
    """
    size_processor = size_processor or SizeProcessor()
    thickness_processor = thickness_processor or ThicknessProcessor(enable_rule_layered=False)
    pressure_processor = pressure_processor or PressureProcessor()
    weak_fallback_processor = weak_fallback_processor or WeakFallbackProcessor(
        size_processor=size_processor,
        thickness_processor=thickness_processor,
    )

    size_result = size_processor.extract_by_rules(text)
    od_pair_decisions = _classify_od_pair_decisions(str(text or ""), size_result, size_processor)
    size_result = _augment_size_result_with_size_pair_echo(size_result, str(text or ""), od_pair_decisions)
    thickness_blocked_spans = list(getattr(size_result, "consumed_spans", []) or size_result.matched_spans)
    for decision in od_pair_decisions:
        if decision.action == "treat_as_size_pair" and decision.second_value_span not in thickness_blocked_spans:
            thickness_blocked_spans.append(decision.second_value_span)
        elif decision.action == "keep_as_thickness" and decision.second_value_span in thickness_blocked_spans:
            thickness_blocked_spans.remove(decision.second_value_span)
    thickness_result = thickness_processor.extract_by_rules(
        text,
        size_context=size_result,
        blocked_spans=thickness_blocked_spans,
    )
    size_result = weak_fallback_processor.apply_size_tail_dn_fallback(
        text,
        size_result,
        blocked_spans=list(getattr(thickness_result, "consumed_spans", []) or thickness_result.matched_spans),
    )
    thickness_blocked_spans = list(getattr(size_result, "consumed_spans", []) or size_result.matched_spans)
    for decision in od_pair_decisions:
        if decision.action == "treat_as_size_pair" and decision.second_value_span not in thickness_blocked_spans:
            thickness_blocked_spans.append(decision.second_value_span)
        elif decision.action == "keep_as_thickness" and decision.second_value_span in thickness_blocked_spans:
            thickness_blocked_spans.remove(decision.second_value_span)
    thickness_result = weak_fallback_processor.apply_thickness_decimal_mm_fallback(
        text,
        thickness_result,
        blocked_spans=thickness_blocked_spans,
    )
    pressure_blocked_spans = thickness_blocked_spans + list(getattr(thickness_result, "consumed_spans", []) or thickness_result.matched_spans)
    pressure_result = pressure_processor.extract_by_rules(text, blocked_spans=pressure_blocked_spans)
    if apply_residual_guard:
        pressure_result = _finalize_pressure_result(
            pressure_result,
            clear_conflicted_multi_pressure=True,
        )

    # 主体裸复合规格存在，但规则侧未拿到任何显式/兜底尺寸时，
    # 说明本句整体应让给大模型，壁厚规则结果也一并作废，避免半规则半模型。
    if (
        apply_residual_guard
        and
        size_processor._has_unconsumed_bare_complex_spec(str(text or ""))
        and not size_result.dn
        and not size_result.od
        and not size_result.inch
    ):
        thickness_result = RuleThicknessExtraction(
            schedule=[],
            mm=[],
            thickness_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
        )
    if apply_residual_guard and _should_clear_due_to_residual_spec(str(text or ""), size_result, thickness_result, pressure_result):
        size_result = RuleSizeExtraction(
            dn=[],
            od=[],
            inch=[],
            length=[],
            size_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
        )
        thickness_result = RuleThicknessExtraction(
            schedule=[],
            mm=[],
            thickness_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
        )
        pressure_result = RulePressureExtraction(
            values=[],
            pressure_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
            cleared=True,
            clear_reason="residual_spec_guard",
        )

    if apply_residual_guard and any(decision.action == "ambiguous_clear" for decision in od_pair_decisions):
        size_result = RuleSizeExtraction(
            dn=[],
            od=[],
            inch=[],
            length=[],
            size_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
        )
        thickness_result = RuleThicknessExtraction(
            schedule=[],
            mm=[],
            thickness_code="",
            matched_texts=[],
            matched_spans=[],
            consumed_spans=[],
            ordered_items=[],
        )

    return RuleExtractionResult(size=size_result, thickness=thickness_result, pressure=pressure_result)
