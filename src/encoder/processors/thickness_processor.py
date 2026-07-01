"""
壁厚处理器
负责壁厚/壁厚规格的格式化和编码转换

处理规则：
1. SCH 格式：SCH80 → S80, SCH40S → S40S
2. S- 格式：S-80 → S80, S-10S → S10S  
3. 特殊值：SCHXXS → XXS, SCHSTD → STD
4. mm 格式：8.1mm → 8.1MM, T=3.0mm → 3MM
5. 异径处理：SCH80XSCH80 → S80 (相同合并), SCH80XSCH40 → S80XS40
6. 分隔符统一：x, *, ×, , → X；但结构化多层壁厚值中的组内 `/` 可保留
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional


class ThicknessProcessor:
    """壁厚处理器"""
    STRUCTURED_SUBTYPE_ORDER = ("MM", "INCH", "SCHEDULE", "SERIES", "BWG")
    # 弱数值 schedule 白名单。
    # 仅用于最弱的 `S数字` / `S-数字` / `数字S` 规则，避免把标准号残片误识别成壁厚。
    WEAK_SCHEDULE_BASE_VALUES = ("5", "10", "20", "30", "40", "60", "80", "100", "120", "140", "160")
    # 只有这些完整 token 存在合法的 `...S` 形式。
    WEAK_SCHEDULE_SUFFIX_S_TOKENS = ("5S", "10S", "20S", "40S", "30S", "60S", "80S", "120S", "160S")
    
    # 分隔符正则（用于异径规格）
    SEPARATOR_PATTERN = re.compile(r'\s*[xX*×/,]\s*')
    
    # SCH 格式正则（注意：XXS 要放在 XS 前面，避免 XS 先匹配）
    # 支持 SCH40S, SCH.40S, SCH 40S 等格式
    SCH_PATTERN = re.compile(r'SCH[.\s]*(XXS|XS|STD|\d+S?)', re.IGNORECASE)
    
    # S- 格式正则
    S_DASH_PATTERN = re.compile(r'S-(XXS|XS|STD|\d+S?)', re.IGNORECASE)
    S_DASH_MM_PATTERN = re.compile(r'^S-(\d+\.\d+)\s*(?:MM|毫米)?$', re.IGNORECASE)
    
    # mm 格式正则（数字+mm）
    MM_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE)
    
    # T=/THK= 前缀正则
    PREFIX_PATTERN = re.compile(r'^(?:T|THK)\s*=\s*', re.IGNORECASE)
    
    # (L)/(S)/L/S 后缀正则（大小端标识）
    SUFFIX_PATTERN = re.compile(r'\s*\([LS]\)|\s+[LS](?=\s|$|[xX*×/,])', re.IGNORECASE)
    
    # 特殊值（不带数字的）
    SPECIAL_VALUES = {'XXS', 'XS', 'STD'}

    # 来源于现有映射表的少量下划线脏样本别名。
    # 这些写法语义不一致，逐条归一比套一条通用规则更安全。
    UNDERSCORE_ALIASES = {
        'SCH20_S': 'SCH20S',
        'SCH40S_S': 'SCH40S/S',
        'SCH40_S': 'SCH40/S',
        'SCH40_STD': 'SCH40/STD',
        'STD_STD': 'STD/STD',
        'STD_SCH2': 'STD/SCH2',
        'SCH20_STD': 'STD/SCH20',
        'XS_XS': 'XS/XS',
    }

    def __init__(self, enable_rule_layered: bool = False):
        # 按当前要求：规则路径暂不处理分层壁厚
        self.enable_rule_layered = enable_rule_layered
        self.config_path = Path(__file__).parent.parent / "config" / "encoder_config.yaml"
        self._common_dn_values = self._load_common_dn_values()

    def _load_common_dn_values(self) -> List[int]:
        try:
            if not self.config_path.exists():
                return []
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            values = config.get('size_processing', {}).get('common_dn_values', [])
            return sorted({int(v) for v in values if str(v).strip()}, key=lambda x: len(str(x)), reverse=True)
        except Exception:
            return []

    @classmethod
    @lru_cache(maxsize=1)
    def _weak_schedule_patterns(cls) -> Tuple[str, str, str, str]:
        """
        生成最弱 schedule 规则使用的公共模式。
        返回：
        1. prefixed: S10 / S10S / S120 ...
        2. s_dash: S-10 / S-10S / S-120 ...
        3. suffix: 10S / 40S ...
        4. combined: 三者 + XS/XXS/STD 的合并模式
        """
        base_group = "|".join(re.escape(v) for v in cls.WEAK_SCHEDULE_BASE_VALUES)
        suffix_tokens = tuple(token[:-1] for token in cls.WEAK_SCHEDULE_SUFFIX_S_TOKENS)
        suffix_base_group = "|".join(re.escape(v) for v in suffix_tokens)
        suffix_token_group = "|".join(re.escape(v) for v in cls.WEAK_SCHEDULE_SUFFIX_S_TOKENS)
        prefixed = rf'(?:S(?:{suffix_base_group})S|S(?:{base_group}))'
        s_dash = rf'(?:S-(?:{suffix_base_group})S|S-(?:{base_group}))'
        suffix = rf'(?:{suffix_token_group})'
        combined = rf'(?:{prefixed}|{s_dash}|{suffix}|XS|XXS|STD)'
        return prefixed, s_dash, suffix, combined

    def process(self, value: Any, original_text: str = "") -> str:
        """
        处理壁厚值，返回标准编码
        
        Args:
            value: 原始壁厚描述，如 "SCH40S", "S-10S x S-40S", "THK=6.3mmX6.3mm"
            
        Returns:
            标准编码，如 "S40S", "S10SXS40S", "6.3MM"
        """
        if not value:
            return ""

        if isinstance(value, dict):
            return self._process_structured(value, original_text=original_text)

        if isinstance(value, list):
            merged_parts: List[str] = []
            for item in value:
                normalized = self.process(item, original_text=original_text)
                if not normalized:
                    continue
                for part in normalized.split('X'):
                    part = part.strip()
                    if part and part not in merged_parts:
                        merged_parts.append(part)
            return 'X'.join(merged_parts)
        
        value = str(value).strip()
        
        # 1. 局部预处理（仅清理壁厚 token 前后缀，不做全文删空格）
        value = self._preprocess(value)

        # 2. 分割异径规格
        parts = self._split_parts(value)
        
        # 3. 转换每个部分
        converted = [self._convert_single(p) for p in parts]
        converted = [c for c in converted if c]  # 过滤空值
        
        if not converted:
            return value.upper()  # 无法识别时返回大写原值
        
        # 4. 合并结果
        return self._merge_parts(converted)

    def _process_structured(self, value: Dict[str, Any], original_text: str = "") -> str:
        ordered = self._process_structured_items(value.get("_ITEMS"))
        if ordered:
            return ordered

        parts: List[str] = []

        ordered_keys = [k for k in self.STRUCTURED_SUBTYPE_ORDER if k in value]
        extra_keys = [k for k in value.keys() if k not in ordered_keys]

        for subtype in ordered_keys + extra_keys:
            raw_items = value.get(subtype)
            if raw_items in (None, "", []):
                continue
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            for item in raw_items:
                normalized = self._normalize_structured_part(str(subtype).upper(), item)
                if normalized and normalized not in parts:
                    parts.append(normalized)
        return 'X'.join(parts)

    def extract_by_rules(
        self,
        text: str,
        blocked_spans: Optional[List[Tuple[int, int]]] = None,
        size_context: Optional[Any] = None,
    ) -> "RuleThicknessExtraction":
        """
        基于显式锚点和高频稳定结构做壁厚抽取。

        规则：
        1. 只认显式 THK/T/壁厚/S=、SCH/STD/XS/XXS、明确 mm 结构
        2. `THK=a mm X b mm` 直接按普通双壁厚处理
        3. 不在规则路径处理分层壁厚
        4. 不从 `DNx整数` 推断壁厚
        """
        source = str(text or "")
        normalized = source.replace('”', '"').replace('“', '"').replace('″', '"')

        # 0) 明显脏小数串直接交给大模型：
        # 例如 6.31.5D / 12.70.31mm
        if self._has_malformed_decimal_chain(normalized):
            return RuleThicknessExtraction(
                schedule=[],
                mm=[],
                thickness_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )

        # 0) 主壁厚(次壁厚) 结构直接交给大模型：
        # 例如 114x4.0(3.0)、THK=8.0(6.0)mm
        if self._has_parenthesized_thickness_variant(normalized):
            return RuleThicknessExtraction(
                schedule=[],
                mm=[],
                thickness_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )

        # 0) 复杂复合规格先基于原始归一化文本判定，避免后续切分把复杂结构改写掉。
        if self._has_complex_composite_size(normalized) and not self._has_phi_dual_od_dual_thk_structure(normalized):
            return RuleThicknessExtraction(
                schedule=[],
                mm=[],
                thickness_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )

        blocked_spans = blocked_spans or []
        explicit = self._extract_explicit_thickness_rules(normalized, blocked_spans)
        if explicit.get("invalid"):
            return RuleThicknessExtraction(
                schedule=[],
                mm=[],
                thickness_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )
        schedule_parts = explicit["schedule"]
        mm_parts = explicit["mm"]
        ordered_items = explicit["ordered_items"]
        matched_texts = explicit["matched_texts"]
        matched_spans = explicit["matched_spans"]

        self._apply_weak_thickness_fallback(
            normalized=normalized,
            blocked_spans=blocked_spans,
            schedule_parts=schedule_parts,
            mm_parts=mm_parts,
            ordered_items=ordered_items,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
        )
        if self._has_invalid_thickness_items(schedule_parts, mm_parts):
            return RuleThicknessExtraction(
                schedule=[],
                mm=[],
                thickness_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )

        if self._has_size_context(size_context):
            self._apply_size_context_glued_schedule_rules(
                normalized=normalized,
                blocked_spans=blocked_spans,
                size_context=size_context,
                schedule_parts=schedule_parts,
                ordered_items=ordered_items,
                matched_texts=matched_texts,
                matched_spans=matched_spans,
            )
            if self._has_invalid_thickness_items(schedule_parts, mm_parts):
                return RuleThicknessExtraction(
                    schedule=[],
                    mm=[],
                    thickness_code="",
                    matched_texts=[],
                    matched_spans=[],
                    ordered_items=[],
                )
            self._apply_size_context_thickness_rules(
                normalized=normalized,
                blocked_spans=blocked_spans,
                size_context=size_context,
                schedule_parts=schedule_parts,
                mm_parts=mm_parts,
                ordered_items=ordered_items,
                matched_texts=matched_texts,
                matched_spans=matched_spans,
            )
            if self._has_invalid_thickness_items(schedule_parts, mm_parts):
                return RuleThicknessExtraction(
                    schedule=[],
                    mm=[],
                    thickness_code="",
                    matched_texts=[],
                    matched_spans=[],
                    ordered_items=[],
                )

        sorted_ordered_items = sorted(ordered_items, key=lambda x: (x["span"][0], x["span"][1]))
        schedule_parts = [str(item["code"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "SCHEDULE"]
        mm_parts = [str(item["code"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "MM"]

        thickness_code = ""
        if sorted_ordered_items:
            thickness_code = 'X'.join(str(item["code"]) for item in sorted_ordered_items)
        elif schedule_parts and mm_parts:
            thickness_code = 'X'.join(schedule_parts + mm_parts)
        elif schedule_parts:
            thickness_code = 'X'.join(schedule_parts)
        elif mm_parts:
            thickness_code = 'X'.join(mm_parts)

        consumed_spans = self._derive_consumed_spans_from_ordered_items(normalized, ordered_items)

        return RuleThicknessExtraction(
            schedule=schedule_parts,
            mm=mm_parts,
            thickness_code=thickness_code,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
            consumed_spans=consumed_spans,
            ordered_items=[
                {"type": str(item["type"]), "value": str(item["value"])}
                for item in sorted_ordered_items
            ],
        )

    @staticmethod
    def _has_size_context(size_context: Optional[Any]) -> bool:
        if not size_context:
            return False
        return bool(
            getattr(size_context, "dn", None)
            or getattr(size_context, "od", None)
            or getattr(size_context, "inch", None)
        )

    @staticmethod
    def _derive_consumed_spans_from_ordered_items(text: str, ordered_items: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        consumed_spans: List[Tuple[int, int]] = []
        cursor_by_span: Dict[Tuple[int, int], int] = {}
        mm_index_by_span: Dict[Tuple[int, int], int] = {}
        mm_items_per_span: Dict[Tuple[int, int], int] = {}
        numeric_tokens_by_span: Dict[Tuple[int, int], List[re.Match[str]]] = {}
        mm_tokens_by_span: Dict[Tuple[int, int], List[re.Match[str]]] = {}
        numeric_token_re = re.compile(r'\d+(?:\.\d+)?')
        mm_token_re = re.compile(r'(?i)\d+(?:\.\d+)?\s*(?:MM|毫米)')
        schedule_like_re = re.compile(r'(?i)SCH\s*\.?\s*(?:\d+S?|STD|XS|XXS)|S-(?:\d+S?|STD|XS|XXS)|S\d+S?|\d+S|STD|XS|XXS')

        for item in ordered_items:
            span = item.get("span")
            if not span or not isinstance(span, tuple) or len(span) != 2:
                continue
            key = (int(span[0]), int(span[1]))
            if str(item.get("type") or "").upper() == "MM":
                mm_items_per_span[key] = mm_items_per_span.get(key, 0) + 1

        for item in ordered_items:
            span = item.get("span")
            if not span or not isinstance(span, tuple) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or end > len(text) or start >= end:
                continue
            full_text = text[start:end]
            cursor = cursor_by_span.get((start, end), 0)
            search_text = full_text[cursor:]
            item_type = str(item.get("type") or "").upper()
            token_start = token_end = None
            if item_type == "MM":
                mm_total = mm_items_per_span.get((start, end), 1)
                mm_idx = mm_index_by_span.get((start, end), 0)

                mm_tokens = mm_tokens_by_span.get((start, end))
                if mm_tokens is None:
                    mm_tokens = list(mm_token_re.finditer(full_text))
                    mm_tokens_by_span[(start, end)] = mm_tokens
                if mm_tokens:
                    base_index = max(0, len(mm_tokens) - mm_total)
                    token_index = min(len(mm_tokens) - 1, base_index + mm_idx)
                    token_match = mm_tokens[token_index]
                    token_start = start + token_match.start()
                    token_end = start + token_match.end()
                else:
                    tokens = numeric_tokens_by_span.get((start, end))
                    if tokens is None:
                        tokens = list(numeric_token_re.finditer(full_text))
                        numeric_tokens_by_span[(start, end)] = tokens
                    if not tokens:
                        continue
                    base_index = max(0, len(tokens) - mm_total)
                    token_index = min(len(tokens) - 1, base_index + mm_idx)
                    token_match = tokens[token_index]
                    token_start = start + token_match.start()
                    token_end = start + token_match.end()
                mm_index_by_span[(start, end)] = mm_idx + 1
            else:
                token_match = schedule_like_re.search(search_text)
                if not token_match:
                    continue
                token_start = start + cursor + token_match.start()
                token_end = start + cursor + token_match.end()
            candidate = (token_start, token_end)
            if candidate not in consumed_spans:
                consumed_spans.append(candidate)
            if item_type != "MM":
                cursor_by_span[(start, end)] = cursor + token_match.end()

        return consumed_spans

    def _extract_explicit_thickness_rules(
        self,
        normalized: str,
        blocked_spans: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        temperature_parenthetical_pattern = re.compile(
            r'(?i)'
            r'(?:'
            r'SCH[.\s]*(?:XXS|XS|STD|\d+S?)|'
            r'S-(?:XXS|XS|STD|\d+S?)|'
            r'S\d+S?|'
            r'\d+S|'
            r'(?:THK|T|壁厚)\s*[:：=]?\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?|'
            r'\d+(?:\.\d+)?\s*(?:MM|毫米)'
            r')'
            r'(?=\s*\(\s*-?\d+(?:\.\d+)?\s*(?:℃|°C|C)\s*\))'
        )
        effective_blocked_spans = list(blocked_spans)
        effective_blocked_spans.extend(
            (m.start(), m.end()) for m in temperature_parenthetical_pattern.finditer(normalized)
        )

        schedule_parts: List[str] = []
        mm_parts: List[str] = []
        ordered_items: List[Dict[str, Any]] = []
        matched_texts: List[str] = []
        matched_spans: List[Tuple[int, int]] = []
        invalid = False

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _add_ordered(item_type: str, raw_value: str, code_value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(raw_value), "code": str(code_value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        def _overlaps_blocked(span: Tuple[int, int]) -> bool:
            for start, end in effective_blocked_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        def _overlaps_recorded(span: Tuple[int, int]) -> bool:
            for start, end in matched_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _overlaps_recorded(span: Tuple[int, int]) -> bool:
            for start, end in matched_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _mark_invalid_if_needed(mm_raw: Optional[str] = None, schedule_raw: Optional[str] = None) -> bool:
            nonlocal invalid
            if mm_raw and not self._is_valid_mm_candidate(mm_raw):
                invalid = True
                return True
            if schedule_raw and not self._is_valid_schedule_candidate(schedule_raw):
                invalid = True
                return True
            return False

        def _mark_invalid_schedule_boundary_conflict(pattern: re.Pattern[str]) -> bool:
            nonlocal invalid
            if pattern.search(normalized):
                invalid = True
                return True
            return False

        def _normalize_two_part_operand(raw_operand: str) -> Optional[Tuple[str, str, str]]:
            operand = str(raw_operand or "").strip()
            if not operand:
                return None

            operand = re.sub(r'(?i)\s*\([LS]\)\s*$', '', operand).strip()

            if re.fullmatch(r'(?i)(?:SCH[.\s]*(?:\d+S?|STD|XS|XXS)|S-(?:\d+S?|STD|XS|XXS)|S\d+S?|\d+S|XS|XXS|STD)', operand):
                if _mark_invalid_if_needed(schedule_raw=operand):
                    return None
                code = self._convert_single(operand).strip()
                return ("SCHEDULE", code, code) if code else None

            prefixed_numeric = re.match(
                r'(?i)^(?:(?:THK|T|壁厚)\s*[:：=]?\s*(\d+(?:\.\d+)?)\s*(?:MM|毫米)?|S\s*[:：=]\s*(\d+(?:\.\d+)?)\s*(?:MM|毫米)?|S-(\d+\.\d+)\s*(?:MM|毫米)?)$',
                operand,
            )
            if prefixed_numeric:
                raw_num = prefixed_numeric.group(1) or prefixed_numeric.group(2) or prefixed_numeric.group(3)
                if _mark_invalid_if_needed(mm_raw=raw_num):
                    return None
                normalized_num = self._normalize_number(raw_num)
                return ("MM", normalized_num, f"{normalized_num}MM")

            plain_mm = re.match(r'(?i)^(\d+(?:\.\d+)?)\s*(?:MM|毫米)$', operand)
            if plain_mm:
                raw_num = plain_mm.group(1)
                if _mark_invalid_if_needed(mm_raw=raw_num):
                    return None
                normalized_num = self._normalize_number(raw_num)
                return ("MM", normalized_num, f"{normalized_num}MM")

            return None

        # 0) 泛化二元组合解析器：
        # 左右两侧只要都是合法壁厚 token，就按组合处理；
        # 不再把组合规则写死成 `mm x schedule`、`schedule x schedule` 等固定方向。
        sch_token = r'SCH[.\s]*(?>(?:XXS|XS|STD|\d+S?(?!\.\d)))'
        s_dash_token = r'S-(?>(?:XXS|XS|STD|\d+S?(?!\.\d)))'
        # 最弱的数值 schedule 只认白名单：
        # SCH5/SCH5S、SCH10/SCH10S ... SCH160/SCH160S 对应的弱写法。
        # 这样可以避免把 S3408 这类标准残片误当壁厚。
        weak_schedule_prefixed_token, weak_schedule_s_dash_token, weak_schedule_numeric_suffix_token, weak_schedule_token = self._weak_schedule_patterns()
        schedule_boundary_conflict_patterns = [
            re.compile(r'(?i)SCH[.\s]*(?>\d+)(?=\d)'),
            re.compile(r'(?i)SCH[.\s]*(?>\d+)S(?=[A-Za-z])'),
            re.compile(r'(?i)SCH[.\s]*(?:STD|XS|XXS)(?=[A-Za-z0-9])'),
            # 合法的 S-10S / S-40S 不应被误判成粘连脏串；
            # 对 `S-40CL300` / `S-40PN16` 这类“壁厚词 + 压力词”粘连，交给后续专门规则处理。
            re.compile(r'(?i)S-\d+(?=(?!(?:CL|CLASS|PN))[A-RT-Za-rt-z])'),
            re.compile(r'(?i)S-(?:STD|XS|XXS)(?=[A-Za-z0-9])'),
        ]
        for schedule_boundary_conflict_pattern in schedule_boundary_conflict_patterns:
            if _mark_invalid_schedule_boundary_conflict(schedule_boundary_conflict_pattern):
                return {
                    "schedule": [],
                    "mm": [],
                    "ordered_items": [],
                    "matched_texts": [],
                    "matched_spans": [],
                    "invalid": invalid,
                }

        schedule_operand = rf'(?:{sch_token}|{s_dash_token}|{weak_schedule_token})(?:\s*\([LS]\))?'
        mm_operand = r'(?:(?:THK|T|壁厚)\s*[:：=]?\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?|S\s*[:：=]\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?|S-\d+\.\d+\s*(?:MM|毫米)?|\d+(?:\.\d+)?\s*(?:MM|毫米))(?:\s*\([LS]\))?'
        generic_operand = rf'(?:{schedule_operand}|{mm_operand})'
        generic_two_part_combo_pattern = re.compile(
            rf'(?i)(?<![A-Za-z0-9])({generic_operand})\s*[xX×*/]\s*({generic_operand})(?=$|[^A-Za-z0-9])'
        )
        for m in generic_two_part_combo_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span) or _overlaps_recorded(span):
                continue

            left = _normalize_two_part_operand(m.group(1))
            right = _normalize_two_part_operand(m.group(2))
            if not left or not right:
                continue

            for operand, operand_span in ((left, m.span(1)), (right, m.span(2))):
                item_type, raw_value, code_value = operand
                if item_type == "MM":
                    _add_unique(mm_parts, code_value)
                    _add_ordered("MM", raw_value, code_value, operand_span)
                else:
                    _add_unique(schedule_parts, code_value)
                    _add_ordered("SCHEDULE", raw_value, code_value, operand_span)
            _record(m.group(0), span)

        thk_mm_schedule_pattern = re.compile(
            r'(?i)(?:\bTHK(?=\s*[:=]?\s*\d)\s*[:=]?\s*|\bT\b\s*[:=]\s*|壁厚\s*[:：=]?\s*|\bS\b\s*[:=]\s*)'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?\s*(?:\([LS]\))?'
            r'\s*[xX×]\s*'
            r'((?:SCH[.\s]*\d+S?|SCH[.\s]*(?:STD|XS|XXS)|STD|XS|XXS|S-(?:\d+S?|STD|XS|XXS)|\d+S))(?:\s*\([LS]\))?'
        )
        for m in thk_mm_schedule_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1), schedule_raw=m.group(2)):
                continue
            mm_value = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, mm_value)
            _add_ordered("MM", self._normalize_number(m.group(1)), mm_value, m.span(1))
            schedule_part = self._convert_single(m.group(2)).strip()
            if schedule_part:
                _add_unique(schedule_parts, schedule_part)
                _add_ordered("SCHEDULE", schedule_part, schedule_part, m.span(2))
            _record(m.group(0), span)

        single_s_dash_mm_pattern = re.compile(
            r'(?i)(?<![A-Za-z0-9])S-(\d+\.\d+)\s*(?:MM|毫米)?(?=$|[^A-Za-z0-9])'
        )
        for m in single_s_dash_mm_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)):
                continue
            mm_value = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, mm_value)
            _add_ordered("MM", self._normalize_number(m.group(1)), mm_value, span)
            _record(m.group(0), span)

        thk_repeated_prefix_pair_pattern = re.compile(
            r'(?i)(?:\bTHK(?=\s*[:=]?\s*\d)\s*[:=]?\s*|\bT\b\s*[:=]\s*|壁厚\s*[:：=]?\s*)'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?\s*(?:\([LS]\))?'
            r'\s*[xX×]\s*'
            r'(?:THK(?=\s*[:=]?\s*\d)\s*[:=]?\s*|T\s*[:=]\s*|壁厚\s*[:：=]?\s*)'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?\s*(?:\([LS]\))?'
        )
        for m in thk_repeated_prefix_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or _mark_invalid_if_needed(mm_raw=m.group(2)):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            right = f"{self._normalize_number(m.group(2))}MM"
            _add_unique(mm_parts, left)
            _add_unique(mm_parts, right)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        thk_pair_pattern = re.compile(
            r'(?i)(?:\bTHK(?=\s*[:=]?\s*\d)\s*[:=]?\s*|\bT\b\s*[:=]\s*|壁厚\s*[:：=]?\s*|\bS\b\s*[:=]\s*)'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?\s*(?:\([LS]\))?'
            r'\s*[xX×]\s*'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?\s*(?:\([LS]\))?'
        )
        for m in thk_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or _mark_invalid_if_needed(mm_raw=m.group(2)):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            right = f"{self._normalize_number(m.group(2))}MM"
            _add_unique(mm_parts, left)
            _add_unique(mm_parts, right)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        dn_pair_dual_mm_pattern = re.compile(
            r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?\s*[xX×*]\s*(?:DN\s*)?\d+(?:\.\d+)?\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)'
        )
        for m in dn_pair_dual_mm_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if any(span[0] < end and start < span[1] for start, end in matched_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or _mark_invalid_if_needed(mm_raw=m.group(2)):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            right = f"{self._normalize_number(m.group(2))}MM"
            _add_unique(mm_parts, left)
            _add_unique(mm_parts, right)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        thk_pattern = re.compile(
            r'(?i)(?:\bTHK(?=\s*[:=]?\s*\d)\s*[:=]?\s*|\bT\b\s*[:=]\s*|壁厚\s*[:：=]?\s*|\bS\b\s*[:=]\s*)'
            r'(\d+(?:\.\d+)?)\s*(?:MM)?'
            r'(?:\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:MM)?)?'
        )
        for m in thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if any(span[0] < end and start < span[1] for start, end in matched_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or (m.group(2) and _mark_invalid_if_needed(mm_raw=m.group(2))):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, left)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            if m.group(2):
                right = f"{self._normalize_number(m.group(2))}MM"
                _add_unique(mm_parts, right)
                _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        if self._common_dn_values:
            dn_pair_mm_pair_pattern = re.compile(
                r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?\s*[xX×*/]\s*DN\s*'
                r'(' + '|'.join(map(re.escape, [str(v) for v in self._common_dn_values])) + r')'
                r'(\d+\.\d+)\s*MM\s*[xX×]\s*(\d+\.\d+)\s*MM'
            )
            for m in dn_pair_mm_pair_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _overlaps_blocked(span):
                    continue
                if _mark_invalid_if_needed(mm_raw=m.group(2)) or _mark_invalid_if_needed(mm_raw=m.group(3)):
                    continue
                _add_unique(mm_parts, f"{self._normalize_number(m.group(2))}MM")
                _add_unique(mm_parts, f"{self._normalize_number(m.group(3))}MM")
                _record(m.group(0), span)

        # 1) 强规则：组合 schedule 优先，避免后面的弱 token 越权。
        #
        # 关键点：
        # - SCH... 和 S-... 不能混在一个大正则里直接“猜”组合关系。
        # - 组合规则必须有明确的 x/X/× 分隔，或是少量已知紧凑拼写（如 SCH10Sch20S）。
        # - 一旦命中，后续规则不能再从该片段内部重复提取。
        schedule_left_boundary = r'(?:(?<=^)|(?<=[^A-Za-z])|(?<=[xX×*]))'
        schedule_right_boundary = r'(?=$|[^A-Za-z0-9]|[xX×*/])'
        single_schedule_left_boundary = r'(?:(?<=^)|(?<=[^A-Za-z]))'
        sch_numeric_right_boundary = r'(?=$|[^0-9])'
        sch_numeric_s_right_boundary = r'(?=$|[^A-Za-z])'
        compact_schedule_pair_patterns = [
            re.compile(
                rf'(?i){schedule_left_boundary}({sch_token}\s*[xX×]\s*{sch_token}){schedule_right_boundary}'
            ),
            re.compile(
                rf'(?i){schedule_left_boundary}({s_dash_token}\s*[xX×]\s*{s_dash_token}){schedule_right_boundary}'
            ),
            re.compile(
                rf'(?i){schedule_left_boundary}((?:{sch_token}|{s_dash_token})\s*[xX×]\s*(?:{sch_token}|{s_dash_token}|{weak_schedule_token})){schedule_right_boundary}'
            ),
            re.compile(
                rf'(?i){schedule_left_boundary}((?:{weak_schedule_token})\s*[xX×]\s*(?:{sch_token}|{s_dash_token}|{weak_schedule_token})){schedule_right_boundary}'
            ),
            # 紧凑拼写：SCH10Sch20S / SCH10SXSCH10S 等，留给 split_concatenated_tokens 做切分。
            re.compile(
                rf'(?i){schedule_left_boundary}(({sch_token}(?:\s*[xX×]\s*)?{sch_token})){schedule_right_boundary}'
            ),
        ]
        consumed_schedule_spans: List[Tuple[int, int]] = []
        for pair_pattern in compact_schedule_pair_patterns:
            for m in pair_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _overlaps_blocked(span):
                    continue
                if _overlaps_recorded(span):
                    continue
                if any(start < span[1] and span[0] < end for start, end in consumed_schedule_spans):
                    continue
                raw_value = m.group(1)
                if _mark_invalid_if_needed(schedule_raw=raw_value):
                    continue
                raw_parts = self._split_parts(raw_value)
                if len(raw_parts) < 2:
                    compact_parts = self._split_concatenated_tokens(raw_value)
                    if len(compact_parts) > 1:
                        raw_parts = compact_parts
                normalized_parts: List[str] = []
                for raw_part in raw_parts:
                    part = self._convert_single(raw_part).strip()
                    if part:
                        normalized_parts.append(part)
                if len(normalized_parts) < 2:
                    continue
                for part in normalized_parts:
                    _add_unique(schedule_parts, part)
                    _add_ordered("SCHEDULE", part, part, span)
                consumed_schedule_spans.append(span)
                _record(m.group(0), span)

        # 1.5) 强规则：壁厚词与压力词粘连
        # 例如 S-40CL300 / SCH40CL150 / STDCL3000。
        glued_schedule_pressure_pattern = re.compile(
            rf'(?i){single_schedule_left_boundary}'
            rf'((?:{sch_token}|{s_dash_token}|STD|XS|XXS|{weak_schedule_token}))'
            rf'(?=(?:\s*CL(?:ASS)?[.\s-]*\d|\s*C\s*\d|\s*PN\s*\d|\s*\d+\s*(?:LBS?|#)))'
        )
        for m in glued_schedule_pressure_pattern.finditer(normalized):
            span = (m.start(1), m.end(1))
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if any(start < span[1] and span[0] < end for start, end in consumed_schedule_spans):
                continue
            raw_value = m.group(1)
            if _mark_invalid_if_needed(schedule_raw=raw_value):
                continue
            part = self._convert_single(raw_value).strip()
            if part:
                _add_unique(schedule_parts, part)
                _add_ordered("SCHEDULE", part, part, span)
                consumed_schedule_spans.append(span)
                _record(raw_value, span)

        # 2) 强规则：单个 SCH... / S-... token
        strong_schedule_token_patterns = [
            re.compile(
                rf'(?i){single_schedule_left_boundary}(?:SCH[.\s]*\d+S){sch_numeric_s_right_boundary}'
            ),
            re.compile(
                rf'(?i){single_schedule_left_boundary}(?:SCH[.\s]*\d+){sch_numeric_right_boundary}'
            ),
            re.compile(
                rf'(?i){single_schedule_left_boundary}(?:SCH[.\s]*(?:STD|XS|XXS)|S-(?:\d+S?|STD|XS|XXS)){schedule_right_boundary}'
            ),
        ]
        for strong_schedule_token_pattern in strong_schedule_token_patterns:
            for m in strong_schedule_token_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _overlaps_blocked(span):
                    continue
                if _overlaps_recorded(span):
                    continue
                if any(start < span[1] and span[0] < end for start, end in consumed_schedule_spans):
                    continue
                if _mark_invalid_if_needed(schedule_raw=m.group(0)):
                    continue
                part = self._convert_single(m.group(0)).strip()
                if part:
                    _add_unique(schedule_parts, part)
                    _add_ordered("SCHEDULE", part, part, span)
                    consumed_schedule_spans.append(span)
                    _record(m.group(0), span)

        # 3) 弱规则拆分：
        # - XS / XXS / STD：弱 token，但可以在无强 schedule 的情况下直接使用
        # - 数字S（10S/40S/...）：只有在前面完全没有提到任何显式壁厚时才允许启用
        #
        # 同时保持硬边界：
        # - 不允许从 B16.9S-40S 里截出 9S
        # - 不把 Mnf Std / MFR STD / MFRS STD / ENR STD 当壁厚
        weak_schedule_alpha_pattern = re.compile(
            r'(?i)(?<![A-Za-z0-9.])(?:XXS|XS|STD)(?![A-Za-z0-9])'
        )
        for m in weak_schedule_alpha_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if any(start < span[1] and span[0] < end for start, end in consumed_schedule_spans):
                continue
            raw_token = m.group(0)
            prefix = normalized[max(0, m.start() - 8):m.start()]
            if raw_token.upper() == 'STD' and re.search(r'(?i)(?:MNF|MFRS?|MFR|ENR)\s*$', prefix):
                continue
            if _mark_invalid_if_needed(schedule_raw=raw_token):
                continue
            part = self._convert_single(raw_token).strip()
            if part:
                _add_unique(schedule_parts, part)
                _add_ordered("SCHEDULE", part, part, span)
                consumed_schedule_spans.append(span)
                _record(raw_token, span)

        # 数字S / S整数 是最弱的一层：
        # 只有在前面没有任何显式壁厚（schedule/mm）时才允许启用，
        # 避免像 P150S40S 这种编码串里的 150S 越权进入壁厚。
        if not schedule_parts and not mm_parts:
            weak_numeric_schedule_pattern = re.compile(
                rf'(?i)(?<![A-Za-z0-9.])(?:{weak_schedule_prefixed_token}|{weak_schedule_s_dash_token}|{weak_schedule_numeric_suffix_token})(?![A-Za-z0-9.]|\.\d)'
            )
            for m in weak_numeric_schedule_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _overlaps_blocked(span):
                    continue
                if _overlaps_recorded(span):
                    continue
                if any(start < span[1] and span[0] < end for start, end in consumed_schedule_spans):
                    continue
                raw_token = m.group(0)
                # 弱规则候选非法时只跳过该候选，不整条作废。
                if not self._is_valid_schedule_candidate(raw_token):
                    continue
                part = self._convert_single(raw_token).strip()
                if part:
                    _add_unique(schedule_parts, part)
                    _add_ordered("SCHEDULE", part, part, span)
                    consumed_schedule_spans.append(span)
                    _record(raw_token, span)

        od_schedule_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*\d+(?:\.\d+)?\s*[xX×*]\s*((?:SCH[.\s]*\d+S?|SCH[.\s]*(?:STD|XS|XXS)|STD|XS|XXS|S-(?:\d+S?|STD|XS|XXS)|\d+S))'
        )
        for m in od_schedule_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _mark_invalid_if_needed(schedule_raw=m.group(1)):
                continue
            raw_parts = self._split_parts(m.group(1))
            for raw_part in raw_parts:
                part = self._convert_single(raw_part)
                part = part.strip()
                if part:
                    _add_unique(schedule_parts, part)
                    _add_ordered("SCHEDULE", part, part, span)
            _record(m.group(0), span)

        # 结构族：ΦA×B/T1×T2 或 ΦA×ΦB/T1×T2
        # 当前两段为尺寸、后两段为壁厚时，直接提取后两段壁厚。
        phi_dual_od_dual_thk_pattern = re.compile(
            r'(?i)[φΦФф]\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(?:[φΦФф]\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)'
        )
        consumed_phi_dual_spans: List[Tuple[int, int]] = []
        for m in phi_dual_od_dual_thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            value_spans = [m.span(3), m.span(4)]
            if any(_overlaps_blocked(value_span) for value_span in value_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(3)) or _mark_invalid_if_needed(mm_raw=m.group(4)):
                continue
            first = f"{self._normalize_number(m.group(3))}MM"
            second = f"{self._normalize_number(m.group(4))}MM"
            _add_unique(mm_parts, first)
            _add_unique(mm_parts, second)
            _add_ordered("MM", self._normalize_number(m.group(3)), first, m.span(3))
            _add_ordered("MM", self._normalize_number(m.group(4)), second, m.span(4))
            consumed_phi_dual_spans.append(span)
            _record(m.group(0), span)

        od_double_head_thk_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*\d+(?:\.\d+)?\s*[xX×]\s*(?:\bOD|[φΦФф]|\bD)\s*\d+(?:\.\d+)?\s*[xX×]\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\b'
        )
        od_three_thk_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:MM)?'
        )
        consumed_thk_spans: List[Tuple[int, int]] = []
        for m in od_double_head_thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            value_spans = [m.span(1)] + ([m.span(2)] if m.group(2) else [])
            if any(_overlaps_blocked(value_span) for value_span in value_spans):
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_phi_dual_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or (m.group(2) and _mark_invalid_if_needed(mm_raw=m.group(2))):
                continue
            first = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, first)
            _add_ordered("MM", self._normalize_number(m.group(1)), first, m.span(1))
            if m.group(2):
                second = f"{self._normalize_number(m.group(2))}MM"
                _add_unique(mm_parts, second)
                _add_ordered("MM", self._normalize_number(m.group(2)), second, m.span(2))
            consumed_thk_spans.append(span)
            _record(m.group(0), span)
        for m in od_three_thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(m.span(1)):
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_phi_dual_spans):
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_thk_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)):
                continue
            value = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, value)
            _add_ordered("MM", self._normalize_number(m.group(1)), value, span)
            consumed_thk_spans.append(span)
            _record(m.group(0), span)

        od_thk_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*\d+(?:\.\d+)?\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*(?:MM)?(?!\s*[xX×*]\s*\d)'
        )
        for m in od_thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(m.span(1)):
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_phi_dual_spans):
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_thk_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)):
                continue
            value = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, value)
            _add_ordered("MM", self._normalize_number(m.group(1)), value, span)
            _record(m.group(0), span)

        mm_pair_pattern = re.compile(
            r'(?i)(\d+(?:\.\d+)?)\s*(?:MM|毫米)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:MM|毫米)'
        )
        for m in mm_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or _mark_invalid_if_needed(mm_raw=m.group(2)):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            right = f"{self._normalize_number(m.group(2))}MM"
            _add_unique(mm_parts, left)
            _add_unique(mm_parts, right)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        mm_mixed_pair_pattern = re.compile(
            r'(?i)(\d+(?:\.\d+)?)\s*(?:MM|毫米)\s*[xX×]\s*(\d+(?:\.\d+)?)(?!\s*(?:MM|毫米))'
        )
        for m in mm_mixed_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)) or _mark_invalid_if_needed(mm_raw=m.group(2)):
                continue
            left = f"{self._normalize_number(m.group(1))}MM"
            right = f"{self._normalize_number(m.group(2))}MM"
            _add_unique(mm_parts, left)
            _add_unique(mm_parts, right)
            _add_ordered("MM", self._normalize_number(m.group(1)), left, m.span(1))
            _add_ordered("MM", self._normalize_number(m.group(2)), right, m.span(2))
            _record(m.group(0), span)

        dn_pair_trailing_mm_pattern = re.compile(
            r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?\s*[xX×*]\s*DN\s*\d+(?:\.\d+)?\s+(\d+(?:\.\d+)?)\s*(?:MM|毫米)\b'
        )
        for m in dn_pair_trailing_mm_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if any(span[0] < end and start < span[1] for start, end in matched_spans):
                continue
            if _mark_invalid_if_needed(mm_raw=m.group(1)):
                continue
            value = f"{self._normalize_number(m.group(1))}MM"
            _add_unique(mm_parts, value)
            _add_ordered("MM", self._normalize_number(m.group(1)), value, span)
            _record(m.group(0), span)

        return {
            "schedule": schedule_parts,
            "mm": mm_parts,
            "ordered_items": ordered_items,
            "matched_texts": matched_texts,
            "matched_spans": matched_spans,
            "invalid": invalid,
        }

    def _apply_weak_thickness_fallback(
        self,
        normalized: str,
        blocked_spans: List[Tuple[int, int]],
        schedule_parts: List[str],
        mm_parts: List[str],
        ordered_items: List[Dict[str, Any]],
        matched_texts: List[str],
        matched_spans: List[Tuple[int, int]],
    ) -> None:
        if schedule_parts or mm_parts:
            return

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _add_ordered(item_type: str, raw_value: str, code_value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(raw_value), "code": str(code_value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        def _overlaps_blocked(span: Tuple[int, int]) -> bool:
            for start, end in blocked_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        for block in self._extract_bare_od_thickness_blocks(normalized):
            span = block["span"]
            if _overlaps_blocked(span):
                continue
            value = f"{self._normalize_number(block['thickness'])}MM"
            _add_unique(mm_parts, value)
            _add_ordered("MM", self._normalize_number(block["thickness"]), value, span)
            _record(block["raw"], span)

        dn_decimal_patterns = (
            re.compile(r'(?i)\bDN\s*\d+(?:\.\d+)?\s*[xX×*]\s*(\d+\.\d+)\b'),
            re.compile(r'(?i)\bDN\s*\d+(?:\.\d+)?\s*-\s*(\d+\.\d+)\b'),
        )
        for pattern in dn_decimal_patterns:
            for m in pattern.finditer(normalized):
                second_value_span = m.span(1)
                if _overlaps_blocked(second_value_span):
                    continue
                value = f"{self._normalize_number(m.group(1))}MM"
                if value not in mm_parts:
                    _add_unique(mm_parts, value)
                    _add_ordered("MM", self._normalize_number(m.group(1)), value, second_value_span)
                    _record(m.group(0), (m.start(), m.end()))

    def _apply_size_context_thickness_rules(
        self,
        normalized: str,
        blocked_spans: List[Tuple[int, int]],
        size_context: Any,
        schedule_parts: List[str],
        mm_parts: List[str],
        ordered_items: List[Dict[str, Any]],
        matched_texts: List[str],
        matched_spans: List[Tuple[int, int]],
    ) -> None:
        """
        当尺寸已经明确存在时，允许把无 THK 前缀但结构稳定的 `数字mm x schedule`
        / `数字mm x 数字mm` / `数字mm x 数字` 视为壁厚组合。
        """

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _add_ordered(item_type: str, raw_value: str, code_value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(raw_value), "code": str(code_value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        def _overlaps_blocked(span: Tuple[int, int]) -> bool:
            for start, end in blocked_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        def _overlaps_recorded(span: Tuple[int, int]) -> bool:
            for start, end in matched_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        # 尺寸右侧 x 厚度值：DN20x5.6mm / OD219.1X4.00 / 6"x10mm
        size_spans = list(getattr(size_context, "consumed_spans", None) or getattr(size_context, "matched_spans", []) or [])
        right_mm_pattern = re.compile(
            r'(?i)^\s*[xX×*]\s*(?:(\d+(?:\.\d+)?)\s*(MM|毫米)|(\d+\.\d+)(?![\dA-Za-z]))'
        )
        for size_start, size_end in size_spans:
            if not (0 <= size_end < len(normalized)):
                continue
            right_tail = normalized[size_end:]
            m = right_mm_pattern.match(right_tail)
            if not m:
                continue
            if m.group(1):
                raw_num = m.group(1)
                unit = m.group(2) or "MM"
                token_start = size_end + m.start(1)
                token_end = size_end + m.end(2)
            else:
                raw_num = m.group(3)
                unit = ""
                token_start = size_end + m.start(3)
                token_end = size_end + m.end(3)
            span = (token_start, token_end)
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if not self._is_valid_mm_candidate(raw_num):
                continue
            value = f"{self._normalize_number(raw_num)}MM"
            _add_unique(mm_parts, value)
            _add_ordered("MM", self._normalize_number(raw_num), value, span)
            record_end = token_end
            if unit:
                record_text = normalized[token_start:token_end]
            else:
                record_text = normalized[token_start:token_end]
            _record(record_text, span)

        # 数字mm x Sch/STD/XS
        mm_schedule_pattern = re.compile(
            r'(?i)(\d+(?:\.\d+)?)\s*(?:MM|毫米)\s*(?:\([LS]\))?\s*[xX×*]\s*'
            r'((?:SCH[.\s]*\d+S?|SCH[.\s]*(?:STD|XS|XXS)|STD|XS|XXS|S-(?:\d+S?|STD|XS|XXS)|\d+S))(?:\s*\([LS]\))?'
        )
        for m in mm_schedule_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _overlaps_blocked(span):
                continue
            if _overlaps_recorded(span):
                continue
            if not self._is_valid_mm_candidate(m.group(1)) or not self._is_valid_schedule_candidate(m.group(2)):
                continue
            mm_value = f"{self._normalize_number(m.group(1))}MM"
            schedule_value = self._convert_single(m.group(2)).strip()
            _add_unique(mm_parts, mm_value)
            _add_ordered("MM", self._normalize_number(m.group(1)), mm_value, m.span(1))
            if schedule_value and schedule_value not in schedule_parts:
                _add_unique(schedule_parts, schedule_value)
                _add_ordered("SCHEDULE", schedule_value, schedule_value, m.span(2))
            _record(m.group(0), span)

    def _apply_size_context_glued_schedule_rules(
        self,
        normalized: str,
        blocked_spans: List[Tuple[int, int]],
        size_context: Any,
        schedule_parts: List[str],
        ordered_items: List[Dict[str, Any]],
        matched_texts: List[str],
        matched_spans: List[Tuple[int, int]],
    ) -> None:
        """
        当尺寸已经提取成功时，允许尺寸 span 直接充当 schedule token 的左右边界。

        例如：
        - DN40S10S
        - S10SDN10
        - S10SOD108

        这里只放宽边界，不放宽 token 本体：
        后续仍要求命中的内容本身是合法 schedule。
        """

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _add_ordered(item_type: str, raw_value: str, code_value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(raw_value), "code": str(code_value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        def _overlaps_blocked(span: Tuple[int, int]) -> bool:
            for start, end in blocked_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        def _overlaps_recorded(span: Tuple[int, int]) -> bool:
            for start, end in matched_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        weak_schedule_prefixed_token, weak_schedule_s_dash_token, weak_schedule_numeric_suffix_token, _ = self._weak_schedule_patterns()
        schedule_token_pattern = (
            rf'(?:SCH[.\s]*(?:\d+S?|STD|XS|XXS)|'
            rf'S-(?:\d+S?|STD|XS|XXS)|'
            rf'{weak_schedule_prefixed_token}|'
            rf'{weak_schedule_s_dash_token}|'
            rf'{weak_schedule_numeric_suffix_token}|'
            rf'XS|XXS|STD)'
        )
        separator_prefixed_right_schedule_pattern = re.compile(
            rf'(?i)^[xX×*/,]\s*({schedule_token_pattern})(?=$|[^A-Za-z0-9.]|\.\d)'
        )
        right_schedule_pattern = re.compile(
            rf'(?i)^({schedule_token_pattern})(?=$|[^A-Za-z0-9.]|\.\d)'
        )
        left_schedule_pattern = re.compile(
            rf'(?i){schedule_token_pattern}$'
        )

        size_spans = list(getattr(size_context, "consumed_spans", None) or getattr(size_context, "matched_spans", []) or [])
        for size_span in size_spans:
            size_start, size_end = size_span

            # 尺寸右侧紧贴 schedule：DN40S10S
            # 尺寸右侧通过分隔符连接 schedule：DN20xXS / OD89xSTD
            if 0 <= size_end < len(normalized):
                right_tail = normalized[size_end:]
                separator_prefixed_match = separator_prefixed_right_schedule_pattern.match(right_tail)
                if separator_prefixed_match:
                    raw_token = separator_prefixed_match.group(1)
                    span = (
                        size_end + separator_prefixed_match.start(1),
                        size_end + separator_prefixed_match.end(1),
                    )
                    if not _overlaps_blocked(span) and not _overlaps_recorded(span):
                        if self._is_valid_schedule_candidate(raw_token):
                            part = self._convert_single(raw_token).strip()
                            if part:
                                _add_unique(schedule_parts, part)
                                _add_ordered("SCHEDULE", part, part, span)
                                _record(raw_token, span)
                else:
                    right_match = right_schedule_pattern.match(right_tail)
                    if right_match:
                        raw_token = right_match.group(1)
                        span = (size_end, size_end + len(raw_token))
                        if not _overlaps_blocked(span) and not _overlaps_recorded(span):
                            if self._is_valid_schedule_candidate(raw_token):
                                part = self._convert_single(raw_token).strip()
                                if part:
                                    _add_unique(schedule_parts, part)
                                    _add_ordered("SCHEDULE", part, part, span)
                                    _record(raw_token, span)

            # 尺寸左侧紧贴 schedule：S10SDN10 / S10SOD108
            if size_start > 0:
                left_head = normalized[:size_start]
                left_match = left_schedule_pattern.search(left_head)
                if left_match:
                    raw_token = left_match.group(0)
                    span = (size_start - len(raw_token), size_start)
                    if not _overlaps_blocked(span) and not _overlaps_recorded(span):
                        if self._is_valid_schedule_candidate(raw_token):
                            part = self._convert_single(raw_token).strip()
                            if part:
                                _add_unique(schedule_parts, part)
                                _add_ordered("SCHEDULE", part, part, span)
                                _record(raw_token, span)

    @staticmethod
    def _is_valid_mm_candidate(raw_value: str) -> bool:
        value = str(raw_value or "").strip()
        match = re.match(r'^\d+(?:\.(\d+))?$', value)
        if not match:
            return False
        decimal_part = match.group(1) or ""
        if len(decimal_part) > 2:
            return False
        try:
            return float(value) <= 100
        except ValueError:
            return False

    @staticmethod
    def _is_valid_schedule_candidate(raw_value: str) -> bool:
        value = str(raw_value or "").upper()
        for numeric_part in re.findall(r'\d+', value):
            if len(numeric_part) >= 4:
                return False
        return True

    def _has_invalid_thickness_items(self, schedule_parts: List[str], mm_parts: List[str]) -> bool:
        for item in mm_parts:
            raw_value = str(item).upper().removesuffix('MM')
            if not self._is_valid_mm_candidate(raw_value):
                return True
        for item in schedule_parts:
            if not self._is_valid_schedule_candidate(item):
                return True
        return False

    @staticmethod
    def _has_complex_composite_size(text: str) -> bool:
        patterns = [
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+\.\d+'),
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?(?:\s*[xX×*]\s*\d+(?:\.\d+)?){2,}'),
            # DN 复合尺寸中，第二个 DN 后若直接连小数厚度，整体交给大模型。
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?\s*[xX×*/]\s*DN\s*\d+(?=\d+\.\d+\s*(?:MM|毫米))'),
            re.compile(r'(?i)(?:\bD|[φΦФф]|(?:\bOD))\s*\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?){2,}'),
            re.compile(r'(?i)(?:\bD|[φΦФф])\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?'),
        ]
        return any(pattern.search(text) for pattern in patterns)

    @staticmethod
    def _has_malformed_decimal_chain(text: str) -> bool:
        """
        识别明显脏小数串：
        - 6.31.5
        - 12.70.31mm
        只要出现“数字.数字.”这种结构，壁厚规则直接失效。
        """
        patterns = (
            re.compile(r'\d+\.\d+\.\d+'),
            re.compile(r'\d+\.\d+\.(?=[A-Za-z\u4e00-\u9fff]|$)'),
        )
        return any(p.search(text) for p in patterns)

    @staticmethod
    def _has_parenthesized_thickness_variant(text: str) -> bool:
        """
        识别主壁厚(次壁厚) 写法，规则层直接放弃：
        - 114x4.0(3.0)
        - THK=8.0(6.0)mm
        - 12.7mm(10.3mm)
        """
        patterns = (
            re.compile(r'(?i)\d+(?:\.\d+)?\s*(?:MM|毫米)?\s*\(\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?\s*\)'),
            re.compile(r'(?i)(?:THK|T|S|壁厚)\s*[:：=]?\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?\s*\(\s*\d+(?:\.\d+)?\s*(?:MM|毫米)?\s*\)'),
        )
        return any(p.search(text) for p in patterns)

    @staticmethod
    def _has_phi_dual_od_dual_thk_structure(text: str) -> bool:
        pattern = re.compile(
            r'(?i)[φΦФф]\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(?:[φΦФф]\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)'
        )
        for m in pattern.finditer(text):
            a = float(m.group(1))
            b = float(m.group(2))
            t1 = float(m.group(3))
            t2 = float(m.group(4))
            if t1 <= a * 0.12 and t2 <= b * 0.12:
                return True
        return False

    @staticmethod
    def _extract_bare_od_thickness_blocks(text: str) -> List[Dict[str, Any]]:
        pattern = re.compile(
            r'(?<![A-Za-z0-9])'
            r'(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+\.\d+)'
            r'(?!\s*")\s*(?:MM|毫米)?(?:[A-Z]{1,6})?'
            r'(?=$|[^A-Za-z0-9])',
            re.IGNORECASE,
        )
        blocks: List[Dict[str, Any]] = []
        for m in pattern.finditer(text):
            # 不能从前一个小数的尾部起跳：
            # 168.3x7.11 不应截成 3x7.11
            if m.start() >= 2 and text[m.start() - 1] == '.' and text[m.start() - 2].isdigit():
                continue
            try:
                od = float(m.group(1))
                thk = float(m.group(2))
            except ValueError:
                continue
            blocks.append({
                "od": od,
                "thickness": thk,
                "raw": m.group(0).strip(),
                "span": (m.start(), m.end()),
            })
        return blocks

    def _process_structured_items(self, items: Any) -> str:
        if not isinstance(items, list):
            return ""

        parts: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subtype = str(item.get("type", "")).strip().upper()
            raw_value = item.get("value")
            if raw_value in (None, ""):
                continue
            normalized = self._normalize_structured_part(subtype, raw_value)
            if normalized and normalized not in parts:
                parts.append(normalized)
        return 'X'.join(parts)

    def _normalize_structured_part(self, subtype: str, item: Any) -> str:
        if isinstance(item, dict):
            item = item.get("value")
        if item in (None, ""):
            return ""

        text = str(item).strip()
        if not text:
            return ""

        if subtype == "MM":
            slash_group = self._normalize_layered_mm_group(text)
            if slash_group:
                return slash_group

            # 兼容 LLM 子类型误标：如 MM: SCH20 / MM: STD。
            # 这类值应走通用壁厚规则，不能强行转成 20MM。
            upper = text.upper()
            if (
                "SCH" in upper
                or upper in self.SPECIAL_VALUES
                or re.search(r'(^|[^A-Z])\d+(?:\.\d+)?S($|[^A-Z])', upper)
                or upper.startswith("S-")
                or re.match(r'^S\d', upper)
            ):
                return self.process(text)

            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if not match:
                return ""
            return f"{self._normalize_number(match.group(1))}MM"

        if subtype == "INCH":
            normalized = text.replace('”', '"').replace('“', '"').replace('″', '"')
            normalized = re.sub(r'\s+', '', normalized)
            return normalized.upper()

        if subtype in ("SCHEDULE", "SERIES"):
            return self.process(text)

        if subtype == "BWG":
            return text.upper()

        return self.process(text)

    def _normalize_layered_mm_group(self, text: str) -> str:
        parts = [part.strip() for part in text.split('/') if part.strip()]
        if len(parts) < 2:
            return ""

        normalized: List[str] = []
        for part in parts:
            if re.search(r'[xX*×,]', part):
                return ""
            match = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*MM)?', part, flags=re.IGNORECASE)
            if not match:
                return ""
            normalized.append(f"{self._normalize_number(match.group(1))}MM")

        return '/'.join(normalized)

    def _preprocess(self, value: str) -> str:
        """
        预处理：仅清理壁厚 token 的局部前缀/后缀和脏字符。
        不做全文删空格，避免破坏字段边界。
        """
        result = value

        result = self._normalize_known_aliases(result)

        # 先去掉裸露的大小端标记（如 "SCH40 L x SCH80 S"）。
        # 这一类 L/S 只是位置标识，不是壁厚值本体。
        result = self.SUFFIX_PATTERN.sub('', result)

        result = result.replace(' ', '')
        # 去掉 II-/Ⅱ- 前缀（只表示系列，不是壁厚本体）。
        result = re.sub(r'(?i)(?:^|(?<=[xX*×/,]))(?:II|Ⅱ)-', '', result)

        # 去除所有位置的 T=/THK= 前缀（开头和分隔符后的）。
        result = re.sub(r'(?:^|(?<=[xX*×/,]))(?:T|THK)\s*=\s*', '', result, flags=re.IGNORECASE)

        # 容错处理：去除单独的 = 号（NER 可能漏掉 T，只识别出 =4.0mm）。
        result = re.sub(r'(?:^|(?<=[xX*×/,]))=\s*(?=\d)', '', result, flags=re.IGNORECASE)

        # 处理 thk 格式：thk3X6 → 3X6, thk3Xthk6 → 3X6
        result = re.sub(r'(?:^|(?<=[xX*×/,]))(?:THK|Thk|thk)', '', result, flags=re.IGNORECASE)

        # 去除 (L)/(S) 后缀。
        result = re.sub(r'\([LS]\)', '', result, flags=re.IGNORECASE)

        # 去除无效字符（如单引号、反引号等）。
        result = re.sub(r"[''`]", '', result)

        return result

    def _normalize_known_aliases(self, value: str) -> str:
        """处理映射表中已知的少量脏别名。"""
        compact = re.sub(r'\s+', '', value or '').upper()
        return self.UNDERSCORE_ALIASES.get(compact, value)
    
    def _split_parts(self, value: str) -> List[str]:
        """
        按分隔符分割异径规格
        
        注意：需要智能分割，避免把 XXS 中的 X 当作分隔符
        分隔符特征：X 两边都有数字或壁厚值
        """
        # 特殊值直接返回，不分割
        if value.upper() in self.SPECIAL_VALUES:
            return [value]
        
        # 先检查是否是 SCH 开头的特殊值（如 SCHXXS, SCHXS）
        sch_special = re.match(r'^SCH\s*(XXS|XS|STD)$', value, re.IGNORECASE)
        if sch_special:
            return [value]
        
        # 智能分割：只在有效分隔位置分割
        # 有效分隔：分隔符前后都是完整的壁厚值
        # 模式：值 + 分隔符 + 值
        # 值的模式：SCH\d+S?|S-?\d+S?|S?-?XXS|S?-?XS|STD|\d+(\.\d+)?\s*mm|\d+(\.\d+)?
        
        # 使用更精确的分割模式
        # 匹配异径分隔符（前后有数字或字母）
        split_pattern = re.compile(
            r'(?<=[0-9SsDdMm])\s*[xX*×/,]\s*(?=[0-9SsTtHhCcXx])',
            re.IGNORECASE
        )
        
        parts = split_pattern.split(value)
        # 过滤空字符串
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 1:
            concatenated_parts = self._split_concatenated_tokens(value)
            if len(concatenated_parts) > 1:
                parts = concatenated_parts
        
        # 处理省略前缀的情况：
        # SCH10/10 → SCH10/SCH10, S10/10 → S10/S10, S-10/10 → S-10/S-10
        # 如果第一个部分有壁厚前缀，后续纯数字部分应继承该前缀
        if len(parts) > 1:
            first_part = parts[0]
            
            # 匹配各种壁厚前缀格式
            # SCH10, SCH10S, S10, S10S, S-10, S-10S
            prefix_match = re.match(r'^(SCH|S-?)\s*(\d+S?)$', first_part, re.IGNORECASE)
            if prefix_match:
                prefix = prefix_match.group(1)  # SCH, S, S-
                for i in range(1, len(parts)):
                    # 如果是纯数字（可能带S后缀），补上前缀
                    if re.match(r'^(\d+S?)$', parts[i], re.IGNORECASE):
                        parts[i] = f"{prefix}{parts[i]}"
        
        return parts

    def _split_concatenated_tokens(self, value: str) -> List[str]:
        """
        尝试拆分无显式分隔符的连续壁厚 token。

        例如：
        - SCH20SCH40 -> ["SCH20", "SCH40"]
        - SCH20SCH40S -> ["SCH20", "SCH40S"]
        - S40S20 -> ["S40", "S20"]
        - 40S20S -> ["40S", "20S"]
        - S40SS20S -> ["S40S", "S20S"]

        仅当整串可以被完整切分为 2 段及以上合法 token 时才生效，
        避免把普通单值误拆。
        """
        if not value:
            return [value]

        upper = value.upper()
        if len(upper) < 4:
            return [value]

        @lru_cache(maxsize=None)
        def dfs(pos: int):
            if pos == len(upper):
                return []

            for candidate in self._iter_concatenated_candidates(upper, pos):
                rest = dfs(pos + len(candidate))
                if rest is not None:
                    return [candidate] + rest
            return None

        parts = dfs(0)
        return parts if parts and len(parts) > 1 else [value]

    def _iter_concatenated_candidates(self, value: str, pos: int) -> List[str]:
        """返回当前位置可能的连续壁厚 token，按更长优先。"""
        candidates: List[str] = []
        rest = value[pos:]

        if rest.startswith('SCH'):
            body = rest[3:]
            for special in ('XXS', 'XS', 'STD'):
                token = f'SCH{special}'
                if rest.startswith(token):
                    candidates.append(token)
            num_match = re.match(r'SCH(\d+(?:\.\d+)?)', rest)
            if num_match:
                base = num_match.group(0)
                if rest.startswith(base + 'S'):
                    candidates.append(base + 'S')
                candidates.append(base)

        if rest.startswith('S-'):
            for special in ('XXS', 'XS', 'STD'):
                token = f'S-{special}'
                if rest.startswith(token):
                    candidates.append(token)
            decimal_mm_match = re.match(r'S-(\d+\.\d+)(MM|毫米)?', rest)
            if decimal_mm_match:
                candidates.append(decimal_mm_match.group(0))
            else:
                num_match = re.match(r'S-(\d+)', rest)
                if num_match:
                    base = num_match.group(0)
                    if rest.startswith(base + 'S'):
                        candidates.append(base + 'S')
                    candidates.append(base)

        if rest.startswith('S'):
            num_match = re.match(r'S(\d+(?:\.\d+)?)', rest)
            if num_match:
                base = num_match.group(0)
                if rest.startswith(base + 'S'):
                    candidates.append(base + 'S')
                candidates.append(base)

        num_s_match = re.match(r'(\d+(?:\.\d+)?)S', rest)
        if num_s_match:
            candidates.append(num_s_match.group(0))

        mm_match = re.match(r'(\d+(?:\.\d+)?)MM', rest)
        if mm_match:
            candidates.append(mm_match.group(0))

        for special in ('XXS', 'XS', 'STD'):
            if rest.startswith(special):
                candidates.append(special)

        # 去重并按长度倒序，优先尝试更完整 token，再回溯处理歧义情况。
        unique = []
        seen = set()
        for token in sorted(candidates, key=len, reverse=True):
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique
    
    def _convert_single(self, value: str) -> str:
        """
        转换单个壁厚值
        """
        if not value:
            return ""
        
        value = value.strip()
        upper_value = value.upper()
        
        # 检查是否是特殊值（XXS, XS, STD）- 精确匹配
        if upper_value in self.SPECIAL_VALUES:
            return upper_value
        
        # 尝试 SCH 格式：SCH80 → S80, SCH40S → S40S, SCHXXS → XXS
        sch_match = self.SCH_PATTERN.fullmatch(value)
        if sch_match:
            suffix = sch_match.group(1).upper()
            # 特殊值不加 S 前缀
            if suffix in self.SPECIAL_VALUES:
                return suffix
            return f"S{suffix}"

        s_dash_mm_match = self.S_DASH_MM_PATTERN.fullmatch(value)
        if s_dash_mm_match:
            num = self._normalize_number(s_dash_mm_match.group(1))
            return f"{num}MM"

        # 尝试 S- 格式：S-80 → S80, S-10S → S10S, S-XS → XS
        s_dash_match = self.S_DASH_PATTERN.fullmatch(value)
        if s_dash_match:
            suffix = s_dash_match.group(1).upper()
            if suffix in self.SPECIAL_VALUES:
                return suffix
            return f"S{suffix}"
        
        # 尝试 S 格式（已经是 S 开头）：S80 → S80, S10S → S10S, S6.3 → S6.3
        s_match = re.match(r'^S(\d+(?:\.\d+)?S?)', value, re.IGNORECASE)
        if s_match:
            suffix = s_match.group(1).upper()
            # 如果是纯数字（可能有小数），保留
            return f"S{suffix}"
        
        # 尝试纯数字+S 格式：40S → S40S, 10S → S10S, 80S → S80S
        num_s_match = re.match(r'^(\d+)(S)$', value, re.IGNORECASE)
        if num_s_match:
            num = num_s_match.group(1)
            return f"S{num}S"
        
        # 尝试 mm 格式：8.1mm → 8.1MM, 22.0mm → 22MM
        mm_match = self.MM_PATTERN.match(value)
        if mm_match:
            num = mm_match.group(1)
            # 去除小数点后的 .0
            num = self._normalize_number(num)
            return f"{num}MM"
        
        # 尝试 Thk 格式：Thk4.0 → 4MM, THK=6.3 → 6.3MM
        thk_match = re.match(r'^THK\s*[=]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?$', value, re.IGNORECASE)
        if thk_match:
            num = self._normalize_number(thk_match.group(1))
            return f"{num}MM"
        
        # 尝试纯数字格式：6.3 → 6.3MM
        num_match = re.match(r'^(\d+(?:\.\d+)?)$', value)
        if num_match:
            num = self._normalize_number(num_match.group(1))
            return f"{num}MM"
        
        # 无法识别，返回大写原值
        return upper_value
    
    def _normalize_number(self, num_str: str) -> str:
        """
        规范化数字：去除尾部的 .0
        """
        try:
            num = float(num_str)
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return num_str
    
    def _merge_parts(self, parts: List[str]) -> str:
        """
        合并异径规格部分
        
        规则：
        - 如果两部分相同，只输出一个
        - 如果不同，用 X 连接
        """
        if not parts:
            return ""
        
        if len(parts) == 1:
            return parts[0]
        
        # 去重（相邻相同的合并）
        unique_parts = [parts[0]]
        for p in parts[1:]:
            if p != unique_parts[-1]:
                unique_parts.append(p)
        
        if len(unique_parts) == 1:
            return unique_parts[0]
        
        return 'X'.join(unique_parts)


# 单例
_processor_instance: Optional[ThicknessProcessor] = None


@dataclass
class RuleThicknessExtraction:
    schedule: List[str]
    mm: List[str]
    thickness_code: str
    matched_texts: List[str]
    matched_spans: List[Tuple[int, int]]
    consumed_spans: List[Tuple[int, int]] = field(default_factory=list)
    ordered_items: List[Dict[str, str]] = field(default_factory=list)


def get_thickness_processor() -> ThicknessProcessor:
    """获取壁厚处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ThicknessProcessor()
    return _processor_instance


# 测试代码
if __name__ == '__main__':
    processor = ThicknessProcessor()
    
    # 测试用例（输入, 期望输出）
    test_cases = [
        # 基础格式
        ("S-80", "S80"),
        ("S-10S", "S10S"),
        ("SCH80", "S80"),
        ("Sch40", "S40"),
        ("SCH40S", "S40S"),
        ("SCH 40S", "S40S"),
        ("SCH5S(L)", "S5S"),
        ("SCH10(L)", "S10"),
        ("SCHXXS", "XXS"),
        ("XXS", "XXS"),
        ("SCHXS", "XS"),
        ("XS", "XS"),
        ("SCHSTD", "STD"),
        ("STD", "STD"),
        
        # 异径格式
        ("Sch40XSch80", "S40XS80"),
        ("SCH80XSCH80", "S80"),
        ("SCH80XSCH80S", "S80XS80S"),
        ("STDXSTD", "STD"),
        ("STDXSCH20", "STDXS20"),
        ("S-10S x S-40S", "S10SXS40S"),
        ("S-60 x S-XS", "S60XXS"),
        ("SCH60(L)*SCH80(S)", "S60XS80"),
        ("SCH40 L x SCH80 S", "S40XS80"),
        ("SCH10S(L)×SCH40S(S)", "S10SXS40S"),
        ("STD(L)*SCH30(S)", "STDXS30"),
        ("SCH20(L)'×SCH40(S)", "S20XS40"),
        ("Sch 20 Sch 40", "S20XS40"),
        ("SCH20SCH40", "S20XS40"),
        ("SCH20SCH40S", "S20XS40S"),
        ("S40S20", "S40XS20"),
        ("40s20S", "S40SXS20S"),
        ("S40SS20S", "S40SXS20S"),
        ("S40S40S", "S40S"),
        ("S-40S, S-80S", "S40SXS80S"),
        ("Sch20_S", "S20S"),
        ("Sch40S_S", "S40SXS"),
        ("Sch40_S", "S40XS"),
        ("Sch40_STD", "S40XSTD"),
        ("STD_STD", "STD"),
        ("STD_Sch2", "STDXS2"),
        ("Sch20_STD", "STDXS20"),
        ("XS_XS", "XS"),
        
        # mm 格式
        ("8.1mm", "8.1MM"),
        ("4mm/16.6mm", "4MMX16.6MM"),
        ("3mm/2.4mm", "3MMX2.4MM"),
        ("8.1mmX5.6mm", "8.1MMX5.6MM"),
        ("22.0mmx20", "22MMX20MM"),
        ("4.5mmX4.5mm", "4.5MM"),
        
        # T=/THK= 格式
        ("T=3.0mm", "3MM"),
        ("T=3.5x3mm", "3.5MMX3MM"),
        ("THK=5.6mmX4.0mm", "5.6MMX4MM"),
        ("THK=6.3mmX6.3mm", "6.3MM"),
        ("THK=6.0X4.5mm", "6MMX4.5MM"),
        ("THK=3.0X3.0mm", "3MM"),
        ("II-3/2.5", "3MMX2.5MM"),
        ("Ⅱ-4×3", "4MMX3MM"),
        ("II-2.5/2.5", "2.5MM"),
        ("Ⅱ-3", "3MM"),
        
        # 混合格式
        ("S6.3XS6.3", "S6.3"),
        ("11.91 mm x S-40", "11.91MMXS40"),
    ]
    
    print("=== 壁厚处理器测试 ===")
    passed = 0
    failed = 0
    
    for input_val, expected in test_cases:
        result = processor.process(input_val)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{input_val}' → '{result}' (期望: '{expected}')")
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
