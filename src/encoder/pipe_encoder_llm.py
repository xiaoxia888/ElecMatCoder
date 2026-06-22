# -*- coding: utf-8 -*-
"""
LLM 管道材料编码器
使用 Qwen3 大语言模型进行编码
"""

import logging
import re
from typing import Any, Dict, List

from .pipe_encoder import PipeEncoderBase, EncodedFieldResult
from .processors import get_type_encoder, get_material_encoder
from ..llm_ner.predictor import Qwen3Predictor

logger = logging.getLogger(__name__)


class LlmPipeEncoder(PipeEncoderBase):
    LLM_DIRECT_SIMILARITY = 0.90
    RULE_FALLBACK_SIMILARITY = 0.75
    NORMALIZED_FALLBACK_SIMILARITY = 0.80

    def __init__(self):
        super().__init__()
        self.type_encoder = get_type_encoder()
        self.material_encoder = get_material_encoder()
        self._last_type_encode_meta: Dict[str, Any] = {}

        encoding_config = self.platform_config.get('encoding', {})
        llm_config = encoding_config.get('llm', {})
        backend = str(llm_config.get('backend', 'ollama')).strip()
        if backend != 'mlx_service':
            raise RuntimeError(f"二阶段编码只支持 mlx_service 后端，当前配置为: {backend}")

        self.llm_encoder = Qwen3Predictor(
            model_name=llm_config.get('model_name', 'coder'),
            backend='mlx_service',
            service_url=llm_config.get('service_url', 'http://127.0.0.1:8200'),
            ollama_num_predict=llm_config.get('num_predict', 512),
            ollama_temperature=llm_config.get('temperature', 0.0),
            ollama_top_p=llm_config.get('top_p', 1.0),
            ollama_logprobs_enabled=False,
            request_timeout=llm_config.get('timeout', 300),
            stage2_system_prompt=llm_config.get('instruction'),
        )
        logger.info(f"编码方法: LLM ({backend})")
        self.backend = backend
        self.fallback_fields = {
            str(field).strip().upper()
            for field in (llm_config.get('fallback_fields') or [])
            if str(field).strip()
        }

    def _make_encode_meta(self, source: str, confidence: float, reason: str, evidence: Dict[str, Any] | None = None):
        return self._build_processor_encode_confidence(
            source=source,
            confidence=confidence,
            reason=reason,
            evidence=evidence or {},
        )

    def _allow_llm_fallback(self, field_type: str) -> bool:
        return str(field_type or '').strip().upper() in self.fallback_fields

    def _build_fallback_input_text(self, values: List[Any]) -> str:
        parts = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]
        return ' ; '.join(parts)

    def _score_from_model_conf(self, model_conf: float) -> float:
        """将模型 token 级概率映射到字段置信度分数。"""
        conf = max(0.0, min(1.0, float(model_conf)))
        # 拉伸低区间，避免分数过低导致全部待审
        return round(0.5 + 0.5 * conf, 4)

    def _is_known_code(self, field_type: str, code: str) -> bool:
        mapping_data = self.matcher.mapping.get(field_type, {})
        code_upper = (code or "").strip().upper()
        if not code_upper:
            return False
        for _, info in mapping_data.items():
            if isinstance(info, dict):
                c = str(info.get('code', '')).strip().upper()
            else:
                c = str(info).strip().upper()
            if c and c == code_upper:
                return True
        return False

    def _score_ollama_encoding(self, field_type: str, value: str, code: str) -> float:
        """
        ollama 无 logprob 时的字段级编码置信度（启发式，不是常量）。
        """
        if not code:
            return 0.0
        v = (value or "").strip().upper()
        c = (code or "").strip().upper()
        score = 0.68

        # 全局可读性信号
        if len(c) <= 2:
            score -= 0.08
        elif len(c) <= 8:
            score += 0.05
        else:
            score += 0.02

        if field_type == 'TYPE':
            if self._is_known_code('TYPE', c):
                score += 0.16
            if '90' in v and '90' in c:
                score += 0.04

        elif field_type == 'MATERIAL':
            if self._is_known_code('MATERIAL', c):
                score += 0.14
            if '+' in v and '+' in c:
                score += 0.03
            if ('GR' in v or '#' in v) and ('GR' in c or '#' in c):
                score += 0.03

        elif field_type == 'SIZE':
            has_pair = bool(re.search(r'[X×]', v))
            code_pair = bool(re.search(r'[X×]', c))
            if has_pair == code_pair:
                score += 0.10
            if re.match(r'^\d+(?:\.\d+)?(?:[X×]\d+(?:\.\d+)?)*$', c):
                score += 0.10
            if has_pair and c.count('X') != v.replace('×', 'X').count('X'):
                score -= 0.08

        elif field_type == 'THICKNESS':
            value_parts = [p for p in re.split(r'[X×*/]+', v.replace(' ', '')) if p]
            code_parts = [p for p in re.split(r'[X×*/]+', c.replace(' ', '')) if p]
            if len(value_parts) == len(code_parts):
                score += 0.08
            if any('MM' in p for p in code_parts) or any('S' in p for p in code_parts):
                score += 0.07
            if re.search(r'XXS|XS|STD|SCH|S\d+', c):
                score += 0.05

        elif field_type == 'PRESSURE':
            if re.search(r'^(PN\s*)?\d+|C\d+|CL\d+', c):
                score += 0.15
            if ('PN' in v and 'PN' in c) or ('LB' in v and ('C' in c or 'CL' in c)):
                score += 0.05

        elif field_type == 'STANDARD':
            if re.search(r'^[A-Z0-9]+$', c):
                score += 0.08
            if self._is_known_code('STANDARD', c):
                score += 0.12
            if 'GB/T' in v and c.startswith('GBT'):
                score += 0.05

        return float(max(0.35, min(0.95, score)))

    def _encode_with_llm_meta(self, field_type: str, value: str):
        """
        返回 (code, similarity, used_model_confidence)。
        - transformers: 使用真实生成概率
        - ollama: 无原生概率，回退到经验分
        """
        if not self.llm_encoder or not value:
            return "", 0.0, False
        try:
            codes, confs = self.llm_encoder.encode_with_confidence({field_type: value})
            code = codes.get(field_type, "")
            if isinstance(code, list):
                code = code[0] if code else ""
            model_conf = confs.get(field_type)
            used_model_conf = model_conf is not None
            if code:
                if used_model_conf:
                    similarity = self._score_from_model_conf(model_conf)
                else:
                    similarity = self._score_ollama_encoding(field_type, value, code)
            else:
                similarity = 0.0
            logger.info(
                f"[LLM编码] {field_type}: '{value}' -> '{code}', "
                f"model_conf={model_conf}, similarity={similarity:.3f}"
            )
            return (str(code) if code else ""), similarity, used_model_conf
        except Exception as e:
            logger.warning(f"[LLM编码] {field_type} 编码失败: {e}")
            return "", 0.0, False

    def _encode_with_llm(self, field_type: str, value: str) -> str:
        code, _, _ = self._encode_with_llm_meta(field_type, value)
        return code

    def _should_use_type_combined(self) -> bool:
        return bool(self.llm_encoder)

    def _encode_type_value(self, merged_value: str, type_value: Dict[str, Any] | None = None):
        self._last_type_encode_meta = {}
        if type_value:
            type_result = self.type_encoder.encode(type_value)
            if type_result.resolved and type_result.code:
                logger.info(
                    "[TYPE编码器] %s -> %s, strategy=%s, key=%s",
                    type_value,
                    type_result.code,
                    type_result.strategy,
                    type_result.matched_key,
                )
                self._last_type_encode_meta = self._make_encode_meta(
                    source='type_mapping',
                    confidence=0.98,
                    reason='type_mapping_resolved',
                    evidence={
                        'strategy': str(type_result.strategy or ''),
                        'matched_key': str(type_result.matched_key or ''),
                        'body_present': bool((type_value or {}).get('BODY')),
                    },
                )
                return type_result.code, 0.995
            logger.info(
                "[TYPE编码器] unresolved, fallback to LLM. merged='%s', type_value=%s",
                merged_value,
                type_value,
            )

        if not self._allow_llm_fallback('TYPE'):
            self._last_type_encode_meta = self._make_encode_meta(
                source='fallback_disabled',
                confidence=0.0,
                reason='type_mapping_unresolved_fallback_disabled',
                evidence={'field_type': 'TYPE'},
            )
            return "", 0.0

        code, similarity, used_model_conf = self._encode_with_llm_meta('TYPE', merged_value)
        self._last_type_encode_meta = self._make_encode_meta(
            source='llm_fallback',
            confidence=similarity if code else 0.0,
            reason='type_mapping_unresolved_llm_used' if code else 'type_mapping_unresolved_llm_failed',
            evidence={
                'code_present': bool(code),
                'used_model_confidence': bool(used_model_conf),
                'merged_value_present': bool(merged_value),
            },
        )
        confidence = similarity if code else 0.0
        return code, confidence

    def _process_material_item_structured(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        material_result = self.material_encoder.encode(item)
        if material_result.resolved and material_result.code:
            logger.info(
                "[MATERIAL编码器] %s -> %s, strategy=%s, base=%s, suffixes=%s",
                item,
                material_result.code,
                material_result.strategy,
                material_result.matched_code,
                material_result.matched_suffixes,
            )
            original = self._flatten_material_item_for_stage2(item)
            matched_parts = [material_result.value, *material_result.special_req]
            return {
                'original': original,
                'matched': ' '.join([p for p in matched_parts if p]).strip() or original,
                'code': material_result.code,
                'similarity': 0.995,
                'encode_meta': self._make_encode_meta(
                    source='material_mapping',
                    confidence=0.98,
                    reason='material_mapping_resolved',
                    evidence={
                        'strategy': str(material_result.strategy or ''),
                        'base_code': str(material_result.matched_code or ''),
                        'suffix_count': len(material_result.matched_suffixes or []),
                    },
                ),
                'is_exact': True,
                'need_review': False,
                'candidates': [],
            }
        logger.info(
            "[MATERIAL编码器] unresolved, fallback to LLM. item=%s, reason=%s",
            item,
            material_result.reason,
        )
        if not self._allow_llm_fallback('MATERIAL'):
            return {
                'original': self._flatten_material_item_for_stage2(item),
                'matched': '',
                'code': '',
                'similarity': 0.0,
                'encode_meta': self._make_encode_meta(
                    source='fallback_disabled',
                    confidence=0.0,
                    reason='material_mapping_unresolved_fallback_disabled',
                    evidence={'field_type': 'MATERIAL'},
                ),
                'is_exact': True,
                'need_review': True,
                'candidates': [],
            }
        return None

    def _encode_size_multi(self, values: List[Any], original_text: str = "") -> EncodedFieldResult:
        display_values = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]
        merged, size_need_review = self.size_processor.process_multi_with_review(values, original_text=original_text)
        normalized_merged = merged
        final_code = merged
        sim = 1.0 if final_code else 0.0

        encode_meta = self._make_encode_meta(
            source='size_processor',
            confidence=0.96 if final_code and not size_need_review else (0.72 if final_code else 0.0),
            reason='size_processor_resolved' if final_code else 'size_processor_failed',
            evidence={
                'item_count': len(display_values),
                'code_present': bool(final_code),
                'need_review': bool(size_need_review),
            },
        )

        if not final_code and self._allow_llm_fallback('SIZE'):
            fallback_input = self._build_fallback_input_text(values)
            code, sim, used_model_conf = self._encode_with_llm_meta('SIZE', fallback_input)
            if code:
                final_code = code
                size_need_review = False
                encode_meta = self._make_encode_meta(
                    source='llm_fallback',
                    confidence=sim,
                    reason='size_processor_failed_llm_used',
                    evidence={
                        'item_count': len(display_values),
                        'used_model_confidence': bool(used_model_conf),
                    },
                )
            else:
                encode_meta = self._make_encode_meta(
                    source='llm_fallback',
                    confidence=0.0,
                    reason='size_processor_failed_llm_failed',
                    evidence={'item_count': len(display_values)},
                )

        return EncodedFieldResult(
            field_type='SIZE',
            stage2_input=self._clone_response_value(values[0] if len(values) == 1 else values),
            encode_confidence_v2=encode_meta,
            code=final_code, codes=[final_code] if final_code else [],
            similarity=sim if final_code else 0.0,
            is_exact_match=True,
            need_review=size_need_review,
            candidates=[],
        )

    def _encode_thickness_value(self, value: Any, original_text: str = "") -> str:
        normalized = self.thickness_processor.process(value, original_text=original_text)
        if normalized:
            return normalized
        if not self._allow_llm_fallback('THICKNESS'):
            return ""
        fallback_input = self._stringify_field_value(value)
        code, _, _ = self._encode_with_llm_meta('THICKNESS', fallback_input)
        return code

    @staticmethod
    def _split_thickness_parts(normalized: str) -> List[str]:
        """
        智能拆分组合壁厚，避免把 XS/XXS 里的 X 当分隔符。
        例如:
          - 16MMXS60 -> ["16MM", "S60"]
          - S40SXS80S -> ["S40S", "S80S"]
          - S40XXS -> ["S40", "XS"]
        若无法完整解析，则返回原字符串（不拆分），保证不误拆。
        """
        if not normalized:
            return []

        s = normalized.strip().upper()
        token_re = re.compile(r'(XXS|XS|STD|S\d+(?:\.\d+)?S?|\d+(?:\.\d+)?MM)', re.IGNORECASE)
        parts: List[str] = []
        i = 0
        n = len(s)

        while i < n:
            m = token_re.match(s, i)
            if not m:
                return [s]
            parts.append(m.group(1).upper())
            i = m.end()
            if i >= n:
                break
            if s[i] in ('X', '×', '*', '/'):
                i += 1
                continue
            return [s]

        return parts if len(parts) > 1 else [s]

    def _process_single_value(self, field_type: str, value: str) -> dict:
        if not value:
            return {'original': '', 'matched': '', 'code': '',
                    'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

        if field_type == 'THICKNESS':
            normalized = self.thickness_processor.process(value)
            if not normalized:
                if self._allow_llm_fallback('THICKNESS'):
                    code, sim, used_model_conf = self._encode_with_llm_meta('THICKNESS', value)
                    return {
                        'original': value, 'matched': value if code else '', 'code': code,
                        'similarity': sim if code else 0.0,
                        'encode_meta': self._make_encode_meta(
                            source='llm_fallback',
                            confidence=sim if code else 0.0,
                            reason='thickness_processor_failed_llm_used' if code else 'thickness_processor_failed_llm_failed',
                            evidence={
                                'field_type': 'THICKNESS',
                                'used_model_confidence': bool(used_model_conf),
                            },
                        ),
                        'is_exact': True,
                        'need_review': not code,
                        'candidates': []
                    }
                return {
                    'original': value, 'matched': '', 'code': '',
                    'similarity': 0.0, 'is_exact': True, 'need_review': True, 'candidates': []
                }
            return {
                'original': value, 'matched': normalized, 'code': normalized,
                'similarity': 1.0 if normalized else 0.0,
                'encode_meta': self._make_encode_meta(
                    source='thickness_processor',
                    confidence=0.96 if normalized else 0.0,
                    reason='thickness_processor_resolved' if normalized else 'thickness_processor_failed',
                    evidence={'code_present': bool(normalized)},
                ),
                'is_exact': True,
                'need_review': False,
                'candidates': []
            }

        if field_type == 'PRESSURE':
            normalized = self._process_pressure(value)
            if not normalized:
                if self._allow_llm_fallback('PRESSURE'):
                    code, sim, used_model_conf = self._encode_with_llm_meta('PRESSURE', value)
                    return {
                        'original': value, 'matched': value if code else '', 'code': code,
                        'similarity': sim if code else 0.0,
                        'encode_meta': self._make_encode_meta(
                            source='llm_fallback',
                            confidence=sim if code else 0.0,
                            reason='pressure_processor_failed_llm_used' if code else 'pressure_processor_failed_llm_failed',
                            evidence={
                                'field_type': 'PRESSURE',
                                'used_model_confidence': bool(used_model_conf),
                            },
                        ),
                        'is_exact': True,
                        'need_review': not code,
                        'candidates': []
                    }
                return {
                    'original': value, 'matched': '', 'code': '',
                    'similarity': 0.0, 'is_exact': True, 'need_review': True, 'candidates': []
                }
            return {
                'original': value, 'matched': normalized, 'code': normalized,
                'similarity': 1.0,
                'encode_meta': self._make_encode_meta(
                    source='pressure_processor',
                    confidence=0.96,
                    reason='pressure_processor_resolved',
                    evidence={'code_present': bool(normalized)},
                ),
                'is_exact': True,
                'need_review': False,
                'candidates': []
            }

        if field_type == 'SIZE':
            normalized = self.size_processor.process(value)
            return {
                'original': value, 'matched': normalized, 'code': normalized,
                'similarity': 1.0 if normalized else 0.0,
                'encode_meta': self._make_encode_meta(
                    source='size_processor',
                    confidence=0.96 if normalized else 0.0,
                    reason='size_processor_resolved' if normalized else 'size_processor_failed',
                    evidence={'code_present': bool(normalized)},
                ),
                'is_exact': True,
                'need_review': not normalized,
                'candidates': []
            }

        if field_type in ('MATERIAL', 'TYPE'):
            if not self._allow_llm_fallback(field_type):
                return {
                    'original': value, 'matched': value, 'code': '',
                    'similarity': 0.0,
                    'encode_meta': self._make_encode_meta(
                        source='fallback_disabled',
                        confidence=0.0,
                        reason=f'{field_type.lower()}_fallback_disabled',
                        evidence={'field_type': field_type},
                    ),
                    'is_exact': True,
                    'need_review': True,
                    'candidates': []
                }
            code, sim, used_model_conf = self._encode_with_llm_meta(field_type, value)
            return {
                'original': value, 'matched': value, 'code': code,
                'similarity': sim if code else 0.0,
                'encode_meta': self._make_encode_meta(
                    source='llm_fallback',
                    confidence=sim if code else 0.0,
                    reason='llm_fallback_used' if code else 'llm_fallback_failed',
                    evidence={
                        'field_type': field_type,
                        'code_present': bool(code),
                        'used_model_confidence': bool(used_model_conf),
                    },
                ),
                'is_exact': True,
                'need_review': not code,
                'candidates': []
            }

        if field_type in self.exact_match_fields:
            match_result = self.matcher.match(field_type, value, use_semantic=False)
            return {
                'original': value,
                'matched': match_result.matched_value,
                'code': match_result.code,
                'similarity': match_result.similarity if match_result.code else 1.0,
                'is_exact': match_result.is_exact_match,
                'need_review': False,
                'candidates': [{'name': c[0], 'code': c[1], 'similarity': c[2]}
                               for c in (match_result.candidates or [])]
            }

        return {'original': value, 'matched': value, 'code': value,
                'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

    def _build_thickness_fallback_result(self, values: List[Any]) -> Dict[str, Any]:
        fallback_input = self._build_fallback_input_text(values)
        code, sim, used_model_conf = self._encode_with_llm_meta('THICKNESS', fallback_input)
        return {
            'code': code,
            'similarity': sim if code else 0.0,
            'encode_meta': self._make_encode_meta(
                source='llm_fallback',
                confidence=sim if code else 0.0,
                reason='thickness_processor_failed_llm_used' if code else 'thickness_processor_failed_llm_failed',
                evidence={
                    'field_type': 'THICKNESS',
                    'item_count': len(values),
                    'used_model_confidence': bool(used_model_conf),
                },
            ),
            'need_review': not code,
        }

    def _process_standard_multi(
        self,
        values: List[str],
        modifier_map: Dict[int, Dict[str, List[str]]] = None,
        original_text: str = "",
    ) -> EncodedFieldResult:
        if not values:
            return EncodedFieldResult(field_type='STANDARD')

        sp = self.standard_processor

        merged_standards = list(values)
        if modifier_map:
            for idx, modifier_info in modifier_map.items():
                if not (0 <= idx < len(merged_standards)):
                    continue
                suffix_parts = []
                for modifier_type in sp.MODIFIER_ORDER:
                    for raw_value in modifier_info.get(modifier_type, []) or []:
                        if raw_value:
                            suffix_parts.append(str(raw_value).strip())
                if suffix_parts:
                    merged_standards[idx] = f"{merged_standards[idx]} {' '.join(suffix_parts)}"

        expanded = sp._expand_slash_standards(merged_standards)
        resolved_items = []

        for std in expanded:
            formatted = sp._format_standard(std)
            category = sp._classify_standard(formatted)

            encoded = sp._encode_standard(formatted)
            if encoded:
                item_similarity = 1.0
                logger.info(
                    "[STANDARD编码器] '%s' -> '%s', strategy=standard_processor",
                    formatted,
                    encoded,
                )
                encode_meta = self._make_encode_meta(
                    source='standard_processor',
                    confidence=0.98,
                    reason='standard_processor_resolved',
                    evidence={
                        'formatted_present': bool(formatted),
                        'category': category,
                    },
                )
            else:
                if self._allow_llm_fallback('STANDARD'):
                    encoded, item_similarity, used_model_conf = self._encode_with_llm_meta('STANDARD', formatted)
                    if encoded:
                        logger.info(
                            "[STANDARD编码器] unresolved, fallback to LLM: '%s' -> '%s'",
                            formatted,
                            encoded,
                        )
                        encode_meta = self._make_encode_meta(
                            source='llm_fallback',
                            confidence=item_similarity,
                            reason='standard_processor_unresolved_llm_used',
                            evidence={
                                'formatted_present': bool(formatted),
                                'category': category,
                                'used_model_confidence': bool(used_model_conf),
                            },
                        )
                    else:
                        logger.warning(
                            "[STANDARD编码器] unresolved and LLM failed: '%s'",
                            formatted,
                        )
                        encode_meta = self._make_encode_meta(
                            source='llm_fallback',
                            confidence=0.0,
                            reason='standard_processor_unresolved_llm_failed',
                            evidence={
                                'formatted_present': bool(formatted),
                                'category': category,
                            },
                        )
                else:
                    encoded = ""
                    item_similarity = 0.0
                    encode_meta = self._make_encode_meta(
                        source='fallback_disabled',
                        confidence=0.0,
                        reason='standard_processor_unresolved_fallback_disabled',
                        evidence={
                            'formatted_present': bool(formatted),
                            'category': category,
                        },
                    )

            resolved_items.append({
                'original': std,
                'matched': formatted,
                'code': encoded,
                'category_key': category,
                'similarity': item_similarity,
                'encode_meta': encode_meta,
            })

        detail = sp.process_standards_with_detail(merged_standards, original_text=original_text)
        ordered_detail_items = detail.get('ordered_items', []) or []
        category_by_original = {}
        for detail_item in ordered_detail_items:
            original_key = str(detail_item.get('original', '') or '').strip()
            category_key = str(detail_item.get('category', '') or '').strip()
            if original_key and original_key not in category_by_original:
                category_by_original[original_key] = sp.CATEGORY_LABELS.get(category_key, '')
        detail_index = {}
        for item in ordered_detail_items:
            key = (item.get('original', ''), item.get('encoded', ''))
            detail_index.setdefault(key, []).append(item)

        items = []
        resolved_index = {}
        resolved_by_original = {}
        for item in resolved_items:
            key = (item.get('original', ''), item.get('code', ''))
            resolved_index.setdefault(key, []).append(item)
            resolved_by_original.setdefault(item.get('original', ''), []).append(item)

        for detail_item in ordered_detail_items:
            key = (detail_item.get('original', ''), detail_item.get('encoded', ''))
            resolved_item = (resolved_index.get(key) or resolved_by_original.get(detail_item.get('original', ''), []) or [{}]).pop(0)
            code = detail_item.get('encoded', '') or resolved_item.get('code', '')
            code_parts = sp._split_code_and_grade(code)
            items.append({
                'original': detail_item.get('original', ''),
                'matched': resolved_item.get('matched', detail_item.get('formatted', '')),
                'code': code,
                'base_code': detail_item.get('base_code', '') or code_parts.get('base', ''),
                'grade': detail_item.get('grade', '') or code_parts.get('grade', ''),
                'standard_subject': detail_item.get('standard_subject', ''),
                'standard_grade': detail_item.get('standard_grade', ''),
                'standard_method': detail_item.get('standard_method', ''),
                'standard_appendix': detail_item.get('standard_appendix', ''),
                'similarity': resolved_item.get('similarity', 1.0), 'is_exact': True, 'need_review': False,
                'candidates': [], 'category': sp.CATEGORY_LABELS.get(detail_item.get('category', 'unknown'), ''), 'encode_meta': resolved_item.get('encode_meta')
            })
        chosen_by_base = {}
        base_order = []
        for item in items:
            code = item.get('code', '')
            base_code = item.get('base_code', '')
            if not code or not base_code:
                continue
            if base_code not in chosen_by_base:
                chosen_by_base[base_code] = item
                base_order.append(base_code)
            elif item.get('grade') and not chosen_by_base[base_code].get('grade'):
                chosen_by_base[base_code] = item
        final_code = ''.join(chosen_by_base[base].get('code', '') for base in base_order)

        stage2_standard_inputs = []
        for idx, body in enumerate(values):
            body_text = str(body or '').strip()
            modifier_info = modifier_map.get(idx, {}) if isinstance(modifier_map, dict) else {}
            stage2_standard_inputs.append({
                'BODY': body_text,
                'GRADE': ' '.join(str(v).strip() for v in (modifier_info.get('STANDARD_GRADE') or []) if str(v).strip()),
                'APPENDIX': ' '.join(str(v).strip() for v in (modifier_info.get('STANDARD_APPENDIX') or []) if str(v).strip()),
                'METHOD': ' '.join(str(v).strip() for v in (modifier_info.get('STANDARD_METHOD') or []) if str(v).strip()),
                'CATEGORY': category_by_original.get(body_text, ''),
            })

        return EncodedFieldResult(
            field_type='STANDARD',
            stage2_input=stage2_standard_inputs,
            encode_confidence_v2=self._aggregate_item_encode_confidence(items, fallback_source='standard_processor'),
            code=final_code,
            codes=[item['code'] for item in items],
            similarity=min((item['similarity'] for item in items), default=1.0), is_exact_match=True, need_review=False,
            candidates=[],
            detail_items=items
        )
