# -*- coding: utf-8 -*-
"""
LLM 管道材料编码器
使用 Qwen3 大语言模型进行编码
"""

import logging
import re
from typing import Any, Dict, List
from pathlib import Path

from .pipe_encoder import PipeEncoderBase, FieldEncoding
from ..llm_ner.predictor import Qwen3Predictor

logger = logging.getLogger(__name__)


class LlmPipeEncoder(PipeEncoderBase):
    LLM_DIRECT_SIMILARITY = 0.90
    RULE_FALLBACK_SIMILARITY = 0.75
    NORMALIZED_FALLBACK_SIMILARITY = 0.80

    def __init__(self):
        super().__init__()

        encoding_config = self.platform_config.get('encoding', {})
        llm_config = encoding_config.get('llm', {})
        backend = llm_config.get('backend', 'ollama')

        if backend == 'ollama':
            self.llm_encoder = Qwen3Predictor(
                model_name=llm_config.get('model_name', 'qwen3-pipe'),
                backend='ollama',
                ollama_url=llm_config.get('ollama_url', 'http://localhost:11434'),
            )
        else:
            model_path = llm_config.get('model_path', 'outputs/qwen3_finetune/merged')
            if not Path(model_path).is_absolute():
                model_path = str(Path(__file__).parent.parent.parent / model_path)
            self.llm_encoder = Qwen3Predictor(
                model_path=model_path,
                backend='transformers',
            )
        logger.info(f"编码方法: LLM ({backend})")
        self.backend = backend

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

    def _encode_type_value(self, merged_value: str):
        code, similarity, _ = self._encode_with_llm_meta('TYPE', merged_value)
        confidence = similarity if code else 0.0
        return code, confidence

    def _encode_size_multi(self, values: List[Any]) -> FieldEncoding:
        merged, size_need_review = self.size_processor.process_multi_with_review(values)
        display_values = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]
        code, sim, _ = self._encode_with_llm_meta('SIZE', merged) if merged else ("", 0.0, False)
        return FieldEncoding(
            field_type='SIZE',
            original_value=' | '.join(display_values),
            original_values=display_values,
            matched_name=merged, matched_names=[merged],
            code=code, codes=[code] if code else [],
            similarity=sim if code else 0.0, is_exact_match=True, need_review=size_need_review or not bool(code),
            candidates=[], display='', items=[]
        )

    def _encode_thickness_value(self, value: Any, original_text: str = "") -> str:
        normalized = self.thickness_processor.process(value, original_text=original_text)
        if not normalized:
            return ""
        return self._encode_with_llm('THICKNESS', normalized) or normalized

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
                return {
                    'original': value, 'matched': '', 'code': '',
                    'similarity': 0.0, 'is_exact': True, 'need_review': True, 'candidates': []
                }
            # 组合壁厚（如 16MMXS60）拆开分别编码再拼回
            parts = self._split_thickness_parts(normalized)
            if len(parts) > 1:
                encoded_parts = []
                any_fallback = False
                part_sims = []
                for p in parts:
                    enc, part_sim, _ = self._encode_with_llm_meta(field_type, p)
                    if enc:
                        encoded_parts.append(enc)
                        part_sims.append(part_sim)
                    else:
                        encoded_parts.append(p)
                        any_fallback = True
                code = 'X'.join(encoded_parts)
                similarity = min(part_sims) if part_sims else 0.0
                if any_fallback:
                    similarity = min(similarity or 1.0, self.NORMALIZED_FALLBACK_SIMILARITY)
            else:
                encoded, model_sim, _ = self._encode_with_llm_meta(field_type, normalized)
                if encoded:
                    code = encoded
                    similarity = model_sim
                else:
                    code = normalized
                    similarity = self.NORMALIZED_FALLBACK_SIMILARITY
            return {
                'original': value, 'matched': normalized, 'code': code,
                'similarity': similarity if code else 0.0,
                'is_exact': True,
                'need_review': not code,
                'candidates': []
            }

        if field_type in ('SIZE', 'PRESSURE', 'MATERIAL', 'TYPE'):
            code, sim, _ = self._encode_with_llm_meta(field_type, value)
            return {
                'original': value, 'matched': value, 'code': code,
                'similarity': sim if code else 0.0,
                'is_exact': True,
                'need_review': not code,
                'candidates': []
            }

        if field_type in self.exact_match_fields:
            match_result = self.matcher.match(field_type, value, use_semantic=False)
            return {
                'original': value,
                'matched': match_result.matched_name,
                'code': match_result.code,
                'similarity': match_result.similarity if match_result.code else 1.0,
                'is_exact': match_result.is_exact_match,
                'need_review': False,
                'candidates': [{'name': c[0], 'code': c[1], 'similarity': c[2]}
                               for c in (match_result.candidates or [])]
            }

        return {'original': value, 'matched': value, 'code': value,
                'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

    def _process_standard_multi(self, values: List[str], modifier_map: Dict[int, Dict[str, List[str]]] = None) -> FieldEncoding:
        if not values:
            return FieldEncoding(field_type='STANDARD')

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

        production_items = []
        manufacturing_items = []
        unknown_items = []

        for std in expanded:
            formatted = sp._format_standard(std)
            category = sp._classify_standard(formatted)

            encoded, item_similarity, _ = self._encode_with_llm_meta('STANDARD', formatted)
            if not encoded:
                encoded = sp._encode_standard(formatted)
                item_similarity = self.RULE_FALLBACK_SIMILARITY
                logger.warning(f"[STANDARD] LLM编码失败，回退规则: '{std}' -> '{encoded}'")

            item = (std, formatted, encoded, item_similarity)
            if category == 'production':
                production_items.append(item)
            elif category == 'manufacturing':
                manufacturing_items.append(item)
            else:
                unknown_items.append(item)

        production_items = sp._sort_items(production_items)
        manufacturing_items = sp._sort_items(manufacturing_items)
        unknown_items = sp._sort_items(unknown_items)

        items = []
        all_encoded = []

        for item_list, category_label in [
            (production_items, '生产'),
            (manufacturing_items, '制造'),
            (unknown_items, ''),
        ]:
            for std, formatted, encoded, item_similarity in item_list:
                structured = sp.parse_standard_structure(std)
                code_parts = sp._split_code_and_grade(encoded)
                items.append({
                    'original': std,
                    'matched': formatted,
                    'code': encoded,
                    'base_code': code_parts['base'],
                    'grade': structured['grade'] or code_parts['grade'],
                    'standard_subject': structured['subject'],
                    'standard_grade': structured['grade'],
                    'standard_method': structured['method'],
                    'standard_appendix': structured['appendix'],
                    'similarity': item_similarity, 'is_exact': True, 'need_review': False,
                    'candidates': [], 'category': category_label
                })
                if encoded:
                    all_encoded.append(encoded)

        unique_encoded = []
        seen = set()
        for e in all_encoded:
            if e not in seen:
                seen.add(e)
                unique_encoded.append(e)

        original_display = ' | '.join([item['original'] for item in items])
        display_parts = []
        for item in items:
            if item['code']:
                if item['category']:
                    display_parts.append(f"{item['code']}({item['category']})")
                else:
                    display_parts.append(item['code'])

        return FieldEncoding(
            field_type='STANDARD',
            original_value=original_display,
            original_values=values,
            matched_name=' '.join(display_parts) if display_parts else '',
            matched_names=[item['code'] for item in items],
            code=''.join(unique_encoded),
            codes=[item['code'] for item in items],
            similarity=min((item['similarity'] for item in items), default=1.0), is_exact_match=True, need_review=False,
            candidates=[],
            display=' '.join(display_parts) if display_parts else '无',
            items=items
        )
