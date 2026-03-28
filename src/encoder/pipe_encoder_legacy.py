# -*- coding: utf-8 -*-
"""
Legacy 管道材料编码器
使用 T5 Seq2Seq + 规则处理器进行编码
"""

import logging
from typing import Dict, List, Any
from pathlib import Path

from .pipe_encoder import PipeEncoderBase, FieldEncoding
from .seq2seq_encoder import get_seq2seq_encoder

logger = logging.getLogger(__name__)


class LegacyPipeEncoder(PipeEncoderBase):

    def __init__(self):
        super().__init__()

        encoding_config = self.platform_config.get('encoding', {})
        self.type_encoding_method = encoding_config.get('type_encoding', 'semantic')
        self.material_encoding_method = encoding_config.get('material_encoding', 'semantic')

        self.seq2seq_encoder = None
        if self.type_encoding_method == 'seq2seq' or self.material_encoding_method == 'seq2seq':
            seq2seq_config = encoding_config.get('seq2seq', {})
            model_path = seq2seq_config.get('model_path', 'outputs/type_seq2seq/final_model')
            device = seq2seq_config.get('device', 'auto')
            if not Path(model_path).is_absolute():
                model_path = str(Path(__file__).parent.parent.parent / model_path)
            self.seq2seq_encoder = get_seq2seq_encoder(model_path, device)
            logger.info(f"编码方法: Legacy (TYPE={self.type_encoding_method}, MATERIAL={self.material_encoding_method})")

    def _should_use_type_combined(self) -> bool:
        return (self.type_encoding_method == 'seq2seq'
                and self.seq2seq_encoder
                and self.seq2seq_encoder.is_available())

    def _encode_type_value(self, merged_value: str):
        result = self.seq2seq_encoder.encode(merged_value)
        logger.info(f"[Seq2Seq] TYPE (combined): '{merged_value}' -> code='{result.code}', conf={result.confidence:.2f}")
        return result.code, result.confidence

    def _encode_size_multi(self, values: List[Any], original_text: str = "") -> FieldEncoding:
        merged, need_review = self.size_processor.process_multi_with_review(values, original_text=original_text)
        display_values = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]
        return FieldEncoding(
            field_type='SIZE',
            original_value=' | '.join(display_values),
            original_values=display_values,
            matched_name=merged, matched_names=[merged],
            code=merged, codes=[merged] if merged else [],
            similarity=1.0, is_exact_match=True, need_review=need_review,
            candidates=[], display='', items=[]
        )

    def _encode_thickness_value(self, value: Any, original_text: str = "") -> str:
        return self.thickness_processor.process(value, original_text=original_text)

    def _process_single_value(self, field_type: str, value: str) -> dict:
        if not value:
            return {'original': '', 'matched': '', 'code': '',
                    'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

        if field_type == 'SIZE':
            processed = self.size_processor.process(value)
            return {'original': value, 'matched': processed, 'code': processed,
                    'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

        if field_type == 'STANDARD':
            detail = self.standard_processor.process_multi_with_detail([value])
            return {'original': value, 'matched': detail['encoded'], 'code': detail['encoded'],
                    'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': [],
                    'display': detail['display']}

        if field_type in self.passthrough_fields:
            if field_type == 'THICKNESS':
                processed = self._process_thickness(value)
            elif field_type == 'PRESSURE':
                processed = self._process_pressure(value)
            else:
                processed = value.strip().upper()
            return {'original': value, 'matched': processed, 'code': processed,
                    'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

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

        if field_type in self.semantic_match_fields:
            use_seq2seq = False
            if field_type == 'TYPE' and self.type_encoding_method == 'seq2seq':
                use_seq2seq = True
            elif field_type == 'MATERIAL' and self.material_encoding_method == 'seq2seq':
                use_seq2seq = True

            if use_seq2seq and self.seq2seq_encoder and self.seq2seq_encoder.is_available():
                result = self.seq2seq_encoder.encode(value)
                logger.info(f"[Seq2Seq] {field_type}: '{value}' -> code='{result.code}', conf={result.confidence:.2f}")
                return {
                    'original': value, 'matched': value, 'code': result.code,
                    'similarity': result.confidence, 'is_exact': True,
                    'need_review': not result.code, 'candidates': []
                }
            else:
                logger.info(f"[Semantic] {field_type}: '{value}' (回退到语义匹配)")
                match_result = self.matcher.semantic_match(field_type, value)
                return {
                    'original': value,
                    'matched': match_result.matched_name,
                    'code': match_result.code,
                    'similarity': match_result.similarity,
                    'is_exact': match_result.is_exact_match,
                    'need_review': match_result.need_review,
                    'candidates': [{'name': c[0], 'code': c[1], 'similarity': c[2]}
                                   for c in (match_result.candidates or [])]
                }

        return {'original': value, 'matched': value, 'code': value,
                'similarity': 1.0, 'is_exact': True, 'need_review': False, 'candidates': []}

    def _process_standard_multi(self, values: List[str], modifier_map: Dict[int, Dict[str, List[str]]] = None) -> FieldEncoding:
        if not values:
            return FieldEncoding(field_type='STANDARD')

        detail = self.standard_processor.process_with_modifiers(values, modifier_map)
        items = []

        for std, encoded in zip(detail.get('production', []), detail.get('production_encoded', [])):
            structured = self.standard_processor.parse_standard_structure(std)
            code_parts = self.standard_processor._split_code_and_grade(encoded)
            items.append({'original': std, 'matched': encoded, 'code': encoded,
                          'base_code': code_parts['base'], 'grade': structured['grade'] or code_parts['grade'],
                          'standard_subject': structured['subject'],
                          'standard_grade': structured['grade'],
                          'standard_method': structured['method'],
                          'standard_appendix': structured['appendix'],
                          'similarity': 1.0,
                          'is_exact': True, 'need_review': False, 'candidates': [], 'category': '生产'})

        for std, encoded in zip(detail.get('manufacturing', []), detail.get('manufacturing_encoded', [])):
            structured = self.standard_processor.parse_standard_structure(std)
            code_parts = self.standard_processor._split_code_and_grade(encoded)
            items.append({'original': std, 'matched': encoded, 'code': encoded,
                          'base_code': code_parts['base'], 'grade': structured['grade'] or code_parts['grade'],
                          'standard_subject': structured['subject'],
                          'standard_grade': structured['grade'],
                          'standard_method': structured['method'],
                          'standard_appendix': structured['appendix'],
                          'similarity': 1.0,
                          'is_exact': True, 'need_review': False, 'candidates': [], 'category': '制造'})

        for std, encoded in zip(detail.get('unknown', []), detail.get('unknown_encoded', [])):
            structured = self.standard_processor.parse_standard_structure(std)
            code_parts = self.standard_processor._split_code_and_grade(encoded)
            items.append({'original': std, 'matched': encoded, 'code': encoded,
                          'base_code': code_parts['base'], 'grade': structured['grade'] or code_parts['grade'],
                          'standard_subject': structured['subject'],
                          'standard_grade': structured['grade'],
                          'standard_method': structured['method'],
                          'standard_appendix': structured['appendix'],
                          'similarity': 1.0,
                          'is_exact': True, 'need_review': False, 'candidates': [], 'category': ''})

        original_display = ' | '.join([item['original'] for item in items])
        return FieldEncoding(
            field_type='STANDARD',
            original_value=original_display, original_values=values,
            matched_name=detail.get('display', ''),
            matched_names=[item['code'] for item in items],
            code=detail.get('encoded', ''),
            codes=[item['code'] for item in items],
            similarity=1.0, is_exact_match=True, need_review=False,
            candidates=[], display=detail.get('display', ''), items=items
        )
