# -*- coding: utf-8 -*-
"""
管道材料编码器
将NER识别结果转换为标准材料编码

编码顺序（固定）: TYPE + MANU + CONN + SIZE + THICKNESS + PRESSURE + MATERIAL + STANDARD

架构：
- PipeEncoderBase: 基类，包含所有公共逻辑（配置加载、预处理、字段收集、组装）
- LegacyPipeEncoder (pipe_encoder_legacy.py): T5 Seq2Seq + 规则处理器
- LlmPipeEncoder   (pipe_encoder_llm.py):   Qwen3 LLM 编码
- get_pipe_encoder(): 工厂函数，根据配置自动选择实现
"""

import re
import math
import logging
import json
import copy
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .semantic_matcher import get_semantic_matcher
from .processors import get_standard_processor
from .processors import get_thickness_processor
from .processors import get_pressure_processor
from .processors import get_size_processor
from .processors import get_regex_extractor

logger = logging.getLogger(__name__)


@dataclass
class FieldItem:
    """多值字段中的单个项"""
    original: str = ""
    matched: str = ""
    code: str = ""
    similarity: float = 1.0
    is_exact: bool = True
    need_review: bool = False
    candidates: List[Dict] = field(default_factory=list)
    category: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FieldEncoding:
    """单个字段的编码结果"""
    field_type: str = ""
    original_value: Any = ""
    original_values: List[str] = field(default_factory=list)
    matched_name: str = ""
    matched_names: List[str] = field(default_factory=list)
    code: str = ""
    codes: List[str] = field(default_factory=list)
    similarity: float = 1.0
    is_exact_match: bool = True
    need_review: bool = False
    candidates: List[Dict] = field(default_factory=list)
    display: str = ""
    items: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipeEncodingResult:
    """管道材料编码结果"""
    original_text: str
    fields: Dict[str, FieldEncoding] = field(default_factory=dict)
    final_code: str = ""
    success: bool = False
    need_review: bool = False
    hard_rule_hit: bool = False
    confidence: float = 0.0
    min_similarity: float = 1.0
    review_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        result = {
            'original_text': self.original_text,
            'final_code': self.final_code,
            'success': self.success,
            'need_review': self.need_review,
            'hard_rule_hit': self.hard_rule_hit,
            'confidence': self.confidence,
            'min_similarity': self.min_similarity,
            'review_fields': self.review_fields,
            'missing_fields': self.missing_fields,
            'errors': self.errors,
            'warnings': self.warnings,
            'fields': {k: v.to_dict() for k, v in self.fields.items()}
        }
        return result


@dataclass
class StandardPosition:
    """STANDARD 在原文中的位置。"""
    value: str
    index: int
    pos: int


@dataclass
class StandardModifierHit:
    """规范修饰项命中（等级/附录/方法）。"""
    label: str
    value: str
    start: int = 0
    end: int = 0
    code: str = ""


class PipeEncoderBase:
    """
    管道材料编码器 - 基类
    
    公共流程：
    1. 接收NER识别结果
    2. 正则补充提取（ENDS/SEAL/CONN/MANU 等）
    3. 三通异径预处理
    4. TYPE 组合字段收集 + 去重
    5. 各字段多值分发处理
    6. 按固定顺序组装最终编码
    
    子类需实现的方法：
    - _should_use_type_combined()
    - _encode_type_value(merged_value)
    - _encode_size_multi(values) -> FieldEncoding
    - _encode_thickness_value(value) -> str
    - _process_single_value(field_type, value) -> dict
    - _process_standard_multi(values, modifier_map) -> FieldEncoding
    """
    
    DEFAULT_FIELD_ORDER = ['TYPE', 'RADIUS', 'ENDS', 'MANU', 'CONN', 'SEAL', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
    TYPE_COMBINED_FIELDS = ['TYPE', 'RADIUS', 'ENDS', 'SEAL', 'MANU', 'CONN']
    STANDARD_MODIFIER_FIELDS = ['STANDARD_GRADE', 'STANDARD_APPENDIX', 'STANDARD_METHOD']
    
    def __init__(self):
        self.matcher = get_semantic_matcher()
        
        config_path = Path(__file__).parent / "config" / "encoder_config.yaml"
        self.config = self._load_config(config_path)
        
        platform_config_path = Path(__file__).parent.parent / "config" / "platform_config.yaml"
        self.platform_config = self._load_config(platform_config_path)
        
        self.FIELD_ORDER = self.config.get('field_order', self.DEFAULT_FIELD_ORDER)
        
        self.size_processor = get_size_processor()
        self.standard_processor = get_standard_processor()
        self.thickness_processor = get_thickness_processor()
        self.regex_extractor = get_regex_extractor()
        
        self.semantic_match_fields = set(self.config.get('semantic_match_fields', ['TYPE', 'MATERIAL']))
        self.exact_match_fields = set(self.config.get('exact_match_fields', ['MANU', 'CONN']))
        self.passthrough_fields = set(self.config.get('passthrough_fields', ['THICKNESS', 'PRESSURE']))
        self.review_rules_config = self.config.get('review_rules', {}) or {}
        self.extract_conf_weight = float(self.review_rules_config.get('extract_weight', 0.55))
        self.encode_conf_weight = float(self.review_rules_config.get('encode_weight', 0.45))
        self.review_threshold = float(self.review_rules_config.get('review_threshold', 0.80))
        self.hard_rule_force_review = bool(self.review_rules_config.get('hard_rule_force_review', True))
        calibration_cfg = self.review_rules_config.get('calibration', {}) or {}
        self.conf_calibration_enabled = bool(calibration_cfg.get('enabled', False))
        self.conf_temperature = float(calibration_cfg.get('temperature', 1.8))
        verification_cfg = self.config.get('field_verification', {}) or {}
        self.verification_enabled = bool(verification_cfg.get('enabled', False))
        self.verify_fields = set(verification_cfg.get('verify_fields', []))
        self.unverified_penalty = float(verification_cfg.get('unverified_penalty', 0.5))
    
    def _load_config(self, config_path: Path) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return {}
    
    # ──────────────────── 子类必须实现的方法 ────────────────────

    def _should_use_type_combined(self) -> bool:
        raise NotImplementedError

    def _encode_type_value(self, merged_value: str):
        """编码合并后的 TYPE 值，返回 (code, confidence)"""
        raise NotImplementedError

    def _encode_size_multi(self, values: List[str]) -> FieldEncoding:
        raise NotImplementedError

    def _encode_thickness_value(self, value: str, original_text: str = "") -> str:
        raise NotImplementedError

    def _process_single_value(self, field_type: str, value: str) -> dict:
        raise NotImplementedError

    def _process_standard_multi(self, values: List[str], modifier_map: Dict[int, Dict[str, List[str]]] = None) -> FieldEncoding:
        raise NotImplementedError

    # ──────────────────── 公共方法 ────────────────────

    def _preprocess_tee_reducing(self, entities: Dict, original_text: str) -> Dict:
        """
        三通异径预处理
        
        如果 TYPE 包含"三通"或"tee"，且原始描述不包含"变径"或"异径"，
        则基于 SIZE 的最终编码结果判断是否异径：
        - 编码后若为多段且数值不相同（如 200X150），判定为异径
        - 编码后若收敛为单段（如 DN200 与 8" 收敛为 200），不判定异径
        """
        type_value = entities.get('TYPE', '')
        type_text = self._get_type_body_text(entities)
        
        if '三通' not in type_text and 'tee' not in type_text:
            return entities
        
        text_lower = (original_text or '').lower()
        if '变径' in text_lower or '异径' in text_lower:
            return entities
        
        size_value = entities.get('SIZE', '')
        if self._is_reducing_size_by_encoded(size_value):
            if isinstance(type_value, dict):
                body = str(type_value.get('BODY') or '').strip()
                if body and not body.startswith('异径'):
                    type_value['BODY'] = '异径' + body
                entities['TYPE'] = type_value
            elif isinstance(type_value, list):
                entities['TYPE'] = ['异径' + type_value[0]] + type_value[1:]
            else:
                entities['TYPE'] = '异径' + (type_value or '')
            logger.info(f"[三通异径] 检测到三通异径，TYPE: {type_value} -> {entities['TYPE']}")
        
        return entities
    
    def _is_reducing_size(self, sizes: List[str]) -> bool:
        """判断尺寸数组是否为异径"""
        if len(sizes) < 2:
            return False
        
        def extract_number(s):
            m = re.search(r'(\d+(?:\.\d+)?)', s)
            return float(m.group(1)) if m else None
        
        nums = [extract_number(s) for s in sizes]
        nums = [n for n in nums if n is not None]
        
        return len(set(nums)) > 1 if len(nums) >= 2 else False

    def _is_reducing_size_by_encoded(self, size_value: Any) -> bool:
        """
        基于 SIZE 编码结果判断是否异径。
        仅在拿到编码值时判定，避免把「同一尺寸的不同表示」误判为异径。
        """
        if isinstance(size_value, dict):
            values = [size_value]
        elif isinstance(size_value, list):
            values = [v for v in size_value if v]
        elif isinstance(size_value, str):
            values = [size_value.strip()] if size_value.strip() else []
        else:
            values = []

        if not values:
            return False

        try:
            size_encoded = self._encode_size_multi(values)
        except Exception as e:
            logger.debug(f"[三通异径] SIZE 预编码失败，跳过异径判定: {e}")
            return False

        encoded = (size_encoded.code or "").strip().upper()
        if not encoded:
            return False

        parts = [p for p in re.split(r'[X×*/]+', encoded) if p]
        if len(parts) < 2:
            return False

        def extract_number(s: str):
            m = re.search(r'(\d+(?:\.\d+)?)', s)
            return float(m.group(1)) if m else None

        nums = [extract_number(p) for p in parts]
        nums = [n for n in nums if n is not None]
        if len(nums) >= 2:
            return len(set(nums)) > 1

        return len(set(parts)) > 1
    
    def _is_valid_field_value(self, field: str, value: str, field_config: Any) -> bool:
        """检查值是否匹配字段的配置"""
        if not value or not field_config:
            return False
        
        value_upper = value.strip().upper()
        
        if isinstance(field_config, dict):
            keywords = field_config.get('keywords', [])
            if value_upper in [kw.upper() for kw in keywords]:
                return True
            no_boundary = field_config.get('no_boundary', [])
            if value_upper in [kw.upper() for kw in no_boundary]:
                return True
            aliases = field_config.get('aliases', {})
            if value_upper in [k.upper() for k in aliases.keys()]:
                return True
            patterns = field_config.get('patterns', [])
            for pattern in patterns:
                try:
                    if re.match(pattern, value.strip(), re.IGNORECASE):
                        return True
                except re.error:
                    logger.warning(f"无效的正则表达式: {pattern}")
            return False
        elif isinstance(field_config, list):
            return value_upper in [kw.upper() for kw in field_config]
        
        return False
    
    def _process_thickness(self, value: str) -> str:
        if not value:
            return ""
        return get_thickness_processor().process(value)
    
    def _process_pressure(self, value: str) -> str:
        if not value:
            return ""
        return get_pressure_processor().process(value)

    @staticmethod
    def _stringify_field_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _clone_response_value(value: Any) -> Any:
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    @staticmethod
    def _ensure_type_dict(type_value: Any, create: bool = False) -> Optional[Dict[str, Any]]:
        if isinstance(type_value, dict):
            return type_value
        if not create:
            return None
        type_dict: Dict[str, Any] = {}
        if isinstance(type_value, list):
            parts = [str(v).strip() for v in type_value if str(v).strip()]
            if parts:
                type_dict['BODY'] = ';'.join(parts)
        elif type_value:
            type_dict['BODY'] = str(type_value).strip()
        return type_dict

    def _get_type_body_text(self, entities: Dict[str, Any]) -> str:
        type_dict = self._ensure_type_dict(entities.get('TYPE'))
        if type_dict is not None:
            body = type_dict.get('BODY', '')
            if isinstance(body, list):
                return ' '.join(str(v).strip() for v in body if str(v).strip()).lower()
            return str(body or '').lower()
        type_value = entities.get('TYPE', '')
        if isinstance(type_value, list):
            return ' '.join(str(v).strip() for v in type_value if str(v).strip()).lower()
        return str(type_value or '').lower()

    def _get_nested_type_value(self, entities: Dict[str, Any], label: str) -> Any:
        type_dict = self._ensure_type_dict(entities.get('TYPE'))
        if type_dict is not None and type_dict.get(label):
            return type_dict.get(label)
        return entities.get(label)

    def _set_nested_type_value(self, entities: Dict[str, Any], label: str, value: Any, overwrite: bool = False):
        if value in (None, "", []):
            return
        type_dict = self._ensure_type_dict(entities.get('TYPE'), create=True)
        if entities.get('TYPE') is not type_dict:
            entities['TYPE'] = type_dict
        if overwrite or not type_dict.get(label):
            type_dict[label] = value

    def _append_radius_to_type_body(self, entities: Dict[str, Any], display_value: str, code_value: str):
        token = (code_value or display_value or '').strip()
        if not token:
            return
        type_dict = self._ensure_type_dict(entities.get('TYPE'), create=True)
        if entities.get('TYPE') is not type_dict:
            entities['TYPE'] = type_dict
        body = str(type_dict.get('BODY') or '').strip()
        type_text = body.lower()
        included_keywords = self.config.get('type_included_keywords', {})
        keywords = included_keywords.get(token.upper(), [])
        is_included = any(kw in type_text for kw in keywords)
        if not is_included and token.lower() in type_text:
            is_included = True
        if is_included:
            return
        type_dict['BODY'] = f"{body};{token}" if body else token

    def _flatten_type_value_for_stage2(self, value: Any) -> str:
        type_dict = self._ensure_type_dict(value)
        if type_dict is None:
            if isinstance(value, list):
                parts = [str(v).strip() for v in value if str(v).strip()]
                return ';'.join(parts)
            return str(value or '').strip()

        parts: List[str] = []

        body = type_dict.get('BODY')
        if isinstance(body, list):
            parts.extend([str(v).strip() for v in body if str(v).strip()])
        elif body:
            parts.extend([p.strip() for p in str(body).split(';') if p.strip()])

        for key in ('ENDS', 'SEAL', 'MANU', 'CONN'):
            raw = type_dict.get(key)
            if not raw:
                continue
            values = raw if isinstance(raw, list) else [raw]
            for item in values:
                item_text = str(item).strip()
                if item_text:
                    parts.append(item_text)

        deduped: List[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return ';'.join(deduped)

    @staticmethod
    def _flatten_material_value_for_stage2(value: Any) -> str:
        if not isinstance(value, dict):
            return str(value or '').strip()

        items = value.get('ITEMS')
        if not isinstance(items, list) or not items:
            return ""

        parts: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                item_text = str(item).strip()
                if item_text:
                    parts.append(item_text)
                continue

            exec_standard = str(item.get('EXEC_STANDARD') or '').strip()
            grade_code = str(item.get('MATERIAL_GRADE_CODE') or '').strip()
            special_req = item.get('SPECIAL_REQ') or []
            special_parts = []
            if isinstance(special_req, list):
                special_parts = [str(v).strip() for v in special_req if str(v).strip()]
            elif special_req:
                special_parts = [str(special_req).strip()]

            item_parts = [p for p in [exec_standard, grade_code, *special_parts] if p]
            if item_parts:
                parts.append(' '.join(item_parts))

        return ' '.join(parts).strip()

    @staticmethod
    def _normalize_material_relation(value: Any) -> str:
        if isinstance(value, list):
            for item in value:
                text = str(item or '').strip().lower()
                if text:
                    return text
            return ""
        return str(value or '').strip().lower()

    @staticmethod
    def _flatten_material_item_for_stage2(item: Any) -> str:
        if item in (None, "", []):
            return ""
        if not isinstance(item, dict):
            return str(item).strip()

        exec_standard = str(item.get('EXEC_STANDARD') or '').strip()
        grade_code = str(item.get('MATERIAL_GRADE_CODE') or '').strip()
        special_req = item.get('SPECIAL_REQ') or []
        if isinstance(special_req, list):
            special_parts = [str(v).strip() for v in special_req if str(v).strip()]
        elif special_req:
            special_parts = [str(special_req).strip()]
        else:
            special_parts = []

        item_parts = [p for p in [exec_standard, grade_code, *special_parts] if p]
        return ' '.join(item_parts).strip()

    def _process_material_structured(self, value: Dict[str, Any]) -> FieldEncoding:
        relation = self._normalize_material_relation(value.get('RELATION'))
        raw_items = value.get('ITEMS')
        item_texts: List[str] = []
        if isinstance(raw_items, list):
            for item in raw_items:
                item_text = self._flatten_material_item_for_stage2(item)
                if item_text:
                    item_texts.append(item_text)

        if not item_texts:
            flattened = self._flatten_material_value_for_stage2(value)
            if not flattened:
                return FieldEncoding(field_type='MATERIAL')
            return self._process_field_multi('MATERIAL', [flattened])

        item_results = [self._process_single_value('MATERIAL', item_text) for item_text in item_texts]
        separator = '/' if relation == 'alternative' else ''

        items = []
        for item_text, item_result in zip(item_texts, item_results):
            items.append({
                'original': item_result['original'] or item_text,
                'matched': item_result['matched'] or item_text,
                'code': item_result['code'],
                'similarity': item_result['similarity'],
                'is_exact': item_result['is_exact'],
                'need_review': item_result['need_review'],
                'candidates': item_result.get('candidates', []),
                'category': '',
                'relation': relation,
            })

        original_values = [str(r['original'] or t) for r, t in zip(item_results, item_texts)]
        matched_names = [str(r['matched'] or t) for r, t in zip(item_results, item_texts)]

        unique_codes: List[str] = []
        seen = set()
        for item_result in item_results:
            code = str(item_result.get('code') or '').strip()
            if code and code not in seen:
                unique_codes.append(code)
                seen.add(code)

        min_similarity = min((r['similarity'] for r in item_results), default=1.0)
        any_need_review = any(r['need_review'] for r in item_results)
        all_exact = all(r['is_exact'] for r in item_results)

        candidates = []
        for item_result in item_results:
            if item_result.get('candidates'):
                candidates = item_result['candidates']
                break

        return FieldEncoding(
            field_type='MATERIAL',
            original_value=' | '.join(original_values),
            original_values=original_values,
            matched_name=' | '.join(matched_names),
            matched_names=matched_names,
            code=separator.join(unique_codes),
            codes=unique_codes,
            similarity=min_similarity,
            is_exact_match=all_exact,
            need_review=any_need_review,
            candidates=candidates,
            display=relation,
            items=items
        )

    def _normalize_field_value_for_stage2(self, field_type: str, value: Any) -> Any:
        if value in (None, "", []):
            return ""
        if field_type == 'TYPE':
            return self._flatten_type_value_for_stage2(value)
        if field_type == 'MATERIAL':
            return self._flatten_material_value_for_stage2(value)
        return value

    @staticmethod
    def _build_loose_entity_pattern(value: str) -> Optional[re.Pattern]:
        """
        构建宽松匹配模式：
        允许实体片段之间出现空白/标点/下划线/连字符差异。
        例如 "GB/T 12459 Series I" 可匹配 "GB/T 12459, Series I"。
        """
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None

        tokens = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)
        if not tokens:
            return None

        sep = r"[\s,;，、。:：_\-]*"
        pattern = sep.join(re.escape(tok) for tok in tokens)
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error:
            return None

    def _find_entity_position(self, text: str, value: str, start_pos: int = 0) -> int:
        """实体定位：先严格匹配，失败后宽松匹配。"""
        if not text or not value:
            return -1

        begin = max(0, int(start_pos or 0))
        text_lower = text.lower()
        value_lower = str(value).lower()
        idx = text_lower.find(value_lower, begin)
        if idx >= 0:
            return idx

        loose_pattern = self._build_loose_entity_pattern(value)
        if not loose_pattern:
            return -1
        m = loose_pattern.search(text, begin)
        return m.start() if m else -1

    def _assemble_code(self, result: PipeEncodingResult):
        """按固定顺序拼接各字段编码"""
        parts = []
        for field_type in self.FIELD_ORDER:
            if field_type in result.fields:
                code = result.fields[field_type].code
                if code:
                    parts.append(code)
        result.final_code = ''.join(parts)

    def _mark_field_review(
        self,
        result: PipeEncodingResult,
        field_type: str,
        warning: str,
        similarity_cap: Optional[float] = None,
    ):
        """将字段标记为待审，并可下调字段置信度上限。"""
        field = result.fields.get(field_type)
        if not field:
            return

        field.need_review = True
        if similarity_cap is not None:
            field.similarity = min(field.similarity, similarity_cap)

        if field_type not in result.review_fields:
            result.review_fields.append(field_type)
        result.hard_rule_hit = True
        result.min_similarity = min(result.min_similarity, field.similarity)

        if warning and warning not in result.warnings:
            result.warnings.append(warning)

    @staticmethod
    def _is_sch_like_token(token: str) -> bool:
        """判断壁厚片段是否为 SCH/S 系列表达。"""
        t = (token or "").strip().upper()
        if not t:
            return False
        return bool(re.match(r"^(SCH\d+(?:\.\d+)?S?|S-?\d+(?:\.\d+)?S?|XXS|XS|STD|\d+S)$", t))

    def _apply_field_verification_penalty(
        self,
        result: PipeEncodingResult,
        field_verified: Dict[str, tuple],
    ):
        """对未通过原文验证的字段做置信度惩罚。"""
        if not field_verified:
            return
        for field, (passed, llm_value) in field_verified.items():
            if passed:
                continue

            target_field = field
            if field not in result.fields and field in self.TYPE_COMBINED_FIELDS:
                target_field = 'TYPE'

            if target_field not in result.fields:
                continue

            fe = result.fields[target_field]
            old_sim = fe.similarity
            fe.similarity = round(old_sim * self.unverified_penalty, 4)
            self._mark_field_review(
                result, target_field,
                f"{field}='{llm_value}' 未在原文中被正则独立匹配到，建议人工审查。",
            )
            logger.info(
                f"[字段验证惩罚] {field}(→{target_field}): "
                f"置信度 {old_sim:.4f} → {fe.similarity:.4f} (×{self.unverified_penalty})"
            )

    def _apply_review_rules(self, result: PipeEncodingResult):
        """
        审查规则（命中即待审）：
        1) 异径场景下壁厚单位混用（如 20MMX30S）
        2) 材质多值
        3) 规范存在“未识别类型”项（category 为空）
        """
        size_field = result.fields.get('SIZE')
        thk_field = result.fields.get('THICKNESS')
        mat_field = result.fields.get('MATERIAL')
        std_field = result.fields.get('STANDARD')
        rules_enabled = self.review_rules_config.get('enabled', True)
        if not rules_enabled:
            return

        mixed_unit_cfg = self.review_rules_config.get('mixed_thickness_units', {}) or {}
        multi_material_cfg = self.review_rules_config.get('multi_material', {}) or {}
        unknown_std_cfg = self.review_rules_config.get('unknown_standard_category', {}) or {}
        over_segment_cfg = self.review_rules_config.get('over_segmented_size_or_thickness', {}) or {}

        # 0) 预编码阶段已判定需要审核的字段，在这里补齐整条审核原因。
        # 典型场景：SIZE 同时出现 DN 与 INCH/OD，且换算后不一致。
        if size_field and size_field.need_review:
            self._mark_field_review(
                result,
                'SIZE',
                "尺寸字段存在单位换算冲突：识别结果中的 DN 与 INCH/OD 换算结果不一致，建议人工审查。",
            )

        def _segment_count(text: str) -> int:
            if not text:
                return 0
            parts = [p for p in re.split(r"[xX×*/]+", str(text).replace(" ", "")) if p]
            return len(parts)

        def _encoded_segment_count(field: Optional[FieldEncoding]) -> int:
            if not field:
                return 0
            # 仅看编码结果：字段主 code + 多值项 code
            candidates: List[str] = []
            if field.code:
                candidates.append(str(field.code))
            if field.codes:
                candidates.extend([str(c) for c in field.codes if c])
            if field.items:
                for it in field.items:
                    cv = it.get('code') if isinstance(it, dict) else None
                    if cv:
                        candidates.append(str(cv))
            return max((_segment_count(t) for t in candidates), default=0)

        # 1) 壁厚异径时单位不一致
        if mixed_unit_cfg.get('enabled', True) and size_field and thk_field:
            size_text = self._stringify_field_value(
                size_field.code or size_field.matched_name or size_field.original_value or ""
            ).upper()
            thk_text = self._stringify_field_value(
                thk_field.code or thk_field.matched_name or thk_field.original_value or ""
            ).upper()
            if 'X' in size_text:
                parts = [p for p in re.split(r"[X×*/]+", thk_text.replace(" ", "")) if p]
                if len(parts) >= 2:
                    has_mm = any('MM' in p for p in parts)
                    has_sch = any(self._is_sch_like_token(p) for p in parts)
                    if has_mm and has_sch:
                        self._mark_field_review(
                            result,
                            'THICKNESS',
                            mixed_unit_cfg.get('warning', "壁厚分段单位混用（mm 与 Sch/S 系列），建议人工审查。"),
                            similarity_cap=float(mixed_unit_cfg.get('similarity_cap', 0.65)),
                        )

        # 2) 材质有多个
        if multi_material_cfg.get('enabled', True) and mat_field:
            material_count = len(mat_field.original_values or [])
            if material_count <= 1 and mat_field.items:
                material_count = len(mat_field.items)
            material_relation = ""
            if mat_field.items:
                material_relation = str(mat_field.items[0].get('relation') or '').strip().lower()
            if material_count > 1 and material_relation != 'alternative':
                self._mark_field_review(
                    result,
                    'MATERIAL',
                    multi_material_cfg.get('warning', "识别到多个材质值，建议人工确认主材质/组合材质编码。"),
                    similarity_cap=float(multi_material_cfg.get('similarity_cap', 0.75)),
                )

        # 3) 规范未识别到类型（生产/制造）
        if unknown_std_cfg.get('enabled', True) and std_field and std_field.items:
            unknown_items = [it for it in std_field.items if not (it.get('category') or '').strip()]
            if unknown_items:
                self._mark_field_review(
                    result,
                    'STANDARD',
                    unknown_std_cfg.get('warning', "部分规范未识别到类型（生产/制造），建议人工审查。"),
                    similarity_cap=float(unknown_std_cfg.get('similarity_cap', 0.70)),
                )

        # 4) 尺寸/壁厚编码结果分段过多（>2 段）
        if over_segment_cfg.get('enabled', True):
            max_segments = int(over_segment_cfg.get('max_segments', 2))

            if size_field:
                size_segments = _encoded_segment_count(size_field)
                if size_segments > max_segments:
                    self._mark_field_review(
                        result,
                        'SIZE',
                        over_segment_cfg.get('size_warning', "尺寸编码结果分段超过2段，建议人工审查。"),
                        similarity_cap=float(over_segment_cfg.get('size_similarity_cap', 0.68)),
                    )

            if thk_field:
                thk_segments = _encoded_segment_count(thk_field)
                if thk_segments > max_segments:
                    self._mark_field_review(
                        result,
                        'THICKNESS',
                        over_segment_cfg.get('thickness_warning', "壁厚编码结果分段超过2段，建议人工审查。"),
                        similarity_cap=float(over_segment_cfg.get('thickness_similarity_cap', 0.68)),
                    )

    def _compute_result_confidence(self, result: PipeEncodingResult) -> float:
        """计算整条编码置信度（字段加权几何平均）。"""
        if not result.fields:
            return 0.0

        weights = self.review_rules_config.get('confidence_weights', {}) or {}
        default_weights = {
            'TYPE': 0.22,
            'SIZE': 0.20,
            'THICKNESS': 0.18,
            'MATERIAL': 0.16,
            'STANDARD': 0.16,
            'MANU': 0.04,
            'CONN': 0.02,
            'ENDS': 0.01,
            'SEAL': 0.01,
        }
        merged_weights = {**default_weights, **weights}

        selected = []
        total_w = 0.0
        for ft, field in result.fields.items():
            w = float(merged_weights.get(ft, 0.0))
            if w <= 0:
                continue
            s = max(1e-6, min(1.0, float(field.similarity)))
            selected.append((w, s))
            total_w += w

        if not selected:
            sims = [max(1e-6, min(1.0, float(f.similarity))) for f in result.fields.values()]
            if not sims:
                return 0.0
            p = 1.0
            for s in sims:
                p *= s
            return p ** (1.0 / len(sims))

        log_sum = 0.0
        for w, s in selected:
            log_sum += (w / total_w) * math.log(s)
        return float(math.exp(log_sum))

    def _temperature_scale_confidence(self, p: float) -> float:
        """对置信度做温度缩放，缓解过度自信。"""
        p = max(1e-6, min(1 - 1e-6, float(p)))
        if not self.conf_calibration_enabled:
            return p
        t = max(1e-6, float(self.conf_temperature))
        # p' = sigmoid(logit(p)/T)
        logit = math.log(p / (1 - p))
        scaled = 1.0 / (1.0 + math.exp(-(logit / t)))
        return float(max(0.0, min(1.0, scaled)))

    def _finalize_review_decision(self, result: PipeEncodingResult):
        """
        最终审核判定只看：
        1) 硬规则命中（可配置是否强制待审）
        2) 总置信度是否低于阈值
        """
        if self.hard_rule_force_review and result.hard_rule_hit:
            result.need_review = True
            return
        any_field_need_review = any(field.need_review for field in result.fields.values())
        result.need_review = bool(any_field_need_review or result.confidence < self.review_threshold)

    def _get_field_extract_confidence(self, field_type: str, extract_confidence: Optional[Dict[str, Any]]) -> Optional[float]:
        """读取字段抽取置信度（支持标量或列表）。"""
        if not extract_confidence:
            return None
        raw = extract_confidence.get(field_type)
        if raw is None:
            return None
        if isinstance(raw, list):
            vals = []
            for v in raw:
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if not vals:
                return None
            # 多值字段取较保守值
            return max(0.0, min(1.0, min(vals)))
        try:
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return None

    def _blend_extract_and_encode_conf(self, extract_conf: Optional[float], encode_conf: float) -> float:
        """
        融合两阶段置信度：
        C_field = C_extract^a * C_encode^b
        """
        e = max(1e-6, min(1.0, float(encode_conf)))
        if extract_conf is None:
            return float(e)
        x = max(1e-6, min(1.0, float(extract_conf)))
        a = max(0.0, min(1.0, self.extract_conf_weight))
        b = max(0.0, min(1.0, self.encode_conf_weight))
        if a + b <= 0:
            a, b = 0.55, 0.45
        else:
            s = a + b
            a, b = a / s, b / s
        return float((x ** a) * (e ** b))

    # ──────────────────── TYPE 组合处理（公共逻辑） ────────────────────

    def _process_type_combined(
        self,
        entities: Dict[str, Any],
        regex_value_code_map: Dict[str, Dict]
    ) -> FieldEncoding:
        """合并 TYPE 新结构及规则补提字段，统一编码。"""
        collected_values = []
        type_dict = self._ensure_type_dict(entities.get('TYPE'))

        if type_dict is not None:
            body_value = type_dict.get('BODY')
            body_values = body_value if isinstance(body_value, list) else [body_value]
            for v in body_values:
                if v and str(v).strip():
                    collected_values.append(('TYPE', str(v).strip(), str(v).strip()))

            for f in ('ENDS', 'SEAL', 'MANU', 'CONN'):
                raw_value = type_dict.get(f) or entities.get(f)
                if not raw_value:
                    continue
                values_to_check = raw_value if isinstance(raw_value, list) else [raw_value]
                for v in values_to_check:
                    if v and str(v).strip():
                        info = regex_value_code_map.get(f, {})
                        code_val = str(info.get('code') or v).strip()
                        collected_values.append((f, str(v).strip(), code_val))
        else:
            for f in self.TYPE_COMBINED_FIELDS:
                if f == 'TYPE':
                    raw_value = entities.get(f)
                    if raw_value:
                        if isinstance(raw_value, list):
                            for v in raw_value:
                                if v:
                                    collected_values.append((f, str(v).strip(), str(v).strip()))
                        elif raw_value:
                            collected_values.append((f, str(raw_value).strip(), str(raw_value).strip()))
                else:
                    if f in regex_value_code_map:
                        info = regex_value_code_map[f]
                        display_val = info.get('value', '')
                        code_val = info.get('code', '')
                        if code_val:
                            collected_values.append((f, str(display_val).strip(), str(code_val).strip()))
                    else:
                        raw_value = entities.get(f)
                        if raw_value:
                            values_to_check = raw_value if isinstance(raw_value, list) else [raw_value]
                            for v in values_to_check:
                                if v:
                                    collected_values.append((f, str(v).strip(), str(v).strip()))
        
        if not collected_values:
            return FieldEncoding(field_type='TYPE', original_value='')
        
        type_values = [display for f, display, code in collected_values if f == 'TYPE']
        type_text = ' '.join(type_values).lower() if type_values else ''
        
        included_keywords = self.config.get('type_included_keywords', {})
        
        filtered_codes = []
        filtered_displays = []
        for f, display, code in collected_values:
            if f == 'TYPE':
                filtered_codes.append(code)
                filtered_displays.append(display)
            else:
                code_upper = code.upper()
                keywords = included_keywords.get(code_upper, [])
                is_included = any(kw in type_text for kw in keywords)
                if not is_included and display.strip():
                    is_included = display.strip().lower() in type_text
                if not is_included:
                    filtered_codes.append(code)
                    filtered_displays.append(display)
                else:
                    logger.debug(f"[TYPE合并] 跳过 {f}='{display}'，已包含在 TYPE 中")
        
        if not filtered_codes:
            return FieldEncoding(field_type='TYPE')
        
        merged_value = ';'.join(filtered_codes)
        original_parts = ' | '.join(filtered_displays)
        
        logger.info(f"[TYPE合并] 合并字段: {[f'{f}={d}({c})' for f, d, c in collected_values]}")
        logger.info(f"[TYPE合并] 编码用: '{merged_value}', 显示用: '{original_parts}'")
        
        code, confidence = self._encode_type_value(merged_value)
        
        return FieldEncoding(
            field_type='TYPE',
            original_value=original_parts,
            original_values=filtered_displays,
            matched_name=merged_value,
            matched_names=[merged_value],
            code=code,
            codes=[code] if code else [],
            similarity=confidence,
            is_exact_match=True,
            need_review=confidence < 0.8,
            candidates=[], display='', items=[]
        )

    # ──────────────────── 多值字段处理（公共逻辑） ────────────────────

    def _process_field_multi(self, field_type: str, values: List[str], original_text: str = "") -> FieldEncoding:
        """处理多值字段"""
        if not values:
            return FieldEncoding(field_type=field_type)

        if field_type == 'MATERIAL' and len(values) == 1 and isinstance(values[0], dict):
            return self._process_material_structured(values[0])
        
        if field_type == 'SIZE':
            return self._encode_size_multi(values)
        
        if field_type == 'STANDARD':
            return self._process_standard_multi(values, {})
        
        if field_type == 'THICKNESS' and any(not isinstance(v, str) for v in values):
            processed_results = []
            for v in values:
                p = self._encode_thickness_value(v, original_text=original_text)
                if p:
                    processed_results.append(p)

            unique_results = []
            seen = set()
            for r in processed_results:
                if r not in seen:
                    seen.add(r)
                    unique_results.append(r)

            processed = 'X'.join(unique_results) if unique_results else ''
            original_values = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]

            return FieldEncoding(
                field_type=field_type,
                original_value=' | '.join(original_values),
                original_values=original_values,
                matched_name=processed, matched_names=[processed],
                code=processed, codes=[processed] if processed else [],
                similarity=1.0, is_exact_match=True, need_review=False,
                candidates=[], display='', items=[]
            )

        if field_type == 'THICKNESS' and len(values) > 1:
            processed_results = []
            for v in values:
                p = self._encode_thickness_value(v, original_text=original_text)
                if p:
                    processed_results.append(p)
            
            unique_results = []
            seen = set()
            for r in processed_results:
                if r not in seen:
                    seen.add(r)
                    unique_results.append(r)

            processed = 'X'.join(unique_results) if unique_results else ''
            
            return FieldEncoding(
                field_type=field_type,
                original_value=' | '.join(values),
                original_values=list(values),
                matched_name=processed, matched_names=[processed],
                code=processed, codes=[processed] if processed else [],
                similarity=1.0, is_exact_match=True, need_review=False,
                candidates=[], display='', items=[]
            )
        
        multi_display_fields = ['MATERIAL']
        
        results = []
        for v in values:
            result = self._process_single_value(field_type, v)
            results.append(result)
        
        items = []
        if field_type in multi_display_fields and len(values) > 1:
            for r in results:
                items.append({
                    'original': r['original'], 'matched': r['matched'],
                    'code': r['code'], 'similarity': r['similarity'],
                    'is_exact': r['is_exact'], 'need_review': r['need_review'],
                    'candidates': r.get('candidates', []), 'category': ''
                })
        
        original_values = [str(r['original']) for r in results if r['original']]
        matched_names = [str(r['matched']) for r in results if r['matched']]
        codes = [str(r['code']) for r in results if r['code']]
        
        unique_codes = []
        seen = set()
        for c in codes:
            if c and c not in seen:
                unique_codes.append(c)
                seen.add(c)
        
        min_similarity = min(r['similarity'] for r in results) if results else 1.0
        any_need_review = any(r['need_review'] for r in results)
        all_exact = all(r['is_exact'] for r in results)
        
        candidates = []
        for r in results:
            if r.get('candidates'):
                candidates = r['candidates']
                break
        
        return FieldEncoding(
            field_type=field_type,
            original_value=' | '.join(original_values),
            original_values=original_values,
            matched_name=' | '.join(matched_names),
            matched_names=matched_names,
            code=''.join(unique_codes),
            codes=unique_codes,
            similarity=min_similarity,
            is_exact_match=all_exact,
            need_review=any_need_review,
            candidates=candidates,
            display='', items=items
        )

    def _build_exclude_ranges(self, entities: Dict[str, Any], original_text: str):
        """构建规则补充提取使用的排除区间。"""
        exclude_ranges = []

        def _collect_values(value: Any):
            if value in (None, "", []):
                return
            if isinstance(value, dict):
                if 'value' in value and value.get('value') not in (None, ""):
                    yield value.get('value')
                    return
                for sub_value in value.values():
                    yield from _collect_values(sub_value)
                return
            if isinstance(value, list):
                for item in value:
                    yield from _collect_values(item)
                return
            yield value

        for entity_value in entities.values():
            for val in _collect_values(entity_value):
                text_val = str(val).strip()
                if not text_val:
                    continue
                idx = self._find_entity_position(original_text, text_val)
                if idx >= 0:
                    exclude_ranges.append((idx, idx + len(text_val)))
        return exclude_ranges

    @staticmethod
    def _split_regex_extractions(regex_all_results):
        """拆分规则提取结果：仅保留当前仍使用的补提字段。"""
        other_extractions = {}
        for extraction in regex_all_results:
            if extraction.label not in other_extractions:
                other_extractions[extraction.label] = {
                    'value': extraction.value, 'code': extraction.code
                }
        return other_extractions

    def _apply_regex_fallbacks(self, entities: Dict[str, Any], regex_value_code_map: Dict[str, Dict], other_extractions: Dict[str, Dict]):
        """将规则补提结果回填到实体。"""
        for label, info in other_extractions.items():
            if label == 'RADIUS':
                self._append_radius_to_type_body(entities, info['value'], info['code'])
                regex_value_code_map[label] = info
            elif label in ('MANU', 'CONN', 'ENDS', 'SEAL'):
                if not self._get_nested_type_value(entities, label):
                    self._set_nested_type_value(entities, label, info['value'])
                regex_value_code_map[label] = info
            elif label not in entities or not entities[label]:
                entities[label] = info['value']
                regex_value_code_map[label] = info

    def _validate_fields_against_text(
        self,
        entities: Dict[str, Any],
        original_text: str,
    ) -> Dict[str, tuple]:
        """
        字段验证：用 RegexExtractor 检查 LLM 提取的字段值能否被正则从原文独立匹配到。
        不丢弃任何值，仅返回每个字段的 (passed, llm_value)。
        后续根据验证结果对置信度做惩罚。

        通过 RegexExtractor 带边界正则匹配，
        粘连情况（如 GB/T4237WELDED）会因缺少边界而匹配不到，判定为"未验证通过"。
        """
        verified: Dict[str, tuple] = {}
        if not self.verification_enabled or not self.verify_fields or not original_text:
            return verified

        regex_results = self.regex_extractor.extract(original_text, exclude_ranges=[])
        regex_found: Dict[str, set] = {}
        for extraction in regex_results:
            if extraction.label not in regex_found:
                regex_found[extraction.label] = set()
            regex_found[extraction.label].add(extraction.value.upper())
            regex_found[extraction.label].add(extraction.code.upper())

        for field in self.verify_fields:
            field_value = self._get_nested_type_value(entities, field)
            if not field_value:
                continue
            llm_value = str(field_value).strip()
            if not llm_value:
                continue
            found = llm_value.upper() in regex_found.get(field, set())
            verified[field] = (found, llm_value)
            if not found:
                logger.info(f"[字段验证] {field}='{llm_value}' 未被正则从原文独立匹配到，置信度将惩罚")
            else:
                logger.debug(f"[字段验证] {field}='{llm_value}' 原文正则验证通过")
        return verified

    def _augment_entities_from_text(
        self,
        entities: Dict[str, Any],
        original_text: str,
        regex_value_code_map: Dict[str, Dict],
    ):
        """基于原文做规则补提。STANDARD 修饰符绑定由标准处理器统一负责。"""
        if not original_text:
            return

        exclude_ranges = self._build_exclude_ranges(entities, original_text)
        regex_all_results = self.regex_extractor.extract(original_text, exclude_ranges)
        other_extractions = self._split_regex_extractions(regex_all_results)
        self._apply_regex_fallbacks(entities, regex_value_code_map, other_extractions)

    def _build_standard_modifier_map(
        self,
        entities: Dict[str, Any],
        original_text: str = ""
    ) -> Dict[int, Dict[str, List[str]]]:
        """将 STANDARD_* 修饰项按 bind_to_index / 位置绑定到对应 STANDARD。"""
        standards = entities.get('STANDARD')
        if not standards:
            return {}

        if not isinstance(standards, list):
            standards = [standards]
        standard_count = len(standards)
        if standard_count == 0:
            return {}

        positions = []
        for pos_info in entities.get('_STANDARD_POSITIONS', []) or []:
            if isinstance(pos_info, dict):
                positions.append(StandardPosition(
                    value=str(pos_info.get('value', '')),
                    index=int(pos_info.get('index', 0) or 0),
                    pos=int(pos_info.get('start_pos', pos_info.get('pos', 0)) or 0),
                ))

        modifier_map: Dict[int, Dict[str, List[str]]] = {}

        def _resolve_bind_idx(item: Any) -> Optional[int]:
            if isinstance(item, dict):
                bind_to_index = item.get('bind_to_index')
                if bind_to_index is not None:
                    idx = int(bind_to_index)
                    if 0 <= idx < standard_count:
                        return idx

                start = item.get('start')
                if start is not None and positions:
                    start = int(start)
                    candidate = None
                    best_distance = float('inf')
                    for pos in positions:
                        if pos.pos <= start:
                            distance = start - pos.pos
                            if distance < best_distance:
                                best_distance = distance
                                candidate = pos.index
                    if candidate is not None:
                        return candidate

            if standard_count == 1:
                return 0
            return None

        for field in self.STANDARD_MODIFIER_FIELDS:
            raw_value = entities.get(field)
            if not raw_value:
                continue

            items = raw_value if isinstance(raw_value, list) else [raw_value]
            for item in items:
                value = item.get('value') if isinstance(item, dict) else item
                if not value:
                    continue

                bind_idx = _resolve_bind_idx(item)
                if bind_idx is None:
                    logger.warning(f"[STANDARD修饰符] {field}='{value}' 缺少 bind_to_index，且无法通过位置推断，已跳过")
                    continue

                field_map = modifier_map.setdefault(bind_idx, {})
                field_values = field_map.setdefault(field, [])
                if str(value) not in field_values:
                    field_values.append(str(value))

        return modifier_map

    def _encode_fields(
        self,
        entities: Dict[str, Any],
        raw_entities_snapshot: Dict[str, Any],
        extract_confidence: Optional[Dict[str, Any]],
        regex_value_code_map: Dict[str, Dict],
        result: PipeEncodingResult,
        type_combined_processed: bool,
        original_text: str = "",
    ):
        """按字段顺序编码并写入结果对象。"""
        for field_type in self.FIELD_ORDER:
            if type_combined_processed and field_type in self.TYPE_COMBINED_FIELDS:
                continue

            raw_value = entities.get(field_type, None)
            if field_type == 'MATERIAL' and isinstance(raw_value, dict):
                field_result = self._process_material_structured(raw_value)

                field_extract_conf = self._get_field_extract_confidence(field_type, extract_confidence)
                field_result.similarity = self._blend_extract_and_encode_conf(
                    field_extract_conf,
                    field_result.similarity
                )
                field_result.similarity = self._temperature_scale_confidence(field_result.similarity)

                if field_type in raw_entities_snapshot:
                    field_result.original_value = self._clone_response_value(raw_entities_snapshot.get(field_type))

                result.fields[field_type] = field_result

                if field_result.need_review and field_type not in result.review_fields:
                    result.review_fields.append(field_type)
                if not field_result.code and field_type not in result.missing_fields:
                    result.missing_fields.append(field_type)
                if field_result.similarity < result.min_similarity:
                    result.min_similarity = field_result.similarity
                continue

            raw_value = self._normalize_field_value_for_stage2(field_type, raw_value)
            if not raw_value:
                continue

            if isinstance(raw_value, list):
                values = [
                    normalized for normalized in
                    (self._normalize_field_value_for_stage2(field_type, v) for v in raw_value)
                    if normalized
                ]
            else:
                values = [raw_value] if raw_value else []

            if not values:
                continue

            if field_type in regex_value_code_map:
                info = regex_value_code_map[field_type]
                field_result = FieldEncoding(
                    field_type=field_type,
                    original_value=info['value'],
                    original_values=[info['value']],
                    matched_name=info['value'],
                    matched_names=[info['value']],
                    code=info['code'],
                    codes=[info['code']],
                    similarity=1.0,
                    is_exact_match=True,
                    need_review=False
                )
            elif field_type == 'STANDARD':
                modifier_map = entities.get('_STANDARD_MODIFIER_MAP', {})
                field_result = self._process_standard_multi(values, modifier_map)
            else:
                field_result = self._process_field_multi(field_type, values, original_text=original_text)

            field_extract_conf = self._get_field_extract_confidence(field_type, extract_confidence)
            field_result.similarity = self._blend_extract_and_encode_conf(
                field_extract_conf,
                field_result.similarity
            )
            field_result.similarity = self._temperature_scale_confidence(field_result.similarity)

            if field_type in raw_entities_snapshot:
                field_result.original_value = self._clone_response_value(raw_entities_snapshot.get(field_type))

            result.fields[field_type] = field_result

            if field_result.need_review and field_type not in result.review_fields:
                result.review_fields.append(field_type)
            if not field_result.code and field_type not in result.missing_fields:
                result.missing_fields.append(field_type)
            if field_result.similarity < result.min_similarity:
                result.min_similarity = field_result.similarity

    @staticmethod
    def _append_entity(
        entities: Dict[str, List[str]],
        current_entity: Dict[str, str],
        current_entity_pos: Dict[str, int],
        standard_positions: List[StandardPosition],
        entity_type: str
    ):
        """将当前实体片段写入实体集合，并维护 STANDARD 的位置信息。"""
        value = current_entity.get(entity_type, "")
        if not value:
            return
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(value)
        if entity_type == 'STANDARD':
            pos = current_entity_pos.get(entity_type, 0)
            standard_positions.append(StandardPosition(
                value=value,
                index=len(entities['STANDARD']) - 1,
                pos=pos
            ))

    def _flush_standard_modifiers_from_current(
        self,
        current_entity: Dict[str, str],
        current_entity_pos: Dict[str, int],
        standard_modifiers: Dict[str, List[StandardModifierHit]]
    ):
        """将当前缓存的 STANDARD_* 修饰项写入列表并清理缓存。"""
        for field in self.STANDARD_MODIFIER_FIELDS:
            if field in current_entity:
                standard_modifiers.setdefault(field, []).append(StandardModifierHit(
                    label=field,
                    value=current_entity[field],
                    start=current_entity_pos.get(field, 0),
                ))
                current_entity.pop(field)
                current_entity_pos.pop(field, None)

    def _flush_current_entities(
        self,
        entities: Dict[str, List[str]],
        current_entity: Dict[str, str],
        current_entity_pos: Dict[str, int],
        standard_positions: List[StandardPosition]
    ):
        """将当前缓存实体全部写入实体集合。"""
        for etype in list(current_entity.keys()):
            self._append_entity(entities, current_entity, current_entity_pos, standard_positions, etype)
        current_entity.clear()
        current_entity_pos.clear()

    @staticmethod
    def _ensure_token_positions(tokens: List[Dict[str, Any]], original_text: str):
        """为缺失 start/end 的 token 基于原文补齐位置信息。"""
        if not original_text:
            return
        current_pos = 0
        for token in tokens:
            word = token.get('word', '')
            if token.get('start') is None and word:
                idx = original_text.find(word, current_pos)
                if idx >= 0:
                    token['start'] = idx
                    token['end'] = idx + len(word)
                    current_pos = token['end']
                else:
                    token['start'] = current_pos
                    token['end'] = current_pos + len(word)
                    current_pos = token['end']
            elif token.get('start') is not None:
                current_pos = token.get('end', token['start'] + len(word))

    # ──────────────────── SIZE / THICKNESS 拆分修正 ────────────────────

    # 明确的壁厚起始标识（参考 ThicknessProcessor 的模式）
    _THICKNESS_TOKEN_RE = re.compile(
        r'SCH[.\s]?(?:XXS|XS|STD|\d+(?:\.\d+)?S?)'   # SCH40, SCH40S, SCHXXS, SCHSTD
        r'|S-(?:\d+(?:\.\d+)?S?|XXS|XS|STD)'          # S-40, S-10S, S-XXS
        r'|S\d{1,3}S(?!\d)'                            # S40S, S10S, S80S
        r'|\d{1,3}S(?!\d)'                             # 40S, 10S, 80S
        r'|(?:T|THK)\s*=\s*\d+(?:\.\d+)?',            # T=3.0, THK=6.3
        re.IGNORECASE,
    )

    def _fix_size_thickness_split(self, entities: Dict[str, Any]):
        """
        修正 LLM 将壁厚误归入尺寸的问题。
        当 SIZE 中包含明确的壁厚标识（如 SCH40）且 THICKNESS 缺失时，
        自动拆分 SIZE 为 SIZE + THICKNESS。

        典型案例:
            DN20xSCH40           → SIZE=DN20,    THICKNESS=SCH40
            DN50x40-SCH40xSCH40  → SIZE=DN50x40, THICKNESS=SCH40xSCH40
            DN200xSCH20          → SIZE=DN200,   THICKNESS=SCH20
        """
        if entities.get('THICKNESS'):
            return

        size_val = entities.get('SIZE')
        if not size_val:
            return

        if isinstance(size_val, str):
            size_part, thickness_part = self._try_split_size_thickness(size_val)
            if thickness_part:
                entities['SIZE'] = size_part
                entities['THICKNESS'] = thickness_part
                logger.info(
                    f"[SIZE/THICKNESS拆分] '{size_val}' → SIZE='{size_part}', THICKNESS='{thickness_part}'"
                )
        elif isinstance(size_val, list):
            new_size: List[str] = []
            thickness_parts: List[str] = []
            for v in size_val:
                if self._THICKNESS_TOKEN_RE.match(v.strip()):
                    thickness_parts.append(v)
                else:
                    s, t = self._try_split_size_thickness(v)
                    if t:
                        new_size.append(s)
                        thickness_parts.append(t)
                    else:
                        new_size.append(v)
            if thickness_parts:
                entities['SIZE'] = new_size if len(new_size) != 1 else new_size[0]
                entities['THICKNESS'] = thickness_parts if len(thickness_parts) != 1 else thickness_parts[0]
                logger.info(
                    f"[SIZE/THICKNESS拆分] 列表 → SIZE={entities['SIZE']}, THICKNESS={entities['THICKNESS']}"
                )

    def _try_split_size_thickness(self, size_val: str) -> tuple:
        """
        尝试从 SIZE 字符串中拆分出壁厚部分。

        Returns:
            (size_part, thickness_part) — 无法拆分时 thickness_part 为空字符串
        """
        m = self._THICKNESS_TOKEN_RE.search(size_val)
        if not m:
            return size_val, ''

        thickness_pos = m.start()
        if thickness_pos == 0:
            return size_val, ''

        # 分隔符：-, x, X, ×, *, /, 逗号, 空格
        prefix = size_val[:thickness_pos]
        sep_match = re.search(r'[-xX×*/,\s]+$', prefix)
        if not sep_match:
            return size_val, ''

        size_part = prefix[:sep_match.start()].strip()
        thickness_part = size_val[thickness_pos:].strip()

        if size_part and thickness_part:
            return size_part, thickness_part
        return size_val, ''

    # ──────────────────── 主编码方法 ────────────────────

    def encode(
        self,
        entities: Dict[str, str],
        original_text: str = "",
        extract_confidence: Optional[Dict[str, Any]] = None,
    ) -> PipeEncodingResult:
        """
        编码材料实体
        
        Args:
            entities: NER识别结果 {字段类型: 值}
            original_text: 原始描述
            
        Returns:
            编码结果
        """
        result = PipeEncodingResult(original_text=original_text)
        
        if not entities:
            result.errors.append("输入实体为空")
            return result

        raw_entities_snapshot = self._clone_response_value(entities) if isinstance(entities, dict) else {}
        
        regex_value_code_map = {}
        
        self._fix_size_thickness_split(entities)
        self._augment_entities_from_text(entities, original_text, regex_value_code_map)
        entities['_STANDARD_MODIFIER_MAP'] = self._build_standard_modifier_map(entities, original_text)
        
        entities = self._preprocess_tee_reducing(entities, original_text)
        
        type_combined_processed = False
        if self._should_use_type_combined():
            type_combined_result = self._process_type_combined(entities, regex_value_code_map)
            if type_combined_result.code:
                type_extract_conf = self._get_field_extract_confidence('TYPE', extract_confidence)
                type_combined_result.similarity = self._blend_extract_and_encode_conf(
                    type_extract_conf,
                    type_combined_result.similarity
                )
                type_combined_result.similarity = self._temperature_scale_confidence(type_combined_result.similarity)
                if 'TYPE' in raw_entities_snapshot:
                    type_combined_result.original_value = self._clone_response_value(raw_entities_snapshot.get('TYPE'))
                result.fields['TYPE'] = type_combined_result
                type_combined_processed = True
                if type_combined_result.need_review and 'TYPE' not in result.review_fields:
                    result.review_fields.append('TYPE')
                if type_combined_result.similarity < result.min_similarity:
                    result.min_similarity = type_combined_result.similarity
        
        field_verified = self._validate_fields_against_text(entities, original_text)

        self._encode_fields(
            entities,
            raw_entities_snapshot,
            extract_confidence,
            regex_value_code_map,
            result,
            type_combined_processed,
            original_text=original_text,
        )

        self._apply_field_verification_penalty(result, field_verified)
        self._apply_review_rules(result)
        
        self._assemble_code(result)
        result.success = bool(result.final_code)
        raw_conf = self._compute_result_confidence(result)
        result.confidence = round(self._temperature_scale_confidence(raw_conf), 4)
        self._finalize_review_decision(result)
        return result

    # ──────────────────── 从分词结果编码 ────────────────────

    def encode_from_tokens(
        self,
        tokens: List[Dict[str, str]],
        original_text: str = ""
    ) -> PipeEncodingResult:
        """从分词结果编码"""
        self._ensure_token_positions(tokens, original_text)
        
        entities: Dict[str, List[str]] = {}
        current_entity: Dict[str, str] = {}
        current_entity_pos: Dict[str, int] = {}
        
        standard_positions: List[StandardPosition] = []
        standard_modifiers: Dict[str, List[StandardModifierHit]] = {}
        
        for token in tokens:
            word = token.get('word', '')
            tag = token.get('tag', 'O')
            start_pos = token.get('start', 0)
            end_pos = token.get('end', start_pos + len(word))
            
            if not tag or tag == 'O':
                self._flush_current_entities(entities, current_entity, current_entity_pos, standard_positions)
                continue
            
            if tag.startswith('B-') or tag.startswith('I-'):
                prefix = tag[0]
                entity_type = tag[2:]
            else:
                prefix = 'B'
                entity_type = tag
            
            if entity_type in self.STANDARD_MODIFIER_FIELDS:
                if prefix == 'B':
                    if 'STANDARD' in current_entity and current_entity['STANDARD']:
                        self._append_entity(
                            entities, current_entity, current_entity_pos, standard_positions, 'STANDARD'
                        )
                        current_entity.pop('STANDARD')
                        current_entity_pos.pop('STANDARD', None)
                    
                    current_entity[entity_type] = word
                    current_entity_pos[entity_type] = start_pos
                else:
                    if entity_type in current_entity:
                        current_entity[entity_type] += word
                    else:
                        current_entity[entity_type] = word
                        current_entity_pos[entity_type] = start_pos
                continue
            
            if prefix == 'B':
                if entity_type in current_entity and current_entity[entity_type]:
                    self._append_entity(
                        entities, current_entity, current_entity_pos, standard_positions, entity_type
                    )
                
                self._flush_standard_modifiers_from_current(
                    current_entity, current_entity_pos, standard_modifiers
                )
                
                current_entity[entity_type] = word
                current_entity_pos[entity_type] = start_pos
            else:
                if entity_type in current_entity:
                    current_entity[entity_type] += word
                else:
                    current_entity[entity_type] = word
                    current_entity_pos[entity_type] = start_pos
        
        self._flush_standard_modifiers_from_current(
            current_entity, current_entity_pos, standard_modifiers
        )
        
        self._flush_current_entities(entities, current_entity, current_entity_pos, standard_positions)
        
        for field, hits in standard_modifiers.items():
            if hits:
                entities[field] = [
                    {'value': hit.value, 'start': hit.start}
                    for hit in hits
                ]
        
        if standard_positions and 'STANDARD' in entities:
            entities['_STANDARD_POSITIONS'] = [
                {'value': p.value, 'index': p.index, 'start_pos': p.pos}
                for p in standard_positions
            ]
        
        return self.encode(entities, original_text, extract_confidence=None)

    # ──────────────────── 批量 / 工具方法 ────────────────────

    def batch_encode(
        self,
        items: List[Dict],
        progress_callback=None
    ) -> List[PipeEncodingResult]:
        """批量编码"""
        results = []
        total = len(items)
        for i, item in enumerate(items):
            if 'tokens' in item:
                r = self.encode_from_tokens(item['tokens'], item.get('text', ''))
            else:
                r = self.encode(
                    item.get('entities', {}),
                    item.get('text', ''),
                    item.get('extract_confidence')
                )
            results.append(r)
            if progress_callback:
                progress_callback(i, total, r)
        return results
    
    def get_threshold(self) -> float:
        return self.matcher.get_threshold()
    
    def set_threshold(self, threshold: float):
        self.matcher.set_threshold(threshold)
    
    def reload_mapping(self):
        self.matcher.reload_mapping()


# ──────────────────── 工厂函数 & 兼容别名 ────────────────────

PipeEncoder = PipeEncoderBase

_encoder_instance: Optional[PipeEncoderBase] = None


def get_pipe_encoder() -> PipeEncoderBase:
    """
    根据 platform_config.yaml 中的 encoding.method 自动选择编码器实现
    - 'llm'  → LlmPipeEncoder
    - 其他   → LegacyPipeEncoder
    """
    global _encoder_instance
    if _encoder_instance is None:
        config_path = Path(__file__).parent.parent / "config" / "platform_config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        
        method = cfg.get('encoding', {}).get('method', 'legacy')
        
        if method == 'llm':
            from .pipe_encoder_llm import LlmPipeEncoder
            _encoder_instance = LlmPipeEncoder()
            logger.info("编码器: LlmPipeEncoder")
        else:
            from .pipe_encoder_legacy import LegacyPipeEncoder
            _encoder_instance = LegacyPipeEncoder()
            logger.info("编码器: LegacyPipeEncoder")
    
    return _encoder_instance
