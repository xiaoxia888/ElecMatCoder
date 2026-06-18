# -*- coding: utf-8 -*-
"""
管道材料编码器
将NER识别结果转换为标准材料编码

编码顺序（固定）: TYPE + MANU + CONN + SIZE + THICKNESS + PRESSURE + MATERIAL + STANDARD

架构：
- PipeEncoderBase: 基类，包含所有公共逻辑（配置加载、预处理、字段收集、组装）
- LlmPipeEncoder   (pipe_encoder_llm.py):   Qwen3 LLM 编码
- get_pipe_encoder(): 工厂函数，返回统一的 LLM 编码实现
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

from ..domain.common import OrderedValueItem
from ..domain.material import MaterialItem
from ..domain.pipeline import (
    ConfidenceDetail,
    EncodeResultPayload,
    FieldResultPayload,
    FieldStatus,
    Stage1RawPayload,
    Stage2InputPayload,
    Stage2OutputPayload,
)
from ..domain.pressure import PressureItem, PressureValue
from ..domain.size import SizeValue
from ..domain.standard import StandardItem
from ..domain.thickness import ThicknessValue
from ..domain.type import TypeGeometry, TypeValue
from .semantic_matcher import get_semantic_matcher
from .processors import get_standard_processor
from .processors import get_thickness_processor
from .processors import get_pressure_processor
from .processors import get_size_processor
from .processors import get_regex_extractor
from .processors import get_thickness_table_processor

logger = logging.getLogger(__name__)


@dataclass
class EncodedFieldDetail:
    """字段编码明细项，主要用于材质/规范等多值字段的辅助审查。"""
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
class EncodedFieldResult:
    """单字段编码结果，内部真值统一为三层：stage1_raw / stage2_input / code。"""
    field_type: str = ""
    stage1_raw: Any = ""
    stage2_input: Any = ""
    stage1_confidence: Optional[float] = None
    stage2_confidence: Optional[float] = None
    field_confidence: Optional[float] = None
    encode_confidence_v2: Dict[str, Any] = field(default_factory=dict)
    code: str = ""
    codes: List[str] = field(default_factory=list)
    similarity: float = 1.0
    is_exact_match: bool = True
    need_review: bool = False
    candidates: List[Dict] = field(default_factory=list)
    detail_items: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipeEncodingResult:
    """管道材料编码结果"""
    original_text: str
    fields: Dict[str, EncodedFieldResult] = field(default_factory=dict)
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
    thickness_conversion_notes: List[str] = field(default_factory=list)
    extract_confidence_v2: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """默认直接返回统一三层 schema。"""
        return self.to_payload_dict()

    def to_payload_dict(
        self,
        *,
        processed_text: str = "",
        route_info: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        转成统一三层 schema。

        这里把新 schema 的组装责任下沉到编码器结果对象本身，避免 server 继续充当结构桥接层。
        """
        fields = {
            field_type: self._build_field_payload(field_type, field_data)
            for field_type, field_data in self.fields.items()
        }
        payload = EncodeResultPayload(
            original_text=self.original_text,
            processed_text=processed_text or self.original_text,
            final_code=self.final_code,
            success=bool(self.success),
            need_review=bool(self.need_review),
            confidence=round(float(self.confidence or 0.0), 4),
            fields=fields,
            route_info=copy.deepcopy(route_info) if isinstance(route_info, dict) else None,
            errors=list(self.errors or []),
            warnings=list(self.warnings or []),
        )
        return payload.to_dict()

    def _extract_stage1_meta(self, field_type: str) -> dict[str, Any]:
        item = self.extract_confidence_v2.get(field_type)
        if not isinstance(item, dict):
            item = {}
        return {
            "source": str(item.get("source", "") or ""),
            "confidence": item.get("confidence"),
            "reason": str(item.get("reason", "") or ""),
            "evidence": copy.deepcopy(item.get("evidence") or {}),
        }

    def _extract_mm_context_from_stage1_field(self, field_obj: Any) -> list[float]:
        result: list[float] = []
        if field_obj is None:
            return result
        stage1_value = _field_obj_get(field_obj, "stage1_raw")
        if not isinstance(stage1_value, dict):
            return result
        ordered_items = stage1_value.get("_ITEMS") or stage1_value.get("ordered_items") or []
        for item in ordered_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().upper() != "MM":
                continue
            raw = str(item.get("value", "") or "").strip().upper()
            if raw.endswith("MM"):
                raw = raw[:-2]
            try:
                result.append(float(raw))
            except (TypeError, ValueError):
                continue
        return result

    def _build_stage2_input_value(self, field_type: str, field_obj: Any) -> Any:
        return _rename_stage_value_keys(copy.deepcopy(_field_obj_get(field_obj, "stage2_input", "")))

    @staticmethod
    def _build_stage2_notes(field_type: str, stage1_value: Any, stage2_value: Any, field_notes: Any = None) -> list[str]:
        notes: list[str] = []
        if field_type == "TYPE" and isinstance(stage1_value, dict) and isinstance(stage2_value, dict):
            stage1_conn = stage1_value.get("CONN") or stage1_value.get("ENDS") or []
            stage2_conn = stage2_value.get("CONN") or []
            if stage1_conn and not stage2_conn:
                notes.append("连接方式未参与编码")
        if field_type == "SIZE" and isinstance(stage2_value, dict) and stage2_value.get("thickness_mm_context"):
            notes.append("附加壁厚上下文用于OD/DN消歧")
        if field_type == "MATERIAL" and stage1_value != stage2_value:
            notes.append("材质值归一化后参与编码")
        if field_type == "STANDARD" and stage1_value != stage2_value:
            notes.append("标准主体归一化并补充分类后参与编码")
        for item in field_notes if isinstance(field_notes, list) else []:
            text = str(item or "").strip()
            if text and text not in notes:
                notes.append(text)
        return notes

    def _build_field_payload(self, field_type: str, field_data: Any) -> FieldResultPayload:
        stage1_value = _rename_stage_value_keys(copy.deepcopy(_field_obj_get(field_data, "stage1_raw", "")))
        stage1_meta = self._extract_stage1_meta(field_type)
        stage2_value = self._build_stage2_input_value(field_type, field_data)
        field_notes = copy.deepcopy(_field_obj_get(field_data, "notes", []))
        return FieldResultPayload(
            field_type=field_type,
            stage1_raw=Stage1RawPayload(
                value=stage1_value,
                source=stage1_meta["source"],
                confidence=stage1_meta["confidence"],
                reason=stage1_meta["reason"],
                evidence=stage1_meta["evidence"],
            ),
            stage2_input=Stage2InputPayload(
                value=stage2_value,
                notes=self._build_stage2_notes(field_type, stage1_value, stage2_value, field_notes),
            ),
            stage2_output=Stage2OutputPayload(
                code=str(_field_obj_get(field_data, "code", "") or "").strip(),
            ),
            confidence_detail=ConfidenceDetail(
                stage1=_field_obj_get(field_data, "stage1_confidence", None),
                stage2=_field_obj_get(field_data, "stage2_confidence", None),
                field=_field_obj_get(field_data, "field_confidence", None),
            ),
            status=FieldStatus(
                need_review=bool(_field_obj_get(field_data, "need_review", False)),
                similarity=round(float(_field_obj_get(field_data, "similarity", 0.0) or 0.0), 4),
                is_exact_match=_field_obj_get(field_data, "is_exact_match", None),
            ),
        )


@dataclass
class StandardPosition:
    """STANDARD 在原文中的位置。"""
    value: str
    index: int
    pos: int


def _rename_stage_value_keys(value: Any) -> Any:
    """统一历史内部键名，避免 `_ITEMS` 等实现细节泄漏到新 schema。"""
    if isinstance(value, list):
        return [_rename_stage_value_keys(item) for item in value]
    if not isinstance(value, dict):
        return value

    renamed: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text == "_ITEMS":
            renamed["ordered_items"] = _rename_stage_value_keys(item)
            continue
        if key_text == "_THICKNESS_MM_CONTEXT":
            renamed["thickness_mm_context"] = _rename_stage_value_keys(item)
            continue
        if key_text.startswith("_"):
            continue
        renamed[key_text] = _rename_stage_value_keys(item)
    return renamed


def _field_obj_get(field_obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dataclass / 普通对象 / dict 三种字段结果读取方式。"""
    if isinstance(field_obj, dict):
        return field_obj.get(key, default)
    return getattr(field_obj, key, default)


def _to_ordered_items(items: Any) -> list[dict[str, str]]:
    """将历史 `_ITEMS` 收敛成统一原子项列表。"""
    result: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "") or "").strip()
        item_value = str(item.get("value", "") or "").strip()
        if not item_type or not item_value:
            continue
        result.append(OrderedValueItem(type=item_type, value=item_value).to_dict())
    return result


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
    - _encode_size_multi(values) -> EncodedFieldResult
    - _encode_thickness_value(value) -> str
    - _process_single_value(field_type, value) -> dict
    - _process_standard_multi(values, modifier_map) -> EncodedFieldResult
    """
    
    DEFAULT_FIELD_ORDER = ['TYPE', 'RADIUS', 'ENDS', 'MANU', 'CONN', 'SEAL', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
    TYPE_COMBINED_FIELDS = ['TYPE', 'RADIUS', 'ENDS', 'SEAL', 'MANU', 'CONN']
    STANDARD_MODIFIER_FIELDS = ['STANDARD_GRADE', 'STANDARD_APPENDIX', 'STANDARD_METHOD']
    
    def __init__(self):
        self.matcher = get_semantic_matcher()
        
        config_path = Path(__file__).parent / "config" / "encoder_config.yaml"
        self.config = self._load_config(config_path)

        confidence_config_path = Path(__file__).parent.parent / "config" / "confidence_config.yaml"
        self.confidence_config = self._load_config(confidence_config_path)
        
        platform_config_path = Path(__file__).parent.parent / "config" / "platform_config.yaml"
        self.platform_config = self._load_config(platform_config_path)
        
        self.FIELD_ORDER = self.config.get('field_order', self.DEFAULT_FIELD_ORDER)
        
        self.size_processor = get_size_processor()
        self.standard_processor = get_standard_processor()
        self.thickness_processor = get_thickness_processor()
        self.thickness_table_processor = get_thickness_table_processor()
        self.regex_extractor = get_regex_extractor()
        thickness_mm_cfg = self.config.get('thickness_mm_conversion', {}) or {}
        self.thickness_mm_conversion_enabled = bool(thickness_mm_cfg.get('enabled', False))
        thickness_mm_dedup_cfg = self.config.get('thickness_mm_dedup', {}) or {}
        self.thickness_mm_dedup_enabled = bool(thickness_mm_dedup_cfg.get('enabled', False))
        
        self.semantic_match_fields = set(self.config.get('semantic_match_fields', ['TYPE', 'MATERIAL']))
        self.exact_match_fields = set(self.config.get('exact_match_fields', ['MANU', 'CONN']))
        self.review_rules_config = (
            self.confidence_config.get('review_rules')
            or self.config.get('review_rules')
            or {}
        )
        self.extract_conf_weight = float(self.review_rules_config.get('extract_weight', 0.55))
        self.encode_conf_weight = float(self.review_rules_config.get('encode_weight', 0.45))
        self.review_threshold = float(self.review_rules_config.get('review_threshold', 0.80))
        self.hard_rule_force_review = bool(self.review_rules_config.get('hard_rule_force_review', True))
        calibration_cfg = self.review_rules_config.get('calibration', {}) or {}
        self.conf_calibration_enabled = bool(calibration_cfg.get('enabled', False))
        self.conf_temperature = float(calibration_cfg.get('temperature', 1.8))
        verification_cfg = (
            self.confidence_config.get('field_verification')
            or self.config.get('field_verification')
            or {}
        )
        self.verification_enabled = bool(verification_cfg.get('enabled', False))
        self.verify_fields = set(verification_cfg.get('verify_fields', []))
        self.unverified_penalty = float(verification_cfg.get('unverified_penalty', 0.5))
        self.evidence_rules_cfg = verification_cfg.get('evidence_rules', {}) or {}
        self.whitelist_rules_cfg = verification_cfg.get('whitelist_rules', {}) or {}
    
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

    def _encode_type_value(self, merged_value: str, type_value: Optional[Dict[str, Any]] = None):
        """编码合并后的 TYPE 值，返回 (code, confidence)"""
        raise NotImplementedError

    def _encode_size_multi(self, values: List[str], original_text: str = "") -> EncodedFieldResult:
        raise NotImplementedError

    def _encode_thickness_value(self, value: str, original_text: str = "") -> str:
        raise NotImplementedError

    def _process_single_value(self, field_type: str, value: str) -> dict:
        raise NotImplementedError

    def _process_standard_multi(
        self,
        values: List[str],
        modifier_map: Dict[int, Dict[str, List[str]]] = None,
        original_text: str = "",
    ) -> EncodedFieldResult:
        raise NotImplementedError

    def _process_material_item_structured(self, item: Dict[str, Any]) -> Optional[dict]:
        return None

    # ──────────────────── 公共方法 ────────────────────

    def _preprocess_tee_reducing(self, entities: Dict, original_text: str) -> Dict:
        """
        三通/管箍异径预处理
        
        如果 TYPE 包含三通/tee/管箍，且原始描述不包含"变径"或"异径"，
        则基于 SIZE 的最终编码结果判断是否异径：
        - 编码后若为多段且数值不相同（如 200X150），判定为异径
        - 编码后若收敛为单段（如 DN200 与 8" 收敛为 200），不判定异径
        """
        type_value = entities.get('TYPE', '')
        type_text = self._get_type_body_text(entities)
        
        if (
            '三通' not in type_text
            and 'tee' not in type_text
            and '管箍' not in type_text
        ):
            return entities
        
        text_lower = (original_text or '').lower()
        if '变径' in text_lower or '异径' in text_lower:
            return entities
        
        size_value = entities.get('SIZE', '')
        if self._is_reducing_size_by_encoded(size_value):
            if isinstance(type_value, dict):
                body = str(type_value.get('BODY') or '').strip()
                replaced = self._insert_reducing_before_body(body)
                if replaced and replaced != body:
                    type_value['BODY'] = replaced
                entities['TYPE'] = type_value
            elif isinstance(type_value, list):
                first = str(type_value[0] or '').strip() if type_value else ''
                replaced = self._insert_reducing_before_body(first)
                entities['TYPE'] = [replaced or first] + type_value[1:]
            else:
                raw = str(type_value or '').strip()
                replaced = self._insert_reducing_before_body(raw)
                entities['TYPE'] = replaced or raw
            logger.info(f"[异径补充] 检测到尺寸不等，TYPE: {type_value} -> {entities['TYPE']}")
        
        return entities

    @staticmethod
    def _insert_reducing_before_body(body: str) -> str:
        text = str(body or '').strip()
        if not text or '异径' in text:
            return text
        if '单头管箍' in text:
            return text.replace('单头管箍', '异径单头管箍', 1)
        if '单口管箍' in text:
            return text.replace('单口管箍', '异径单口管箍', 1)
        if '双头管箍' in text:
            return text.replace('双头管箍', '异径双头管箍', 1)
        if '双口管箍' in text:
            return text.replace('双口管箍', '异径双口管箍', 1)
        if '管箍' in text:
            return text.replace('管箍', '异径管箍', 1)
        if '斜三通' in text:
            return text.replace('斜三通', '异径斜三通', 1)
        if '三通' in text:
            return text.replace('三通', '异径三通', 1)
        return text
    
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

    def _get_regex_alias_code(self, label: str, raw_value: Any) -> str:
        """读取 regex_extraction.aliases 中对原始值的编码映射。"""
        text = str(raw_value or '').strip()
        if not text:
            return ""
        regex_cfg = self.config.get('regex_extraction', {}) or {}
        field_cfg = regex_cfg.get(label, {}) or {}
        if not isinstance(field_cfg, dict):
            return ""
        aliases = field_cfg.get('aliases', {}) or {}
        value_upper = text.upper()
        for alias_key, alias_code in aliases.items():
            if value_upper == str(alias_key).strip().upper():
                return str(alias_code or '').strip()
        return ""

    def _normalize_regex_display_value(self, label: str, raw_value: Any, code_value: Any) -> str:
        """
        对命中 aliases 的规则提取值，显示层也统一为映射后的编码。
        例如：承插/插焊/承插焊 -> SW
        """
        raw_text = str(raw_value or '').strip()
        code_text = str(code_value or '').strip()
        alias_code = self._get_regex_alias_code(label, raw_text)
        if alias_code and code_text and alias_code.upper() == code_text.upper():
            return code_text
        return raw_text
    
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
    def _get_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

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

    def _get_nested_type_geometry_value(self, entities: Dict[str, Any], label: str) -> Any:
        type_dict = self._ensure_type_dict(entities.get('TYPE'))
        if type_dict is None:
            return None
        geometry = self._get_dict(type_dict.get('GEOMETRY'))
        if geometry.get(label):
            return geometry.get(label)
        return None

    def _set_nested_type_geometry_value(self, entities: Dict[str, Any], label: str, value: Any, overwrite: bool = False):
        if value in (None, "", []):
            return
        type_dict = self._ensure_type_dict(entities.get('TYPE'), create=True)
        if entities.get('TYPE') is not type_dict:
            entities['TYPE'] = type_dict
        geometry = self._get_dict(type_dict.get('GEOMETRY'))
        if type_dict.get('GEOMETRY') is not geometry:
            type_dict['GEOMETRY'] = geometry
        if overwrite or not geometry.get(label):
            geometry[label] = value

    def _set_radius_to_type_geometry(self, entities: Dict[str, Any], display_value: str, code_value: str):
        token = (code_value or display_value or '').strip()
        if not token:
            return
        if self._get_nested_type_geometry_value(entities, 'RADIUS'):
            return
        self._set_nested_type_geometry_value(entities, 'RADIUS', token)

    def _flatten_type_value_for_stage2(self, value: Any) -> str:
        type_dict = self._ensure_type_dict(value)
        if type_dict is None:
            if isinstance(value, list):
                parts = [str(v).strip() for v in value if str(v).strip()]
                return ';'.join(parts)
            return str(value or '').strip()

        return self._flatten_type_encoding_key(type_dict)

    def _flatten_type_encoding_key(self, value: Any, separator: str = ';') -> str:
        type_dict = self._ensure_type_dict(value)
        if type_dict is None:
            return str(value or '').strip()

        parts: List[str] = []

        def _extend(raw: Any):
            if not raw:
                return
            values = raw if isinstance(raw, list) else [raw]
            for item in values:
                item_text = str(item).strip()
                if item_text:
                    parts.append(item_text)

        _extend(type_dict.get('FLANGE_STYLE'))

        body = type_dict.get('BODY')
        if isinstance(body, list):
            _extend(body)
        elif body:
            _extend([p.strip() for p in str(body).split(';') if p.strip()])

        geometry = self._get_dict(type_dict.get('GEOMETRY'))
        _extend(geometry.get('ANGLE'))
        _extend(geometry.get('RADIUS'))
        _extend(type_dict.get('SEAL'))

        conn_sources = []
        for source_key in ('CONN', 'ENDS'):
            source_raw = type_dict.get(source_key)
            if source_raw:
                conn_sources.extend(source_raw if isinstance(source_raw, list) else [source_raw])
        _extend(conn_sources)
        _extend(type_dict.get('MANU'))

        deduped: List[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return separator.join(deduped)

    def _build_type_encoding_input(
        self,
        entities: Dict[str, Any],
        regex_value_code_map: Dict[str, Dict]
    ) -> Dict[str, Any]:
        type_dict = self._ensure_type_dict(entities.get('TYPE')) or {}
        geometry = self._get_dict(type_dict.get('GEOMETRY'))

        body_value = type_dict.get('BODY')
        if isinstance(body_value, list):
            body = next((str(v).strip() for v in body_value if str(v).strip()), '')
        else:
            body = str(body_value or '').strip()

        angle = str(geometry.get('ANGLE') or '').strip()

        radius = str(geometry.get('RADIUS') or '').strip()
        if not radius:
            radius_info = regex_value_code_map.get('RADIUS', {}) or {}
            radius = str(radius_info.get('code') or radius_info.get('value') or entities.get('RADIUS') or '').strip()

        def _collect_values(field: str) -> List[str]:
            values: List[str] = []
            source_fields = [field]
            if field == 'CONN':
                source_fields.append('ENDS')

            for source_field in source_fields:
                raw_type_value = type_dict.get(source_field)
                raw_entity_value = entities.get(source_field)
                raw_value = raw_type_value if raw_type_value not in (None, '', []) else raw_entity_value

                if raw_value:
                    candidates = raw_value if isinstance(raw_value, list) else [raw_value]
                    for item in candidates:
                        item_text = str(item).strip()
                        if item_text and item_text not in values:
                            values.append(item_text)

                info = regex_value_code_map.get(source_field, {}) or {}
                code_value = str(info.get('code') or '').strip()
                if code_value and code_value not in values:
                    values.insert(0, code_value)
            return values
        raw_structured = {
            'FLANGE_STYLE': str(type_dict.get('FLANGE_STYLE') or type_dict.get('flange_style') or '').strip(),
            'BODY': body,
            'GEOMETRY': {
                'ANGLE': angle,
                'RADIUS': radius,
            },
            'SEAL': _collect_values('SEAL'),
            'CONN': _collect_values('CONN'),
            'MANU': _collect_values('MANU'),
        }
        return self._filter_type_encoding_input(raw_structured)

    def _get_type_included_text(self, type_input: Dict[str, Any]) -> str:
        body = str(type_input.get('BODY') or '').strip()
        geometry = self._get_dict(type_input.get('GEOMETRY'))
        angle = str(geometry.get('ANGLE') or '').strip()
        radius = str(geometry.get('RADIUS') or '').strip()
        return ' '.join([body, angle, radius]).strip().lower()

    def _is_type_component_included(self, type_text: str, code_value: str, display_value: str) -> bool:
        if not type_text:
            return False
        included_keywords = self.config.get('type_included_keywords', {})
        code_upper = str(code_value or '').strip().upper()
        display_text = str(display_value or '').strip().lower()
        keywords = included_keywords.get(code_upper, [])
        if any(str(kw).lower() in type_text for kw in keywords):
            return True
        if display_text and display_text in type_text:
            return True
        return False

    def _filter_type_component_list(self, type_text: str, values: List[str]) -> List[str]:
        filtered: List[str] = []
        for item in values:
            item_text = str(item or '').strip()
            if not item_text:
                continue
            if self._is_type_component_included(type_text, item_text, item_text):
                logger.debug(f"[TYPE结构化过滤] 跳过组件 '{item_text}'，已包含在 TYPE 中")
                continue
            if item_text not in filtered:
                filtered.append(item_text)
        return filtered

    def _filter_type_encoding_input(self, type_input: Dict[str, Any]) -> Dict[str, Any]:
        geometry = self._get_dict(type_input.get('GEOMETRY'))
        filtered_input = {
            'FLANGE_STYLE': str(type_input.get('FLANGE_STYLE') or type_input.get('flange_style') or '').strip(),
            'BODY': str(type_input.get('BODY') or '').strip(),
            'GEOMETRY': {
                'ANGLE': str(geometry.get('ANGLE') or '').strip(),
                'RADIUS': str(geometry.get('RADIUS') or '').strip(),
            },
            'SEAL': list(type_input.get('SEAL') or []),
            'CONN': list(type_input.get('CONN') or []),
            'MANU': list(type_input.get('MANU') or []),
        }
        type_text = self._get_type_included_text(filtered_input)
        for field in ('SEAL', 'CONN', 'MANU'):
            filtered_input[field] = self._filter_type_component_list(type_text, filtered_input[field])
        # 直管/管子类的端部或连接方式不参与 TYPE 编码，只保留在 stage1_raw 中展示。
        if filtered_input['BODY'].strip().upper() in {'直管', '钢管', '管子', 'PIPE'}:
            filtered_input['CONN'] = []
        return filtered_input

    @staticmethod
    def _format_type_body_for_fallback(body: str, angle: str) -> str:
        body_text = str(body or '').strip()
        angle_text = str(angle or '').strip()
        if angle_text and body_text:
            return f"{angle_text}度{body_text}"
        return body_text or angle_text

    @staticmethod
    def _build_processor_encode_confidence(
        source: str,
        confidence: float,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            'source': source,
            'confidence': max(0.0, min(1.0, float(confidence))),
            'reason': reason,
            'evidence': evidence or {},
        }

    def _aggregate_item_encode_confidence(
        self,
        item_results: List[Dict[str, Any]],
        fallback_source: str = 'unknown',
    ) -> Dict[str, Any]:
        if not item_results:
            return self._build_processor_encode_confidence(
                source=fallback_source,
                confidence=0.0,
                reason='no_item_results',
                evidence={'item_count': 0},
            )

        metas = [r.get('encode_meta') for r in item_results if isinstance(r.get('encode_meta'), dict)]
        if not metas:
            return self._build_processor_encode_confidence(
                source=fallback_source,
                confidence=min((float(r.get('similarity', 0.0)) for r in item_results), default=0.0),
                reason='fallback_from_similarity',
                evidence={'item_count': len(item_results)},
            )

        sources = [str(meta.get('source') or fallback_source) for meta in metas]
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)

        if len(unique_sources) == 1:
            source_name = unique_sources[0]
        else:
            source_name = 'mixed'

        llm_count = sum(1 for source in sources if source == 'llm_fallback')
        mapping_count = sum(1 for source in sources if source.endswith('_mapping'))
        processor_count = sum(1 for source in sources if source.endswith('_processor'))

        return self._build_processor_encode_confidence(
            source=source_name,
            confidence=min((float(meta.get('confidence', 0.0)) for meta in metas), default=0.0),
            reason='multi_item_aggregated',
            evidence={
                'item_count': len(item_results),
                'source_count': len(unique_sources),
                'mapping_item_count': mapping_count,
                'processor_item_count': processor_count,
                'llm_item_count': llm_count,
            },
        )

    @staticmethod
    def _flatten_material_value_for_stage2(value: Any) -> str:
        parts = [
            PipeEncoder._flatten_material_item_for_stage2(item)
            for item in PipeEncoder._normalize_material_entries(value)
        ]
        return ' | '.join([part for part in parts if part]).strip()

    @staticmethod
    def _normalize_material_special_req(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if value in (None, "", []):
            return []
        return [str(value).strip()]

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

        # 新结构: {"ROLE": "...", "VALUE": "...", "SPECIAL_REQ": [...]}
        if 'VALUE' in item or 'ROLE' in item:
            base_value = str(item.get('VALUE') or '').strip()
            special_parts = PipeEncoder._normalize_material_special_req(item.get('SPECIAL_REQ'))
            item_parts = [p for p in [base_value, *special_parts] if p]
            return ' '.join(item_parts).strip()

        # 旧结构: {"EXEC_STANDARD": "...", "GRADE": "...", "SPECIAL_REQ": [...]}
        exec_standard = str(item.get('EXEC_STANDARD') or '').strip()
        grade_code = str(item.get('GRADE') or '').strip()
        special_parts = PipeEncoder._normalize_material_special_req(item.get('SPECIAL_REQ'))

        item_parts = [p for p in [exec_standard, grade_code, *special_parts] if p]
        return ' '.join(item_parts).strip()

    @staticmethod
    def _normalize_material_entries(value: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        if value in (None, "", []):
            return entries

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    role = str(item.get('ROLE') or 'MAIN').strip() or 'MAIN'
                    item_value = str(item.get('VALUE') or '').strip()
                    special_req = PipeEncoder._normalize_material_special_req(item.get('SPECIAL_REQ'))
                    if item_value or special_req:
                        entries.append({
                            'ROLE': role,
                            'VALUE': item_value,
                            'SPECIAL_REQ': special_req,
                        })
                else:
                    item_text = str(item).strip()
                    if item_text:
                        entries.append({
                            'ROLE': 'MAIN',
                            'VALUE': item_text,
                            'SPECIAL_REQ': [],
                        })
            return entries

        if isinstance(value, dict):
            # 新结构单项兜底
            if 'VALUE' in value or 'ROLE' in value:
                role = str(value.get('ROLE') or 'MAIN').strip() or 'MAIN'
                item_value = str(value.get('VALUE') or '').strip()
                special_req = PipeEncoder._normalize_material_special_req(value.get('SPECIAL_REQ'))
                if item_value or special_req:
                    entries.append({
                        'ROLE': role,
                        'VALUE': item_value,
                        'SPECIAL_REQ': special_req,
                    })
                return entries

            # 旧结构兼容
            items = value.get('ITEMS')
            if isinstance(items, list):
                for item in items:
                    item_text = PipeEncoder._flatten_material_item_for_stage2(item)
                    special_req = []
                    if isinstance(item, dict):
                        special_req = PipeEncoder._normalize_material_special_req(item.get('SPECIAL_REQ'))
                        item_text = ' '.join([
                            p for p in [
                                str(item.get('EXEC_STANDARD') or '').strip(),
                                str(item.get('GRADE') or '').strip(),
                            ] if p
                        ]).strip()
                    if item_text or special_req:
                        entries.append({
                            'ROLE': 'MAIN',
                            'VALUE': item_text,
                            'SPECIAL_REQ': special_req,
                        })
                return entries

        scalar_text = str(value).strip()
        if scalar_text:
            entries.append({
                'ROLE': 'MAIN',
                'VALUE': scalar_text,
                'SPECIAL_REQ': [],
            })
        return entries

    def _process_material_structured(self, value: Any) -> EncodedFieldResult:
        relation = ""
        is_legacy_material = isinstance(value, dict) and isinstance(value.get('ITEMS'), list)
        if is_legacy_material:
            relation = self._normalize_material_relation(value.get('RELATION'))

        entries = self._normalize_material_entries(value)
        item_texts = [
            self._flatten_material_item_for_stage2(item)
            for item in entries
            if self._flatten_material_item_for_stage2(item)
        ]

        if not item_texts:
            flattened = self._flatten_material_value_for_stage2(value)
            if not flattened:
                return EncodedFieldResult(field_type='MATERIAL')
            return self._process_field_multi('MATERIAL', [flattened])

        item_results = []
        for entry, item_text in zip(entries, item_texts):
            item_result = self._process_material_item_structured(entry)
            if item_result is None:
                item_result = self._process_single_value('MATERIAL', item_text)
            item_results.append(item_result)
        separator = '/' if relation == 'alternative' else ''

        items = []
        for entry, item_text, item_result in zip(entries, item_texts, item_results):
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
                'role': str(entry.get('ROLE') or 'MAIN').strip() or 'MAIN',
            })

        unique_codes: List[str] = []
        if is_legacy_material:
            seen = set()
            for item_result in item_results:
                code = str(item_result.get('code') or '').strip()
                if code and code not in seen:
                    unique_codes.append(code)
                    seen.add(code)
        else:
            unique_codes = [
                str(item_result.get('code') or '').strip()
                for item_result in item_results
                if str(item_result.get('code') or '').strip()
            ]

        min_similarity = min((r['similarity'] for r in item_results), default=1.0)
        any_need_review = any(r['need_review'] for r in item_results)
        all_exact = all(r['is_exact'] for r in item_results)

        candidates = []
        for item_result in item_results:
            if item_result.get('candidates'):
                candidates = item_result['candidates']
                break

        normalized_stage2_input = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            normalized_stage2_input.append(
                MaterialItem(
                    role=str(entry.get('ROLE') or 'MAIN').strip() or 'MAIN',
                    # stage2_input 表示“实际送入二阶段编码前的结构”，
                    # 这里只保留编码前材质项本身，不再回写最终编码结果。
                    value=str(entry.get('VALUE') or '').strip(),
                    special_req=self._normalize_material_special_req(entry.get('SPECIAL_REQ')),
                ).to_dict()
            )

        return EncodedFieldResult(
            field_type='MATERIAL',
            stage2_input=normalized_stage2_input,
            encode_confidence_v2=self._aggregate_item_encode_confidence(item_results, fallback_source='material_mapping'),
            code=separator.join(unique_codes),
            codes=unique_codes,
            similarity=min_similarity,
            is_exact_match=all_exact,
            need_review=any_need_review,
            candidates=candidates,
            detail_items=items
        )

    def _normalize_field_value_for_stage2(self, field_type: str, value: Any) -> Any:
        if value in (None, "", []):
            return ""
        if field_type == 'TYPE':
            return self._flatten_type_value_for_stage2(value)
        if field_type == 'MATERIAL':
            return self._flatten_material_value_for_stage2(value)
        if field_type == 'STANDARD':
            return self._flatten_standard_value_for_stage2(value)
        return value

    @staticmethod
    def _extract_explicit_thickness_mm_values(value: Any) -> List[float]:
        """仅提取显式 MM 壁厚，用于 SIZE 的 OD→DN 消歧，不处理 SCH/XS/STD。"""
        result: List[float] = []

        def _append_num(raw: Any):
            if raw in (None, ""):
                return
            if isinstance(raw, (int, float)):
                result.append(float(raw))
                return
            text = str(raw).strip().upper()
            if not text:
                return
            # 仅接受显式 mm 或结构化 MM 字段中的纯数字
            m = re.search(r'(\d+(?:\.\d+)?)\s*MM\b', text)
            if m:
                try:
                    result.append(float(m.group(1)))
                except (ValueError, TypeError):
                    pass
                return
            if re.fullmatch(r'\d+(?:\.\d+)?', text):
                try:
                    result.append(float(text))
                except (ValueError, TypeError):
                    pass

        if isinstance(value, list):
            for item in value:
                result.extend(PipeEncoderBase._extract_explicit_thickness_mm_values(item))
            return result

        if isinstance(value, dict):
            for item in value.get('_ITEMS') or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get('type', '')).strip().upper() != 'MM':
                    continue
                _append_num(item.get('value'))
            for item in value.get('MM') or []:
                raw = item.get('value') if isinstance(item, dict) else item
                _append_num(raw)
            return result

        _append_num(value)
        return result

    def _attach_thickness_mm_context_to_size(self, size_value: Any, thickness_value: Any) -> Any:
        """将显式 MM 壁厚作为临时上下文挂到 SIZE 结构上，供尺寸处理器消歧同 OD 多候选行。"""
        mm_values = self._extract_explicit_thickness_mm_values(thickness_value)
        if not mm_values:
            return size_value

        cloned = self._clone_response_value(size_value)
        if isinstance(cloned, dict):
            cloned['_THICKNESS_MM_CONTEXT'] = list(mm_values)
            return cloned
        if isinstance(cloned, list):
            updated = []
            for item in cloned:
                if isinstance(item, dict):
                    entry = self._clone_response_value(item)
                    entry['_THICKNESS_MM_CONTEXT'] = list(mm_values)
                    updated.append(entry)
                else:
                    updated.append(item)
            return updated
        return cloned

    @staticmethod
    def _flatten_standard_value_for_stage2(value: Any) -> Any:
        if value in (None, "", []):
            return ""
        if isinstance(value, list):
            flattened = []
            for item in value:
                body = PipeEncoder._flatten_standard_value_for_stage2(item)
                if body:
                    if isinstance(body, list):
                        flattened.extend([v for v in body if v])
                    else:
                        flattened.append(body)
            return flattened
        if isinstance(value, dict):
            return str(value.get('BODY') or '').strip()
        return str(value).strip()

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

    @staticmethod
    def _split_encoded_parts(value: Any) -> List[str]:
        text = str(value or '').strip()
        if not text:
            return []
        return [p for p in re.split(r'[xX×*/]+', text.replace(' ', '')) if p]

    @staticmethod
    def _format_mm_code(mm_value: str) -> str:
        text = str(mm_value or '').strip().upper()
        if not text:
            return ''
        if text.endswith('MM'):
            return text
        return f'{text}MM'

    @staticmethod
    def _is_mm_code(value: Any) -> bool:
        return bool(re.fullmatch(r'\d+(?:\.\d+)?MM', str(value or '').strip().upper()))

    def _build_thickness_items_for_conversion(self, value: Any, original_text: str = "") -> List[Dict[str, str]]:
        return self.thickness_table_processor.build_thickness_items(value, original_text=original_text)

    @staticmethod
    def _build_thickness_conversion_note(source_part: str, dn: str, formatted: str) -> str:
        source = str(source_part or '').strip().upper()
        target = str(formatted or '').strip().upper()
        dn_text = str(dn or '').strip()
        if not source or not target:
            return ''
        if dn_text:
            return f"壁厚 {source} 按 DN{dn_text} 换算为 {target}"
        return f"壁厚 {source} 换算为 {target}"

    def _apply_thickness_mm_conversion(self, result: PipeEncodingResult):
        """
        使用 STANDARD + SIZE + THICKNESS 查壁厚表，将 Sch/S/Series 转成 mm。

        规则：
        1) 查到则输出 `3.91MM`
        2) 多段换算后若全部相同，只保留一个（如 `SCH30X3.91` -> `3.91MM`）
        3) 完全查不到则保持现有编码原样
        """
        if not self.thickness_mm_conversion_enabled and not self.thickness_mm_dedup_enabled:
            return

        thk_field = result.fields.get('THICKNESS')
        size_field = result.fields.get('SIZE')
        std_field = result.fields.get('STANDARD')
        if not thk_field or not thk_field.code or not size_field or not size_field.code or not std_field:
            return

        standards = std_field.stage1_raw or std_field.detail_items or ''
        if isinstance(size_field.stage1_raw, dict):
            dn_values = size_field.stage1_raw
        else:
            size_code_for_dn = str(size_field.code or '').strip()
            if not size_code_for_dn:
                return

            length_prefix = self.size_processor.extract_length_prefix(
                size_field.stage1_raw,
                original_text=result.original_text,
            )
            if length_prefix and size_code_for_dn.upper().endswith(length_prefix.upper()):
                size_code_for_dn = size_code_for_dn[:-len(length_prefix)]
            dn_values = size_code_for_dn
        thickness_input = thk_field.stage2_input or thk_field.stage1_raw or thk_field.code
        thickness_values = thk_field.code

        thickness_items = self._build_thickness_items_for_conversion(
            thickness_input,
            original_text=result.original_text,
        )
        if not thickness_items:
            thickness_items = [
                {"type": "", "value": part, "normalized": part}
                for part in (self._split_encoded_parts(thickness_values) or [str(thickness_values).strip()])
                if str(part or '').strip()
            ]

        explicit_mm_codes = [
            str(item.get("normalized") or "").strip().upper()
            for item in thickness_items
            if self._is_mm_code(item.get("normalized"))
        ]
        has_explicit_mm = bool(explicit_mm_codes)
        has_schedule_like = any(
            self.thickness_table_processor.is_schedule_like(item.get("normalized"))
            for item in thickness_items
        )

        if self.thickness_mm_dedup_enabled and not self.thickness_mm_conversion_enabled:
            if not (has_explicit_mm and has_schedule_like):
                return

        conversion_details = self.thickness_table_processor.convert_to_mm_details(
            standards,
            dn_values,
            thickness_input,
            original_text=result.original_text,
        )
        if not conversion_details:
            return

        changed = False
        formatted_parts: List[str] = []
        note_lines: List[str] = []
        for detail in conversion_details:
            converted = str(detail.get('converted') or '').strip()
            source_part = str(detail.get('source') or '').strip()
            source_type = str(detail.get('source_type') or '').strip().upper()
            if not converted:
                formatted_parts.append(source_part)
                continue
            formatted = self._format_mm_code(converted) if re.fullmatch(r'\d+(?:\.\d+)?', converted) else converted

            if self.thickness_table_processor.is_schedule_like(source_part):
                if has_explicit_mm and self.thickness_mm_dedup_enabled:
                    if any(
                        self.thickness_table_processor.mm_values_equivalent(formatted, existing_mm)
                        for existing_mm in explicit_mm_codes
                    ):
                        changed = True
                        dn = str(detail.get('dn') or '').strip()
                        note = self._build_thickness_conversion_note(source_part, dn, formatted)
                        if note:
                            note_lines.append(note)
                        continue
                    formatted_parts.append(source_part)
                    continue

                if self.thickness_mm_conversion_enabled and not has_explicit_mm:
                    if formatted.upper() != source_part.upper():
                        changed = True
                        dn = str(detail.get('dn') or '').strip()
                        note = self._build_thickness_conversion_note(source_part, dn, formatted)
                        if note:
                            note_lines.append(note)
                    formatted_parts.append(formatted)
                    continue

                formatted_parts.append(source_part)
                continue

            if source_type == 'MM':
                formatted_parts.append(source_part)
                continue

            formatted_parts.append(source_part)

        if not changed or not formatted_parts:
            return

        deduped_parts: List[str] = []
        for part in formatted_parts:
            if part not in deduped_parts:
                deduped_parts.append(part)

        final_code = deduped_parts[0] if len(deduped_parts) == 1 else 'X'.join(deduped_parts)
        thk_field.code = final_code
        thk_field.codes = [final_code]
        thk_field.notes = list(dict.fromkeys(note_lines))
        result.thickness_conversion_notes = note_lines

        logger.info(
            "[THICKNESS毫米换算] standards=%s, dn=%s, thickness=%s -> %s",
            standards,
            dn_values,
            thickness_values,
            final_code,
        )

    @staticmethod
    def _format_standard_source_for_note(standards: Any) -> str:
        def _format_item(item: Any) -> str:
            if isinstance(item, dict):
                body = str(item.get('BODY') or '').strip()
                grade = str(item.get('GRADE') or '').strip()
                appendix = str(item.get('APPENDIX') or '').strip()
                method = str(item.get('METHOD') or '').strip()
                text = body
                if grade:
                    text = f"{text}({grade})" if text else grade
                if appendix:
                    text = f"{text}-{appendix}" if text else appendix
                if method:
                    text = f"{text}-{method}" if text else method
                return text
            return str(item or '').strip()

        if isinstance(standards, list):
            parts = [_format_item(item) for item in standards]
            parts = [part for part in parts if part]
            return '；'.join(parts) if parts else ''
        return _format_item(standards)

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
        """对未通过原文验证的字段打提示并标记待审，不直接改写二阶段相似度。"""
        if not field_verified:
            return
        for field, payload in field_verified.items():
            if len(payload) >= 3:
                passed, llm_value, reason = payload[0], payload[1], payload[2]
            else:
                passed, llm_value = payload[0], payload[1]
                reason = "未在原文中被正则独立匹配到"
            if passed:
                continue

            target_field = field
            if '.' in field:
                target_field = field.split('.', 1)[0]
            if field not in result.fields and field in self.TYPE_COMBINED_FIELDS:
                target_field = 'TYPE'

            if target_field not in result.fields:
                continue

            self._mark_field_review(
                result, target_field,
                f"{field}='{llm_value}' {reason}，建议人工审查。",
            )
            logger.info(
                f"[字段验证标记] {field}(→{target_field}): "
                f"命中验证问题，已标记待审"
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

        def _encoded_segment_count(field: Optional[EncodedFieldResult]) -> int:
            if not field:
                return 0
            # 仅看编码结果：字段主 code + 多值项 code
            candidates: List[str] = []
            if field.code:
                candidates.append(str(field.code))
            if field.codes:
                candidates.extend([str(c) for c in field.codes if c])
            if field.detail_items:
                for it in field.detail_items:
                    cv = it.get('code') if isinstance(it, dict) else None
                    if cv:
                        candidates.append(str(cv))
            return max((_segment_count(t) for t in candidates), default=0)

        # 1) 壁厚异径时单位不一致
        if mixed_unit_cfg.get('enabled', True) and size_field and thk_field:
            size_text = self._stringify_field_value(
                size_field.code or size_field.stage2_input or size_field.stage1_raw or ""
            ).upper()
            thk_text = self._stringify_field_value(
                thk_field.code or thk_field.stage2_input or thk_field.stage1_raw or ""
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
            material_count = len(mat_field.stage1_raw or []) if isinstance(mat_field.stage1_raw, list) else 0
            if material_count <= 1 and mat_field.detail_items:
                material_count = len(mat_field.detail_items)
            material_relation = ""
            if mat_field.detail_items:
                material_relation = str(mat_field.detail_items[0].get('relation') or '').strip().lower()
            if material_count > 1 and material_relation != 'alternative':
                self._mark_field_review(
                    result,
                    'MATERIAL',
                    multi_material_cfg.get('warning', "识别到多个材质值，建议人工确认主材质/组合材质编码。"),
                    similarity_cap=float(multi_material_cfg.get('similarity_cap', 0.75)),
                )

        # 3) 规范未识别到类型（生产/制造）
        if unknown_std_cfg.get('enabled', True) and std_field and std_field.detail_items:
            unknown_items = [it for it in std_field.detail_items if not (it.get('category') or '').strip()]
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
        """计算整条编码置信度（按字段最终执行度做加权几何平均）。"""
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
            raw_conf = field.field_confidence if field.field_confidence is not None else field.similarity
            s = max(1e-6, min(1.0, float(raw_conf)))
            selected.append((w, s))
            total_w += w

        if not selected:
            sims = [
                max(
                    1e-6,
                    min(1.0, float(f.field_confidence if f.field_confidence is not None else f.similarity))
                )
                for f in result.fields.values()
            ]
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

    def _map_validation_target_field(self, field: str) -> str:
        """将验证失败项映射到字段级置信度所属字段。"""
        target_field = field
        if '.' in field:
            target_field = field.split('.', 1)[0]
        if field not in self.FIELD_ORDER and field in self.TYPE_COMBINED_FIELDS:
            target_field = 'TYPE'
        return target_field

    def _apply_field_verification_to_stage1_confidence(
        self,
        extract_confidence_v2: Optional[Dict[str, Any]],
        field_verified: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """将字段验证失败回写到一阶段置信度，而不是直接修改二阶段编码分。"""
        adjusted = self._clone_response_value(extract_confidence_v2) if isinstance(extract_confidence_v2, dict) else {}
        if not field_verified:
            return adjusted

        issues_by_field: Dict[str, List[str]] = {}
        for field, payload in field_verified.items():
            if not payload or payload[0]:
                continue
            llm_value = payload[1] if len(payload) >= 2 else ""
            reason = payload[2] if len(payload) >= 3 else "未在原文中被正则独立匹配到"
            target_field = self._map_validation_target_field(field)
            issue_text = f"{field}='{llm_value}' {reason}"
            issues_by_field.setdefault(target_field, []).append(issue_text)

        for field_type, issues in issues_by_field.items():
            item = adjusted.get(field_type)
            if not isinstance(item, dict):
                item = {
                    "source": "validation_adjusted",
                    "confidence": 0.0,
                    "reason": "field_missing",
                    "evidence": {},
                }
                adjusted[field_type] = item

            old_conf = 0.0
            try:
                old_conf = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            except Exception:
                old_conf = 0.0

            penalty_factor = self.unverified_penalty ** len(issues)
            new_conf = round(old_conf * penalty_factor, 4)
            item["confidence"] = new_conf

            evidence = item.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {}
                item["evidence"] = evidence

            evidence["base_confidence"] = round(old_conf, 4)
            evidence["validation_issue_count"] = len(issues)
            evidence["validation_penalty_factor"] = round(penalty_factor, 4)
            evidence["validation_reasons"] = "；".join(issues)
            logger.info(
                f"[一阶段置信度惩罚] {field_type}: "
                f"基础分 {old_conf:.4f} → {new_conf:.4f} (×{round(penalty_factor, 4)})"
            )

        return adjusted

    def _get_field_extract_confidence(
        self,
        field_type: str,
        extract_confidence: Optional[Dict[str, Any]],
        extract_confidence_v2: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """读取字段一阶段置信度，优先使用结构化 V2，再回退旧标量。"""
        if extract_confidence_v2 and isinstance(extract_confidence_v2, dict):
            raw_v2 = extract_confidence_v2.get(field_type)
            if isinstance(raw_v2, dict):
                value = raw_v2.get("confidence")
                if value is not None:
                    try:
                        return max(0.0, min(1.0, float(value)))
                    except Exception:
                        pass
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

    @staticmethod
    def _get_field_stage2_confidence(field_result: EncodedFieldResult) -> Optional[float]:
        """读取字段二阶段编码置信度，优先取结构化 V2，再回退当前相似度。"""
        meta = getattr(field_result, 'encode_confidence_v2', {}) or {}
        if isinstance(meta, dict):
            value = meta.get("confidence")
            if value is not None:
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    pass
        try:
            return max(0.0, min(1.0, float(field_result.similarity)))
        except Exception:
            return None

    def _set_field_confidence_triplet(
        self,
        field_result: EncodedFieldResult,
        stage1_confidence: Optional[float],
        stage2_confidence: Optional[float],
    ):
        """回填字段级三层执行度：一阶段、二阶段、字段最终。"""
        field_result.stage1_confidence = stage1_confidence
        field_result.stage2_confidence = stage2_confidence
        try:
            field_result.field_confidence = max(0.0, min(1.0, float(field_result.similarity)))
        except Exception:
            field_result.field_confidence = None

    def _refresh_field_confidence_snapshot(self, result: PipeEncodingResult):
        """在所有审查/封顶规则执行后，同步字段最终执行度。"""
        for field in result.fields.values():
            try:
                field.field_confidence = max(0.0, min(1.0, float(field.similarity)))
            except Exception:
                field.field_confidence = None

    # ──────────────────── TYPE 组合处理（公共逻辑） ────────────────────

    def _process_type_combined(
        self,
        entities: Dict[str, Any],
        regex_value_code_map: Dict[str, Dict]
    ) -> EncodedFieldResult:
        """合并 TYPE 新结构及规则补提字段，统一编码。"""
        collected_values = []
        type_dict = self._ensure_type_dict(entities.get('TYPE'))

        if type_dict is not None:
            geometry = self._get_dict(type_dict.get('GEOMETRY'))
            angle_value = str(geometry.get('ANGLE') or '').strip()
            body_value = type_dict.get('BODY')
            body_values = body_value if isinstance(body_value, list) else [body_value]
            for v in body_values:
                if v and str(v).strip():
                    body_text = str(v).strip()
                    fallback_text = self._format_type_body_for_fallback(body_text, angle_value)
                    collected_values.append(('TYPE', fallback_text, fallback_text))

            radius_value = str(geometry.get('RADIUS') or '').strip()
            if not radius_value:
                radius_info = regex_value_code_map.get('RADIUS', {}) or {}
                radius_value = str(radius_info.get('code') or radius_info.get('value') or entities.get('RADIUS') or '').strip()
            if radius_value:
                collected_values.append(('RADIUS', radius_value, radius_value))

            for f in ('SEAL', 'MANU', 'CONN'):
                if f == 'CONN':
                    raw_sources = []
                    for source_key in ('CONN', 'ENDS'):
                        source_value = type_dict.get(source_key) or entities.get(source_key)
                        if source_value:
                            raw_sources.extend(source_value if isinstance(source_value, list) else [source_value])
                    raw_value = raw_sources
                else:
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
            return EncodedFieldResult(field_type='TYPE')
        
        type_values = [display for f, display, code in collected_values if f == 'TYPE']
        type_text = ' '.join(type_values).lower() if type_values else ''

        filtered_codes = []
        filtered_displays = []
        for f, display, code in collected_values:
            if f == 'TYPE':
                filtered_codes.append(code)
                filtered_displays.append(display)
            else:
                is_included = self._is_type_component_included(type_text, code, display)
                if not is_included:
                    filtered_codes.append(code)
                    filtered_displays.append(display)
                else:
                    logger.debug(f"[TYPE合并] 跳过 {f}='{display}'，已包含在 TYPE 中")
        
        if not filtered_codes:
            return EncodedFieldResult(field_type='TYPE')
        
        type_encoding_input = self._build_type_encoding_input(entities, regex_value_code_map)
        merged_value = self._flatten_type_encoding_key(type_encoding_input)
        original_parts = self._flatten_type_encoding_key(type_encoding_input, separator=' | ')

        logger.info(f"[TYPE合并] 合并字段: {[f'{f}={d}({c})' for f, d, c in collected_values]}")
        logger.info(f"[TYPE合并] 编码用: '{merged_value}', 显示用: '{original_parts}'")

        code, confidence = self._encode_type_value(merged_value, type_encoding_input)
        
        return EncodedFieldResult(
            field_type='TYPE',
            stage2_input=self._clone_response_value(type_encoding_input),
            encode_confidence_v2=getattr(self, '_last_type_encode_meta', {}) or {},
            code=code,
            codes=[code] if code else [],
            similarity=confidence,
            is_exact_match=True,
            need_review=confidence < 0.8,
            candidates=[],
        )

    # ──────────────────── 多值字段处理（公共逻辑） ────────────────────

    def _process_field_multi(self, field_type: str, values: List[str], original_text: str = "") -> EncodedFieldResult:
        """处理多值字段"""
        if not values:
            return EncodedFieldResult(field_type=field_type)

        if field_type == 'MATERIAL' and len(values) == 1 and isinstance(values[0], (dict, list)):
            return self._process_material_structured(values[0])
        
        if field_type == 'SIZE':
            return self._encode_size_multi(values, original_text=original_text)
        
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
            input_values = [self._stringify_field_value(v) for v in values if self._stringify_field_value(v)]

            return EncodedFieldResult(
                field_type=field_type,
                stage2_input=self._clone_response_value(values[0] if len(values) == 1 else values),
                encode_confidence_v2=self._build_processor_encode_confidence(
                    source='thickness_processor',
                    confidence=0.96 if processed else 0.0,
                    reason='thickness_processor_resolved' if processed else 'thickness_processor_failed',
                    evidence={
                        'value_present': bool(input_values),
                        'item_count': len(input_values),
                        'code_present': bool(processed),
                    },
                ),
                code=processed, codes=[processed] if processed else [],
                similarity=1.0, is_exact_match=True, need_review=False,
                candidates=[],
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
            
            return EncodedFieldResult(
                field_type=field_type,
                stage2_input=self._clone_response_value(values),
                encode_confidence_v2=self._build_processor_encode_confidence(
                    source='thickness_processor',
                    confidence=0.96 if processed else 0.0,
                    reason='thickness_processor_resolved' if processed else 'thickness_processor_failed',
                    evidence={
                        'value_present': bool(values),
                        'item_count': len(values),
                        'code_present': bool(processed),
                    },
                ),
                code=processed, codes=[processed] if processed else [],
                similarity=1.0, is_exact_match=True, need_review=False,
                candidates=[],
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
        
        return EncodedFieldResult(
            field_type=field_type,
            stage2_input=self._clone_response_value(values if len(values) != 1 else values[0]),
            encode_confidence_v2=self._aggregate_item_encode_confidence(results, fallback_source=f'{field_type.lower()}_processor'),
            code=''.join(unique_codes),
            codes=unique_codes,
            similarity=min_similarity,
            is_exact_match=all_exact,
            need_review=any_need_review,
            candidates=candidates,
            detail_items=items
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
                self._set_radius_to_type_geometry(entities, info['value'], info['code'])
                regex_value_code_map[label] = info
            elif label in ('MANU', 'CONN', 'ENDS', 'SEAL'):
                display_value = self._normalize_regex_display_value(
                    label,
                    info.get('value', ''),
                    info.get('code', ''),
                )
                if display_value and display_value != str(info.get('value', '') or '').strip():
                    self._set_nested_type_value(entities, label, display_value, overwrite=True)
                elif not self._get_nested_type_value(entities, label):
                    self._set_nested_type_value(entities, label, display_value or info.get('value'))
                regex_value_code_map[label] = {
                    'value': display_value or info.get('value', ''),
                    'code': info.get('code', ''),
                }
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
        if not self.verification_enabled or not original_text:
            return verified

        regex_results = self.regex_extractor.extract(original_text, exclude_ranges=[])
        regex_found: Dict[str, set] = {}
        for extraction in regex_results:
            if extraction.label not in regex_found:
                regex_found[extraction.label] = set()
            regex_found[extraction.label].add(extraction.value.upper())
            regex_found[extraction.label].add(extraction.code.upper())

        def _normalize_verification_values(value: Any) -> List[str]:
            if value in (None, "", []):
                return []
            if isinstance(value, list):
                normalized: List[str] = []
                for item in value:
                    normalized.extend(_normalize_verification_values(item))
                return normalized
            text = str(value).strip()
            return [text] if text else []

        def _record_failure(field: str, values: List[str], reason: str) -> None:
            if not values:
                return
            joined = ' | '.join(values)
            existing = verified.get(field)
            if existing and not existing[0]:
                old_values = [v.strip() for v in str(existing[1]).split('|') if v.strip()]
                merged_values: List[str] = []
                for value in [*old_values, *values]:
                    if value not in merged_values:
                        merged_values.append(value)
                old_reason = str(existing[2]) if len(existing) >= 3 else ""
                merged_reason = old_reason
                if reason and reason not in old_reason:
                    merged_reason = f"{old_reason}；{reason}" if old_reason else reason
                verified[field] = (False, ' | '.join(merged_values), merged_reason)
                return
            verified[field] = (False, joined, reason)

        material_special_req_aliases = {
            str(key).strip().upper(): [
                str(alias).strip() for alias in aliases or [] if str(alias).strip()
            ]
            for key, aliases in (
                (self.evidence_rules_cfg.get('material_special_req_requires_text') or {}).get('aliases', {}) or {}
            ).items()
        }
        type_manu_aliases = {
            str(key).strip().upper(): [
                str(alias).strip() for alias in aliases or [] if str(alias).strip()
            ]
            for key, aliases in (
                (self.evidence_rules_cfg.get('type_manu_requires_text') or {}).get('aliases', {}) or {}
            ).items()
        }

        def _text_contains_any_alias(aliases: List[str]) -> bool:
            text = original_text or ""
            text_upper = text.upper()
            for alias in aliases:
                alias_upper = alias.upper()
                if re.search(r'[A-Z0-9#/\-"]', alias_upper):
                    if alias_upper in text_upper:
                        return True
                else:
                    if alias and alias in text:
                        return True
            return False

        for field in self.verify_fields:
            if field == 'MANU' and (self.evidence_rules_cfg.get('type_manu_requires_text') or {}).get('enabled', False):
                continue
            field_value = self._get_nested_type_value(entities, field)
            if not field_value:
                continue

            candidate_values = _normalize_verification_values(field_value)
            if not candidate_values:
                continue

            found_values = regex_found.get(field, set())
            missing_values = [value for value in candidate_values if value.upper() not in found_values]
            found = len(missing_values) == 0
            llm_value = ' | '.join(candidate_values)
            verified_value = ' | '.join(missing_values) if missing_values else llm_value
            verified[field] = (found, verified_value, "")
            if not found:
                logger.info(
                    f"[字段验证] {field}='{verified_value}' 未被正则从原文独立匹配到，置信度将惩罚"
                )
            else:
                logger.debug(f"[字段验证] {field}='{llm_value}' 原文正则验证通过")

        size_dn_rule = self.evidence_rules_cfg.get('size_dn_requires_anchor') or {}
        if size_dn_rule.get('enabled', False):
            size_dict = entities.get('SIZE') if isinstance(entities.get('SIZE'), dict) else {}
            dn_values = _normalize_verification_values(size_dict.get('DN')) if size_dict else []
            if dn_values:
                dn_anchor_pattern = size_dn_rule.get('dn_anchor_pattern', r'(?i)\bDN\s*\d')
                if not re.search(dn_anchor_pattern, original_text or ''):
                    verified_value = ' | '.join(dn_values)
                    _record_failure('SIZE.DN', dn_values, '原文缺少 DN 锚点')
                    logger.info(
                        f"[字段验证] SIZE.DN='{verified_value}' 原文缺少 DN 锚点，置信度将惩罚"
                    )

        material_req_rule = self.evidence_rules_cfg.get('material_special_req_requires_text') or {}
        if material_req_rule.get('enabled', False):
            missing_special_req: List[str] = []
            for entry in self._normalize_material_entries(entities.get('MATERIAL')):
                for req in self._normalize_material_special_req(entry.get('SPECIAL_REQ')):
                    aliases = material_special_req_aliases.get(req.upper(), [req])
                    if not _text_contains_any_alias(aliases):
                        missing_special_req.append(req)
            if missing_special_req:
                verified_value = ' | '.join(missing_special_req)
                _record_failure('MATERIAL.SPECIAL_REQ', missing_special_req, '原文缺少特殊要求证据')
                logger.info(
                    f"[字段验证] MATERIAL.SPECIAL_REQ='{verified_value}' 原文缺少特殊要求证据，置信度将惩罚"
                )

        manu_rule = self.evidence_rules_cfg.get('type_manu_requires_text') or {}
        if manu_rule.get('enabled', False):
            manu_values = _normalize_verification_values(self._get_nested_type_value(entities, 'MANU'))
            if manu_values:
                missing_manu: List[str] = []
                for manu in manu_values:
                    aliases = type_manu_aliases.get(manu.upper(), [manu])
                    if not _text_contains_any_alias(aliases):
                        missing_manu.append(manu)
                if missing_manu:
                    verified_value = ' | '.join(missing_manu)
                    _record_failure('MANU', missing_manu, '原文缺少工艺证据')
                    logger.info(
                        f"[字段验证] MANU='{verified_value}' 原文缺少工艺证据，置信度将惩罚"
                    )

        type_whitelist_rule = self.whitelist_rules_cfg.get('type_subfields') or {}
        if type_whitelist_rule.get('enabled', False):
            for subtype in ('SEAL', 'MANU', 'CONN'):
                allowed_raw = type_whitelist_rule.get(subtype) or []
                allowed = {str(item).strip().upper() for item in allowed_raw if str(item).strip()}
                if not allowed:
                    continue
                values = _normalize_verification_values(self._get_nested_type_value(entities, subtype))
                if not values:
                    continue
                invalid = [value for value in values if value.upper() not in allowed]
                if not invalid:
                    continue
                _record_failure(subtype, invalid, '不在白名单中')
                logger.info(
                    f"[字段验证] {subtype}='{ ' | '.join(invalid) }' 不在白名单中，置信度将惩罚"
                )
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

        # 新 schema: STANDARD 为对象数组 [{BODY, GRADE, APPENDIX, METHOD}]
        if isinstance(standards, list):
            embedded_modifier_map = {
                'GRADE': 'STANDARD_GRADE',
                'APPENDIX': 'STANDARD_APPENDIX',
                'METHOD': 'STANDARD_METHOD',
            }
            for idx, standard_item in enumerate(standards):
                if not isinstance(standard_item, dict):
                    continue
                for src_key, dst_key in embedded_modifier_map.items():
                    raw_val = standard_item.get(src_key)
                    if raw_val in (None, "", []):
                        continue
                    vals = raw_val if isinstance(raw_val, list) else [raw_val]
                    field_map = modifier_map.setdefault(idx, {})
                    bucket = field_map.setdefault(dst_key, [])
                    for val in vals:
                        text = str(val).strip()
                        if text and text not in bucket:
                            bucket.append(text)

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
        stage1_final_snapshot: Dict[str, Any],
        extract_confidence: Optional[Dict[str, Any]],
        extract_confidence_v2: Optional[Dict[str, Any]],
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
            if field_type == 'MATERIAL' and isinstance(raw_value, (dict, list)):
                field_result = self._process_material_structured(raw_value)

                field_extract_conf = self._get_field_extract_confidence(
                    field_type,
                    extract_confidence,
                    extract_confidence_v2,
                )
                field_result.similarity = self._blend_extract_and_encode_conf(
                    field_extract_conf,
                    field_result.similarity
                )
                field_result.similarity = self._temperature_scale_confidence(field_result.similarity)
                self._set_field_confidence_triplet(
                    field_result,
                    field_extract_conf,
                    self._get_field_stage2_confidence(field_result),
                )

                if field_type in raw_entities_snapshot:
                    field_result.stage1_raw = self._clone_response_value(raw_entities_snapshot.get(field_type))
                if field_type in stage1_final_snapshot:
                    if field_result.stage2_input in ("", None, [], {}):
                        field_result.stage2_input = self._clone_response_value(stage1_final_snapshot.get(field_type))

                result.fields[field_type] = field_result

                if field_result.need_review and field_type not in result.review_fields:
                    result.review_fields.append(field_type)
                if not field_result.code and field_type not in result.missing_fields:
                    result.missing_fields.append(field_type)
                if field_result.similarity < result.min_similarity:
                    result.min_similarity = field_result.similarity
                continue

            if field_type == 'SIZE':
                raw_value = self._attach_thickness_mm_context_to_size(
                    raw_value,
                    entities.get('THICKNESS'),
                )

            raw_value = self._normalize_field_value_for_stage2(field_type, raw_value)
            if not raw_value:
                if field_type in {'SIZE', 'THICKNESS', 'PRESSURE'}:
                    raw_stage1 = raw_entities_snapshot.get(field_type)
                    final_stage1 = stage1_final_snapshot.get(field_type)
                    if raw_stage1 not in (None, "", [], {}) or final_stage1 not in (None, "", [], {}):
                        logger.warning(
                            "[编码前差异][%s] 一阶段原始识别有值，但送编码前为空。raw_stage1=%s | stage1_final=%s | original_text=%s",
                            field_type,
                            raw_stage1,
                            final_stage1,
                            original_text,
                        )
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
                field_result = EncodedFieldResult(
                    field_type=field_type,
                    stage2_input=copy.deepcopy(info.get('value')),
                    encode_confidence_v2={
                        'source': 'regex_direct',
                        'confidence': 0.98,
                        'reason': 'regex_direct_match',
                        'evidence': {
                            'value_present': bool(info.get('value')),
                            'code_present': bool(info.get('code')),
                        }
                    },
                    code=info['code'],
                    codes=[info['code']],
                    similarity=1.0,
                    is_exact_match=True,
                    need_review=False
                )
            elif field_type == 'STANDARD':
                modifier_map = entities.get('_STANDARD_MODIFIER_MAP', {})
                field_result = self._process_standard_multi(values, modifier_map, original_text=original_text)
            else:
                field_result = self._process_field_multi(field_type, values, original_text=original_text)

            if field_type in {'SIZE', 'THICKNESS', 'PRESSURE'} and not field_result.code:
                logger.warning(
                    "[编码结果为空][%s] 送编码值=%s | stage2_input=%s | original_text=%s",
                    field_type,
                    values,
                    field_result.stage2_input,
                    original_text,
                )

            field_extract_conf = self._get_field_extract_confidence(
                field_type,
                extract_confidence,
                extract_confidence_v2,
            )
            field_result.similarity = self._blend_extract_and_encode_conf(
                field_extract_conf,
                field_result.similarity
            )
            field_result.similarity = self._temperature_scale_confidence(field_result.similarity)
            self._set_field_confidence_triplet(
                field_result,
                field_extract_conf,
                self._get_field_stage2_confidence(field_result),
            )

            if field_type in raw_entities_snapshot:
                field_result.stage1_raw = self._clone_response_value(raw_entities_snapshot.get(field_type))
            if field_type in stage1_final_snapshot:
                if field_result.stage2_input in ("", None, [], {}):
                    field_result.stage2_input = self._clone_response_value(stage1_final_snapshot.get(field_type))

            result.fields[field_type] = field_result

            if field_result.need_review and field_type not in result.review_fields:
                result.review_fields.append(field_type)
            if not field_result.code and field_type not in result.missing_fields:
                result.missing_fields.append(field_type)
            if field_result.similarity < result.min_similarity:
                result.min_similarity = field_result.similarity

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
        extract_confidence_v2: Optional[Dict[str, Any]] = None,
        stage1_raw_snapshot: Optional[Dict[str, Any]] = None,
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

        raw_entities_snapshot = self._clone_response_value(stage1_raw_snapshot) if isinstance(stage1_raw_snapshot, dict) else (
            self._clone_response_value(entities) if isinstance(entities, dict) else {}
        )
        
        regex_value_code_map = {}
        
        self._fix_size_thickness_split(entities)
        self._augment_entities_from_text(entities, original_text, regex_value_code_map)
        self._normalize_standard_body_grade_suffix(entities)
        entities['_STANDARD_MODIFIER_MAP'] = self._build_standard_modifier_map(entities, original_text)
        
        entities = self._preprocess_tee_reducing(entities, original_text)
        stage1_final_snapshot = self._clone_response_value(entities) if isinstance(entities, dict) else {}
        field_verified = self._validate_fields_against_text(entities, original_text)
        adjusted_extract_confidence_v2 = self._apply_field_verification_to_stage1_confidence(
            extract_confidence_v2,
            field_verified,
        )
        result.extract_confidence_v2 = self._clone_response_value(adjusted_extract_confidence_v2)
        
        type_combined_processed = False
        if self._should_use_type_combined():
            type_combined_result = self._process_type_combined(entities, regex_value_code_map)
            if type_combined_result.code:
                type_extract_conf = self._get_field_extract_confidence(
                    'TYPE',
                    extract_confidence,
                    adjusted_extract_confidence_v2,
                )
                type_combined_result.similarity = self._blend_extract_and_encode_conf(
                    type_extract_conf,
                    type_combined_result.similarity
                )
                type_combined_result.similarity = self._temperature_scale_confidence(type_combined_result.similarity)
                self._set_field_confidence_triplet(
                    type_combined_result,
                    type_extract_conf,
                    self._get_field_stage2_confidence(type_combined_result),
                )
                if 'TYPE' in raw_entities_snapshot:
                    type_combined_result.stage1_raw = self._clone_response_value(raw_entities_snapshot.get('TYPE'))
                if 'TYPE' in stage1_final_snapshot:
                    if type_combined_result.stage2_input in ("", None, [], {}):
                        type_combined_result.stage2_input = self._clone_response_value(stage1_final_snapshot.get('TYPE'))
                result.fields['TYPE'] = type_combined_result
                type_combined_processed = True
                if type_combined_result.need_review and 'TYPE' not in result.review_fields:
                    result.review_fields.append('TYPE')
                if type_combined_result.similarity < result.min_similarity:
                    result.min_similarity = type_combined_result.similarity

        self._encode_fields(
            entities,
            raw_entities_snapshot,
            stage1_final_snapshot,
            extract_confidence,
            adjusted_extract_confidence_v2,
            regex_value_code_map,
            result,
            type_combined_processed,
            original_text=original_text,
        )

        self._apply_thickness_mm_conversion(result)

        self._apply_field_verification_penalty(result, field_verified)
        self._apply_review_rules(result)
        self._refresh_field_confidence_snapshot(result)
        
        self._assemble_code(result)
        result.success = bool(result.final_code)
        raw_conf = self._compute_result_confidence(result)
        result.confidence = round(self._temperature_scale_confidence(raw_conf), 4)
        self._finalize_review_decision(result)
        return result

    @staticmethod
    def _normalize_standard_body_grade_suffix(entities: Dict[str, Any]) -> None:
        """窄归一：仅将 STANDARD.BODY 末尾 IA 规范化为 Ia，不拆字段。"""
        standards = entities.get('STANDARD')
        if not isinstance(standards, list):
            return
        for item in standards:
            if not isinstance(item, dict):
                continue
            body = str(item.get('BODY') or '').strip()
            if not body:
                continue
            if re.search(r'\dIA$', body):
                item['BODY'] = re.sub(r'IA$', 'Ia', body)

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
    返回统一的 LLM 管道编码器实现。
    """
    global _encoder_instance
    if _encoder_instance is None:
        from .pipe_encoder_llm import LlmPipeEncoder
        _encoder_instance = LlmPipeEncoder()
        logger.info("编码器: LlmPipeEncoder")
    
    return _encoder_instance
