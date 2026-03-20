# -*- coding: utf-8 -*-
"""
信息补全器
负责推断缺失的材料属性
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .code_mapping import CodeMappingManager, get_mapping_manager
from .prompts import build_info_completion_prompt

logger = logging.getLogger(__name__)


@dataclass
class FieldInfo:
    """字段信息"""
    value: str                  # 字段值
    is_inferred: bool = False   # 是否是推断的
    confidence: float = 1.0     # 置信度 (0-1)
    reason: str = ""            # 来源/推断原因
    is_new: bool = False        # 是否是新值（不在标准列表中）


@dataclass
class CompletionResult:
    """补全结果"""
    name: FieldInfo = None
    material: FieldInfo = None
    type: FieldInfo = None
    spec: FieldInfo = None
    
    # 处理信息
    missing_fields: List[str] = field(default_factory=list)
    inferred_fields: List[str] = field(default_factory=list)
    failed_fields: List[str] = field(default_factory=list)
    used_llm: bool = False


class InfoCompleter:
    """
    信息补全器
    
    处理流程：
    1. 检查哪些字段缺失
    2. 尝试从规则推断（如仪表桥架->槽式）
    3. 如果规则无法推断，调用LLM
    """
    
    # 必需字段
    REQUIRED_FIELDS = ['name', 'material', 'type', 'spec']
    # 可推断的字段（规格不可推断，必须从描述中提取）
    INFERABLE_FIELDS = ['name', 'material', 'type']
    
    def __init__(
        self,
        mapping_manager: CodeMappingManager = None,
        llm_service = None
    ):
        """
        初始化补全器
        
        Args:
            mapping_manager: 编码映射管理器
            llm_service: LLM服务实例
        """
        self.mapping = mapping_manager or get_mapping_manager()
        self.llm = llm_service
    
    def set_llm_service(self, llm_service):
        """设置LLM服务"""
        self.llm = llm_service
    
    async def complete(
        self,
        entities: Dict[str, str],
        use_llm: bool = True,
        original_text: str = ""
    ) -> CompletionResult:
        """
        补全缺失的实体信息
        
        Args:
            entities: 已识别的实体 {"name": "桥架", "material": "铝合金", ...}
            use_llm: 是否使用LLM补全
            original_text: 完整的原始材料描述（用于LLM推断上下文）
            
        Returns:
            补全结果
        """
        result = CompletionResult()
        self._original_text = original_text  # 保存原始描述供后续使用
        
        # 初始化已有字段
        for field in self.REQUIRED_FIELDS:
            value = entities.get(field)
            if value and value.strip():
                setattr(result, field, FieldInfo(
                    value=value.strip(),
                    is_inferred=False,
                    confidence=1.0,  # NER识别的置信度为1.0
                    reason="NER识别"
                ))
            else:
                result.missing_fields.append(field)
        
        # 如果没有缺失字段，直接返回
        if not result.missing_fields:
            return result
        
        # 获取已知信息用于推断
        known_info = {
            f: getattr(result, f).value 
            for f in self.REQUIRED_FIELDS 
            if getattr(result, f) is not None
        }
        
        # Step 1: 尝试规则推断（只推断可推断的字段，不包括spec）
        self._infer_by_rules(result, known_info)
        
        # 更新缺失字段列表（只推断可推断的字段）
        still_missing = [
            f for f in result.missing_fields 
            if getattr(result, f) is None and f in self.INFERABLE_FIELDS
        ]
        
        # Step 2: 如果还有缺失，尝试LLM推断（不推断规格）
        if still_missing and use_llm and self.llm:
            await self._infer_with_llm(result, known_info, still_missing)
            result.used_llm = True
        
        # 更新最终缺失列表
        result.failed_fields = [
            f for f in result.missing_fields 
            if getattr(result, f) is None
        ]
        
        return result
    
    def _infer_by_rules(
        self, 
        result: CompletionResult, 
        known_info: Dict[str, str]
    ):
        """
        使用规则推断
        
        Args:
            result: 补全结果对象
            known_info: 已知信息
        """
        name = known_info.get('name', '')
        # 获取完整描述，用于关键词匹配
        original_text = getattr(self, '_original_text', '')
        
        # 根据名称推断类型
        if 'type' in result.missing_fields:
            # 先尝试从完整描述中匹配关键词
            inferred_type = None
            matched_key = None
            for key in self.mapping.get_inference_keys('name_to_type'):
                if key in original_text:
                    inferred_type = self.mapping.get_inferred_type(key)
                    matched_key = key
                    break
            
            # 如果完整描述没匹配上，再尝试NER识别的名称
            if not inferred_type:
                inferred_type = self.mapping.get_inferred_type(name)
                matched_key = name
            
            if inferred_type:
                result.type = FieldInfo(
                    value=inferred_type,
                    is_inferred=True,
                    confidence=0.9,  # 规则推断置信度
                    reason=f"规则推断: {matched_key} -> {inferred_type}"
                )
                result.inferred_fields.append('type')
        
        # 根据名称推断材质
        if 'material' in result.missing_fields:
            # 先尝试从完整描述中匹配关键词
            inferred_material = None
            matched_key = None
            for key in self.mapping.get_inference_keys('name_to_material'):
                if key in original_text:
                    inferred_material = self.mapping.get_inferred_material(key)
                    matched_key = key
                    break
            
            # 如果完整描述没匹配上，再尝试NER识别的名称
            if not inferred_material:
                inferred_material = self.mapping.get_inferred_material(name)
                matched_key = name
            
            if inferred_material:
                result.material = FieldInfo(
                    value=inferred_material,
                    is_inferred=True,
                    confidence=0.9,  # 规则推断置信度
                    reason=f"规则推断: {matched_key} -> {inferred_material}"
                )
                result.inferred_fields.append('material')
    
    async def _infer_with_llm(
        self,
        result: CompletionResult,
        known_info: Dict[str, str],
        missing_fields: List[str]
    ):
        """
        使用LLM推断
        
        Args:
            result: 补全结果对象
            known_info: 已知信息
            missing_fields: 缺失字段列表
        """
        try:
            # 准备选项
            options = {}
            for field in missing_fields:
                if field == 'name':
                    options[field] = self.mapping.get_all_categories('name')
                elif field == 'material':
                    options[field] = self.mapping.get_all_categories('material')
                elif field == 'type':
                    options[field] = self.mapping.get_all_categories('type')
            
            # 构建提示词（包含完整原始描述）
            messages = build_info_completion_prompt(
                known_info=known_info,
                missing_fields=missing_fields,
                options=options,
                original_text=getattr(self, '_original_text', '')
            )
            
            # 记录完整的prompt
            logger.info(f"[LLM推断] 完整Prompt:\n{'-'*50}")
            for msg in messages:
                logger.info(f"[{msg['role']}]: {msg['content']}")
            logger.info(f"{'-'*50}")
            
            # 调用LLM
            response = await self.llm.chat(messages, format="json")
            llm_result = json.loads(response)
            
            # 记录LLM原始返回
            logger.info(f"[LLM推断] 响应: {llm_result}")
            
            inferred = llm_result.get('inferred', {})
            
            # 容错处理：如果LLM错误地使用"字段名"作为key，尝试修复
            if '字段名' in inferred and len(missing_fields) == 1:
                # 只有一个缺失字段时，可以安全地将"字段名"映射到实际字段
                actual_field = missing_fields[0]
                inferred[actual_field] = inferred.pop('字段名')
                logger.info(f"[LLM推断] 修正字段名: 字段名 -> {actual_field}")
            cannot_infer = llm_result.get('cannot_infer', [])
            
            # 处理推断结果（只处理可推断的字段）
            # 检查是否是扁平格式（reason/confidence 与字段同级）
            top_level_reason = inferred.get('reason', '')
            top_level_conf = inferred.get('confidence')
            
            for field, info in inferred.items():
                # 跳过 reason 和 confidence 字段本身
                if field in ['reason', 'confidence']:
                    continue
                    
                if field in missing_fields and field in self.INFERABLE_FIELDS:
                    # 兼容多种格式
                    if isinstance(info, str):
                        # 格式1: {'type': '槽式', 'reason': '...', 'confidence': 0.5}
                        # 或格式2: {'type': {'value': '...'}} 但 value 也可能是字符串
                        value = info
                        # 尝试使用顶层的 reason 和 confidence
                        reason = top_level_reason
                        if isinstance(top_level_conf, (int, float)):
                            confidence = float(top_level_conf)
                        else:
                            confidence = 0.7
                    elif isinstance(info, dict):
                        # 格式3: {'type': {'value': '槽式', 'reason': '...', 'confidence': 0.5}}
                        value = info.get('value')
                        raw_conf = info.get('confidence')
                        if isinstance(raw_conf, (int, float)):
                            confidence = float(raw_conf)
                        else:
                            confidence = 0.7
                        reason = info.get('reason', '')
                    else:
                        continue
                    
                    # 过滤无效值（LLM 可能把 cannot_infer 当作值返回）
                    invalid_values = ['cannot_infer', '无法推断', '不能确定', '不能归类', 'null', 'none', '']
                    if value and value.lower() in [v.lower() for v in invalid_values]:
                        logger.info(f"[LLM推断] {field}: 跳过无效值 '{value}'，保留原因: {reason}")
                        # 记录失败原因，即便值无效
                        setattr(result, field, FieldInfo(
                            value=None, 
                            is_inferred=True, 
                            confidence=confidence, 
                            reason=reason
                        ))
                        continue
                    
                    # 动态逻辑过滤：防止将属于其他类目的词填入当前字段
                    # 比如防止将产品名称（name）填入结构类型（type）
                    other_terms = []
                    for other_field in ['name', 'material', 'type']:
                        if other_field != field:
                            # 获取其他字段的所有已知词汇（标准类+缓存变体）
                            other_terms.extend(self.mapping.get_all_known_terms(other_field))
                    
                    # 检查是否命中了其他字段的词库
                    if value in other_terms:
                        logger.info(f"[LLM推断] {field}: 拒绝将属于其他字段的值 '{value}' 填入当前字段")
                        continue
                    
                    # 读取 is_new 字段（LLM 标记该值是否不在常见列表中）
                    is_new = False
                    if isinstance(info, dict):
                        is_new = info.get('is_new', False)
                    
                    # 只有置信度 >= 0.3 才接受推断结果
                    if value and confidence >= 0.3:
                        logger.info(f"[LLM推断] {field}: value={value}, confidence={confidence}, reason={reason}, is_new={is_new}")
                        setattr(result, field, FieldInfo(
                            value=value,
                            is_inferred=True,
                            confidence=confidence,
                            reason=reason,  # 直接使用LLM返回的reason
                            is_new=is_new   # 标记是否为新值
                        ))
                        result.inferred_fields.append(field)
                        
        except Exception as e:
            logger.error(f"LLM推断失败: {e}")
    
    def to_dict(self, result: CompletionResult) -> Dict[str, Any]:
        """
        将补全结果转换为字典
        
        Args:
            result: 补全结果
            
        Returns:
            字典格式的结果
        """
        output = {
            'entities': {},
            'missing_fields': result.missing_fields,
            'inferred_fields': result.inferred_fields,
            'failed_fields': result.failed_fields,
            'used_llm': result.used_llm
        }
        
        for field in self.REQUIRED_FIELDS:
            field_info = getattr(result, field)
            if field_info:
                output['entities'][field] = {
                    'value': field_info.value,
                    'is_inferred': field_info.is_inferred,
                    'confidence': field_info.confidence,
                    'reason': field_info.reason
                }
            else:
                output['entities'][field] = None
        
        return output

