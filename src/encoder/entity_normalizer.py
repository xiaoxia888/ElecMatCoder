# -*- coding: utf-8 -*-
"""
实体归一化器
负责将变体描述归一化到标准大类，并转换为编码
"""

import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .code_mapping import CodeMappingManager, get_mapping_manager
from .prompts import build_classification_prompt, build_code_generation_prompt

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """归一化结果"""
    original: str           # 原始值
    category: str           # 标准大类
    code: str               # 编码
    is_cached: bool         # 是否从缓存获取
    is_new_category: bool   # 是否是新类别
    is_new_code: bool       # 是否是新生成的编码
    confidence: float       # 置信度 (0-1)
    reason: str             # 归类/编码原因


class EntityNormalizer:
    """
    实体归一化器
    
    处理流程：
    1. 查询分类缓存
    2. 如果缓存未命中，调用LLM分类
    3. 查询编码映射表
    4. 如果编码不存在，调用LLM生成
    """
    
    def __init__(
        self, 
        mapping_manager: CodeMappingManager = None,
        llm_service = None
    ):
        """
        初始化归一化器
        
        Args:
            mapping_manager: 编码映射管理器
            llm_service: LLM服务实例
        """
        self.mapping = mapping_manager or get_mapping_manager()
        self.llm = llm_service
    
    def set_llm_service(self, llm_service):
        """设置LLM服务"""
        self.llm = llm_service
    
    async def normalize(
        self, 
        entity_type: str, 
        value: str,
        use_llm: bool = True,
        original_text: str = ""
    ) -> NormalizationResult:
        """
        归一化实体值
        
        Args:
            entity_type: 实体类型 (name, material, type)
            value: 实体值
            use_llm: 是否使用LLM（缓存未命中时）
            original_text: 完整的原始材料描述（用于LLM分类上下文）
            
        Returns:
            归一化结果
        """
        self._original_text = original_text  # 保存供后续使用
        if not value or not value.strip():
            return NormalizationResult(
                original=value,
                category=None,
                code=None,
                is_cached=False,
                is_new_category=False,
                is_new_code=False,
                confidence=0.0,
                reason="空值"
            )
        
        value = value.strip()
        
        # Step 1: 检查是否直接是标准大类
        direct_code = self.mapping.get_code(entity_type, value)
        if direct_code:
            return NormalizationResult(
                original=value,
                category=value,
                code=direct_code,
                is_cached=True,
                is_new_category=False,
                is_new_code=False,
                confidence=1.0,
                reason="直接匹配标准大类"
            )
        
        # Step 2: 查询分类缓存
        cached_category = self.mapping.get_cached_category(entity_type, value)
        if cached_category:
            code = self.mapping.get_code(entity_type, cached_category)
            if code:
                return NormalizationResult(
                    original=value,
                    category=cached_category,
                    code=code,
                    is_cached=True,
                    is_new_category=False,
                    is_new_code=False,
                    confidence=0.95,
                    reason=f"从缓存获取: {value} -> {cached_category}"
                )
        
        # Step 3: 调用LLM分类
        if use_llm and self.llm:
            return await self._normalize_with_llm(entity_type, value, original_text)
        
        # 无法归一化
        return NormalizationResult(
            original=value,
            category=None,
            code=None,
            is_cached=False,
            is_new_category=True,
            is_new_code=False,
            confidence=0.0,
            reason="无法归一化，需要LLM或人工处理"
        )
    
    async def _normalize_with_llm(
        self, 
        entity_type: str, 
        value: str,
        original_text: str = ""
    ) -> NormalizationResult:
        """
        使用LLM进行归一化
        
        Args:
            entity_type: 实体类型
            value: 实体值
            original_text: 完整的原始材料描述
            
        Returns:
            归一化结果
        """
        try:
            # 获取所有标准大类
            categories = self.mapping.get_all_categories(entity_type)
            
            # 构建提示词（包含完整原始描述）
            messages = build_classification_prompt(
                entity_type=entity_type,
                value=value,
                categories=categories,
                original_text=original_text
            )
            
            # 记录完整的prompt
            logger.info(f"[LLM归一化] 完整Prompt:\n{'-'*50}")
            for msg in messages:
                logger.info(f"[{msg['role']}]: {msg['content']}")
            logger.info(f"{'-'*50}")
            
            # 调用LLM
            response = await self.llm.chat(messages, format="json")
            result = json.loads(response)
            
            # 记录完整响应日志
            logger.info(f"[LLM归一化] 响应: {result}")
            
            category = result.get('category')
            is_new = result.get('is_new', False)
            
            # 核心逻辑修改：如果返回的分类不在预设列表中，强制视为新类别
            if category and category not in categories:
                logger.info(f"[LLM归一化] 检测到自创分类: {category}，转为新类别处理")
                is_new = True
            
            # 直接使用 LLM 返回的置信度
            raw_conf = result.get('confidence')
            if isinstance(raw_conf, (int, float)):
                confidence = float(raw_conf)
            else:
                confidence = 0.7  # 仅当 LLM 没有返回 confidence 时使用默认值
            reason = result.get('reason', '')
            
            logger.info(f"[LLM归一化] {entity_type}.{value} -> {category}, LLM返回置信度: {raw_conf}, 使用置信度: {confidence}")
            
            if is_new or not category:
                # 新类别
                suggested_name = result.get('suggested_name', value)
                # 防止使用模板文本作为类别名
                invalid_names = ['建议的新大类名称', '新大类名称', '标准大类名称', '']
                if not suggested_name or suggested_name in invalid_names:
                    suggested_name = value
                return await self._handle_new_category(
                    entity_type, value, suggested_name, confidence, reason
                )
            
            # 归入已有类别
            code = self.mapping.get_code(entity_type, category)
            
            if not code:
                # 类别存在但没有编码（异常情况）
                logger.warning(f"类别 {category} 没有对应编码")
                return NormalizationResult(
                    original=value,
                    category=category,
                    code=None,
                    is_cached=False,
                    is_new_category=False,
                    is_new_code=True,
                    confidence=confidence,
                    reason=reason
                )
            
            # 添加到缓存
            if value != category:
                self.mapping.add_to_cache(entity_type, value, category)
            
            return NormalizationResult(
                original=value,
                category=category,
                code=code,
                is_cached=False,
                is_new_category=False,
                is_new_code=False,
                confidence=confidence,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"LLM归一化失败: {e}")
            return NormalizationResult(
                original=value,
                category=None,
                code=None,
                is_cached=False,
                is_new_category=False,
                is_new_code=False,
                confidence=0.0,
                reason=f"LLM调用失败: {str(e)}"
            )
    
    async def _handle_new_category(
        self,
        entity_type: str,
        original_value: str,
        category_name: str,
        confidence: float,
        reason: str
    ) -> NormalizationResult:
        """
        处理新类别
        
        Args:
            entity_type: 实体类型
            original_value: 原始值
            category_name: 新类别名称
            confidence: 置信度
            reason: 原因
            
        Returns:
            归一化结果
        """
        if not self.llm:
            return NormalizationResult(
                original=original_value,
                category=category_name,
                code=None,
                is_cached=False,
                is_new_category=True,
                is_new_code=True,
                confidence=confidence,
                reason=f"新类别，需要生成编码: {reason}"
            )
        
        try:
            # 获取已有编码用于参考
            existing_codes = self.mapping.get_all_codes(entity_type)
            used_codes = list(existing_codes.values())
            
            # 构建提示词
            messages = build_code_generation_prompt(
                entity_type=entity_type,
                category=category_name,
                existing_codes=existing_codes,
                used_codes=used_codes
            )
            
            # 调用LLM生成编码
            response = await self.llm.chat(messages, format="json")
            if not response or not response.strip():
                logger.error(f"[生成编码] LLM返回内容为空")
                return NormalizationResult(
                    original=original_value,
                    category=category_name,
                    code=None,
                    is_cached=False,
                    is_new_category=True,
                    is_new_code=True,
                    confidence=confidence,
                    reason=f"{reason} (生成编码失败: LLM无返回)"
                )
                
            try:
                result = json.loads(response)
            except Exception as e:
                logger.error(f"[生成编码] JSON解析失败: {e}, 原始响应: {response}")
                return NormalizationResult(
                    original=original_value,
                    category=category_name,
                    code=None,
                    is_cached=False,
                    is_new_category=True,
                    is_new_code=True,
                    confidence=confidence,
                    reason=f"{reason} (生成编码失败: JSON格式错误)"
                )
            
            code = result.get('code')
            derivation = result.get('derivation', '')
            has_conflict = result.get('has_conflict', False)
            
            if has_conflict:
                # 使用备选编码
                alternatives = result.get('alternatives', [])
                if alternatives:
                    code = alternatives[0]
            
            if code:
                # 保存新编码
                self.mapping.add_code(entity_type, category_name, code)
                
                # 如果原始值不同于类别名，添加缓存
                if original_value != category_name:
                    self.mapping.add_to_cache(entity_type, original_value, category_name)
                
                return NormalizationResult(
                    original=original_value,
                    category=category_name,
                    code=code,
                    is_cached=False,
                    is_new_category=True,
                    is_new_code=True,
                    confidence=confidence,
                    reason=f"新类别，生成编码: {derivation}"
                )
            
        except Exception as e:
            logger.error(f"生成编码失败: {e}")
        
        return NormalizationResult(
            original=original_value,
            category=category_name,
            code=None,
            is_cached=False,
            is_new_category=True,
            is_new_code=False,
            confidence=0.0,
            reason=f"新类别，编码生成失败: {reason}"
        )
    
    async def batch_normalize(
        self,
        entity_type: str,
        values: list,
        use_llm: bool = True
    ) -> list:
        """
        批量归一化
        
        Args:
            entity_type: 实体类型
            values: 值列表
            use_llm: 是否使用LLM
            
        Returns:
            归一化结果列表
        """
        results = []
        for v in values:
            results.append(await self.normalize(entity_type, v, use_llm))
        return results

