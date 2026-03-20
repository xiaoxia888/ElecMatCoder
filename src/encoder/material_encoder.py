# -*- coding: utf-8 -*-
"""
材料编码器
整合所有模块，实现完整的材料描述到编码的转换流程
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

from .code_mapping import CodeMappingManager, get_mapping_manager
from .entity_normalizer import EntityNormalizer, NormalizationResult
from .spec_parser import SpecParser, SpecParseResult
from .info_completer import InfoCompleter, CompletionResult, FieldInfo

logger = logging.getLogger(__name__)


@dataclass
class EntityDetail:
    """实体编码详情"""
    original: str               # 原始值
    normalized: str             # 归一化后的值（标准大类）
    code: str                   # 编码
    is_inferred: bool = False   # 是否是推断的
    is_new: bool = False        # 是否是新类别/编码
    confidence: float = 1.0     # 置信度 (0-1)
    reason: str = ""            # 处理说明


@dataclass
class EncodingResult:
    """编码结果"""
    # 输入
    original_text: str          # 原始材料描述
    
    # 各字段详情
    name: EntityDetail = None
    material: EntityDetail = None
    type: EntityDetail = None
    spec: EntityDetail = None
    
    # 最终编码
    final_code: str = None
    
    # 处理状态
    success: bool = False
    has_inferred: bool = False  # 是否有推断字段
    has_new: bool = False       # 是否有新类别/编码
    has_missing: bool = False   # 是否有缺失字段
    
    # 缺失和错误信息
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


class MaterialEncoder:
    """
    材料编码器
    
    完整处理流程：
    1. 文本预处理
    2. NER实体识别
    3. 信息补全（推断缺失字段）
    4. 实体归一化 + 编码转换
    5. 规格解析
    6. 组装最终编码
    """
    
    def __init__(
        self,
        ner_predictor = None,
        llm_service = None,
        mapping_manager: CodeMappingManager = None,
        preprocessor = None
    ):
        """
        初始化编码器
        
        Args:
            ner_predictor: NER预测器实例
            llm_service: LLM服务实例
            mapping_manager: 编码映射管理器
            preprocessor: 文本预处理器
        """
        self.ner = ner_predictor
        self.llm = llm_service
        self.mapping = mapping_manager or get_mapping_manager()
        self.preprocessor = preprocessor
        
        # 初始化子模块
        self.normalizer = EntityNormalizer(self.mapping, llm_service)
        self.spec_parser = SpecParser(llm_service)
        self.completer = InfoCompleter(self.mapping, llm_service)
    
    def set_ner_predictor(self, ner_predictor):
        """设置NER预测器"""
        self.ner = ner_predictor
    
    def set_llm_service(self, llm_service):
        """设置LLM服务"""
        self.llm = llm_service
        self.normalizer.set_llm_service(llm_service)
        self.spec_parser.set_llm_service(llm_service)
        self.completer.set_llm_service(llm_service)
    
    def set_preprocessor(self, preprocessor):
        """设置预处理器"""
        self.preprocessor = preprocessor
    
    async def encode(
        self,
        text: str,
        use_llm: bool = True,
        material_type: str = "桥架"
    ) -> EncodingResult:
        """
        编码材料描述
        
        Args:
            text: 材料描述文本
            use_llm: 是否使用LLM
            material_type: 材料类型（用于规格解析）
            
        Returns:
            编码结果
        """
        result = EncodingResult(original_text=text)
        
        if not text or not text.strip():
            result.errors.append("输入文本为空")
            return result
        
        try:
            # Step 1: 预处理
            processed_text = self._preprocess(text)
            
            # Step 2: NER识别
            entities = self._extract_entities(processed_text)
            if not entities:
                result.errors.append("NER识别失败或未识别到实体")
                return result
            
            # Step 3: 信息补全（传入完整原始描述）
            completion = await self.completer.complete(
                entities, 
                use_llm, 
                original_text=text  # 传递原始描述给LLM
            )
            
            # Step 4: 处理各字段（传入完整原始描述）
            await self._process_name(result, completion, use_llm, original_text=text)
            await self._process_material(result, completion, use_llm, original_text=text)
            await self._process_type(result, completion, use_llm, original_text=text)
            await self._process_spec(result, completion, material_type, use_llm)
            
            # Step 5: 组装最终编码
            self._assemble_code(result)
            
            # 设置状态标记
            result.has_inferred = any([
                result.name and result.name.is_inferred,
                result.material and result.material.is_inferred,
                result.type and result.type.is_inferred,
                result.spec and result.spec.is_inferred,
            ])
            
            result.has_new = any([
                result.name and result.name.is_new,
                result.material and result.material.is_new,
                result.type and result.type.is_new,
            ])
            
            # 检查原始值是否缺失（NER未识别到），而不是编码是否缺失
            def is_original_missing(field):
                entity = getattr(result, field)
                if entity is None:
                    return True
                # 原始值为 None 表示是推断的，也算缺失
                return entity.original is None or entity.original == ''
            
            result.missing_fields = [
                f for f in ['name', 'material', 'type', 'spec']
                if is_original_missing(f)
            ]
            result.has_missing = len(result.missing_fields) > 0
            
            result.success = result.final_code is not None
            
        except Exception as e:
            logger.error(f"编码失败: {e}", exc_info=True)
            result.errors.append(f"编码失败: {str(e)}")
        
        return result
    
    def _preprocess(self, text: str) -> str:
        """预处理文本"""
        if self.preprocessor:
            return self.preprocessor.process(text)
        
        # 默认预处理：全角转半角
        result = text
        
        # 全角数字转半角
        for i in range(10):
            result = result.replace(chr(0xFF10 + i), str(i))
        
        # 全角字母转半角
        for i in range(26):
            result = result.replace(chr(0xFF21 + i), chr(0x41 + i))  # 大写
            result = result.replace(chr(0xFF41 + i), chr(0x61 + i))  # 小写
        
        # 统一乘号
        result = result.replace('×', 'X').replace('*', 'X')
        
        # 统一冒号
        result = result.replace('：', ':')
        
        return result.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """提取实体"""
        if not self.ner:
            logger.warning("NER预测器未设置，返回空实体")
            return {}
        
        try:
            # 调用NER预测
            ner_result = self.ner.predict_raw(text)
            entities = ner_result.get('entities', [])
            
            # 转换为字典格式
            entity_dict = {}
            for entity in entities:
                entity_type = entity.get('type', '').upper()
                entity_text = entity.get('text', '')
                
                # 映射到标准字段名
                field_map = {
                    'NAME': 'name',
                    'MATERIAL': 'material',
                    'TYPE': 'type',
                    'SPEC': 'spec',
                    'FEATURE': 'feature',  # 可选字段
                }
                
                field_name = field_map.get(entity_type)
                if field_name and entity_text:
                    # 如果同一字段有多个值，合并
                    if field_name in entity_dict:
                        entity_dict[field_name] += entity_text
                    else:
                        entity_dict[field_name] = entity_text
            
            return entity_dict
            
        except Exception as e:
            logger.error(f"NER提取失败: {e}")
            return {}
    
    def _combine_reasons(self, field_info: FieldInfo, norm_result: NormalizationResult) -> str:
        """结构化合并处理原因"""
        reasons = []
        
        # 1. 处理来源/推断原因
        if field_info.is_inferred:
            if field_info.value is None: # 推断失败
                reasons.append(f"【缺失】{field_info.reason}")
            else:
                reasons.append(f"【推断】{field_info.reason}")
        elif field_info.reason and field_info.reason != "NER识别":
            reasons.append(field_info.reason)
            
        # 2. 处理归一化原因
        if norm_result and norm_result.reason:
            if norm_result.reason == "直接匹配标准大类":
                if not reasons: # 如果前面没东西，才加这个
                    reasons.append("标准匹配")
            else:
                reasons.append(f"【归一化】{norm_result.reason}")
                
        return " | ".join(reasons) if reasons else "正常识别"

    async def _process_name(
        self, 
        result: EncodingResult, 
        completion: CompletionResult,
        use_llm: bool,
        original_text: str = ""
    ):
        """处理名称字段"""
        field_info = completion.name
        if not field_info:
            result.warnings.append("名称字段缺失")
            return
        
        # 归一化（传入完整原始描述）
        norm_result = await self.normalizer.normalize(
            'name', field_info.value, use_llm, original_text
        )
        
        result.name = EntityDetail(
            original=None if field_info.is_inferred else field_info.value,
            normalized=norm_result.category or field_info.value,
            code=norm_result.code,
            is_inferred=field_info.is_inferred,
            is_new=norm_result.is_new_category or norm_result.is_new_code,
            confidence=round((field_info.confidence * norm_result.confidence) ** 0.5, 2),
            reason=self._combine_reasons(field_info, norm_result)
        )
        
        if not norm_result.code:
            result.warnings.append(f"名称编码失败: {field_info.value}")
    
    async def _process_material(
        self, 
        result: EncodingResult, 
        completion: CompletionResult,
        use_llm: bool,
        original_text: str = ""
    ):
        """处理材质字段"""
        field_info = completion.material
        if not field_info:
            result.warnings.append("材质字段缺失")
            return
        
        # 如果推断结果中 value 为空，但有 reason，则记录为带原因的缺失
        if field_info.value is None:
            result.material = EntityDetail(
                original=None,
                normalized="",
                code=None,
                is_inferred=True,
                confidence=field_info.confidence,
                reason=field_info.reason
            )
            result.warnings.append(f"材质字段缺失: {field_info.reason}")
            return
        
        # 归一化（传入完整原始描述）
        norm_result = await self.normalizer.normalize(
            'material', field_info.value, use_llm, original_text
        )
        
        result.material = EntityDetail(
            original=None if field_info.is_inferred else field_info.value,
            normalized=norm_result.category or field_info.value,
            code=norm_result.code,
            is_inferred=field_info.is_inferred,
            is_new=norm_result.is_new_category or norm_result.is_new_code,
            confidence=round((field_info.confidence * norm_result.confidence) ** 0.5, 2),
            reason=self._combine_reasons(field_info, norm_result)
        )
        
        if not norm_result.code:
            result.warnings.append(f"材质编码失败: {field_info.value}")
    
    async def _process_type(
        self, 
        result: EncodingResult, 
        completion: CompletionResult,
        use_llm: bool,
        original_text: str = ""
    ):
        """处理类型字段"""
        field_info = completion.type
        if not field_info:
            result.warnings.append("类型字段缺失")
            return
        
        # 如果推断结果中 value 为空，但有 reason，则记录为带原因的缺失
        if field_info.value is None:
            result.type = EntityDetail(
                original=None,
                normalized="",
                code=None,
                is_inferred=True,
                confidence=field_info.confidence,
                reason=field_info.reason
            )
            result.warnings.append(f"类型字段缺失: {field_info.reason}")
            return
            
        # 归一化（传入完整原始描述）
        norm_result = await self.normalizer.normalize(
            'type', field_info.value, use_llm, original_text
        )
        
        result.type = EntityDetail(
            original=None if field_info.is_inferred else field_info.value,
            normalized=norm_result.category or field_info.value,
            code=norm_result.code,
            is_inferred=field_info.is_inferred,
            is_new=norm_result.is_new_category or norm_result.is_new_code,
            confidence=round((field_info.confidence * norm_result.confidence) ** 0.5, 2),
            reason=self._combine_reasons(field_info, norm_result)
        )
        
        if not norm_result.code:
            result.warnings.append(f"类型编码失败: {field_info.value}")
    
    async def _process_spec(
        self, 
        result: EncodingResult, 
        completion: CompletionResult,
        material_type: str,
        use_llm: bool
    ):
        """处理规格字段"""
        field_info = completion.spec
        if not field_info:
            result.warnings.append("规格字段缺失")
            return
        
        # 解析规格
        parse_result = await self.spec_parser.parse(
            field_info.value, 
            material_type, 
            use_llm
        )
        
        result.spec = EntityDetail(
            original=None if field_info.is_inferred else field_info.value,
            normalized=parse_result.code_format or field_info.value,
            code=parse_result.code_format,
            is_inferred=field_info.is_inferred,
            is_new=False,
            # 置信度 = 推断置信度 * 解析置信度
            confidence=round(field_info.confidence * parse_result.confidence, 2),
            reason=parse_result.parse_method
        )
        
        if not parse_result.success:
            result.warnings.append(f"规格解析失败: {field_info.value}")
    
    def _assemble_code(self, result: EncodingResult):
        """组装最终编码"""
        parts = []
        
        # 名称编码
        if result.name and result.name.code:
            parts.append(result.name.code)
        
        # 材质编码
        if result.material and result.material.code:
            parts.append(result.material.code)
        
        # 类型编码
        if result.type and result.type.code:
            parts.append(result.type.code)
        
        # 规格编码
        if result.spec and result.spec.code:
            parts.append(result.spec.code)
        
        if parts:
            result.final_code = ''.join(parts)
        else:
            result.final_code = None
    
    async def batch_encode(
        self,
        texts: List[str],
        use_llm: bool = True,
        material_type: str = "桥架",
        max_concurrent: int = 10
    ) -> List[EncodingResult]:
        """
        批量编码（并行处理）
        
        Args:
            texts: 材料描述列表
            use_llm: 是否使用LLM
            material_type: 材料类型
            max_concurrent: 最大并发数
            
        Returns:
            编码结果列表
        """
        import asyncio
        
        # 去重处理：相同描述只处理一次
        unique_texts = list(dict.fromkeys(texts))  # 保持顺序去重
        text_to_result: Dict[str, EncodingResult] = {}
        
        # 分批并行处理
        async def process_batch(batch_texts):
            tasks = [
                self.encode(text, use_llm, material_type)
                for text in batch_texts
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # 按批次处理
        for i in range(0, len(unique_texts), max_concurrent):
            batch = unique_texts[i:i + max_concurrent]
            batch_results = await process_batch(batch)
            
            for text, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    # 处理异常
                    text_to_result[text] = EncodingResult(
                        original_text=text,
                        errors=[f"处理失败: {str(result)}"]
                    )
                else:
                    text_to_result[text] = result
        
        # 按原始顺序返回结果（包括重复项）
        return [text_to_result[text] for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.mapping.get_stats()

