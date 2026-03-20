"""
电力材料NER标注器主类
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .config import settings
from .models import Entity, AnnotationResult, BatchAnnotationResult, EntityType
from .services.ollama_service import OllamaService
from src.prompts.cable_ner_prompt import NERPromptTemplate
from .utils import (
    extract_json_from_text,
    validate_entity,
    merge_overlapping_entities,
    format_annotation_result,
)

logger = logging.getLogger(__name__)


class CableNERAnnotator:
    """电力材料NER标注器"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        custom_examples: List[Dict] = None,
    ):
        """
        初始化标注器
        
        Args:
            model: Ollama模型名称
            base_url: Ollama API地址
            custom_examples: 自定义few-shot示例
        """
        self.ollama = OllamaService(
            model=model,
            base_url=base_url,
        )
        self.prompt_template = NERPromptTemplate(custom_examples=custom_examples)
        
        logger.info(f"标注器初始化完成，使用模型: {self.ollama.model}")
    
    async def annotate(self, text: str) -> AnnotationResult:
        """
        对单条文本进行标注
        
        Args:
            text: 待标注的电力材料描述文本
            
        Returns:
            标注结果
        """
        if not text or not text.strip():
            return AnnotationResult(
                text=text,
                entities=[],
                success=True,
            )
        
        try:
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_template.get_full_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self.prompt_template.get_user_prompt(text),
                },
            ]
            
            # 调用LLM
            response = await self.ollama.chat(
                messages=messages,
                format="json",
            )
            
            logger.info(f"LLM响应长度: {len(response) if response else 0}")
            
            if not response or not response.strip():
                logger.error("LLM返回空响应")
                return AnnotationResult(
                    text=text,
                    entities=[],
                    success=False,
                    error_message="LLM返回空响应",
                )
            
            # 解析响应
            result = self._parse_response(text, response)
            result.raw_response = response
            
            return result
            
        except Exception as e:
            logger.error(f"标注失败: {e}, text={text[:50]}...")
            return AnnotationResult(
                text=text,
                entities=[],
                success=False,
                error_message=str(e),
            )
    
    def annotate_sync(self, text: str) -> AnnotationResult:
        """
        同步版本的标注方法
        
        Args:
            text: 待标注文本
            
        Returns:
            标注结果
        """
        return asyncio.run(self.annotate(text))
    
    async def annotate_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
    ) -> BatchAnnotationResult:
        """
        批量标注
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            批量标注结果
        """
        batch_size = batch_size or settings.batch_size
        batch_result = BatchAnnotationResult()
        
        # 使用进度条
        iterator = tqdm(texts, desc="标注进度") if show_progress else texts
        
        for text in iterator:
            result = await self.annotate(text)
            batch_result.add_result(result)
        
        logger.info(
            f"批量标注完成: 总数={batch_result.total}, "
            f"成功={batch_result.success_count}, "
            f"失败={batch_result.failed_count}"
        )
        
        return batch_result
    
    def annotate_batch_sync(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True,
    ) -> BatchAnnotationResult:
        """
        同步版本的批量标注方法
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            批量标注结果
        """
        return asyncio.run(
            self.annotate_batch(
                texts=texts,
                batch_size=batch_size,
                show_progress=show_progress,
            )
        )
    
    def _parse_response(self, original_text: str, response: str) -> AnnotationResult:
        """
        解析LLM响应
        
        Args:
            original_text: 原始文本
            response: LLM响应
            
        Returns:
            标注结果
        """
        # 提取JSON
        json_data = extract_json_from_text(response)
        
        if not json_data:
            logger.warning(f"无法解析LLM响应: {response[:200]}...")
            return AnnotationResult(
                text=original_text,
                entities=[],
                success=False,
                error_message="无法解析LLM响应为JSON",
            )
        
        # 提取实体列表
        entities_data = json_data.get("entities", [])
        
        if not isinstance(entities_data, list):
            logger.warning(f"entities不是列表: {type(entities_data)}")
            return AnnotationResult(
                text=original_text,
                entities=[],
                success=False,
                error_message="entities字段格式错误",
            )
        
        # 验证并创建实体对象
        entities = []
        for entity_dict in entities_data:
            entity = validate_entity(entity_dict, original_text)
            if entity:
                entities.append(entity)
        
        # 合并重叠实体
        entities = merge_overlapping_entities(entities)
        
        return AnnotationResult(
            text=original_text,
            entities=entities,
            success=True,
        )
    
    async def check_service(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            服务是否可用
        """
        return await self.ollama.check_health()
    
    async def list_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            模型名称列表
        """
        return await self.ollama.list_models()
    
    def add_custom_example(self, input_text: str, entities: List[Dict]):
        """
        添加自定义标注示例
        
        Args:
            input_text: 输入文本
            entities: 实体列表，格式: [{"text": "...", "label": "...", "start": 0, "end": 4}]
        """
        self.prompt_template.add_example(
            input_text=input_text,
            output={"entities": entities},
        )
        logger.info(f"添加自定义示例: {input_text[:30]}...")
    
    def format_result(self, result: AnnotationResult) -> str:
        """
        格式化标注结果为可读字符串
        
        Args:
            result: 标注结果
            
        Returns:
            格式化的字符串
        """
        return format_annotation_result(result.text, result.entities)

