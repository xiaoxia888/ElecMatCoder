"""
LLM分词器

使用大语言模型进行电力材料描述的分词
支持多平台：电缆/桥架(cable)、管道(pipe)
"""

import logging
import asyncio
import json
import re
from typing import List, Optional, Dict

from .llm.config import settings
from .llm.services.ollama_service import OllamaService
from .llm.services.deepseek_service import DeepSeekService
from src.prompts.cable_tokenize_prompt import TokenizePromptTemplate
from src.prompts.pipe_tokenize_prompt import PipeTokenizePromptTemplate

logger = logging.getLogger(__name__)


class LLMTokenizer:
    """基于LLM的分词器"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        platform: str = "cable",
    ):
        """
        初始化LLM分词器
        
        Args:
            model: 模型名称 (如 qwen3:8b, deepseek)
            base_url: API地址
            platform: 平台类型 ('cable' 或 'pipe')
        """
        self.model_name = model
        self.platform = platform
        
        # 判断是 DeepSeek 还是 Ollama
        is_deepseek = model and "deepseek" in model.lower()
        logger.info(f"初始化LLM分词器: model={model}, is_deepseek={is_deepseek}")
        
        if is_deepseek:
            self.service = DeepSeekService(
                model=model,
                base_url=base_url,
            )
            logger.info(f"使用 DeepSeek 服务: {self.service.model}")
        else:
            self.service = OllamaService(
                model=model,
                base_url=base_url,
            )
            logger.info(f"使用 Ollama 服务: {self.service.model}")
        
        # 根据平台选择不同的提示词模板
        if platform == "pipe":
            self.prompt_template = PipeTokenizePromptTemplate()
        else:
            self.prompt_template = TokenizePromptTemplate()
        
        logger.info(f"LLM分词器初始化完成，平台: {platform}，模型: {self.service.model}")
    
    async def tokenize(self, text: str) -> Dict:
        """
        对文本进行分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            结果字典 {"tokens": [...], "type_class": "..."}
        """
        if not text or not text.strip():
            return {"tokens": [], "type_class": None}
        
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
            
            # 调用服务
            response = await self.service.chat(
                messages=messages,
                format="json",
            )
            
            # --- 打印原始响应内容进行调试 ---
            print(f"\n" + "="*50)
            print(f"DEBUG: 大模型原始响应 (原文: {text})")
            print("-" * 50)
            print(response)
            print("="*50 + "\n")
            
            logger.info(f"LLM分词响应: {response[:200] if response else 'empty'}...")
            
            if not response:
                logger.error("LLM返回空响应")
                return {"tokens": self._fallback_tokenize(text), "type_class": None}
            
            # 解析响应
            json_data = self._extract_json(response)
            if not json_data:
                logger.warning("解析分词结果失败，使用回退方案")
                return {"tokens": self._fallback_tokenize(text), "type_class": None}
                
            tokens = self._parse_tokens(text, json_data)
            type_class = json_data.get("type_class")
            
            return {"tokens": tokens, "type_class": type_class}
            
        except Exception as e:
            logger.error(f"LLM分词失败: {e}")
            import traceback
            traceback.print_exc()
            return {"tokens": self._fallback_tokenize(text), "type_class": None}
    
    def tokenize_sync(self, text: str) -> Dict:
        """同步版本的分词方法"""
        return asyncio.run(self.tokenize(text))
    
    def _parse_tokens(self, original_text: str, json_data: Dict) -> List[Dict]:
        """
        从JSON数据中解析tokens，并与原文进行对齐。
        针对LLM可能私自修改空格或标点的情况进行鲁棒性处理。
        """
        tokens_list = json_data.get("tokens", [])
        if not isinstance(tokens_list, list):
            return []
        
        result = []
        current_pos = 0
        
        for token_item in tokens_list:
            if isinstance(token_item, str):
                token_text = token_item
                token_tag = "O"
            elif isinstance(token_item, dict):
                token_text = token_item.get("word", "")
                token_tag = token_item.get("tag", "O")
            else:
                continue
            
            if not token_text:
                continue
            
            # --- 对齐逻辑：寻找 token_text 在 original_text 中的位置 ---
            found_pos = -1
            matched_text = ""
            
            # 1. 尝试精确匹配
            found_pos = original_text.find(token_text, current_pos)
            if found_pos != -1:
                matched_text = token_text
            
            # 2. 如果精确匹配失败，尝试忽略空格进行模糊匹配（解决LLM删减空格的问题）
            if found_pos == -1:
                token_text_clean = re.sub(r'\s+', '', token_text)
                if token_text_clean:
                    # 在原文 current_pos 之后寻找能拼出 token_text_clean 的最短片段
                    search_range = original_text[current_pos:]
                    # 构建正则：字符之间允许有任意空格
                    pattern = r'\s*'.join([re.escape(c) for c in token_text_clean])
                    match = re.search(pattern, search_range)
                    if match:
                        found_pos = current_pos + match.start()
                        matched_text = match.group() # 使用原文中的实际文本（包含原始空格）
            
            # 3. 如果还是找不到，跳过这个token（避免弄乱后续位置）
            if found_pos == -1:
                logger.warning(f"在原文中无法对齐token: '{token_text}'")
                continue
            
            # --- 填充间隙 (Gap Filling) ---
            if found_pos > current_pos:
                gap_text = original_text[current_pos:found_pos]
                if gap_text:
                    result.append({
                        "word": gap_text,
                        "start": current_pos,
                        "end": found_pos,
                        "tag": "O"
                    })
            
            # --- 添加当前 token ---
            result.append({
                "word": matched_text,
                "start": found_pos,
                "end": found_pos + len(matched_text),
                "tag": token_tag
            })
            
            current_pos = found_pos + len(matched_text)
            
        # --- 处理尾部残留 ---
        if current_pos < len(original_text):
            tail = original_text[current_pos:]
            if tail:
                result.append({
                    "word": tail,
                    "start": current_pos,
                    "end": len(original_text),
                    "tag": "O"
                })
        
        return result
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON"""
        if not text:
            return None
        
        # 移除可能的think标签
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取{...}
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _fallback_tokenize(self, text: str) -> List[Dict]:
        """
        回退分词方案：按字符分词
        """
        logger.info("使用回退方案：按字符分词")
        result = []
        for i, char in enumerate(text):
            result.append({
                "word": char,
                "start": i,
                "end": i + 1,
                "tag": "O"
            })
        return result
