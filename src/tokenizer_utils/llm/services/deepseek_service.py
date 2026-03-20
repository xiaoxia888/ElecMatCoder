"""
DeepSeek API服务调用模块
"""

import httpx
import json
import logging
import re
from typing import Optional, Dict, Any, List
import asyncio

from ..config import settings

logger = logging.getLogger(__name__)


class DeepSeekService:
    """DeepSeek API调用服务"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        timeout: int = None,
    ):
        """
        初始化DeepSeek服务
        
        Args:
            api_key: DeepSeek API Key
            base_url: DeepSeek API 地址
            model: 模型名称
            timeout: 请求超时时间
        """
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.timeout = timeout or settings.request_timeout
        
        # 确保base_url没有尾部斜杠
        self.base_url = self.base_url.rstrip("/")
    
    async def chat(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None,
        format: str = "json",
    ) -> str:
        """
        聊天接口
        
        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大token数
            format: 输出格式
            
        Returns:
            生成的文本内容
        """
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens or settings.max_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # DeepSeek API 不一定支持 format="json" 参数，我们通常在 prompt 中要求返回 JSON
        # 但有些 provider 支持。这里为了通用，默认不传 format 参数，除非明确支持。
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 如果要求是JSON，尝试清洗掉 Markdown 标记
                if format == "json":
                    content = self._clean_json_content(content)
                    
                return content
                    
            except httpx.TimeoutException as e:
                logger.error(f"DeepSeek请求超时: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"DeepSeek HTTP错误: {e}, 响应内容: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"DeepSeek请求失败: {e}")
                raise

    def _clean_json_content(self, content: str) -> str:
        """清洗LLM返回的JSON内容，移除 markdown 代码块等"""
        # 移除可能存在的 think 标签
        content = re.sub(r'<think>[\s\S]*?</think>', '', content)
        
        # 提取 ```json ... ``` 中的内容
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            return json_match.group(1).strip()
        
        # 提取 ``` ... ``` 中的内容
        code_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
        if code_match:
            return code_match.group(1).strip()
            
        return content.strip()
