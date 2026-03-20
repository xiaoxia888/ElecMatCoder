"""
Ollama服务调用模块
"""

import httpx
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio

from ..config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama API调用服务"""
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: int = None,
    ):
        """
        初始化Ollama服务
        
        Args:
            base_url: Ollama API地址
            model: 模型名称
            timeout: 请求超时时间
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.request_timeout
        
        # 确保base_url没有尾部斜杠
        self.base_url = self.base_url.rstrip("/")
    
    async def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False,
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 用户输入
            system: 系统提示词
            temperature: 生成温度
            max_tokens: 最大token数
            stream: 是否流式输出
            
        Returns:
            生成的文本
        """
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens or settings.max_tokens
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                
                if stream:
                    # 流式响应需要特殊处理
                    return await self._handle_stream_response(response)
                else:
                    result = response.json()
                    return result.get("response", "")
                    
            except httpx.TimeoutException as e:
                logger.error(f"Ollama请求超时: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP错误: {e}")
                raise
            except Exception as e:
                logger.error(f"Ollama请求失败: {e}")
                raise
    
    async def chat(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None,
        format: str = "json",
    ) -> str:
        """
        聊天接口（推荐使用，更好的对话管理）
        
        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大token数
            format: 输出格式，"json" 强制JSON输出
            
        Returns:
            生成的文本
        """
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens or settings.max_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        # 强制JSON输出
        if format == "json":
            payload["format"] = "json"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "")
                    
            except httpx.TimeoutException as e:
                logger.error(f"Ollama请求超时: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP错误: {e}")
                raise
            except Exception as e:
                logger.error(f"Ollama请求失败: {e}")
                raise
    
    async def _handle_stream_response(self, response: httpx.Response) -> str:
        """处理流式响应"""
        full_response = ""
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    full_response += data.get("response", "")
                except json.JSONDecodeError:
                    continue
        return full_response
    
    async def check_health(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            服务是否可用
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama健康检查失败: {e}")
            return False
    
    async def list_models(self) -> list:
        """
        获取可用模型列表
        
        Returns:
            模型列表
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def generate_sync(
        self,
        prompt: str,
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        同步版本的generate方法
        
        Args:
            prompt: 用户输入
            system: 系统提示词
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            生成的文本
        """
        return asyncio.run(
            self.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
    
    def chat_sync(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None,
        format: str = "json",
    ) -> str:
        """
        同步版本的chat方法
        
        Args:
            messages: 消息列表
            temperature: 生成温度
            max_tokens: 最大token数
            format: 输出格式
            
        Returns:
            生成的文本
        """
        return asyncio.run(
            self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                format=format,
            )
        )

