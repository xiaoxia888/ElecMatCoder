"""
配置管理模块
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""

    # Ollama 配置
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API 地址"
    )
    ollama_model: str = Field(
        default="qwen3:8b",
        description="使用的模型名称"
    )
    
    # DeepSeek 配置
    deepseek_api_key: str = Field(
        default="sk-134b39a758a147ac87b9da7af886a848",
        description="DeepSeek API Key"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 地址"
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek 模型名称"
    )
    
    # 请求配置
    request_timeout: int = Field(
        default=120,
        description="请求超时时间（秒）"
    )
    max_retries: int = Field(
        default=3,
        description="最大重试次数"
    )
    
    # 生成配置
    temperature: float = Field(
        default=0.1,
        description="生成温度，越低越确定性"
    )
    max_tokens: int = Field(
        default=2048,
        description="最大生成token数"
    )
    
    # 批处理配置
    batch_size: int = Field(
        default=10,
        description="批处理大小"
    )
    
    class Config:
        env_prefix = "LLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()

