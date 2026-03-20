"""
Prompt 模板模块
"""

from .cable_tokenize_prompt import TokenizePromptTemplate
from .pipe_tokenize_prompt import PipeTokenizePromptTemplate

__all__ = [
    'TokenizePromptTemplate',
    'PipeTokenizePromptTemplate',
]
