"""
LLM预标注模块

用于电力材料描述的命名实体识别预标注
"""

from .annotator import CableNERAnnotator
from .models import Entity, AnnotationResult, EntityType
from .config import settings

__all__ = [
    "CableNERAnnotator",
    "Entity",
    "AnnotationResult",
    "EntityType",
    "settings",
]

__version__ = "0.2.0"
