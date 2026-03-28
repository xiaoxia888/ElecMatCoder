# -*- coding: utf-8 -*-
"""
材料编码模块
将NER识别结果转换为标准材料编码

支持两种编码模式：
1. 桥架材料编码 - MaterialEncoder
2. 管道材料编码 - PipeEncoder
"""

from .material_encoder import MaterialEncoder, EncodingResult, EntityDetail
from .entity_normalizer import EntityNormalizer, NormalizationResult
from .spec_parser import SpecParser, SpecParseResult
from .info_completer import InfoCompleter, CompletionResult, FieldInfo
from .code_mapping import CodeMappingManager, get_mapping_manager

# 管道材料编码模块
from .pipe_encoder import (
    PipeEncoder,
    PipeEncodingResult,
    FieldEncoding,
    get_pipe_encoder
)
from .processors import (
    SizeProcessor,
    StandardProcessor,
    StandardTargetMapper,
    ThicknessTableProcessor,
    get_size_processor,
    get_standard_processor,
    get_standard_target_mapper,
    get_thickness_table_processor,
    map_standard_to_target,
    convert_thickness_to_mm,
)
from .semantic_matcher import (
    SemanticMatcher,
    MatchResult,
    get_semantic_matcher
)

__all__ = [
    # 桥架材料编码器
    'MaterialEncoder',
    'EncodingResult',
    'EntityDetail',
    
    # 归一化器
    'EntityNormalizer',
    'NormalizationResult',
    
    # 规格解析器
    'SpecParser',
    'SpecParseResult',
    
    # 信息补全器
    'InfoCompleter',
    'CompletionResult',
    'FieldInfo',
    
    # 映射管理
    'CodeMappingManager',
    'get_mapping_manager',
    
    # 管道材料编码器
    'PipeEncoder',
    'PipeEncodingResult',
    'FieldEncoding',
    'get_pipe_encoder',
    
    # 处理器
    'SizeProcessor',
    'StandardProcessor',
    'StandardTargetMapper',
    'ThicknessTableProcessor',
    'get_size_processor',
    'get_standard_processor',
    'get_standard_target_mapper',
    'get_thickness_table_processor',
    'map_standard_to_target',
    'convert_thickness_to_mm',
    
    # 语义匹配器
    'SemanticMatcher',
    'MatchResult',
    'get_semantic_matcher',
]
