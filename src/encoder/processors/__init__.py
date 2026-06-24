"""
编码处理器模块

包含各种字段的处理器：
- StandardProcessor: 规范/标准处理器
- ThicknessProcessor: 壁厚处理器
- PressureProcessor: 磅级处理器
- SizeProcessor: 尺寸处理器
- RegexExtractor: 规则提取器（ENDS, SEAL 等）
"""

from .standard_processor import StandardProcessor, get_standard_processor
from .thickness_processor import ThicknessProcessor, get_thickness_processor
from .pressure_processor import PressureProcessor, get_pressure_processor
from .size_processor import SizeProcessor, get_size_processor
from .weak_fallback_processor import WeakFallbackProcessor, get_weak_fallback_processor
from .thickness_table_processor import ThicknessTableProcessor, get_thickness_table_processor, convert_thickness_to_mm
from .regex_extractor import RegexExtractor, get_regex_extractor
from .standard_target_mapper import StandardTargetMapper, get_standard_target_mapper, map_standard_to_target
from .type_encoder import TypeEncoder, TypeEncodingResult, get_type_encoder
from .material_encoder import MaterialEncoder, MaterialEncodingResult, get_material_encoder
from .rule_audit import build_rule_audit_excel
from .rule_extraction import RuleExtractionResult, extract_size_and_thickness_by_rules

__all__ = [
    'StandardProcessor',
    'get_standard_processor',
    'ThicknessProcessor', 
    'get_thickness_processor',
    'PressureProcessor',
    'get_pressure_processor',
    'SizeProcessor',
    'get_size_processor',
    'WeakFallbackProcessor',
    'get_weak_fallback_processor',
    'ThicknessTableProcessor',
    'get_thickness_table_processor',
    'convert_thickness_to_mm',
    'RegexExtractor',
    'get_regex_extractor',
    'StandardTargetMapper',
    'get_standard_target_mapper',
    'map_standard_to_target',
    'TypeEncoder',
    'TypeEncodingResult',
    'get_type_encoder',
    'MaterialEncoder',
    'MaterialEncodingResult',
    'get_material_encoder',
    'build_rule_audit_excel',
    'RuleExtractionResult',
    'extract_size_and_thickness_by_rules',
]
