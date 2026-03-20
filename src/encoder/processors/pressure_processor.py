"""
磅级处理器
负责压力等级的格式化和编码转换

处理规则：
1. Class 体系：CL150, 150LB, 150#, Class150 → C150
2. PN 体系：PN16, 16RF → PN16
3. 拼写修正：CL15O → CL150, CI3000 → CL3000
"""

import re
from typing import Optional, Tuple
import yaml
from pathlib import Path


class PressureProcessor:
    """磅级处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径，默认使用 encoder_config.yaml
        """
        self.config = self._load_config(config_path)
        
        # 拼写修正映射
        self.typo_fix = self.config.get('typo_fix', {
            'O': '0',
            'CI': 'CL'
        })
        
        # Class 体系识别特征
        class_indicators = self.config.get('class_indicators', {})
        self.class_prefixes = class_indicators.get('prefixes', ['CL', 'CLASS', 'C'])
        self.class_suffixes = class_indicators.get('suffixes', ['LB', 'LBS', '#'])
        
        # PN 体系识别特征
        pn_indicators = self.config.get('pn_indicators', {})
        self.pn_prefixes = pn_indicators.get('prefixes', ['PN'])
        self.pn_suffixes = pn_indicators.get('suffixes', ['RF'])
        
        # 构建正则表达式
        self._build_patterns()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'encoder_config.yaml'
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('pressure_processing', {})
        return {}
    
    def _build_patterns(self):
        """构建正则表达式"""
        # Class 前缀模式（按长度降序，先匹配 CLASS 再匹配 CL/C）
        class_prefixes_sorted = sorted(self.class_prefixes, key=len, reverse=True)
        class_prefix_pattern = '|'.join(re.escape(p) for p in class_prefixes_sorted)
        
        # Class 后缀模式
        class_suffix_pattern = '|'.join(re.escape(s) for s in self.class_suffixes)
        
        # PN 前缀模式
        pn_prefix_pattern = '|'.join(re.escape(p) for p in self.pn_prefixes)
        
        # PN 后缀模式
        pn_suffix_pattern = '|'.join(re.escape(s) for s in self.pn_suffixes)
        
        # Class 体系正则：前缀+数字 或 数字+后缀
        # 匹配：CL150, CL.150, CLASS150, C150, CL2.5, 150LB, 150#
        self.class_prefix_re = re.compile(
            rf'(?:{class_prefix_pattern})\.?\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )
        # 注意：# 是非单词字符，后面不能用 \b，改用 (?:\b|$)
        self.class_suffix_re = re.compile(
            rf'(\d+(?:\.\d+)?)\s*(?:{class_suffix_pattern})(?:\b|$)',
            re.IGNORECASE
        )
        
        # PN 体系正则：前缀+数字 或 数字+后缀
        # 匹配：PN16, PN2.5, 16RF
        self.pn_prefix_re = re.compile(
            rf'(?:{pn_prefix_pattern})\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )
        self.pn_suffix_re = re.compile(
            rf'(\d+(?:\.\d+)?)\s*(?:{pn_suffix_pattern})(?:\b|$)',
            re.IGNORECASE
        )
    
    def process(self, value: str) -> str:
        """
        处理磅级值，返回标准编码
        
        Args:
            value: 原始磅级描述，如 "CL 150", "150LB", "PN16", "16 RF"
            
        Returns:
            标准编码，如 "C150", "PN16"
        """
        if not value:
            return ""
        
        # 1. 预处理
        processed = self._preprocess(value)
        
        # 2. 识别体系并提取数字
        system, number = self._identify_and_extract(processed)
        
        if system is None or number is None:
            # 无法识别，返回原值（大写）
            return value.strip().upper()
        
        # 3. 生成编码
        if system == 'CLASS':
            return f"C{number}"
        elif system == 'PN':
            return f"PN{number}"
        else:
            return value.strip().upper()
    
    def _preprocess(self, value: str) -> str:
        """
        预处理：去除空格、修正拼写
        """
        result = value.strip().upper()
        
        # 去除空格
        result = result.replace(' ', '')
        
        # 修正拼写
        for wrong, correct in self.typo_fix.items():
            result = result.replace(wrong.upper(), correct.upper())
        
        return result
    
    def _identify_and_extract(self, value: str) -> Tuple[Optional[str], Optional[str]]:
        """
        识别体系并提取数字
        
        Returns:
            (体系, 数字) 或 (None, None)
        """
        # 优先检查 PN 体系（因为 PN 前缀更明确）
        # PN 前缀匹配：PN16
        match = self.pn_prefix_re.search(value)
        if match:
            return ('PN', match.group(1))
        
        # PN 后缀匹配：16RF
        match = self.pn_suffix_re.search(value)
        if match:
            return ('PN', match.group(1))
        
        # Class 前缀匹配：CL150, CLASS150, C150
        match = self.class_prefix_re.search(value)
        if match:
            return ('CLASS', match.group(1))
        
        # Class 后缀匹配：150LB, 150#
        match = self.class_suffix_re.search(value)
        if match:
            return ('CLASS', match.group(1))
        
        return (None, None)


# 单例模式
_processor_instance: Optional[PressureProcessor] = None


def get_pressure_processor() -> PressureProcessor:
    """获取处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = PressureProcessor()
    return _processor_instance
