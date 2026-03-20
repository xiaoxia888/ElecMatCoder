"""
jieba分词器模块 - 针对电力材料描述优化

型号规则从词典中动态加载：
- TYPE 标签的词 → 前缀（如 ZR, NH, ZRA）
- FEATURE 标签且为纯大写字母的词 → 基本型号（如 YJV, KVV）
- ARMOR 标签且为纯数字的词 → 铠装代码（如 22, 32）
"""

import jieba
import os
import re
import logging
from typing import List, Dict, Set

from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class CableTokenizer:
    """电力材料分词器 - 混合正则+jieba方案
    
    型号规则从词典中动态加载，无需硬编码
    """
    
    def __init__(self, user_dict_path: str = None):
        """
        初始化分词器
        
        Args:
            user_dict_path: 自定义词典路径
        """
        self.preprocessor = TextPreprocessor()
        
        # 从词典动态加载的规则
        self.word_to_tag: Dict[str, str] = {}      # 词 -> 标签
        self.model_prefixes: Set[str] = set()      # 阻燃/耐火前缀（如ZR, NH, ZRA）
        self.base_models: Dict[str, str] = {}      # 基本型号（如YJV, KVV）
        self.armor_codes: Set[str] = set()         # 铠装代码（如22, 32）
        
        # 确定词典路径
        if user_dict_path and os.path.exists(user_dict_path):
            dict_path = user_dict_path
        else:
            dict_path = os.path.join(os.path.dirname(__file__), "dict", "cable_dict.txt")
        
        # 加载词典
        if os.path.exists(dict_path):
            self._load_dict(dict_path)
            print(f"[CableTokenizer] 已加载词典: {dict_path}")
            print(f"  - 前缀: {len(self.model_prefixes)} 个")
            print(f"  - 基本型号: {len(self.base_models)} 个")
            print(f"  - 铠装代码: {len(self.armor_codes)} 个")
        else:
            print(f"[CableTokenizer] 词典文件不存在: {dict_path}")
        
        # 编译正则表达式
        self._compile_patterns()
    
    def _load_dict(self, dict_path: str):
        """加载词典并动态构建型号规则"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    word, freq, tag = parts[0], int(parts[1]), parts[2]
                    
                    # 添加到jieba
                    jieba.add_word(word, freq=freq)
                    # 添加到词-标签映射表
                    self.word_to_tag[word] = tag
                    
                    # 动态构建型号规则
                    # 纯大写字母且标签为TYPE -> 前缀
                    if tag == 'TYPE' and word.isalpha() and word.isupper():
                        self.model_prefixes.add(word)
                    
                    # 纯大写字母且标签为FEATURE -> 基本型号
                    if tag == 'FEATURE' and word.isalpha() and word.isupper():
                        self.base_models[word] = 'FEATURE'
                    
                    # 纯数字且标签为ARMOR -> 铠装代码
                    if tag == 'ARMOR' and word.isdigit():
                        self.armor_codes.add(word)
                        
                elif len(parts) == 2:
                    word, freq = parts[0], int(parts[1])
                    jieba.add_word(word, freq=freq)
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        # 电压模式: 0.6/1kV, 8.7/15KV, 450V/750V, 10KV 等
        self.voltage_pattern = re.compile(
            r'\d+\.?\d*/\d+\.?\d*[kK]?[vV]'  # 如 0.6/1kV
            r'|\d+[vV]/\d+[vV]'  # 如 450V/750V
            r'|\d+\.?\d*[kK]?[vV]'  # 如 10KV, 1kV
        )
        
        # 电缆规格模式: 5×16, 3×25+1×16, 16×2×1.5 等
        self.spec_pattern = re.compile(
            r'\d+×\d+\.?\d*(?:×\d+\.?\d*)*(?:\+\d+×\d+\.?\d*(?:×\d+\.?\d*)*)*'
        )
        
        # 桥架尺寸规格模式
        # W600mm, H=150mm, L=6m, W400mm×H150mm, W×H=200×150mm, W×H×L:200×200×6000
        self.bridge_spec_pattern = re.compile(
            r'[WHL]=?\d+\.?\d*(?:mm|m)?'  # W600mm, H=150mm, L=6m, L=6.0m
            r'|[WHL]×[WHL]=?\d+\.?\d*(?:×\d+\.?\d*)*(?:mm|m)?'  # W×H=200×150mm
            r'|[WHL]×[WHL]×[WHL]:?\d+×\d+×\d+'  # W×H×L:200×200×6000
            r'|\d+×\d+(?:mm)?'  # 100×50, 100×50mm (简单尺寸)
        )
        
        # 型号模式: YJV, YJV22, ZR-YJV, NH-KVV 等
        self.model_pattern = re.compile(
            r'[A-Z]{2,}[0-9]{0,2}'  # 基本型号如 YJV, YJV22
        )
        
        # 完整电缆型号模式（用于识别并拆分）
        # 如: NH-YJV22, ZR-DJYPVP, WDZN-YJV32
        self.full_model_pattern = re.compile(
            r'([A-Z]{2,})-([A-Z]{2,})(\d{2})?'  # 前缀-基本型号+可选铠装
        )
    
    def tokenize_with_position(self, text: str, preprocess: bool = True) -> List[dict]:
        """
        分词并返回位置信息和标签
        
        采用混合策略：
        1. 先用正则表达式识别电压、规格等特殊模式
        2. 对其余部分使用jieba分词
        3. 根据词典映射标签
        
        Args:
            text: 输入文本
            preprocess: 是否进行预处理
            
        Returns:
            分词结果列表，每个元素包含 word, start, end, tag
        """
        if not text:
            return []
        
        # 预处理
        if preprocess:
            text = self.preprocessor.process(text)
        
        # 第一步：用正则识别特殊模式，记录位置
        special_tokens = []  # [(start, end, word, tag), ...]
        
        # 识别电压
        for m in self.voltage_pattern.finditer(text):
            special_tokens.append((m.start(), m.end(), m.group(), 'VOLTAGE'))
        
        # 识别电缆规格
        for m in self.spec_pattern.finditer(text):
            # 检查是否与已有token重叠
            overlap = False
            for st in special_tokens:
                if not (m.end() <= st[0] or m.start() >= st[1]):
                    overlap = True
                    break
            if not overlap:
                special_tokens.append((m.start(), m.end(), m.group(), 'SPEC'))
        
        # 识别桥架规格 (W600mm, H=150mm, L=6m 等)
        for m in self.bridge_spec_pattern.finditer(text):
            # 检查是否与已有token重叠
            overlap = False
            for st in special_tokens:
                if not (m.end() <= st[0] or m.start() >= st[1]):
                    overlap = True
                    break
            if not overlap:
                special_tokens.append((m.start(), m.end(), m.group(), 'SPEC'))
        
        # 按位置排序
        special_tokens.sort(key=lambda x: x[0])
        
        # 第二步：对非特殊部分使用jieba分词
        result = []
        last_end = 0
        
        for start, end, word, tag in special_tokens:
            # 处理特殊token之前的文本
            if start > last_end:
                segment = text[last_end:start]
                segment_tokens = self._jieba_tokenize(segment, last_end)
                result.extend(segment_tokens)
            
            # 添加特殊token
            result.append({
                "word": word,
                "start": start,
                "end": end,
                "tag": tag
            })
            last_end = end
        
        # 处理最后一段文本
        if last_end < len(text):
            segment = text[last_end:]
            segment_tokens = self._jieba_tokenize(segment, last_end)
            result.extend(segment_tokens)
        
        return result
    
    def _jieba_tokenize(self, text: str, offset: int) -> List[dict]:
        """
        使用jieba分词并查表获取标签
        
        Args:
            text: 待分词文本
            offset: 在原文中的偏移量
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        words = list(jieba.cut(text))
        result = []
        pos = 0
        
        for word in words:
            # 计算在segment中的位置
            start_in_segment = text.find(word, pos)
            if start_in_segment == -1:
                start_in_segment = pos
            
            # 转换为原文位置
            start = offset + start_in_segment
            
            # 尝试解析电缆型号（如 NH-YJV22）
            parsed = self._parse_cable_model(word, start)
            if parsed:
                result.extend(parsed)
            else:
                # 普通词
                end = start + len(word)
                tag = self._get_tag(word)
                result.append({
                    "word": word,
                    "start": start,
                    "end": end,
                    "tag": tag
                })
            
            pos = start_in_segment + len(word)
        
        return result
    
    def _parse_cable_model(self, word: str, start_offset: int) -> List[dict]:
        """
        解析电缆型号，拆分成细粒度部分
        
        如: NH-YJV22 -> [NH(TYPE), -(O), YJV(FEATURE), 22(ARMOR)]
        
        Args:
            word: 型号字符串
            start_offset: 在原文中的起始位置
            
        Returns:
            拆分后的token列表，如果不是型号则返回None
        """
        # 检查是否是完整型号模式：前缀-基本型号+铠装
        match = self.full_model_pattern.fullmatch(word)
        if match:
            prefix, base, armor = match.groups()
            tokens = []
            pos = start_offset
            
            # 前缀（如 NH, ZR）
            if prefix in self.model_prefixes:
                tokens.append({
                    "word": prefix,
                    "start": pos,
                    "end": pos + len(prefix),
                    "tag": "TYPE"
                })
                pos += len(prefix)
            else:
                tokens.append({
                    "word": prefix,
                    "start": pos,
                    "end": pos + len(prefix),
                    "tag": "O"
                })
                pos += len(prefix)
            
            # 连接符 -
            tokens.append({
                "word": "-",
                "start": pos,
                "end": pos + 1,
                "tag": "O"
            })
            pos += 1
            
            # 基本型号（如 YJV）
            if base in self.base_models:
                tokens.append({
                    "word": base,
                    "start": pos,
                    "end": pos + len(base),
                    "tag": self.base_models[base]
                })
            else:
                tokens.append({
                    "word": base,
                    "start": pos,
                    "end": pos + len(base),
                    "tag": "FEATURE"  # 默认当作特征
                })
            pos += len(base)
            
            # 铠装代码（如 22）
            if armor:
                if armor in self.armor_codes:
                    tokens.append({
                        "word": armor,
                        "start": pos,
                        "end": pos + len(armor),
                        "tag": "ARMOR"
                    })
                else:
                    tokens.append({
                        "word": armor,
                        "start": pos,
                        "end": pos + len(armor),
                        "tag": "O"
                    })
            
            return tokens
        
        # 检查是否是带铠装的基本型号（如 YJV22, KVV32）
        base_armor_match = re.fullmatch(r'([A-Z]{2,})(\d{2})', word)
        if base_armor_match:
            base, armor = base_armor_match.groups()
            if base in self.base_models or armor in self.armor_codes:
                tokens = []
                pos = start_offset
                
                # 基本型号
                tokens.append({
                    "word": base,
                    "start": pos,
                    "end": pos + len(base),
                    "tag": self.base_models.get(base, "FEATURE")
                })
                pos += len(base)
                
                # 铠装代码
                tokens.append({
                    "word": armor,
                    "start": pos,
                    "end": pos + len(armor),
                    "tag": "ARMOR" if armor in self.armor_codes else "O"
                })
                
                return tokens
        
        return None
    
    def _get_tag(self, word: str) -> str:
        """
        根据词获取标签
        
        优先级：
        1. 词典映射表精确匹配
        2. base_models 中的基本型号 -> FEATURE
        3. model_prefixes 中的前缀 -> TYPE
        4. 正则模式匹配（作为补充）
        5. 默认返回 'O'
        """
        # 1. 精确匹配词典
        if word in self.word_to_tag:
            return self.word_to_tag[word]
        
        # 2. 检查是否是基本型号（YJV, KVV等） -> FEATURE
        if word in self.base_models:
            return self.base_models[word]
        
        # 3. 检查是否是阻燃/耐火前缀（ZR, NH等） -> TYPE
        if word in self.model_prefixes:
            return 'TYPE'
        
        # 4. 正则匹配作为补充
        # 电压模式
        if self.voltage_pattern.fullmatch(word):
            return 'VOLTAGE'
        
        # 规格模式
        if self.spec_pattern.fullmatch(word):
            return 'SPEC'
        
        # 注意：不再用正则把所有大写字母当成 TYPE
        # 未知的型号应该由用户手动标注
        
        return 'O'
    
    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        """
        简单分词，只返回词列表
        """
        tokens = self.tokenize_with_position(text, preprocess)
        return [t["word"] for t in tokens]
    
    def add_word(self, word: str, freq: int = 50000, tag: str = None):
        """动态添加词汇"""
        jieba.add_word(word, freq=freq)
        if tag:
            self.word_to_tag[word] = tag
    
    def del_word(self, word: str):
        """删除词汇"""
        jieba.del_word(word)
        if word in self.word_to_tag:
            del self.word_to_tag[word]


class PipeTokenizer:
    """管道平台分词器
    
    用于识别管道相关标准编号（国标GB、美标ANSI、欧标EN）
    """
    
    def __init__(self, user_dict_path: str = None):
        """初始化分词器"""
        self.preprocessor = TextPreprocessor()
        self.word_to_tag: Dict[str, str] = {}
        
        # 确定词典路径
        if user_dict_path and os.path.exists(user_dict_path):
            dict_path = user_dict_path
        else:
            dict_path = os.path.join(os.path.dirname(__file__), "dict", "pipe_dict.txt")
        
        # 加载词典
        if os.path.exists(dict_path):
            self._load_dict(dict_path)
            print(f"[PipeTokenizer] 已加载词典: {dict_path}")
            print(f"  - 词条数: {len(self.word_to_tag)} 个")
        else:
            print(f"[PipeTokenizer] 词典文件不存在: {dict_path}")
    
    def _load_dict(self, dict_path: str):
        """加载词典"""
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    word, freq, tag = parts[0], int(parts[1]), parts[2]
                    jieba.add_word(word, freq=freq)
                    self.word_to_tag[word] = tag
                elif len(parts) == 2:
                    word, freq = parts[0], int(parts[1])
                    jieba.add_word(word, freq=freq)
    
    def tokenize_with_position(self, text: str, preprocess: bool = True) -> List[dict]:
        """分词并返回位置信息和标签"""
        if not text:
            return []
        
        # 预处理
        if preprocess:
            text = self.preprocessor.process(text)
        
        tokens = []
        pos = 0
        
        # 使用jieba分词
        words = list(jieba.cut(text))
        
        for word in words:
            if not word:
                continue
            
            # 查找词在文本中的位置
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            
            # 确定标签
            tag = self._get_tag(word)
            
            tokens.append({
                "word": word,
                "start": start,
                "end": end,
                "tag": tag
            })
            
            pos = end
        
        return tokens
    
    def _get_tag(self, word: str) -> str:
        """获取词的标签"""
        # 精确匹配词典
        if word in self.word_to_tag:
            return self.word_to_tag[word]
        return 'O'
    
    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        """简单分词，只返回词列表"""
        tokens = self.tokenize_with_position(text, preprocess)
        return [t["word"] for t in tokens]


# 分词器缓存
_tokenizer_cache: Dict[str, object] = {}


def get_tokenizer(platform: str = 'cable'):
    """
    获取指定平台的分词器（工厂函数）
    
    Args:
        platform: 平台名称 ('cable' 或 'pipe')
        
    Returns:
        对应平台的分词器实例
    """
    if platform not in _tokenizer_cache:
        if platform == 'pipe':
            _tokenizer_cache[platform] = PipeTokenizer()
        else:
            # 默认使用电缆/桥架分词器
            _tokenizer_cache[platform] = CableTokenizer()
    
    return _tokenizer_cache[platform]
