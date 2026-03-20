# -*- coding: utf-8 -*-
"""
规格解析器
负责从各种格式的规格描述中提取关键尺寸
"""

import json
import logging
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .prompts import build_spec_parsing_prompt

logger = logging.getLogger(__name__)


@dataclass
class SpecParseResult:
    """规格解析结果"""
    original: str               # 原始规格描述
    code_format: str            # 编码格式 (如 "600X200")
    dimensions: Dict[str, Any]  # 解析出的尺寸
    parse_method: str           # 解析方法说明
    success: bool               # 是否成功
    used_llm: bool              # 是否使用了LLM
    confidence: float = 1.0     # 置信度 (0-1)


class SpecParser:
    """
    规格解析器
    
    支持多种格式：
    - W×H×L：800×200×6000
    - WXHXL:300X200X6000
    - 50W×50H
    - 400mm×150mm×2000mm
    - W=600,H=200,L=6000
    - 600宽×200高
    - φ89×4 (管道)
    - DN100 (公称直径)
    """
    
    def __init__(self, llm_service=None):
        """
        初始化解析器
        
        Args:
            llm_service: LLM服务实例（用于复杂格式）
        """
        self.llm = llm_service
        
        # 预编译正则表达式
        self._compile_patterns()
    
    def set_llm_service(self, llm_service):
        """设置LLM服务"""
        self.llm = llm_service
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        # 统一乘号
        self.multiply_pattern = re.compile(r'[×xX\*]')
        
        # 模式1: W×H×L：800×200×6000 或 WXHXL:300X200X6000
        self.pattern_wxhxl = re.compile(
            r'[WwＷｗ]\s*[×xX\*]?\s*[HhＨｈ]\s*[×xX\*]?\s*[LlＬｌ]?\s*[:：]?\s*'
            r'(\d+(?:\.\d+)?)\s*[×xX\*]\s*(\d+(?:\.\d+)?)\s*(?:[×xX\*]\s*(\d+(?:\.\d+)?))?'
        )
        
        # 模式2: 800×200×6000 (纯数字格式)
        self.pattern_numbers = re.compile(
            r'(\d+(?:\.\d+)?)\s*[×xX\*]\s*(\d+(?:\.\d+)?)\s*(?:[×xX\*]\s*(\d+(?:\.\d+)?))?'
        )
        
        # 模式3: 50W×50H 或 50w*50h
        self.pattern_num_label = re.compile(
            r'(\d+(?:\.\d+)?)\s*[WwＷｗ宽]\s*[×xX\*]?\s*(\d+(?:\.\d+)?)\s*[HhＨｈ高]'
        )
        
        # 模式4: W=600,H=200 或 W:600 H:200
        self.pattern_equals = re.compile(
            r'[WwＷｗ宽]\s*[=:：]\s*(\d+(?:\.\d+)?)[,，\s]*[HhＨｈ高]\s*[=:：]\s*(\d+(?:\.\d+)?)'
        )
        
        # 模式5: 600宽×200高 或 600宽200高
        self.pattern_chinese = re.compile(
            r'(\d+(?:\.\d+)?)\s*宽\s*[×xX\*]?\s*(\d+(?:\.\d+)?)\s*高'
        )
        
        # 模式6: φ89×4 或 Φ89*4 (管道外径×壁厚)
        self.pattern_pipe = re.compile(
            r'[φΦ]\s*(\d+(?:\.\d+)?)\s*[×xX\*]\s*(\d+(?:\.\d+)?)'
        )
        
        # 模式7: DN100 或 DN 100 (公称直径)
        self.pattern_dn = re.compile(
            r'DN\s*(\d+(?:\.\d+)?)', re.IGNORECASE
        )
        
        # 模式8: 400mm×150mm 或 400*150mm
        self.pattern_with_unit = re.compile(
            r'(\d+(?:\.\d+)?)\s*(?:mm|MM|cm|CM|m|M)?\s*[×xX\*]\s*'
            r'(\d+(?:\.\d+)?)\s*(?:mm|MM|cm|CM|m|M)?'
            r'(?:\s*[×xX\*]\s*(\d+(?:\.\d+)?)\s*(?:mm|MM|cm|CM|m|M)?)?'
        )
    
    async def parse(
        self, 
        spec_text: str, 
        material_type: str = "桥架",
        use_llm: bool = True
    ) -> SpecParseResult:
        """
        解析规格描述
        
        Args:
            spec_text: 规格描述文本
            material_type: 材料类型 (桥架/管道/电缆)
            use_llm: 是否使用LLM
            
        Returns:
            解析结果
        """
        if not spec_text or not spec_text.strip():
            return SpecParseResult(
                original=spec_text,
                code_format=None,
                dimensions={},
                parse_method="空值",
                success=False,
                used_llm=False,
                confidence=0.0
            )
        
        spec_text = spec_text.strip()
        
        # 由于规格格式多种多样，直接使用LLM解析
        # LLM能更好地理解各种变体格式
        if use_llm and self.llm:
            return await self._parse_with_llm(spec_text, material_type)
        
        # 如果不使用LLM，则尝试规则解析作为后备
        result = self._parse_by_rules(spec_text, material_type)
        if result.success:
            return result
        
        return SpecParseResult(
            original=spec_text,
            code_format=None,
            dimensions={},
            parse_method="无法解析（未启用LLM）",
            success=False,
            used_llm=False
        )
    
    def _parse_by_rules(
        self, 
        spec_text: str, 
        material_type: str
    ) -> SpecParseResult:
        """
        使用规则解析
        
        Args:
            spec_text: 规格文本
            material_type: 材料类型
            
        Returns:
            解析结果
        """
        # 管道类特殊处理
        if material_type in ["管道", "pipe"]:
            # 尝试DN格式
            match = self.pattern_dn.search(spec_text)
            if match:
                dn = match.group(1)
                return SpecParseResult(
                    original=spec_text,
                    code_format=f"DN{dn}",
                    dimensions={"dn": float(dn)},
                    parse_method="DN公称直径格式",
                    success=True,
                    used_llm=False,
                    confidence=1.0
                )
            
            # 尝试φ格式
            match = self.pattern_pipe.search(spec_text)
            if match:
                diameter = match.group(1)
                thickness = match.group(2)
                return SpecParseResult(
                    original=spec_text,
                    code_format=f"{diameter}X{thickness}",
                    dimensions={
                        "diameter": float(diameter),
                        "thickness": float(thickness)
                    },
                    parse_method="管道外径×壁厚格式",
                    success=True,
                    used_llm=False,
                    confidence=1.0
                )
        
        # 桥架类解析
        # 尝试各种模式
        patterns_to_try = [
            (self.pattern_wxhxl, "W×H×L格式"),
            (self.pattern_num_label, "数值+标签格式"),
            (self.pattern_equals, "等号赋值格式"),
            (self.pattern_chinese, "中文标识格式"),
            (self.pattern_with_unit, "带单位格式"),
            (self.pattern_numbers, "纯数字格式"),
        ]
        
        for pattern, method in patterns_to_try:
            match = pattern.search(spec_text)
            if match:
                groups = match.groups()
                width = groups[0] if len(groups) > 0 else None
                height = groups[1] if len(groups) > 1 else None
                length = groups[2] if len(groups) > 2 else None
                
                if width and height:
                    # 去除小数点后的0
                    w = self._format_number(width)
                    h = self._format_number(height)
                    
                    dimensions = {
                        "width": float(width),
                        "height": float(height)
                    }
                    if length:
                        dimensions["length"] = float(length)
                    
                    return SpecParseResult(
                        original=spec_text,
                        code_format=f"{w}X{h}",
                        dimensions=dimensions,
                        parse_method=method,
                        success=True,
                        used_llm=False
                    )
        
        return SpecParseResult(
            original=spec_text,
            code_format=None,
            dimensions={},
            parse_method="规则无法匹配",
            success=False,
            used_llm=False
        )
    
    def _format_number(self, num_str: str) -> str:
        """格式化数字，去除不必要的小数"""
        try:
            num = float(num_str)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return num_str
    
    async def _parse_with_llm(
        self, 
        spec_text: str, 
        material_type: str
    ) -> SpecParseResult:
        """
        使用LLM解析
        
        Args:
            spec_text: 规格文本
            material_type: 材料类型
            
        Returns:
            解析结果
        """
        try:
            messages = build_spec_parsing_prompt(
                material_type=material_type,
                spec_text=spec_text
            )
            
            # 记录完整的prompt
            logger.info(f"[LLM规格解析] 完整Prompt:\n{'-'*50}")
            for msg in messages:
                logger.info(f"[{msg['role']}]: {msg['content']}")
            logger.info(f"{'-'*50}")
            
            response = await self.llm.chat(messages, format="json")
            result = json.loads(response)
            
            # 记录完整响应日志
            logger.info(f"[LLM规格解析] 响应: {result}")
            
            code_format = result.get('code_format')
            width = result.get('width')
            height = result.get('height')
            
            # 验证code_format是否包含具体数值（不是模板如"W×H"）
            if code_format and any(char.isdigit() for char in str(code_format)):
                dimensions = {
                    "width": width,
                    "height": height
                }
                return SpecParseResult(
                    original=spec_text,
                    code_format=code_format,
                    dimensions=dimensions,
                    parse_method=f"LLM解析: 宽{width}, 高{height}",
                    success=True,
                    used_llm=True,
                    confidence=0.9
                )
            
            # 如果LLM返回的code_format不包含数字，尝试从width/height构建
            if width is not None and height is not None:
                code_format = f"{int(width)}X{int(height)}"
                return SpecParseResult(
                    original=spec_text,
                    code_format=code_format,
                    dimensions={"width": width, "height": height},
                    parse_method=f"LLM解析: 宽{width}, 高{height}",
                    success=True,
                    used_llm=True,
                    confidence=0.9
                )
            
            error = result.get('error', 'LLM未返回有效数值')
            return SpecParseResult(
                original=spec_text,
                code_format=None,
                dimensions={},
                parse_method=f"LLM解析失败: {error}",
                success=False,
                used_llm=True
            )
            
        except Exception as e:
            logger.error(f"LLM规格解析失败: {e}")
            return SpecParseResult(
                original=spec_text,
                code_format=None,
                dimensions={},
                parse_method=f"LLM调用失败: {str(e)}",
                success=False,
                used_llm=True
            )
    
    async def batch_parse(
        self,
        spec_texts: list,
        material_type: str = "桥架",
        use_llm: bool = True
    ) -> list:
        """
        批量解析规格
        
        Args:
            spec_texts: 规格文本列表
            material_type: 材料类型
            use_llm: 是否使用LLM
            
        Returns:
            解析结果列表
        """
        results = []
        for text in spec_texts:
            results.append(await self.parse(text, material_type, use_llm))
        return results

