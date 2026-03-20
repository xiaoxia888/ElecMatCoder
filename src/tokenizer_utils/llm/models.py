"""
数据模型定义
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """实体类型枚举"""
    
    NAME = "NAME"           # 名称：电力电缆、控制电缆、光纤等
    MATERIAL = "MATERIAL"   # 材质：铜芯、铝芯、本安型等
    TYPE = "TYPE"  # 类型：阻燃、阻燃耐火、阻燃A类等
    ARMOR = "ARMOR"         # 铠装：22、32、42、53、非铠装
    FEATURE = "FEATURE"     # 特征：交联聚乙烯绝缘聚氯乙烯护套等
    VOLTAGE = "VOLTAGE"     # 额定电压：0.6/1KV、8.7/15KV等
    SPEC = "SPEC"           # 规格：3*6、3×95+2×50等


# 实体类型的中文名称映射
ENTITY_TYPE_NAMES = {
    EntityType.NAME: "名称",
    EntityType.MATERIAL: "材质",
    EntityType.TYPE: "类型",
    EntityType.ARMOR: "铠装",
    EntityType.FEATURE: "特征",
    EntityType.VOLTAGE: "额定电压",
    EntityType.SPEC: "规格",
}


class Entity(BaseModel):
    """单个实体"""
    
    text: str = Field(..., description="实体文本")
    label: EntityType = Field(..., description="实体类型")
    start: Optional[int] = Field(None, description="起始位置（字符索引）")
    end: Optional[int] = Field(None, description="结束位置（字符索引）")
    
    class Config:
        use_enum_values = True


class AnnotationResult(BaseModel):
    """标注结果"""
    
    text: str = Field(..., description="原始文本")
    entities: List[Entity] = Field(default_factory=list, description="识别出的实体列表")
    raw_response: Optional[str] = Field(None, description="LLM原始响应")
    success: bool = Field(True, description="标注是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    def to_bio_format(self) -> List[tuple]:
        """
        转换为BIO格式
        
        Returns:
            List of (char, tag) tuples
        """
        # 初始化所有字符为O
        bio_tags = ["O"] * len(self.text)
        
        # 根据实体标注BIO标签
        for entity in self.entities:
            if entity.start is not None and entity.end is not None:
                for i in range(entity.start, entity.end):
                    if i < len(bio_tags):
                        if i == entity.start:
                            bio_tags[i] = f"B-{entity.label}"
                        else:
                            bio_tags[i] = f"I-{entity.label}"
        
        return list(zip(self.text, bio_tags))
    
    def to_bio_string(self) -> str:
        """
        转换为BIO格式字符串
        
        Returns:
            BIO格式的字符串，每行一个字符和标签
        """
        bio_data = self.to_bio_format()
        lines = [f"{char} {tag}" for char, tag in bio_data]
        return "\n".join(lines)


class BatchAnnotationResult(BaseModel):
    """批量标注结果"""
    
    results: List[AnnotationResult] = Field(default_factory=list)
    total: int = Field(0, description="总数")
    success_count: int = Field(0, description="成功数")
    failed_count: int = Field(0, description="失败数")
    
    def add_result(self, result: AnnotationResult):
        """添加单条结果"""
        self.results.append(result)
        self.total += 1
        if result.success:
            self.success_count += 1
        else:
            self.failed_count += 1

