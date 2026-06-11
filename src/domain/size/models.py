from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.common.ordered_item import OrderedValueItem


@dataclass
class SizeValue:
    """尺寸字段在编码前的统一结构。"""

    # 公称直径列表。
    dn: list[str] = field(default_factory=list)
    # 外径列表。
    od: list[str] = field(default_factory=list)
    # 英制尺寸列表。
    inch: list[str] = field(default_factory=list)
    # 长度列表。
    length: list[str] = field(default_factory=list)
    # 按原文顺序保留的原子项。
    ordered_items: list[OrderedValueItem] = field(default_factory=list)
    # 来自壁厚字段的毫米上下文，供 OD/DN 消歧使用。
    thickness_mm_context: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "DN": list(self.dn),
            "OD": list(self.od),
            "INCH": list(self.inch),
            "LENGTH": list(self.length),
            "ordered_items": [item.to_dict() for item in self.ordered_items],
            "thickness_mm_context": list(self.thickness_mm_context),
        }
