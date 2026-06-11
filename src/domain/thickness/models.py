from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.common.ordered_item import OrderedValueItem


@dataclass
class ThicknessValue:
    """壁厚字段在编码前的统一结构。"""

    # 毫米壁厚。
    mm: list[str] = field(default_factory=list)
    # 英寸壁厚。
    inch: list[str] = field(default_factory=list)
    # SCH/STD/XS/XXS 等等级。
    schedule: list[str] = field(default_factory=list)
    # 系列码。
    series: list[str] = field(default_factory=list)
    # BWG。
    bwg: list[str] = field(default_factory=list)
    # 按原文顺序保留的原子项。
    ordered_items: list[OrderedValueItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "MM": list(self.mm),
            "INCH": list(self.inch),
            "SCHEDULE": list(self.schedule),
            "SERIES": list(self.series),
            "BWG": list(self.bwg),
            "ordered_items": [item.to_dict() for item in self.ordered_items],
        }
