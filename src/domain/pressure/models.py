from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PressureItem:
    """压力字段单项结构。"""

    # 压力类别，例如 PN / CLASS / LB。
    type: str = ""
    # 压力值。
    value: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "TYPE": self.type,
            "VALUE": self.value,
        }


@dataclass
class PressureValue:
    """压力字段编码前统一结构。"""

    items: list[PressureItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "items": [item.to_dict() for item in self.items],
        }
