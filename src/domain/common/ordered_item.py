from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderedValueItem:
    """有序原子项，用于尺寸/壁厚等字段在编码阶段保留出现顺序。"""

    # 原子项类别，如 DN / OD / MM / SCHEDULE。
    type: str
    # 原子项值，如 450 / 8 / S80。
    value: str
    # 结构角色，如 BASE / LINING / INNER / OUTER；普通管子管件为空。
    role: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"type": self.type, "value": self.value}
        if self.role:
            payload = {"role": self.role, **payload}
        return payload
