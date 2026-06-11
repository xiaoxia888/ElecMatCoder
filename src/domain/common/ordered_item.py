from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OrderedValueItem:
    """有序原子项，用于尺寸/壁厚等字段在编码阶段保留出现顺序。"""

    # 原子项类别，如 DN / OD / MM / SCHEDULE。
    type: str
    # 原子项值，如 450 / 8 / S80。
    value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
