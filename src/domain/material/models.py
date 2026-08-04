from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialItem:
    """材质字段单项结构。"""

    # 物理部件，例如 BODY / LINING / INNER_PIPE / OUTER_PIPE / FLANGE。
    part: str = "BODY"
    # 二阶段输入材质值。这里表示编码前的材质项，不应回写最终编码结果。
    value: str = ""
    # 特殊要求，例如 ZN / CE。
    special_req: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "PART": self.part,
            "VALUE": self.value,
            "SPECIAL_REQ": list(self.special_req),
        }
