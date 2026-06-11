from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialItem:
    """材质字段单项结构。"""

    # 材质角色，例如 MAIN / LINING。
    role: str = "MAIN"
    # 二阶段输入材质值。这里表示编码前的材质项，不应回写最终编码结果。
    value: str = ""
    # 特殊要求，例如 ZN / CE。
    special_req: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ROLE": self.role,
            "VALUE": self.value,
            "SPECIAL_REQ": list(self.special_req),
        }
