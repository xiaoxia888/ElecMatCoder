from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TypeGeometry:
    """种类几何信息，当前只承载角度和半径。"""

    # 角度，例如 45 / 90。
    angle: str = ""
    # 半径，例如 1.5D / LR / SR。
    radius: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "ANGLE": self.angle,
            "RADIUS": self.radius,
        }


@dataclass
class TypeValue:
    """种类字段结构，供一阶段与二阶段统一使用。"""

    # 主体种类，例如 直管 / 弯头 / 法兰。
    body: str = ""
    # 几何信息。
    geometry: TypeGeometry = field(default_factory=TypeGeometry)
    # 制造方式，例如 SMLS / ERW。
    manu: list[str] = field(default_factory=list)
    # 连接方式，例如 BW / SW / MNPT。
    conn: list[str] = field(default_factory=list)
    # 密封面，例如 RF / RTJ / FF。
    seal: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "BODY": self.body,
            "GEOMETRY": self.geometry.to_dict(),
            "MANU": list(self.manu),
            "CONN": list(self.conn),
            "SEAL": list(self.seal),
        }
