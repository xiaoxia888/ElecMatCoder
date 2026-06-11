from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StandardItem:
    """规范字段单项结构，承载编码前排序和修饰信息。"""

    # 规范主体，例如 GBT8163。
    body: str = ""
    # 等级。
    grade: str = ""
    # 附录。
    appendix: str = ""
    # 方法。
    method: str = ""
    # 规范分类，例如 manufacturing / production。
    category: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "BODY": self.body,
            "GRADE": self.grade,
            "APPENDIX": self.appendix,
            "METHOD": self.method,
            "CATEGORY": self.category,
        }
