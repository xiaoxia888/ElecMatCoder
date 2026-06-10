"""兼容导入入口。

新的 TYPE 编码器已迁移到 `src.encoder.processors.type_encoder`。
这里保留旧路径，避免后续接入主链时重复改 import。
"""

from .processors.type_encoder import TypeEncoder, TypeEncodingResult, get_type_encoder

__all__ = ["TypeEncoder", "TypeEncodingResult", "get_type_encoder"]
