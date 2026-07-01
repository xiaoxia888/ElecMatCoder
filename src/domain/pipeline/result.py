from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Stage1RawPayload:
    """一阶段统一输出的字段结构，保留模型/规则最原始的识别结果。"""
    value: Any
    source: str = ""
    confidence: float | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2InputPayload:
    """实际送入二阶段编码器的字段结构，所有归一化/补提都体现在这里。"""
    value: Any
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2OutputPayload:
    """二阶段字段最终输出，只保留最终编码值。"""
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceDetail:
    """字段级置信度快照，和三层结构并列，不混入输出 code。"""
    stage1: float | None = None
    stage2: float | None = None
    field: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FieldStatus:
    """字段状态快照，供前端判断是否需审核及展示相似度。"""
    need_review: bool = False
    similarity: float | None = None
    is_exact_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FieldResultPayload:
    """单字段统一结果：一阶段原始结构、二阶段输入结构、二阶段输出结构。"""
    field_type: str
    stage1_raw: Stage1RawPayload
    stage2_input: Stage2InputPayload
    stage2_output: Stage2OutputPayload
    encode_confidence_v2: dict[str, Any] = field(default_factory=dict)
    confidence_detail: ConfidenceDetail = field(default_factory=ConfidenceDetail)
    status: FieldStatus = field(default_factory=FieldStatus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_type": self.field_type,
            "stage1_raw": self.stage1_raw.to_dict(),
            "stage2_input": self.stage2_input.to_dict(),
            "stage2_output": self.stage2_output.to_dict(),
            "encode_confidence_v2": dict(self.encode_confidence_v2 or {}),
            "confidence_detail": self.confidence_detail.to_dict(),
            "status": self.status.to_dict(),
        }


@dataclass
class EncodeResultPayload:
    """单次或批量 item 的统一返回载体，单条与批量详情共用这一套 schema。"""
    original_text: str
    processed_text: str
    final_code: str
    success: bool
    need_review: bool
    confidence: float | None
    fields: dict[str, FieldResultPayload] = field(default_factory=dict)
    route_info: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None
    difficulty_split: dict[str, Any] | None = None
    second_pass: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_encoding: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "final_code": self.final_code,
            "success": self.success,
            "need_review": self.need_review,
            "confidence": self.confidence,
            "fields": {key: value.to_dict() for key, value in self.fields.items()},
            "route_info": self.route_info,
            "routing": self.routing,
            "difficulty_split": self.difficulty_split,
            "second_pass": self.second_pass,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "skipped_encoding": self.skipped_encoding,
            "skip_reason": self.skip_reason,
        }
