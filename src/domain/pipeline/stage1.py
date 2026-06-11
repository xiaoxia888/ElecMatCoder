from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stage1Snapshot:
    """一阶段统一快照，显式区分原始输出与送入编码的统一决策。"""

    decisions: dict[str, Any] = field(default_factory=dict)
    raw_values: dict[str, Any] = field(default_factory=dict)
    field_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": copy.deepcopy(self.decisions),
            "raw_values": copy.deepcopy(self.raw_values),
            "field_meta": copy.deepcopy(self.field_meta),
        }


class Stage1DecisionNormalizer:
    """把不同一阶段模型的输出收敛成统一 decisions/raw_values 结构。"""

    @staticmethod
    def _serialize_entity_value(entity: dict[str, Any]) -> Any:
        """
        将旧 predictor 的 entity 结构压平成统一值。

        仅 STANDARD 修饰项等少数场景会保留 bind_to_index / 位置信息。
        """
        value = entity.get("value") or entity.get("text", "")
        payload: dict[str, Any] = {"value": value}
        for key in ("bind_to_index", "start", "end", "subtype"):
            if entity.get(key) is not None:
                payload[key] = entity.get(key)
        if len(payload) == 1:
            return value
        return payload

    @staticmethod
    def _append_entity(entities: dict[str, Any], field: str, value: Any) -> None:
        """
        聚合旧 `entities[]` 输出。

        - 普通字段保留 string/list
        - 带 subtype 的结构化字段还原为嵌套对象
        """
        if isinstance(value, dict) and value.get("subtype") is not None:
            subtype = str(value["subtype"])
            subtype_value = value.get("value")
            if subtype_value in (None, ""):
                return
            field_obj = entities.get(field)
            if not isinstance(field_obj, dict):
                field_obj = {}
                entities[field] = field_obj
            bucket = field_obj.get(subtype)
            if not isinstance(bucket, list):
                bucket = []
                field_obj[subtype] = bucket
            bucket.append(subtype_value)
            return

        if field in entities:
            prev = entities[field]
            entities[field] = [prev, value] if not isinstance(prev, list) else prev + [value]
        else:
            entities[field] = value

    @classmethod
    def build_decisions(cls, predict_result: dict[str, Any]) -> dict[str, Any]:
        """
        构造统一的一阶段 decisions。

        优先级：
        1. `model_output.decisions`
        2. `model_output` 中非调试键
        3. 旧 `entities[]`
        """
        model_output = predict_result.get("model_output")
        if isinstance(model_output, dict):
            decisions = model_output.get("decisions")
            if isinstance(decisions, dict) and decisions:
                return copy.deepcopy(decisions)
            structured = {
                key: copy.deepcopy(value)
                for key, value in model_output.items()
                if not str(key).startswith("_") and key != "model_raw_response"
            }
            if structured:
                return structured

        entities: dict[str, Any] = {}
        for entity in predict_result.get("entities", []) if isinstance(predict_result.get("entities"), list) else []:
            if not isinstance(entity, dict):
                continue
            field = str(entity.get("type", "") or "").strip()
            if not field:
                continue
            cls._append_entity(entities, field, cls._serialize_entity_value(entity))
        return entities

    @staticmethod
    def build_field_meta(predict_result: dict[str, Any]) -> dict[str, Any]:
        """抽取一阶段字段级元信息。"""
        return copy.deepcopy(predict_result.get("extract_confidence_v2", {}) or {})

    @classmethod
    def build_raw_values(cls, predict_result: dict[str, Any]) -> dict[str, Any]:
        """
        提取真正的一阶段原始字段输出。

        规则：
        1. 优先读取各模型/规则源头输出
        2. 仅在缺失时回退到 decisions
        3. 不使用编码前补提/归并后的结构
        """
        raw_values: dict[str, Any] = {}
        model_output = predict_result.get("model_output")
        decisions = cls.build_decisions(predict_result)

        if isinstance(model_output, dict):
            type_output = model_output.get("_TYPE_MODEL_OUTPUT")
            if isinstance(type_output, dict) and type_output.get("TYPE") not in (None, "", [], {}):
                raw_values["TYPE"] = copy.deepcopy(type_output.get("TYPE"))

            material_output = model_output.get("_MATERIAL_MODEL_OUTPUT")
            if isinstance(material_output, dict) and material_output.get("MATERIAL") not in (None, "", [], {}):
                raw_values["MATERIAL"] = copy.deepcopy(material_output.get("MATERIAL"))

            standard_output = model_output.get("_STANDARD_MODEL_OUTPUT")
            if isinstance(standard_output, dict) and standard_output.get("STANDARD") not in (None, "", [], {}):
                raw_values["STANDARD"] = copy.deepcopy(standard_output.get("STANDARD"))

            structural_output = model_output.get("_STRUCTURAL_PROMPT")
            if isinstance(structural_output, dict):
                for field in ("SIZE", "THICKNESS", "PRESSURE"):
                    if structural_output.get(field) not in (None, "", [], {}):
                        raw_values[field] = copy.deepcopy(structural_output.get(field))

        for field, value in decisions.items():
            if field.startswith("_"):
                continue
            if field not in raw_values and value not in (None, "", [], {}):
                raw_values[field] = copy.deepcopy(value)

        return raw_values

    @classmethod
    def build_snapshot(cls, predict_result: dict[str, Any]) -> Stage1Snapshot:
        """一次性生成统一一阶段快照。"""
        return Stage1Snapshot(
            decisions=cls.build_decisions(predict_result),
            raw_values=cls.build_raw_values(predict_result),
            field_meta=cls.build_field_meta(predict_result),
        )
