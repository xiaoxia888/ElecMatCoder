from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional

from .structural_field_output_normalizer import StructuralFieldOutputNormalizer
from .structured_llamafactory_adapter import build_structured_predictor_from_config

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURAL_FIELD_INSTRUCTION = (
    "你是工业管道材料描述结构化抽取助手。请从材料描述中抽取尺寸、长度、壁厚和磅级信息，"
    "并输出严格 JSON。输出字段只能包含 SIZE_ITEMS、LENGTH、THICKNESS_ITEMS、PRESSURE。"
    "LENGTH 统一转换为毫米单位，SIZE_ITEMS 和 THICKNESS_ITEMS 按原文顺序输出，不要解释，"
    "不要输出 JSON 以外的内容。"
)


class StructuralFieldModelExtractor:
    """使用本地/服务化微调模型抽取 SIZE / THICKNESS / PRESSURE。"""

    SIZE_KEYS = StructuralFieldOutputNormalizer.SIZE_KEYS
    THICKNESS_KEYS = StructuralFieldOutputNormalizer.THICKNESS_KEYS
    ITEM_TYPES = StructuralFieldOutputNormalizer.ITEM_TYPES

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config or {})
        self.backend = str(self.config.get("backend", "")).strip()
        self.predictor = build_structured_predictor_from_config(
            self.config,
            default_instruction=DEFAULT_STRUCTURAL_FIELD_INSTRUCTION,
            log_label="结构字段模型",
        )

    @classmethod
    def empty_result(cls) -> Dict[str, Any]:
        return StructuralFieldOutputNormalizer.empty_result()

    def extract_with_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        run_size_length: bool = True,
        run_thickness: bool = True,
        run_pressure: bool = True,
    ) -> Dict[str, Any]:
        del context  # 本地微调结构字段模型直接基于原文抽取，不消费额外上下文。
        result = self.empty_result()
        status = {
            "size_length": "skipped" if not run_size_length else "ok",
            "thickness": "skipped" if not run_thickness else "ok",
            "pressure": "skipped" if not run_pressure else "ok",
        }
        errors: Dict[str, str] = {}

        try:
            predict_result = self.predictor.predict(str(text or ""))
            model_output = predict_result.get("model_output") if isinstance(predict_result, dict) else {}
            normalized = self._normalize(model_output if isinstance(model_output, dict) else {})
            result.update(normalized)
            result["_raw"] = str((predict_result or {}).get("model_raw_response", "") or "")
            result["_extract_confidence_v2"] = self._map_model_confidences(predict_result or {})
        except Exception as exc:  # pragma: no cover - 运行时兜底
            logger.warning("[结构字段模型] 调用失败: %s", exc)
            if run_size_length:
                status["size_length"] = "error"
                errors["size_length"] = str(exc)
            if run_thickness:
                status["thickness"] = "error"
                errors["thickness"] = str(exc)
            if run_pressure:
                status["pressure"] = "error"
                errors["pressure"] = str(exc)
            result["_raw"] = ""

        if not run_size_length:
            result["SIZE"] = {"_ITEMS": []}
            result["SIZE_ITEMS"] = []
            result["LENGTH"] = ""
        if not run_thickness:
            result["THICKNESS"] = {"_ITEMS": []}
            result["THICKNESS_ITEMS"] = []
        if not run_pressure:
            result["PRESSURE"] = ""

        result["_status"] = status
        result["_errors"] = errors
        result["_usage"] = {}
        return result

    @staticmethod
    def _map_model_confidences(predict_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        source = predict_result.get("extract_confidence_v2")
        source = source if isinstance(source, dict) else {}

        def aggregate(*keys: str) -> Optional[Dict[str, Any]]:
            items = [source.get(key) for key in keys if isinstance(source.get(key), dict)]
            values = [item.get("confidence") for item in items if item.get("confidence") is not None]
            if not values:
                return None
            confidences = [max(1e-12, min(1.0, float(value))) for value in values]
            confidence = math.exp(sum(math.log(value) for value in confidences) / len(confidences))
            token_count = sum(int((item.get("evidence") or {}).get("token_count") or 0) for item in items)
            return {
                "source": "model_token_logprobs",
                "confidence": confidence,
                "reason": "generated_value_token_probability",
                "evidence": {"source_fields": list(keys), "token_count": token_count},
            }

        mapped = {
            "SIZE": aggregate("SIZE_ITEMS", "LENGTH"),
            "THICKNESS": aggregate("THICKNESS_ITEMS"),
            "PRESSURE": aggregate("PRESSURE"),
        }
        return {field: value for field, value in mapped.items() if value is not None}

    @classmethod
    def _normalize(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return StructuralFieldOutputNormalizer.normalize(parsed)

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        return StructuralFieldOutputNormalizer.normalize_list(value)

    @classmethod
    def _normalize_items(cls, value: Any, allowed_types: set[str]) -> List[Dict[str, str]]:
        return StructuralFieldOutputNormalizer.normalize_items(value, allowed_types)

    @staticmethod
    def _normalize_item_value(item_type: str, raw_value: Any) -> str:
        return StructuralFieldOutputNormalizer.normalize_item_value(item_type, raw_value)
