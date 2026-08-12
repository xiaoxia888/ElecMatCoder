from __future__ import annotations

import copy
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from src.domain.structural_v2 import (
    STRUCTURAL_V2_FIELD,
    canonical_structural_v2,
    has_v2_pressure,
    has_v2_size,
    has_v2_thickness,
    is_structural_v2,
)

logger = logging.getLogger(__name__)


def _is_empty_structural_value(field: str, value: Any) -> bool:
    if field in {"SIZE", "THICKNESS"}:
        return not isinstance(value, dict) or not any(
            v not in (None, "", [], {}) for v in value.values()
        )
    if field == "PRESSURE":
        return value in (None, "", [], {})
    return True


_PRESSURE_RE = re.compile(r"\b(?:PN\s*\d+(?:\.\d+)?|CL\s*\d+|CLASS\s*\d+|\d+\s*(?:LB|LBS)|\d+#|\d+(?:\.\d+)?\s*MPA|\d+(?:\.\d+)?\s*BAR)\b", re.IGNORECASE)
_THICKNESS_ANCHOR_RE = re.compile(r"(?:THK|壁厚|厚度|T=|S=|SCH|STD|XS|XXS|BWG)", re.IGNORECASE)
_SIZE_ANCHOR_RE = re.compile(r"(?:DN\s*\d|OD\s*\d|[Φφ]\s*\d|\bL\s*=)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _load_encoder_config() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "encoder" / "config" / "encoder_config.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("加载 encoder_config 失败: %s", exc)
        return {}


@lru_cache(maxsize=1)
def _load_material_special_req_reverse_map() -> Dict[str, str]:
    cfg = _load_encoder_config()
    raw = (
        ((cfg.get("material_special_req_supplement") or {}).get("suffix_aliases")) or {}
    )
    if not isinstance(raw, dict):
        return {}
    reverse: Dict[str, str] = {}
    for suffix, aliases in raw.items():
        suffix_text = str(suffix or "").strip().upper()
        if not suffix_text:
            continue
        reverse[suffix_text] = suffix_text
        if isinstance(aliases, list):
            for alias in aliases:
                alias_text = str(alias or "").strip().upper()
                if alias_text:
                    reverse[alias_text] = suffix_text
    return reverse


def _make_empty_extract_conf_v2(
    source: str,
    *,
    applicable: bool = True,
    reason: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "confidence": 0.0 if applicable else None,
        "reason": reason or ("field_missing" if applicable else "not_applicable"),
        "evidence": evidence or {},
    }


def _has_structural_anchor(text: str, field: str) -> bool:
    raw = text or ""
    if field == "SIZE":
        return bool(_SIZE_ANCHOR_RE.search(raw))
    if field == "THICKNESS":
        return bool(_THICKNESS_ANCHOR_RE.search(raw))
    if field == "PRESSURE":
        return bool(_PRESSURE_RE.search(raw))
    return False


def _summarize_stage1_structural_field(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            if item not in (None, "", [], {}):
                result[key] = item
        return result
    return value


class Stage1FieldOrchestrator:
    """
    阶段一统一编排器：
    1. 路由判断类别
    2. 并行执行 TYPE / MATERIAL / STANDARD / 结构提示词
    3. 合并为平台可消费的结构
    """

    def __init__(
        self,
        *,
        router: Any,
        type_factories: Dict[str, Callable[[], Any]],
        default_type_factory: Callable[[], Any],
        material_factory: Callable[[], Any],
        standard_factory: Callable[[], Any],
        structural_extractor_factory: Optional[Callable[[], Any]] = None,
        fallback_category: str = "其他管件",
        direct_threshold: float = 0.9,
        review_threshold: float = 0.7,
        encodable_categories: Optional[set[str]] = None,
        max_workers: int = 4,
    ):
        self.router = router
        self.type_factories = type_factories or {}
        self.default_type_factory = default_type_factory
        self.material_factory = material_factory
        self.standard_factory = standard_factory
        self.structural_extractor_factory = structural_extractor_factory
        self.fallback_category = fallback_category
        self.direct_threshold = float(direct_threshold)
        self.review_threshold = float(review_threshold)
        self.encodable_categories = set(encodable_categories or set())
        self.max_workers = max(1, int(max_workers))

        self._default_type_predictor = None
        self._type_predictor_cache: Dict[str, Any] = {}
        self._material_predictor = None
        self._standard_predictor = None
        self._structural_extractor = None

    def _get_default_type_predictor(self) -> Any:
        if self._default_type_predictor is None:
            self._default_type_predictor = self.default_type_factory()
        return self._default_type_predictor

    def _get_type_predictor(self, category: str) -> Any:
        category = str(category or "").strip()
        if not category:
            return self._get_default_type_predictor()
        if category in self._type_predictor_cache:
            return self._type_predictor_cache[category]
        factory = self.type_factories.get(category)
        if factory is None:
            return self._get_default_type_predictor()
        predictor = factory()
        self._type_predictor_cache[category] = predictor
        return predictor

    def _get_material_predictor(self) -> Any:
        if self._material_predictor is None:
            self._material_predictor = self.material_factory()
        return self._material_predictor

    def _get_standard_predictor(self) -> Any:
        if self._standard_predictor is None:
            self._standard_predictor = self.standard_factory()
        return self._standard_predictor

    def _get_structural_extractor(self) -> Any:
        if self.structural_extractor_factory is None:
            return None
        if self._structural_extractor is None:
            self._structural_extractor = self.structural_extractor_factory()
        return self._structural_extractor

    def predict(self, text: str, category_override: str = "") -> Dict[str, Any]:
        route_info = None
        selected_category = str(category_override or "").strip()
        if selected_category:
            route_info = {
                "category": selected_category,
                "confidence": 1.0,
                "reason": "imported_category",
                "source": "imported_category",
                "candidates": [{"category": selected_category, "score": 1.0}],
                "review_required": False,
                "imported_category": selected_category,
            }
        else:
            try:
                if self.router is not None:
                    route_info = self.router.route(text)
                    selected_category = str(route_info.get("category") or "").strip()
                else:
                    route_info = {
                        "category": "",
                        "confidence": 1.0,
                        "reason": "router_disabled",
                        "source": "router_disabled",
                        "candidates": [],
                        "review_required": False,
                    }
            except Exception as exc:
                logger.exception("路由失败，回退默认类别: %s", exc)
                route_info = {
                    "category": self.fallback_category,
                    "confidence": 0.0,
                    "reason": "路由异常，回退默认类别",
                    "source": "router_error",
                    "candidates": [{"category": self.fallback_category, "score": 0.0}],
                    "review_required": True,
                    "error": str(exc),
                }
                selected_category = self.fallback_category

        should_encode = (
            (not self.encodable_categories)
            or (selected_category in self.encodable_categories)
            or (not category_override and self.router is None)
        )
        route_payload = dict(route_info or {})
        confidence = float(route_payload.get("confidence") or 0.0)
        if category_override:
            route_payload["review_required"] = False
            route_payload["route_level"] = "imported"
        elif self.router is None:
            route_payload["review_required"] = False
            route_payload["route_level"] = "router_disabled"
        elif confidence >= self.direct_threshold:
            route_payload["review_required"] = False
            route_payload["route_level"] = "direct"
        elif confidence >= self.review_threshold:
            route_payload["review_required"] = True
            route_payload["route_level"] = "suggest_review"
        else:
            route_payload["review_required"] = True
            route_payload["route_level"] = "low_confidence"
        route_payload["selected_category"] = selected_category or ""
        route_payload["encoding_enabled"] = should_encode
        route_payload["skip_encoding_reason"] = "" if should_encode else f"类别“{selected_category}”只分类，不参与编码"

        if not should_encode:
            return {
                "tokens": [],
                "entities": [],
                "type_class": selected_category,
                "model_output": {},
                "model_raw_response": "",
                "extract_confidence": {},
                "extract_confidence_v2": {},
                "route_info": route_payload,
            }

        type_predictor = self._get_type_predictor(selected_category)
        material_predictor = self._get_material_predictor()
        standard_predictor = self._get_standard_predictor()
        structural_extractor = self._get_structural_extractor()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            predictor_futures: Dict[int, Any] = {}

            def submit_predict(predictor: Any):
                key = id(predictor)
                if key not in predictor_futures:
                    predictor_futures[key] = pool.submit(predictor.predict, text)
                return predictor_futures[key]

            type_future = submit_predict(type_predictor)
            material_future = submit_predict(material_predictor)
            standard_future = submit_predict(standard_predictor)
            structural_future = (
                pool.submit(structural_extractor.extract, text)
                if structural_extractor is not None
                else None
            )

            type_result = type_future.result()
            material_result = material_future.result()
            standard_result = standard_future.result()
            structural_result = structural_future.result() if structural_future is not None else None

        merged = self._merge_results(
            text=text,
            selected_category=selected_category,
            route_payload=route_payload,
            type_result=type_result,
            material_result=material_result,
            standard_result=standard_result,
            structural_result=structural_result,
        )
        return merged

    @staticmethod
    def _extract_field(model_result: Dict[str, Any], field: str) -> Any:
        if not isinstance(model_result, dict):
            return None
        model_output = model_result.get("model_output")
        if isinstance(model_output, dict):
            decisions = model_output.get("decisions")
            if isinstance(decisions, dict) and field in decisions:
                return copy.deepcopy(decisions.get(field))
            if field in model_output:
                return copy.deepcopy(model_output.get(field))
        return None

    @staticmethod
    def _extract_model_category(type_result: Dict[str, Any]) -> str:
        if not isinstance(type_result, dict):
            return ""
        model_output = type_result.get("model_output")
        if not isinstance(model_output, dict):
            return ""
        return str(model_output.get("CATEGORY") or model_output.get("category") or "").strip()

    @staticmethod
    def _extract_field_confidence(model_result: Dict[str, Any], field: str) -> Optional[float]:
        if not isinstance(model_result, dict):
            return None
        conf = model_result.get("extract_confidence")
        if isinstance(conf, dict):
            value = conf.get(field)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_field_confidence_v2(model_result: Dict[str, Any], field: str) -> Optional[Dict[str, Any]]:
        if not isinstance(model_result, dict):
            return None
        conf = model_result.get("extract_confidence_v2")
        if isinstance(conf, dict):
            value = conf.get(field)
            if isinstance(value, dict):
                return copy.deepcopy(value)
        return None

    @staticmethod
    def _build_structural_field_confidence_v2(
        *,
        text: str,
        field: str,
        field_value: Any,
        source: str = "prompt_extraction",
        prompt_status: str = "",
        prompt_error: str = "",
    ) -> Dict[str, Any]:
        if _is_empty_structural_value(field, field_value):
            applicable = _has_structural_anchor(text, field)
            reason = "field_missing_with_anchor" if applicable else "not_applicable"
            evidence = {"anchor_present": applicable}
            if source == "prompt_extraction" and prompt_status:
                reason = f"prompt_{prompt_status}"
                if prompt_error:
                    evidence["prompt_error"] = prompt_error
            return _make_empty_extract_conf_v2(
                source,
                applicable=applicable,
                reason=reason,
                evidence=evidence,
            )

        if source == "rule_extraction":
            return {
                "source": source,
                "confidence": 1.0,
                "reason": "deterministic_rule_extraction",
                "evidence": {"field_present": True, "deterministic": True},
            }

        return {
            "source": source,
            "confidence": None,
            "reason": "model_token_logprobs_unavailable",
            "evidence": {"field_present": True, "token_count": 0},
        }

    def _merge_results(
        self,
        *,
        text: str,
        selected_category: str,
        route_payload: Dict[str, Any],
        type_result: Dict[str, Any],
        material_result: Dict[str, Any],
        standard_result: Dict[str, Any],
        structural_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        decisions: Dict[str, Any] = {}
        extract_confidence: Dict[str, Any] = {}
        extract_confidence_v2: Dict[str, Any] = {
            "TYPE": _make_empty_extract_conf_v2("finetuned_model"),
            "MATERIAL": _make_empty_extract_conf_v2("finetuned_model"),
            "STANDARD": _make_empty_extract_conf_v2("finetuned_model"),
            "SIZE": _make_empty_extract_conf_v2(
                "prompt_extraction",
                applicable=_has_structural_anchor(text, "SIZE"),
                reason="field_missing_with_anchor" if _has_structural_anchor(text, "SIZE") else "not_applicable",
                evidence={"anchor_present": _has_structural_anchor(text, "SIZE")},
            ),
            "THICKNESS": _make_empty_extract_conf_v2(
                "prompt_extraction",
                applicable=_has_structural_anchor(text, "THICKNESS"),
                reason="field_missing_with_anchor" if _has_structural_anchor(text, "THICKNESS") else "not_applicable",
                evidence={"anchor_present": _has_structural_anchor(text, "THICKNESS")},
            ),
            "PRESSURE": _make_empty_extract_conf_v2(
                "prompt_extraction",
                applicable=_has_structural_anchor(text, "PRESSURE"),
                reason="field_missing_with_anchor" if _has_structural_anchor(text, "PRESSURE") else "not_applicable",
                evidence={"anchor_present": _has_structural_anchor(text, "PRESSURE")},
            ),
        }

        for field, result in (
            ("TYPE", type_result),
            ("MATERIAL", material_result),
            ("STANDARD", standard_result),
        ):
            value = self._extract_field(result, field)
            if value not in (None, "", [], {}):
                decisions[field] = value
                field_confidence = self._extract_field_confidence(result, field)
                if field_confidence is not None:
                    extract_confidence[field] = field_confidence
            field_conf_v2 = self._extract_field_confidence_v2(result, field)
            if isinstance(field_conf_v2, dict):
                extract_confidence_v2[field] = field_conf_v2

        self._apply_material_special_req_supplement(text, decisions)

        structural_visible = None
        structural_sources: Dict[str, str] = {}
        structural_model_confidences: Dict[str, Any] = {}
        if isinstance(structural_result, dict):
            if is_structural_v2(structural_result):
                structural_visible = canonical_structural_v2(structural_result)
                decisions[STRUCTURAL_V2_FIELD] = copy.deepcopy(structural_visible)
                structural_sources = {
                    str(k): str(v)
                    for k, v in (structural_result.get("_sources") or {}).items()
                    if k in {"SIZE", "THICKNESS", "PRESSURE"}
                }
                structural_model_confidences = copy.deepcopy(
                    structural_result.get("_extract_confidence_v2", {}) or {}
                )
                field_presence = {
                    "SIZE": has_v2_size(structural_result),
                    "THICKNESS": has_v2_thickness(structural_result),
                    "PRESSURE": has_v2_pressure(structural_result),
                }
                for field, present in field_presence.items():
                    provided_confidence = structural_model_confidences.get(field)
                    if isinstance(provided_confidence, dict):
                        extract_confidence_v2[field] = provided_confidence
                    else:
                        field_value = structural_result if field != "PRESSURE" else structural_result.get("PRESSURE")
                        extract_confidence_v2[field] = self._build_structural_field_confidence_v2(
                            text=text,
                            field=field,
                            field_value=field_value,
                            source=structural_sources.get(field, "finetuned_structural_model"),
                            prompt_status="",
                            prompt_error="",
                        )
                    confidence = extract_confidence_v2[field].get("confidence")
                    if present and confidence is not None:
                        extract_confidence[field] = float(confidence)
                logger.debug("[结构字段最终] 使用V2结构，不生成旧结构字段: %s", structural_visible)
            else:
                structural_visible = {
                    k: copy.deepcopy(v)
                    for k, v in structural_result.items()
                    if not str(k).startswith("_")
                }
                structural_sources = {
                    str(k): str(v)
                    for k, v in (structural_result.get("_sources") or {}).items()
                    if k in {"SIZE", "THICKNESS", "PRESSURE"}
                }
                structural_model_confidences = copy.deepcopy(
                    structural_result.get("_extract_confidence_v2", {}) or {}
                )
                for field in ("SIZE", "THICKNESS", "PRESSURE"):
                    field_value = copy.deepcopy(structural_result.get(field))
                    if field in {"SIZE", "THICKNESS"} and isinstance(field_value, dict):
                        items_key = f"{field}_ITEMS"
                        items_value = structural_result.get(items_key)
                        if isinstance(items_value, list) and items_value:
                            field_value["_ITEMS"] = copy.deepcopy(items_value)
                    if isinstance(structural_visible, dict):
                        structural_visible[field] = copy.deepcopy(field_value)
                    if not _is_empty_structural_value(field, field_value):
                        decisions[field] = field_value
                    provided_confidence = structural_model_confidences.get(field)
                    if isinstance(provided_confidence, dict):
                        extract_confidence_v2[field] = provided_confidence
                    else:
                        extract_confidence_v2[field] = self._build_structural_field_confidence_v2(
                            text=text,
                            field=field,
                            field_value=field_value,
                            source=structural_sources.get(field, "prompt_extraction"),
                            prompt_status="",
                            prompt_error="",
                        )
                    structural_confidence = extract_confidence_v2[field].get("confidence")
                    if structural_confidence is not None and not _is_empty_structural_value(field, field_value):
                        extract_confidence[field] = float(structural_confidence)
            logger.debug(
                "[结构字段最终] sources=%s, SIZE=%s, THICKNESS=%s, PRESSURE=%s, complex_trigger=%s",
                structural_sources,
                _summarize_stage1_structural_field(decisions.get("SIZE")),
                _summarize_stage1_structural_field(decisions.get("THICKNESS")),
                decisions.get("PRESSURE"),
                structural_result.get("_complex_structural_trigger") or {},
            )

        model_category = self._extract_model_category(type_result)
        if model_category:
            route_payload["model_category"] = model_category

        model_output = {
            "decisions": decisions,
            "_TYPE_MODEL_OUTPUT": copy.deepcopy(type_result.get("model_output", {})),
            "_TYPE_MODEL_RAW_RESPONSE": type_result.get("model_raw_response", ""),
            "_MATERIAL_MODEL_OUTPUT": copy.deepcopy(material_result.get("model_output", {})),
            "_MATERIAL_MODEL_RAW_RESPONSE": material_result.get("model_raw_response", ""),
            "_STANDARD_MODEL_OUTPUT": copy.deepcopy(standard_result.get("model_output", {})),
            "_STANDARD_MODEL_RAW_RESPONSE": standard_result.get("model_raw_response", ""),
        }
        if structural_visible is not None:
            model_output["_STRUCTURAL_PROMPT"] = structural_visible

        return {
            "text": text,
            "tokens": [],
            "entities": [],
            "type_class": selected_category,
            "model_output": model_output,
            "model_raw_response": "",
            "extract_confidence": extract_confidence,
            "extract_confidence_v2": extract_confidence_v2,
            "route_info": route_payload,
        }

    @staticmethod
    def _contains_material_special_req_alias(upper_text: str, alias_upper: str) -> bool:
        if not upper_text or not alias_upper:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in alias_upper):
            return alias_upper in upper_text
        pattern = re.compile(rf'(?<![A-Z0-9]){re.escape(alias_upper)}(?![A-Z0-9])', re.IGNORECASE)
        return bool(pattern.search(upper_text))

    @staticmethod
    def _normalize_special_req_values(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip().upper() for v in value if str(v).strip()]
        if value in (None, "", []):
            return []
        return [str(value).strip().upper()]

    def _apply_material_special_req_supplement(self, text: str, decisions: Dict[str, Any]) -> None:
        supplement_cfg = _load_encoder_config().get("material_special_req_supplement") or {}
        if not supplement_cfg.get("enabled", False):
            return

        material_value = decisions.get("MATERIAL")
        if not isinstance(material_value, list):
            return

        explicit_parts = [
            item
            for item in material_value
            if isinstance(item, dict) and item.get("PART") not in (None, "")
        ]
        # 原文级附加要求无法确定属于哪个物理部件。多部件结构必须使用
        # 模型已绑定到各 PART 的 SPECIAL_REQ，不能把法兰镀锌广播给主体。
        if len(explicit_parts) > 1:
            return

        reverse_map = _load_material_special_req_reverse_map()
        if not reverse_map:
            return

        upper_text = str(text or "").upper()
        matched_suffixes: list[str] = []
        for alias_upper, suffix in reverse_map.items():
            if alias_upper == suffix:
                continue
            if self._contains_material_special_req_alias(upper_text, alias_upper):
                if suffix not in matched_suffixes:
                    matched_suffixes.append(suffix)
        if not matched_suffixes:
            return

        for item in material_value:
            if not isinstance(item, dict):
                continue
            value = str(item.get("VALUE") or "").strip()
            if not value:
                continue
            current_raw = self._normalize_special_req_values(item.get("SPECIAL_REQ"))
            current: list[str] = []
            seen = set()
            for req in current_raw:
                canonical = reverse_map.get(req.upper(), req.upper())
                if canonical and canonical not in seen:
                    current.append(canonical)
                    seen.add(canonical)

            value_upper = value.upper()
            for suffix in matched_suffixes:
                if value_upper.endswith(suffix):
                    continue
                if suffix in seen:
                    continue
                current.append(suffix)
                seen.add(suffix)
            item["SPECIAL_REQ"] = current
