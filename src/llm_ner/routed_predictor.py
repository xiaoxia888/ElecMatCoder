from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class RoutedNERPredictor:
    def __init__(
        self,
        *,
        router: Any,
        default_factory: Callable[[], Any],
        category_factories: Optional[Dict[str, Callable[[], Any]]] = None,
        fallback_category: str = "其他管件",
        direct_threshold: float = 0.9,
        review_threshold: float = 0.7,
        encodable_categories: Optional[set[str]] = None,
    ):
        self.router = router
        self.default_factory = default_factory
        self.category_factories = category_factories or {}
        self.fallback_category = fallback_category
        self.direct_threshold = float(direct_threshold)
        self.review_threshold = float(review_threshold)
        self.encodable_categories = set(encodable_categories or set())
        self._default_predictor = None
        self._predictor_cache: Dict[str, Any] = {}

    def _get_default_predictor(self) -> Any:
        if self._default_predictor is None:
            self._default_predictor = self.default_factory()
        return self._default_predictor

    def _get_predictor(self, category: str) -> Any:
        category = str(category or "").strip()
        if not category:
            return self._get_default_predictor()
        if category in self._predictor_cache:
            return self._predictor_cache[category]
        factory = self.category_factories.get(category)
        if factory is None:
            return self._get_default_predictor()
        predictor = factory()
        self._predictor_cache[category] = predictor
        return predictor

    def predict(self, text: str) -> Dict[str, Any]:
        route_info = None
        selected_category = ""
        try:
            if self.router is not None:
                route_info = self.router.route(text)
                selected_category = str(route_info.get("category") or "").strip()
        except Exception as exc:
            logger.exception("路由失败，回退默认模型: %s", exc)
            route_info = {
                "category": self.fallback_category,
                "confidence": 0.0,
                "reason": "路由异常，回退默认模型",
                "source": "router_error",
                "candidates": [{"category": self.fallback_category, "score": 0.0}],
                "review_required": True,
                "error": str(exc),
            }
            selected_category = self.fallback_category

        should_encode = (not self.encodable_categories) or (selected_category in self.encodable_categories)
        route_payload = dict(route_info or {})
        confidence = float(route_payload.get("confidence") or 0.0)
        if confidence >= self.direct_threshold:
            route_payload["review_required"] = False
            route_payload["route_level"] = "direct"
        elif confidence >= self.review_threshold:
            route_payload["review_required"] = True
            route_payload["route_level"] = "suggest_review"
        else:
            route_payload["review_required"] = True
            route_payload["route_level"] = "low_confidence"
        route_payload["selected_category"] = selected_category or ""
        route_payload["selected_model_scope"] = (
            selected_category if selected_category in self.category_factories else "default"
        )
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
                "route_info": route_payload,
            }

        predictor = self._get_predictor(selected_category)
        result = predictor.predict(text)
        if not isinstance(result, dict):
            raise RuntimeError("下游 predictor 返回结果不是 dict")
        result["route_info"] = route_payload
        return result
