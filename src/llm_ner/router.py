from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ROUTER_NOISE_TOKENS = {
    "",
    "管",
    "件",
    "法兰",
    "弯头",
    "三通",
    "接头",
    "大小头",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip_confidence(value: Any) -> float:
    score = _safe_float(value, 0.0)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _normalize_text(text: str) -> str:
    return str(text or "").strip()


def _normalize_api_kind(api_kind: str) -> str:
    key = str(api_kind or "chat_completions").strip().lower().replace("-", "_")
    aliases = {
        "openai_completions": "completions",
        "openai_chat_completions": "chat_completions",
        "chat": "chat_completions",
        "completion": "completions",
    }
    return aliases.get(key, key)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_json_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_fragment(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_empty_route(
    *,
    category: str = "",
    confidence: float = 0.0,
    reason: str = "",
    source: str = "",
    candidates: Optional[List[Dict[str, Any]]] = None,
    error: str = "",
    confidence_breakdown: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "category": category,
        "confidence": _clip_confidence(confidence),
        "reason": reason,
        "source": source,
        "candidates": candidates or [],
        "confidence_breakdown": confidence_breakdown or {},
        "review_required": False,
        "error": error,
    }


def _rounded_score(value: Any) -> float:
    return round(_clip_confidence(value), 4)


def _compute_margin_score(scores: List[float]) -> float:
    ordered = sorted((_clip_confidence(score) for score in scores), reverse=True)
    if not ordered:
        return 0.0
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    gap = max(0.0, top1 - top2)
    gap_ratio = 1.0 if top1 <= 0 else min(1.0, gap / top1)
    return _clip_confidence(0.6 * top1 + 0.4 * gap_ratio)


def _compute_keyword_anchor_score(strong_hits: int, keyword_hits: int) -> float:
    raw = strong_hits * 1.0 + keyword_hits * 0.35
    return _clip_confidence(raw / 2.0)


class RuleBasedCategoryRouter:
    def __init__(self, schema: Dict[str, Any], *, fallback_category: str = "其他管件"):
        self.schema = schema
        self.fallback_category = fallback_category
        self.categories = self._normalize_categories(schema)

    @staticmethod
    def _normalize_categories(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        categories = schema.get("categories", [])
        normalized: List[Dict[str, Any]] = []
        for item in categories:
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("name"))
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "definition": _normalize_text(item.get("definition")),
                    "keywords": [_normalize_text(v) for v in item.get("keywords", []) if _normalize_text(v)],
                    "strong_keywords": [_normalize_text(v) for v in item.get("strong_keywords", []) if _normalize_text(v)],
                    "examples": [_normalize_text(v) for v in item.get("examples", []) if _normalize_text(v)],
                }
            )
        return normalized

    def route(self, text: str) -> Dict[str, Any]:
        raw = str(text or "")
        lowered = raw.lower()
        scored: List[Dict[str, Any]] = []

        for item in self.categories:
            score = 0.0
            hits: List[str] = []
            strong_hit_count = 0
            keyword_hit_count = 0
            for token in item["strong_keywords"]:
                if token and token.lower() in lowered:
                    score += 3.0
                    hits.append(token)
                    strong_hit_count += 1
            for token in item["keywords"]:
                if token and token.lower() in lowered:
                    score += 1.0
                    hits.append(token)
                    keyword_hit_count += 1
            hits = _dedupe_keep_order(hits)
            if score > 0:
                scored.append(
                    {
                        "category": item["name"],
                        "raw_score": score,
                        "strong_hit_count": strong_hit_count,
                        "keyword_hit_count": keyword_hit_count,
                        "hits": hits[:8],
                    }
                )

        if not scored:
            return _build_empty_route(
                category=self.fallback_category,
                confidence=0.25,
                reason="未命中规则关键词，回退到默认类别",
                source="rules",
                candidates=[{"category": self.fallback_category, "score": 0.25}],
                confidence_breakdown={
                    "mode": "rules",
                    "evidence_score": 0.0,
                    "margin_score": 0.0,
                    "anchor_score": 0.0,
                },
            )

        scored.sort(key=lambda x: (-x["raw_score"], x["category"]))
        total = sum(item["raw_score"] for item in scored) or 1.0
        top = scored[0]
        candidates = [
            {"category": item["category"], "score": _rounded_score(max(0.01, min(0.99, item["raw_score"] / total)))}
            for item in scored[:5]
        ]
        evidence_score = _clip_confidence(top["raw_score"] / 6.0)
        margin_score = _compute_margin_score([item["score"] for item in candidates])
        anchor_score = _compute_keyword_anchor_score(
            top.get("strong_hit_count", 0),
            top.get("keyword_hit_count", 0),
        )
        top_conf = _clip_confidence(
            0.45 * evidence_score + 0.35 * margin_score + 0.20 * anchor_score
        )
        if top.get("strong_hit_count", 0) > 0:
            top_conf = max(top_conf, 0.6)
        return _build_empty_route(
            category=top["category"],
            confidence=top_conf,
            reason=f"规则命中关键词: {', '.join(top['hits'][:5])}",
            source="rules",
            candidates=candidates,
            confidence_breakdown={
                "mode": "rules",
                "evidence_score": _rounded_score(evidence_score),
                "margin_score": _rounded_score(margin_score),
                "anchor_score": _rounded_score(anchor_score),
                "strong_hit_count": top.get("strong_hit_count", 0),
                "keyword_hit_count": top.get("keyword_hit_count", 0),
                "top_raw_score": round(float(top["raw_score"]), 4),
            },
        )


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        norm = _normalize_text(value)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


def _is_noise_token(token: str) -> bool:
    token = _normalize_text(token)
    if not token:
        return True
    if token in ROUTER_NOISE_TOKENS:
        return True
    if len(token) == 1:
        return True
    return False


def _flatten_preview_keywords(node: Any) -> List[str]:
    values: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if not _is_noise_token(str(key)):
                values.append(str(key))
            values.extend(_flatten_preview_keywords(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(_flatten_preview_keywords(item))
    elif isinstance(node, str):
        if not _is_noise_token(node):
            values.append(node)
    return _dedupe_keep_order(values)


def _build_schema_from_preview_mapping(
    preview: Dict[str, Any],
    category_groups: Dict[str, List[str]],
) -> Dict[str, Any]:
    categories: List[Dict[str, Any]] = []
    for target_category, source_categories in category_groups.items():
        merged_keywords: List[str] = []
        for source_category in source_categories:
            source_node = preview.get(source_category)
            if source_node is None:
                continue
            merged_keywords.extend(_flatten_preview_keywords(source_node))
        merged_keywords = _dedupe_keep_order(merged_keywords)
        strong_keywords = [kw for kw in merged_keywords if len(kw) >= 4 or " " in kw or kw.isupper()]
        if not strong_keywords:
            strong_keywords = merged_keywords[:]
        categories.append(
            {
                "name": target_category,
                "definition": f"由 {', '.join(source_categories)} 汇总得到的路由类别",
                "strong_keywords": strong_keywords[:80],
                "keywords": merged_keywords[:160],
                "examples": [],
            }
        )
    return {"version": "preview-derived", "categories": categories}


class OpenAICompatibleCategoryRouter:
    def __init__(
        self,
        schema: Dict[str, Any],
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        api_kind: str = "chat_completions",
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 256,
        fallback_category: str = "其他管件",
    ):
        self.schema = schema
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model_name = str(model_name or "")
        self.api_kind = _normalize_api_kind(api_kind)
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.fallback_category = fallback_category
        self.category_meta: Dict[str, Dict[str, Any]] = {}
        self.category_names: List[str] = []
        for item in schema.get("categories", []):
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("name"))
            if not name:
                continue
            self.category_names.append(name)
            self.category_meta[name] = {
                "keywords": [_normalize_text(v) for v in item.get("keywords", []) if _normalize_text(v)],
                "strong_keywords": [_normalize_text(v) for v in item.get("strong_keywords", []) if _normalize_text(v)],
            }

        if not self.base_url:
            raise ValueError("router.base_url 不能为空")
        if not self.model_name:
            raise ValueError("router.model_name 不能为空")

    def route(self, text: str) -> Dict[str, Any]:
        content = self._call_model(text)
        parsed = _parse_json_fragment(content)
        if not parsed:
            raise RuntimeError(f"路由模型返回无法解析为 JSON: {content[:200]}")

        category = _normalize_text(parsed.get("category"))
        model_confidence = _clip_confidence(parsed.get("confidence", 0.0))
        reason = _normalize_text(parsed.get("reason"))
        candidates = self._normalize_candidates(parsed.get("candidates"))

        if category not in self.category_names:
            if candidates and candidates[0]["category"] in self.category_names:
                category = candidates[0]["category"]
            else:
                category = self.fallback_category
                model_confidence = min(model_confidence or 0.0, 0.5)
                reason = reason or "路由模型未返回合法类别，回退默认类别"

        if not candidates:
            candidates = [{"category": category, "score": model_confidence or 0.5}]

        anchor_score = self._compute_anchor_score(text, category)
        margin_score = _compute_margin_score([item["score"] for item in candidates])
        final_confidence = _clip_confidence(
            0.30 * anchor_score + 0.40 * margin_score + 0.30 * model_confidence
        )
        if category == self.fallback_category:
            final_confidence = min(final_confidence, 0.5)

        return _build_empty_route(
            category=category,
            confidence=final_confidence or 0.5,
            reason=reason or "LLM 路由完成",
            source="llm",
            candidates=candidates,
            confidence_breakdown={
                "mode": "llm",
                "model_confidence": _rounded_score(model_confidence),
                "margin_score": _rounded_score(margin_score),
                "anchor_score": _rounded_score(anchor_score),
            },
        )

    def _normalize_candidates(self, candidates: Any) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if not isinstance(candidates, list):
            return result
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            category = _normalize_text(item.get("category"))
            if not category:
                continue
            score = _clip_confidence(item.get("score", 0.0))
            result.append({"category": category, "score": score})
        result.sort(key=lambda x: (-x["score"], x["category"]))
        return result

    def _compute_anchor_score(self, text: str, category: str) -> float:
        meta = self.category_meta.get(category) or {}
        lowered = str(text or "").lower()
        strong_hits = 0
        keyword_hits = 0
        for token in meta.get("strong_keywords", []):
            if token and token.lower() in lowered:
                strong_hits += 1
        for token in meta.get("keywords", []):
            if token and token.lower() in lowered:
                keyword_hits += 1
        return _compute_keyword_anchor_score(strong_hits, keyword_hits)

    def _call_model(self, text: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        system_prompt = self._build_system_prompt()
        user_prompt = f"待分类描述：{text}"

        if self.api_kind == "completions":
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            resp = requests.post(
                f"{self.base_url}/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            return str(choices[0].get("text", "")).strip()

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    def _build_system_prompt(self) -> str:
        lines = [
            "你是工业管道材料描述路由器。",
            "你的任务不是抽取字段，而是把输入描述路由到一个最合适的专项模型。",
            "只能从给定类别中选择一个主类别。",
            "不得输出 markdown，不得输出解释段落，只能输出 JSON。",
            "JSON 格式固定为：",
            '{"category":"类别名","confidence":0.0,"reason":"一句话理由","candidates":[{"category":"类别名","score":0.0}]}',
            "类别定义如下：",
        ]
        for item in self.schema.get("categories", []):
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("name"))
            if not name:
                continue
            definition = _normalize_text(item.get("definition"))
            keywords = "、".join([_normalize_text(v) for v in item.get("keywords", [])[:8] if _normalize_text(v)])
            examples = "；".join([_normalize_text(v) for v in item.get("examples", [])[:3] if _normalize_text(v)])
            lines.append(f"- {name}：{definition}")
            if keywords:
                lines.append(f"  关键词：{keywords}")
            if examples:
                lines.append(f"  例子：{examples}")
        lines.extend(
            [
                "判定规则：",
                "1. 优先看主体物料名称，不要根据材质、标准、尺寸、壁厚、压力决定类别。",
                "2. 若存在多个候选类别，选择最贴近主体名词的那一个。",
                "3. 若把握不足，降低 confidence，并在 candidates 中给出次选类别。",
                "4. 只有“直管、法兰、管件”三类会继续进入后续编码；“阀门、螺栓、垫片、仪表、特殊件”只做分类，不做编码。",
                "5. 过滤器、疏水阀、阻火器、视镜、孔板、室内消火栓等，不要因为带有法兰、DN、压力等级就误判为法兰或管件，应按主体归到对应非编码类。",
                "6. 必须先识别主体物项名词，再看其余属性。主体物项名词如 Pipe、Tube、Flange、Elbow、Tee、Reducer、Valve、Gauge、Strainer、Gasket、Bolt 等，优先级最高。",
                "7. 标准、压力等级、密封面、尺寸、公称直径、连接面型等都只是配套属性，不能覆盖主体物项名词。典型属性包括 ASME B16.5、RF、RTJ、CL150、CL300、PN16、DN50、SCH40 等。",
                "8. 如果主体不是法兰，就算同时出现 ASME B16.5、RF、CL300、DN20，也不能判成法兰；如果主体不是管件，也不能因为出现对焊、承插焊、NPT 就判成管件。",
                "9. 当描述以 Gauge、Strainer、Trap、Sight Glass、Orifice、Hydrant 等非三类主体开头或为核心名词时，应优先判到 仪表 或 特殊件，而不是法兰。",
                "10. 对于 'Gauge, CL300, ASME B16.5 RF, DN20' 这类描述，主体是 Gauge，正确类别应为 仪表；CL300、ASME B16.5、RF、DN20 只是配套属性，不得据此改判为 法兰。",
            ]
        )
        return "\n".join(lines)


class HybridCategoryRouter:
    def __init__(
        self,
        rules_router: RuleBasedCategoryRouter,
        llm_router: OpenAICompatibleCategoryRouter,
        *,
        rules_direct_threshold: float = 0.92,
        fallback_category: str = "其他管件",
    ):
        self.rules_router = rules_router
        self.llm_router = llm_router
        self.rules_direct_threshold = float(rules_direct_threshold)
        self.fallback_category = fallback_category

    def route(self, text: str) -> Dict[str, Any]:
        rule_result = self.rules_router.route(text)
        if rule_result["confidence"] >= self.rules_direct_threshold:
            rule_result["review_required"] = False
            breakdown = dict(rule_result.get("confidence_breakdown") or {})
            breakdown["mode"] = "hybrid_rules_direct"
            breakdown["rules_direct_threshold"] = _rounded_score(self.rules_direct_threshold)
            rule_result["confidence_breakdown"] = breakdown
            return rule_result

        llm_result = self.llm_router.route(text)
        llm_result["rule_result"] = {
            "category": rule_result.get("category"),
            "confidence": rule_result.get("confidence"),
            "reason": rule_result.get("reason"),
            "candidates": rule_result.get("candidates", []),
            "confidence_breakdown": rule_result.get("confidence_breakdown", {}),
        }
        llm_confidence = _clip_confidence(llm_result.get("confidence"))
        rule_confidence = _clip_confidence(rule_result.get("confidence"))
        same_category = llm_result.get("category") == rule_result.get("category")
        agreement_score = 1.0 if same_category else 0.0
        rule_support_score = rule_confidence if same_category else 0.0
        llm_result["confidence"] = _clip_confidence(
            0.55 * llm_confidence + 0.25 * rule_support_score + 0.20 * agreement_score
        )
        llm_breakdown = dict(llm_result.get("confidence_breakdown") or {})
        llm_breakdown.update(
            {
                "mode": "hybrid",
                "llm_confidence": _rounded_score(llm_confidence),
                "rule_confidence": _rounded_score(rule_confidence),
                "rule_support_score": _rounded_score(rule_support_score),
                "agreement_score": _rounded_score(agreement_score),
                "same_category": same_category,
                "rules_direct_threshold": _rounded_score(self.rules_direct_threshold),
            }
        )
        llm_result["confidence_breakdown"] = llm_breakdown
        return llm_result


def build_category_router(router_config: Dict[str, Any], *, project_root: Path) -> Any:
    if not isinstance(router_config, dict):
        return None
    if not router_config.get("enabled", False):
        return None

    schema_path_value = router_config.get("category_schema_path")
    if not schema_path_value:
        raise ValueError("router.category_schema_path 未配置")
    schema_path = Path(schema_path_value)
    if not schema_path.is_absolute():
        schema_path = project_root / schema_path
    schema = _load_json(schema_path)
    category_groups = router_config.get("category_groups") or {}
    if "categories" not in schema and isinstance(category_groups, dict) and category_groups:
        schema = _build_schema_from_preview_mapping(schema, category_groups)

    backend = str(router_config.get("backend", "hybrid")).strip().lower()
    fallback_category = str(router_config.get("fallback_category", "其他管件")).strip() or "其他管件"
    rules_router = RuleBasedCategoryRouter(schema, fallback_category=fallback_category)

    if backend == "rules":
        return rules_router

    llm_router = OpenAICompatibleCategoryRouter(
        schema,
        base_url=router_config.get("base_url", ""),
        api_key=router_config.get("api_key", ""),
        model_name=router_config.get("model_name", ""),
        api_kind=router_config.get("api", "chat_completions"),
        timeout=int(router_config.get("timeout", 60)),
        temperature=float(router_config.get("temperature", 0.0)),
        max_tokens=int(router_config.get("max_tokens", 256)),
        fallback_category=fallback_category,
    )
    if backend == "openai_compatible":
        return llm_router
    if backend == "hybrid":
        return HybridCategoryRouter(
            rules_router,
            llm_router,
            rules_direct_threshold=float(router_config.get("rules_direct_threshold", 0.92)),
            fallback_category=fallback_category,
        )
    raise ValueError(f"不支持的 router.backend: {backend}")
