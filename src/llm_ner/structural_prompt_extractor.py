# -*- coding: utf-8 -*-
"""
Prompt-based extractor for strong-evidence structural fields.

This is intentionally separate from the finetuned semantic parser.  It only
extracts SIZE / THICKNESS / PRESSURE and never attempts material/type/standard
normalization.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from src.tokenizer_utils.preprocessor import TextPreprocessor

from .llm_http_transport import (
    call_ollama_chat,
    call_openai_compatible,
)
from .structural_field_output_normalizer import StructuralFieldOutputNormalizer
from .structural_prompt import (
    get_pressure_system_prompt,
    get_size_length_system_prompt,
    get_thickness_system_prompt,
)

logger = logging.getLogger(__name__)


class StructuralPromptExtractor:
    """Extract SIZE / THICKNESS / PRESSURE with an instruction-following model."""

    SIZE_KEYS = StructuralFieldOutputNormalizer.SIZE_KEYS
    THICKNESS_KEYS = StructuralFieldOutputNormalizer.THICKNESS_KEYS
    ITEM_TYPES = StructuralFieldOutputNormalizer.ITEM_TYPES

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        self.config = config or {}
        self.debug = bool(debug or self.config.get("debug", False))
        self.backend = str(self.config.get("backend", "ollama")).strip() or "ollama"
        self.model_name = str(self.config.get("model_name", "")).strip()
        self.base_url = str(
            self.config.get("base_url")
            or self.config.get("ollama_url")
            or "http://localhost:11434"
        ).rstrip("/")
        self.api = str(self.config.get("api", "chat_completions")).strip()
        self.provider = self._infer_provider(
            str(self.config.get("provider", "") or ""),
            self.base_url,
            self.model_name,
        )
        self.api_key = str(self.config.get("api_key", "")).strip()
        self.timeout = float(self.config.get("timeout", 60))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", self.config.get("num_predict", 768)))
        self.max_workers = max(1, int(self.config.get("max_workers", 3)))
        self.reasoning_split = bool(self.config.get("reasoning_split", False))
        self.thinking_mode = str(self.config.get("thinking_mode", "auto") or "auto").strip().lower()
        self.reasoning_effort = str(self.config.get("reasoning_effort", "") or "").strip()
        self.prompt_version = str(
            self.config.get("prompt_version")
            or self.config.get("thickness_prompt_version")
            or "v1"
        ).strip().lower() or "v1"
        self.thickness_prompt_version = self.prompt_version
        self.size_length_prompt = get_size_length_system_prompt(debug=self.debug, version=self.prompt_version)
        self.thickness_prompt = get_thickness_system_prompt(debug=self.debug, version=self.prompt_version)
        self.pressure_prompt = get_pressure_system_prompt(debug=self.debug, version=self.prompt_version)
        self._last_usage: Dict[str, Any] = {}
        self.text_preprocessor = TextPreprocessor()
        if not self.model_name:
            raise RuntimeError("缺少结构字段模型配置: model_name")

    def extract(self, text: str) -> Dict[str, Any]:
        return self.extract_with_context(text)

    def extract_with_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        run_size_length: bool = True,
        run_thickness: bool = True,
        run_pressure: bool = True,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return self.empty_result()
        prompt_text = self._preprocess_text(text)
        partials, statuses, errors, usage = self._extract_partials(
            prompt_text,
            context=context,
            run_size_length=run_size_length,
            run_thickness=run_thickness,
            run_pressure=run_pressure,
        )
        merged = self._merge_partials(partials)
        normalized = self._normalize(merged)
        normalized["_raw"] = "\n\n".join(
            f"[{name.upper()}]\n{raw}" for name, raw in partials.items() if raw
        )
        normalized["_status"] = statuses
        normalized["_errors"] = errors
        normalized["_usage"] = usage
        return normalized

    def debug_extract(self, text: str) -> Dict[str, Any]:
        return self.debug_extract_with_context(text)

    def debug_extract_with_context(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        run_size_length: bool = True,
        run_thickness: bool = True,
        run_pressure: bool = True,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"trace": [], "field_diagnostics": {}, "final": self.empty_result(), "_raw": ""}
        prompt_text = self._preprocess_text(text)
        partials, statuses, errors, usage = self._extract_partials(
            prompt_text,
            context=context,
            run_size_length=run_size_length,
            run_thickness=run_thickness,
            run_pressure=run_pressure,
        )
        merged = self._merge_partials(partials)
        normalized_final = self._normalize(merged)
        normalized_final["_status"] = statuses
        normalized_final["_errors"] = errors
        normalized_final["_usage"] = usage
        return {
            "trace": [],
            "field_diagnostics": {
                "SIZE_LENGTH": {"raw": partials.get("size_length", "")},
                "THICKNESS": {"raw": partials.get("thickness", "")},
                "PRESSURE": {"raw": partials.get("pressure", "")},
            },
            "final": normalized_final,
            "_raw": "\n\n".join(
                f"[{name.upper()}]\n{raw}" for name, raw in partials.items() if raw
            ),
            "_parsed": merged,
            "_status": statuses,
            "_errors": errors,
            "_usage": usage,
        }

    @classmethod
    def empty_result(cls) -> Dict[str, Any]:
        return StructuralFieldOutputNormalizer.empty_result()

    def _extract_partials(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        run_size_length: bool = True,
        run_thickness: bool = True,
        run_pressure: bool = True,
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, Any]]:
        results: Dict[str, str] = {}
        statuses: Dict[str, str] = {}
        errors: Dict[str, str] = {}
        usage: Dict[str, Any] = {}

        size_context_items = self._extract_context_size_items(context)
        thickness_context_items = self._extract_context_thickness_items(context)

        if run_size_length:
            try:
                # 尺寸这一步也带上已识别壁厚上下文，便于在「外径x壁厚」等结构里反推尺寸
                size_user = self._build_user_content(text, thickness_items=thickness_context_items)
                results["size_length"] = self._generate(self.size_length_prompt, size_user)
                statuses["size_length"] = "ok"
                usage["size_length"] = dict(self._last_usage or {})
            except Exception as exc:
                logger.warning("[结构字段提示词][size_length] 调用失败: %s", exc)
                results["size_length"] = ""
                statuses["size_length"] = self._classify_error(exc)
                errors["size_length"] = str(exc)
                usage["size_length"] = {}
        else:
            results["size_length"] = ""
            statuses["size_length"] = "skipped"
            usage["size_length"] = {}

        if not size_context_items:
            parsed_size = self._parse_json(results.get("size_length", "") or "") or {}
            merged_size = self._merge_partials({"size_length": results.get("size_length", "") or ""})
            normalized_size = self._normalize(merged_size)
            size_context_items = normalized_size.get("SIZE_ITEMS") or self._normalize_items(
                parsed_size.get("SIZE_ITEMS"), self.ITEM_TYPES["SIZE_ITEMS"]
            )

        if run_thickness:
            try:
                thickness_user = self._build_user_content(text, size_items=size_context_items)
                results["thickness"] = self._generate(self.thickness_prompt, thickness_user)
                statuses["thickness"] = "ok"
                usage["thickness"] = dict(self._last_usage or {})
            except Exception as exc:
                logger.warning("[结构字段提示词][thickness] 调用失败: %s", exc)
                results["thickness"] = ""
                statuses["thickness"] = self._classify_error(exc)
                errors["thickness"] = str(exc)
                usage["thickness"] = {}
        else:
            results["thickness"] = ""
            statuses["thickness"] = "skipped"
            usage["thickness"] = {}

        if not thickness_context_items:
            merged_thickness = self._merge_partials({"thickness": results.get("thickness", "") or ""})
            normalized_thickness = self._normalize(merged_thickness)
            thickness_context_items = normalized_thickness.get("THICKNESS_ITEMS") or []

        if run_pressure:
            try:
                pressure_user = self._build_user_content(
                    text,
                    size_items=size_context_items,
                    thickness_items=thickness_context_items,
                )
                results["pressure"] = self._generate(self.pressure_prompt, pressure_user)
                statuses["pressure"] = "ok"
                usage["pressure"] = dict(self._last_usage or {})
            except Exception as exc:
                logger.warning("[结构字段提示词][pressure] 调用失败: %s", exc)
                results["pressure"] = ""
                statuses["pressure"] = self._classify_error(exc)
                errors["pressure"] = str(exc)
                usage["pressure"] = {}
        else:
            results["pressure"] = ""
            statuses["pressure"] = "skipped"
            usage["pressure"] = {}

        usage["total"] = self._merge_usage_totals(usage)
        return results, statuses, errors, usage

    def _merge_partials(self, partials: Dict[str, str]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "STRUCTURE_KIND": "",
            "SIZE_ITEMS": [],
            "THICKNESS_ITEMS": [],
            "PRESSURE": "",
            "LENGTH": "",
        }
        for name, raw in partials.items():
            parsed = self._parse_json(raw)
            if not isinstance(parsed, dict):
                if raw:
                    logger.warning("[结构字段提示词][%s] JSON解析失败: %s", name, raw[:200])
                continue
            self._merge_structure_kind(merged, parsed.get("STRUCTURE_KIND"))
            if name == "size_length":
                if isinstance(parsed.get("SIZE_ITEMS"), list):
                    merged["SIZE_ITEMS"] = parsed.get("SIZE_ITEMS") or []
                merged["LENGTH"] = str(parsed.get("LENGTH", "") or "").strip()
            elif name == "thickness":
                if isinstance(parsed.get("THICKNESS_ITEMS"), list):
                    merged["THICKNESS_ITEMS"] = parsed.get("THICKNESS_ITEMS") or []
            elif name == "pressure":
                merged["PRESSURE"] = str(parsed.get("PRESSURE", "") or "").strip()
        return merged

    @staticmethod
    def _merge_structure_kind(merged: Dict[str, Any], value: Any) -> None:
        incoming = str(value or "").strip().upper()
        if incoming not in {"NORMAL", "LINED", "JACKETED"}:
            return
        priority = {"": 0, "NORMAL": 1, "LINED": 2, "JACKETED": 3}
        current = str(merged.get("STRUCTURE_KIND") or "").strip().upper()
        if priority.get(incoming, 0) >= priority.get(current, 0):
            merged["STRUCTURE_KIND"] = incoming

    def _generate(self, system_prompt: str, user_content: str) -> str:
        stop = ["\n\n输入：", "\n\n【", "\n【"]
        if self.backend == "ollama":
            response = call_ollama_chat(
                base_url=self.base_url,
                model_name=self.model_name,
                system_prompt=system_prompt,
                user_content=user_content,
                timeout=self.timeout,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )
            self._last_usage = response.usage
            return response.text
        if self.backend in {"openai_compatible", "openai"}:
            response = call_openai_compatible(
                base_url=self.base_url,
                model_name=self.model_name,
                system_prompt=system_prompt,
                user_content=user_content,
                timeout=self.timeout,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api=self.api,
                api_key=self.api_key,
                stop=stop,
                reasoning_split=self.reasoning_split,
                extra_body=self._build_provider_extra_body(),
            )
            self._last_usage = response.usage
            return response.text
        raise RuntimeError(f"不支持的结构字段提示词后端: {self.backend}")

    @staticmethod
    def _infer_provider(provider: str, base_url: str, model_name: str) -> str:
        explicit = str(provider or "").strip().lower()
        if explicit and explicit != "auto":
            return explicit
        url = str(base_url or "").lower()
        model = str(model_name or "").lower()
        if "deepseek" in url or "deepseek" in model:
            return "deepseek"
        if "minimax" in url or model.startswith("minimax-"):
            return "minimax"
        return "generic"

    def _build_provider_extra_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self.provider in {"deepseek", "minimax"}:
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            if self.thinking_mode and self.thinking_mode != "auto":
                body["thinking"] = {"type": self.thinking_mode}
            return body
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.thinking_mode and self.thinking_mode not in {"", "auto"}:
            body["thinking"] = self.thinking_mode
        return body

    @classmethod
    def _extract_context_size_items(cls, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        structured = (context or {}).get("SIZE")
        items = (context or {}).get("SIZE_ITEMS")
        if isinstance(items, list):
            normalized = cls._normalize_items(items, cls.ITEM_TYPES["SIZE_ITEMS"])
            if normalized:
                return normalized
        if isinstance(structured, dict):
            collected: List[Dict[str, str]] = []
            for key in cls.SIZE_KEYS:
                if key == "LENGTH":
                    continue
                for value in cls._normalize_list(structured.get(key)):
                    collected.append({"type": key, "value": str(value)})
            return cls._normalize_items(collected, cls.ITEM_TYPES["SIZE_ITEMS"])
        return []

    @classmethod
    def _extract_context_thickness_items(cls, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        structured = (context or {}).get("THICKNESS")
        items = (context or {}).get("THICKNESS_ITEMS")
        if isinstance(items, list):
            normalized = cls._normalize_items(items, cls.ITEM_TYPES["THICKNESS_ITEMS"])
            if normalized:
                return normalized
        if isinstance(structured, dict):
            collected: List[Dict[str, str]] = []
            for key in cls.THICKNESS_KEYS:
                for value in cls._normalize_list(structured.get(key)):
                    collected.append({"type": key, "value": str(value)})
            return cls._normalize_items(collected, cls.ITEM_TYPES["THICKNESS_ITEMS"])
        return []

    @staticmethod
    def _build_user_content(
        text: str,
        size_items: Optional[List[Dict[str, str]]] = None,
        thickness_items: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        parts = [f"输入：\n{text}"]
        if size_items:
            size_lines = "\n".join(
                f'- {item.get("type", "").upper()}: {item.get("value", "")}'
                for item in size_items
                if item.get("type") and item.get("value")
            )
            if size_lines:
                parts.append(
                    "已识别尺寸结果，已被识别的不要重复识别：\n"
                    f"{size_lines}"
                )
        if thickness_items:
            thickness_lines = "\n".join(
                f'- {item.get("type", "").upper()}: {item.get("value", "")}'
                for item in thickness_items
                if item.get("type") and item.get("value")
            )
            if thickness_lines:
                parts.append(
                    "已识别壁厚结果，已被识别的不要重复识别：\n"
                    f"{thickness_lines}"
                )
        parts.append("只输出一个 JSON 对象，输出到最后一个 } 后立即停止。")
        return "\n\n".join(parts)

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError):
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                return "rate_limit"
            if status_code == 503:
                return "service_unavailable"
            if status_code == 401:
                return "unauthorized"
            if status_code == 403:
                return "forbidden"
        if isinstance(exc, requests.ReadTimeout):
            return "timeout"
        if isinstance(exc, requests.ConnectTimeout):
            return "connect_timeout"
        if isinstance(exc, requests.Timeout):
            return "timeout"
        if isinstance(exc, requests.RequestException):
            return "request_error"
        return "error"

    def _preprocess_text(self, text: str) -> str:
        raw = str(text or "")
        if not raw:
            return raw
        raw = self.text_preprocessor.process(raw)
        # 仅在前一段属于这些 token 时，把连接到 `SCH...` 前面的 `X/x` 统一改为 `*`：
        # - 数字 / 小数
        # - STD / XS / XXS
        # - SCH数字 / SCH数字S
        # - S-数字 / S-数字S
        # 例如：
        # - φ139.7XSCH10S -> φ139.7*SCH10S
        # - XXSXSCH40 -> XXS*SCH40
        # - SCH40XSCH80 -> SCH40*SCH80
        pattern = re.compile(
            r"(?i)(\d+(?:\.\d+)?|STD|XS|XXS|SCH\d+S?|S-\d+S?)\s*[xX](?=\s*SCH\d)"
        )
        return pattern.sub(lambda m: f"{m.group(1)}*", raw)

    @staticmethod
    def _merge_usage_totals(usage_map: Dict[str, Any]) -> Dict[str, int]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for key, usage in usage_map.items():
            if key == "total" or not isinstance(usage, dict):
                continue
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        cleaned = str(raw or "").strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            balanced = StructuralPromptExtractor._extract_first_json_object(cleaned)
            if balanced:
                try:
                    return json.loads(balanced)
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return ""

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

    @staticmethod
    def _group_items(items: List[Dict[str, str]], keys: tuple[str, ...]) -> Dict[str, List[str]]:
        return StructuralFieldOutputNormalizer.group_items(items, keys)
