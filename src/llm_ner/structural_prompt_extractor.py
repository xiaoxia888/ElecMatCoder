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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

from src.tokenizer_utils.preprocessor import TextPreprocessor

from .structural_prompt import (
    get_pressure_system_prompt,
    get_size_length_system_prompt,
    get_thickness_system_prompt,
)

logger = logging.getLogger(__name__)


class StructuralPromptExtractor:
    """Extract SIZE / THICKNESS / PRESSURE with an instruction-following model."""

    SIZE_KEYS = ("DN", "OD", "INCH", "LENGTH")
    THICKNESS_KEYS = ("MM", "SCHEDULE", "BWG", "INCH")
    ITEM_TYPES = {
        "SIZE_ITEMS": set(SIZE_KEYS),
        "THICKNESS_ITEMS": set(THICKNESS_KEYS),
    }

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
        self.api_key = str(self.config.get("api_key", "")).strip()
        self.timeout = float(self.config.get("timeout", 60))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", self.config.get("num_predict", 768)))
        self.max_workers = max(1, int(self.config.get("max_workers", 3)))
        self.reasoning_split = bool(self.config.get("reasoning_split", False))
        self.thickness_prompt_version = str(self.config.get("thickness_prompt_version", "v1")).strip().lower() or "v1"
        self.prompt_version = self.thickness_prompt_version
        self.size_length_prompt = get_size_length_system_prompt(debug=self.debug, version=self.prompt_version)
        self.thickness_prompt = get_thickness_system_prompt(debug=self.debug, version=self.prompt_version)
        self.pressure_prompt = get_pressure_system_prompt(debug=self.debug, version=self.prompt_version)
        self._last_usage: Dict[str, Any] = {}
        self.text_preprocessor = TextPreprocessor()
        if not self.model_name:
            raise RuntimeError("缺少配置: ner.structural_prompt.model_name")

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
        return {
            "SIZE": {key: [] for key in cls.SIZE_KEYS},
            "SIZE_ITEMS": [],
            "LENGTH": "",
            "THICKNESS": {key: [] for key in cls.THICKNESS_KEYS},
            "THICKNESS_ITEMS": [],
            "PRESSURE": "",
        }

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

    def _generate(self, system_prompt: str, user_content: str) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_content)
        if self.backend in {"openai_compatible", "openai"}:
            return self._generate_openai_compatible(system_prompt, user_content)
        raise RuntimeError(f"不支持的结构字段提示词后端: {self.backend}")

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

    def _generate_ollama(self, system_prompt: str, user_content: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stop": ["\n\n输入：", "\n\n【", "\n【"],
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._last_usage = self._extract_usage(payload)
        return str(payload.get("message", {}).get("content", "")).strip()

    def _generate_openai_compatible(self, system_prompt: str, user_content: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.api == "openai-completions":
            prompt = (
                f"{system_prompt}\n\n"
                f"{user_content}\n\n"
                "输出：\n"
            )
            resp = requests.post(
                f"{self.base_url}/completions",
                headers=headers,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": ["\n\n输入：", "\n\n【", "\n【"],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._last_usage = self._extract_usage(payload)
            choices = payload.get("choices") or []
            return str((choices[0] if choices else {}).get("text", "")).strip()

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": ["\n\n输入：", "\n\n【", "\n【"],
                **({"reasoning_split": True} if self.reasoning_split else {}),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._last_usage = self._extract_usage(payload)

        choices = payload.get("choices") or []
        message = (choices[0] if choices else {}).get("message") or {}
        return str(message.get("content", "")).strip()

    @staticmethod
    def _extract_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
        usage = payload.get("usage") or {}
        if usage:
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }

        # Ollama 常见字段
        prompt_eval = payload.get("prompt_eval_count")
        eval_count = payload.get("eval_count")
        if prompt_eval is not None or eval_count is not None:
            prompt_tokens = int(prompt_eval or 0)
            completion_tokens = int(eval_count or 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return {}

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
        result = cls.empty_result()

        size = parsed.get("SIZE")
        if isinstance(size, dict):
            for key in cls.SIZE_KEYS:
                result["SIZE"][key] = cls._normalize_list(size.get(key))
        result["SIZE_ITEMS"] = cls._normalize_items(parsed.get("SIZE_ITEMS"), cls.ITEM_TYPES["SIZE_ITEMS"])
        top_level_length = str(parsed.get("LENGTH", "") or "").strip()
        result["LENGTH"] = top_level_length
        if top_level_length:
            result["SIZE"]["LENGTH"] = cls._normalize_list([top_level_length])
            if ("LENGTH", top_level_length) not in {
                (str(item.get("type", "")).strip().upper(), str(item.get("value", "")).strip())
                for item in result["SIZE_ITEMS"]
            }:
                result["SIZE_ITEMS"].append({"type": "LENGTH", "value": top_level_length})
        if result["SIZE_ITEMS"]:
            result["SIZE"] = cls._group_items(result["SIZE_ITEMS"], cls.SIZE_KEYS)
            result["LENGTH"] = result["SIZE"]["LENGTH"][0] if result["SIZE"]["LENGTH"] else top_level_length

        thickness = parsed.get("THICKNESS")
        if isinstance(thickness, dict):
            for key in cls.THICKNESS_KEYS:
                result["THICKNESS"][key] = cls._normalize_list(thickness.get(key))
        result["THICKNESS_ITEMS"] = cls._normalize_items(
            parsed.get("THICKNESS_ITEMS"), cls.ITEM_TYPES["THICKNESS_ITEMS"]
        )
        if result["THICKNESS_ITEMS"]:
            result["THICKNESS"] = cls._group_items(result["THICKNESS_ITEMS"], cls.THICKNESS_KEYS)

        pressure = parsed.get("PRESSURE")
        result["PRESSURE"] = "" if pressure in (None, [], {}) else str(pressure).strip()
        return result

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        if value in (None, "", []):
            return []
        if not isinstance(value, list):
            value = [value]
        result: List[str] = []
        seen = set()
        for item in value:
            if item in (None, ""):
                continue
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @classmethod
    def _normalize_items(cls, value: Any, allowed_types: set[str]) -> List[Dict[str, str]]:
        if value in (None, "", []):
            return []
        if not isinstance(value, list):
            return []
        result: List[Dict[str, str]] = []
        seen = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().upper()
            item_value = cls._normalize_item_value(item_type, item.get("value", ""))
            if not item_type or not item_value or item_type not in allowed_types:
                continue
            key = (item_type, item_value)
            if key in seen:
                continue
            seen.add(key)
            result.append({"type": item_type, "value": item_value})
        return result

    @staticmethod
    def _normalize_item_value(item_type: str, raw_value: Any) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""

        kind = str(item_type or "").strip().upper()
        normalized = text.replace("”", "\"").replace("“", "\"").replace("″", "\"").strip()

        if kind == "DN":
            matched = re.fullmatch(r'(?i)DN\s*(\d+(?:\.\d+)?)', normalized)
            if matched:
                return matched.group(1)
            return normalized

        if kind == "OD":
            matched = re.fullmatch(r'(?i)(?:OD|[ΦφФфØø])\s*(\d+(?:\.\d+)?)', normalized)
            if matched:
                return matched.group(1)
            return normalized

        if kind == "INCH":
            normalized = re.sub(r'(?i)^NPS\s*', '', normalized)
            if normalized.endswith('"'):
                normalized = normalized[:-1].strip()
            return re.sub(r'\s+', '', normalized)

        if kind == "MM":
            matched = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*MM)?', normalized, flags=re.IGNORECASE)
            if matched:
                return matched.group(1)
            return normalized.upper()

        if kind == "BWG":
            matched = re.fullmatch(r'(?i)(?:BWG\s*)?(\d+(?:\.\d+)?)', normalized)
            if matched:
                return matched.group(1)
            return normalized.upper()

        return normalized.upper() if kind in {"SCHEDULE", "SERIES"} else normalized

    @staticmethod
    def _group_items(items: List[Dict[str, str]], keys: tuple[str, ...]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {key: [] for key in keys}
        for item in items:
            item_type = str(item.get("type", "")).strip().upper()
            item_value = str(item.get("value", "")).strip()
            if not item_type or not item_value or item_type not in grouped:
                continue
            grouped[item_type].append(item_value)
        return grouped
