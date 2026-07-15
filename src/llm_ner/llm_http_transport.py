# -*- coding: utf-8 -*-
"""
Shared HTTP transport helpers for LLM backends.

This module only deals with request/response transport and token usage
extraction. Prompt orchestration and task-specific parsing stay in the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import requests


@dataclass
class LlmHttpResponse:
    text: str
    usage: Dict[str, int]
    payload: Dict[str, Any]


def extract_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    usage = payload.get("usage") or {}
    if usage:
        result = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        if "prompt_cache_hit_tokens" in usage:
            result["prompt_cache_hit_tokens"] = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        if "prompt_cache_miss_tokens" in usage:
            result["prompt_cache_miss_tokens"] = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        if "cache_creation_input_tokens" in usage:
            result["cache_creation_input_tokens"] = int(usage.get("cache_creation_input_tokens", 0) or 0)
        if "cache_read_input_tokens" in usage:
            result["cache_read_input_tokens"] = int(usage.get("cache_read_input_tokens", 0) or 0)
        return result

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


def call_ollama_chat(
    *,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    stop: Sequence[str] | None = None,
) -> LlmHttpResponse:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **({"stop": list(stop)} if stop else {}),
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return LlmHttpResponse(
        text=str(payload.get("message", {}).get("content", "")).strip(),
        usage=extract_usage(payload),
        payload=payload,
    )


def call_openai_compatible(
    *,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    api: str = "chat_completions",
    api_key: str = "",
    stop: Sequence[str] | None = None,
    reasoning_split: bool = False,
    extra_body: Dict[str, Any] | None = None,
) -> LlmHttpResponse:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if api == "openai-completions":
        prompt = f"{system_prompt}\n\n{user_content}\n\n输出：\n"
        resp = requests.post(
            f"{base_url.rstrip('/')}/completions",
            headers=headers,
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **({"stop": list(stop)} if stop else {}),
                **(extra_body or {}),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        choices = payload.get("choices") or []
        text = str((choices[0] if choices else {}).get("text", "")).strip()
        return LlmHttpResponse(text=text, usage=extract_usage(payload), payload=payload)

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **({"stop": list(stop)} if stop else {}),
            **({"reasoning_split": True} if reasoning_split else {}),
            **(extra_body or {}),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    choices = payload.get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    return LlmHttpResponse(
        text=str(message.get("content", "")).strip(),
        usage=extract_usage(payload),
        payload=payload,
    )


def call_ollama_generate(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: float,
    temperature: float,
    max_new_tokens: int,
    raw: bool = True,
    format_json: bool = False,
) -> LlmHttpResponse:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "raw": raw,
            "stream": False,
            **({"format": "json"} if format_json else {}),
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return LlmHttpResponse(
        text=str(payload.get("response", "")).strip(),
        usage=extract_usage(payload),
        payload=payload,
    )


def call_hf_lazy_service_predict(
    *,
    service_url: str,
    model_name: str,
    text: str,
    timeout: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> LlmHttpResponse:
    resp = requests.post(
        f"{service_url.rstrip('/')}/predict",
        json={
            "model": model_name,
            "text": text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    raw = payload.get("raw_response")
    parsed = payload.get("parsed_json")
    if isinstance(raw, str) and raw.strip():
        text_output = raw.strip()
    elif isinstance(parsed, dict):
        text_output = json.dumps(parsed, ensure_ascii=False)
    else:
        text_output = ""
    return LlmHttpResponse(
        text=text_output,
        usage=extract_usage(payload),
        payload=payload,
    )
