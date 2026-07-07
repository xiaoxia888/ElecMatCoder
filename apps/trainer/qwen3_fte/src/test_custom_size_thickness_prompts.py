#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试自定义尺寸 / 壁厚提示词。

用途：
1. 直接读取单独的 prompt 文件
2. 分别测试尺寸与壁厚提示词效果
3. 可选走平台同款 TextPreprocessor
4. 输出原始响应、解析后的 JSON、token usage

示例：
python apps/trainer/qwen3_fte/src/test_custom_size_thickness_prompts.py \
  --backend openai_compatible \
  --base-url http://192.168.31.201:8085/v1 \
  --model-name MiniMax-M2.5-Q5_K_M-00001-of-00005.gguf \
  --api-key ollama-local \
  --text "HG/T20538;GB/T 14976;SH/T 3405 PIPE Lined L=1796mm CL150 RF 60.3x3.91(2.5) S30408/PP DN50"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_ner.llm_http_transport import call_ollama_chat, call_openai_compatible  # noqa: E402
from src.tokenizer_utils.preprocessor import TextPreprocessor  # noqa: E402


DEFAULT_SIZE_PROMPT = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "prompt"
    / "size_extraction_structured_rules_v1.txt"
)
DEFAULT_THICKNESS_PROMPT = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "prompt"
    / "thickness_extraction_structured_rules_v1.txt"
)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def try_parse_json(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def strip_think_blocks(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    return cleaned or raw


def extract_first_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    start = raw.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
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
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return ""


def build_user_content(text: str) -> str:
    return (
        f"输入：\n{text}\n\n"
        "只输出一个 JSON 对象。\n"
        "禁止输出思考过程。\n"
        "禁止输出 <think>、</think>、解释文字或代码块。\n"
        "输出到最后一个 } 后立即停止。"
    )


def infer_provider(provider: str, base_url: str, model_name: str) -> str:
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


def _parse_json_object_arg(raw: str, arg_name: str) -> dict[str, Any]:
    parsed, error = try_parse_json(raw)
    if error or not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} 不是合法 JSON 对象: {raw}")
    return parsed


def build_provider_extra_body(
    *,
    provider: str,
    base_url: str,
    model_name: str,
    thinking_mode: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    resolved = infer_provider(provider, base_url, model_name)
    body: dict[str, Any] = {}

    if resolved == "deepseek":
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if thinking_mode and thinking_mode != "auto":
            body["thinking"] = {"type": thinking_mode}
        return body

    if resolved == "minimax":
        if thinking_mode and thinking_mode != "auto":
            # 这是基于 MiniMax-M3 接口报错“ThinkingConfig 需要对象”的适配。
            body["thinking"] = {"type": thinking_mode}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return body

    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if thinking_mode and thinking_mode not in {"", "auto"}:
        body["thinking"] = thinking_mode
    return body


def call_model(
    *,
    provider: str,
    backend: str,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    api: str,
    api_key: str,
    thinking: str,
    thinking_json: str,
    thinking_mode: str,
    reasoning_effort: str,
    extra_body_json: str,
) -> tuple[str, dict[str, int]]:
    stop = ["\n\n输入：", "\n\n【", "\n【"]
    extra_body = build_provider_extra_body(
        provider=provider,
        base_url=base_url,
        model_name=model_name,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    if extra_body_json:
        extra_body.update(_parse_json_object_arg(extra_body_json, "--extra-body-json"))
    if thinking_json:
        extra_body["thinking"] = _parse_json_object_arg(thinking_json, "--thinking-json")
    elif thinking:
        extra_body["thinking"] = thinking
    if backend == "ollama":
        response = call_ollama_chat(
            base_url=base_url,
            model_name=model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.text, response.usage

    response = call_openai_compatible(
        base_url=base_url,
        model_name=model_name,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        api=api,
        api_key=api_key,
        stop=stop,
        reasoning_split=False,
        extra_body=extra_body,
    )
    return response.text, response.usage


def run_task(
    *,
    task_name: str,
    prompt_path: Path,
    text: str,
    provider: str,
    backend: str,
    base_url: str,
    model_name: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    api: str,
    api_key: str,
    thinking: str,
    thinking_json: str,
    thinking_mode: str,
    reasoning_effort: str,
    extra_body_json: str,
    show_prompt: bool,
) -> dict[str, Any]:
    system_prompt = load_prompt(prompt_path)
    user_content = build_user_content(text)
    result: dict[str, Any] = {
        "prompt_path": str(prompt_path),
        "raw_response": "",
        "cleaned_response": "",
        "json_candidate": "",
        "parsed_json": None,
        "parse_error": "",
        "usage": {},
    }
    try:
        raw_text, usage = call_model(
            provider=provider,
            backend=backend,
            base_url=base_url,
            model_name=model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            api=api,
            api_key=api_key,
            thinking=thinking,
            thinking_json=thinking_json,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            extra_body_json=extra_body_json,
        )
        cleaned_response = strip_think_blocks(raw_text)
        json_candidate = extract_first_json_object(cleaned_response)
        parsed_json, parse_error = try_parse_json(json_candidate or cleaned_response)
        result["raw_response"] = raw_text
        result["cleaned_response"] = cleaned_response
        result["json_candidate"] = json_candidate
        result["parsed_json"] = parsed_json
        result["parse_error"] = parse_error or ""
        result["usage"] = usage
        if (
            isinstance(usage, dict)
            and int(usage.get("completion_tokens", 0) or 0) >= max_tokens
            and not parsed_json
        ):
            result["warning"] = (
                f"completion_tokens 命中上限 {max_tokens}，输出大概率被截断；"
                "建议提高 --max-tokens，或进一步缩短提示词。"
            )
    except requests.HTTPError as exc:
        response = exc.response
        body = ""
        if response is not None:
            try:
                body = response.text
            except Exception:  # noqa: BLE001
                body = ""
        result["request_error"] = {
            "type": "HTTPError",
            "message": str(exc),
            "status_code": getattr(response, "status_code", None),
            "response_text": body,
        }
    except Exception as exc:  # noqa: BLE001
        result["request_error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
    if show_prompt:
        result["system_prompt"] = system_prompt
    return result


def summarize_cache_usage(usage: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}

    summary: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        if key in usage:
            summary[key] = usage.get(key)

    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    if hit or miss:
        total = hit + miss
        summary["prompt_cache_hit_ratio"] = round(hit / total, 4) if total else 0.0
        summary["prompt_cache_status"] = "hit" if hit > 0 else "miss"

    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        read_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
        created_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
        if read_tokens > 0:
            summary["context_cache_status"] = "hit"
        elif created_tokens > 0:
            summary["context_cache_status"] = "write"
        else:
            summary["context_cache_status"] = "miss"

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="测试自定义尺寸/壁厚提示词")
    parser.add_argument("--text", required=True, help="待测试描述")
    parser.add_argument(
        "--task",
        choices=["size", "thickness", "both"],
        default="both",
        help="测试尺寸、壁厚或两者",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "minimax", "deepseek", "generic"],
        default="auto",
        help="接口提供商；默认根据 base_url/model_name 自动判断",
    )
    parser.add_argument(
        "--backend",
        choices=["openai_compatible", "ollama"],
        default="openai_compatible",
        help="模型后端",
    )
    parser.add_argument("--base-url", required=True, help="服务地址")
    parser.add_argument("--model-name", required=True, help="模型名")
    parser.add_argument(
        "--api",
        choices=["chat_completions", "openai-completions"],
        default="chat_completions",
        help="仅 openai_compatible 后端使用",
    )
    parser.add_argument("--api-key", default="", help="仅 openai_compatible 后端使用")
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度")
    parser.add_argument("--max-tokens", type=int, default=1536, help="最大生成 token")
    parser.add_argument(
        "--thinking",
        choices=["enabled", "adaptive", "disabled", ""],
        default="",
        help="旧版透传参数；仅在接口确实接受 thinking 字符串时使用",
    )
    parser.add_argument(
        "--thinking-json",
        default="",
        help='透传给兼容接口的 thinking JSON 对象，例如 \'{"type":"disabled"}\'',
    )
    parser.add_argument(
        "--thinking-mode",
        choices=["auto", "enabled", "adaptive", "disabled"],
        default="auto",
        help="语义化思考模式，由脚本按 provider 自动转换成具体请求格式",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", ""],
        default="",
        help="推理强度参数；由脚本按 provider 自动透传",
    )
    parser.add_argument(
        "--extra-body-json",
        default="",
        help='透传给兼容接口的额外 JSON 对象，例如 \'{"thinking":{"type":"enabled"}}\'',
    )
    parser.add_argument(
        "--size-prompt",
        default=str(DEFAULT_SIZE_PROMPT),
        help="尺寸提示词路径",
    )
    parser.add_argument(
        "--thickness-prompt",
        default=str(DEFAULT_THICKNESS_PROMPT),
        help="壁厚提示词路径",
    )
    parser.add_argument(
        "--no-platform-preprocess",
        action="store_true",
        help="不走平台 TextPreprocessor，直接用原文",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="输出 system prompt 内容",
    )
    args = parser.parse_args()

    original_text = str(args.text or "").strip()
    if not original_text:
        raise SystemExit("text 不能为空")

    processed_text = original_text
    if not args.no_platform_preprocess:
        processed_text = TextPreprocessor().process(original_text)

    output: dict[str, Any] = {
        "config": {
            "task": args.task,
            "provider": infer_provider(args.provider, args.base_url, args.model_name),
            "backend": args.backend,
            "base_url": args.base_url,
            "model_name": args.model_name,
            "api": args.api,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking,
            "thinking_json": args.thinking_json,
            "thinking_mode": args.thinking_mode,
            "reasoning_effort": args.reasoning_effort,
            "extra_body_json": args.extra_body_json,
        },
        "input": {
            "original_text": original_text,
            "processed_text": processed_text,
        },
    }

    if args.task in {"size", "both"}:
        output["size"] = run_task(
            task_name="size",
            prompt_path=Path(args.size_prompt),
            text=processed_text,
            provider=args.provider,
            backend=args.backend,
            base_url=args.base_url,
            model_name=args.model_name,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api=args.api,
            api_key=args.api_key,
            thinking=args.thinking,
            thinking_json=args.thinking_json,
            thinking_mode=args.thinking_mode,
            reasoning_effort=args.reasoning_effort,
            extra_body_json=args.extra_body_json,
            show_prompt=args.show_prompt,
        )
        output["size"]["cache_summary"] = summarize_cache_usage(output["size"].get("usage", {}))

    if args.task in {"thickness", "both"}:
        output["thickness"] = run_task(
            task_name="thickness",
            prompt_path=Path(args.thickness_prompt),
            text=processed_text,
            provider=args.provider,
            backend=args.backend,
            base_url=args.base_url,
            model_name=args.model_name,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api=args.api,
            api_key=args.api_key,
            thinking=args.thinking,
            thinking_json=args.thinking_json,
            thinking_mode=args.thinking_mode,
            reasoning_effort=args.reasoning_effort,
            extra_body_json=args.extra_body_json,
            show_prompt=args.show_prompt,
        )
        output["thickness"]["cache_summary"] = summarize_cache_usage(output["thickness"].get("usage", {}))

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
