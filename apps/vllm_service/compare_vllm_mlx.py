#!/usr/bin/env python3
"""Compare deterministic type-model responses from vLLM and MLX services."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VLLM_URL = (
    "https://f98b8bda606b412895aac9fb355f023e.region2.waas.aigate.cc"
)
DEFAULT_MLX_URL = "http://192.168.31.201:8200"
DEFAULT_PROMPT = (
    PROJECT_ROOT
    / "apps"
    / "trainer"
    / "qwen3_fte"
    / "prompt"
    / "type_extraction_sft_instruction_v1.txt"
)
DEFAULT_TEXT = '法兰偏心异径管, PTFElined GB/T 8163-20, RF, PN16, HG/T20538, SMLS , DN50 x DN40, 3.5mm x 3.5 mm'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="向 vLLM 和 MLX /predict 发送相同请求并生成差异报告",
    )
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    parser.add_argument("--mlx-url", default=DEFAULT_MLX_URL)
    parser.add_argument("--model", default="type")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path, help="从文本文件读取测试描述")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="生成上限；默认256，避免长提示词与1024上下文模型合计超限",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--insecure", action="store_true", help="关闭 HTTPS 证书校验")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vllm_mlx_comparison.json"),
    )
    return parser.parse_args()


def normalize_url(value: str) -> str:
    return value.rstrip("/")


def canonical_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def response_signature(result: dict[str, Any]) -> str:
    parsed = result.get("parsed_json")
    if parsed is not None:
        return f"json:{canonical_json(parsed)}"
    return f"raw:{str(result.get('raw_response') or '').strip()}"


def post_predict(
    client: httpx.Client,
    *,
    base_url: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(f"{normalize_url(base_url)}/predict", json=body)
        elapsed = round(time.perf_counter() - started, 4)
        payload: Any
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"raw_http_body": response.text}
        if response.is_error:
            return {
                "ok": False,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "error": payload,
            }
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "error": f"响应不是JSON对象: {type(payload).__name__}",
            }
        payload["ok"] = True
        payload["status_code"] = response.status_code
        payload["client_elapsed_seconds"] = elapsed
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def get_probe(client: httpx.Client, base_url: str, path: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.get(f"{normalize_url(base_url)}{path}")
        try:
            payload: Any = response.json()
        except json.JSONDecodeError:
            payload = response.text
        return {
            "ok": not response.is_error,
            "status_code": response.status_code,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_endpoint(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item.get("ok")]
    signatures = [response_signature(item) for item in successful]
    instructions = {
        str(item.get("instruction") or "")
        for item in successful
    }
    prompts = {
        canonical_json(item.get("prompt"))
        if not isinstance(item.get("prompt"), str)
        else str(item.get("prompt"))
        for item in successful
    }
    return {
        "requests": len(results),
        "successes": len(successful),
        "failures": len(results) - len(successful),
        "stable_across_repeats": len(set(signatures)) <= 1 and bool(successful),
        "unique_response_count": len(set(signatures)),
        "unique_instruction_count": len(instructions),
        "unique_prompt_count": len(prompts),
    }


def summarize_phase(
    vllm_results: list[dict[str, Any]],
    mlx_results: list[dict[str, Any]],
) -> dict[str, Any]:
    vllm_ok = [item for item in vllm_results if item.get("ok")]
    mlx_ok = [item for item in mlx_results if item.get("ok")]
    paired = min(len(vllm_ok), len(mlx_ok))
    pair_agreement = [
        response_signature(vllm_ok[index]) == response_signature(mlx_ok[index])
        for index in range(paired)
    ]
    vllm_instruction = str(vllm_ok[0].get("instruction") or "") if vllm_ok else ""
    mlx_instruction = str(mlx_ok[0].get("instruction") or "") if mlx_ok else ""
    return {
        "vllm": summarize_endpoint(vllm_results),
        "mlx": summarize_endpoint(mlx_results),
        "paired_successes": paired,
        "paired_equal_responses": sum(pair_agreement),
        "all_paired_responses_equal": bool(pair_agreement) and all(pair_agreement),
        "instructions_equal": bool(vllm_ok and mlx_ok)
        and vllm_instruction == mlx_instruction,
        "vllm_instruction_length": len(vllm_instruction),
        "mlx_instruction_length": len(mlx_instruction),
    }


def diagnosis(configured: dict[str, Any], forced: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    if not configured.get("instructions_equal"):
        messages.append("两端配置中的 instruction 不同。")
    if configured.get("all_paired_responses_equal"):
        messages.append("两端使用各自配置时输出已经一致。")
        return messages
    if forced.get("all_paired_responses_equal"):
        if not configured.get("instructions_equal"):
            messages.append("强制相同 instruction 后输出一致，差异来自服务端提示词配置。")
        else:
            messages.append("相同 instruction 和生成参数下输出一致。")
        return messages
    if not forced.get("paired_successes"):
        messages.append(
            "强制相同 instruction 阶段没有成功配对，当前结果不能用于判断推理后端差异。"
        )
        return messages
    if (
        forced["vllm"].get("successes")
        and not forced["vllm"].get("stable_across_repeats")
    ):
        messages.append("vLLM 在 temperature=0 下重复输出不稳定。")
    if (
        forced["mlx"].get("successes")
        and not forced["mlx"].get("stable_across_repeats")
    ):
        messages.append("MLX 在 temperature=0 下重复输出不稳定。")
    if (
        forced["vllm"].get("stable_across_repeats")
        and forced["mlx"].get("stable_across_repeats")
        and not forced.get("all_paired_responses_equal")
    ):
        messages.append(
            "相同 instruction 下两端各自稳定但彼此不同，继续检查 chat template、"
            "MLX量化/权重转换以及首token logits。"
        )
    return messages


def run_phase(
    client: httpx.Client,
    *,
    vllm_url: str,
    mlx_url: str,
    request_body: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    vllm_results: list[dict[str, Any]] = []
    mlx_results: list[dict[str, Any]] = []
    for _ in range(repeats):
        vllm_results.append(
            post_predict(client, base_url=vllm_url, body=request_body)
        )
        mlx_results.append(
            post_predict(client, base_url=mlx_url, body=request_body)
        )
    return {
        "request": request_body,
        "summary": summarize_phase(vllm_results, mlx_results),
        "vllm_results": vllm_results,
        "mlx_results": mlx_results,
    }


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats 必须大于0")

    text = (
        args.text_file.expanduser().read_text(encoding="utf-8").strip()
        if args.text_file
        else args.text.strip()
    )
    instruction = args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
    base_request = {
        "model": args.model,
        "text": text,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    with httpx.Client(timeout=args.timeout, verify=not args.insecure) as client:
        probes = {
            "vllm": {
                "health": get_probe(client, args.vllm_url, "/health"),
                "models": get_probe(client, args.vllm_url, "/models"),
            },
            "mlx": {
                "health": get_probe(client, args.mlx_url, "/health"),
                "models": get_probe(client, args.mlx_url, "/models"),
            },
        }
        configured = run_phase(
            client,
            vllm_url=args.vllm_url,
            mlx_url=args.mlx_url,
            request_body=base_request,
            repeats=args.repeats,
        )
        forced_request = dict(base_request)
        forced_request["instruction"] = instruction
        forced = run_phase(
            client,
            vllm_url=args.vllm_url,
            mlx_url=args.mlx_url,
            request_body=forced_request,
            repeats=args.repeats,
        )

    report = {
        "vllm_url": normalize_url(args.vllm_url),
        "mlx_url": normalize_url(args.mlx_url),
        "model": args.model,
        "text": text,
        "parameters": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repeats": args.repeats,
        },
        "prompt_file": str(args.prompt_file.expanduser().resolve()),
        "probes": probes,
        "configured_instruction_phase": configured,
        "forced_same_instruction_phase": forced,
        "diagnosis": diagnosis(configured["summary"], forced["summary"]),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "configured": configured["summary"],
                "forced_same_instruction": forced["summary"],
                "diagnosis": report["diagnosis"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
