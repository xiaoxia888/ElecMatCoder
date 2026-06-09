# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests


DEFAULT_INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)


def _call_predict(
    *,
    service_url: str,
    model: str,
    text: str,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    resp = requests.post(
        f"{service_url.rstrip('/')}/predict",
        json={
            "model": model,
            "text": text,
            "instruction": instruction,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    payload = resp.json()
    return {
        "model": model,
        "wall_seconds": round(elapsed, 4),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "json_parse_ok": payload.get("json_parse_ok"),
        "raw_preview": str(payload.get("raw_response", ""))[:120],
    }


def _run_sequential(args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for model in args.models:
        results.append(
            _call_predict(
                service_url=args.service_url,
                model=model,
                text=args.text,
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
        )
    return results


def _run_parallel(args: argparse.Namespace) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        futures = [
            pool.submit(
                _call_predict,
                service_url=args.service_url,
                model=model,
                text=args.text,
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
            for model in args.models
        ]
        return [future.result() for future in futures]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 mlx_service 的并发/串行行为")
    parser.add_argument("--service-url", default="http://127.0.0.1:8200")
    parser.add_argument("--models", nargs="+", required=True, help="例如: type material-standard material-standard")
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    seq_started = time.perf_counter()
    sequential = _run_sequential(args)
    seq_total = time.perf_counter() - seq_started

    par_started = time.perf_counter()
    parallel = _run_parallel(args)
    par_total = time.perf_counter() - par_started

    print(
        json.dumps(
            {
                "service_url": args.service_url,
                "models": args.models,
                "sequential_total_seconds": round(seq_total, 4),
                "parallel_total_seconds": round(par_total, 4),
                "sequential": sequential,
                "parallel": parallel,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
