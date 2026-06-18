#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比同一条 prompt 在 HF(merged) 与 MLX 服务上的实际输出。

默认口径：
1. HF 直接加载 LlamaFactory merged 目录
2. MLX 走本地 /predict 服务
3. 双方都使用同一条 instruction、同一条用户输入、同一组解码参数

示例：
python apps/trainer/qwen3_fte/src/test_hf_mlx_consistency.py \
  --text "Eccentric Reducer, ASTM A 234 Gr.WPB + Internally PTFE Lined, SMLS, RF, 150#, MFR STD, ASME NM.1 / ASTM D4894 / ASTM D4895 Φ100x80"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_HF_MODEL = (
    "/Users/guoxi/Desktop/workspace/NJNCC/python_code/LlamaFactory/"
    "saves/qwen3-4b-base/lora/material-standard"
)
DEFAULT_MLX_URL = "http://127.0.0.1:8200/predict"
DEFAULT_MLX_MODEL = "material-standard"
DEFAULT_INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)


def build_prompt(instruction: str, text: str) -> str:
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_json_output(raw: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\\w*\\n?", "", cleaned)
        cleaned = re.sub(r"\\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def load_hf(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_hf(
    model,
    tokenizer,
    *,
    text: str,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt = build_prompt(instruction, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    bad_words_ids = []
    for marker in ("<think>", "</think>"):
        token_ids = tokenizer.encode(marker, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)

    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids or None,
        )
    elapsed = time.perf_counter() - started
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "elapsed_seconds": round(elapsed, 4),
        "prompt": prompt,
        "raw_response": raw,
        "parsed_json": parse_json_output(raw),
    }


def generate_mlx_via_http(
    *,
    url: str,
    model_name: str,
    text: str,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    body = {
        "model": model_name,
        "text": text,
        "instruction": instruction,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    started = time.perf_counter()
    response = requests.post(url, json=body, timeout=300)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    payload.setdefault("elapsed_seconds", round(elapsed, 4))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 HF merged 与 MLX 服务输出")
    parser.add_argument("--text", required=True, help="待测试描述")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HF merged 模型目录")
    parser.add_argument("--mlx-url", default=DEFAULT_MLX_URL, help="MLX 服务 /predict URL")
    parser.add_argument("--mlx-model", default=DEFAULT_MLX_MODEL, help="MLX 服务模型名")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="system instruction")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    text = str(args.text or "").strip()
    if not text:
        raise SystemExit("text 不能为空")

    print("加载 HF merged 模型...")
    model, tokenizer = load_hf(args.hf_model)

    print("执行 HF 推理...")
    hf_result = generate_hf(
        model,
        tokenizer,
        text=text,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("执行 MLX 服务推理...")
    mlx_result = generate_mlx_via_http(
        url=args.mlx_url,
        model_name=args.mlx_model,
        text=text,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output = {
        "input": {
            "text": text,
            "instruction": args.instruction,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "hf": hf_result,
        "mlx": {
            "elapsed_seconds": mlx_result.get("elapsed_seconds"),
            "prompt": mlx_result.get("prompt", ""),
            "raw_response": mlx_result.get("raw_response", ""),
            "parsed_json": mlx_result.get("parsed_json"),
            "json_parse_ok": mlx_result.get("json_parse_ok"),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
