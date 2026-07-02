#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单条测试当前结构字段提示词链路。

用途：
1. 复用平台当前配置
2. 可选走平台同款预处理
3. 打印真正送入结构提示词模型前的文本
4. 打印模型原始返回与解析结果
5. 观察 SIZE_ITEMS / THICKNESS_ITEMS 顺序输出
6. 可切到 debug 模式，查看候选值、丢弃值和原因

示例：
python apps/trainer/qwen3_fte/src/test_structural_prompt_extract.py \
  --text "SPECTACLE BLANK CL300(PN50) RF NB/T 47008 20 ENR STD 40T018 DN150" \
  --debug
"""

from __future__ import annotations

import argparse
import json
import sys
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_ner_config, reload_config  # noqa: E402
from src.llm_ner.structural_prompt_extractor import StructuralPromptExtractor  # noqa: E402
from src.tokenizer_utils.preprocessor import TextPreprocessor  # noqa: E402


def _resolve_structural_cfg() -> dict:
    reload_config()
    ner_config = get_ner_config()
    qwen3_cfg = ner_config.get("qwen3", {}) or {}
    stage1_cfg = qwen3_cfg.get("stage1", {}) or {}
    structural_cfg = stage1_cfg.get("structural_prompt", {}) or {}
    if not structural_cfg.get("enabled", False):
        raise RuntimeError("当前配置未启用 structural_prompt")
    return structural_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="测试结构字段提示词抽取")
    parser.add_argument("--text", required=True, help="待测试描述")
    parser.add_argument(
        "--no-platform-preprocess",
        action="store_true",
        help="不走平台 TextPreprocessor，直接用原文",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="额外打印 system prompt",
    )
    parser.add_argument(
        "--force-api",
        choices=["chat_completions", "openai-completions"],
        help="强制指定一次请求方式",
    )
    parser.add_argument(
        "--compare-request-modes",
        action="store_true",
        help="同一描述同时测试 chat_completions 与 openai-completions",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试提示词，输出 candidates / dropped / final",
    )
    args = parser.parse_args()

    text = str(args.text or "").strip()
    if not text:
        raise SystemExit("text 不能为空")

    processed_text = text
    if not args.no_platform_preprocess:
        processed_text = TextPreprocessor().process(text)

    base_cfg = _resolve_structural_cfg()

    def run_once(api_mode: str | None) -> dict:
        cfg = copy.deepcopy(base_cfg)
        if api_mode:
            cfg["api"] = api_mode
        extractor = StructuralPromptExtractor(cfg, debug=args.debug)
        prompt_text = extractor._preprocess_text(processed_text)
        if args.debug:
            debug_result = extractor.debug_extract(processed_text)
            item = {
                "config": {
                    "backend": extractor.backend,
                    "model_name": extractor.model_name,
                    "base_url": extractor.base_url,
                    "api": extractor.api,
                    "temperature": extractor.temperature,
                    "max_tokens": extractor.max_tokens,
                    "debug": True,
                },
                "input": {
                    "original_text": text,
                    "processed_text": processed_text,
                    "prompt_text": prompt_text,
                },
                "raw_response": debug_result.get("_raw", ""),
                "parsed_json": debug_result.get("_parsed"),
                "usage": debug_result.get("_usage", {}),
                "debug": {
                    "trace": debug_result.get("trace", []),
                    "field_diagnostics": debug_result.get("field_diagnostics", {}),
                    "final": debug_result.get("final", extractor.empty_result()),
                },
            }
        else:
            normalized = extractor.extract(processed_text)
            item = {
                "config": {
                    "backend": extractor.backend,
                    "model_name": extractor.model_name,
                    "base_url": extractor.base_url,
                    "api": extractor.api,
                    "temperature": extractor.temperature,
                    "max_tokens": extractor.max_tokens,
                    "debug": False,
                },
                "input": {
                    "original_text": text,
                    "processed_text": processed_text,
                    "prompt_text": prompt_text,
                },
                "raw_response": normalized.get("_raw", ""),
                "parsed_json": None,
                "usage": normalized.get("_usage", {}),
                "normalized": normalized,
            }
        if args.show_prompt:
            item["system_prompt"] = {
                "size_length": extractor.size_length_prompt,
                "thickness": extractor.thickness_prompt,
                "pressure": extractor.pressure_prompt,
            }
        return item

    if args.compare_request_modes:
        output = {
            "chat_completions": run_once("chat_completions"),
            "openai_completions": run_once("openai-completions"),
        }
    else:
        output = run_once(args.force_api)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
