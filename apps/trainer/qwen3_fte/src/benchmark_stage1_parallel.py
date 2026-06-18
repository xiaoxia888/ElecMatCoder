#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试阶段一四路字段抽取是否并行执行。

默认流程：
1. 使用当前平台配置完成类别路由
2. 按路由结果构造 TYPE / MATERIAL / STANDARD / STRUCTURAL 四路执行器
3. 分别跑一遍顺序执行和并行执行
4. 输出各路耗时、总耗时、并行收益

示例：
python apps/trainer/qwen3_fte/src/benchmark_stage1_parallel.py \
  --text "非标对焊法兰 WN-RF 300LB 064MC09-20 THK=3.5mm 16MnD,NB/T47009 DN50"
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.platform.server import (  # noqa: E402
    _build_qwen3_predictor_from_config,
    _merge_nested_dict,
    _resolve_qwen3_stage1_config,
)
from src.config import get_ner_config, reload_config  # noqa: E402
from src.llm_ner.router import build_category_router  # noqa: E402
from src.llm_ner.structural_prompt_extractor import StructuralPromptExtractor  # noqa: E402


def _normalize_model_override(value: Any) -> Dict[str, Any]:
    if value in (None, "", {}):
        return {}
    if isinstance(value, str):
        return {"model_name": value}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise TypeError(f"不支持的模型覆盖配置类型: {type(value).__name__}")


def _timed_call(name: str, func: Callable[[], Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    result = func()
    ended = time.perf_counter()
    return {
        "name": name,
        "duration_s": round(ended - started, 4),
        "result_preview": _preview(result),
    }


def _preview(result: Any) -> Any:
    if isinstance(result, dict):
        model_output = result.get("model_output")
        if isinstance(model_output, dict):
            decisions = model_output.get("decisions")
            if isinstance(decisions, dict):
                keys = [k for k in ("TYPE", "MATERIAL", "STANDARD", "SIZE", "THICKNESS", "PRESSURE") if k in decisions]
                return {k: decisions.get(k) for k in keys}
        if "SIZE" in result or "THICKNESS" in result or "PRESSURE" in result:
            return {
                "SIZE": result.get("SIZE"),
                "THICKNESS": result.get("THICKNESS"),
                "PRESSURE": result.get("PRESSURE"),
            }
    return str(result)[:200]


def _build_stage1_components() -> Dict[str, Any]:
    reload_config()
    ner_config = get_ner_config()
    qwen3_config = copy.deepcopy(ner_config.get("qwen3") or {})
    if not qwen3_config:
        raise RuntimeError("缺少配置: ner.qwen3")

    stage1_cfg = _resolve_qwen3_stage1_config(qwen3_config, ner_config)
    router_cfg = copy.deepcopy(stage1_cfg.get("router") or {})
    type_models = copy.deepcopy(stage1_cfg.get("type_models") or {})
    material_model = _normalize_model_override(stage1_cfg.get("material_model"))
    standard_model = _normalize_model_override(stage1_cfg.get("standard_model"))
    structural_cfg = copy.deepcopy(stage1_cfg.get("structural_prompt") or {})

    base_qwen3_config = copy.deepcopy(qwen3_config)
    base_qwen3_config.pop("router", None)
    base_qwen3_config.pop("category_models", None)
    base_qwen3_config.pop("stage1", None)

    return {
        "router": build_category_router(router_cfg, project_root=PROJECT_ROOT),
        "router_cfg": router_cfg,
        "base_qwen3_config": base_qwen3_config,
        "type_models": {k: _normalize_model_override(v) for k, v in type_models.items()},
        "material_model": material_model,
        "standard_model": standard_model,
        "structural_cfg": structural_cfg,
    }


def _select_type_override(selected_category: str, type_models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not selected_category:
        return {}
    return copy.deepcopy(type_models.get(selected_category) or {})


def _route_text(router: Any, text: str) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    route_info = router.route(text)
    ended = time.perf_counter()
    return route_info, round(ended - started, 4)


def _build_predictor(base_cfg: Dict[str, Any], override: Dict[str, Any]) -> Any:
    cfg = _merge_nested_dict(base_cfg, override or {})
    return _build_qwen3_predictor_from_config(cfg)


def _build_runners(text: str, components: Dict[str, Any], selected_category: str) -> Dict[str, Callable[[], Any]]:
    base_cfg = components["base_qwen3_config"]
    type_predictor = _build_predictor(base_cfg, _select_type_override(selected_category, components["type_models"]))
    material_predictor = _build_predictor(base_cfg, components["material_model"])
    standard_predictor = _build_predictor(base_cfg, components["standard_model"])

    structural_runner: Callable[[], Any]
    if components["structural_cfg"].get("enabled", False):
        structural_extractor = StructuralPromptExtractor(components["structural_cfg"])
        structural_runner = lambda: structural_extractor.extract(text)
    else:
        structural_runner = lambda: {"SIZE": {}, "THICKNESS": {}, "PRESSURE": ""}

    return {
        "TYPE": lambda: type_predictor.predict(text),
        "MATERIAL": lambda: material_predictor.predict(text),
        "STANDARD": lambda: standard_predictor.predict(text),
        "STRUCTURAL": structural_runner,
    }


def _run_sequential(runners: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    branches = {name: _timed_call(name, func) for name, func in runners.items()}
    total = time.perf_counter() - started
    return {
        "total_s": round(total, 4),
        "branches": branches,
    }


def _run_parallel(runners: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(runners)) as pool:
        futures = {name: pool.submit(_timed_call, name, func) for name, func in runners.items()}
        branches = {name: futures[name].result() for name in runners}
    total = time.perf_counter() - started
    return {
        "total_s": round(total, 4),
        "branches": branches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="测试阶段一并行调用耗时")
    parser.add_argument("--text", required=True, help="用于测试的一条描述文本")
    parser.add_argument("--mode", choices=["parallel", "sequential", "both"], default="both")
    parser.add_argument("--pretty", action="store_true", help="美化 JSON 输出")
    args = parser.parse_args()

    text = str(args.text or "").strip()
    if not text:
        raise SystemExit("text 不能为空")

    components = _build_stage1_components()
    route_info, route_time = _route_text(components["router"], text)
    selected_category = str(route_info.get("category") or "").strip()
    runners = _build_runners(text, components, selected_category)

    output: Dict[str, Any] = {
        "input": text,
        "route": {
            "category": selected_category,
            "confidence": route_info.get("confidence"),
            "source": route_info.get("source"),
            "reason": route_info.get("reason"),
            "duration_s": route_time,
        },
    }

    sequential = None
    parallel = None
    if args.mode in {"sequential", "both"}:
        sequential = _run_sequential(runners)
        output["sequential"] = sequential
    if args.mode in {"parallel", "both"}:
        parallel = _run_parallel(runners)
        output["parallel"] = parallel

    if sequential and parallel:
        seq_total = float(sequential["total_s"])
        par_total = float(parallel["total_s"])
        output["comparison"] = {
            "speedup_ratio": round(seq_total / par_total, 4) if par_total > 0 else None,
            "saved_s": round(seq_total - par_total, 4),
            "parallel_effective": par_total < seq_total,
        }

    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
