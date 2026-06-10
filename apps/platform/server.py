# -*- coding: utf-8 -*-
"""
材料智能处理平台 - 统一后端服务
整合标注和编码功能
"""

import os
import sys
import logging
import yaml
import time
import uuid
from collections import deque
from typing import List, Optional, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json
import asyncio
import copy

# 导入配置模块
from src.config import get_platform_config, get_tokenizer_config, get_semantic_config, get_ner_config

# 导入分词模块
from src.tokenizer_utils.llm_tokenizer import LLMTokenizer
from src.tokenizer_utils.preprocessor import TextPreprocessor

# 导入NER预测器
from src.bert_ner.predictor import PipePredictor


# 导入编码模块
from src.encoder.pipe_encoder import PipeEncoder, get_pipe_encoder
from src.encoder.semantic_matcher import get_semantic_matcher
from src.material_description_splitter.platform_integration import (
    build_base_difficulty,
    finalize_batch_difficulty,
)
from src.material_description_splitter.second_pass import PlatformSecondPassRunner

# 导入第三方集成模块
from src.integrations import get_h3yun_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

H3YUN_REVIEW_TASK_SCHEMA_CODE = "D148357c862f0c8cdfa41418c55cef288f8d83c"
H3YUN_REVIEW_TASK_SUBTABLE_CODE = "D148357F17c2e0548b94497f873300934ea06164"
H3YUN_REVIEW_TASK_APP_CODE = "D148357CLDGGL"

# ============================================================
# FastAPI应用
# ============================================================

app = FastAPI(
    title="材料智能处理平台",
    description="提供材料标注和编码功能",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    platform_config = get_platform_config()
    server_config = platform_config.get("server", {})
    
    # 根据配置决定是否预加载模型
    if server_config.get("preload_models", False):
        logger.info("预加载模型已启用，正在加载...")
        
        # 预加载语义匹配器（包含 SentenceTransformer 模型）
        logger.info("加载语义匹配模型...")
        get_semantic_matcher()
        
        # 预加载管道编码器
        logger.info("加载管道编码器...")
        get_pipe_encoder()
        
        logger.info("模型预加载完成！")
    else:
        logger.info("预加载模型已禁用，将在首次使用时加载")


# ============================================================
# 全局实例
# ============================================================

_tokenizers: Dict[str, LLMTokenizer] = {}
_preprocessor = TextPreprocessor()
_ner_predictor = None
_ner_model_type: str = "bert"
_pipe_router = None
_structural_prompt_extractor = None
_second_pass_runner = PlatformSecondPassRunner()
_batch_jobs: Dict[str, Dict[str, Any]] = {}
_batch_job_queue: deque[str] = deque()
_batch_job_active_id: Optional[str] = None
_batch_job_scheduler_task: Optional[asyncio.Task] = None
_batch_job_lock = asyncio.Lock()

_BATCH_JOB_KEEP_SECONDS = 60.0
_BATCH_JOB_TERMINAL_STATUSES = {"finished", "cancelled", "failed"}
_BATCH_JOB_ACTIVE_STATUSES = {"queued", "running", "cancelling"}


def _merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _utc_ts() -> float:
    return time.time()


def _batch_job_public(job: Dict[str, Any], *, include_results: bool = True) -> Dict[str, Any]:
    queue_position = 0
    if job.get("status") == "queued":
        try:
            queue_position = list(_batch_job_queue).index(job["job_id"]) + 1
        except ValueError:
            queue_position = 0
    result: Dict[str, Any] = {
        "job_id": job["job_id"],
        "status": job.get("status", ""),
        "total": int(job.get("total", 0) or 0),
        "processed": int(job.get("processed", 0) or 0),
        "success_count": int(job.get("success_count", 0) or 0),
        "review_count": int(job.get("review_count", 0) or 0),
        "threshold": job.get("threshold"),
        "queue_position": queue_position,
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "updated_at": job.get("updated_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "error": job.get("error", ""),
        "items": copy.deepcopy(job.get("items_meta", [])),
    }
    if include_results:
        result["results"] = copy.deepcopy(job.get("results", {}))
    return result


def _build_batch_item_trace_base(
    *,
    text: str,
    processed_text: str,
    route_info: Dict[str, Any],
    stage1_output: Dict[str, Any],
    stage1_raw_response: str,
    structural_prompt_output: Any,
    structural_prompt_raw_response: str,
) -> Dict[str, Any]:
    return {
        "original_text": text,
        "processed_text": processed_text,
        "route_info": copy.deepcopy(route_info),
        "stage1_output": copy.deepcopy(stage1_output),
        "stage1_raw_response": stage1_raw_response,
        "structural_prompt_output": copy.deepcopy(structural_prompt_output),
        "structural_prompt_raw_response": structural_prompt_raw_response,
        "difficulty_stage1_before_project": None,
        "second_pass_before_project": None,
        "difficulty_stage1_after_project": None,
        "second_pass_after_project": None,
        "status": "pending",
        "skip_reason": "",
        "error": "",
    }


def _build_batch_item_trace_summary(
    converted: Optional[Dict[str, Any]],
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    converted = converted or {}
    trace = trace or {}
    route_info = trace.get("route_info") or converted.get("route_info") or {}
    stage1_output = trace.get("stage1_output") or converted.get("stage1_output") or {}
    extract_confidence_v2 = converted.get("extract_confidence_v2") or {}

    def _source_label(source: str) -> str:
        source = str(source or "").strip()
        if source == "finetuned_model":
            return "微调模型"
        if source == "rule_extraction":
            return "规则/正则"
        if source == "prompt_extraction":
            return "提示词模型"
        return source or "未知"

    def _field_source(field: str) -> str:
        info = extract_confidence_v2.get(field)
        if isinstance(info, dict):
            return str(info.get("source", "") or "")
        return ""

    def _stringify_stage1_value(field: str, value: Any) -> str:
        if value is None:
            return ""
        if field == "TYPE" and isinstance(value, dict):
            body = str(value.get("BODY", "") or "").strip()
            return body
        if field == "MATERIAL" and isinstance(value, list):
            values = []
            for item in value:
                if isinstance(item, dict):
                    raw = str(item.get("VALUE", "") or "").strip()
                    if raw:
                        values.append(raw)
            return "; ".join(values)
        if field == "STANDARD" and isinstance(value, list):
            values = []
            for item in value:
                if isinstance(item, dict):
                    raw = str(item.get("BODY", "") or "").strip()
                    if raw:
                        values.append(raw)
            return "; ".join(values)
        if isinstance(value, dict):
            parts = []
            for key in ("DN", "OD", "INCH", "MM", "SCHEDULE", "SERIES", "BWG"):
                raw = value.get(key)
                if isinstance(raw, list):
                    raw = " | ".join(str(v).strip() for v in raw if str(v).strip())
                elif raw is None:
                    raw = ""
                raw = str(raw or "").strip()
                if raw:
                    parts.append(f"{key}: {raw}")
            return "; ".join(parts)
        if isinstance(value, list):
            return "; ".join(str(v).strip() for v in value if str(v).strip())
        return str(value or "").strip()

    def _stage1_field(field: str, value: Any) -> Dict[str, Any]:
        source = _field_source(field)
        return {
            "value": _stringify_stage1_value(field, value),
            "source": source,
            "source_label": _source_label(source),
        }

    return {
        "route": {
            "category": route_info.get("category", ""),
            "confidence": route_info.get("confidence"),
            "reason": route_info.get("reason", ""),
        },
        "stage1": {
            "type": _stage1_field("TYPE", stage1_output.get("TYPE")),
            "size": _stage1_field("SIZE", stage1_output.get("SIZE")),
            "thickness": _stage1_field("THICKNESS", stage1_output.get("THICKNESS")),
            "pressure": _stage1_field("PRESSURE", stage1_output.get("PRESSURE")),
            "material": _stage1_field("MATERIAL", stage1_output.get("MATERIAL")),
            "standard": _stage1_field("STANDARD", stage1_output.get("STANDARD")),
        },
        "split": {
            "difficulty_before_project": (trace.get("difficulty_stage1_before_project") or {}).get("difficulty", ""),
            "difficulty_before_project_reason": (trace.get("difficulty_stage1_before_project") or {}).get("reason", ""),
            "second_pass_before_project": (trace.get("second_pass_before_project") or {}).get("final_level", ""),
            "second_pass_before_project_reason": (trace.get("second_pass_before_project") or {}).get("reason", ""),
            "difficulty_after_project": (trace.get("difficulty_stage1_after_project") or {}).get("difficulty", ""),
            "difficulty_after_project_reason": (trace.get("difficulty_stage1_after_project") or {}).get("reason", ""),
            "second_pass_after_project": (trace.get("second_pass_after_project") or {}).get("final_level", ""),
            "second_pass_after_project_reason": (trace.get("second_pass_after_project") or {}).get("reason", ""),
        },
        "skip": {
            "skipped": bool(converted.get("skipped_encoding")),
            "reason": trace.get("skip_reason", "") or converted.get("skip_reason", ""),
        },
        "status": trace.get("status", ""),
    }


async def _batch_job_emit(job: Dict[str, Any], event: Dict[str, Any]) -> None:
    event_payload = copy.deepcopy(event)
    event_payload["job_id"] = job["job_id"]
    event_payload["job_status"] = job.get("status", "")
    subscribers = list(job.get("subscribers", []))
    for queue in subscribers:
        try:
            queue.put_nowait(copy.deepcopy(event_payload))
        except asyncio.QueueFull:
            continue


async def _batch_job_cleanup_locked() -> None:
    now = _utc_ts()
    remove_ids: list[str] = []
    for job_id, job in list(_batch_jobs.items()):
        status = str(job.get("status", "") or "")
        finished_at = float(job.get("finished_at") or 0.0)
        if status in _BATCH_JOB_TERMINAL_STATUSES and finished_at and (now - finished_at) >= _BATCH_JOB_KEEP_SECONDS:
            remove_ids.append(job_id)
    for job_id in remove_ids:
        _batch_jobs.pop(job_id, None)
        try:
            _batch_job_queue.remove(job_id)
        except ValueError:
            pass


async def _batch_job_get(job_id: str) -> Optional[Dict[str, Any]]:
    async with _batch_job_lock:
        await _batch_job_cleanup_locked()
        return _batch_jobs.get(job_id)


async def _batch_job_mark_finished(job: Dict[str, Any], status: str, error: str = "") -> None:
    job["status"] = status
    job["updated_at"] = _utc_ts()
    job["finished_at"] = _utc_ts()
    job["error"] = str(error or "")


async def _run_batch_job(job_id: str) -> None:
    async with _batch_job_lock:
        job = _batch_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = _utc_ts()
        job["updated_at"] = _utc_ts()
        job["error"] = ""
    await _batch_job_emit(job, {"type": "start", "snapshot": _batch_job_public(job)})

    encoder = get_pipe_encoder()
    predictor = get_ner_predictor()
    items = list(job.get("items", []))
    total = len(items)
    threshold = encoder.get_threshold()
    job["threshold"] = threshold
    max_concurrent = int(job.get("max_concurrent") or get_batch_max_concurrent())
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    index_lock = asyncio.Lock()
    next_index = 0
    converted_results: list[Optional[Dict[str, Any]]] = [None] * total

    async def process_one(order_index: int, item: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        text = item.get("text", "")
        preprocess = bool(item.get("preprocess", True))
        processed_text = _preprocessor.process(text) if preprocess else text

        async with semaphore:
            predict_result = await asyncio.to_thread(predictor.predict, processed_text)
        if predict_result.get("structural_prompt_output") is None:
            _apply_structural_prompt_override(predict_result, processed_text)

        route_info = _decorate_predict_route_info(predict_result.get("route_info"))
        stage1_output = dict(predict_result.get("model_output", {}) or {})
        stage1_output["_STRUCTURAL_PROMPT"] = predict_result.get("structural_prompt_output")
        stage1_raw_response = "".join([
            predict_result.get("model_raw_response", "") or "",
            f"\n\n[STRUCTURAL_PROMPT_RAW]\n{predict_result.get('structural_prompt_raw_response')}"
            if predict_result.get("structural_prompt_raw_response")
            else "",
        ])
        trace = _build_batch_item_trace_base(
            text=text,
            processed_text=processed_text,
            route_info=route_info,
            stage1_output=stage1_output,
            stage1_raw_response=stage1_raw_response,
            structural_prompt_output=predict_result.get("structural_prompt_output"),
            structural_prompt_raw_response=predict_result.get("structural_prompt_raw_response", ""),
        )

        if route_info.get("encoding_enabled") is False:
            converted: Dict[str, Any] = {
                "original_text": text,
                "processed_text": processed_text,
                "final_code": "",
                "success": True,
                "need_review": False,
                "skipped_encoding": True,
                "skip_reason": route_info.get("skip_encoding_reason", "") or "",
                "errors": [],
                "fields": {},
                "route_info": route_info,
                "stage1_output": stage1_output,
                "stage1_raw_response": stage1_raw_response,
                "structural_prompt_output": predict_result.get("structural_prompt_output"),
                "structural_prompt_raw_response": predict_result.get("structural_prompt_raw_response", ""),
            }
            trace["status"] = "skipped"
            trace["skip_reason"] = converted.get("skip_reason", "")
            return converted, trace

        entities = _build_pipe_entities_for_encode(predict_result)
        extract_confidence = predict_result.get("extract_confidence", {}) or {}
        extract_confidence_v2 = predict_result.get("extract_confidence_v2", {}) or {}

        async with semaphore:
            result = await asyncio.to_thread(
                encoder.encode,
                entities,
                text,
                extract_confidence,
                extract_confidence_v2,
            )

        converted = _convert_pipe_result(result)
        converted["processed_text"] = processed_text
        converted["extract_confidence_v2"] = converted.get("extract_confidence_v2") or extract_confidence_v2
        converted["route_info"] = route_info
        converted["stage1_output"] = stage1_output
        converted["stage1_raw_response"] = stage1_raw_response
        converted["structural_prompt_output"] = predict_result.get("structural_prompt_output")
        converted["structural_prompt_raw_response"] = predict_result.get("structural_prompt_raw_response", "")
        converted = _attach_base_difficulty(converted, str(item.get("project_name", "") or "").strip())
        converted = _attach_second_pass(converted)
        trace["status"] = "processed"
        trace["difficulty_stage1_before_project"] = copy.deepcopy(converted.get("difficulty_split"))
        trace["second_pass_before_project"] = copy.deepcopy(converted.get("second_pass"))
        return converted, trace

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with _batch_job_lock:
                current_job = _batch_jobs.get(job_id)
                if not current_job:
                    return
                if current_job.get("cancel_requested"):
                    current_job["status"] = "cancelling"
                    current_job["updated_at"] = _utc_ts()
                    return
            async with index_lock:
                if next_index >= total:
                    return
                order_index = next_index
                next_index += 1
            item = items[order_index]
            try:
                converted, trace = await process_one(order_index, item)
                converted_results[order_index] = converted
                async with _batch_job_lock:
                    current_job = _batch_jobs.get(job_id)
                    if not current_job:
                        return
                    current_job["processed"] = int(current_job.get("processed", 0) or 0) + 1
                    if converted and converted.get("success"):
                        current_job["success_count"] = int(current_job.get("success_count", 0) or 0) + 1
                    if converted and converted.get("need_review"):
                        current_job["review_count"] = int(current_job.get("review_count", 0) or 0) + 1
                    current_job["results"][str(order_index)] = copy.deepcopy(converted)
                    current_job["item_traces"][str(order_index)] = copy.deepcopy(trace)
                    current_job["updated_at"] = _utc_ts()
                    snapshot = _batch_job_public(current_job)
                await _batch_job_emit(job, {
                    "type": "progress",
                    "index": int(item.get("client_index", order_index)),
                    "order_index": order_index,
                    "result": converted,
                    "trace_summary": _build_batch_item_trace_summary(converted, trace),
                    "snapshot": snapshot,
                })
            except Exception as exc:
                logger.exception("批量编码任务处理失败: job=%s index=%s", job_id, order_index)
                failed_result = {
                    "original_text": item.get("text", ""),
                    "final_code": "",
                    "success": False,
                    "need_review": True,
                    "errors": [str(exc)],
                    "fields": {},
                }
                failed_trace = {
                    "original_text": item.get("text", ""),
                    "processed_text": "",
                    "route_info": None,
                    "stage1_output": None,
                    "stage1_raw_response": "",
                    "structural_prompt_output": None,
                    "structural_prompt_raw_response": "",
                    "difficulty_stage1_before_project": None,
                    "second_pass_before_project": None,
                    "difficulty_stage1_after_project": None,
                    "second_pass_after_project": None,
                    "status": "failed",
                    "skip_reason": "",
                    "error": str(exc),
                }
                converted_results[order_index] = failed_result
                async with _batch_job_lock:
                    current_job = _batch_jobs.get(job_id)
                    if not current_job:
                        return
                    current_job["processed"] = int(current_job.get("processed", 0) or 0) + 1
                    current_job["review_count"] = int(current_job.get("review_count", 0) or 0) + 1
                    current_job["results"][str(order_index)] = copy.deepcopy(failed_result)
                    current_job["item_traces"][str(order_index)] = failed_trace
                    current_job["updated_at"] = _utc_ts()
                    snapshot = _batch_job_public(current_job)
                await _batch_job_emit(job, {
                    "type": "progress",
                    "index": int(item.get("client_index", order_index)),
                    "order_index": order_index,
                    "result": failed_result,
                    "trace_summary": _build_batch_item_trace_summary(failed_result, failed_trace),
                    "snapshot": snapshot,
                })

    try:
        workers = [asyncio.create_task(worker()) for _ in range(max(1, max_concurrent))]
        await asyncio.gather(*workers)

        async with _batch_job_lock:
            current_job = _batch_jobs.get(job_id)
            if not current_job:
                return
            cancel_requested = bool(current_job.get("cancel_requested"))

        if cancel_requested:
            async with _batch_job_lock:
                current_job = _batch_jobs.get(job_id)
                if not current_job:
                    return
                await _batch_job_mark_finished(current_job, "cancelled")
                snapshot = _batch_job_public(current_job)
            await _batch_job_emit(job, {"type": "cancelled", "snapshot": snapshot})
            return

        finalized = finalize_batch_difficulty(
            [
                {
                    "text": (converted or {}).get("original_text", "") if isinstance(converted, dict) else "",
                    "project_name": str(item.get("project_name", "") or "").strip(),
                    "type_code": _extract_code_from_field(converted or {}, "TYPE") if isinstance(converted, dict) else "",
                    "material_code": _extract_code_from_field(converted or {}, "MATERIAL") if isinstance(converted, dict) else "",
                    "standard_code": _extract_code_from_field(converted or {}, "STANDARD") if isinstance(converted, dict) else "",
                    "standard_codes": _extract_standard_codes_from_field(converted or {}) if isinstance(converted, dict) else [],
                    "base_difficulty": (converted or {}).get("difficulty_split") if isinstance(converted, dict) else None,
                }
                for item, converted in zip(items, converted_results)
            ]
        )

        for order_index, (item, converted, difficulty) in enumerate(zip(items, converted_results, finalized)):
            if not isinstance(converted, dict) or converted.get("skipped_encoding"):
                continue
            converted["difficulty_split"] = difficulty
            _attach_second_pass(converted)
            async with _batch_job_lock:
                current_job = _batch_jobs.get(job_id)
                if not current_job:
                    return
                current_job["results"][str(order_index)] = copy.deepcopy(converted)
                trace = current_job["item_traces"].setdefault(str(order_index), {})
                trace["difficulty_stage1_after_project"] = copy.deepcopy(converted.get("difficulty_split"))
                trace["second_pass_after_project"] = copy.deepcopy(converted.get("second_pass"))
                trace["status"] = "finalized"
                current_job["updated_at"] = _utc_ts()
                snapshot = _batch_job_public(current_job)
            await _batch_job_emit(job, {
                "type": "finalize",
                "index": int(item.get("client_index", order_index)),
                "order_index": order_index,
                "result": converted,
                "trace_summary": _build_batch_item_trace_summary(converted, trace),
                "snapshot": snapshot,
            })

        async with _batch_job_lock:
            current_job = _batch_jobs.get(job_id)
            if not current_job:
                return
            await _batch_job_mark_finished(current_job, "finished")
            snapshot = _batch_job_public(current_job)
        await _batch_job_emit(job, {"type": "end", "snapshot": snapshot})
    except Exception as exc:
        logger.exception("批量编码任务失败: job=%s", job_id)
        async with _batch_job_lock:
            current_job = _batch_jobs.get(job_id)
            if current_job:
                await _batch_job_mark_finished(current_job, "failed", str(exc))
                snapshot = _batch_job_public(current_job)
            else:
                snapshot = {}
        await _batch_job_emit(job, {"type": "failed", "error": str(exc), "snapshot": snapshot})


async def _batch_job_scheduler() -> None:
    global _batch_job_active_id, _batch_job_scheduler_task
    while True:
        async with _batch_job_lock:
            await _batch_job_cleanup_locked()
            next_job_id = None
            while _batch_job_queue:
                candidate = _batch_job_queue.popleft()
                candidate_job = _batch_jobs.get(candidate)
                if candidate_job and candidate_job.get("status") == "queued":
                    next_job_id = candidate
                    break
            if not next_job_id:
                _batch_job_active_id = None
                _batch_job_scheduler_task = None
                return
            _batch_job_active_id = next_job_id
        try:
            await _run_batch_job(next_job_id)
        finally:
            async with _batch_job_lock:
                if _batch_job_active_id == next_job_id:
                    _batch_job_active_id = None


async def _batch_job_create(request: "PipeBatchEncodeRequest") -> Dict[str, Any]:
    global _batch_job_scheduler_task
    job_id = uuid.uuid4().hex
    threshold = get_pipe_encoder().get_threshold()
    job = {
        "job_id": job_id,
        "status": "queued",
        "items": copy.deepcopy(request.items),
        "items_meta": [
            {
                "index": int(item.get("client_index", idx)),
                "text": str(item.get("text", "") or ""),
                "project_name": str(item.get("project_name", "") or ""),
            }
            for idx, item in enumerate(request.items)
        ],
        "results": {},
        "item_traces": {},
        "processed": 0,
        "success_count": 0,
        "review_count": 0,
        "total": len(request.items),
        "threshold": threshold,
        "max_concurrent": request.max_concurrent or get_batch_max_concurrent(),
        "cancel_requested": False,
        "created_at": _utc_ts(),
        "started_at": None,
        "finished_at": None,
        "updated_at": _utc_ts(),
        "error": "",
        "subscribers": [],
    }
    async with _batch_job_lock:
        await _batch_job_cleanup_locked()
        _batch_jobs[job_id] = job
        _batch_job_queue.append(job_id)
        if _batch_job_scheduler_task is None or _batch_job_scheduler_task.done():
            _batch_job_scheduler_task = asyncio.create_task(_batch_job_scheduler())
    return _batch_job_public(job)


def _resolve_qwen3_stage1_config(
    qwen3_config: Dict[str, Any],
    ner_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    统一解析阶段一配置。

    新结构优先：
    qwen3.stage1.{router,type_models,material_model,standard_model,structural_prompt,structural_rules}

    旧结构兼容：
    qwen3.router / qwen3.category_models / ner.structural_prompt
    """
    qwen3_config = qwen3_config or {}
    ner_config = ner_config or {}

    stage1 = copy.deepcopy(qwen3_config.get("stage1") or {})
    if not stage1:
        stage1 = {
            "router": copy.deepcopy(qwen3_config.get("router") or {}),
            "type_models": copy.deepcopy(qwen3_config.get("category_models") or {}),
            "material_model": copy.deepcopy(qwen3_config.get("material_model") or {}),
            "standard_model": copy.deepcopy(qwen3_config.get("standard_model") or {}),
            "structural_prompt": copy.deepcopy(ner_config.get("structural_prompt") or {}),
            "structural_rules": {},
        }
    else:
        stage1.setdefault("router", copy.deepcopy(qwen3_config.get("router") or {}))
        stage1.setdefault("type_models", copy.deepcopy(qwen3_config.get("category_models") or {}))
        stage1.setdefault("material_model", copy.deepcopy(qwen3_config.get("material_model") or {}))
        stage1.setdefault("standard_model", copy.deepcopy(qwen3_config.get("standard_model") or {}))
        stage1.setdefault("structural_rules", {})
        if not stage1.get("structural_prompt"):
            stage1["structural_prompt"] = copy.deepcopy(ner_config.get("structural_prompt") or {})
    return stage1


def _build_qwen3_predictor_from_config(qwen3_config: Dict[str, Any]):
    adapter = qwen3_config.get("adapter")
    backend = qwen3_config.get("backend")
    if not adapter:
        raise RuntimeError("缺少配置: ner.qwen3.adapter")
    if not backend:
        raise RuntimeError("缺少配置: ner.qwen3.backend")

    if adapter == "structured_llamafactory":
        from src.llm_ner.structured_llamafactory_adapter import StructuredLlamaFactoryPredictor

        max_new_tokens = qwen3_config.get("num_predict")
        temperature = qwen3_config.get("temperature")
        if max_new_tokens is None:
            raise RuntimeError("缺少配置: ner.qwen3.num_predict")
        if temperature is None:
            raise RuntimeError("缺少配置: ner.qwen3.temperature")
        if backend == "ollama":
            device = qwen3_config.get("device")
            if device is None:
                raise RuntimeError("缺少配置: ner.qwen3.device")
            model_name = qwen3_config.get("model_name")
            ollama_url = qwen3_config.get("ollama_url")
            if not model_name:
                raise RuntimeError("缺少配置: ner.qwen3.model_name")
            if not ollama_url:
                raise RuntimeError("缺少配置: ner.qwen3.ollama_url")
            logger.info(
                "加载 Qwen3 结构化一阶段模型: model=%s, backend=%s, adapter=%s",
                model_name,
                backend,
                adapter,
            )
            return StructuredLlamaFactoryPredictor(
                backend="ollama",
                model_name=model_name,
                ollama_url=ollama_url,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        if backend == "hf_lazy_service":
            model_name = qwen3_config.get("model_name")
            service_url = qwen3_config.get("service_url")
            if not model_name:
                raise RuntimeError("缺少配置: ner.qwen3.model_name")
            if not service_url:
                raise RuntimeError("缺少配置: ner.qwen3.service_url")
            logger.info(
                "加载 Qwen3 结构化一阶段模型: model=%s, backend=%s, adapter=%s, service=%s",
                model_name,
                backend,
                adapter,
                service_url,
            )
            return StructuredLlamaFactoryPredictor(
                backend="hf_lazy_service",
                model_name=model_name,
                service_url=service_url,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        device = qwen3_config.get("device")
        if device is None:
            raise RuntimeError("缺少配置: ner.qwen3.device")
        configured_model_path = qwen3_config.get("model_path")
        if not configured_model_path:
            raise RuntimeError("缺少配置: ner.qwen3.model_path")
        model_path = str(PROJECT_ROOT / configured_model_path)
        if Path(configured_model_path).is_absolute():
            model_path = configured_model_path
        logger.info(
            "加载 Qwen3 结构化一阶段模型: 路径=%s, 设备=%s, adapter=%s",
            model_path,
            device,
            adapter,
        )
        return StructuredLlamaFactoryPredictor(
            backend="transformers",
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    from src.llm_ner.predictor import Qwen3Predictor
    if backend == "ollama":
        model_name = qwen3_config.get("model_name")
        ollama_url = qwen3_config.get("ollama_url")
        output_mode = qwen3_config.get("output_mode")
        ollama_num_predict = qwen3_config.get("num_predict")
        ollama_temperature = qwen3_config.get("temperature")
        ollama_top_p = qwen3_config.get("top_p")
        ollama_logprobs_enabled = qwen3_config.get("logprobs_enabled")
        if not model_name:
            raise RuntimeError("缺少配置: ner.qwen3.model_name")
        if not ollama_url:
            raise RuntimeError("缺少配置: ner.qwen3.ollama_url")
        if output_mode is None:
            raise RuntimeError("缺少配置: ner.qwen3.output_mode")
        if ollama_num_predict is None:
            raise RuntimeError("缺少配置: ner.qwen3.num_predict")
        if ollama_temperature is None:
            raise RuntimeError("缺少配置: ner.qwen3.temperature")
        if ollama_top_p is None:
            raise RuntimeError("缺少配置: ner.qwen3.top_p")
        if ollama_logprobs_enabled is None:
            raise RuntimeError("缺少配置: ner.qwen3.logprobs_enabled")
        logger.info("加载 Qwen3 NER 模型: Ollama 后端, 模型: %s, 输出模式: %s", model_name, output_mode)
        return Qwen3Predictor(
            model_name=model_name,
            backend="ollama",
            ollama_url=ollama_url,
            output_mode=output_mode,
            ollama_num_predict=ollama_num_predict,
            ollama_temperature=ollama_temperature,
            ollama_top_p=ollama_top_p,
            ollama_logprobs_enabled=ollama_logprobs_enabled,
        )

    configured_model_path = qwen3_config.get("model_path")
    device = qwen3_config.get("device")
    output_mode = qwen3_config.get("output_mode")
    if not configured_model_path:
        raise RuntimeError("缺少配置: ner.qwen3.model_path")
    if device is None:
        raise RuntimeError("缺少配置: ner.qwen3.device")
    if output_mode is None:
        raise RuntimeError("缺少配置: ner.qwen3.output_mode")
    model_path = str(PROJECT_ROOT / configured_model_path)
    if Path(configured_model_path).is_absolute():
        model_path = configured_model_path
    logger.info(
        "加载 Qwen3 NER 模型: Transformers 后端, 路径: %s, 设备: %s, 输出模式: %s",
        model_path,
        device,
        output_mode,
    )
    return Qwen3Predictor(
        model_path=model_path,
        backend="transformers",
        device=device,
        output_mode=output_mode,
    )


def _build_routed_qwen3_predictor(qwen3_config: Dict[str, Any]):
    from src.llm_ner.stage1_orchestrator import Stage1FieldOrchestrator

    ner_config = get_ner_config()
    stage1_cfg = _resolve_qwen3_stage1_config(qwen3_config, ner_config)
    router_cfg = copy.deepcopy(stage1_cfg.get("router", {}) or {})
    type_models = copy.deepcopy(stage1_cfg.get("type_models") or {})
    material_model = copy.deepcopy(stage1_cfg.get("material_model") or {})
    standard_model = copy.deepcopy(stage1_cfg.get("standard_model") or {})
    structural_prompt_cfg = copy.deepcopy(stage1_cfg.get("structural_prompt") or {})
    structural_rules_cfg = copy.deepcopy(stage1_cfg.get("structural_rules") or {})

    base_qwen3_config = copy.deepcopy(qwen3_config)
    base_qwen3_config.pop("router", None)
    base_qwen3_config.pop("category_models", None)
    base_qwen3_config.pop("stage1", None)

    router = None
    if router_cfg.get("enabled", False):
        from src.llm_ner.router import build_category_router

        router = build_category_router(router_cfg, project_root=PROJECT_ROOT)
    fallback_category = str(router_cfg.get("fallback_category", "其他管件")).strip() or "其他管件"

    shared_default_predictor: Dict[str, Any] = {"instance": None}

    def default_factory():
        if shared_default_predictor["instance"] is None:
            shared_default_predictor["instance"] = _build_qwen3_predictor_from_config(base_qwen3_config)
        return shared_default_predictor["instance"]

    type_factories = {}
    for category, override in type_models.items():
        if category in {"默认", "default"}:
            continue
        merged_cfg = _merge_nested_dict(base_qwen3_config, override or {})
        type_factories[category] = (
            lambda cfg=merged_cfg: _build_qwen3_predictor_from_config(cfg)
        )
    if material_model:
        material_cfg = _merge_nested_dict(base_qwen3_config, material_model or {})
        material_factory = lambda cfg=material_cfg: _build_qwen3_predictor_from_config(cfg)
    else:
        material_cfg = base_qwen3_config
        material_factory = default_factory
    if standard_model:
        standard_cfg = _merge_nested_dict(base_qwen3_config, standard_model or {})
        standard_factory = lambda cfg=standard_cfg: _build_qwen3_predictor_from_config(cfg)
    else:
        standard_cfg = base_qwen3_config
        standard_factory = default_factory

    share_material_standard = material_cfg == standard_cfg

    def structural_extractor_factory():
        if not structural_prompt_cfg.get("enabled", False) and not structural_rules_cfg.get("enabled", False):
            return None
        from src.llm_ner.structural_field_resolver import StructuralFieldResolver

        return StructuralFieldResolver.from_configs(
            prompt_config=structural_prompt_cfg,
            rule_config=structural_rules_cfg,
        )

    logger.info(
        "加载路由版 Qwen3 一阶段编排器: router=%s, TYPE覆盖分类数=%s, material_override=%s, standard_override=%s, structural_prompt=%s, structural_rules=%s",
        router_cfg.get("backend", "disabled"),
        len(type_factories),
        bool(material_model),
        bool(standard_model),
        bool(structural_prompt_cfg.get("enabled", False)),
        bool(structural_rules_cfg.get("enabled", False)),
    )
    return Stage1FieldOrchestrator(
        router=router,
        default_type_factory=default_factory,
        type_factories=type_factories,
        material_factory=material_factory,
        standard_factory=standard_factory,
        share_material_standard=share_material_standard,
        structural_extractor_factory=structural_extractor_factory if (structural_prompt_cfg.get("enabled", False) or structural_rules_cfg.get("enabled", False)) else None,
        fallback_category=fallback_category,
        direct_threshold=float(router_cfg.get("direct_threshold", 0.9)),
        review_threshold=float(router_cfg.get("review_threshold", 0.7)),
        encodable_categories=set(router_cfg.get("encodable_categories") or []),
    )


def get_ner_predictor():
    """获取NER预测器实例（惰性加载，根据配置选择模型）"""
    global _ner_predictor, _ner_model_type
    
    ner_config = get_ner_config()
    model_type = ner_config.get("model_type", "bert")
    
    # 如果模型类型改变，需要重新加载
    if _ner_predictor is not None and _ner_model_type == model_type:
        return _ner_predictor
    
    if model_type == "qwen3":
        # 使用 Qwen3-4B 微调模型
        logger.info("使用 Qwen3 NER 模型")

        qwen3_config = ner_config.get("qwen3", {})
        stage1_cfg = _resolve_qwen3_stage1_config(qwen3_config, ner_config)
        router_enabled = bool((stage1_cfg.get("router") or {}).get("enabled", False))
        stage1_orchestrator_enabled = bool(
            router_enabled
            or (stage1_cfg.get("type_models") or {})
            or (stage1_cfg.get("material_model") or {})
            or (stage1_cfg.get("standard_model") or {})
            or (stage1_cfg.get("structural_prompt") or {}).get("enabled", False)
            or (stage1_cfg.get("structural_rules") or {}).get("enabled", False)
        )
        if stage1_orchestrator_enabled:
            _ner_predictor = _build_routed_qwen3_predictor(qwen3_config)
        else:
            _ner_predictor = _build_qwen3_predictor_from_config(qwen3_config)
        _ner_model_type = "qwen3"
    elif model_type == "globalpointer":
        # 使用 GlobalPointer 模型
        logger.info("使用 GlobalPointer NER 模型")
        
        try:
            from apps.trainer.globalpointer_ner.predict import GlobalPointerPredictor
        except ImportError as e:
            logger.error(f"无法导入 GlobalPointer 预测器: {e}")
            raise RuntimeError("GlobalPointer NER 模型依赖未安装")
        
        gp_config = ner_config.get("globalpointer", {})
        model_path = str(PROJECT_ROOT / gp_config.get("model_path", "outputs/globalpointer_ner/best_model"))
        threshold = gp_config.get("threshold", 0.0)
        device = gp_config.get("device", "auto")
        
        logger.info(f"加载 GlobalPointer NER 模型: {model_path}, threshold={threshold}")
        _ner_predictor = GlobalPointerPredictor(
            model_path=model_path,
            threshold=threshold,
            device=device
        )
        _ner_model_type = "globalpointer"
    else:
        # 使用 BERT NER 模型（默认）
        logger.info("使用 BERT NER 模型")
        bert_config = ner_config.get("bert", {})
        model_path = PROJECT_ROOT / bert_config.get("model_path", "models/pipe_model")
        if not model_path.exists():
            raise RuntimeError(f"NER模型不存在: {model_path}")
        
        # O标签偏置：正值使模型更倾向于预测O（对未知token更保守）
        o_bias = ner_config.get("o_bias", 0.0)
        # 设备配置
        device = bert_config.get("device", "auto")
        
        logger.info(f"加载 BERT NER 模型: {model_path}, O标签偏置: {o_bias}, 设备: {device}")
        _ner_predictor = PipePredictor(str(model_path), device=device, o_bias=o_bias)
        _ner_model_type = "bert"
    
    return _ner_predictor


def get_pipe_router():
    """获取管道描述路由器实例（惰性加载）"""
    global _pipe_router
    if _pipe_router is not None:
        return _pipe_router

    ner_config = get_ner_config()
    qwen3_config = ner_config.get("qwen3", {}) or {}
    stage1_cfg = _resolve_qwen3_stage1_config(qwen3_config, ner_config)
    router_cfg = stage1_cfg.get("router", {}) or {}
    if not router_cfg.get("enabled", False):
        return None

    from src.llm_ner.router import build_category_router

    _pipe_router = build_category_router(router_cfg, project_root=PROJECT_ROOT)
    return _pipe_router


def get_structural_prompt_extractor():
    """获取结构字段决策器（规则优先 + 提示词回退，惰性加载）。"""
    global _structural_prompt_extractor

    ner_config = get_ner_config()
    qwen3_config = ner_config.get("qwen3", {}) or {}
    stage1_cfg = _resolve_qwen3_stage1_config(qwen3_config, ner_config)
    prompt_cfg = stage1_cfg.get("structural_prompt", {}) or {}
    rule_cfg = stage1_cfg.get("structural_rules", {}) or {}
    if not prompt_cfg.get("enabled", False) and not rule_cfg.get("enabled", False):
        return None

    if _structural_prompt_extractor is None:
        from src.llm_ner.structural_field_resolver import StructuralFieldResolver

        _structural_prompt_extractor = StructuralFieldResolver.from_configs(
            prompt_config=prompt_cfg,
            rule_config=rule_cfg,
        )
        logger.info(
            "启用结构字段决策: prompt=%s, rules=%s, model=%s",
            bool(prompt_cfg.get("enabled", False)),
            bool(rule_cfg.get("enabled", False)),
            prompt_cfg.get("model_name"),
        )
    return _structural_prompt_extractor


def _is_empty_structural_value(field: str, value: Any) -> bool:
    if field in {"SIZE", "THICKNESS"}:
        return not isinstance(value, dict) or not any(
            v not in (None, "", [], {}) for v in value.values()
        )
    if field == "PRESSURE":
        return value in (None, "", [], {})
    return True


def _size_debug_snapshot(value: Any) -> Any:
    """收缩 SIZE 调试输出，避免日志过长。"""
    if isinstance(value, dict):
        snapshot = {
            "DN": copy.deepcopy(value.get("DN", [])),
            "OD": copy.deepcopy(value.get("OD", [])),
            "INCH": copy.deepcopy(value.get("INCH", [])),
            "LENGTH": copy.deepcopy(value.get("LENGTH", [])),
        }
        if isinstance(value.get("_ITEMS"), list):
            snapshot["_ITEMS"] = copy.deepcopy(value.get("_ITEMS"))
        return snapshot
    return copy.deepcopy(value)


def _apply_structural_prompt_override(result: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    """
    用结构字段决策器（规则优先 + 提示词回退）覆盖强结构字段。

    微调模型仍负责 TYPE / MATERIAL / STANDARD。SIZE / THICKNESS / PRESSURE
    以统一结构字段决策结果为准，避免微调模型按训练分布补 SCH/CL 等不存在的值。
    """
    structural = _extract_structural_prompt_fields(text)
    return _merge_structural_prompt_output(result, structural)


def _extract_structural_prompt_fields(text: str) -> Optional[Dict[str, Any]]:
    extractor = get_structural_prompt_extractor()
    if extractor is None:
        return None
    try:
        structural = extractor.extract(text)
        return structural
    except Exception as exc:
        logger.warning("[结构字段提示词] 抽取失败，保留微调结果: %s", exc)
        return None


def _merge_structural_prompt_output(
    result: Dict[str, Any],
    structural: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not structural:
        return None

    model_output = result.get("model_output")
    if not isinstance(model_output, dict):
        model_output = {}
        result["model_output"] = model_output

    target = model_output.get("decisions") if isinstance(model_output.get("decisions"), dict) else model_output
    for field in ("SIZE", "THICKNESS", "PRESSURE"):
        value = structural.get(field)
        target[field] = copy.deepcopy(value)

    structural_visible = {
        k: copy.deepcopy(v) for k, v in structural.items() if not str(k).startswith("_")
    }
    structural_status = copy.deepcopy(structural.get("_status", {}) or {})
    structural_errors = copy.deepcopy(structural.get("_errors", {}) or {})
    model_output["_STRUCTURAL_PROMPT"] = structural_visible
    model_output["_STRUCTURAL_PROMPT_RAW"] = structural.get("_raw", "")
    model_output["_STRUCTURAL_PROMPT_STATUS"] = structural_status
    model_output["_STRUCTURAL_PROMPT_ERRORS"] = structural_errors
    result["structural_prompt_output"] = structural_visible
    result["structural_prompt_raw_response"] = structural.get("_raw", "")
    result["structural_prompt_status"] = structural_status
    result["structural_prompt_errors"] = structural_errors

    logger.info(
        "[结构字段覆盖] source=%s SIZE=%s THICKNESS=%s PRESSURE=%s",
        structural.get("_sources"),
        structural_visible.get("SIZE"),
        structural_visible.get("THICKNESS"),
        structural_visible.get("PRESSURE"),
    )

    extract_confidence = result.get("extract_confidence")
    if not isinstance(extract_confidence, dict):
        extract_confidence = {}
        result["extract_confidence"] = extract_confidence
    for field in ("SIZE", "THICKNESS", "PRESSURE"):
        extract_confidence[field] = 1.0 if not _is_empty_structural_value(field, structural.get(field)) else 0.0

    extract_confidence_v2 = result.get("extract_confidence_v2")
    if not isinstance(extract_confidence_v2, dict):
        extract_confidence_v2 = {}
        result["extract_confidence_v2"] = extract_confidence_v2
    source_map = structural.get("_sources") if isinstance(structural.get("_sources"), dict) else {}
    prompt_status_map = structural_status if isinstance(structural_status, dict) else {}
    field_to_prompt_task = {
        "SIZE": "size_length",
        "THICKNESS": "thickness",
        "PRESSURE": "pressure",
    }
    for field in ("SIZE", "THICKNESS", "PRESSURE"):
        source = str(source_map.get(field, "prompt_extraction"))
        prompt_task = field_to_prompt_task[field]
        prompt_status = str(prompt_status_map.get(prompt_task, "") or "")
        if _is_empty_structural_value(field, structural.get(field)):
            reason = "field_missing"
            evidence: Dict[str, Any] = {}
            if source == "prompt_extraction" and prompt_status:
                reason = f"prompt_{prompt_status}"
                if prompt_task in structural_errors:
                    evidence["prompt_error"] = structural_errors[prompt_task]
            extract_confidence_v2[field] = {
                "source": source,
                "confidence": 0.0,
                "reason": reason,
                "evidence": evidence,
            }
        else:
            evidence = {}
            if source == "prompt_extraction" and prompt_status:
                evidence["prompt_status"] = prompt_status
            extract_confidence_v2[field] = {
                "source": source,
                "confidence": 1.0,
                "reason": "rule_based_extraction" if source == "rule_extraction" else "prompt_extraction",
                "evidence": evidence,
            }

    return structural_visible


def get_ner_confidence_threshold() -> float:
    """获取NER置信度阈值"""
    ner_config = get_ner_config()
    return ner_config.get("confidence_threshold", 0.9)


def get_batch_max_concurrent(default_value: int = 3) -> int:
    """获取批处理并发上限（最小为1）"""
    platform_config = get_platform_config()
    batch_cfg = platform_config.get("batch_processing", {}) or {}
    try:
        n = int(batch_cfg.get("max_concurrent", default_value))
    except Exception:
        n = default_value
    return max(1, n)


def get_tokenizer(model: str, platform: str = "pipe") -> LLMTokenizer:
    """获取分词器实例"""
    cache_key = f"{model}_{platform}"
    if cache_key not in _tokenizers:
        logger.info(f"创建分词器: model={model}, platform={platform}")
        _tokenizers[cache_key] = LLMTokenizer(model=model, platform=platform)
    return _tokenizers[cache_key]


# ============================================================
# 请求/响应模型
# ============================================================

class TokenizeRequest(BaseModel):
    """分词请求"""
    text: str = Field(..., description="待分词文本")
    preprocess: bool = Field(True, description="是否预处理")
    model: str = Field("deepseek-chat", description="使用的模型")
    platform: str = Field("pipe", description="平台类型")


class BatchTokenizeRequest(BaseModel):
    """批量分词请求"""
    texts: List[str] = Field(..., description="待分词文本列表")
    preprocess: bool = Field(True, description="是否预处理")
    model: str = Field("deepseek-chat", description="使用的模型")
    platform: str = Field("pipe", description="平台类型")


class TokenInfo(BaseModel):
    """分词信息"""
    word: str
    tag: str


class PipeEncodeRequest(BaseModel):
    """管道材料编码请求"""
    entities: Dict[str, Any] = Field(..., description="NER识别结果")
    text: str = Field("", description="原始描述")
    project_name: str = Field("", description="项目名称，可选")
    extract_confidence: Optional[Dict[str, Any]] = Field(None, description="第一阶段抽取置信度（按字段）")
    extract_confidence_v2: Optional[Dict[str, Any]] = Field(None, description="第一阶段抽取置信度V2（结构化）")


class PipeEncodeFromTokensRequest(BaseModel):
    """管道材料编码请求（基于分词结果）"""
    tokens: List[TokenInfo] = Field(..., description="分词结果")
    text: str = Field("", description="原始描述")
    project_name: str = Field("", description="项目名称，可选")


class PipeBatchEncodeRequest(BaseModel):
    """批量管道材料编码请求"""
    items: List[Dict] = Field(..., description="编码项列表")
    max_concurrent: Optional[int] = Field(None, description="批量编码并发数，可选")


class PipeDifficultyFinalizeRequest(BaseModel):
    """批量分流修正请求（项目维度）"""
    items: List[Dict[str, Any]] = Field(..., description="已编码项列表")


class PipeBatchJobCreateResponse(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int


class H3yunImportItem(BaseModel):
    """氚云导入数据项"""
    description: str = Field(..., description="描述")
    code: str = Field(..., description="编码")
    type_raw: str = Field("", description="原始种类")
    type_code: str = Field("", description="标准化种类")
    size_raw: str = Field("", description="原始尺寸")
    size_code: str = Field("", description="标准化尺寸")
    thickness_raw: str = Field("", description="原始壁厚")
    thickness_code: str = Field("", description="标准化壁厚")
    pressure_raw: str = Field("", description="原始磅级")
    pressure_code: str = Field("", description="标准化磅级")
    material_raw: str = Field("", description="原始材质")
    material_code: str = Field("", description="标准化材质")
    standard_raw: str = Field("", description="原始规范")
    standard_code: str = Field("", description="标准化规范")


class H3yunImportRequest(BaseModel):
    """氚云导入请求"""
    items: List[H3yunImportItem] = Field(..., description="导入数据列表")
    encode_date: str = Field(..., description="编码日期时间，格式：YYYY-MM-DD HH:MM")


def _serialize_pipe_entity_for_encode(entity: Dict[str, Any]) -> Any:
    """保留 STANDARD_* 修饰项的 bind_to_index / 位置信息，其他字段保持原样值。"""
    value = entity.get("value") or entity.get("text", "")
    payload: Dict[str, Any] = {"value": value}

    for key in ("bind_to_index", "start", "end", "subtype"):
        if entity.get(key) is not None:
            payload[key] = entity.get(key)

    if len(payload) == 1:
        return value
    return payload


def _append_pipe_entity(entities: Dict[str, Any], field: str, val: Any):
    """
    聚合 predictor 输出的实体。

    - 普通字段: 保持旧的 string / list 形式
    - 带 subtype 的结构化字段（如 SIZE/THICKNESS）: 还原为嵌套对象
    """
    if isinstance(val, dict) and val.get("subtype") is not None:
        subtype = str(val["subtype"])
        value = val.get("value")
        if value in (None, ""):
            return
        field_obj = entities.get(field)
        if not isinstance(field_obj, dict):
            field_obj = {}
            entities[field] = field_obj
        bucket = field_obj.get(subtype)
        if not isinstance(bucket, list):
            bucket = []
            field_obj[subtype] = bucket
        bucket.append(value)
        return

    if field in entities:
        prev = entities[field]
        entities[field] = [prev, val] if not isinstance(prev, list) else prev + [val]
    else:
        entities[field] = val


def _build_pipe_entities_for_encode(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    优先使用语义解析器输出中的 decisions 作为编码输入。
    若不存在结构化输出，则回退到旧的打平聚合逻辑。
    """
    model_output = result.get("model_output")
    if isinstance(model_output, dict):
        decisions = model_output.get("decisions")
        if isinstance(decisions, dict) and decisions:
            return copy.deepcopy(decisions)
        structured = {
            k: copy.deepcopy(v)
            for k, v in model_output.items()
            if not str(k).startswith("_") and k != "model_raw_response"
        }
        if structured:
            return structured

    entities: Dict[str, Any] = {}
    for e in result.get("entities", []):
        field = e["type"]
        val = _serialize_pipe_entity_for_encode(e)
        _append_pipe_entity(entities, field, val)
    return entities


def _decorate_predict_route_info(route_info: Any) -> Dict[str, Any]:
    info = copy.deepcopy(route_info) if isinstance(route_info, dict) else {}
    encodable_categories = set(
        (((_resolve_qwen3_stage1_config((get_ner_config().get("qwen3", {}) or {}), get_ner_config()).get("router", {}) or {}).get("encodable_categories")) or [])
    )
    selected_category = str(info.get("category") or "").strip()
    if selected_category:
        encoding_enabled = selected_category in encodable_categories if encodable_categories else True
        info["encoding_enabled"] = encoding_enabled
        info["skip_encoding_reason"] = "" if encoding_enabled else f"类别“{selected_category}”只分类，不参与编码"
    else:
        info.setdefault("encoding_enabled", True)
        info.setdefault("skip_encoding_reason", "")
    return info


# ============================================================
# 标注相关API
# ============================================================

@app.get("/")
async def root():
    """根路径"""
    return {"message": "材料智能处理平台", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    tokenizer_config = get_tokenizer_config()
    default_model = tokenizer_config.get("default_model", "deepseek-chat")
    
    models = [
        {"value": "ner_pipe", "label": "NER模型（本地）", "available": True},
        {"value": "deepseek-chat", "label": "DeepSeek API", "available": True},
        {"value": "qwen3:8b", "label": "Qwen3 8B", "available": True},
        {"value": "qwen3:0.6b", "label": "Qwen3 0.6B", "available": True},
    ]
    return {
        "models": models,
        "default": default_model
    }


@app.get("/api/config")
async def get_config():
    """获取平台配置"""
    tokenizer_config = get_tokenizer_config()
    semantic_config = get_semantic_config()
    platform_config = get_platform_config()
    batch_config = platform_config.get("batch_processing", {})
    
    return {
        "tokenizer": {
            "default_model": tokenizer_config.get("default_model", "deepseek-chat"),
            "default_platform": tokenizer_config.get("default_platform", "pipe"),
        },
        "semantic": {
            "model_name": semantic_config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            "similarity_threshold": semantic_config.get("similarity_threshold", 0.9),
        },
        "batch_processing": {
            "max_concurrent": int(batch_config.get("max_concurrent", 3)),
            "progress_interval_ms": int(batch_config.get("progress_interval_ms", 100)),
        },
    }


@app.post("/api/tokenize")
async def tokenize(request: TokenizeRequest):
    """分词接口"""
    text = request.text
    
    if not text or not text.strip():
        return {"success": False, "error": "文本为空"}
    
    # 预处理
    processed_text = text
    if request.preprocess:
        processed_text = _preprocessor.process(text)
    
    # 分词
    try:
        # 如果是 NER 模型，使用本地 BERT 预测器
        if request.model == "ner_pipe":
            predictor = get_ner_predictor()
            result = predictor.predict(processed_text)
            
            # 打印 STANDARD 实体识别结果（调试用）
            standard_tokens = [t for t in result["tokens"] if t["tag"] == "STANDARD"]
            if standard_tokens:
                logger.info(f"[NER识别] STANDARD实体: {[(t['word'], t.get('confidence', 'N/A')) for t in standard_tokens]}")
            
            # 获取置信度阈值
            confidence_threshold = get_ner_confidence_threshold()
            
            # 过滤低置信度的实体（将其标记为 O）
            tokens = []
            for t in result["tokens"]:
                conf = t.get("confidence", 1.0)
                tag = t["tag"]
                # 如果置信度低于阈值，将非 O 标签改为 O
                if tag != "O" and conf < confidence_threshold:
                    logger.debug(f"过滤低置信度实体: {t['word']} ({tag}) 置信度={conf:.2%}")
                    tag = "O"
                tokens.append({
                    "word": t["word"], 
                    "tag": tag, 
                    "confidence": conf,
                    "start": t.get("start"),
                    "end": t.get("end")
                })
            
            type_class = result.get("type_class")
        else:
            # 使用 LLM 分词器
            tokenizer = get_tokenizer(request.model, request.platform)
            result = await tokenizer.tokenize(processed_text)
            tokens = result.get("tokens", [])
            type_class = result.get("type_class")
        
        return {
            "success": True,
            "original_text": text,
            "processed_text": processed_text,
            "tokens": tokens,
            "type_class": type_class
        }
    except Exception as e:
        logger.error(f"分词失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/tokenize/batch")
async def batch_tokenize(request: BatchTokenizeRequest):
    """批量分词（异步并发）"""
    
    # 如果是 NER 模型，使用本地预测器（同步）
    if request.model == "ner_pipe":
        predictor = get_ner_predictor()
        confidence_threshold = get_ner_confidence_threshold()
        results = []
        for text in request.texts:
            if not text or not text.strip():
                results.append({"success": False, "error": "文本为空"})
                continue
            
            processed_text = text
            if request.preprocess:
                processed_text = _preprocessor.process(text)
            
            try:
                result = predictor.predict(processed_text)
                # 过滤低置信度的实体
                tokens = []
                for t in result["tokens"]:
                    conf = t.get("confidence", 1.0)
                    tag = t["tag"]
                    if tag != "O" and conf < confidence_threshold:
                        tag = "O"
                    tokens.append({"word": t["word"], "tag": tag, "confidence": conf})
                results.append({
                    "success": True,
                    "original_text": text,
                    "processed_text": processed_text,
                    "tokens": tokens,
                    "type_class": result.get("type_class")
                })
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return {
            "success": True,
            "total": len(results),
            "results": results
        }
    
    # LLM 模型，使用异步并发
    async def process_single(text: str):
        if not text or not text.strip():
            return {"success": False, "error": "文本为空"}
        
        processed_text = text
        if request.preprocess:
            processed_text = _preprocessor.process(text)
        
        try:
            tokenizer = get_tokenizer(request.model, request.platform)
            result = await tokenizer.tokenize(processed_text)
            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "tokens": result.get("tokens", []),
                "type_class": result.get("type_class")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 并发执行
    tasks = [process_single(text) for text in request.texts]
    results = await asyncio.gather(*tasks)
    
    return {
        "success": True,
        "total": len(results),
        "results": results
    }


# ============================================================
# 编码相关API
# ============================================================

class PipePredictRequest(BaseModel):
    """管道材料NER预测请求（直接返回实体，不走token中转）"""
    text: str = Field(..., description="待提取文本")
    preprocess: bool = Field(True, description="是否预处理")


class PipeBatchPredictRequest(BaseModel):
    """批量管道材料NER预测请求"""
    texts: List[str] = Field(..., description="待提取文本列表")
    preprocess: bool = Field(True, description="是否预处理")


class PipeRouteRequest(BaseModel):
    """管道材料路由请求"""
    text: str = Field(..., description="待路由文本")
    preprocess: bool = Field(True, description="是否预处理")


class PipeBatchRouteRequest(BaseModel):
    """批量管道材料路由请求"""
    texts: List[str] = Field(..., description="待路由文本列表")
    preprocess: bool = Field(True, description="是否预处理")


@app.post("/api/pipe/predict")
async def pipe_predict(request: PipePredictRequest):
    """管道材料NER预测（直接返回JSON实体，供编码平台使用）"""
    text = request.text
    if not text or not text.strip():
        return {"success": False, "error": "文本为空"}

    processed_text = text
    if request.preprocess:
        processed_text = _preprocessor.process(text)

    try:
        predictor = get_ner_predictor()
        result = await asyncio.to_thread(predictor.predict, processed_text)
        if result.get("structural_prompt_output") is None:
            _apply_structural_prompt_override(result, processed_text)
        entities = _build_pipe_entities_for_encode(result)
        extract_confidence = result.get("extract_confidence", {}) or {}
        extract_confidence_v2 = result.get("extract_confidence_v2", {}) or {}

        return {
            "success": True,
            "original_text": text,
            "processed_text": processed_text,
            "entities": entities,
            "extract_confidence": extract_confidence,
            "extract_confidence_v2": extract_confidence_v2,
            "type_class": result.get("type_class"),
            "model_output": result.get("model_output", {}),
            "model_raw_response": result.get("model_raw_response", ""),
            "structural_prompt_output": result.get("structural_prompt_output"),
            "structural_prompt_raw_response": result.get("structural_prompt_raw_response", ""),
            "route_info": result.get("route_info"),
        }
    except Exception as e:
        logger.error(f"NER预测失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/pipe/predict/batch")
async def pipe_batch_predict(request: PipeBatchPredictRequest):
    """批量管道材料NER预测"""
    predictor = get_ner_predictor()
    semaphore = asyncio.Semaphore(get_batch_max_concurrent())

    async def process_one(text: str):
        if not text or not text.strip():
            return {"success": False, "error": "文本为空"}

        processed_text = _preprocessor.process(text) if request.preprocess else text
        try:
            async with semaphore:
                result = await asyncio.to_thread(predictor.predict, processed_text)
            if result.get("structural_prompt_output") is None:
                _apply_structural_prompt_override(result, processed_text)
            entities = _build_pipe_entities_for_encode(result)
            extract_confidence = result.get("extract_confidence", {}) or {}
            extract_confidence_v2 = result.get("extract_confidence_v2", {}) or {}

            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "entities": entities,
                "extract_confidence": extract_confidence,
                "extract_confidence_v2": extract_confidence_v2,
                "type_class": result.get("type_class"),
                "model_output": result.get("model_output", {}),
                "model_raw_response": result.get("model_raw_response", ""),
                "structural_prompt_output": result.get("structural_prompt_output"),
                "structural_prompt_raw_response": result.get("structural_prompt_raw_response", ""),
                "route_info": result.get("route_info"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*(process_one(t) for t in request.texts))

    return {"success": True, "total": len(results), "results": results}


@app.post("/api/pipe/route")
def pipe_route(request: PipeRouteRequest):
    """管道材料类别路由，仅返回路由结果，不执行一阶段抽取"""
    text = request.text
    if not text or not text.strip():
        return {"success": False, "error": "文本为空"}

    processed_text = _preprocessor.process(text) if request.preprocess else text

    try:
        router = get_pipe_router()
        if router is None:
            return {"success": False, "error": "当前未启用路由器"}

        route_info = router.route(processed_text)
        encodable_categories = set(
            (((_resolve_qwen3_stage1_config((get_ner_config().get("qwen3", {}) or {}), get_ner_config()).get("router", {}) or {}).get("encodable_categories")) or [])
        )
        selected_category = str(route_info.get("category") or "").strip()
        encoding_enabled = selected_category in encodable_categories
        route_info["encoding_enabled"] = encoding_enabled
        route_info["skip_encoding_reason"] = "" if encoding_enabled else f"类别“{selected_category}”只分类，不参与编码"
        return {
            "success": True,
            "original_text": text,
            "processed_text": processed_text,
            "route_info": route_info,
        }
    except Exception as e:
        logger.error(f"路由失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/pipe/route/batch")
async def pipe_batch_route(request: PipeBatchRouteRequest):
    """批量管道材料类别路由，仅返回路由结果"""
    router = get_pipe_router()
    if router is None:
        return {"success": False, "error": "当前未启用路由器"}

    semaphore = asyncio.Semaphore(get_batch_max_concurrent())

    async def process_one(text: str):
        if not text or not text.strip():
            return {"success": False, "error": "文本为空"}

        processed_text = _preprocessor.process(text) if request.preprocess else text
        try:
            async with semaphore:
                route_info = await asyncio.to_thread(router.route, processed_text)
            encodable_categories = set(
                (((_resolve_qwen3_stage1_config((get_ner_config().get("qwen3", {}) or {}), get_ner_config()).get("router", {}) or {}).get("encodable_categories")) or [])
            )
            selected_category = str(route_info.get("category") or "").strip()
            encoding_enabled = selected_category in encodable_categories
            route_info["encoding_enabled"] = encoding_enabled
            route_info["skip_encoding_reason"] = "" if encoding_enabled else f"类别“{selected_category}”只分类，不参与编码"
            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "route_info": route_info,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*(process_one(t) for t in request.texts))
    return {"success": True, "total": len(results), "results": results}


@app.post("/api/pipe/encode")
def pipe_encode(request: PipeEncodeRequest):
    """管道材料编码（基于实体）"""
    encoder = get_pipe_encoder()
    result = encoder.encode(
        request.entities,
        request.text,
        request.extract_confidence,
        request.extract_confidence_v2,
    )
    converted = _convert_pipe_result(result)
    converted = _attach_base_difficulty(converted, request.project_name)
    return _attach_second_pass(converted)


@app.post("/api/pipe/encode/tokens")
def pipe_encode_from_tokens(request: PipeEncodeFromTokensRequest):
    """管道材料编码（基于分词结果）"""
    encoder = get_pipe_encoder()
    tokens = [{"word": t.word, "tag": t.tag} for t in request.tokens]
    result = encoder.encode_from_tokens(tokens, request.text)
    converted = _convert_pipe_result(result)
    converted = _attach_base_difficulty(converted, request.project_name)
    return _attach_second_pass(converted)


@app.post("/api/pipe/encode/batch")
async def pipe_batch_encode(request: PipeBatchEncodeRequest):
    """批量管道材料编码"""
    encoder = get_pipe_encoder()
    semaphore = asyncio.Semaphore(get_batch_max_concurrent())

    async def process_item(item: Dict[str, Any]):
        async with semaphore:
            if 'tokens' in item:
                return await asyncio.to_thread(
                    encoder.encode_from_tokens,
                    item['tokens'],
                    item.get('text', '')
                )
            return await asyncio.to_thread(
                encoder.encode,
                item.get('entities', {}),
                item.get('text', ''),
                item.get('extract_confidence')
            )

    results = await asyncio.gather(*(process_item(item) for item in request.items))
    
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    review_count = sum(1 for r in results if r.need_review)
    
    converted_results = []
    for item, result in zip(request.items, results):
        converted_results.append(
            _attach_base_difficulty(
                _convert_pipe_result(result),
                str(item.get("project_name", "") or "").strip(),
            )
        )

    finalized = finalize_batch_difficulty(
        [
            {
                "text": converted.get("original_text", "") or "",
                "project_name": str(item.get("project_name", "") or "").strip(),
                "type_code": _extract_code_from_field(converted, "TYPE"),
                "material_code": _extract_code_from_field(converted, "MATERIAL"),
                "standard_code": _extract_code_from_field(converted, "STANDARD"),
                "standard_codes": _extract_standard_codes_from_field(converted),
                "base_difficulty": converted.get("difficulty_split"),
            }
            for item, converted in zip(request.items, converted_results)
        ]
    )
    for converted, difficulty in zip(converted_results, finalized):
        converted["difficulty_split"] = difficulty
        _attach_second_pass(converted)

    return {
        "total": total,
        "success_count": success_count,
        "review_count": review_count,
        "threshold": encoder.get_threshold(),
        "results": converted_results
    }


@app.post("/api/pipe/difficulty/finalize")
async def pipe_finalize_difficulty(request: PipeDifficultyFinalizeRequest):
    """根据项目维度对已编码结果做最终分流修正。"""
    finalized = finalize_batch_difficulty(request.items)
    return {
        "success": True,
        "total": len(finalized),
        "results": finalized,
    }


@app.post("/api/pipe/encode/batch/jobs")
async def pipe_batch_encode_create_job(request: PipeBatchEncodeRequest):
    snapshot = await _batch_job_create(request)
    return {"success": True, "job": snapshot}


@app.get("/api/pipe/encode/batch/jobs")
async def pipe_batch_encode_list_jobs():
    async with _batch_job_lock:
        await _batch_job_cleanup_locked()
        jobs = [
            _batch_job_public(job, include_results=False)
            for job in _batch_jobs.values()
            if str(job.get("status", "") or "") in _BATCH_JOB_ACTIVE_STATUSES
        ]
    jobs.sort(key=lambda item: float(item.get("created_at") or 0.0))
    return {"success": True, "jobs": jobs}


@app.get("/api/pipe/encode/batch/jobs/{job_id}")
async def pipe_batch_encode_get_job(job_id: str):
    job = await _batch_job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"success": True, "job": _batch_job_public(job)}


@app.get("/api/pipe/encode/batch/jobs/{job_id}/items/{item_index}")
async def pipe_batch_encode_get_job_item(job_id: str, item_index: int):
    job = await _batch_job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    items_meta = list(job.get("items_meta", []))
    order_index = -1
    item_meta: Dict[str, Any] | None = None
    for idx, meta in enumerate(items_meta):
        if int(meta.get("index", idx)) == item_index:
            order_index = idx
            item_meta = copy.deepcopy(meta)
            break
    if order_index < 0:
        raise HTTPException(status_code=404, detail="任务条目不存在")

    result = copy.deepcopy(job.get("results", {}).get(str(order_index)))
    trace = copy.deepcopy(job.get("item_traces", {}).get(str(order_index), {}))
    item_status = str(trace.get("status", "") or "")
    if not item_status:
        item_status = "pending" if result is None else "processed"

    return {
        "success": True,
        "job_id": job_id,
        "job_status": str(job.get("status", "") or ""),
        "item_index": int(item_index),
        "order_index": order_index,
        "item": item_meta or {"index": int(item_index)},
        "status": item_status,
        "result": result,
        "trace": trace,
    }


@app.post("/api/pipe/encode/batch/jobs/{job_id}/cancel")
async def pipe_batch_encode_cancel_job(job_id: str):
    async with _batch_job_lock:
        await _batch_job_cleanup_locked()
        job = _batch_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        status = str(job.get("status", "") or "")
        if status == "queued":
            job["cancel_requested"] = True
            try:
                _batch_job_queue.remove(job_id)
            except ValueError:
                pass
            await _batch_job_mark_finished(job, "cancelled")
            snapshot = _batch_job_public(job)
        elif status in {"running", "cancelling"}:
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            job["updated_at"] = _utc_ts()
            snapshot = _batch_job_public(job)
        else:
            snapshot = _batch_job_public(job)
    await _batch_job_emit(job, {"type": "cancel_requested", "snapshot": snapshot})
    return {"success": True, "job": snapshot}


@app.get("/api/pipe/encode/batch/jobs/{job_id}/stream")
async def pipe_batch_encode_job_stream(job_id: str):
    async with _batch_job_lock:
        await _batch_job_cleanup_locked()
        job = _batch_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        subscriber_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=2000)
        job.setdefault("subscribers", []).append(subscriber_queue)
        snapshot = _batch_job_public(job)

    async def generate_sse():
        try:
            yield f"data: {json.dumps({'type': 'snapshot', 'snapshot': snapshot, 'job_id': job_id}, ensure_ascii=False)}\n\n"
            while True:
                event = await subscriber_queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in {"end", "cancelled", "failed"}:
                    break
        finally:
            async with _batch_job_lock:
                current_job = _batch_jobs.get(job_id)
                if current_job and subscriber_queue in current_job.get("subscribers", []):
                    current_job["subscribers"].remove(subscriber_queue)

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/pipe/encode/batch/stream")
async def pipe_batch_encode_stream(request: PipeBatchEncodeRequest):
    snapshot = await _batch_job_create(request)
    job_id = snapshot["job_id"]
    return await pipe_batch_encode_job_stream(job_id)


@app.get("/api/pipe/config")
async def get_pipe_config():
    """获取管道编码配置"""
    encoder = get_pipe_encoder()
    config_path = PROJECT_ROOT / "src" / "encoder" / "config" / "encoder_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return {
            "threshold": encoder.get_threshold(),
            "field_order": encoder.FIELD_ORDER,
            "config": config
        }
    except Exception as e:
        return {
            "threshold": encoder.get_threshold(),
            "field_order": encoder.FIELD_ORDER,
            "error": str(e)
        }


@app.post("/api/pipe/threshold")
async def set_pipe_threshold(threshold: float = Query(..., ge=0.0, le=1.0)):
    """设置相似度阈值"""
    encoder = get_pipe_encoder()
    encoder.set_threshold(threshold)
    return {"message": "阈值更新成功", "threshold": threshold}


@app.get("/api/pipe/mapping/{field_type}")
async def get_pipe_mapping(field_type: str):
    """获取管道材料指定字段的映射表"""
    valid_types = ['TYPE', 'MATERIAL', 'MANU', 'CONN', 'SEAL', 'ENDS']
    if field_type.upper() not in valid_types:
        raise HTTPException(status_code=400, detail=f"无效的字段类型，可选: {valid_types}")
    
    matcher = get_semantic_matcher()
    mapping_data = matcher.mapping.get(field_type.upper(), {})
    
    return {"field_type": field_type.upper(), "mapping": mapping_data}


@app.post("/api/pipe/mapping/reload")
async def reload_pipe_mapping():
    """重新加载管道编码映射表"""
    encoder = get_pipe_encoder()
    encoder.reload_mapping()
    return {"message": "映射表重新加载成功"}


# ============================================================
# 氚云导入API
# ============================================================

@app.post("/api/h3yun/import")
async def import_to_h3yun(request: H3yunImportRequest):
    """导入编码结果到氚云"""
    # 转换请求数据
    items = [
        {
            "description": item.description,
            "code": item.code,
            "type_raw": item.type_raw,
            "type_code": item.type_code,
            "size_raw": item.size_raw,
            "size_code": item.size_code,
            "thickness_raw": item.thickness_raw,
            "thickness_code": item.thickness_code,
            "pressure_raw": item.pressure_raw,
            "pressure_code": item.pressure_code,
            "material_raw": item.material_raw,
            "material_code": item.material_code,
            "standard_raw": item.standard_raw,
            "standard_code": item.standard_code,
        }
        for item in request.items
    ]
    
    # 调用氚云客户端
    client = get_h3yun_client()
    result = client.import_encodings(items, request.encode_date)
    
    return {
        "success": result.success,
        "message": result.message,
        "count": result.count,
        "ids": result.ids,
        "task_code": result.task_code
    }


class TaskListRequest(BaseModel):
    """任务列表请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("ReviewTaskListApiController", description="控制器名称")
    pageIndex: int = Field(1, ge=1, description="页码")
    pageSize: int = Field(20, ge=1, le=100, description="每页数量")
    filterTaskCode: Optional[str] = Field(None, description="任务编号筛选")
    filterReviewer: Optional[str] = Field(None, description="审核人筛选")
    filterCreatedTimeStart: Optional[str] = Field(None, description="创建时间开始")
    filterCreatedTimeEnd: Optional[str] = Field(None, description="创建时间结束")
    filterFeedbackTimeStart: Optional[str] = Field(None, description="反馈时间开始")
    filterFeedbackTimeEnd: Optional[str] = Field(None, description="反馈时间结束")


class TaskDetailRequest(BaseModel):
    """任务详情请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("MyApiController", description="控制器名称")
    taskCode: str = Field(..., description="任务编号")
    pageIndex: int = Field(1, ge=1, description="页码")
    pageSize: int = Field(50, ge=1, le=500, description="每页数量")
    sortField: str = Field("encodeDate", description="排序字段")
    sortOrder: str = Field("desc", description="排序方向")
    # 筛选条件
    filterDescription: Optional[str] = Field(None, description="描述筛选")
    filterCode: Optional[str] = Field(None, description="编码筛选")
    filterCorrectedCode: Optional[str] = Field(None, description="修正编码筛选")
    filterReasonCategory: Optional[str] = Field(None, description="原因分类筛选")
    filterIsResolved: Optional[str] = Field(None, description="是否解决筛选")
    filterVerifyDateStart: Optional[str] = Field(None, description="核对日期开始")
    filterVerifyDateEnd: Optional[str] = Field(None, description="核对日期结束")
    filterEncodeDateStart: Optional[str] = Field(None, description="编码日期开始")
    filterEncodeDateEnd: Optional[str] = Field(None, description="编码日期结束")


class TaskObjectDetailRequest(BaseModel):
    """按业务对象ID查询任务详情请求"""
    bizObjectId: str = Field(..., description="氚云业务对象ID")
    schemaCode: str = Field(H3YUN_REVIEW_TASK_SCHEMA_CODE, description="表单编码")


class ReviewCorrectionItemRequest(BaseModel):
    """审核修正项"""
    id: str = Field(..., description="子表行ObjectId")
    correctedCode: str = Field("", description="修正后编码")
    correctedType: str = Field("", description="修正种类")
    correctedSize: str = Field("", description="修正尺寸")
    correctedThickness: str = Field("", description="修正壁厚")
    correctedPressure: str = Field("", description="修正磅级")
    correctedMaterial: str = Field("", description="修正材质")
    correctedStandard: str = Field("", description="修正规范")


class TaskCorrectionWriteRequest(BaseModel):
    """审核修正写入请求"""
    bizObjectId: str = Field(..., description="主表业务对象ID")
    appCode: str = Field(H3YUN_REVIEW_TASK_APP_CODE, description="应用编码")
    controller: str = Field("ReviewTaskListApiController", description="控制器名称")
    items: List[ReviewCorrectionItemRequest] = Field(default_factory=list, description="需要写入的修正项")


class ReasonCategoryRequest(BaseModel):
    """原因分类请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("MyApiController", description="控制器名称")


@app.post("/api/h3yun/tasks")
async def get_h3yun_task_list(request: TaskListRequest):
    """获取氚云任务列表（代理接口）"""
    try:
        created_start = request.filterCreatedTimeStart
        created_end = request.filterCreatedTimeEnd
        feedback_start = request.filterFeedbackTimeStart
        feedback_end = request.filterFeedbackTimeEnd

        if created_start and not created_end:
            created_end = created_start
        elif created_end and not created_start:
            created_start = created_end

        if feedback_start and not feedback_end:
            feedback_end = feedback_start
        elif feedback_end and not feedback_start:
            feedback_start = feedback_end

        filters = {}
        if request.filterTaskCode:
            filters["filterTaskCode"] = request.filterTaskCode
        if request.filterReviewer:
            filters["filterReviewer"] = request.filterReviewer
        if created_start:
            filters["filterCreatedTimeStart"] = created_start
        if created_end:
            filters["filterCreatedTimeEnd"] = created_end
        if feedback_start:
            filters["filterFeedbackTimeStart"] = feedback_start
        if feedback_end:
            filters["filterFeedbackTimeEnd"] = feedback_end

        client = get_h3yun_client()
        result = client.get_task_list(
            app_code=request.appCode,
            controller=request.controller,
            page_index=request.pageIndex,
            page_size=request.pageSize,
            filters=filters if filters else None
        )

        for item in result.get("data", []) or []:
            review_date = (item.get("reviewDate") or "").strip()
            item["status"] = "已核对" if review_date else "待核对"

        return result
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/detail")
async def get_h3yun_task_detail(request: TaskDetailRequest):
    """获取氚云任务详情（代理接口，支持排序和筛选）"""
    try:
        # 构建筛选条件
        filters = {}
        if request.filterDescription:
            filters["filterDescription"] = request.filterDescription
        if request.filterCode:
            filters["filterCode"] = request.filterCode
        if request.filterCorrectedCode:
            filters["filterCorrectedCode"] = request.filterCorrectedCode
        if request.filterReasonCategory:
            filters["filterReasonCategory"] = request.filterReasonCategory
        if request.filterIsResolved:
            filters["filterIsResolved"] = request.filterIsResolved
        if request.filterVerifyDateStart:
            filters["filterVerifyDateStart"] = request.filterVerifyDateStart
        if request.filterVerifyDateEnd:
            filters["filterVerifyDateEnd"] = request.filterVerifyDateEnd
        if request.filterEncodeDateStart:
            filters["filterEncodeDateStart"] = request.filterEncodeDateStart
        if request.filterEncodeDateEnd:
            filters["filterEncodeDateEnd"] = request.filterEncodeDateEnd
        
        client = get_h3yun_client()
        result = client.get_task_detail(
            task_code=request.taskCode,
            app_code=request.appCode,
            controller=request.controller,
            page_index=request.pageIndex,
            page_size=request.pageSize,
            sort_field=request.sortField,
            sort_order=request.sortOrder,
            filters=filters if filters else None
        )
        return result
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/object-detail")
async def get_h3yun_task_object_detail(request: TaskObjectDetailRequest):
    """按BizObjectId获取审核任务主表及子表详情"""
    try:
        client = get_h3yun_client()
        biz_object = client.load_biz_object(
            schema_code=request.schemaCode,
            biz_object_id=request.bizObjectId
        )

        sub_items = biz_object.get(H3YUN_REVIEW_TASK_SUBTABLE_CODE, []) or []
        items = []
        for row in sub_items:
            if not isinstance(row, dict):
                continue
            items.append({
                "id": row.get("ObjectId", "") or "",
                "name": row.get("Name", "") or "",
                "description": row.get("F0000003", "") or "",
                "code": row.get("F0000004", "") or "",
                "correctedCode": row.get("F0000008", "") or "",
                "typeRaw": row.get("F0000009", "") or "",
                "typeCode": row.get("F0000010", "") or "",
                "correctedType": row.get("F0000011", "") or "",
                "sizeRaw": row.get("F0000012", "") or "",
                "sizeCode": row.get("F0000013", "") or "",
                "correctedSize": row.get("F0000014", "") or "",
                "thicknessRaw": row.get("F0000015", "") or "",
                "thicknessCode": row.get("F0000016", "") or "",
                "correctedThickness": row.get("F0000017", "") or "",
                "pressureRaw": row.get("F0000018", "") or "",
                "pressureCode": row.get("F0000019", "") or "",
                "correctedPressure": row.get("F0000020", "") or "",
                "materialRaw": row.get("F0000021", "") or "",
                "materialCode": row.get("F0000022", "") or "",
                "correctedMaterial": row.get("F0000023", "") or "",
                "standardRaw": row.get("F0000024", "") or "",
                "standardCode": row.get("F0000025", "") or "",
                "correctedStandard": row.get("F0000026", "") or "",
            })

        review_date = (biz_object.get("F0000043", "") or "").strip()

        return {
            "Success": True,
            "data": {
                "id": biz_object.get("ObjectId", "") or "",
                "taskCode": biz_object.get("SeqNo", "") or "",
                "name": biz_object.get("Name", "") or "",
                "reviewer": biz_object.get("F0000042", "") or "",
                "reviewerId": biz_object.get("F0000048", "") or "",
                "reviewDate": review_date,
                "creator": biz_object.get("CreatedBy", "") or "",
                "createdTime": biz_object.get("CreatedTime", "") or "",
                "modifiedBy": biz_object.get("ModifiedBy", "") or "",
                "modifiedTime": biz_object.get("ModifiedTime", "") or "",
                "status": "已核对" if review_date else "待核对",
                "items": items,
            }
        }
    except Exception as e:
        logger.error(f"按对象ID获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/write-corrections")
async def write_h3yun_task_corrections(request: TaskCorrectionWriteRequest):
    """批量写入审核修正结果到氚云"""
    try:
        client = get_h3yun_client()
        result = client.save_review_task_corrections(
            app_code=request.appCode,
            controller=request.controller,
            biz_object_id=request.bizObjectId,
            items=[item.dict() for item in request.items]
        )
        return {
            "Success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"写入审核修正结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/reason-categories")
async def get_h3yun_reason_categories(request: ReasonCategoryRequest):
    """获取原因分类列表"""
    try:
        client = get_h3yun_client()
        result = client.get_reason_categories(
            app_code=request.appCode,
            controller=request.controller
        )
        return {"Success": True, "data": result}
    except Exception as e:
        logger.error(f"获取原因分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 辅助函数
# ============================================================

def _convert_pipe_result(result) -> dict:
    """转换管道材料编码结果为响应格式"""
    fields = {}
    for field_type, field_data in result.fields.items():
        fields[field_type] = {
            "field_type": field_data.field_type,
            "original_value": field_data.original_value,
            "stage1_final_value": getattr(field_data, "stage1_final_value", ""),
            "stage1_confidence": getattr(field_data, "stage1_confidence", None),
            "stage2_confidence": getattr(field_data, "stage2_confidence", None),
            "field_confidence": getattr(field_data, "field_confidence", None),
            "matched_name": field_data.matched_name,
            "encoding_input": getattr(field_data, "encoding_input", ""),
            "encode_confidence_v2": getattr(field_data, "encode_confidence_v2", {}) or {},
            "code": field_data.code,
            "similarity": round(field_data.similarity, 4),
            "is_exact_match": field_data.is_exact_match,
            "need_review": field_data.need_review,
            "candidates": field_data.candidates,
            "display": field_data.display or "",  # 分类显示信息
            "items": field_data.items or []  # 多值分行显示
        }
    
    return {
        "original_text": result.original_text,
        "final_code": result.final_code,
        "success": result.success,
        "need_review": result.need_review,
        "hard_rule_hit": getattr(result, "hard_rule_hit", False),
        "confidence": round(getattr(result, "confidence", 0.0), 4),
        "min_similarity": round(result.min_similarity, 4),
        "review_fields": result.review_fields,
        "missing_fields": result.missing_fields,
        "errors": result.errors,
        "warnings": result.warnings,
        "thickness_conversion_notes": getattr(result, "thickness_conversion_notes", []),
        "extract_confidence_v2": getattr(result, "extract_confidence_v2", {}) or {},
        "fields": fields
    }


def _extract_code_from_field(result_dict: Dict[str, Any], field: str) -> str:
    fields = result_dict.get("fields", {}) if isinstance(result_dict, dict) else {}
    field_data = fields.get(field, {}) if isinstance(fields, dict) else {}
    return str(field_data.get("code", "") or "").strip()


def _extract_standard_codes_from_field(result_dict: Dict[str, Any]) -> List[str]:
    fields = result_dict.get("fields", {}) if isinstance(result_dict, dict) else {}
    field_data = fields.get("STANDARD", {}) if isinstance(fields, dict) else {}
    items = field_data.get("items", []) if isinstance(field_data, dict) else []
    codes: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "") or "").strip()
        if code:
            codes.append(code)
    if codes:
        return codes
    fallback = str(field_data.get("code", "") or "").strip() if isinstance(field_data, dict) else ""
    return [fallback] if fallback else []


def _attach_base_difficulty(result_dict: Dict[str, Any], project_name: str = "") -> Dict[str, Any]:
    difficulty = build_base_difficulty(
        result_dict.get("original_text", "") or "",
        type_code=_extract_code_from_field(result_dict, "TYPE"),
        material_code=_extract_code_from_field(result_dict, "MATERIAL"),
        standard_code=_extract_code_from_field(result_dict, "STANDARD"),
        standard_codes=_extract_standard_codes_from_field(result_dict),
    )
    difficulty["project_name"] = str(project_name or "").strip()
    result_dict["difficulty_split"] = difficulty
    return result_dict


def _attach_second_pass(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result_dict, dict):
        return result_dict

    difficulty_info = result_dict.get("difficulty_split") if isinstance(result_dict.get("difficulty_split"), dict) else {}
    payload = {
        "text": result_dict.get("original_text", "") or "",
        "stage1_difficulty": difficulty_info.get("difficulty"),
        "fields": result_dict.get("fields", {}) if isinstance(result_dict.get("fields"), dict) else {},
    }
    try:
        result_dict["second_pass"] = _second_pass_runner.analyze_payload(payload)
    except Exception as exc:
        logger.exception("二次校验执行失败: %s", exc)
        result_dict["second_pass"] = {
            "final_level": None,
            "results": {},
            "skipped_fields": {},
            "error": str(exc),
        }
    return result_dict


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="材料智能处理平台")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    logger.info(f"启动材料智能处理平台: http://{args.host}:{args.port}")
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
