# -*- coding: utf-8 -*-
"""
LlamaFactory 一阶段结构化输出适配器。

适用于：
- merge 后的 HuggingFace 模型目录
- 模型直接输出顶层结构化 JSON
- 不再依赖 mentions / semantics / decisions 旧协议
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from .llm_http_transport import (
    call_hf_lazy_service_predict,
    call_ollama_generate,
)
from ..confidence.model_token_confidence import build_field_token_confidences

INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)


def build_structured_predictor_from_config(
    config: Dict[str, Any],
    *,
    default_instruction: str | None = None,
    model_base_dir: str | Path | None = None,
    log_label: str = "结构化模型",
) -> "StructuredLlamaFactoryPredictor":
    cfg = dict(config or {})
    backend = str(cfg.get("backend", "")).strip()
    if backend not in {"transformers", "hf_lazy_service", "ollama"}:
        raise RuntimeError(f"{log_label}后端不支持: {backend}")

    max_new_tokens = int(
        cfg.get("max_new_tokens")
        or cfg.get("num_predict")
        or cfg.get("max_tokens")
        or 512
    )
    temperature = float(cfg.get("temperature", 0.0) or 0.0)
    instruction = str(cfg.get("instruction") or default_instruction or INSTRUCTION)

    predictor_kwargs: Dict[str, Any] = {
        "backend": backend,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "instruction": instruction,
    }

    if backend == "hf_lazy_service":
        service_url = str(cfg.get("service_url", "")).strip()
        model_name = str(cfg.get("model_name", "")).strip()
        if not service_url:
            raise RuntimeError(f"{log_label}缺少配置: service_url")
        if not model_name:
            raise RuntimeError(f"{log_label}缺少配置: model_name")
        predictor_kwargs["service_url"] = service_url
        predictor_kwargs["model_name"] = model_name
        logger.info(
            "[%s] HF Lazy Service 后端, 模型: %s, 服务: %s",
            log_label,
            model_name,
            service_url,
        )
        return StructuredLlamaFactoryPredictor(**predictor_kwargs)

    if backend == "ollama":
        ollama_url = str(cfg.get("ollama_url") or cfg.get("base_url") or "").strip()
        model_name = str(cfg.get("model_name", "")).strip()
        if not ollama_url:
            raise RuntimeError(f"{log_label}缺少配置: ollama_url/base_url")
        if not model_name:
            raise RuntimeError(f"{log_label}缺少配置: model_name")
        predictor_kwargs["ollama_url"] = ollama_url
        predictor_kwargs["model_name"] = model_name
        predictor_kwargs["device"] = str(cfg.get("device") or "ollama")
        logger.info(
            "[%s] Ollama 后端, 模型: %s, 服务: %s",
            log_label,
            model_name,
            ollama_url,
        )
        return StructuredLlamaFactoryPredictor(**predictor_kwargs)

    configured_model_path = str(cfg.get("model_path", "")).strip()
    device = str(cfg.get("device", "")).strip()
    if not configured_model_path:
        raise RuntimeError(f"{log_label}缺少配置: model_path")
    if not device:
        raise RuntimeError(f"{log_label}缺少配置: device")

    model_path = Path(configured_model_path)
    if not model_path.is_absolute():
        base_dir = Path(model_base_dir) if model_base_dir is not None else Path(__file__).resolve().parents[2]
        model_path = base_dir / configured_model_path

    predictor_kwargs["model_path"] = str(model_path)
    predictor_kwargs["device"] = device
    logger.info(
        "[%s] Transformers 后端, 模型: %s, 设备: %s",
        log_label,
        model_path,
        device,
    )
    return StructuredLlamaFactoryPredictor(**predictor_kwargs)

TYPE_CLASS_MAP = {
    "管": "管子", "pipe": "管子", "tube": "管子",
    "弯头": "管件", "三通": "管件", "四通": "管件", "异径": "管件",
    "大小头": "管件", "管帽": "管件", "封头": "管件", "接头": "管件",
    "elbow": "管件", "tee": "管件", "reducer": "管件", "cap": "管件",
    "法兰": "法兰", "flange": "法兰",
    "螺栓": "螺栓", "螺母": "螺栓", "螺柱": "螺栓", "bolt": "螺栓",
    "nut": "螺栓", "stud": "螺栓",
    "阀": "阀门", "valve": "阀门",
    "垫片": "垫片", "垫圈": "垫片", "gasket": "垫片",
}


def _is_non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _type_present(type_value: Any) -> bool:
    if not isinstance(type_value, dict):
        return False
    return any(_is_non_empty(v) for v in type_value.values())


def _material_present(material_value: Any) -> bool:
    return isinstance(material_value, list) and any(isinstance(item, dict) and _is_non_empty(item.get("VALUE")) for item in material_value)


def _standard_present(standard_value: Any) -> bool:
    return isinstance(standard_value, list) and any(isinstance(item, dict) and _is_non_empty(item.get("BODY")) for item in standard_value)


def _build_raw_chatml_prompt(input_text: str, instruction: str) -> str:
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_chat_prompt(tokenizer: Any, input_text: str, instruction: str) -> str:
    return _build_raw_chatml_prompt(input_text, instruction)


class StructuredLlamaFactoryPredictor:
    def __init__(
        self,
        *,
        backend: str,
        model_path: str | None = None,
        model_name: str | None = None,
        ollama_url: str | None = None,
        service_url: str | None = None,
        device: str | None = None,
        max_new_tokens: int,
        temperature: float,
        type_value_whitelist: Optional[Dict[str, list[str]]] = None,
        instruction: str | None = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.service_url = service_url
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.instruction = str(instruction or INSTRUCTION)
        self.type_value_whitelist = {
            str(k).strip().upper(): {
                str(v).strip().upper() for v in (values or []) if str(v).strip()
            }
            for k, values in (type_value_whitelist or {}).items()
        }
        self.model = None
        self.tokenizer = None

        if backend == "ollama":
            if not self.model_name:
                raise ValueError("structured_llamafactory + ollama 必须配置 model_name")
            if not self.ollama_url:
                raise ValueError("structured_llamafactory + ollama 必须配置 ollama_url")
            self.model_path = None
            self.device = "ollama"
            logger.info(
                f"[结构化适配器] Ollama 后端, 模型: {self.model_name}, 服务: {self.ollama_url}"
            )
            return

        if backend == "hf_lazy_service":
            if not self.model_name:
                raise ValueError("structured_llamafactory + hf_lazy_service 必须配置 model_name")
            if not self.service_url:
                raise ValueError("structured_llamafactory + hf_lazy_service 必须配置 service_url")
            self.model_path = None
            self.device = "hf_lazy_service"
            logger.info(
                f"[结构化适配器] HF Lazy Service 后端, 模型: {self.model_name}, 服务: {self.service_url}"
            )
            return

        if backend != "transformers":
            raise ValueError(f"不支持的 backend: {backend}")
        if model_path is None:
            raise ValueError("transformers 后端必须提供 model_path")
        if not device:
            raise ValueError("structured_llamafactory + transformers 必须配置 device")

        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        logger.info(
            f"[结构化适配器] Transformers 后端, 模型: {self.model_path}, 设备: {self.device}"
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = self._resolve_dtype(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"[结构化适配器] 模型加载完成, 设备: {self.model.device}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(device: str):
        import torch

        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "text": text,
                "tokens": [],
                "entities": [],
                "type_class": None,
                "model_output": {},
                "extract_confidence": {},
                "extract_confidence_v2": {},
                "model_raw_response": "",
            }

        started = time.perf_counter()
        raw, token_logprobs = self._generate(text)
        elapsed = time.perf_counter() - started
        logger.debug("[结构化适配器] 推理耗时: %.2fs", elapsed)

        parsed = self._parse_json_output(raw)
        if parsed is None:
            logger.warning(
                "[结构化适配器] JSON解析失败, raw_len=%s, raw_preview=%r",
                len(str(raw or "")),
                str(raw or "")[:500],
            )
            return {
                "text": text,
                "tokens": [],
                "entities": [],
                "type_class": None,
                "model_output": {},
                "extract_confidence": {},
                "extract_confidence_v2": {},
                "model_raw_response": raw,
            }
        logger.debug("[结构化适配器] JSON解析成功, keys=%s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))

        structured = self._normalize_model_output(parsed)
        type_class = self._infer_type_class(structured)
        field_scores = build_field_token_confidences(raw, structured, token_logprobs)
        extract_confidence = {
            field: float(payload["confidence"])
            for field, payload in field_scores.items()
        }
        extract_confidence_v2 = self._build_extract_confidence_v2(structured, field_scores)

        return {
            "text": text,
            "tokens": [],
            "entities": [],
            "type_class": type_class,
            "model_output": structured,
            "extract_confidence": extract_confidence,
            "extract_confidence_v2": extract_confidence_v2,
            "model_raw_response": raw,
        }

    def _generate(self, input_text: str) -> tuple[str, list[dict[str, Any]]]:
        if self.backend == "ollama":
            return self._generate_ollama(input_text)
        if self.backend == "hf_lazy_service":
            return self._generate_hf_lazy_service(input_text)

        import torch

        text = _build_chat_prompt(self.tokenizer, input_text, self.instruction)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 0.9

        bad_words_ids = []
        for marker in ("<think>", "</think>"):
            token_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if token_ids:
                bad_words_ids.append(token_ids)
        if bad_words_ids:
            generate_kwargs["bad_words_ids"] = bad_words_ids

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        raw_unstripped = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        token_logprobs = self._build_transformers_token_logprobs(new_tokens, outputs.scores)
        return self._strip_response_with_offsets(raw_unstripped, token_logprobs)

    def _generate_ollama(self, input_text: str) -> tuple[str, list[dict[str, Any]]]:
        prompt = _build_raw_chatml_prompt(input_text, self.instruction)
        response = call_ollama_generate(
            base_url=self.ollama_url,
            model_name=self.model_name,
            prompt=prompt,
            timeout=120,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            raw=True,
            format_json=True,
        )
        return response.text, []

    def _generate_hf_lazy_service(self, input_text: str) -> tuple[str, list[dict[str, Any]]]:
        response = call_hf_lazy_service_predict(
            service_url=self.service_url,
            model_name=self.model_name,
            text=input_text,
            timeout=180,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=0.9 if self.temperature > 0 else 1.0,
        )
        return response.text, list(response.payload.get("token_logprobs") or [])

    def _build_transformers_token_logprobs(
        self,
        generated: Any,
        scores: Any,
    ) -> list[dict[str, Any]]:
        import torch

        records: list[dict[str, Any]] = []
        previous = ""
        max_steps = min(len(generated), len(scores or []))
        for index in range(max_steps):
            token_id = int(generated[index].item())
            decoded = self.tokenizer.decode(generated[: index + 1], skip_special_tokens=True)
            if decoded.startswith(previous):
                piece = decoded[len(previous):]
                start = len(previous)
            else:
                piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
                start = len(previous)
                decoded = previous + piece
            if piece:
                logprob = float(torch.log_softmax(scores[index][0], dim=-1)[token_id].item())
                records.append({
                    "token": piece,
                    "logprob": logprob,
                    "start": start,
                    "end": start + len(piece),
                })
            previous = decoded
        return records

    @staticmethod
    def _strip_response_with_offsets(
        raw: str,
        records: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        stripped = raw.strip()
        if stripped == raw:
            return raw, records
        left = len(raw) - len(raw.lstrip())
        adjusted: list[dict[str, Any]] = []
        for item in records:
            start = max(0, int(item["start"]) - left)
            end = min(len(stripped), int(item["end"]) - left)
            if end > start:
                adjusted.append({**item, "token": stripped[start:end], "start": start, "end": end})
        return stripped, adjusted

    @staticmethod
    def _parse_json_output(raw: str) -> Optional[dict]:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

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

    def _normalize_model_output(self, parsed: dict) -> dict:
        if not isinstance(parsed, dict):
            return {}
        if isinstance(parsed.get("decisions"), dict):
            structured = parsed["decisions"]
        else:
            structured = {
                k: v for k, v in parsed.items()
                if not str(k).startswith("_")
            }
        return self._apply_type_value_whitelist(structured)

    def _build_extract_confidence_v2(
        self,
        structured: Dict[str, Any],
        field_scores: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field, value in structured.items():
            if str(field).startswith("_"):
                continue
            score = field_scores.get(str(field))
            present = value not in (None, "", [], {})
            result[str(field)] = {
                "source": "model_token_logprobs" if score else "model_token_logprobs_unavailable",
                "confidence": round(float(score["confidence"]), 6) if score else None,
                "reason": "generated_value_token_probability" if score else (
                    "field_missing" if not present else "token_logprobs_unavailable"
                ),
                "evidence": {
                    "field_present": present,
                    "token_count": int(score["token_count"]) if score else 0,
                    "mean_logprob": round(float(score["mean_logprob"]), 6) if score else None,
                    "min_probability": round(float(score["min_probability"]), 6) if score else None,
                },
            }
        return result

    def _apply_type_value_whitelist(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        if not self.type_value_whitelist:
            return structured
        type_dict = structured.get("TYPE")
        if not isinstance(type_dict, dict):
            return structured

        for subtype, allowed in self.type_value_whitelist.items():
            if not allowed:
                continue
            raw_value = type_dict.get(subtype)
            if raw_value in (None, "", []):
                continue
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            filtered = [
                str(v).strip()
                for v in values
                if str(v).strip() and str(v).strip().upper() in allowed
            ]
            type_dict[subtype] = filtered
        return structured

    @staticmethod
    def _merge_type_geometry_into_body(structured: Dict[str, Any]) -> Dict[str, Any]:
        type_dict = structured.get("TYPE")
        if not isinstance(type_dict, dict):
            return structured

        geometry = type_dict.get("GEOMETRY")
        if not isinstance(geometry, dict):
            return structured

        body = str(type_dict.get("BODY") or "").strip()
        angle = str(geometry.get("ANGLE") or "").strip()
        radius = str(geometry.get("RADIUS") or "").strip()

        parts = []
        if angle:
            parts.append(angle)
        if body:
            parts.append(body)
        if radius:
            parts.append(radius)

        if parts:
            type_dict["BODY"] = ";".join(parts)
        type_dict.pop("GEOMETRY", None)
        return structured

    @staticmethod
    def _infer_type_class(structured: Dict[str, Any]) -> Optional[str]:
        type_val = structured.get("TYPE")
        if not type_val:
            return None
        if isinstance(type_val, dict):
            type_val = type_val.get("BODY") or ""
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else ""
        type_lower = str(type_val).lower()
        for keyword, cls in TYPE_CLASS_MAP.items():
            if keyword in type_lower:
                return cls
        return None
