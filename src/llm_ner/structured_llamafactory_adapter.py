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

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)

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


def _build_raw_chatml_prompt(input_text: str) -> str:
    return (
        f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_chat_prompt(tokenizer: Any, input_text: str) -> str:
    return _build_raw_chatml_prompt(input_text)


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
    ):
        self.backend = backend
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.service_url = service_url
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
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
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(device: str):
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
        raw = self._generate(text)
        elapsed = time.perf_counter() - started
        logger.debug("[结构化适配器] 推理耗时: %.2fs", elapsed)

        parsed = self._parse_json_output(raw)
        if parsed is None:
            logger.warning("[结构化适配器] JSON解析失败")
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

        structured = self._normalize_model_output(parsed)
        type_class = self._infer_type_class(structured)
        extract_confidence = {
            field: 1.0 for field, value in structured.items()
            if field and value not in (None, "", [], {})
        }
        extract_confidence_v2 = self._build_extract_confidence_v2(structured)

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

    def _generate(self, input_text: str) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(input_text)
        if self.backend == "hf_lazy_service":
            return self._generate_hf_lazy_service(input_text)

        text = _build_chat_prompt(self.tokenizer, input_text)
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
            outputs = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_ollama(self, input_text: str) -> str:
        prompt = _build_raw_chatml_prompt(input_text)
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        return str(payload.get("response", "")).strip()

    def _generate_hf_lazy_service(self, input_text: str) -> str:
        resp = requests.post(
            f"{self.service_url.rstrip('/')}/predict",
            json={
                "model": self.model_name,
                "text": input_text,
                "instruction": INSTRUCTION,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": 0.9 if self.temperature > 0 else 1.0,
            },
            timeout=180,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw = payload.get("raw_response")
        parsed = payload.get("parsed_json")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return ""

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

    def _build_extract_confidence_v2(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        type_value = structured.get("TYPE")
        type_present = _type_present(type_value)
        type_body_present = isinstance(type_value, dict) and _is_non_empty(type_value.get("BODY"))
        type_aux_count = 0
        if isinstance(type_value, dict):
            for key in ("MANU", "CONN", "SEAL", "ENDS"):
                value = type_value.get(key)
                if isinstance(value, list) and any(_is_non_empty(item) for item in value):
                    type_aux_count += 1
            geometry = type_value.get("GEOMETRY")
            if isinstance(geometry, dict) and any(_is_non_empty(v) for v in geometry.values()):
                type_aux_count += 1
        type_conf = 0.0
        if type_present:
            type_conf = min(0.95, 0.55 + (0.20 if isinstance(type_value, dict) else 0.0) + (0.15 if type_body_present else 0.0) + min(type_aux_count, 2) * 0.05)
        result["TYPE"] = {
            "source": "finetuned_model",
            "confidence": round(type_conf, 4),
            "reason": "field_present_and_schema_valid" if type_present else "field_missing",
            "evidence": {
                "field_present": type_present,
                "structure_valid": isinstance(type_value, dict),
                "body_present": bool(type_body_present),
                "aux_signal_count": type_aux_count,
            },
        }

        material_value = structured.get("MATERIAL")
        material_items = material_value if isinstance(material_value, list) else []
        material_valid = sum(1 for item in material_items if isinstance(item, dict) and _is_non_empty(item.get("VALUE")))
        material_roles = sum(1 for item in material_items if isinstance(item, dict) and _is_non_empty(item.get("ROLE")))
        material_present = material_valid > 0
        material_ratio = (material_valid / len(material_items)) if material_items else 0.0
        material_conf = 0.0
        if material_present:
            material_conf = min(0.95, 0.58 + 0.22 * material_ratio + (0.10 if material_roles == material_valid else 0.0) + (0.05 if material_valid > 1 else 0.0))
        result["MATERIAL"] = {
            "source": "finetuned_model",
            "confidence": round(material_conf, 4),
            "reason": "field_present_and_items_valid" if material_present else "field_missing",
            "evidence": {
                "field_present": material_present,
                "item_count": len(material_items),
                "valid_item_count": material_valid,
                "role_present_count": material_roles,
                "valid_ratio": round(material_ratio, 4),
            },
        }

        standard_value = structured.get("STANDARD")
        standard_items = standard_value if isinstance(standard_value, list) else []
        standard_valid = sum(1 for item in standard_items if isinstance(item, dict) and _is_non_empty(item.get("BODY")))
        standard_present = standard_valid > 0
        standard_ratio = (standard_valid / len(standard_items)) if standard_items else 0.0
        standard_conf = 0.0
        if standard_present:
            standard_conf = min(0.95, 0.58 + 0.24 * standard_ratio + (0.08 if standard_valid > 1 else 0.0) + (0.05 if all(isinstance(item, dict) for item in standard_items) else 0.0))
        result["STANDARD"] = {
            "source": "finetuned_model",
            "confidence": round(standard_conf, 4),
            "reason": "field_present_and_items_valid" if standard_present else "field_missing",
            "evidence": {
                "field_present": standard_present,
                "item_count": len(standard_items),
                "valid_item_count": standard_valid,
                "valid_ratio": round(standard_ratio, 4),
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
