# -*- coding: utf-8 -*-
"""
Qwen3 语义解析预测器。

当前一阶段统一为：
原始描述 -> {mentions, semantics, decisions}
"""

from __future__ import annotations

import json
import logging
import math
import copy
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from .prompts import (
    get_stage1_decisions_only_prompt,
    get_stage1_platform_predict_prompt,
    get_stage2_platform_predict_prompt,
)

logger = logging.getLogger(__name__)
STAGE2_SYSTEM_PROMPT = get_stage2_platform_predict_prompt()

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


@lru_cache(maxsize=1)
def _load_encoder_config() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "encoder" / "config" / "encoder_config.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("[Qwen3预测器] 加载 encoder_config 失败: %s", e)
        return {}


@lru_cache(maxsize=1)
def _load_material_special_req_supplement_map() -> Dict[str, List[str]]:
    cfg = _load_encoder_config()
    raw = (
        (cfg.get("material_special_req_supplement") or {}).get("suffix_aliases") or {}
    )
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for suffix, aliases in raw.items():
        suffix_text = str(suffix or "").strip().upper()
        if not suffix_text or not isinstance(aliases, list):
            continue
        clean_aliases = [str(alias or "").strip() for alias in aliases if str(alias or "").strip()]
        if clean_aliases:
            result[suffix_text] = clean_aliases
    return result


@lru_cache(maxsize=1)
def _load_material_special_req_reverse_map() -> Dict[str, str]:
    reverse: Dict[str, str] = {}
    for suffix, aliases in _load_material_special_req_supplement_map().items():
        reverse[suffix.upper()] = suffix.upper()
        for alias in aliases:
            reverse[str(alias).strip().upper()] = suffix.upper()
    return reverse


class Qwen3Predictor:
    """Qwen3 一阶段语义解析预测器。"""

    ENCODABLE_FIELDS = {"TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD"}

    SEMANTIC_TO_FIELD = {
        "TYPE_BODY": "TYPE",
        "TYPE_ANGLE": "TYPE",
        "TYPE_RADIUS": "TYPE",
        "TYPE_MANU": "TYPE",
        "TYPE_CONN": "TYPE",
        "TYPE_ENDS": "TYPE",
        "TYPE_SEAL": "TYPE",
        "SIZE_DN": "SIZE",
        "SIZE_DN_PAIR": "SIZE",
        "SIZE_OD": "SIZE",
        "SIZE_OD_PAIR": "SIZE",
        "SIZE_INCH": "SIZE",
        "SIZE_INCH_PAIR": "SIZE",
        "SIZE_LENGTH": "SIZE",
        "PRESSURE_CLASS": "PRESSURE",
        "THICKNESS_MM": "THICKNESS",
        "THICKNESS_MM_PAIR": "THICKNESS",
        "THICKNESS_INCH": "THICKNESS",
        "THICKNESS_SCHEDULE": "THICKNESS",
        "THICKNESS_SCHEDULE_PAIR": "THICKNESS",
        "THICKNESS_SERIES": "THICKNESS",
        "THICKNESS_BWG": "THICKNESS",
        "MATERIAL_EXEC_STANDARD": "MATERIAL",
        "MATERIAL_GRADE": "MATERIAL",
        "MATERIAL_SPECIAL_REQ": "MATERIAL",
        "STANDARD_BODY": "STANDARD",
        "STANDARD_GRADE": "STANDARD",
        "STANDARD_APPENDIX": "STANDARD",
        "STANDARD_METHOD": "STANDARD",
        "STANDARD_COMPOSITE": "STANDARD",
    }

    MENTION_TYPE_TO_FIELD = {
        "TYPE_TERM": "TYPE",
        "SIZE_TERM": "SIZE",
        "PRESSURE_TERM": "PRESSURE",
        "THICKNESS_TERM": "THICKNESS",
        "MATERIAL_TERM": "MATERIAL",
        "SPECIAL_REQ_TERM": "MATERIAL",
        "STANDARD_TERM": "STANDARD",
    }

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str = "qwen3-pipe",
        backend: str = "ollama",
        device: str = "auto",
        ollama_url: str = "http://localhost:11434",
        output_mode: str = "full",
        ollama_num_predict: int = 768,
        ollama_temperature: float = 0.1,
        ollama_top_p: float = 0.9,
        ollama_logprobs_enabled: bool = True,
        type_value_whitelist: Optional[Dict[str, List[str]]] = None,
    ):
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self.output_mode = output_mode if output_mode in {"full", "decisions_only"} else "full"
        self.ollama_num_predict = int(ollama_num_predict)
        self.ollama_temperature = float(ollama_temperature)
        self.ollama_top_p = float(ollama_top_p)
        self.ollama_logprobs_enabled = bool(ollama_logprobs_enabled)
        self.type_value_whitelist = {
            str(k).strip().upper(): {
                str(v).strip().upper() for v in (values or []) if str(v).strip()
            }
            for k, values in (type_value_whitelist or {}).items()
        }
        self.stage1_system_prompt = (
            get_stage1_decisions_only_prompt()
            if self.output_mode == "decisions_only"
            else get_stage1_platform_predict_prompt()
        )

        if backend == "ollama":
            self.ollama_url = ollama_url
            self.model_name = model_name
            logger.info(f"[Qwen3预测器] Ollama 后端, 模型: {model_name}, 输出模式: {self.output_mode}")
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model_path = Path(model_path) if model_path is not None else None
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "mps":
                logger.warning("[Qwen3预测器] MPS 与 Qwen3 存在兼容性问题，自动回退到 CPU")
                device = "cpu"
            self.device = device

            logger.info(f"[Qwen3预测器] Transformers 后端, 模型: {self.model_path}, 设备: {device}, 输出模式: {self.output_mode}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                dtype=dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self.model = self.model.to(device)
            self.model.eval()
            logger.info("[Qwen3预测器] 模型加载完成")

    def predict(self, text: str) -> Dict[str, Any]:
        """返回平台仍可消费的 tokens/entities/type_class，同时保留 model_output。"""
        if not text or not text.strip():
            return {
                "text": text,
                "tokens": [],
                "entities": [],
                "type_class": None,
                "model_output": {},
                "extract_confidence": {},
                "model_raw_response": "",
            }

        start_time = time.perf_counter()
        extracted = self._extract(text)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"[Qwen3预测器] 推理耗时: {elapsed:.2f}s")

        if extracted.get("_parse_error"):
            logger.warning(f"[Qwen3预测器] JSON解析失败: {extracted.get('_raw', '')[:200]}")
            return self._fallback_result(text)

        model_conf = extracted.get("_model_confidence")
        if model_conf is None and self.ollama_logprobs_enabled:
            raise RuntimeError("模型未返回 logprobs，无法计算置信度（严格模式）")
        if model_conf is None:
            model_conf = 1.0

        decisions = extracted.get("decisions")
        if not isinstance(decisions, dict):
            raise RuntimeError("语义解析模型输出缺少 decisions 对象")
        self._apply_type_value_whitelist(decisions)
        self._apply_material_special_req_supplement(text, decisions)

        default_conf = float(model_conf)
        tokens = self._build_tokens(text, extracted, default_confidence=default_conf)
        entities = self._build_entities(decisions, default_confidence=default_conf)
        extract_confidence = {field: default_conf for field in self._iter_decision_fields(decisions)}
        type_class = self._infer_type_class(decisions)

        return {
            "text": text,
            "tokens": tokens,
            "entities": entities,
            "type_class": type_class,
            "model_output": extracted,
            "extract_confidence": extract_confidence,
            "model_raw_response": extracted.get("_raw", ""),
        }

    def _apply_type_value_whitelist(self, decisions: Dict[str, Any]) -> None:
        if not self.type_value_whitelist:
            return
        type_dict = decisions.get("TYPE")
        if not isinstance(type_dict, dict):
            return

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

    def _apply_material_special_req_supplement(self, text: str, decisions: Dict[str, Any]) -> None:
        supplement_cfg = _load_encoder_config().get("material_special_req_supplement") or {}
        if not supplement_cfg.get("enabled", False):
            return

        material_value = decisions.get("MATERIAL")
        if material_value in (None, "", [], {}):
            return

        suffix_aliases = _load_material_special_req_supplement_map()
        if not suffix_aliases:
            return

        matched_suffixes: List[str] = []
        upper_text = str(text or "").upper()
        for suffix, aliases in suffix_aliases.items():
            for alias in aliases:
                alias_upper = alias.upper()
                if self._contains_material_special_req_alias(upper_text, alias_upper):
                    matched_suffixes.append(suffix)
                    break
        if not matched_suffixes:
            return

        if isinstance(material_value, list):
            for item in material_value:
                if not isinstance(item, dict):
                    continue
                self._supplement_material_item_special_req(item, matched_suffixes)
            return

        if isinstance(material_value, dict):
            if "VALUE" in material_value or "ROLE" in material_value:
                self._supplement_material_item_special_req(material_value, matched_suffixes)
                return
            items = material_value.get("ITEMS")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        self._supplement_material_item_special_req(item, matched_suffixes, value_key="EXEC_STANDARD")

    @staticmethod
    def _contains_material_special_req_alias(upper_text: str, alias_upper: str) -> bool:
        if not upper_text or not alias_upper:
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in alias_upper):
            return alias_upper in upper_text
        pattern = re.compile(rf'(?<![A-Z0-9]){re.escape(alias_upper)}(?![A-Z0-9])', re.IGNORECASE)
        return bool(pattern.search(upper_text))

    @staticmethod
    def _normalize_special_req_values(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip().upper() for v in value if str(v).strip()]
        if value in (None, "", []):
            return []
        return [str(value).strip().upper()]

    def _supplement_material_item_special_req(
        self,
        item: Dict[str, Any],
        matched_suffixes: List[str],
        *,
        value_key: str = "VALUE",
    ) -> None:
        value = str(item.get(value_key) or "").strip()
        if not value:
            return

        reverse_map = _load_material_special_req_reverse_map()
        current_raw = self._normalize_special_req_values(item.get("SPECIAL_REQ"))
        current: List[str] = []
        seen_current = set()
        for req in current_raw:
            canonical = reverse_map.get(req.upper(), req.upper())
            if canonical and canonical not in seen_current:
                current.append(canonical)
                seen_current.add(canonical)
        existing = set(current)
        value_upper = value.upper()
        for suffix in matched_suffixes:
            if value_upper.endswith(suffix):
                continue
            if suffix in existing:
                continue
            current.append(suffix)
            existing.add(suffix)
        item["SPECIAL_REQ"] = current

    def encode(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        codes, _ = self.encode_with_confidence(entities)
        return codes

    def encode_with_confidence(self, entities: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """保留第二阶段 LLM 编码辅助能力。"""
        single: Dict[str, Any] = {}
        multi: Dict[str, Any] = {}
        for key, value in entities.items():
            if key not in self.ENCODABLE_FIELDS or not value:
                continue
            if isinstance(value, list):
                multi[key] = value
            else:
                single[key] = value

        codes: Dict[str, Any] = {}
        confidences: Dict[str, Any] = {}

        if single:
            result = self._call_model(STAGE2_SYSTEM_PROMPT, json.dumps(single, ensure_ascii=False))
            if not result.get("_parse_error"):
                model_conf = result.get("_model_confidence")
                for key in single.keys():
                    if key in result:
                        codes[key] = result[key]
                        confidences[key] = float(model_conf) if model_conf is not None else None

        for field, values in multi.items():
            encoded_list = []
            conf_list = []
            for value in values:
                result = self._call_model(STAGE2_SYSTEM_PROMPT, json.dumps({field: value}, ensure_ascii=False))
                if not result.get("_parse_error"):
                    encoded_list.append(result.get(field, value))
                    model_conf = result.get("_model_confidence")
                    conf_list.append(float(model_conf) if model_conf is not None else None)
            codes[field] = encoded_list
            confidences[field] = conf_list

        return codes, confidences

    def _merge_type_geometry_into_body(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        decisions = extracted.get("decisions")
        if not isinstance(decisions, dict):
            return extracted

        type_dict = decisions.get("TYPE")
        if not isinstance(type_dict, dict):
            return extracted

        geometry = type_dict.get("GEOMETRY")
        if not isinstance(geometry, dict):
            return extracted

        body = str(type_dict.get("BODY") or "").strip()
        angle = str(geometry.get("ANGLE") or "").strip()
        radius = str(geometry.get("RADIUS") or "").strip()

        parts: List[str] = []
        if angle:
            parts.append(angle)
        if body:
            parts.append(body)
        if radius:
            parts.append(radius)

        if parts:
            type_dict["BODY"] = ";".join(parts)

        type_dict.pop("GEOMETRY", None)
        return extracted

    def _extract(self, text: str) -> dict:
        return self._call_model(self.stage1_system_prompt, text)

    def _call_model(self, system_prompt: str, user_content: str) -> dict:
        if self.backend == "ollama":
            return self._call_ollama(system_prompt, user_content)
        return self._call_transformers(system_prompt, user_content)

    def _call_ollama(self, system_prompt: str, user_content: str) -> dict:
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "logprobs": self.ollama_logprobs_enabled,
                "options": {
                    "temperature": self.ollama_temperature,
                    "top_p": self.ollama_top_p,
                    "repeat_penalty": 1.05,
                    "num_predict": self.ollama_num_predict,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        response = payload.get("message", {}).get("content", "")
        model_confidence = self._compute_ollama_confidence(payload) if self.ollama_logprobs_enabled else 1.0
        if model_confidence is None and self.ollama_logprobs_enabled:
            raise RuntimeError("Ollama 未返回可用 logprobs，无法计算置信度（严格模式）")
        return self._parse_response(response, model_confidence=model_confidence)

    def _compute_ollama_confidence(self, payload: Dict[str, Any]) -> Optional[float]:
        logprobs = payload.get("logprobs")
        if not isinstance(logprobs, list) or not logprobs:
            return None

        probs: List[float] = []
        for item in logprobs:
            if not isinstance(item, dict):
                continue
            lp = item.get("logprob")
            if lp is None:
                continue
            try:
                p = math.exp(float(lp))
            except Exception:
                continue
            probs.append(max(1e-12, min(1.0, p)))

        if not probs:
            return None
        geo = math.exp(sum(math.log(p) for p in probs) / len(probs))
        return float(max(0.0, min(1.0, geo)))

    def _call_transformers(self, system_prompt: str, user_content: str) -> dict:
        import torch

        input_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        generation_config = copy.deepcopy(self.model.generation_config)
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, attr):
                try:
                    setattr(generation_config, attr, None)
                except Exception:
                    pass

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_len = inputs["input_ids"].shape[1]
        sequence = outputs.sequences[0]
        generated = sequence[input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        model_confidence = self._compute_generation_confidence(outputs, generated)
        return self._parse_response(response, model_confidence=model_confidence)

    def _compute_generation_confidence(self, outputs, generated_ids) -> Optional[float]:
        try:
            import torch

            scores = getattr(outputs, "scores", None)
            if not scores or generated_ids is None or len(generated_ids) == 0:
                return None

            probs = []
            max_steps = min(len(scores), len(generated_ids))
            for i in range(max_steps):
                token_id = int(generated_ids[i].item())
                step_logits = scores[i][0]
                step_prob = float(torch.softmax(step_logits, dim=-1)[token_id].item())
                probs.append(max(1e-12, min(1.0, step_prob)))

            if not probs:
                return None
            geo = math.exp(sum(math.log(p) for p in probs) / len(probs))
            return float(max(0.0, min(1.0, geo)))
        except Exception as e:
            logger.debug(f"[Qwen3预测器] 计算模型置信度失败: {e}")
            return None

    def _parse_response(self, response: str, model_confidence: Optional[float] = None) -> dict:
        cleaned = response.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                result["_raw"] = response
                if model_confidence is not None:
                    result["_model_confidence"] = float(model_confidence)
                return result
        except json.JSONDecodeError:
            pass

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            try:
                result = json.loads(cleaned[start_idx:end_idx + 1])
                if isinstance(result, dict):
                    result["_raw"] = response
                    if model_confidence is not None:
                        result["_model_confidence"] = float(model_confidence)
                    return result
            except json.JSONDecodeError:
                pass

        err = {"_parse_error": True, "_raw": response}
        if model_confidence is not None:
            err["_model_confidence"] = float(model_confidence)
        return err

    def _build_tokens(self, text: str, extracted: dict, default_confidence: float = 1.0) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        mentions = extracted.get("mentions")
        semantics = extracted.get("semantics")
        if not isinstance(mentions, list):
            return tokens

        semantic_map: Dict[str, str] = {}
        if isinstance(semantics, list):
            for item in semantics:
                if not isinstance(item, dict):
                    continue
                mention_id = str(item.get("mention_id") or "").strip()
                semantic_tag = str(item.get("semantic_tag") or "").strip()
                if mention_id and semantic_tag:
                    semantic_map[mention_id] = semantic_tag

        tagged_ranges = []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            mention_id = str(mention.get("id") or "").strip()
            mention_text = str(mention.get("text") or "").strip()
            if not mention_text:
                continue
            semantic_tag = semantic_map.get(mention_id, "")
            field = self.SEMANTIC_TO_FIELD.get(
                semantic_tag,
                self.MENTION_TYPE_TO_FIELD.get(str(mention.get("type") or "").strip().upper(), "O"),
            )
            if field == "O":
                continue
            pos = self._find_in_text(text, mention_text)
            if pos >= 0:
                tagged_ranges.append((pos, pos + len(mention_text), field, mention_text))

        tagged_ranges.sort(key=lambda x: x[0])

        cursor = 0
        for start, end, field, value in tagged_ranges:
            if start < cursor:
                continue
            if cursor < start:
                segment = text[cursor:start]
                for seg_start, word in self._split_segment(segment, cursor):
                    tokens.append({
                        "word": word,
                        "tag": "O",
                        "confidence": default_confidence,
                        "start": seg_start,
                        "end": seg_start + len(word),
                    })

            for i, (word_start, word) in enumerate(self._split_segment(value, start)):
                tokens.append({
                    "word": word,
                    "tag": f"B-{field}" if i == 0 else f"I-{field}",
                    "confidence": default_confidence,
                    "start": word_start,
                    "end": word_start + len(word),
                })
            cursor = end

        if cursor < len(text):
            segment = text[cursor:]
            for seg_start, word in self._split_segment(segment, cursor):
                tokens.append({
                    "word": word,
                    "tag": "O",
                    "confidence": default_confidence,
                    "start": seg_start,
                    "end": seg_start + len(word),
                })

        return tokens

    def _find_in_text(self, text: str, value: str) -> int:
        pos = text.find(value)
        if pos >= 0:
            return pos
        return text.lower().find(value.lower())

    def _split_segment(self, segment: str, offset: int) -> List[tuple[int, str]]:
        result = []
        pattern = re.compile(r"[\w./#°φΦ×x\-]+|[^\s]", re.UNICODE)
        for match in pattern.finditer(segment):
            result.append((offset + match.start(), match.group()))
        return result

    def _build_entities(self, decisions: dict, default_confidence: float = 1.0) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        for field, values in decisions.items():
            if field not in self.ENCODABLE_FIELDS or values in (None, "", [], {}):
                continue

            if field in {"TYPE", "SIZE", "THICKNESS"} and isinstance(values, dict):
                for subtype, subvalues in values.items():
                    if subvalues in (None, "", []):
                        continue
                    if not isinstance(subvalues, list):
                        subvalues = [subvalues]
                    for value in subvalues:
                        if value in (None, ""):
                            continue
                        entities.append({
                            "type": field,
                            "subtype": str(subtype),
                            "value": str(value).strip(),
                            "confidence": default_confidence,
                        })
                continue

            if field == "PRESSURE":
                entities.append({
                    "type": field,
                    "value": str(values).strip(),
                    "confidence": default_confidence,
                })
                continue

            if field == "MATERIAL":
                # 新结构: [{"ROLE":"MAIN","VALUE":"...","SPECIAL_REQ":[...]}]
                if isinstance(values, list):
                    for idx, item in enumerate(values):
                        if isinstance(item, dict):
                            role = item.get("ROLE")
                            if role not in (None, ""):
                                entities.append({
                                    "type": field,
                                    "subtype": "ROLE",
                                    "value": str(role).strip(),
                                    "bind_to_index": idx,
                                    "confidence": default_confidence,
                                })
                            value = item.get("VALUE")
                            if value not in (None, ""):
                                entities.append({
                                    "type": field,
                                    "subtype": "VALUE",
                                    "value": str(value).strip(),
                                    "bind_to_index": idx,
                                    "confidence": default_confidence,
                                })
                            special_req = item.get("SPECIAL_REQ")
                            if special_req not in (None, "", []):
                                if not isinstance(special_req, list):
                                    special_req = [special_req]
                                for value in special_req:
                                    if value in (None, ""):
                                        continue
                                    entities.append({
                                        "type": field,
                                        "subtype": "SPECIAL_REQ",
                                        "value": str(value).strip(),
                                        "bind_to_index": idx,
                                        "confidence": default_confidence,
                                    })
                        else:
                            item_text = str(item).strip()
                            if item_text:
                                entities.append({
                                    "type": field,
                                    "subtype": "VALUE",
                                    "value": item_text,
                                    "bind_to_index": idx,
                                    "confidence": default_confidence,
                                })
                    continue

                # 旧结构兼容: {"RELATION":"...","ITEMS":[...]}
                if isinstance(values, dict):
                    relation = values.get("RELATION")
                    if relation:
                        entities.append({
                            "type": field,
                            "subtype": "RELATION",
                            "value": str(relation).strip(),
                            "confidence": default_confidence,
                        })
                    items = values.get("ITEMS")
                    if isinstance(items, list):
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            for subtype in ("EXEC_STANDARD", "GRADE", "SPECIAL_REQ"):
                                subvalues = item.get(subtype)
                                if subvalues in (None, "", []):
                                    continue
                                if not isinstance(subvalues, list):
                                    subvalues = [subvalues]
                                for value in subvalues:
                                    if value in (None, ""):
                                        continue
                                    entities.append({
                                        "type": field,
                                        "subtype": subtype,
                                        "value": str(value).strip(),
                                        "confidence": default_confidence,
                                    })
                continue

            if field == "STANDARD" and isinstance(values, list):
                for idx, item in enumerate(values):
                    if not isinstance(item, dict):
                        continue
                    for subtype in ("BODY", "GRADE", "APPENDIX", "METHOD"):
                        value = item.get(subtype)
                        if value in (None, ""):
                            continue
                        entities.append({
                            "type": field,
                            "subtype": subtype,
                            "value": str(value).strip(),
                            "bind_to_index": idx,
                            "confidence": default_confidence,
                        })
        return entities

    def _iter_decision_fields(self, decisions: Dict[str, Any]) -> List[str]:
        return [field for field, value in decisions.items() if field in self.ENCODABLE_FIELDS and value not in (None, "", [], {})]

    def _infer_type_class(self, decisions: dict) -> Optional[str]:
        type_val = decisions.get("TYPE")
        if not type_val:
            return None
        if isinstance(type_val, dict):
            type_val = type_val.get("BODY") or ""
        if isinstance(type_val, list):
            type_val = type_val[0]
        type_lower = str(type_val).lower()
        for keyword, cls in TYPE_CLASS_MAP.items():
            if keyword in type_lower:
                return cls
        return None

    def _fallback_result(self, text: str) -> Dict[str, Any]:
        tokens = []
        for start, word in self._split_segment(text, 0):
            tokens.append({
                "word": word,
                "tag": "O",
                "confidence": 0.0,
                "start": start,
                "end": start + len(word),
            })
        return {
            "text": text,
            "tokens": tokens,
            "entities": [],
            "type_class": None,
            "model_output": {},
            "extract_confidence": {},
            "model_raw_response": "",
        }
