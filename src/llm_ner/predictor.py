# -*- coding: utf-8 -*-
"""
Qwen3 微调模型预测器

与 PipePredictor (BERT NER) 接口兼容，可通过 platform_config.yaml 切换。

输入: 管道材料描述文本
输出: 与 BERT NER 相同格式的 tokens + entities + type_class
"""

import json
import logging
import re
import time
import requests
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个管道材料信息提取助手。"
    "从描述中提取结构化信息，以JSON格式输出。\n"
    "可能包含的字段：TYPE(种类), SIZE(尺寸), THICKNESS(壁厚), "
    "PRESSURE(压力等级), MATERIAL(材质), STANDARD(规范主体), "
    "STANDARD_GRADE(规范等级，使用对象数组，元素格式为{\"value\":\"Series I\",\"bind_to_index\":0}), "
    "STANDARD_APPENDIX(规范附录，使用对象数组，元素格式为{\"value\":\"附录B\",\"bind_to_index\":0}), "
    "STANDARD_METHOD(规范方法，使用对象数组，元素格式为{\"value\":\"方法E\",\"bind_to_index\":0}或{\"value\":\"Method E\",\"bind_to_index\":0})。\n"
    "字段格式要求："
    "TYPE 使用对象结构 {\"BODY\":\"\",\"CONN\":\"\",\"ENDS\":\"\",\"SEAL\":\"\",\"MANU\":\"\"}，只输出识别到的非空 key；"
    "TYPE 各子字段都按原文表面形式提取，不要改写、翻译或补全原文中不存在的信息；"
    "SIZE 使用对象结构 {\"DN\":[],\"OD\":[],\"INCH\":[],\"LENGTH\":[]}；"
    "SIZE.LENGTH 只提取原文中明确出现的长度表达，按原样保留，如 L=300mm、L=100、Length=200mm；"
    "THICKNESS 使用对象结构 {\"MM\":[],\"INCH\":[],\"SCHEDULE\":[],\"SERIES\":[],\"BWG\":[]}；"
    "MATERIAL 使用对象结构 {\"RELATION\":\"single|alternative\",\"ITEMS\":[{\"EXEC_STANDARD\":\"\",\"MATERIAL_GRADE_CODE\":\"\",\"SPECIAL_REQ\":[]}]}；"
    "MATERIAL 中各字段按当前标注规范抽取，优先保留原文表面形式；"
    "PRESSURE 保持原样提取的单字符串，不做结构化。"
    "不要输出顶层旧字段 SEAL、ENDS、CONN、MANU；如果识别到这些信息，应放入 TYPE 对象内部对应子字段。"
    "只输出识别到的字段，直接输出JSON。"
)

ENCODING_SYSTEM_PROMPT = (
    "你是一个管道材料编码助手。"
    "将实体原始值转换为标准编码，以JSON格式输出。直接输出JSON。"
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


class Qwen3Predictor:
    """
    Qwen3 微调模型预测器

    支持两种后端：
        - ollama: 通过 Ollama HTTP API 调用（推荐，快速）
        - transformers: 直接加载 HF 模型（备用，慢）

    用法:
        # Ollama 后端（默认）
        predictor = Qwen3Predictor(model_name="qwen3-pipe", backend="ollama")

        # HF transformers 后端
        predictor = Qwen3Predictor(model_path="models/qwen3_pipe", backend="transformers")
    """

    @staticmethod
    def _iter_output_fields(extracted: Dict[str, Any]) -> List[str]:
        """按模型输出动态遍历字段，忽略内部元字段。"""
        return [field for field in extracted.keys() if not str(field).startswith("_")]

    @staticmethod
    def _iter_field_values(field: str, values: Any):
        """
        将字段值展开为 (field, subtype, value, bind_to_index) 迭代器。

        兼容:
        - 普通字符串 / 列表
        - STANDARD_* 的 {"value": "...", "bind_to_index": 0}
        - 新 schema 的 TYPE / SIZE / THICKNESS / MATERIAL 嵌套对象
        """
        if values is None:
            return

        if isinstance(values, dict):
            if "value" in values:
                yield field, None, values.get("value"), values.get("bind_to_index")
                return

            for subtype, subvalues in values.items():
                if subvalues in (None, "", []):
                    continue
                if not isinstance(subvalues, list):
                    subvalues = [subvalues]
                for item in subvalues:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        yield field, subtype, item.get("value"), item.get("bind_to_index")
                    else:
                        yield field, subtype, item, None
            return

        if not isinstance(values, list):
            values = [values]

        for item in values:
            if item is None:
                continue
            if isinstance(item, dict):
                yield field, None, item.get("value"), item.get("bind_to_index")
            else:
                yield field, None, item, None

    def __init__(
        self,
        model_path: str = None,
        model_name: str = "qwen3-pipe",
        backend: str = "ollama",
        device: str = "auto",
        ollama_url: str = "http://localhost:11434",
    ):
        self.backend = backend
        self.model = None
        self.tokenizer = None

        if backend == "ollama":
            self.ollama_url = ollama_url
            self.model_name = model_name
            logger.info(f"[Qwen3预测器] Ollama 后端, 模型: {model_name}")
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model_path = Path(model_path)
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            if device == "mps":
                logger.warning("[Qwen3预测器] MPS 与 Qwen3 存在兼容性问题，自动回退到 CPU")
                device = "cpu"
            self.device = device

            logger.info(f"[Qwen3预测器] Transformers 后端, 模型: {self.model_path}, 设备: {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True, padding_side="left",
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
        """
        预测文本，返回与 PipePredictor 兼容的格式

        Returns:
            {
                'text': 原始文本,
                'tokens': [{'word': ..., 'tag': ..., 'confidence': ..., 'start': ..., 'end': ...}],
                'entities': [{'type': ..., 'value': ..., 'start': ..., 'end': ...}],
                'type_class': 材料大类（管子/管件/法兰/螺栓/阀门/垫片）
            }
        """
        if not text or not text.strip():
            return {
                "text": text,
                "tokens": [],
                "entities": [],
                "type_class": None,
                "model_output": {},
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
        if model_conf is None:
            raise RuntimeError("模型未返回 logprobs，无法计算置信度（严格模式）")
        default_conf = float(model_conf)
        field_confidence = {}
        tokens = self._build_tokens(
            text, extracted,
            default_confidence=default_conf,
            field_confidence=field_confidence
        )
        entities = self._build_entities(
            extracted,
            default_confidence=default_conf,
            field_confidence=field_confidence
        )
        type_class = self._infer_type_class(extracted)

        return {
            "text": text,
            "tokens": tokens,
            "entities": entities,
            "type_class": type_class,
            "model_output": extracted,
            "model_raw_response": extracted.get("_raw", ""),
        }

    ENCODABLE_FIELDS = {"TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD"}

    def encode(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧接口：仅返回编码结果。"""
        codes, _ = self.encode_with_confidence(entities)
        return codes

    def encode_with_confidence(self, entities: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        对 NER 提取的实体进行编码（第二次调用）

        Args:
            entities: NER 提取结果，如 {"TYPE": {"BODY": "90°长半径弯头"}, "SIZE": {"DN": ["DN100"]}, ...}
                      值可以是字符串或列表（多值字段如多个 STANDARD）

        Returns:
            (codes, confidences)
            - codes: {"TYPE": "90EL", "SIZE": "100", ...}
            - confidences: {"TYPE": 0.83, "STANDARD": [0.71, 0.76], ...}
        """
        single = {}
        multi = {}
        for k, v in entities.items():
            if k not in self.ENCODABLE_FIELDS or not v:
                continue
            if isinstance(v, list):
                multi[k] = v
            else:
                single[k] = v

        codes: Dict[str, Any] = {}
        confidences: Dict[str, Any] = {}

        if single:
            result = self._call_model(ENCODING_SYSTEM_PROMPT,
                                      json.dumps(single, ensure_ascii=False))
            if not result.get("_parse_error"):
                model_conf = result.get("_model_confidence")
                for k in single.keys():
                    if k in result:
                        codes[k] = result[k]
                        confidences[k] = float(model_conf) if model_conf is not None else None

        for field, values in multi.items():
            encoded_list = []
            conf_list = []
            for val in values:
                result = self._call_model(ENCODING_SYSTEM_PROMPT,
                                          json.dumps({field: val}, ensure_ascii=False))
                if not result.get("_parse_error"):
                    encoded_list.append(result.get(field, val))
                    model_conf = result.get("_model_confidence")
                    conf_list.append(float(model_conf) if model_conf is not None else None)
            codes[field] = encoded_list
            confidences[field] = conf_list

        return codes, confidences

    def _extract(self, text: str) -> dict:
        """调用模型提取结构化信息（NER）"""
        return self._call_model(SYSTEM_PROMPT, text)

    def _call_model(self, system_prompt: str, user_content: str) -> dict:
        """通用模型调用：指定 system prompt 和用户输入，返回解析后的 JSON"""
        if self.backend == "ollama":
            return self._call_ollama(system_prompt, user_content)
        return self._call_transformers(system_prompt, user_content)

    def _call_ollama(self, system_prompt: str, user_content: str) -> dict:
        """通过 Ollama HTTP API 调用"""
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "logprobs": True,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                    "num_predict": 256,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        response = payload.get("message", {}).get("content", "")
        model_confidence = self._compute_ollama_confidence(payload)
        if model_confidence is None:
            raise RuntimeError("Ollama 未返回可用 logprobs，无法计算置信度（严格模式）")
        return self._parse_response(response, model_confidence=model_confidence)

    def _compute_ollama_confidence(self, payload: Dict[str, Any]) -> Optional[float]:
        """从 Ollama 响应中的 logprobs 计算几何平均概率。"""
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
        """通过 HuggingFace transformers 直接推理"""
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
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
        """根据 transformers 生成分数计算平均 token 概率。"""
        try:
            import torch
            scores = getattr(outputs, "scores", None)
            if not scores:
                return None
            if generated_ids is None or len(generated_ids) == 0:
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
            # 几何平均，抑制“某几步很低概率”的风险
            geo = math.exp(sum(math.log(p) for p in probs) / len(probs))
            return float(max(0.0, min(1.0, geo)))
        except Exception as e:
            logger.debug(f"[Qwen3预测器] 计算模型置信度失败: {e}")
            return None

    def _parse_response(self, response: str, model_confidence: Optional[float] = None) -> dict:
        """解析模型输出的 JSON"""
        cleaned = response.strip()

        # 去除 think 标签
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
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

    def _build_tokens(
        self,
        text: str,
        extracted: dict,
        default_confidence: float = 1.0,
        field_confidence: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        将 LLM 提取结果转换为 token 级别的标注

        在原文中定位每个字段值的位置，生成与 BERT NER 兼容的 tokens 列表。
        """
        tokens = []
        tagged_ranges = []

        for field in self._iter_output_fields(extracted):
            if field not in extracted or field.startswith("_"):
                continue

            for _, _, val, _ in self._iter_field_values(field, extracted[field]):
                if val is None:
                    continue
                val_str = str(val).strip()
                if not val_str:
                    continue

                pos = self._find_in_text(text, val_str)
                if pos >= 0:
                    tagged_ranges.append((pos, pos + len(val_str), field, val_str))

        tagged_ranges.sort(key=lambda x: x[0])

        # 合并标注与未标注的文本片段
        cursor = 0
        for start, end, field, val in tagged_ranges:
            if start < cursor:
                continue

            # 未标注区间 → O
            if cursor < start:
                segment = text[cursor:start]
                for ch_start, word in self._split_segment(segment, cursor):
                    tokens.append({
                        "word": word, "tag": "O", "confidence": default_confidence,
                        "start": ch_start, "end": ch_start + len(word),
                    })

            # 标注区间 → B-FIELD / I-FIELD
            words = self._split_segment(val, start)
            for i, (w_start, word) in enumerate(words):
                tag = f"B-{field}" if i == 0 else f"I-{field}"
                tokens.append({
                    "word": word, "tag": tag, "confidence": default_confidence,
                    "start": w_start, "end": w_start + len(word),
                })
                if field_confidence and field in field_confidence:
                    tokens[-1]["confidence"] = field_confidence[field]
            cursor = end

        # 尾部未标注
        if cursor < len(text):
            segment = text[cursor:]
            for ch_start, word in self._split_segment(segment, cursor):
                tokens.append({
                    "word": word, "tag": "O", "confidence": default_confidence,
                    "start": ch_start, "end": ch_start + len(word),
                })

        return tokens

    def _find_in_text(self, text: str, value: str) -> int:
        """在原文中查找值的位置（大小写不敏感）"""
        pos = text.find(value)
        if pos >= 0:
            return pos
        pos = text.lower().find(value.lower())
        return pos

    def _split_segment(self, segment: str, offset: int) -> List[tuple]:
        """将文本片段按空格/标点分割为 word 列表"""
        result = []
        pattern = re.compile(r"[\w./#°φΦ×x\-]+|[^\s]", re.UNICODE)
        for m in pattern.finditer(segment):
            result.append((offset + m.start(), m.group()))
        return result

    def _build_entities(
        self,
        extracted: dict,
        default_confidence: float = 1.0,
        field_confidence: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """构建 entities 列表"""
        entities = []
        for field in self._iter_output_fields(extracted):
            if field not in extracted or field.startswith("_"):
                continue

            for _, subtype, val, bind_to_index in self._iter_field_values(field, extracted[field]):
                if val is None:
                    continue
                conf = default_confidence
                if field_confidence and field in field_confidence:
                    conf = field_confidence[field]
                entity = {"type": field, "value": str(val).strip(), "confidence": conf}
                if subtype is not None:
                    entity["subtype"] = subtype
                if bind_to_index is not None:
                    entity["bind_to_index"] = bind_to_index
                entities.append(entity)
        return entities

    def _infer_type_class(self, extracted: dict) -> Optional[str]:
        """根据 TYPE 字段推断材料大类"""
        type_val = extracted.get("TYPE")
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
        """JSON 解析失败时的兜底结果"""
        tokens = []
        for ch_start, word in self._split_segment(text, 0):
            tokens.append({
                "word": word, "tag": "O", "confidence": 0.0,
                "start": ch_start, "end": ch_start + len(word),
            })
        return {
            "text": text,
            "tokens": tokens,
            "entities": [],
            "type_class": None,
            "model_output": {},
            "model_raw_response": "",
        }
