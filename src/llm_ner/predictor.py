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
import copy
import requests
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .rule_extractors import (
    canonicalize_size_token,
    canonicalize_thickness_token,
    extract_dnxn_ambiguity_rules,
    extract_material_special_req_rules,
    extract_size_rules,
    extract_thickness_rules,
    extract_type_rules,
)

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
    "THICKNESS 是必须重点检查的字段：凡是原文中明确出现的壁厚表达，不能因为同时存在 SIZE、PRESSURE、MATERIAL、STANDARD、TYPE 就省略；"
    "常见壁厚表达如 SCH/Sch/SCH. + 数字或数字+S、10S、20S、40S、80S、XS、XXS、STD，应优先提取到 THICKNESS；"
    "其中 SCH/Sch/SCH. + 数字或数字+S 优先放入 THICKNESS.SCHEDULE；XS、XXS、STD 优先放入 THICKNESS.SERIES；"
    "这些壁厚 token 即使独立出现、没有紧跟在 SIZE 后面，也仍然属于 THICKNESS；"
    "如果遇到复合壁厚表达，应整体优先判断为 THICKNESS，不要把其中片段错误放入 TYPE、ENDS、CONN、MANU；拆分时优先保留原文可对应的表面形式。"
    "要区分 MATERIAL 与 THICKNESS：20、20#、A105、316L 这类材质表达不是 THICKNESS，不要把材质数字误放入 THICKNESS；"
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
        extracted_raw = self._extract(text)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"[Qwen3预测器] 推理耗时: {elapsed:.2f}s")

        if extracted_raw.get("_parse_error"):
            logger.warning(f"[Qwen3预测器] JSON解析失败: {extracted_raw.get('_raw', '')[:200]}")
            return self._fallback_result(text)

        model_conf = extracted_raw.get("_model_confidence")
        if model_conf is None:
            raise RuntimeError("模型未返回 logprobs，无法计算置信度（严格模式）")

        # 双轨输出：保留模型原始输出，再在副本上执行 Hybrid 后处理
        extracted = copy.deepcopy(extracted_raw)

        # Hybrid 后处理：规则抽取 + canonical 对齐（当前仅 THICKNESS）。
        self._apply_type_rule_reconciliation(text, extracted)
        self._apply_size_structural_correction(text, extracted)
        self._apply_size_rule_reconciliation(text, extracted)
        self._apply_dnxn_ambiguity_reconciliation(text, extracted)
        self._apply_thickness_structural_correction(text, extracted)
        self._apply_thickness_rule_reconciliation(text, extracted)
        self._apply_material_rule_reconciliation(text, extracted)

        default_conf = float(model_conf)
        field_confidence: Dict[str, float] = {}
        review_fields: List[str] = []
        review_reasons: List[str] = []
        rule_alerts = extracted.get("_rule_alerts", [])
        if isinstance(rule_alerts, list) and rule_alerts:
            # 规则告警意味着模型输出与高精度规则存在差异，标记待审并下调字段置信度。
            if any(p.get("field") == "THICKNESS" for p in rule_alerts if isinstance(p, dict)):
                field_confidence["THICKNESS"] = min(default_conf, 0.78)
                review_fields.append("THICKNESS")
                review_reasons.append("THICKNESS 与规则识别结果不一致，建议人工复核。")
            if any(p.get("field") == "TYPE" for p in rule_alerts if isinstance(p, dict)):
                field_confidence["TYPE"] = min(default_conf, 0.80)
                review_fields.append("TYPE")
                review_reasons.append("TYPE 子字段与规则识别结果不一致，建议人工复核。")
            if any(p.get("field") == "SIZE" for p in rule_alerts if isinstance(p, dict)):
                field_confidence["SIZE"] = min(default_conf, 0.80)
                review_fields.append("SIZE")
                review_reasons.append("SIZE 与规则识别结果不一致，建议人工复核。")
            if any(p.get("field") == "MATERIAL" for p in rule_alerts if isinstance(p, dict)):
                field_confidence["MATERIAL"] = min(default_conf, 0.80)
                review_fields.append("MATERIAL")
                review_reasons.append("MATERIAL 特殊要求与规则识别结果不一致，建议人工复核。")
            if any(p.get("reason") in {"dn_x_decimal", "dn_x_small_integer", "dn_x_large_integer"}
                   for p in rule_alerts if isinstance(p, dict)):
                # DNx数字是结构歧义，统一降 SIZE 置信度并提示审核
                field_confidence["SIZE"] = min(field_confidence.get("SIZE", default_conf), 0.75)
                if "SIZE" not in review_fields:
                    review_fields.append("SIZE")
                review_reasons.append("命中 DNx数字 歧义模式，需人工确认第二段是尺寸还是壁厚。")

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
            "model_output_raw": extracted_raw,
            "model_output_hybrid": extracted,
            "decision_log": {
                "rule_fixes": extracted.get("_rule_fixes", []),
                "rule_alerts": extracted.get("_rule_alerts", []),
            },
            "model_raw_response": extracted.get("_raw", ""),
            "need_review": bool(review_fields),
            "review_fields": review_fields,
            "review_reasons": review_reasons,
        }

    @staticmethod
    def _collect_model_type_canonicals(extracted: Dict[str, Any]) -> set:
        """
        收集 TYPE 子字段值（仅 MANU/ENDS/SEAL/CONN）。
        """
        t = extracted.get("TYPE")
        if not isinstance(t, dict):
            return set()
        allowed = {"MANU", "ENDS", "SEAL", "CONN"}
        result = set()
        for subtype, vals in t.items():
            st = str(subtype).upper()
            if st not in allowed:
                continue
            if vals in (None, "", []):
                continue
            if not isinstance(vals, list):
                vals = [vals]
            for v in vals:
                s = str(v).strip().upper()
                if s:
                    result.add((st, s))
        return result

    def _apply_type_rule_reconciliation(self, text: str, extracted: Dict[str, Any]):
        """
        TYPE 子字段规则校验：仅 MANU/ENDS/SEAL/CONN，BODY 不参与规则。
        当前策略：仅告警，不自动补位。
        """
        if not isinstance(extracted, dict):
            return
        rule_hits = extract_type_rules(text)
        if not rule_hits:
            return

        model_vals = self._collect_model_type_canonicals(extracted)
        alerts: List[Dict[str, str]] = []
        for h in rule_hits:
            st = str(h.get("subtype") or "").upper()
            cv = str(h.get("canonical") or "").upper()
            if not st or not cv:
                continue
            if (st, cv) not in model_vals:
                alerts.append(
                    {
                        "field": "TYPE",
                        "reason": "rule_missing_in_model",
                        "subtype": st,
                        "canonical": cv,
                        "raw": h.get("raw", ""),
                        "rule_id": str(h.get("rule_id") or ""),
                        "source": str(h.get("source") or ""),
                    }
                )

        if alerts:
            extracted["_rule_alerts"] = extracted.get("_rule_alerts", []) + alerts
            logger.info(f"[Qwen3预测器] TYPE规则告警: {alerts}")

    @staticmethod
    def _collect_model_size_canonicals(extracted: Dict[str, Any]) -> set:
        size = extracted.get("SIZE")
        if not isinstance(size, dict):
            return set()
        result = set()
        for subtype, vals in size.items():
            if vals in (None, "", []):
                continue
            if not isinstance(vals, list):
                vals = [vals]
            for v in vals:
                c = canonicalize_size_token(str(v).strip(), subtype=str(subtype).upper())
                if c:
                    result.add((str(subtype).upper(), c))
        return result

    @staticmethod
    def _is_size_value_legal(subtype: str, canonical: str) -> bool:
        st = str(subtype or "").upper()
        cv = str(canonical or "").upper().replace(" ", "")
        if not st or not cv:
            return False
        if st == "DN":
            return bool(re.match(r"^DN\d+(?:\.\d+)?$", cv))
        if st == "OD":
            return bool(re.match(r"^OD\d+(?:\.\d+)?(?:MM)?$", cv))
        if st == "INCH":
            return bool(re.match(r"^\d+(?:\.\d+)?(?:-\d+/\d+|\s+\d+/\d+|/\d+)?(?:\"|”|″)?$", cv))
        if st == "LENGTH":
            return bool(re.match(r"^L=\d+(?:\.\d+)?(?:MM|CM|M)?$", cv))
        return True

    def _apply_size_structural_correction(self, text: str, extracted: Dict[str, Any]):
        """
        SIZE 结构纠错：
        - 统一 canonical；
        - 过滤明显非法值（仅非法才丢弃）；
        - 不做低置信推断补位。
        """
        if not isinstance(extracted, dict):
            return
        size = extracted.get("SIZE")
        if not isinstance(size, dict):
            return

        fixes: List[Dict[str, str]] = []
        allowed_subtypes = {"DN", "OD", "INCH", "LENGTH"}
        normalized_size: Dict[str, Any] = {}

        for subtype, vals in size.items():
            st = str(subtype or "").upper()
            if st not in allowed_subtypes:
                continue
            if vals in (None, "", []):
                continue
            if not isinstance(vals, list):
                vals = [vals]

            out_vals: List[str] = []
            for v in vals:
                raw = str(v).strip()
                if not raw:
                    continue
                cv = canonicalize_size_token(raw, subtype=st)
                if not cv or not self._is_size_value_legal(st, cv):
                    fixes.append(
                        {
                            "field": "SIZE",
                            "action": "drop_illegal_size_value",
                            "subtype": st,
                            "from": raw,
                            "canonical": cv,
                        }
                    )
                    continue
                if cv not in out_vals:
                    out_vals.append(cv)
                    if cv != raw:
                        fixes.append(
                            {
                                "field": "SIZE",
                                "action": "normalize_size_value",
                                "subtype": st,
                                "from": raw,
                                "to": cv,
                            }
                        )

            if out_vals:
                normalized_size[st] = out_vals

        extracted["SIZE"] = normalized_size
        if fixes:
            extracted["_rule_fixes"] = extracted.get("_rule_fixes", []) + fixes

    def _apply_size_rule_reconciliation(self, text: str, extracted: Dict[str, Any]):
        """
        SIZE 规则校验：仅告警，不自动补位。
        """
        if not isinstance(extracted, dict):
            return
        rule_hits = extract_size_rules(text)
        if not rule_hits:
            return

        model_canonicals = self._collect_model_size_canonicals(extracted)
        alerts: List[Dict[str, str]] = []
        for hit in rule_hits:
            subtype = str(hit.get("subtype") or "").upper()
            canonical = str(hit.get("canonical") or "")
            if not subtype or not canonical:
                continue
            if (subtype, canonical) not in model_canonicals:
                alerts.append(
                    {
                        "field": "SIZE",
                        "reason": "rule_missing_in_model",
                        "subtype": subtype,
                        "canonical": canonical,
                        "raw": hit.get("raw", ""),
                        "rule_id": str(hit.get("rule_id") or ""),
                        "source": str(hit.get("source") or ""),
                    }
                )

        if alerts:
            extracted["_rule_alerts"] = extracted.get("_rule_alerts", []) + alerts
            logger.info(f"[Qwen3预测器] SIZE规则告警: {alerts}")

    def _apply_dnxn_ambiguity_reconciliation(self, text: str, extracted: Dict[str, Any]):
        """
        DNx数字歧义校验：
        - 仅在“模型缺失/与规则建议冲突”时告警，不改写模型输出。
        """
        if not isinstance(extracted, dict):
            return
        hits = extract_dnxn_ambiguity_rules(text)
        if not hits:
            return

        def _collect_subtype_values(field: str, subtype: str) -> set:
            result = set()
            obj = extracted.get(field)
            if not isinstance(obj, dict):
                return result
            vals = obj.get(subtype)
            if vals in (None, "", []):
                return result
            if not isinstance(vals, list):
                vals = [vals]
            for v in vals:
                s = str(v).strip()
                if s:
                    result.add(s.upper())
            return result

        size_dn_vals = _collect_subtype_values("SIZE", "DN")
        thk_mm_vals = _collect_subtype_values("THICKNESS", "MM")

        alerts: List[Dict[str, str]] = []
        for h in hits:
            right_value = str(h.get("right_value") or "").strip()
            suggest_field = str(h.get("suggest_field") or "").strip().upper()
            suggest_subtype = str(h.get("suggest_subtype") or "").strip().upper()
            reason = str(h.get("reason") or "").strip()
            if not suggest_field or not suggest_subtype:
                continue

            # 一致性判断：规则建议与模型已提取是否一致
            if suggest_field == "SIZE" and suggest_subtype == "DN":
                expected = f"DN{right_value}".upper()
                matched = expected in size_dn_vals
            elif suggest_field == "THICKNESS" and suggest_subtype == "MM":
                expected_raw = right_value.upper()
                expected_mm = f"{right_value}MM".upper()
                matched = expected_raw in thk_mm_vals or expected_mm in thk_mm_vals
            else:
                matched = False

            # 一致则不告警，仅冲突/缺失时告警
            if matched:
                continue

            alerts.append(
                {
                    "field": "SIZE",
                    "reason": reason or "dn_x_ambiguous",
                    "raw": h.get("raw", ""),
                    "suggest_field": suggest_field,
                    "suggest_subtype": suggest_subtype,
                    "right_value": right_value,
                }
            )
        extracted["_rule_alerts"] = extracted.get("_rule_alerts", []) + alerts
        if alerts:
            logger.info(f"[Qwen3预测器] DNx数字歧义告警: {alerts}")

    @staticmethod
    def _collect_model_thickness_canonicals(extracted: Dict[str, Any]) -> set:
        thk = extracted.get("THICKNESS")
        if not isinstance(thk, dict):
            return set()

        pairs: List[tuple] = []
        for subtype, sub_vals in thk.items():
            if sub_vals in (None, "", []):
                continue
            if not isinstance(sub_vals, list):
                sub_vals = [sub_vals]
            for v in sub_vals:
                s = str(v).strip()
                if s:
                    pairs.append((str(subtype).upper(), s))

        result = set()
        for subtype, value in pairs:
            c = canonicalize_thickness_token(value, subtype=subtype)
            if c:
                result.add((subtype, c))
        return result

    @staticmethod
    def _is_thickness_rule_covered(subtype: str, canonical: str, model_canonicals: set) -> bool:
        key = (subtype, canonical)
        if key in model_canonicals:
            return True

        # 复合表达等价：规则是 A X B，模型分成 [A, B] 也视为覆盖。
        if "X" in canonical:
            parts = [p for p in canonical.split("X") if p]
            if parts and all((subtype, p) in model_canonicals for p in parts):
                return True
        return False

    def _apply_thickness_structural_correction(self, text: str, extracted: Dict[str, Any]):
        """
        THICKNESS 结构纠错层（规则优先）：
        - 将明显非法 SERIES（如 XS80）纠正为 SCHEDULE（SCH80）；
        - SERIES 仅保留独立合法 token（XS/XXS/STD）；
        - 用高确定规则命中补齐 SCHEDULE/SERIES。
        """
        if not isinstance(extracted, dict):
            return
        thk = extracted.get("THICKNESS")
        if not isinstance(thk, dict):
            return

        def _as_list(v):
            if v in (None, "", []):
                return []
            return v if isinstance(v, list) else [v]

        allowed_series = {"XS", "XXS", "STD"}
        model_schedule = _as_list(thk.get("SCHEDULE"))
        model_series = _as_list(thk.get("SERIES"))

        # 规则命中（高确定）
        rule_hits = extract_thickness_rules(text)
        rule_schedule = []
        rule_series = []
        for h in rule_hits:
            st = str(h.get("subtype") or "").upper()
            cv = str(h.get("canonical") or "").upper()
            if not cv:
                continue
            if st == "SCHEDULE":
                rule_schedule.append(cv)
            elif st == "SERIES":
                rule_series.append(cv)

        fixes: List[Dict[str, str]] = []

        schedule_out: List[str] = []
        series_out: List[str] = []

        def _append_unique(dst: List[str], val: str):
            if val and val not in dst:
                dst.append(val)

        # 保留/规范模型 SCHEDULE
        for raw in model_schedule:
            v = str(raw).strip()
            if not v:
                continue
            c = canonicalize_thickness_token(v, subtype="SCHEDULE")
            if c.startswith("SCH"):
                _append_unique(schedule_out, c)
            elif c in allowed_series:
                _append_unique(series_out, c)

        # 处理模型 SERIES（含纠错）
        for raw in model_series:
            v = str(raw).strip().upper()
            if not v:
                continue
            if v in allowed_series:
                _append_unique(series_out, v)
                continue

            # XS80 / XXS80 / STD80 -> SCH80
            m = re.match(r"^(XXS|XS|STD)[\s\-_/.]*(\d+(?:\.\d+)?S?)$", v)
            if m:
                sch = f"SCH{m.group(2)}"
                _append_unique(schedule_out, sch)
                fixes.append(
                    {
                        "field": "THICKNESS",
                        "action": "series_to_schedule",
                        "from": v,
                        "to": sch,
                    }
                )
                continue

            # 其他可规范化 schedule 的串也转到 SCHEDULE
            c = canonicalize_thickness_token(v, subtype="SCHEDULE")
            if c.startswith("SCH"):
                _append_unique(schedule_out, c)
                fixes.append(
                    {
                        "field": "THICKNESS",
                        "action": "series_to_schedule",
                        "from": v,
                        "to": c,
                    }
                )
                continue

            # 不合法 series 丢弃
            fixes.append(
                {
                    "field": "THICKNESS",
                    "action": "drop_invalid_series",
                    "from": v,
                }
            )

        # 规则补齐（高确定）
        for c in rule_schedule:
            _append_unique(schedule_out, c)
        for c in rule_series:
            _append_unique(series_out, c)

        # 若原文有规则化 schedule 命中，则仅保留被规则支持的 series（避免 S-10SXS-80 中误提 XS）
        if rule_schedule:
            filtered = [s for s in series_out if s in set(rule_series)]
            removed = [s for s in series_out if s not in set(rule_series)]
            for s in removed:
                fixes.append(
                    {
                        "field": "THICKNESS",
                        "action": "drop_unanchored_series",
                        "from": s,
                    }
                )
            series_out = filtered

        # 回写
        if schedule_out:
            thk["SCHEDULE"] = schedule_out
        else:
            thk.pop("SCHEDULE", None)

        if series_out:
            thk["SERIES"] = series_out
        else:
            thk.pop("SERIES", None)

        if fixes:
            extracted["_rule_fixes"] = extracted.get("_rule_fixes", []) + fixes

    def _apply_thickness_rule_reconciliation(self, text: str, extracted: Dict[str, Any]):
        """
        使用高精度规则抽取结果与模型 THICKNESS 做 canonical 对齐。
        当前策略：只做告警，不直接改写模型输出（避免歧义样本被规则误补位）。
        """
        if not isinstance(extracted, dict):
            return

        rule_hits = extract_thickness_rules(text)
        if not rule_hits:
            return

        model_canonicals = self._collect_model_thickness_canonicals(extracted)
        size_inch_vals = set()
        size_obj = extracted.get("SIZE")
        if isinstance(size_obj, dict):
            vals = size_obj.get("INCH")
            if vals not in (None, "", []):
                if not isinstance(vals, list):
                    vals = [vals]
                for v in vals:
                    cv = canonicalize_size_token(str(v).strip(), subtype="INCH")
                    if cv:
                        size_inch_vals.add(cv.upper())

        alerts: List[Dict[str, str]] = []
        for hit in rule_hits:
            canonical = hit.get("canonical", "")
            subtype = str(hit.get("subtype") or "").upper()
            if not canonical:
                continue
            # 同一英寸值若已在 SIZE.INCH 命中，不再要求 THICKNESS.INCH，避免交叉误报
            if subtype == "INCH" and str(canonical).upper() in size_inch_vals:
                continue
            if not self._is_thickness_rule_covered(subtype, canonical, model_canonicals):
                alerts.append(
                    {
                        "field": "THICKNESS",
                        "reason": "rule_missing_in_model",
                        "subtype": subtype,
                        "canonical": canonical,
                        "raw": hit.get("raw", ""),
                    }
                )

        if alerts:
            extracted["_rule_alerts"] = extracted.get("_rule_alerts", []) + alerts
            logger.info(f"[Qwen3预测器] THICKNESS规则告警: {alerts}")

    def _apply_material_rule_reconciliation(self, text: str, extracted: Dict[str, Any]):
        """
        材质规则校验：识别抗硫术语并与模型 MATERIAL.SPECIAL_REQ 对比。
        当前策略：仅告警，不自动补位。
        """
        if not isinstance(extracted, dict):
            return

        req_hits = extract_material_special_req_rules(text)
        if not req_hits:
            return

        model_req_set = set()
        mat = extracted.get("MATERIAL")
        if isinstance(mat, dict):
            items = mat.get("ITEMS")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    sr = it.get("SPECIAL_REQ")
                    if isinstance(sr, list):
                        model_req_set.update(str(v).strip() for v in sr if str(v).strip())
                    elif sr:
                        model_req_set.add(str(sr).strip())

        alerts: List[Dict[str, str]] = []
        for hit in req_hits:
            req = str(hit.get("special_req") or "").strip()
            if not req:
                continue
            if req not in model_req_set:
                alerts.append(
                    {
                        "field": "MATERIAL",
                        "reason": "rule_missing_in_model",
                        "semantic": hit.get("semantic", ""),
                        "special_req": req,
                        "raw": hit.get("raw", ""),
                    }
                )

        if alerts:
            extracted["_rule_alerts"] = extracted.get("_rule_alerts", []) + alerts
            logger.info(f"[Qwen3预测器] MATERIAL规则告警: {alerts}")

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
