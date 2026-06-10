# -*- coding: utf-8 -*-
"""
一阶段 / 二阶段提示词统一维护文件。

维护约定：
1. 只在本文件中维护提示词正文。
2. 外部代码通过 get_prompt(...) 或具体 getter 获取，不再在各处散落常量。
3. 当前一阶段与二阶段分别共享同一套提示词：
   - 一阶段：微调 / 平台预测 / 脚本推理 共用同一提示词
   - 二阶段：微调 / 平台编码 共用同一提示词
"""

from __future__ import annotations


STAGE1_SEMANTIC_PARSER_PROMPT = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并严格输出 JSON。"
    "输出包含 mentions、semantics、decisions。"
    "其中 TYPE 结构包含 BODY、GEOMETRY(ANGLE/RADIUS)、MANU、CONN、SEAL、ENDS。"
    "不要输出解释文字，不要输出 markdown，不要补充原文中不存在的信息，只输出合法 JSON。"
)

STAGE1_DECISIONS_ONLY_PROMPT = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取最终结构化决策结果，并严格输出 JSON。"
    "只输出 decisions。"
    "其中 TYPE 结构包含 BODY、GEOMETRY(ANGLE/RADIUS)、MANU、CONN、SEAL、ENDS。"
    "不要输出解释文字，不要输出 markdown，不要补充原文中不存在的信息，只输出合法 JSON。"
)


STAGE2_ENCODING_PROMPT = (
    "你是一个管道材料编码助手。"
    "将实体原始值转换为标准编码，以JSON格式输出。直接输出JSON。"
)


PROMPT_REGISTRY = {
    "stage1_finetune": STAGE1_SEMANTIC_PARSER_PROMPT,
    "stage1_platform_predict": STAGE1_SEMANTIC_PARSER_PROMPT,
    "stage1_inference": STAGE1_SEMANTIC_PARSER_PROMPT,
    "stage1_decisions_only": STAGE1_DECISIONS_ONLY_PROMPT,
    "stage2_finetune": STAGE2_ENCODING_PROMPT,
    "stage2_platform_predict": STAGE2_ENCODING_PROMPT,
}


def get_prompt(name: str) -> str:
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        raise KeyError(f"未知提示词键: {name}。可用键: {available}") from exc


def get_stage1_finetune_prompt() -> str:
    return get_prompt("stage1_finetune")


def get_stage1_platform_predict_prompt() -> str:
    return get_prompt("stage1_platform_predict")


def get_stage1_inference_prompt() -> str:
    return get_prompt("stage1_inference")


def get_stage1_decisions_only_prompt() -> str:
    return get_prompt("stage1_decisions_only")


def get_stage2_finetune_prompt() -> str:
    return get_prompt("stage2_finetune")


def get_stage2_platform_predict_prompt() -> str:
    return get_prompt("stage2_platform_predict")
