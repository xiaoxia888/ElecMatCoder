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
    "你是一个管道材料语义解析助手。"
    "从材料描述中提取语义解析结果，并严格输出 JSON。\n"
    "输出结构固定为："
    "{\"mentions\":[{\"id\":\"m1\",\"text\":\"\",\"type\":\"\"}],"
    "\"semantics\":[{\"mention_id\":\"m1\",\"semantic_tag\":\"\"}],"
    "\"decisions\":{...}}。\n"
    "要求："
    "1. mentions 必须尽量保留原文证据片段，优先保留连续原文片段；"
    "2. semantics 用于标记 mention 在当前上下文中的语义角色；"
    "3. decisions 用于给出最终结构化决策结果，可以做轻度规范化与结构化拆分；"
    "4. 不要输出解释文字，不要输出 markdown，不要补充原文中不存在的信息；"
    "5. 只输出合法 JSON。"
)

STAGE1_DECISIONS_ONLY_PROMPT = (
    "你是一个管道材料语义解析助手。"
    "从材料描述中提取最终结构化决策结果，并严格输出 JSON。\n"
    "输出结构固定为："
    "{\"decisions\":{"
    "\"TYPE\":{\"BODY\":\"\",\"CONN\":\"\",\"ENDS\":\"\",\"SEAL\":\"\",\"MANU\":\"\"},"
    "\"SIZE\":{\"DN\":[],\"OD\":[],\"INCH\":[],\"LENGTH\":[]},"
    "\"PRESSURE\":\"\","
    "\"THICKNESS\":{\"MM\":[],\"INCH\":[],\"SCHEDULE\":[],\"SERIES\":[],\"BWG\":[]},"
    "\"MATERIAL\":{\"RELATION\":\"single|alternative|composite\",\"ITEMS\":[{\"EXEC_STANDARD\":\"\",\"GRADE\":\"\",\"SPECIAL_REQ\":[]}]},"
    "\"STANDARD\":[{\"BODY\":\"\",\"GRADE\":\"\",\"APPENDIX\":\"\",\"METHOD\":\"\"}]"
    "}}。\n"
    "要求："
    "1. 只输出 decisions 对象，不输出 mentions，不输出 semantics；"
    "2. decisions 必须严格保持上述结构：TYPE/SIZE/THICKNESS/MATERIAL 为对象，STANDARD 为对象数组；"
    "3. decisions 用于给出最终结构化决策结果，可以做轻度规范化与结构化拆分；"
    "4. 未识别到的子字段可省略，但不要改变已识别字段的数据类型；"
    "5. 不要输出解释文字，不要输出 markdown，不要补充原文中不存在的信息；"
    "6. 只输出合法 JSON。"
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
