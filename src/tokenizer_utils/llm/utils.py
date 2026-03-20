"""
工具函数模块
"""

import json
import re
import logging
from typing import List, Dict, Optional, Tuple

from .models import Entity, EntityType

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    从文本中提取JSON对象
    
    Args:
        text: 可能包含JSON的文本
        
    Returns:
        提取的JSON对象，如果失败返回None
    """
    if not text or not text.strip():
        logger.warning("输入文本为空")
        return None
    
    # 打印原始响应用于调试
    logger.info(f"LLM原始响应: {text[:500]}...")
    
    # 首先尝试直接解析
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # 尝试提取```json```代码块中的内容
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    
    # 尝试提取第一个完整的{...}结构（处理嵌套括号）
    def find_json_object(s):
        start = s.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i, c in enumerate(s[start:], start):
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None
    
    json_str = find_json_object(text)
    if json_str:
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, json_str={json_str[:200]}...")
    
    # 尝试修复常见的JSON格式问题
    # 1. 移除可能的思考过程（qwen3的<think>标签）
    text_cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text_cleaned = re.sub(r'```\w*\n?', '', text_cleaned)  # 移除代码块标记
    
    json_str = find_json_object(text_cleaned)
    if json_str:
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    logger.warning(f"无法从文本中提取JSON: {text[:300]}...")
    return None


def validate_entity(entity_dict: Dict, original_text: str) -> Optional[Entity]:
    """
    验证并创建Entity对象
    
    Args:
        entity_dict: 实体字典
        original_text: 原始文本
        
    Returns:
        验证通过的Entity对象，失败返回None
    """
    try:
        # 检查必要字段
        if "text" not in entity_dict or "label" not in entity_dict:
            logger.warning(f"实体缺少必要字段: {entity_dict}")
            return None
        
        text = entity_dict["text"]
        label = entity_dict["label"]
        
        # 验证标签是否有效
        try:
            entity_type = EntityType(label)
        except ValueError:
            logger.warning(f"无效的实体标签: {label}")
            return None
        
        # 获取或计算位置
        start = entity_dict.get("start")
        end = entity_dict.get("end")
        
        # 如果没有位置信息，尝试在原文中查找
        if start is None or end is None:
            found_start = original_text.find(text)
            if found_start != -1:
                start = found_start
                end = found_start + len(text)
            else:
                # 尝试忽略空格的匹配
                start, end = find_entity_position(original_text, text)
        
        return Entity(
            text=text,
            label=entity_type,
            start=start,
            end=end,
        )
        
    except Exception as e:
        logger.error(f"验证实体失败: {e}, entity_dict={entity_dict}")
        return None


def find_entity_position(text: str, entity_text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    在文本中查找实体的位置
    
    Args:
        text: 原始文本
        entity_text: 实体文本
        
    Returns:
        (start, end) 位置元组
    """
    # 直接查找
    idx = text.find(entity_text)
    if idx != -1:
        return idx, idx + len(entity_text)
    
    # 尝试标准化后查找（处理×和*的差异）
    normalized_text = normalize_spec_chars(text)
    normalized_entity = normalize_spec_chars(entity_text)
    
    idx = normalized_text.find(normalized_entity)
    if idx != -1:
        return idx, idx + len(entity_text)
    
    return None, None


def normalize_spec_chars(text: str) -> str:
    """
    标准化规格字符（将各种乘号统一）
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    # 统一乘号：×、*、x、X 都转为 ×
    text = re.sub(r"[*xX×]", "×", text)
    # 统一大小写
    text = text.upper()
    return text


def merge_overlapping_entities(entities: List[Entity]) -> List[Entity]:
    """
    合并重叠的实体（保留更长的）
    
    Args:
        entities: 实体列表
        
    Returns:
        合并后的实体列表
    """
    if not entities:
        return []
    
    # 按start排序
    sorted_entities = sorted(
        [e for e in entities if e.start is not None],
        key=lambda x: (x.start, -(x.end or 0))
    )
    
    if not sorted_entities:
        return entities
    
    merged = [sorted_entities[0]]
    
    for entity in sorted_entities[1:]:
        last = merged[-1]
        
        # 检查是否重叠
        if entity.start < last.end:
            # 重叠，保留更长的
            if (entity.end - entity.start) > (last.end - last.start):
                merged[-1] = entity
        else:
            merged.append(entity)
    
    return merged


def convert_to_bio_tags(text: str, entities: List[Entity]) -> List[Tuple[str, str]]:
    """
    将实体列表转换为BIO标签序列
    
    Args:
        text: 原始文本
        entities: 实体列表
        
    Returns:
        [(字符, 标签), ...] 列表
    """
    # 初始化所有字符为O
    tags = ["O"] * len(text)
    
    # 按位置排序
    sorted_entities = sorted(
        [e for e in entities if e.start is not None and e.end is not None],
        key=lambda x: x.start
    )
    
    for entity in sorted_entities:
        for i in range(entity.start, min(entity.end, len(text))):
            if i == entity.start:
                tags[i] = f"B-{entity.label}"
            else:
                tags[i] = f"I-{entity.label}"
    
    return list(zip(text, tags))


def bio_tags_to_entities(text: str, tags: List[str]) -> List[Entity]:
    """
    将BIO标签序列转换为实体列表
    
    Args:
        text: 原始文本
        tags: 标签序列
        
    Returns:
        实体列表
    """
    entities = []
    current_entity = None
    
    for i, (char, tag) in enumerate(zip(text, tags)):
        if tag.startswith("B-"):
            # 保存之前的实体
            if current_entity:
                entities.append(current_entity)
            
            # 开始新实体
            label = tag[2:]
            current_entity = {
                "text": char,
                "label": label,
                "start": i,
                "end": i + 1,
            }
        elif tag.startswith("I-") and current_entity:
            # 继续当前实体
            expected_label = tag[2:]
            if current_entity["label"] == expected_label:
                current_entity["text"] += char
                current_entity["end"] = i + 1
            else:
                # 标签不匹配，保存当前实体，开始新实体
                entities.append(current_entity)
                current_entity = {
                    "text": char,
                    "label": expected_label,
                    "start": i,
                    "end": i + 1,
                }
        else:
            # O标签，保存当前实体
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # 保存最后一个实体
    if current_entity:
        entities.append(current_entity)
    
    return [
        Entity(
            text=e["text"],
            label=EntityType(e["label"]),
            start=e["start"],
            end=e["end"],
        )
        for e in entities
    ]


def format_annotation_result(text: str, entities: List[Entity]) -> str:
    """
    格式化标注结果为可读字符串
    
    Args:
        text: 原始文本
        entities: 实体列表
        
    Returns:
        格式化的字符串
    """
    lines = [f"原文: {text}", "实体:"]
    
    for entity in entities:
        position = f"[{entity.start}:{entity.end}]" if entity.start is not None else ""
        lines.append(f"  - {entity.text} ({entity.label}) {position}")
    
    if not entities:
        lines.append("  (无实体)")
    
    return "\n".join(lines)

