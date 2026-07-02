#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将结构抽取训练集 JSON 转换为 LlamaFactory 所需的 instruction/input/output 格式，
并按“表达形式覆盖优先”的方式切分 train / val。

输入 JSON 期望格式：
[
  {
    "input": "材料描述",
    "output": {
      "SIZE_ITEMS": [...],
      "LENGTH": "",
      "THICKNESS_ITEMS": [...],
      "PRESSURE": ""
    }
  }
]

输出：
- train.json
- val.json

示例：
python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/尺寸壁厚磅级训练集转换结构抽取训练集为LlamaFactory.py \
  --input /Users/guoxi/Documents/尺寸壁厚磅级C1训练集.json \
  --val-ratio 0.1 \
  --output-dir /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/尺寸壁厚磅级

  

  
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION = (
    "你是工业管道材料描述结构化抽取助手。请从材料描述中抽取尺寸、长度、壁厚和磅级信息，并输出严格 JSON。"
    "输出字段只能包含 SIZE_ITEMS、LENGTH、THICKNESS_ITEMS、PRESSURE。"
    "LENGTH 统一转换为毫米单位，SIZE_ITEMS 和 THICKNESS_ITEMS 按原文顺序输出，不要解释，不要输出 JSON 以外的内容。"
)


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入文件必须是 JSON 数组。")
    return data


def compact_output(output_obj: dict[str, Any]) -> str:
    return json.dumps(output_obj, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    return str(text or "").upper()


def add_if_match(features: set[str], text: str, name: str, pattern: str) -> None:
    if re.search(pattern, text, flags=re.IGNORECASE):
        features.add(name)


def extract_surface_features(input_text: str, output_obj: dict[str, Any]) -> set[str]:
    text = normalize_text(input_text)
    features: set[str] = set()

    size_items = output_obj.get("SIZE_ITEMS") or []
    thickness_items = output_obj.get("THICKNESS_ITEMS") or []
    length_value = str(output_obj.get("LENGTH") or "").strip()
    pressure_value = str(output_obj.get("PRESSURE") or "").strip()

    if size_items:
        features.add("has_size")
    if thickness_items:
        features.add("has_thickness")
    if length_value:
        features.add("has_length")
    if pressure_value:
        features.add("has_pressure")

    size_types = {str(item.get("type") or "").upper() for item in size_items if isinstance(item, dict)}
    thickness_types = {str(item.get("type") or "").upper() for item in thickness_items if isinstance(item, dict)}

    for size_type in sorted(size_types):
        features.add(f"size_type:{size_type}")
    for thickness_type in sorted(thickness_types):
        features.add(f"thickness_type:{thickness_type}")

    if len(size_items) >= 2:
        features.add("size_multi")
    if len(thickness_items) >= 2:
        features.add("thickness_multi")

    if any(item.get("type") == "DN" for item in size_items if isinstance(item, dict)):
        add_if_match(features, text, "size_surface:dn_prefix", r"\bDN\s*\d")
        add_if_match(features, text, "size_surface:dn_compound", r"\bDN\s*\d+\s*[X×]\s*\d+")
    if any(item.get("type") == "OD" for item in size_items if isinstance(item, dict)):
        add_if_match(features, text, "size_surface:od_symbol", r"[ΦφØø]\s*\d")
        add_if_match(features, text, "size_surface:od_prefix", r"\bOD\s*\d")
        add_if_match(features, text, "size_surface:od_pair", r"\b\d+(?:\.\d+)?\s*[X×]\s*\d+(?:\.\d+)?")
    if any(item.get("type") == "INCH" for item in size_items if isinstance(item, dict)):
        add_if_match(features, text, "size_surface:inch_quote", r"\d+(?:\.\d+)?\s*[\"”″]")
        add_if_match(features, text, "size_surface:inch_fraction", r"\b\d+/\d+\s*[\"”″]?\b")
        add_if_match(features, text, "size_surface:inch_mixed_fraction", r"\b\d+\s+\d+/\d+\s*[\"”″]?\b")
        add_if_match(features, text, "size_surface:nps_prefix", r"\bNPS\s*\d")

    if length_value:
        add_if_match(features, text, "length_surface:mm", r"\b(?:LENGTH|LEN|L)\s*[:=]?\s*\d+(?:\.\d+)?\s*MM\b")
        add_if_match(features, text, "length_surface:number_mm", r"\b\d+(?:\.\d+)?\s*MM\b")

    if any(item.get("type") == "MM" for item in thickness_items if isinstance(item, dict)):
        add_if_match(features, text, "thickness_surface:mm_decimal", r"\b\d+\.\d+\s*(?:MM|OMM)\b")
        add_if_match(features, text, "thickness_surface:mm_integer", r"\b\d+\s*(?:MM|OMM)\b")
        add_if_match(features, text, "thickness_surface:thk_marker", r"\bTHK\s*[:=\-]?\s*\d")
        add_if_match(features, text, "thickness_surface:t_marker", r"(?<![A-Z])T\s*[:=\-]?\s*\d")
    if any(item.get("type") == "SCHEDULE" for item in thickness_items if isinstance(item, dict)):
        add_if_match(features, text, "thickness_surface:sch_plain", r"\bSCH\d+(?:\.\d+)?S?\b")
        add_if_match(features, text, "thickness_surface:sch_space", r"\bSCH\s+\d+(?:\.\d+)?S?\b")
        add_if_match(features, text, "thickness_surface:sch_dot", r"\bSCH\.\s*\d+(?:\.\d+)?S?\b")
        add_if_match(features, text, "thickness_surface:s_dash", r"\bS-\d+(?:\.\d+)?S?\b")
        add_if_match(features, text, "thickness_surface:s_suffix", r"(?<![A-Z])\d+(?:\.\d+)?S\b")
        add_if_match(features, text, "thickness_surface:std", r"\bSTD\b")
        add_if_match(features, text, "thickness_surface:xs", r"\bXS\b")
        add_if_match(features, text, "thickness_surface:xxs", r"\bXXS\b")
    if any(item.get("type") == "BWG" for item in thickness_items if isinstance(item, dict)):
        add_if_match(features, text, "thickness_surface:bwg", r"\b\d+\s*BWG\b")
    if any(item.get("type") == "SERIES" for item in thickness_items if isinstance(item, dict)):
        add_if_match(features, text, "thickness_surface:series", r"\b(?:SERIES|SER)\b")

    if len(size_items) >= 2:
        add_if_match(features, text, "size_surface:multi_x", r"[X×]")
    if len(thickness_items) >= 2:
        add_if_match(features, text, "thickness_surface:multi_x", r"[X×]")

    if pressure_value:
        add_if_match(features, text, "pressure_surface:pn", r"\bPN\s*\d+(?:\.\d+)?\b")
        add_if_match(features, text, "pressure_surface:cl", r"\bCL\s*\d+\b")
        add_if_match(features, text, "pressure_surface:c", r"\bC\d+\b")
        add_if_match(features, text, "pressure_surface:lbs", r"\b\d+\s*(?:#|LBS?|LB)\b")
        add_if_match(features, text, "pressure_surface:mpa", r"\b\d+(?:\.\d+)?\s*MPA\b")
        add_if_match(features, text, "pressure_surface:bar", r"\b\d+(?:\.\d+)?\s*BAR\b")
        add_if_match(features, text, "pressure_surface:dual_notation", r"\b\d+\s*CL\s*\(\s*PN\d+\s*\)")

    return features


def build_llamafactory_record(record: dict[str, Any], instruction: str) -> dict[str, str]:
    input_text = str(record.get("input") or "").strip()
    output_obj = record.get("output")
    if not input_text:
        raise ValueError("样本缺少 input。")
    if not isinstance(output_obj, dict):
        raise ValueError("样本缺少 output 对象。")
    return {
        "instruction": instruction,
        "input": input_text,
        "output": compact_output(output_obj),
    }


def choose_validation_indices(
    records: list[dict[str, Any]],
    val_count: int,
    seed: int,
) -> list[int]:
    if val_count <= 0:
        return []
    if val_count >= len(records):
        return list(range(len(records)))

    features_per_index = [extract_surface_features(rec["input"], rec["output"]) for rec in records]
    feature_freq = Counter(feature for features in features_per_index for feature in features)
    uncovered = set(feature_freq)
    selected: list[int] = []
    selected_set: set[int] = set()
    rng = random.Random(seed)

    all_indices = list(range(len(records)))

    def rarity_score(index: int, feature_subset: set[str] | None = None) -> float:
        features = feature_subset if feature_subset is not None else features_per_index[index]
        return sum(1.0 / feature_freq[f] for f in features if feature_freq[f] > 0)

    # 第一阶段：贪心覆盖尽量多的表达形式
    while len(selected) < val_count and uncovered:
        best_index = None
        best_gain = -1
        best_rarity = -1.0
        for index in all_indices:
            if index in selected_set:
                continue
            gain_features = features_per_index[index] & uncovered
            gain = len(gain_features)
            if gain <= 0:
                continue
            rarity = rarity_score(index, gain_features)
            if (
                gain > best_gain
                or (gain == best_gain and rarity > best_rarity)
                or (gain == best_gain and abs(rarity - best_rarity) < 1e-12 and rng.random() < 0.5)
            ):
                best_index = index
                best_gain = gain
                best_rarity = rarity

        if best_index is None:
            break
        selected.append(best_index)
        selected_set.add(best_index)
        uncovered -= features_per_index[best_index]

    # 第二阶段：剩余名额优先放稀有表达
    remaining = [idx for idx in all_indices if idx not in selected_set]
    remaining.sort(
        key=lambda idx: (
            -rarity_score(idx),
            -len(features_per_index[idx]),
            idx,
        )
    )
    for index in remaining:
        if len(selected) >= val_count:
            break
        selected.append(index)
        selected_set.add(index)

    selected.sort()
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将结构抽取训练集转换为 LlamaFactory 的 train/val JSON。")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--train-name", default="train.json", help="训练集文件名")
    parser.add_argument("--val-name", default="val.json", help="验证集文件名")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="固定 instruction 文案")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_json(input_path)
    records: list[dict[str, Any]] = []
    skipped = 0
    for record in raw_records:
        try:
            lf_record = build_llamafactory_record(record, args.instruction)
        except Exception:
            skipped += 1
            continue
        records.append(
            {
                "input": lf_record["input"],
                "output": json.loads(lf_record["output"]),
                "llamafactory": lf_record,
            }
        )

    total = len(records)
    if total == 0:
        raise ValueError("没有可转换的有效样本。")

    val_count = max(1, int(round(total * args.val_ratio))) if args.val_ratio > 0 else 0
    val_count = min(val_count, total)
    val_indices = set(choose_validation_indices(records, val_count, args.seed))

    train_records = [records[i]["llamafactory"] for i in range(total) if i not in val_indices]
    val_records = [records[i]["llamafactory"] for i in range(total) if i in val_indices]

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")

    train_features = set()
    val_features = set()
    all_features = set()
    for i, record in enumerate(records):
        features = extract_surface_features(record["input"], record["output"])
        all_features |= features
        if i in val_indices:
            val_features |= features
        else:
            train_features |= features

    coverage_ratio = (len(val_features) / len(all_features) * 100.0) if all_features else 100.0
    print(f"输入样本数: {len(raw_records)}")
    print(f"有效样本数: {total}")
    print(f"跳过样本数: {skipped}")
    print(f"训练集: {len(train_records)}")
    print(f"验证集: {len(val_records)}")
    print(f"验证集占比: {len(val_records) / total:.2%}")
    print(f"验证集表达覆盖率: {coverage_ratio:.2f}% ({len(val_features)}/{len(all_features)})")
    print(f"已生成训练集: {train_path}")
    print(f"已生成验证集: {val_path}")


if __name__ == "__main__":
    main()
