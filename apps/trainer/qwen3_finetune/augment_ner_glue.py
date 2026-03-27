# -*- coding: utf-8 -*-
"""
NER 粘连数据增强模块

从已有带分隔符的样本中自动生成"粘连"变体，提升 NER 模型边界识别能力。

原理：
  原样本: "管帽 DN32 CF415 SH/T 3410"  → {"MATERIAL":"CF415", "STANDARD":"SH/T 3410"}
  增强后: "管帽 DN32 CF415SH/T 3410"   → {"MATERIAL":"CF415", "STANDARD":"SH/T 3410"}
                       ↑ 删除分隔符
  标签不变，只改输入文本。模型学到：即使粘在一起，也要正确切分。

增强策略（替换式，非追加式）：
  对每条可粘连的样本，以概率 P 用粘连变体"替换"原样本。
  总数据量不变，数据分布不变，只是部分样本变成了粘连形式。

使用方式：
  1. 作为模块被 prepare_training_data.py 调用（推荐）
  2. 独立运行进行分析：
     python -m apps.trainer.qwen3_finetune.augment_ner_glue --analyze
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

HIGH_RISK_PAIRS = [
    ('MATERIAL', 'STANDARD'),
    ('STANDARD', 'STANDARD'),
    ('STANDARD', 'STANDARD_GRADE'),
    ('STANDARD', 'STANDARD_APPENDIX'),
    ('STANDARD', 'STANDARD_METHOD'),
    ('STANDARD', 'SIZE'),
    ('CONN', 'STANDARD'),
    ('THICKNESS', 'SIZE'),
    ('SEAL', 'STANDARD'),
    ('ENDS', 'STANDARD'),
    ('PRESSURE', 'STANDARD'),
    ('MATERIAL', 'SIZE'),
]

SEPARATOR_PATTERN = re.compile(r'^[\s,;，；、\-]+$')


def load_ner_data(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _collect_leaf_values(field: str, value, out: List[Tuple[str, str]]) -> None:
    """
    递归收集可定位到 input 中的字符串叶子值。
    - 保留顶层 field 作为字段归属，兼容新旧 schema。
    - 对 {"value": "..."} 结构优先取 value 字段。
    """
    if value is None:
        return

    if isinstance(value, str):
        v = value.strip()
        if v:
            out.append((field, v))
        return

    if isinstance(value, (int, float, bool)):
        out.append((field, str(value)))
        return

    if isinstance(value, list):
        for item in value:
            _collect_leaf_values(field, item, out)
        return

    if isinstance(value, dict):
        if "value" in value and isinstance(value.get("value"), str):
            _collect_leaf_values(field, value.get("value"), out)
            return
        for _, sub_v in value.items():
            _collect_leaf_values(field, sub_v, out)
        return


def locate_values_in_text(text: str, output: dict) -> List[Tuple[int, int, str, str]]:
    """
    在 input 文本中定位每个 output 字段值的位置。

    Returns:
        [(start, end, field_type, value), ...] 按 start 排序
    """
    positions = []

    for field, value in output.items():
        leaf_values: List[Tuple[str, str]] = []
        _collect_leaf_values(field, value, leaf_values)
        for f, v in leaf_values:
            _find_and_add(text, v, f, positions)

    positions.sort(key=lambda x: x[0])
    return positions


def _find_and_add(text: str, value: str, field: str, positions: list):
    if not value:
        return

    idx = text.find(value)
    if idx >= 0:
        positions.append((idx, idx + len(value), field, value))
    else:
        idx_lower = text.lower().find(value.lower())
        if idx_lower >= 0:
            positions.append((idx_lower, idx_lower + len(value), field, value))


def find_glueable_pairs(
    text: str,
    positions: List[Tuple[int, int, str, str]]
) -> List[Tuple[int, int, int, str, str]]:
    """
    找到可以粘连的相邻字段对。

    Returns:
        [(pos1_end, sep_start, pos2_start, field1, field2), ...]
    """
    pairs = []

    for i in range(len(positions) - 1):
        _, end1, field1, _ = positions[i]
        start2, _, field2, _ = positions[i + 1]

        if end1 >= start2:
            continue

        sep = text[end1:start2]

        if not sep or len(sep) > 3:
            continue

        if not SEPARATOR_PATTERN.match(sep):
            continue

        pair_key = (field1, field2)
        if pair_key in HIGH_RISK_PAIRS:
            pairs.append((end1, end1, start2, field1, field2))

    return pairs


def generate_glued_sample(
    sample: dict,
    text: str,
    pairs: List[Tuple[int, int, int, str, str]],
    max_glue: int = 1
) -> Optional[dict]:
    """
    生成粘连变体：随机选择 1-max_glue 个分隔符删除。
    从后往前删除，避免位置偏移。
    """
    if not pairs:
        return None

    n_glue = min(len(pairs), max_glue)
    selected = sorted(random.sample(pairs, n_glue), key=lambda x: x[0], reverse=True)

    new_text = text
    for _, sep_start, sep_end, _, _ in selected:
        new_text = new_text[:sep_start] + new_text[sep_end:]

    if new_text == text:
        return None

    return {
        "input": new_text,
        "output": sample["output"]
    }


# ─── 对外接口：替换式增强 ──────────────────────────────────────────

def augment_ner_samples_inplace(
    samples: list,
    glue_prob: float = 0.3,
    seed: int = 42
) -> Tuple[list, dict]:
    """
    替换式增强：对可粘连的样本，以概率 glue_prob 用粘连变体替换原样本。

    特点：
      - 总数据量不变（替换，非追加）
      - 数据分布不变（每条样本要么是原始版，要么是粘连版）
      - 粘连样本占比约 glue_prob * 可粘连样本占比

    Args:
        samples: 原始 NER 样本列表 [{"input": ..., "output": ...}, ...]
        glue_prob: 对可粘连样本的替换概率 (0~1)
        seed: 随机种子

    Returns:
        (augmented_samples, stats)
        - augmented_samples: 替换后的样本列表（长度与输入相同）
        - stats: {"total": 总数, "glueable": 可粘连数, "replaced": 实际替换数,
                  "pair_counts": {字段对: 替换数}}
    """
    rng = random.Random(seed)
    result = []
    glueable_count = 0
    replaced_count = 0
    pair_counts = defaultdict(int)

    for sample in samples:
        text = sample.get("input", "")
        output = sample.get("output", {})

        if not text or not output:
            result.append(sample)
            continue

        positions = locate_values_in_text(text, output)
        pairs = find_glueable_pairs(text, positions)

        if not pairs:
            result.append(sample)
            continue

        glueable_count += 1

        if rng.random() < glue_prob:
            glued = generate_glued_sample(sample, text, pairs, max_glue=1)
            if glued and glued["input"] != text:
                result.append(glued)
                replaced_count += 1
                selected_pair = rng.choice(pairs)
                pair_counts[f"{selected_pair[3]}+{selected_pair[4]}"] += 1
                continue

        result.append(sample)

    stats = {
        "total": len(samples),
        "glueable": glueable_count,
        "replaced": replaced_count,
        "pair_counts": dict(pair_counts),
    }
    return result, stats


# ─── 独立分析命令 ──────────────────────────────────────────────────

def analyze(data: list) -> Dict[str, list]:
    """分析模式：统计各字段对可增强的样本数量"""
    pair_stats = defaultdict(list)

    for i, sample in enumerate(data):
        text = sample.get("input", "")
        output = sample.get("output", {})

        if not text or not output:
            continue

        positions = locate_values_in_text(text, output)
        pairs = find_glueable_pairs(text, positions)

        for _, _, _, f1, f2 in pairs:
            pair_key = f"{f1}+{f2}"
            pair_stats[pair_key].append(i)

    return pair_stats


def main():
    parser = argparse.ArgumentParser(description="NER 粘连数据增强 — 分析工具")
    parser.add_argument("--analyze", action="store_true", default=True,
                        help="分析可增强的样本（默认）")
    parser.add_argument("--ner_data", type=str,
                        default="data/pipe/llm_lora/ner_data_new_schema.json",
                        help="NER 数据路径")
    args = parser.parse_args()

    ner_path = PROJECT_ROOT / args.ner_data
    if not ner_path.exists():
        print(f"文件不存在: {ner_path}")
        return

    data = load_ner_data(ner_path)
    print(f"加载 NER 数据: {len(data)} 条\n")

    pair_stats = analyze(data)
    print(f"{'字段对':<30} {'可增强样本数':>10}")
    print("-" * 45)
    total = 0
    for pair_key in sorted(pair_stats.keys(), key=lambda k: -len(pair_stats[k])):
        count = len(pair_stats[pair_key])
        total += count
        print(f"{pair_key:<30} {count:>10}")

        for idx in pair_stats[pair_key][:3]:
            sample = data[idx]
            text = sample["input"]
            if len(text) > 80:
                text = text[:80] + "..."
            print(f"  例: {text}")

    print("-" * 45)
    print(f"{'合计':<30} {total:>10}")
    print(f"\n提示: 增强在 prepare_training_data.py 中通过 --augment 开关启用")
    print(f"  python prepare_training_data.py --augment")
    print(f"  python prepare_training_data.py --augment --glue_prob 0.4")


if __name__ == "__main__":
    main()
