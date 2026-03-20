# -*- coding: utf-8 -*-
"""
从单字段映射表生成编码训练数据

映射表: data/pipe/llm_lora/encoding_mappings.json
输出:   data/pipe/llm_lora/encoding_data.json

注意:
    - 本脚本服务于二阶段编码训练数据生成。
    - 当前 `encoding_mappings.json` / `encoding_data.json` 仍使用二阶段既有输入格式，
      其中 TYPE / MATERIAL / SIZE 主要是字符串值，而不是一阶段新引入的结构化对象。
    - 因此这里的 TYPE 顺序增强逻辑仅适用于分号分隔的字符串 TYPE，
      不负责处理一阶段 `TYPE={"BODY":...}`、`MATERIAL={"RELATION":...}`、
      `SIZE={"DN":...,"LENGTH":...}` 这类新 schema。

使用方法:
    python -m apps.trainer.qwen3_finetune.gen_encoding_data
    python -m apps.trainer.qwen3_finetune.gen_encoding_data --single-repeat 3 --multi-count 2000
"""

import json
import random
import argparse
import itertools
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAPPING_FILE = PROJECT_ROOT / "data/pipe/llm_lora/encoding_mappings.json"
OUTPUT_FILE = PROJECT_ROOT / "data/pipe/llm_lora/encoding_data.json"

MULTI_FIELDS = ["TYPE", "MATERIAL", "STANDARD", "PRESSURE", "SIZE", "THICKNESS"]


def load_mappings(path: Path, skip_fields: set = None):
    """加载映射表，格式: {field: {output_code: [input_val1, input_val2, ...]}}"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mappings = {}
    for field, info in raw.items():
        if skip_fields and field in skip_fields:
            continue
        # 展开为 [(input_value, output_code), ...] 列表
        pairs = []
        for outv, inputs in info["mappings"].items():
            for inv in inputs:
                pairs.append((inv, outv))
        mappings[field] = pairs
    return mappings


def _normalize_type_value(value: str) -> str:
    """统一 TYPE 组合标签格式，消除分号两侧多余空格。"""
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    return ";".join(tokens)


def _build_type_variants(value: str, max_tokens: int, max_permutations: int) -> list[str]:
    """
    为 TYPE 生成顺序置换增强样本。

    只对分号分隔的多标签组合做增强，避免打乱普通短语内部词序。
    返回结果始终包含规范化后的原始值。
    """
    normalized = _normalize_type_value(value)
    if not normalized:
        return []

    tokens = normalized.split(";")
    if len(tokens) < 2 or len(tokens) > max_tokens:
        return [normalized]

    all_variants = []
    seen = set()
    for perm in itertools.permutations(tokens):
        candidate = ";".join(perm)
        if candidate in seen:
            continue
        seen.add(candidate)
        all_variants.append(candidate)

    if normalized in all_variants:
        all_variants.remove(normalized)

    if max_permutations > 0 and len(all_variants) > max_permutations:
        all_variants = random.sample(all_variants, max_permutations)

    return [normalized] + all_variants


def augment_type_mappings(
    mappings: dict,
    enable_type_permute: bool = False,
    type_permute_max_tokens: int = 4,
    type_permute_max_variants: int = 4,
) -> dict:
    """
    仅对 TYPE 字段做顺序置换增强。

    说明：
    - 原始映射始终保留。
    - 顺序置换只用于训练样本增强，不回写映射表。
    - 若增强后出现重复 (input, output) 对，则自动去重。
    """
    if not enable_type_permute or "TYPE" not in mappings:
        return mappings

    augmented = {}
    for field, pairs in mappings.items():
        if field != "TYPE":
            augmented[field] = pairs
            continue

        expanded_pairs = []
        seen_pairs = set()
        for inv, outv in pairs:
            variants = _build_type_variants(
                inv,
                max_tokens=type_permute_max_tokens,
                max_permutations=type_permute_max_variants,
            )
            for variant in variants:
                key = (variant, outv)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                expanded_pairs.append((variant, outv))
        augmented[field] = expanded_pairs

    return augmented


def gen_single_field_samples(mappings: dict, repeat: int = 3) -> list:
    """每个映射对生成 repeat 条单字段样本"""
    samples = []
    for field, pairs in mappings.items():
        for inv, outv in pairs:
            for _ in range(repeat):
                samples.append({
                    "input": {field: inv},
                    "output": {field: outv},
                })
    return samples


def gen_multi_field_samples(mappings: dict, count: int = 2000, min_fields: int = 2, max_fields: int = 4) -> list:
    """随机组合多个字段生成多字段样本"""
    fields = list(mappings.keys())
    samples = []

    for _ in range(count):
        n = random.randint(min_fields, min(max_fields, len(fields)))
        chosen = random.sample(fields, n)
        inp = {}
        out = {}
        for f in chosen:
            inv, outv = random.choice(mappings[f])
            inp[f] = inv
            out[f] = outv
        samples.append({"input": inp, "output": out})

    return samples


def main():
    parser = argparse.ArgumentParser(description="生成编码训练数据")
    parser.add_argument("--mapping-file", type=str, default=str(MAPPING_FILE))
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--single-repeat", type=int, default=3, help="每个单字段映射重复次数")
    parser.add_argument("--multi-count", type=int, default=2000, help="多字段组合样本数")
    parser.add_argument("--min-fields", type=int, default=2, help="多字段样本最少字段数")
    parser.add_argument("--max-fields", type=int, default=4, help="多字段样本最多字段数")
    parser.add_argument("--type-permute", action="store_true", help="仅对 TYPE 字段做顺序置换增强")
    parser.add_argument("--type-permute-max-tokens", type=int, default=4, help="TYPE 顺序增强允许的最大 token 数")
    parser.add_argument("--type-permute-max-variants", type=int, default=4, help="每条 TYPE 映射最多新增多少个置换变体")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    mappings = load_mappings(Path(args.mapping_file))
    mappings = augment_type_mappings(
        mappings,
        enable_type_permute=args.type_permute,
        type_permute_max_tokens=args.type_permute_max_tokens,
        type_permute_max_variants=args.type_permute_max_variants,
    )
    print(f"加载映射表: {sum(len(pairs) for pairs in mappings.values())} 个映射, {len(mappings)} 个字段")
    for f, pairs in mappings.items():
        print(f"  {f}: {len(pairs)} 个映射")

    single = gen_single_field_samples(mappings, repeat=args.single_repeat)
    print(f"\n单字段样本: {len(single)} 条 (每对 x{args.single_repeat})")

    multi = gen_multi_field_samples(
        mappings, count=args.multi_count,
        min_fields=args.min_fields, max_fields=args.max_fields,
    )
    print(f"多字段样本: {len(multi)} 条")

    all_samples = single + multi
    random.shuffle(all_samples)
    print(f"总计: {len(all_samples)} 条")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"已写入: {args.output_file}")


if __name__ == "__main__":
    main()
