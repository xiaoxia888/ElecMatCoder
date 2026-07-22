#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
QWEN_ROOT = PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte"
DEFAULT_INPUT = (
    QWEN_ROOT
    / "output"
    / "按8类拆分数据集"
    / "type_stage2"
    / "encoding_mappings_type_stage2.json"
)
DEFAULT_OUTPUT_DIR = (
    QWEN_ROOT
    / "output"
    / "按8类拆分llamafactory数据集"
    / "coder_stage2_from_mappings"
)
DEFAULT_SEED = 20260722
FIELD_ORDER = ("MATERIAL", "PRESSURE", "SIZE", "STANDARD", "THICKNESS", "TYPE")
TYPE_INPUT_ORDER = (
    "BODY",
    "ANGLE",
    "RADIUS",
    "FLANGE_STYLE",
    "CONN",
    "SEAL",
    "MANU",
)
INSTRUCTION = (
    "你是工业管道材料字段编码助手。请根据字段类型和规范字段值，"
    "输出唯一的标准化编码。只输出编码，不要解释。"
)


def normalize_type_payload(value: Any, *, code: str, index: int) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"TYPE 编码 {code} 第 {index + 1} 个输入不是有效 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"TYPE 编码 {code} 第 {index + 1} 个输入必须是 JSON 对象")

    unknown_fields = sorted(set(value) - set(TYPE_INPUT_ORDER))
    if unknown_fields:
        raise ValueError(f"TYPE 编码 {code} 包含未知字段: {unknown_fields}")

    normalized: dict[str, Any] = {}
    for field in TYPE_INPUT_ORDER:
        field_value = value.get(field)
        if field_value in (None, "", []):
            continue
        if isinstance(field_value, list):
            items: list[str] = []
            for item in field_value:
                text = str(item).strip()
                if text and text not in items:
                    items.append(text)
            if items:
                normalized[field] = items
            continue
        if isinstance(field_value, dict):
            raise ValueError(f"TYPE 编码 {code} 的字段 {field} 不允许使用对象")
        text = str(field_value).strip()
        if text:
            normalized[field] = text

    if not normalized:
        raise ValueError(f"TYPE 编码 {code} 第 {index + 1} 个输入为空")
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def normalize_value(field: str, value: Any, *, code: str, index: int) -> str:
    if field == "TYPE":
        return normalize_type_payload(value, code=code, index=index)
    if isinstance(value, (dict, list)):
        raise ValueError(f"{field} 编码 {code} 第 {index + 1} 个输入必须是字符串")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} 编码 {code} 第 {index + 1} 个输入为空")
    return text


def make_sample(pair: dict[str, str]) -> dict[str, str]:
    value_label = "规范值" if pair["field"] == "TYPE" else "原始值"
    return {
        "instruction": INSTRUCTION,
        "input": f"字段类型: {pair['field']}\n{value_label}: {pair['value']}",
        "output": pair["code"],
    }


def load_pairs(
    path: Path, selected_fields: tuple[str, ...]
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    if not path.is_file():
        raise FileNotFoundError(f"映射文件不存在: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("映射文件顶层必须是对象")

    pairs: list[dict[str, str]] = []
    metadata: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for field in selected_fields:
        section = document.get(field)
        mappings = section.get("mappings") if isinstance(section, dict) else None
        if not isinstance(mappings, dict):
            raise ValueError(f"输入文件缺少对象节点 {field}.mappings")

        value_to_codes: dict[str, set[str]] = defaultdict(set)
        seen_pairs: set[tuple[str, str]] = set()
        duplicate_count = 0
        for raw_code, raw_values in mappings.items():
            code = str(raw_code).strip()
            if not code:
                raise ValueError(f"{field}.mappings 包含空编码")
            if not isinstance(raw_values, list):
                raise ValueError(f"{field} 编码 {code} 的映射值必须是数组")
            for index, raw_value in enumerate(raw_values):
                value = normalize_value(field, raw_value, code=code, index=index)
                pair_key = (value, code)
                if pair_key in seen_pairs:
                    duplicate_count += 1
                    continue
                seen_pairs.add(pair_key)
                value_to_codes[value].add(code)

        field_pairs: list[dict[str, str]] = []
        field_conflicts = 0
        for value, codes in sorted(value_to_codes.items()):
            if len(codes) > 1:
                field_conflicts += 1
                conflicts.append(
                    {"field": field, "value": value, "candidate_codes": sorted(codes)}
                )
                continue
            field_pairs.append({"field": field, "value": value, "code": next(iter(codes))})
        field_pairs.sort(key=lambda item: (item["code"], item["value"]))
        pairs.extend(field_pairs)
        metadata[field] = {
            "declared_unique_outputs": section.get("unique_outputs"),
            "declared_unique_pairs": section.get("unique_pairs"),
            "parsed_unique_outputs": len({pair["code"] for pair in field_pairs}),
            "parsed_unique_pairs": len(field_pairs),
            "source_unique_input_code_pairs": len(seen_pairs),
            "conflicting_inputs_excluded": field_conflicts,
            "duplicate_pairs_removed": duplicate_count,
        }
    return pairs, metadata, conflicts


def split_pairs(
    pairs: list[dict[str, str]], val_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["field"], pair["code"])].append(pair)

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for group_key in sorted(grouped):
        items = sorted(grouped[group_key], key=lambda item: item["value"])
        rng.shuffle(items)
        if len(items) < 3 or val_ratio == 0:
            train.extend(items)
            continue
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 2)
        val.extend(items[:val_count])
        train.extend(items[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def count_by_field(pairs: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(pair["field"] for pair in pairs).items()))


def count_codes_by_field(pairs: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        counts[pair["field"]][pair["code"]] += 1
    return {
        field: dict(sorted(field_counts.items()))
        for field, field_counts in sorted(counts.items())
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将完整编码映射转换为 LlamaFactory Alpaca 训练集和验证集"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fields", nargs="+", choices=FIELD_ORDER, default=list(FIELD_ORDER))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围内")
    selected_fields = tuple(field for field in FIELD_ORDER if field in args.fields)

    pairs, source_metadata, conflicts = load_pairs(args.input, selected_fields)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    all_labels = {(pair["field"], pair["code"]) for pair in pairs}
    train_labels = {(pair["field"], pair["code"]) for pair in train_pairs}
    val_labels = {(pair["field"], pair["code"]) for pair in val_pairs}
    if all_labels - train_labels:
        raise RuntimeError(f"训练集缺少字段编码: {sorted(all_labels - train_labels)}")

    report = {
        "source": str(args.input.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "fields": list(selected_fields),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "statistics": {
            "all_rows": len(pairs),
            "train_rows": len(train_pairs),
            "val_rows": len(val_pairs),
            "all_field_codes": len(all_labels),
            "train_field_codes": len(train_labels),
            "val_field_codes": len(val_labels),
            "conflicting_inputs_excluded": len(conflicts),
            "val_field_codes_missing_from_train": [
                f"{field}:{code}" for field, code in sorted(val_labels - train_labels)
            ],
            "all_rows_by_field": count_by_field(pairs),
            "train_rows_by_field": count_by_field(train_pairs),
            "val_rows_by_field": count_by_field(val_pairs),
        },
        "source_field_statistics": source_metadata,
        "train_code_counts": count_codes_by_field(train_pairs),
        "val_code_counts": count_codes_by_field(val_pairs),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "coder_stage2_all.json", [make_sample(pair) for pair in pairs])
    write_json(
        args.output_dir / "coder_stage2_train.json",
        [make_sample(pair) for pair in train_pairs],
    )
    write_json(
        args.output_dir / "coder_stage2_val.json",
        [make_sample(pair) for pair in val_pairs],
    )
    write_json(args.output_dir / "coder_stage2_split_report.json", report)
    write_json(args.output_dir / "coder_stage2_conflicts.json", conflicts)

    print(json.dumps(report["statistics"], ensure_ascii=False, indent=2))
    print(f"输出目录: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
