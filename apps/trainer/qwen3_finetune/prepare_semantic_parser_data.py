# -*- coding: utf-8 -*-
"""
一阶段语义解析器训练数据准备。

维护层建议分成:
- main
- contrast
- noise
- hard
- eval_frozen
- eval_hard

训练前由本脚本合并 train 类数据，并生成 ChatML JSONL。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_ner.prompts import SEMANTIC_PARSER_SYSTEM_PROMPT


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_chatml(user_text: str, assistant_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SEMANTIC_PARSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
        ]
    }


def dedupe_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    removed = 0
    for sample in samples:
        key = json.dumps(
            {"input": sample.get("input"), "output": sample.get("output")},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(sample)
    return out, removed


def load_config(config_path: str | None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path else Path(__file__).with_name("config_semantic_parser.yaml")
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_optional_source(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_json_or_jsonl(path)


def build_train_samples(data_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources = {
        "main": resolve(data_cfg["main_source"]),
        "contrast": resolve(data_cfg["contrast_source"]),
        "noise": resolve(data_cfg["noise_source"]),
        "hard": resolve(data_cfg["hard_source"]),
    }
    merged: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    for name, path in sources.items():
        items = load_optional_source(path)
        stats[name] = len(items)
        merged.extend(items)
    merged, removed = dedupe_samples(merged)
    stats["deduped_removed"] = removed
    stats["merged"] = len(merged)
    return merged, stats


def build_eval_samples(data_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = load_optional_source(resolve(data_cfg["eval_frozen_source"]))
    hard = load_optional_source(resolve(data_cfg["eval_hard_source"]))
    return frozen, hard


def main() -> None:
    parser = argparse.ArgumentParser(description="准备一阶段语义解析器训练数据")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    split_seed = int(args.seed if args.seed is not None else data_cfg.get("split_seed", 42))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else data_cfg.get("val_ratio", 0.1))
    random.seed(split_seed)

    train_samples, train_stats = build_train_samples(data_cfg)
    if not train_samples:
        raise SystemExit("未找到任何一阶段语义解析训练样本。")

    frozen_eval, hard_eval = build_eval_samples(data_cfg)

    random.shuffle(train_samples)
    if frozen_eval:
        val_samples = frozen_eval
        train_only = train_samples
    else:
        val_count = int(len(train_samples) * val_ratio)
        val_samples = train_samples[:val_count]
        train_only = train_samples[val_count:]

    train_chatml = [to_chatml(item["input"], item["output"]) for item in train_only]
    val_chatml = [to_chatml(item["input"], item["output"]) for item in val_samples]
    hard_chatml = [to_chatml(item["input"], item["output"]) for item in hard_eval]

    prepared_dir = resolve(data_cfg["prepared_dir"])
    train_file = resolve(data_cfg["train_file"])
    val_file = resolve(data_cfg["val_file"])
    hard_file = prepared_dir / "eval_hard.jsonl"
    summary_file = prepared_dir / "summary.json"

    write_jsonl(train_file, train_chatml)
    write_jsonl(val_file, val_chatml)
    if hard_chatml:
        write_jsonl(hard_file, hard_chatml)

    summary = {
        "train_sources": train_stats,
        "eval_frozen": len(frozen_eval),
        "eval_hard": len(hard_eval),
        "train_chatml": len(train_chatml),
        "val_chatml": len(val_chatml),
        "hard_chatml": len(hard_chatml),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "hard_file": str(hard_file) if hard_chatml else "",
        "seed": split_seed,
        "val_ratio": val_ratio,
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

