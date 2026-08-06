#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILES = ["管件.json", "法兰.json", "直管.json"]
OUTPUT_FILE = "种类0629.json"
SKELETON_DIR = Path(__file__).resolve().parents[3] / "skeletons"


def load_json_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    return data


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return data


def deep_clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: deep_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_clone(v) for v in value]
    return value


def apply_skeleton(data: Any, skeleton: Any) -> Any:
    """按骨架裁剪并补全结构。"""
    if isinstance(skeleton, dict):
        source = data if isinstance(data, dict) else {}
        return {
            key: apply_skeleton(source.get(key), skeleton_value)
            for key, skeleton_value in skeleton.items()
        }
    if isinstance(skeleton, list):
        return deep_clone(data) if isinstance(data, list) else deep_clone(skeleton)
    if data is None:
        return deep_clone(skeleton)
    return deep_clone(data)


def transform_record(record: dict[str, Any], output_skeleton: dict[str, Any]) -> dict[str, Any]:
    output = record.get("output")
    if not isinstance(output, dict):
        output = {}

    return {
        "input": record.get("input", ""),
        "output": apply_skeleton(output, output_skeleton),
    }


def merge_type_datasets() -> Path:
    merged: list[dict] = []
    for filename in SOURCE_FILES:
        src = BASE_DIR / filename
        skeleton_path = SKELETON_DIR / filename
        output_skeleton = load_json_object(skeleton_path)

        records = load_json_list(src)
        merged.extend(transform_record(record, output_skeleton) for record in records)

    output_path = BASE_DIR / OUTPUT_FILE
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    output_path = merge_type_datasets()
    print(f"MERGED {output_path}")


if __name__ == "__main__":
    main()
