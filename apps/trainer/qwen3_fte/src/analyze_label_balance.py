#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


DEFAULT_INPUT = Path("apps/trainer/qwen3_fte/output/按8类拆分数据集/法兰.json")
DEFAULT_OUTPUT = Path("apps/trainer/qwen3_fte/output/按8类拆分数据集/法兰_标签分布分析.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        norm = normalize_scalar(item)
        if norm is not None:
            result.append(norm)
    return result


def flatten_special_req(materials: Any) -> list[str]:
    if not isinstance(materials, list):
        return []
    out: list[str] = []
    for item in materials:
        if isinstance(item, dict):
            out.extend(normalize_list(item.get("SPECIAL_REQ", [])))
    return out


def flatten_object_field(items: Any, field: str) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            norm = normalize_scalar(item.get(field))
            if norm is not None:
                out.append(norm)
    return out


def get_path(row: dict[str, Any], path: list[str]) -> Any:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def extract_scalar(path: str) -> Callable[[dict[str, Any]], list[str]]:
    parts = path.split(".")

    def _fn(row: dict[str, Any]) -> list[str]:
        norm = normalize_scalar(get_path(row, parts))
        return [norm] if norm is not None else []

    return _fn


def extract_list(path: str) -> Callable[[dict[str, Any]], list[str]]:
    parts = path.split(".")

    def _fn(row: dict[str, Any]) -> list[str]:
        return normalize_list(get_path(row, parts))

    return _fn


FIELD_EXTRACTORS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "TYPE.BODY": extract_scalar("output.TYPE.BODY"),
    "TYPE.GEOMETRY.ANGLE": extract_scalar("output.TYPE.GEOMETRY.ANGLE"),
    "TYPE.GEOMETRY.RADIUS": extract_scalar("output.TYPE.GEOMETRY.RADIUS"),
    "TYPE.MANU": extract_list("output.TYPE.MANU"),
    "TYPE.CONN": extract_list("output.TYPE.CONN"),
    "TYPE.SEAL": extract_list("output.TYPE.SEAL"),
    "TYPE.ENDS": extract_list("output.TYPE.ENDS"),
    "SIZE.DN": extract_list("output.SIZE.DN"),
    "SIZE.OD": extract_list("output.SIZE.OD"),
    "SIZE.INCH": extract_list("output.SIZE.INCH"),
    "SIZE.LENGTH": extract_list("output.SIZE.LENGTH"),
    "THICKNESS.MM": extract_list("output.THICKNESS.MM"),
    "THICKNESS.SCHEDULE": extract_list("output.THICKNESS.SCHEDULE"),
    "THICKNESS.SERIES": extract_list("output.THICKNESS.SERIES"),
    "THICKNESS.BWG": extract_list("output.THICKNESS.BWG"),
    "THICKNESS.INCH": extract_list("output.THICKNESS.INCH"),
    "PRESSURE": extract_scalar("output.PRESSURE"),
    "MATERIAL.ROLE": lambda row: flatten_object_field(get_path(row, ["output", "MATERIAL"]), "ROLE"),
    "MATERIAL.VALUE": lambda row: flatten_object_field(get_path(row, ["output", "MATERIAL"]), "VALUE"),
    "MATERIAL.SPECIAL_REQ": lambda row: flatten_special_req(get_path(row, ["output", "MATERIAL"])),
    "STANDARD.BODY": lambda row: flatten_object_field(get_path(row, ["output", "STANDARD"]), "BODY"),
    "STANDARD.GRADE": lambda row: flatten_object_field(get_path(row, ["output", "STANDARD"]), "GRADE"),
    "STANDARD.METHOD": lambda row: flatten_object_field(get_path(row, ["output", "STANDARD"]), "METHOD"),
    "STANDARD.APPENDIX": lambda row: flatten_object_field(get_path(row, ["output", "STANDARD"]), "APPENDIX"),
}


def calc_distribution_metrics(counter: Counter[str]) -> dict[str, Any]:
    if not counter:
        return {
            "distinct_values": 0,
            "entropy_norm": None,
            "simpson_diversity": None,
            "cv": None,
            "max_min_ratio": None,
            "imbalance_level": "empty",
        }

    counts = list(counter.values())
    total = sum(counts)
    k = len(counts)
    probs = [c / total for c in counts]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    entropy_norm = entropy / math.log(k) if k > 1 else 1.0
    simpson = 1 - sum(p * p for p in probs)
    mean = total / k
    variance = sum((c - mean) ** 2 for c in counts) / k
    cv = (variance ** 0.5) / mean if mean else 0.0
    max_min_ratio = max(counts) / min(counts) if min(counts) else None

    if entropy_norm < 0.55 or cv > 1.5 or (max_min_ratio is not None and max_min_ratio > 25):
        imbalance = "严重不均"
    elif entropy_norm < 0.75 or cv > 1.0 or (max_min_ratio is not None and max_min_ratio > 10):
        imbalance = "中度不均"
    else:
        imbalance = "相对均匀"

    return {
        "distinct_values": k,
        "entropy_norm": round(entropy_norm, 4),
        "simpson_diversity": round(simpson, 4),
        "cv": round(cv, 4),
        "max_min_ratio": round(max_min_ratio, 4) if max_min_ratio is not None else None,
        "imbalance_level": imbalance,
    }


def analyze_field(rows: list[dict[str, Any]], extractor: Callable[[dict[str, Any]], list[str]], topn: int) -> dict[str, Any]:
    value_counter: Counter[str] = Counter()
    non_empty_rows = 0
    multi_value_rows = 0

    for row in rows:
        values = extractor(row)
        if values:
            non_empty_rows += 1
            if len(values) > 1:
                multi_value_rows += 1
            value_counter.update(values)

    total_rows = len(rows)
    metrics = calc_distribution_metrics(value_counter)

    return {
        "total_rows": total_rows,
        "non_empty_rows": non_empty_rows,
        "coverage": round(non_empty_rows / total_rows, 4) if total_rows else 0.0,
        "multi_value_rows": multi_value_rows,
        "total_values": sum(value_counter.values()),
        "top_values": [{"value": k, "count": v} for k, v in value_counter.most_common(topn)],
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计结构化数据集所有标签的分布均匀性")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 JSON 数据集")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出分析 JSON")
    parser.add_argument("--topn", type=int, default=20, help="每个标签输出前 N 个值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json(args.input)
    if not isinstance(rows, list):
        raise TypeError(f"输入不是数组: {args.input}")

    result: dict[str, Any] = {
        "dataset": str(args.input),
        "total_rows": len(rows),
        "fields": {},
    }

    for field_name, extractor in FIELD_EXTRACTORS.items():
        result["fields"][field_name] = analyze_field(rows, extractor, args.topn)

    dump_json(args.output, result)

    print(f"dataset: {args.input}")
    print(f"rows: {len(rows)}")
    for field_name, info in result["fields"].items():
        print(
            f"{field_name}: coverage={info['coverage']:.4f}, "
            f"distinct={info['distinct_values']}, "
            f"imbalance={info['imbalance_level']}"
        )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
