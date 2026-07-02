#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path("data/manual_samples/法兰管件描述样本.txt")
DEFAULT_OUTPUT_FILE = Path("outputs/种类数据集_法兰管件.json")


def normalize_text(text: str) -> str:
    value = str(text or "").strip()
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;")


def uniq_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_fitting_type(
    *,
    body: str,
    angle: str = "",
    radius: str = "",
    flange_style: str = "",
    manu: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "BODY": body,
        "GEOMETRY": {
            "ANGLE": angle,
            "RADIUS": radius,
        },
        "MANU": uniq_keep_order(manu or []),
        "CONN": [],
    }
    if flange_style:
        payload["FLANGE_STYLE"] = flange_style
    return payload


def build_flange_type(desc: str) -> dict[str, Any]:
    upper = desc.upper()
    payload: dict[str, Any] = {
        "BODY": "带颈平焊法兰",
        "CONN": [],
    }
    if "RF" in upper:
        payload["SEAL"] = ["RF"]
    return payload


def infer_type(desc: str) -> dict[str, Any]:
    upper = desc.upper()
    manu: list[str] = []
    if "SMLS" in upper:
        manu.append("SMLS")
    if "WELDED" in upper or "焊接" in desc:
        manu.append("WELDED")

    if desc.startswith("45度法兰弯头"):
        return build_fitting_type(body="弯头", angle="45", flange_style="FLANGED", manu=manu)
    if desc.startswith("90度法兰弯头"):
        return build_fitting_type(body="弯头", angle="90", flange_style="FLANGED", manu=manu)
    if desc.startswith("法兰等径三通"):
        return build_fitting_type(body="等径三通", flange_style="FLANGED", manu=manu)
    if desc.startswith("法兰异径三通"):
        return build_fitting_type(body="异径三通", flange_style="FLANGED", manu=manu)
    if desc.startswith("法兰偏心异径管"):
        return build_fitting_type(body="偏心异径管", flange_style="FLANGED", manu=manu)
    if desc.startswith("法兰同心异径管"):
        return build_fitting_type(body="同心异径管", flange_style="FLANGED", manu=manu)
    if desc.startswith("带颈平焊法兰"):
        return build_flange_type(desc)
    if desc.startswith("焊接同心异径管接头"):
        return build_fitting_type(body="同心异径管接头", manu=manu)
    raise ValueError(f"未覆盖的描述模式: {desc}")


def load_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [normalize_text(line) for line in lines if normalize_text(line)]


def build_dataset(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        rows.append(
            {
                "input": line,
                "output": {
                    "TYPE": infer_type(line),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="根据一批法兰管件描述生成 TYPE 种类数据集")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="一行一条描述的文本文件")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="输出 json 文件")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    lines = load_lines(args.input_file)
    dataset = build_dataset(lines)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    counter = Counter()
    for row in dataset:
        type_obj = row["output"]["TYPE"]
        key = type_obj.get("BODY", "")
        if type_obj.get("FLANGE_STYLE"):
            key = f"{key}+{type_obj['FLANGE_STYLE']}"
        counter[key] += 1

    print(f"[完成] 输入条数: {len(lines)}")
    print(f"[完成] 输出文件: {args.output_file}")
    print("[完成] 主体分布:")
    for key, count in counter.items():
        print(f"  - {key}: {count}")


if __name__ == "__main__":
    main()
