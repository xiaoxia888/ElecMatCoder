#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "种类0629.json"
OUTPUT_FILE = BASE_DIR / "种类0629分布统计.xlsx"


def text(value: object) -> str:
    return str(value or "").strip()


def ensure_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [text(v) for v in value if text(v)]
    if text(value):
        return [text(value)]
    return []


def canonical_complete_type(type_obj: dict) -> str:
    body = text(type_obj.get("BODY"))
    geometry = type_obj.get("GEOMETRY") or {}
    angle = text(geometry.get("ANGLE"))
    radius = text(geometry.get("RADIUS"))
    manu = "|".join(ensure_list(type_obj.get("MANU")))
    conn = "|".join(ensure_list(type_obj.get("CONN")))
    seal = "|".join(ensure_list(type_obj.get("SEAL")))

    body_with_geometry = body
    if angle:
        body_with_geometry = f"{angle}度{body}" if body else f"{angle}度"

    parts = [body_with_geometry, radius, manu, conn, seal]
    return ";".join([part for part in parts if part])


def build_counter_df(counter: Counter, column_name: str) -> pd.DataFrame:
    total = sum(counter.values())
    rows = []
    for value, count in counter.most_common():
        rows.append(
            {
                column_name: value,
                "数量": count,
                "占比(%)": round(count / total * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def analyze_type_distribution(input_path: Path = INPUT_FILE, output_path: Path = OUTPUT_FILE) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{input_path} 不是 JSON 数组")

    complete_counter: Counter[str] = Counter()
    body_counter: Counter[str] = Counter()
    manu_counter: Counter[str] = Counter()
    conn_counter: Counter[str] = Counter()
    seal_counter: Counter[str] = Counter()

    for item in data:
        output = item.get("output") or {}
        type_obj = output.get("TYPE") or {}
        if not isinstance(type_obj, dict):
            continue

        complete_counter[canonical_complete_type(type_obj)] += 1

        body = text(type_obj.get("BODY")) or "（空）"
        body_counter[body] += 1

        manus = ensure_list(type_obj.get("MANU")) or ["（空）"]
        conns = ensure_list(type_obj.get("CONN")) or ["（空）"]
        seals = ensure_list(type_obj.get("SEAL")) or ["（空）"]

        for manu in manus:
            manu_counter[manu] += 1
        for conn in conns:
            conn_counter[conn] += 1
        for seal in seals:
            seal_counter[seal] += 1

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        build_counter_df(complete_counter, "完整种类").to_excel(writer, sheet_name="完整种类", index=False)
        build_counter_df(body_counter, "BODY").to_excel(writer, sheet_name="BODY分布", index=False)
        build_counter_df(manu_counter, "MANU").to_excel(writer, sheet_name="MANU分布", index=False)
        build_counter_df(conn_counter, "CONN").to_excel(writer, sheet_name="CONN分布", index=False)
        build_counter_df(seal_counter, "SEAL").to_excel(writer, sheet_name="SEAL分布", index=False)

    return output_path


def main() -> None:
    output_path = analyze_type_distribution()
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
