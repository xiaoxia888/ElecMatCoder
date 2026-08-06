#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = BASE_DIR / "管件.json"
OUTPUT_JSON = BASE_DIR / "管件_去重.json"
REPORT_JSON = BASE_DIR / "管件_去重报告.json"


def text(value: object) -> str:
    return str(value or "").strip()


def body_of(item: dict) -> str:
    return text((((item or {}).get("output") or {}).get("TYPE") or {}).get("BODY"))


def dedupe_dataset(data: list[dict]) -> tuple[list[dict], list[dict]]:
    seen: dict[tuple[str, str], int] = {}
    deduped: list[dict] = []
    duplicates: list[dict] = []

    for idx, item in enumerate(data):
        input_text = text((item or {}).get("input"))
        body = body_of(item)
        key = (input_text, body)
        if key in seen:
            duplicates.append(
                {
                    "first_index": seen[key],
                    "duplicate_index": idx,
                    "body": body,
                    "input": input_text,
                    "_source": text((item or {}).get("_source")) or "原始",
                }
            )
            continue
        seen[key] = idx
        deduped.append(item)
    return deduped, duplicates


def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{INPUT_JSON} 不是 JSON 数组")

    deduped, duplicates = dedupe_dataset(data)
    OUTPUT_JSON.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"INPUT={INPUT_JSON} rows={len(data)}")
    print(f"OUTPUT={OUTPUT_JSON} rows={len(deduped)}")
    print(f"REPORT={REPORT_JSON} duplicate_rows={len(duplicates)}")
    for row in duplicates:
        print(
            "DUPLICATE"
            f" first={row['first_index']}"
            f" dup={row['duplicate_index']}"
            f" body={row['body']}"
            f" input={row['input']}"
        )


if __name__ == "__main__":
    main()
