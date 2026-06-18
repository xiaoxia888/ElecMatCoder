#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove LINING from pipe MATERIAL and merge it into VALUE as 主材/衬里."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "apps/trainer/qwen3_fte/output/pipe_project_sampling_full"
TARGET_FILES = [
    BASE / "直管训练草稿_语义补强版.json",
    BASE / "直管训练草稿.json",
    BASE / "直管语义真实补样草稿.json",
    BASE / "直管语义增强草稿.json",
    BASE / "直管语义补样增强合并草稿.json",
]


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    return value if isinstance(value, list) else [value]


def uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = text(value)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def infer_base_from_input(inp: str, current_value: str) -> str:
    patterns = [
        r"\b(20#|20|304|304L|316L|S30403|S30408|S31603|CS|Q235B)\s*/\s*(?:PTFE|RPTFE|PE|EAA)\b",
        r"\b(?:PTFE|RPTFE)\s*/\s*(20#|20|304|304L|316L|S30403|S30408|S31603|CS|Q235B)\b",
        r"\b(20#|20|304|304L|316L|S30403|S30408|S31603|CS|Q235B)\s*\+\s*(?:PTFE|RPTFE|PE|EAA)\b",
        r"\b(20#|20|304|304L|316L|S30403|S30408|S31603|CS|Q235B)\s+GLASS\s+LINED\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, inp, re.I)
        if m:
            return m.group(1)

    if re.match(r"^T(?:20553|20615|2437|12771)\b", current_value):
        for pattern in [
            r"\b(S30403|S30408|S31603|CS|Q235B|304|316L|20#|20)\b",
        ]:
            m = re.search(pattern, inp, re.I)
            if m:
                return m.group(1)
    return current_value


def merge_lining_into_value(inp: str, value: str, lining: list[str]) -> str:
    value = text(value)
    lining = uniq(lining)
    if not lining:
        return value

    base = infer_base_from_input(inp, value)
    existing_parts = [part for part in re.split(r"[+/]", base) if text(part)]
    normalized_lining = uniq(lining)

    if existing_parts:
        base_part = existing_parts[0]
        suffix = existing_parts[1:]
    else:
        base_part = base
        suffix = []

    suffix_upper = {part.upper() for part in suffix}
    for token in normalized_lining:
        if token.upper() not in suffix_upper and token.upper() != base_part.upper():
            suffix.append(token)
            suffix_upper.add(token.upper())

    if not base_part:
        return "/".join(suffix)
    return "/".join([base_part] + suffix)


def migrate_file(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    item_count = 0
    changed_count = 0
    for item in data:
        mats = item.get("output", {}).get("MATERIAL", []) or []
        for mat in mats:
            if not isinstance(mat, dict):
                continue
            item_count += 1
            lining = [text(x) for x in as_list(mat.get("LINING")) if text(x)]
            if lining:
                new_value = merge_lining_into_value(text(item.get("input")), text(mat.get("VALUE")), lining)
                if new_value != text(mat.get("VALUE")):
                    mat["VALUE"] = new_value
                    changed_count += 1
            if "LINING" in mat:
                del mat["LINING"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return item_count, changed_count


def main() -> None:
    for path in TARGET_FILES:
        if not path.exists():
            continue
        total, changed = migrate_file(path)
        print(f"{path.name}\ttotal_material_items={total}\tmerged_lining={changed}")


if __name__ == "__main__":
    main()
