#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "type_raw/种类_train.json"
OUTPUT_FILE = BASE_DIR / "种类分布统计.xlsx"
ORIGINAL_SOURCE = "原始/未标识"


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


def source_name(item: dict[str, Any]) -> str:
    return text(item.get("来源")) or ORIGINAL_SOURCE


def description_skeleton(value: object) -> str:
    """将尺寸、壁厚、磅级和数字归一化，用于发现模板型重复。"""
    value = text(value).upper()
    replacements = (
        (r"\b(?:DN|NPS)\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*(?:DN|NPS|D)?\s*\d+(?:\.\d+)?){0,2}\b", "<SIZE>"),
        (r"(?:Φ|φ|Ø)\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?){0,2}", "<SIZE>"),
        (r"\b\d+(?:\.\d+)?\s*(?:INCH|IN|\")", "<SIZE>"),
        (r"\b(?:SCH(?:EDULE)?\.?|S-)\s*\d+[A-Z]*(?:\s*[X×*/]\s*(?:SCH(?:EDULE)?\.?|S-)?\s*\d+[A-Z]*)?\b", "<THICKNESS>"),
        (r"\b(?:STD|XS|XXS)(?:\s*[X×*/]\s*(?:STD|XS|XXS))?\b", "<THICKNESS>"),
        (r"\b(?:CL(?:ASS)?\.?|PN)\s*\d+(?:\.\d+)?\b|\b\d+\s*(?:LB|LBS|#)\b", "<PRESSURE>"),
        (r"\b\d+(?:\.\d+)?\s*MM\b", "<MM>"),
        (r"\d+(?:\.\d+)?", "<N>"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r"(?:<N>\s*){2,}", "<N>", value)
    value = re.sub(r"\s*([,;:/()\[\]])\s*", r"\1", value)
    return re.sub(r"\s+", " ", value).strip(" ,;")


def body_appears_literally(body: str, description: str) -> bool:
    """仅统计归一化 BODY 文本是否在原文直接出现，不把该结果当作语义正确性。"""
    return bool(body and body.casefold() in description.casefold())


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


def build_overview_df(
    *,
    rows: int,
    descriptions: Counter[str],
    source_counter: Counter[str],
    body_counter: Counter[str],
) -> pd.DataFrame:
    augmented_rows = rows - source_counter.get(ORIGINAL_SOURCE, 0)
    duplicate_groups = sum(count > 1 for count in descriptions.values())
    duplicate_rows = sum(count - 1 for count in descriptions.values() if count > 1)
    metrics = (
        ("总样本数", rows),
        ("唯一描述数", len(descriptions)),
        ("重复描述组数", duplicate_groups),
        ("重复行数", duplicate_rows),
        ("原始/未标识样本数", source_counter.get(ORIGINAL_SOURCE, 0)),
        ("增强样本数", augmented_rows),
        ("增强样本占比(%)", round(augmented_rows / rows * 100, 2) if rows else 0.0),
        ("BODY种类数", len(body_counter)),
        ("BODY样本<10的种类数", sum(count < 10 for count in body_counter.values())),
        ("BODY样本<30的种类数", sum(count < 30 for count in body_counter.values())),
    )
    return pd.DataFrame(metrics, columns=["指标", "数值"])


def build_source_df(source_counter: Counter[str], total: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "来源": source,
                "数量": count,
                "占比(%)": round(count / total * 100, 2) if total else 0.0,
            }
            for source, count in source_counter.most_common()
        ]
    )


def build_body_quality_df(
    *,
    body_counter: Counter[str],
    body_sources: dict[str, Counter[str]],
    body_descriptions: dict[str, set[str]],
    body_skeletons: dict[str, Counter[str]],
    body_literal_counter: Counter[str],
) -> pd.DataFrame:
    total_rows = sum(body_counter.values())
    rows: list[dict[str, Any]] = []
    for body, count in body_counter.most_common():
        original_count = body_sources[body].get(ORIGINAL_SOURCE, 0)
        augmented_count = count - original_count
        skeletons = body_skeletons[body]
        largest_skeleton = max(skeletons.values(), default=0)
        literal_count = body_literal_counter[body]
        risks: list[str] = []
        if augmented_count == count:
            risks.append("无原始样本")
        elif count >= 20 and augmented_count / count >= 0.5:
            risks.append("增强占比过高")
        if count < 10:
            risks.append("样本过少")
        if count >= 20 and largest_skeleton / count >= 0.25:
            risks.append("骨架过度集中")
        rows.append(
            {
                "BODY": body,
                "总数": count,
                "总占比(%)": round(count / total_rows * 100, 2) if total_rows else 0.0,
                "原始/未标识": original_count,
                "增强": augmented_count,
                "增强占比(%)": round(augmented_count / count * 100, 2) if count else 0.0,
                "唯一描述数": len(body_descriptions[body]),
                "描述骨架数": len(skeletons),
                "最大骨架样本数": largest_skeleton,
                "最大骨架占比(%)": round(largest_skeleton / count * 100, 2) if count else 0.0,
                "BODY在原文直接出现数": literal_count,
                "BODY在原文直接出现率(%)": round(literal_count / count * 100, 2) if count else 0.0,
                "风险提示": "|".join(risks),
            }
        )
    return pd.DataFrame(rows)


def build_body_source_matrix(body_sources: dict[str, Counter[str]]) -> pd.DataFrame:
    sources = sorted({source for counter in body_sources.values() for source in counter})
    rows = []
    for body, counter in sorted(
        body_sources.items(), key=lambda item: (-sum(item[1].values()), item[0])
    ):
        row: dict[str, Any] = {"BODY": body, "总数": sum(counter.values())}
        row.update({source: counter.get(source, 0) for source in sources})
        rows.append(row)
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
    source_counter: Counter[str] = Counter()
    descriptions: Counter[str] = Counter()
    body_sources: dict[str, Counter[str]] = defaultdict(Counter)
    body_descriptions: dict[str, set[str]] = defaultdict(set)
    body_skeletons: dict[str, Counter[str]] = defaultdict(Counter)
    body_literal_counter: Counter[str] = Counter()
    nonliteral_examples: dict[str, list[str]] = defaultdict(list)

    for item in data:
        description = text(item.get("input"))
        source = source_name(item)
        source_counter[source] += 1
        descriptions[description] += 1

        output = item.get("output") or {}
        type_obj = output.get("TYPE") or {}
        if not isinstance(type_obj, dict):
            continue

        complete_counter[canonical_complete_type(type_obj)] += 1

        body = text(type_obj.get("BODY")) or "（空）"
        body_counter[body] += 1
        body_sources[body][source] += 1
        body_descriptions[body].add(description)
        body_skeletons[body][description_skeleton(description)] += 1
        if body_appears_literally(body, description):
            body_literal_counter[body] += 1
        elif len(nonliteral_examples[body]) < 5:
            nonliteral_examples[body].append(description)

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
        build_overview_df(
            rows=len(data),
            descriptions=descriptions,
            source_counter=source_counter,
            body_counter=body_counter,
        ).to_excel(writer, sheet_name="数据概览", index=False)
        build_source_df(source_counter, len(data)).to_excel(writer, sheet_name="来源分布", index=False)
        build_body_quality_df(
            body_counter=body_counter,
            body_sources=body_sources,
            body_descriptions=body_descriptions,
            body_skeletons=body_skeletons,
            body_literal_counter=body_literal_counter,
        ).to_excel(writer, sheet_name="BODY质量分析", index=False)
        build_body_source_matrix(body_sources).to_excel(writer, sheet_name="BODY来源贡献", index=False)
        pd.DataFrame(
            [
                {"BODY": body, "原文未直接出现BODY的描述示例": example}
                for body, examples in nonliteral_examples.items()
                for example in examples
            ]
        ).to_excel(writer, sheet_name="BODY归一化示例", index=False)
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
