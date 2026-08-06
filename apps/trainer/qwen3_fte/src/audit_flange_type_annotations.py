#!/usr/bin/env python3
"""Audit flange TYPE annotations without modifying source datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_TYPE_KEYS = {"BODY", "CONN", "SEAL"}
ALLOWED_CONN = {"SW", "THD", "SCRD", "NPT", "FNPT", "MNPT", "FTE", "MTE", "GROOVED"}
ALLOWED_SEAL = {
    "RF",
    "FF",
    "RJ",
    "RTJ",
    "FRJ",
    "MRJ",
    "MFM",
    "LM",
    "LF",
    "MF",
    "M",
    "FM",
    "T",
    "G",
    "TG",
    "SERRATED",
}

SPECTACLE_PATTERN = re.compile(
    r"八字盲板|8字盲板|SPECTACLE\s*(?:BLIND|BLANK)|FIGURE\s*-?\s*8\s*(?:BLIND|BLANK)?",
    re.IGNORECASE,
)
COMPONENT_PATTERN = re.compile(
    r"成套供货|"
    r"(?:PADDLE\s*)?BLANK\s*(?:AND|&|\+)\s*SPACER|"
    r"SPACER\s*(?:AND|&|\+)\s*BLANKS?|"
    r"插板\s*(?:[/+、和及&]|与)\s*垫环|"
    r"盲板\s*(?:[,，/+、和及&]|与)\s*垫环",
    re.IGNORECASE,
)
AUXILIARY_PORT_PATTERN = re.compile(
    r"DRILLED\s*&\s*TAPPED|TAP\s*HOLE|TAPHOLE|PRESSURE\s*TAPS?|取压孔|排放孔",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_output(body: str, conn: list[str], seal: list[str]) -> dict[str, Any]:
    return {
        "CATEGORY": "法兰",
        "TYPE": {
            "BODY": body,
            "CONN": conn,
            "SEAL": seal,
        },
    }


def type_fields(row: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    output = row.get("output", {})
    type_data = output.get("TYPE", {})
    body = type_data.get("BODY", "")
    conn = type_data.get("CONN", [])
    seal = type_data.get("SEAL", [])
    return body, conn, seal


def add_suggestion(
    suggestions: dict[tuple[str, int], dict[str, Any]],
    *,
    split: str,
    index: int,
    row: dict[str, Any],
    issue: str,
    reason: str,
    mutate: Any,
) -> None:
    key = (split, index)
    if key not in suggestions:
        suggestions[key] = {
            "来源数据集": split,
            "source_index": index,
            "原始描述": row.get("input", ""),
            "修正前标签": copy.deepcopy(row.get("output", {})),
            "建议修正标签": copy.deepcopy(row.get("output", {})),
            "问题类别": [],
            "中文原因": [],
            "置信等级": "高",
        }
    item = suggestions[key]
    mutate(item["建议修正标签"])
    if issue not in item["问题类别"]:
        item["问题类别"].append(issue)
    if reason not in item["中文原因"]:
        item["中文原因"].append(reason)


def audit_split(
    split: str,
    rows: list[dict[str, Any]],
    suggestions: dict[tuple[str, int], dict[str, Any]],
    manual_review: dict[str, list[dict[str, Any]]],
    structural_errors: list[dict[str, Any]],
) -> None:
    for index, row in enumerate(rows):
        text = str(row.get("input", ""))
        output = row.get("output", {})
        type_data = output.get("TYPE", {}) if isinstance(output, dict) else {}
        body, conn, seal = type_fields(row)

        structure_problems: list[str] = []
        if output.get("CATEGORY") != "法兰":
            structure_problems.append("CATEGORY不是法兰")
        if set(type_data) != ALLOWED_TYPE_KEYS:
            structure_problems.append("TYPE字段不完整或存在额外字段")
        if not isinstance(body, str) or not body:
            structure_problems.append("BODY为空或类型错误")
        if not isinstance(conn, list) or not all(isinstance(value, str) for value in conn):
            structure_problems.append("CONN不是字符串数组")
        if not isinstance(seal, list) or not all(isinstance(value, str) for value in seal):
            structure_problems.append("SEAL不是字符串数组")
        if isinstance(conn, list) and set(conn) - ALLOWED_CONN:
            structure_problems.append(f"CONN存在非法标签: {sorted(set(conn) - ALLOWED_CONN)}")
        # FWRF is reported as a deterministic label correction below.
        invalid_seal = set(seal) - ALLOWED_SEAL - {"FWRF"} if isinstance(seal, list) else set()
        if invalid_seal:
            structure_problems.append(f"SEAL存在未知标签: {sorted(invalid_seal)}")
        if structure_problems:
            structural_errors.append(
                {
                    "来源数据集": split,
                    "source_index": index,
                    "原始描述": text,
                    "当前标签": output,
                    "问题": structure_problems,
                }
            )
            continue

        has_spectacle = bool(SPECTACLE_PATTERN.search(text))
        has_component = bool(COMPONENT_PATTERN.search(text))
        if has_spectacle:
            if body != "8字盲板":
                add_suggestion(
                    suggestions,
                    split=split,
                    index=index,
                    row=row,
                    issue="8字盲板主体错误",
                    reason="原文明示SPECTACLE BLIND/BLANK；即使同时出现插板与垫环语义，也按一体式8字盲板标注。",
                    mutate=lambda value: value["TYPE"].update({"BODY": "8字盲板"}),
                )
        elif body == "盲板垫环组件" or (body == "插板" and has_component):
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="管道盲板成套主体错误",
                reason="原文明示插板与垫环成套，或仍使用历史标签“盲板垫环组件”，统一标为“插板垫环组件”。",
                mutate=lambda value: value["TYPE"].update({"BODY": "插板垫环组件"}),
            )

        if body == "板式平焊法兰" and re.search(r"(?:突面)?对焊钢制管法兰.*(?:^|[^A-Z])WN", text, re.IGNORECASE):
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="对焊法兰主体误标",
                reason="原文明示对焊钢制管法兰且包含WN，主体应为带颈对焊法兰，不是板式平焊法兰。",
                mutate=lambda value: value["TYPE"].update({"BODY": "带颈对焊法兰"}),
            )

        if body == "异径法兰" and re.search(r"REDUCING\s*FLANGE\s*SO", text, re.IGNORECASE):
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="异径平焊法兰主体过度简化",
                reason="原文明示REDUCING FLANGE SO，SO是带颈平焊结构，应标为异径带颈平焊法兰。",
                mutate=lambda value: value["TYPE"].update({"BODY": "异径带颈平焊法兰"}),
            )

        if body == "承插焊法兰" and not conn:
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="承插焊连接形式漏标",
                reason="BODY已明确为承插焊法兰，CONN必须保留SW，避免一阶段丢失连接语义。",
                mutate=lambda value: value["TYPE"].update({"CONN": ["SW"]}),
            )

        if body == "螺纹法兰" and not conn:
            has_thd = bool(re.search(r"THREADED|THD", text, re.IGNORECASE))
            has_chinese_threaded = "螺纹法兰" in text
            has_undefined_sc = bool(re.search(r"(?<![A-Z])SC(?![A-Z])", text, re.IGNORECASE))
            if has_thd:
                add_suggestion(
                    suggestions,
                    split=split,
                    index=index,
                    row=row,
                    issue="螺纹连接形式漏标",
                    reason="原文明示THD或THREADED，且没有更具体螺纹制式，CONN应保留THD。",
                    mutate=lambda value: value["TYPE"].update({"CONN": ["THD"]}),
                )
            elif has_undefined_sc:
                # SC含义未定义，既不据此补连接标签，也不作为标注问题上报。
                pass
            elif has_chinese_threaded:
                add_suggestion(
                    suggestions,
                    split=split,
                    index=index,
                    row=row,
                    issue="螺纹连接形式漏标",
                    reason="原文明示螺纹法兰但没有更具体螺纹制式，CONN按标注规范保留THD。",
                    mutate=lambda value: value["TYPE"].update({"CONN": ["THD"]}),
                )
            else:
                add_suggestion(
                    suggestions,
                    split=split,
                    index=index,
                    row=row,
                    issue="普通法兰误标为螺纹法兰",
                    reason="原文没有螺纹、THD、SCREWED、SCRD、NPT等证据，只能标为普通法兰。",
                    mutate=lambda value: value["TYPE"].update({"BODY": "法兰"}),
                )

        if body == "盲板法兰" and conn == ["NPT"] and AUXILIARY_PORT_PATTERN.search(text):
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="辅助孔误作主体连接",
                reason="NPT明确属于Drilled & Tapped或Tap Hole辅助孔，不是盲板法兰主体连接形式，CONN应为空。",
                mutate=lambda value: value["TYPE"].update({"CONN": []}),
            )

        if "FWRF" in seal:
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="非标准密封面标签",
                reason="FWRF是端部/项目写法，不在当前固定SEAL枚举中；一阶段不得自行输出该标签。",
                mutate=lambda value: value["TYPE"].update(
                    {"SEAL": [item for item in value["TYPE"]["SEAL"] if item != "FWRF"]}
                ),
            )

        if re.search(r"(?<![A-Z0-9])MFM(?![A-Z0-9])", text, re.IGNORECASE) and "MFM" not in seal:
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="明确MFM密封面漏标",
                reason="原文明示独立MFM密封面，SEAL应保留MFM。",
                mutate=lambda value: value["TYPE"].update({"SEAL": value["TYPE"]["SEAL"] + ["MFM"]}),
            )

        has_rf = bool(re.search(r"(?<![A-Z0-9])RF(?![A-Z0-9])", text, re.IGNORECASE))
        has_rj = bool(re.search(r"(?<![A-Z0-9])RJ(?![A-Z0-9])", text, re.IGNORECASE))
        if has_rf and has_rj and seal != ["RJ"]:
            add_suggestion(
                suggestions,
                split=split,
                index=index,
                row=row,
                issue="RF与RJ冲突",
                reason="原文同时出现独立RF与RJ，按当前标注规则以RJ为准，SEAL只保留RJ。",
                mutate=lambda value: value["TYPE"].update({"SEAL": ["RJ"]}),
            )


def duplicate_report(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in (("train", train), ("val", val)):
        for index, row in enumerate(rows):
            positions[str(row.get("input", ""))].append(
                {"来源数据集": split, "source_index": index, "标签": row.get("output", {})}
            )
    duplicates: list[dict[str, Any]] = []
    for text, items in positions.items():
        if len(items) < 2:
            continue
        signatures = {json.dumps(item["标签"], ensure_ascii=False, sort_keys=True) for item in items}
        duplicates.append(
            {
                "原始描述": text,
                "出现位置": items,
                "标签是否一致": len(signatures) == 1,
                "建议保留位置": items[0] if len(signatures) == 1 else None,
                "建议删除位置": items[1:] if len(signatures) == 1 else [],
                "中文建议": (
                    "标签完全一致，建议保留第一条并删除其余重复项。"
                    if len(signatures) == 1
                    else "同一描述对应不同标签，需先确定正确标签，再保留一条并删除其余项。"
                ),
            }
        )
    return duplicates


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_hash_before = sha256(args.train)
    val_hash_before = sha256(args.val)
    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val)

    suggestions: dict[tuple[str, int], dict[str, Any]] = {}
    manual_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structural_errors: list[dict[str, Any]] = []
    audit_split("train", train_rows, suggestions, manual_review, structural_errors)
    audit_split("val", val_rows, suggestions, manual_review, structural_errors)

    suggestion_rows = [suggestions[key] for key in sorted(suggestions)]
    grouped_suggestions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in suggestion_rows:
        for issue in item["问题类别"]:
            grouped_suggestions[issue].append(item)

    duplicates = duplicate_report(train_rows, val_rows)
    suggestion_path = args.output_dir / "法兰_train_val_标注修改建议_待审核.json"
    manual_path = args.output_dir / "法兰_train_val_需人工判断.json"
    summary_path = args.output_dir / "法兰_train_val_审查统计.json"

    write_json(
        suggestion_path,
        {
            "说明": "本文件仅包含高置信修改建议，尚未写回训练集或验证集。",
            "源文件": {"train": str(args.train), "val": str(args.val)},
            "按问题类别分组": dict(grouped_suggestions),
            "去重后的建议记录": suggestion_rows,
        },
    )
    write_json(
        manual_path,
        {
            "说明": "以下项目存在语义冲突或端部归属不明确，不应自动修改。",
            "需人工判断": dict(manual_review),
            "重复样本建议": duplicates,
            "结构或未知枚举问题": structural_errors,
        },
    )

    issue_counts = Counter(issue for item in suggestion_rows for issue in item["问题类别"])
    manual_counts = {name: len(items) for name, items in manual_review.items()}
    train_hash_after = sha256(args.train)
    val_hash_after = sha256(args.val)
    summary = {
        "source": {
            "train": str(args.train),
            "val": str(args.val),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "total_rows": len(train_rows) + len(val_rows),
        },
        "statistics": {
            "高置信建议记录数": len(suggestion_rows),
            "高置信问题计数": dict(issue_counts),
            "需人工判断记录数": sum(manual_counts.values()),
            "需人工判断问题计数": manual_counts,
            "重复描述组数": len(duplicates),
            "重复描述额外行数": sum(len(item["出现位置"]) - 1 for item in duplicates),
            "结构或未知枚举问题数": len(structural_errors),
        },
        "output": {
            "标注修改建议": str(suggestion_path),
            "需人工判断": str(manual_path),
            "审查统计": str(summary_path),
        },
        "source_integrity": {
            "train_sha256_before": train_hash_before,
            "train_sha256_after": train_hash_after,
            "val_sha256_before": val_hash_before,
            "val_sha256_after": val_hash_after,
            "源文件未修改": train_hash_before == train_hash_after and val_hash_before == val_hash_after,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
