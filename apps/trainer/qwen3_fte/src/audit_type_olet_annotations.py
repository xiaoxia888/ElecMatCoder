#!/usr/bin/env python3
"""Audit OLET-family BODY annotations in the fitting type dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


QWEN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = QWEN_ROOT / "output" / "按8类拆分数据集" / "种类" / "管件.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET.parent

OLET_BODIES = {
    "SWEEPOLET",
    "支管台",
    "对焊支管台",
    "承插焊支管台",
    "螺纹支管台",
    "斜支管台",
}

SCOPE_RE = re.compile(r"(?i)OLET|OUTLET|支管台|支管座|管接台")

EXPLICIT_BODY_PATTERNS = (
    (
        "SWEEPOLET",
        re.compile(r"(?i)SWEEP\s*OLET"),
        "明确 SWEEPOLET 产品词",
    ),
    (
        "斜支管台",
        re.compile(r"(?i)LATROLET|斜支管(?:台|座)|嵌入式支管台"),
        "明确斜支管台产品词",
    ),
    (
        "承插焊支管台",
        re.compile(
            r"(?i)SOCKOLET|SOCKET(?:\s+WELD(?:ED|ING)?)?\s*(?:OLET|OUTLET)|"
            r"承插(?:焊)?(?:管接台|支管台|支管座)|"
            r"(?<![A-Z])SOL(?:-\d+)?(?![A-Z])"
        ),
        "明确承插焊支管台产品词",
    ),
    (
        "螺纹支管台",
        re.compile(
            r"(?i)THR(?:EAD|ED)?OLET|THREADOLET|THREADED?\s*(?:OLET|OUTLET)|"
            r"螺纹(?:管接台|支管台|支管座)|(?<![A-Z])TOL(?:-\d+)?(?![A-Z])"
        ),
        "明确螺纹支管台产品词",
    ),
    (
        "对焊支管台",
        re.compile(
            r"(?i)WELDOLET|(?<!SOCKET )WELD(?:ING|ED)?\s*(?:OLET|OUTLET)|"
            r"对焊(?:管接台|支管台|支管座)|"
            r"(?<![A-Z])WOL(?:-\d+)?(?![A-Z])"
        ),
        "明确对焊支管台产品词",
    ),
)

CONNECTION_PATTERNS = {
    "BW": re.compile(
        r"(?i)(?<![A-Z])B\s*\.?\s*W\s*\.?(?![A-Z])|"
        r"BW(?=OLET|OUTLET)|(?<=OLET)BW|(?<=OUTLET)BW|"
        r"BUTT\s*WELD(?:ED|ING)?|(?<!承插)对焊"
    ),
    "SW": re.compile(
        r"(?i)(?<![A-Z])S\s*\.?\s*W\s*\.?(?![A-Z])|"
        r"SW(?=OLET|OUTLET)|(?<=OLET)SW|(?<=OUTLET)SW|"
        r"SOCKET\s*WELD(?:ED|ING)?|承插焊"
    ),
    "THREAD": re.compile(
        r"(?i)(?<![A-Z0-9])(?:FNPT|MNPT|NPTF|NPT|THD|SCRD)(?![A-Z0-9])|"
        r"(?:FNPT|MNPT|NPTF|NPT|THD|SCRD)(?=OLET|OUTLET)|"
        r"(?<=OLET)(?:FNPT|MNPT|NPTF|NPT|THD|SCRD)|"
        r"(?<=OUTLET)(?:FNPT|MNPT|NPTF|NPT|THD|SCRD)|"
        r"THREAD(?:ED)?|螺纹"
    ),
}

BE_RE = re.compile(
    r"(?i)(?<![A-Z])B\s*\.?\s*E\s*\.?(?![A-Z])|"
    r"BE(?=OLET|OUTLET)|(?<=OLET)BE|(?<=OUTLET)BE|BEVEL(?:ED)?\s*END"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计管件种类数据集中的 OLET 家族 BODY 标注")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def detect_explicit_bodies(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for body, pattern, reason in EXPLICIT_BODY_PATTERNS:
        if pattern.search(text):
            result.append((body, reason))
    return result


def detect_connections(text: str) -> list[str]:
    result = [name for name, pattern in CONNECTION_PATTERNS.items() if pattern.search(text)]
    if BE_RE.search(text) and "BW" not in result:
        result.append("BW")
    return result


def expected_body_from_connections(connections: list[str]) -> str:
    if connections == ["BW"]:
        return "对焊支管台"
    if connections == ["SW"]:
        return "承插焊支管台"
    if connections == ["THREAD"]:
        return "螺纹支管台"
    if not connections:
        return "支管台"
    return ""


def audit_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    description = str(row.get("input") or "")
    output = row.get("output") if isinstance(row.get("output"), dict) else {}
    type_value = output.get("TYPE") if isinstance(output.get("TYPE"), dict) else {}
    current_body = str(type_value.get("BODY") or "").strip()
    current_conn = type_value.get("CONN") if isinstance(type_value.get("CONN"), list) else []

    explicit_bodies = detect_explicit_bodies(description)
    explicit_body_values = list(dict.fromkeys(body for body, _ in explicit_bodies))
    connections = detect_connections(description)

    expected_body = ""
    status = "correct"
    reason = ""
    if len(explicit_body_values) > 1:
        status = "manual_review"
        reason = f"同时命中多个明确产品词: {explicit_body_values}"
    elif explicit_body_values:
        expected_body = explicit_body_values[0]
        if expected_body == "对焊支管台" and any(item in connections for item in ("SW", "THREAD")):
            status = "manual_review"
            reason = f"明确对焊产品词与连接证据冲突: {connections}"
        elif expected_body == "承插焊支管台" and any(item in connections for item in ("BW", "THREAD")):
            status = "manual_review"
            reason = f"明确承插焊产品词与连接证据冲突: {connections}"
        elif expected_body == "螺纹支管台" and any(item in connections for item in ("BW", "SW")):
            status = "manual_review"
            reason = f"明确螺纹产品词与连接证据冲突: {connections}"
        else:
            reason = explicit_bodies[0][1]
    else:
        expected_body = expected_body_from_connections(connections)
        if not expected_body:
            status = "manual_review"
            reason = f"裸 OLET 同时存在多种连接证据: {connections}"
        elif connections:
            reason = f"裸 OLET 根据连接证据 {connections[0]} 判断"
        else:
            reason = "裸 OLET 未明确支管端方式"

    if status != "manual_review" and current_body != expected_body:
        status = "body_mismatch"

    return {
        "source_index": source_index,
        "input": description,
        "current_body": current_body,
        "expected_body": expected_body,
        "current_conn": current_conn,
        "connection_evidence": connections,
        "explicit_body_evidence": [
            {"body": body, "reason": evidence_reason}
            for body, evidence_reason in explicit_bodies
        ],
        "status": status,
        "reason": reason,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown(
    dataset_path: Path,
    all_items: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# 管件 OLET 标注审计报告",
        "",
        f"- 数据集：`{dataset_path}`",
        f"- OLET 家族样本：{summary['olet_rows']} 条",
        f"- 标注一致：{summary['status_counts'].get('correct', 0)} 条",
        f"- BODY 疑似错标：{summary['status_counts'].get('body_mismatch', 0)} 条",
        f"- 证据冲突需人工复核：{summary['status_counts'].get('manual_review', 0)} 条",
        "",
        "## 建议修改分布",
        "",
        "| 当前 BODY | 建议 BODY | 数量 |",
        "|---|---|---:|",
    ]
    for transition, count in summary["mismatch_transitions"].items():
        current_body, expected_body = transition.split(" -> ", 1)
        lines.append(f"| {current_body} | {expected_body} | {count} |")

    lines.extend(
        [
            "",
            "## BODY 疑似错标明细",
            "",
        ]
    )
    mismatches = [item for item in all_items if item["status"] == "body_mismatch"]
    if not mismatches:
        lines.append("无。")
    for index, item in enumerate(mismatches, 1):
        lines.extend(
            [
                f"### {index}. {item['current_body']} -> {item['expected_body']}",
                "",
                f"- 描述：`{item['input']}`",
                f"- 依据：{item['reason']}",
                f"- 连接证据：{item['connection_evidence'] or '无'}",
                "",
            ]
        )

    lines.extend(
        [
            "## 人工复核项",
            "",
        ]
    )
    manual_items = [item for item in all_items if item["status"] == "manual_review"]
    if not manual_items:
        lines.append("无。")
    for index, item in enumerate(manual_items, 1):
        lines.extend(
            [
                f"### {index}. 当前 {item['current_body']}",
                "",
                f"- 描述：`{item['input']}`",
                f"- 原因：{item['reason']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("管件数据集顶层必须是数组")

    all_items = [
        audit_row(row, index)
        for index, row in enumerate(rows)
        if isinstance(row, dict) and SCOPE_RE.search(str(row.get("input") or ""))
    ]
    status_counts = Counter(item["status"] for item in all_items)
    current_body_counts = Counter(item["current_body"] for item in all_items)
    expected_body_counts = Counter(
        item["expected_body"]
        for item in all_items
        if item["expected_body"]
    )
    mismatch_transitions = Counter(
        f"{item['current_body']} -> {item['expected_body']}"
        for item in all_items
        if item["status"] == "body_mismatch"
    )
    unexpected_current_bodies = sorted(
        body for body in current_body_counts if body not in OLET_BODIES
    )

    summary = {
        "dataset": str(dataset_path),
        "dataset_rows": len(rows),
        "olet_rows": len(all_items),
        "status_counts": dict(status_counts),
        "current_body_counts": dict(current_body_counts),
        "expected_body_counts": dict(expected_body_counts),
        "mismatch_transitions": dict(mismatch_transitions),
        "unexpected_current_bodies": unexpected_current_bodies,
    }

    prefix = output_dir / "管件_OLET标注审计"
    write_json(prefix.with_name(prefix.name + "_汇总.json"), summary)
    write_json(prefix.with_name(prefix.name + "_全部明细.json"), all_items)
    write_json(
        prefix.with_name(prefix.name + "_疑似问题.json"),
        [item for item in all_items if item["status"] != "correct"],
    )
    prefix.with_name(prefix.name + "_报告.md").write_text(
        render_markdown(dataset_path, all_items, summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
