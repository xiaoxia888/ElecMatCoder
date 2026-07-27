#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集"
DEFAULT_INPUT = BASE / "种类_train.json"
DEFAULT_OUTPUT = BASE / "种类_train_主词归一.json"


ANCHOR_RULES = [
    {
        "name": "WELDOLET",
        "body": "对焊支管台",
        "patterns": [r"\bWELDOLET\b", r"\bWELD\s*OLET\b", r"\bWELDING\s*OUTLET\b"],
    },
    {
        "name": "SOCKOLET",
        "body": "承插焊支管台",
        "patterns": [r"\bSOCKOLET\b", r"\bSOCKET\s*OLET\b", r"\bSOCK\s*OLET\b"],
    },
    {
        "name": "THREDOLET",
        "body": "螺纹支管台",
        "patterns": [r"\bTHREDOLET\b", r"\bTHREADOLET\b", r"\bTHREAD\s*OLET\b", r"\bTHRED\s*OLET\b"],
    },
    {
        "name": "LATROLET",
        "body": "斜支管台",
        "patterns": [r"\bLATROLET\b"],
    },
    {
        "name": "SWEEPOLET",
        "body": "SWEEPOLET",
        "patterns": [r"\bSWEEP\s*OLET\b"],
    },
    {
        "name": "GENERIC_OLET",
        "body": "支管台",
        "patterns": [r"\bOLET\b"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按高置信主词归一化种类训练集标签")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compile_rules() -> list[dict[str, Any]]:
    compiled = []
    for rule in ANCHOR_RULES:
        compiled.append(
            {
                "name": rule["name"],
                "body": rule["body"],
                "patterns": [re.compile(p, re.I) for p in rule["patterns"]],
            }
        )
    return compiled


def match_rule(text: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    for rule in rules:
        if any(p.search(text) for p in rule["patterns"]):
            return rule
    return None


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rules = compile_rules()
    changed = 0
    changed_by_rule: dict[str, int] = {}
    output_rows: list[dict[str, Any]] = []

    for row in rows:
        cloned = json.loads(json.dumps(row, ensure_ascii=False))
        text = cloned.get("input", "")
        output = cloned.get("output", {}) or {}
        type_obj = (output.get("TYPE", {}) or {})
        current_body = type_obj.get("BODY", "")
        rule = match_rule(text, rules)
        if rule and current_body != rule["body"]:
            type_obj["BODY"] = rule["body"]
            output["TYPE"] = type_obj
            cloned["output"] = output
            cloned["_normalized_by"] = rule["name"]
            cloned["_normalized_from_body"] = current_body
            changed += 1
            changed_by_rule[rule["name"]] = changed_by_rule.get(rule["name"], 0) + 1
        output_rows.append(cloned)

    args.output.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成: {args.output}")
    print(f"总条数: {len(output_rows)}")
    print(f"修正条数: {changed}")
    for name, count in sorted(changed_by_rule.items(), key=lambda x: (-x[1], x[0])):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
