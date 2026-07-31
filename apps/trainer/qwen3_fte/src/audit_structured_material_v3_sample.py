#!/usr/bin/env python3
"""Build a deterministic stratified audit sample for the v3 material dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = (
    ROOT
    / "apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范"
    / "结构化原始牌号/重新划分_v3"
)
TRAIN_NAME = "材质规范_结构化原始牌号_train.json"
VAL_NAME = "材质规范_结构化原始牌号_val.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "材质规范_v3_分层抽查400条.json"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_DATA_DIR / "材质规范_v3_分层抽查400条_报告.md"
SEED = "20260731-material-v3-audit"

MANUAL_FINDINGS: dict[tuple[str, int], dict[str, Any]] = {
    ("train", 64037): {
        "status": "明确错误",
        "category": "材质值残片",
        "reason": "原文为ASTM A234 WP11Cl.2，当前值截成WP11Cl；Cl.2不参与编码时也应完整去除，不能留下Cl残片。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "ASTM A234 WP11", "SPECIAL_REQ": []}
            ],
            "STANDARD": [{"BODY": "GBT12459B"}, {"BODY": "AB169"}],
        },
    },
    ("train", 64036): {
        "status": "明确错误",
        "category": "材质值残片",
        "reason": "原文为ASTM A234 WP11Cl.2，当前值截成WP11Cl；Cl.2不参与编码时也应完整去除，不能留下Cl残片。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "ASTM A234 WP11", "SPECIAL_REQ": []}
            ],
            "STANDARD": [{"BODY": "GBT12459B"}, {"BODY": "AB169"}],
        },
    },
    ("train", 24936): {
        "status": "明确错误",
        "category": "完整牌号信息丢失",
        "reason": "原文明示HDPE(PE100)，PE100是材料等级信息，不能只保留HDPE。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "HDPE(PE100)", "SPECIAL_REQ": []}
            ],
            "STANDARD": [{"BODY": "GBT13663.3"}, {"BODY": "HGT20592"}],
        },
    },
    ("train", 34256): {
        "status": "明确错误",
        "category": "双牌号前缀丢失",
        "reason": "原文明示F316/F316L，当前值F316/316L丢失第二个F前缀，不符合v3保留完整材质表达的规则。",
        "suggested_output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A182 F316/F316L",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [{"BODY": "AB1611"}],
        },
    },
    ("train", 50102): {
        "status": "明确错误",
        "category": "双牌号前缀丢失",
        "reason": "原文明示F316/F316L，当前值F316/316L丢失第二个F前缀，不符合v3保留完整材质表达的规则。",
        "suggested_output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A182 F316/F316L",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [{"BODY": "AB165"}],
        },
    },
    ("train", 59345): {
        "status": "明确错误",
        "category": "产品规范无原文证据",
        "reason": "原文未出现MSS SP-97、MS-97或SP-97，不能仅凭Olet产品名称补入MS97。",
        "suggested_output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A182 F304/304L",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [],
        },
    },
    ("val", 6634): {
        "status": "明确错误",
        "category": "产品规范无原文证据",
        "reason": "原文未出现MSS SP-97、MS-97或SP-97，不能仅凭Olet产品名称补入MS97。",
        "suggested_output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A182 F304/304L",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [],
        },
    },
    ("train", 50491): {
        "status": "疑似问题",
        "category": "防腐要求漏提",
        "reason": "原文明示“4PE加强级外防腐”，具有明确涂层语义；当前SPECIAL_REQ为空，但现有枚举没有4PE，需要确认归为PE还是新增4PE。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "20", "SPECIAL_REQ": ["PE"]}
            ],
            "STANDARD": [{"BODY": "GBT8163"}, {"BODY": "SHT3405"}],
        },
    },
    ("train", 33493): {
        "status": "明确错误",
        "category": "产品规范漏提",
        "reason": "原文明示02S403钢制管件标准图集，当前顶层STANDARD为空。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "Q235B", "SPECIAL_REQ": []}
            ],
            "STANDARD": [{"BODY": "02S403"}],
        },
    },
    ("train", 45253): {
        "status": "明确错误",
        "category": "并列材料标准漏提",
        "reason": "同一管材原文同时出现A269-TP316和ASTM A312 TP316（06Cr17Ni12Mo2），当前仅保留后者，遗漏主描述中的A269体系。",
        "suggested_output": {
            "MATERIAL": [
                {
                    "PART": "BODY",
                    "VALUE": "ASTM A269 TP316 / ASTM A312 TP316(06Cr17Ni12Mo2)",
                    "SPECIAL_REQ": [],
                }
            ],
            "STANDARD": [],
        },
    },
    ("train", 40928): {
        "status": "明确错误",
        "category": "原始牌号字符丢失",
        "reason": "原文为SS316L-EP，EP在该上下文是表面状态可不提取，但材质主体应保留SS316L，当前误成S316L。",
        "suggested_output": {
            "MATERIAL": [
                {"PART": "BODY", "VALUE": "SS316L", "SPECIAL_REQ": []}
            ],
            "STANDARD": [],
        },
    },
}


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "split": split,
            "source_index": index,
            "input": row["input"],
            "output": row["output"],
        }
        for index, row in enumerate(rows)
    ]


def stable_key(row: dict[str, Any]) -> str:
    payload = f"{SEED}\0{row['split']}\0{row['source_index']}\0{row['input']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def material_values(row: dict[str, Any]) -> list[str]:
    return [
        str(item.get("VALUE") or "")
        for item in row["output"].get("MATERIAL", [])
        if isinstance(item, dict)
    ]


def has_compound_value(row: dict[str, Any]) -> bool:
    return any(
        "/" in value or " or " in value.lower() or "(" in value or "（" in value
        for value in material_values(row)
    )


def has_special_req(row: dict[str, Any]) -> bool:
    return any(
        item.get("SPECIAL_REQ")
        for item in row["output"].get("MATERIAL", [])
        if isinstance(item, dict)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    args = parser.parse_args()

    rows = load_rows(args.data_dir / TRAIN_NAME, "train")
    rows.extend(load_rows(args.data_dir / VAL_NAME, "val"))
    rows.sort(key=stable_key)

    value_counts = Counter(
        value
        for row in rows
        for value in material_values(row)
        if value
    )
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    strata: list[tuple[str, int, Callable[[dict[str, Any]], bool]]] = [
        (
            "多物理部件",
            50,
            lambda row: len(row["output"].get("MATERIAL", [])) > 1,
        ),
        ("复合或双牌号", 50, has_compound_value),
        ("特殊要求", 40, has_special_req),
        (
            "无产品规范",
            30,
            lambda row: not row["output"].get("STANDARD"),
        ),
        (
            "低频材质",
            30,
            lambda row: any(
                value and value_counts[value] <= 3 for value in material_values(row)
            ),
        ),
        ("总体随机", 200, lambda row: True),
    ]

    for stratum, quota, predicate in strata:
        count = 0
        for row in rows:
            key = (row["split"], row["source_index"])
            if key in selected_keys or not predicate(row):
                continue
            selected_row = dict(row)
            selected_row["stratum"] = stratum
            selected.append(selected_row)
            selected_keys.add(key)
            count += 1
            if count == quota:
                break
        if count != quota:
            raise RuntimeError(f"{stratum} only selected {count}/{quota} rows")

    selected.sort(key=lambda row: (row["stratum"], stable_key(row)))
    for sample_no, row in enumerate(selected, start=1):
        row["sample_no"] = sample_no
        default_audit = {
            "status": "正确",
            "category": "",
            "reason": "",
            "suggested_output": None,
        }
        row["audit"] = MANUAL_FINDINGS.get(
            (row["split"], row["source_index"]),
            default_audit,
        )

    status_counts = Counter(row["audit"]["status"] for row in selected)
    category_counts = Counter(
        row["audit"]["category"]
        for row in selected
        if row["audit"]["category"]
    )
    general_rows = [
        row for row in selected if row["stratum"] == "总体随机"
    ]
    general_issue_count = sum(
        row["audit"]["status"] != "正确" for row in general_rows
    )
    edge_rows = [
        row for row in selected if row["stratum"] != "总体随机"
    ]
    edge_issue_count = sum(
        row["audit"]["status"] != "正确" for row in edge_rows
    )

    report = {
        "source": {
            "train": str(args.data_dir / TRAIN_NAME),
            "val": str(args.data_dir / VAL_NAME),
            "total_rows": len(rows),
        },
        "sampling": {
            "seed": SEED,
            "sample_rows": len(selected),
            "strata": {name: quota for name, quota, _ in strata},
            "deduplicated_across_strata": True,
        },
        "summary": {
            "正确": status_counts["正确"],
            "疑似问题": status_counts["疑似问题"],
            "明确错误": status_counts["明确错误"],
            "明确错误率": round(status_counts["明确错误"] / len(selected), 6),
            "含疑似问题率": round(
                (status_counts["明确错误"] + status_counts["疑似问题"])
                / len(selected),
                6,
            ),
            "总体随机200条问题数": general_issue_count,
            "强化边界200条问题数": edge_issue_count,
            "问题类别": dict(category_counts),
        },
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    issue_rows = [
        row for row in selected if row["audit"]["status"] != "正确"
    ]
    markdown = [
        "# 材质规范 v3 分层抽查 400 条报告",
        "",
        "## 抽查结果",
        "",
        f"- 正确：{status_counts['正确']} 条",
        f"- 明确错误：{status_counts['明确错误']} 条"
        f"（{status_counts['明确错误'] / len(selected):.2%}）",
        f"- 疑似问题：{status_counts['疑似问题']} 条",
        f"- 含疑似问题："
        f"{(status_counts['明确错误'] + status_counts['疑似问题']) / len(selected):.2%}",
        f"- 总体随机 200 条：{general_issue_count} 条问题",
        f"- 强化边界 200 条：{edge_issue_count} 条问题",
        "",
        "该比例是分层审计命中率，不是全量数据的无偏错误率；样本主动提高了"
        "低频材质、复合牌号、多物理部件和特殊要求的占比。",
        "",
        "## 问题明细",
        "",
        "| # | 状态 | 类别 | 来源 | 原始描述 | 原因 |",
        "|---:|---|---|---|---|---|",
    ]
    for row in issue_rows:
        audit = row["audit"]
        escaped_input = row["input"].replace("|", "\\|").replace("\n", " ")
        escaped_reason = audit["reason"].replace("|", "\\|").replace("\n", " ")
        markdown.append(
            f"| {row['sample_no']} | {audit['status']} | "
            f"{audit['category']} | {row['split']}:{row['source_index']} | "
            f"{escaped_input} | {escaped_reason} |"
        )

    markdown.extend(
        [
            "",
            "## 结论",
            "",
            "普通样本标注较稳定，问题主要集中在边界结构。当前最需要处理的是："
            "材质值截断、显式双牌号前缀丢失、Olet 根据产品名称补入未明示的"
            "MS97、并列材料标准漏提，以及少量原始牌号字符丢失。",
            "",
            "完整 400 条样本、当前标签和建议修正值见同目录 JSON 报告。",
            "",
        ]
    )
    args.summary_output.write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps(report["sampling"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")
    print(f"summary_output: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
