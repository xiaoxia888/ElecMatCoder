#!/usr/bin/env python3
"""Audit fitting type train/val labels without modifying source datasets."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "apps/trainer/qwen3_fte/output/按8类拆分数据集/种类"
DEFAULT_OUTPUT = (
    DATA_DIR
    / "管件标注审查_20260805"
    / "管件_train_val_基础错误标签审核_20260805.json"
)
DATASETS = {
    "train": DATA_DIR / "管件_train.json",
    "val": DATA_DIR / "管件_val.json",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return rows


def add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def remove_values(values: list[str], targets: set[str]) -> list[str]:
    return [value for value in values if value not in targets]


def has_token(text: str, token: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, re.I) is not None


def audit_row(dataset: str, index: int, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    text = str(row.get("input", ""))
    current = row.get("output", {})
    proposed = copy.deepcopy(current)
    type_data = proposed.get("TYPE", {})
    body = str(type_data.get("BODY", ""))
    geometry = type_data.get("GEOMETRY", {})
    manu = list(type_data.get("MANU", []))
    conn = list(type_data.get("CONN", []))
    issues: list[str] = []
    reasons: list[str] = []
    evidence: list[str] = []

    if body == "加强管咀":
        type_data["BODY"] = "加强管嘴"
        issues.append("BODY同义词未统一")
        reasons.append("当前正式BODY词表统一使用“加强管嘴”，“加强管咀”不再作为独立标签。")
        evidence.append("加强管咀")

    if has_token(text, "STS"):
        sts_changed = False
        if type_data.get("BODY") in {"三通", "等径三通"}:
            if type_data.get("BODY") != "等径三通":
                type_data["BODY"] = "等径三通"
                sts_changed = True
        if "SW" not in conn:
            add_unique(conn, "SW")
            sts_changed = True
        if sts_changed:
            issues.append("STS产品代号语义漏标")
            reasons.append("STS是承插焊等径三通代号，应标为等径三通并保留CONN=SW。")
            evidence.append("STS")

    if has_token(text, "WTS") and "WELDED" not in manu:
        add_unique(manu, "WELDED")
        issues.append("WTS焊接产品代号漏标")
        reasons.append("WTS是原文明示的焊接三通产品代号，应保留MANU=WELDED。")
        evidence.append("WTS")

    angled_olet = bool(
        re.search(r"WOL\s*-\s*45", text, re.I)
        or re.search(r"45\s*[°度]?\s*WELD[ -]?OLET", text, re.I)
        or re.search(r"(?:对焊|承插焊)\s*45\s*[°度]\s*支管台", text, re.I)
    )
    if angled_olet and (type_data.get("BODY") != "斜支管台" or geometry.get("ANGLE") != "45"):
        type_data["BODY"] = "斜支管台"
        geometry["ANGLE"] = "45"
        issues.append("45度支管台主体未统一")
        reasons.append("WOL-45、45° WELDOLET及明确45°支管台按当前口径统一为斜支管台，角度标为45。")
        evidence.append("45°支管台/WOL-45")

    if has_token(text, "GRV") and "GRV" not in conn:
        add_unique(conn, "GRV")
        issues.append("GRV沟槽连接漏标")
        reasons.append("原文明示GRV，属于需要由一阶段提取的沟槽连接形式。")
        evidence.append("GRV")

    deprecated_end_forms = {"TSE", "TBE"}
    present_deprecated = [value for value in conn if value in deprecated_end_forms]
    if present_deprecated:
        conn = remove_values(conn, deprecated_end_forms)
        issues.append("组合端部形式不应进入CONN")
        reasons.append("TSE/TBE及BLE、PLE组合端部暂由正则处理，不进入种类模型CONN。")
        evidence.extend(present_deprecated)

    explicit_smls = re.search(r"SMLS|SEAMLESS|无缝", text, re.I)
    if "SMLS" in manu and not explicit_smls:
        manu = remove_values(manu, {"SMLS"})
        issues.append("SMLS缺少原文明示证据")
        reasons.append("一阶段只抽取原文明示的制造方式，不能依据材质或标准推导SMLS。")
        evidence.append("原文未出现SMLS/SEAMLESS/无缝")

    construction_only_weld = re.search(r"焊接方法\s*[:：]|连接方式\s*[:：]\s*焊接", text)
    product_weld_evidence = re.search(
        r"(?<![A-Za-z0-9])WELDED(?![A-Za-z0-9])|(?<![A-Za-z0-9])WELD(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])WLD(?![A-Za-z0-9])|有缝|焊制|钢板制|SEAM|"
        r"(?<![A-Za-z0-9])W(?:90|45)E(?:L|S)?(?![A-Za-z0-9])|"
        r"(?<![A-Za-z0-9])W(?:TS|RT|RE|RC)(?![A-Za-z0-9])",
        text,
        re.I,
    )
    if "WELDED" in manu and construction_only_weld and not product_weld_evidence:
        manu = remove_values(manu, {"WELDED"})
        issues.append("施工焊接信息误标为制造方式")
        reasons.append("焊接方法或现场连接方式不等于产品制造工艺，不能据此标注MANU=WELDED。")
        evidence.append(construction_only_weld.group(0))

    if re.search(r"埋弧焊|(?<![A-Za-z0-9])SAW(?![A-Za-z0-9])", text, re.I) and "SAW" not in manu:
        manu = remove_values(manu, {"WELDED"})
        add_unique(manu, "SAW")
        issues.append("SAW制造方式未使用具体标签")
        reasons.append("原文明示埋弧焊/SAW，应使用具体MANU=SAW，而不是泛化为WELDED。")
        evidence.append("埋弧焊/SAW")

    absolute_radius = re.search(r"\bR\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*mm\b", text, re.I)
    if absolute_radius and not geometry.get("RADIUS"):
        geometry["RADIUS"] = f"{absolute_radius.group(1)}MM"
        issues.append("绝对弯曲半径漏标")
        reasons.append("原文明示绝对半径时应保留数值和MM单位，不能留空或丢失单位。")
        evidence.append(absolute_radius.group(0))

    type_data["GEOMETRY"] = geometry
    type_data["MANU"] = manu
    type_data["CONN"] = conn
    proposed["TYPE"] = type_data

    definite: dict[str, Any] | None = None
    if proposed != current:
        definite = {
            "来源数据集": dataset,
            "source_index": index,
            "问题类别": issues,
            "原始描述": text,
            "修正前标签": current,
            "建议修正标签": proposed,
            "原文证据": evidence,
            "中文原因": reasons,
        }

    manual: dict[str, Any] | None = None
    current_conn = current.get("TYPE", {}).get("CONN", [])
    if (
        re.search(r"FNPT", text, re.I)
        and re.search(r"连接方式\s*[:：].*承插焊", text)
        and "FNPT" in current_conn
        and "SW" in current_conn
    ):
        manual = {
            "来源数据集": dataset,
            "source_index": index,
            "问题类别": "同一描述存在FNPT与承插焊连接冲突",
            "原始描述": text,
            "当前标签": current,
            "中文原因": "原文明示FNPT，同时施工连接方式又写承插焊，不能自动决定是否追加SW，应人工确认产品端部语义。",
        }

    return definite, manual


def build_report() -> dict[str, Any]:
    definite: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}

    for dataset, path in DATASETS.items():
        rows = load_rows(path)
        row_counts[dataset] = len(rows)
        for index, row in enumerate(rows):
            definite_item, manual_item = audit_row(dataset, index, row)
            if definite_item:
                definite.append(definite_item)
            if manual_item:
                manual.append(manual_item)

    category_counts = Counter(
        category
        for item in definite
        for category in item["问题类别"]
    )
    dataset_counts = Counter(item["来源数据集"] for item in definite)

    return {
        "生成时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "说明": "本文件仅提供审核建议，未修改管件_train.json或管件_val.json。人工确认后再执行写回。",
        "数据源": {name: str(path) for name, path in DATASETS.items()},
        "统计": {
            "训练集条数": row_counts.get("train", 0),
            "验证集条数": row_counts.get("val", 0),
            "明确建议修改条数": len(definite),
            "需要人工判断条数": len(manual),
            "按数据集": dict(sorted(dataset_counts.items())),
            "按问题类别": dict(category_counts.most_common()),
        },
        "明确建议修改": definite,
        "需要人工判断": manual,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report["统计"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
