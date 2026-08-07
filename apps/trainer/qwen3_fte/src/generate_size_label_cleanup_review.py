#!/usr/bin/env python3
"""Generate a consolidated, review-only SIZE_ITEMS cleanup proposal."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from audit_size_annotation_consistency import audit_rows as audit_consistency
from audit_size_dataset_second_round import (
    _item_position,
    audit as audit_explicit_evidence,
    canonical_number,
    insert_size_item_in_source_order,
)


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "按8类拆分数据集"
    / "尺寸壁厚磅级"
    / "2026_0729_Archive"
    / "尺寸壁厚磅级C1训练集.json"
)
DEFAULT_OUTPUT = DEFAULT_DATASET.with_name(
    "尺寸壁厚磅级C1训练集_SIZE_ITEMS标签清洗审核.json"
)

CONFIDENCE_ORDER = {"高": 2, "中": 1, "低": 0}
SPECIFICATION_OD_THICKNESS_RE = re.compile(
    r"规格\s*[:：]\s*[ΦφØ]\s*"
    r"(?P<od>\d+(?:\.\d+)?(?:\s*[xX×*]\s*\d+(?:\.\d+)?)+)\s*/\s*"
    r"(?P<thickness>\d+(?:\.\d+)?(?:\s*[xX×*]\s*\d+(?:\.\d+)?)+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_input(text: str) -> str:
    return re.sub(
        r"[\s，,;；:：/\\()（）\[\]{}_-]+",
        "",
        str(text or ""),
    ).upper()


def size_signature(items: Any) -> str:
    return json.dumps(items, ensure_ascii=False, sort_keys=True)


def canonical_item(item: dict[str, Any]) -> tuple[str, str]:
    item_type = str(item.get("type") or "").upper()
    raw_value = str(item.get("value") or "")
    value = raw_value if item_type == "INCH" else canonical_number(raw_value)
    return item_type, value


def deduplicate_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = canonical_item(item)
        if key in seen:
            continue
        result.append(deepcopy(item))
        seen.add(key)
    return result


def add_item_in_source_order(
    text: str,
    items: list[dict[str, str]],
    item: dict[str, str],
) -> list[dict[str, str]]:
    result = deepcopy(items)
    if canonical_item(item) in {canonical_item(existing) for existing in result}:
        return result
    insert_size_item_in_source_order(text, result, item, _item_position(text, item))
    return deduplicate_items(result)


def merge_proposal(
    proposals: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    source_index: int,
    category: str,
    confidence: str,
    reason: str,
    transform: Callable[[list[dict[str, str]]], list[dict[str, str]]],
    evidence: dict[str, Any] | None = None,
) -> None:
    row = rows[source_index]
    current = deepcopy(row["output"]["SIZE_ITEMS"])
    proposal = proposals.setdefault(
        source_index,
        {
            "source_index": source_index,
            "原始描述": row.get("input", ""),
            "修改字段": "SIZE_ITEMS",
            "当前标签": current,
            "建议标签": current,
            "问题类别": [],
            "置信等级": confidence,
            "中文原因": [],
            "证据": [],
        },
    )
    before = deepcopy(proposal["建议标签"])
    after = deduplicate_items(transform(before))
    if before == after:
        return
    proposal["建议标签"] = after
    if category not in proposal["问题类别"]:
        proposal["问题类别"].append(category)
    if reason not in proposal["中文原因"]:
        proposal["中文原因"].append(reason)
    if evidence and evidence not in proposal["证据"]:
        proposal["证据"].append(evidence)
    if CONFIDENCE_ORDER[confidence] < CONFIDENCE_ORDER[proposal["置信等级"]]:
        proposal["置信等级"] = confidence


def replace_dn_with_inch(
    items: list[dict[str, str]],
    values: set[str],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in items:
        item_type, value = canonical_item(item)
        if item_type == "DN" and value in values:
            result.append({"type": "INCH", "value": str(item.get("value") or "")})
        else:
            result.append(deepcopy(item))
    return result


def remove_dn_values(
    items: list[dict[str, str]],
    values: set[str],
) -> list[dict[str, str]]:
    return [
        deepcopy(item)
        for item in items
        if not (canonical_item(item)[0] == "DN" and canonical_item(item)[1] in values)
    ]


def add_od_values(
    text: str,
    items: list[dict[str, str]],
    values: list[str],
) -> list[dict[str, str]]:
    result = deepcopy(items)
    for value in values:
        result = add_item_in_source_order(text, result, {"type": "OD", "value": value})
    return result


def specification_size_items(text: str) -> list[dict[str, str]] | None:
    """Extract product ODs from `规格:ΦOD.../thickness...` structures."""
    match = SPECIFICATION_OD_THICKNESS_RE.search(text)
    if not match:
        return None
    return [
        {"type": "OD", "value": canonical_number(value)}
        for value in re.split(r"\s*[xX×*]\s*", match.group("od"))
    ]


def repair_truncated_or_missing_od(
    text: str,
    items: list[dict[str, str]],
    missing_values: list[str],
) -> list[dict[str, str]]:
    result = deepcopy(items)
    for missing in missing_values:
        missing_number = canonical_number(missing)
        replacement_index = None
        if "." in missing_number:
            integer_part = missing_number.split(".", 1)[0]
            for index, item in enumerate(result):
                item_type, value = canonical_item(item)
                if item_type == "OD" and value == integer_part:
                    replacement_index = index
                    break
        if replacement_index is not None:
            result[replacement_index] = {"type": "OD", "value": missing_number}
        else:
            result = add_item_in_source_order(
                text,
                result,
                {"type": "OD", "value": missing_number},
            )
    return result


def conflict_recommendation(
    text: str,
    variants: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    specification = specification_size_items(text)
    if specification is not None:
        return (
            specification,
            "规格块中斜杠前是外径组合，斜杠后是壁厚组合；壁厚不得进入SIZE_ITEMS。",
        )

    installation = re.search(
        r"管道安装\s+(\d+(?:\.\d+)?)\s*mm\b",
        text,
        flags=re.IGNORECASE,
    )
    if installation:
        return (
            [{"type": "DN", "value": canonical_number(installation.group(1))}],
            "管道安装模板中的裸mm值表示公称口径，统一标为DN。",
        )

    pipe = re.search(
        r"(?:CS|SS)\s+PIPE\s+(\d+(?:\.\d+)?)\s+SCH\w*.*?\bDN\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if pipe:
        return (
            [
                {"type": "INCH", "value": canonical_number(pipe.group(1))},
                {"type": "DN", "value": canonical_number(pipe.group(2))},
            ],
            "受控PIPE模板同时给出裸寸径和明确DN，两项均保留并按原文顺序输出。",
        )

    winner = max(variants, key=lambda variant: (len(variant["source_indexes"]), -len(variant["label"])))
    return deepcopy(winner["label"]), "无更强结构证据，暂按同描述中的多数标签推荐，必须人工复核。"


def duplicate_statistics_and_proposals(
    rows: list[dict[str, Any]],
    proposals: dict[int, dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[normalize_input(row.get("input", ""))].append((index, row))

    duplicate_groups = {key: members for key, members in groups.items() if len(members) > 1}
    full_conflict_groups = 0
    size_conflict_summaries: list[dict[str, Any]] = []

    for members in duplicate_groups.values():
        full_outputs = {
            json.dumps(row.get("output"), ensure_ascii=False, sort_keys=True)
            for _, row in members
        }
        full_conflict_groups += len(full_outputs) > 1

        variants_by_signature: dict[str, dict[str, Any]] = {}
        for source_index, row in members:
            label = row["output"]["SIZE_ITEMS"]
            signature = size_signature(label)
            variant = variants_by_signature.setdefault(
                signature,
                {"label": deepcopy(label), "source_indexes": []},
            )
            variant["source_indexes"].append(source_index)
        if len(variants_by_signature) <= 1:
            continue

        variants = list(variants_by_signature.values())
        text = str(members[0][1].get("input") or "")
        recommended, reason = conflict_recommendation(text, variants)
        changed_indexes: list[int] = []
        for source_index, row in members:
            if row["output"]["SIZE_ITEMS"] == recommended:
                continue
            changed_indexes.append(source_index)
            merge_proposal(
                proposals,
                rows,
                source_index,
                "同描述SIZE_ITEMS标签冲突",
                "高" if not reason.startswith("无更强") else "低",
                reason,
                lambda _items, value=deepcopy(recommended): deepcopy(value),
                {"同组源索引": [index for index, _ in members]},
            )
        size_conflict_summaries.append(
            {
                "原始描述": text,
                "同组源索引": [index for index, _ in members],
                "标签变体": variants,
                "推荐标签": recommended,
                "推荐原因": reason,
                "建议修改源索引": changed_indexes,
            }
        )

    return (
        {
            "归一化后唯一描述数": len(groups),
            "重复描述组数": len(duplicate_groups),
            "重复描述组涉及行数": sum(len(members) for members in duplicate_groups.values()),
            "任意输出字段冲突组数": full_conflict_groups,
            "SIZE_ITEMS冲突组数": len(size_conflict_summaries),
        },
        size_conflict_summaries,
    )


def build_review(rows: list[dict[str, Any]], dataset_path: Path) -> dict[str, Any]:
    proposals: dict[int, dict[str, Any]] = {}
    consistency_issues = audit_consistency(rows)
    structured_size_indexes: set[int] = set()

    for source_index, row in enumerate(rows):
        text = str(row.get("input") or "")
        expected = specification_size_items(text)
        if expected is None:
            continue
        structured_size_indexes.add(source_index)
        merge_proposal(
            proposals,
            rows,
            source_index,
            "规格外径/壁厚结构误标",
            "高",
            "规格块中斜杠前是产品外径，斜杠后是对应壁厚；斜杠后的数值不得进入SIZE_ITEMS。",
            lambda _items, value=deepcopy(expected): deepcopy(value),
            {"强结构规则": "Φ外径组合/壁厚组合"},
        )

    for issue in consistency_issues:
        source_index = issue["source_index"]
        text = str(rows[source_index].get("input") or "")
        category = issue["category"]
        evidence = issue["evidence"]
        if category == "radius_dn_material_mislabeled_as_dn":
            values = set(evidence["radius_material_values"])
            merge_proposal(
                proposals,
                rows,
                source_index,
                "R=倍数DN后的材质数字误标为DN",
                "高",
                "R=倍数DN表示弯曲半径，紧随其后的“数字#”是材质，不是公称尺寸。",
                lambda items, values=values: remove_dn_values(items, values),
                evidence,
            )
        elif category == "inch_mislabeled_as_dn":
            values = set(evidence["explicit_inch_values"])
            merge_proposal(
                proposals,
                rows,
                source_index,
                "明确英制尺寸误标为DN",
                "高",
                "原文带英寸引号或NPS锚点，同值只能标为INCH，不能标为DN。",
                lambda items, values=values: replace_dn_with_inch(items, values),
                evidence,
            )
        elif category == "pipe_od_mislabeled_as_dn":
            missing_values = list(evidence["missing_od_values"])
            merge_proposal(
                proposals,
                rows,
                source_index,
                "PIPE公制外径误标为DN",
                "高",
                "PIPE后的公制管径与SCH/壁厚构成外径规格，当前DN标签无DN锚点。",
                lambda items, values=missing_values, text=text: add_od_values(
                    text,
                    [item for item in items if str(item.get("type") or "").upper() != "DN"],
                    values,
                ),
                evidence,
            )
        elif category == "explicit_od_missing":
            missing_values = list(evidence["missing_od_values"])
            merge_proposal(
                proposals,
                rows,
                source_index,
                "明确外径OD漏标或数值截断",
                "高",
                "原文通过OD/Φ明确给出外径，当前标签漏标或丢失小数部分。",
                lambda items, values=missing_values, text=text: repair_truncated_or_missing_od(
                    text,
                    items,
                    values,
                ),
                evidence,
            )

    duplicate_stats, conflict_summaries = duplicate_statistics_and_proposals(rows, proposals)

    explicit_report = audit_explicit_evidence(rows)
    explicit_categories = {
        "明确英制尺寸漏标": "中",
        "管道裸寸径漏标": "中",
        "明确DN尺寸漏标": "中",
        "明确外径OD漏标": "中",
    }
    for category, confidence in explicit_categories.items():
        for item in explicit_report["待确认修改"][category]:
            source_index = item["source_index"]
            if source_index in structured_size_indexes:
                continue
            existing_categories = proposals.get(source_index, {}).get("问题类别", [])
            if (
                category == "明确DN尺寸漏标"
                and "R=倍数DN后的材质数字误标为DN" in existing_categories
            ):
                continue
            text = str(rows[source_index].get("input") or "")
            additions = deepcopy(item.get("建议新增") or [])
            merge_proposal(
                proposals,
                rows,
                source_index,
                category,
                confidence,
                item["中文原因"] + " 仍需确认该尺寸属于产品主体，而非技术要求或附加加工。",
                lambda items, additions=additions, text=text: _add_items(text, items, additions),
                {"建议新增": additions},
            )

    for issue in consistency_issues:
        if issue["category"] not in {
            "pipe_od_missing_with_explicit_dn",
            "pipe_od_missing_with_explicit_inch",
        }:
            continue
        source_index = issue["source_index"]
        if source_index in structured_size_indexes:
            continue
        text = str(rows[source_index].get("input") or "")
        missing_values = list(issue["evidence"]["missing_od_values"])
        merge_proposal(
            proposals,
            rows,
            source_index,
            "PIPE公制外径证据漏标",
            "中",
            "PIPE公制外径与明确DN/英寸同时出现；按一阶段保留原文明示证据的口径建议补充OD。",
            lambda items, values=missing_values, text=text: add_od_values(text, items, values),
            issue["evidence"],
        )

    effective = [
        proposal
        for proposal in proposals.values()
        if proposal["当前标签"] != proposal["建议标签"]
    ]
    for proposal in effective:
        proposal["中文原因"] = "；".join(proposal["中文原因"])

    high = sorted(
        (proposal for proposal in effective if proposal["置信等级"] == "高"),
        key=lambda item: item["source_index"],
    )
    medium = sorted(
        (proposal for proposal in effective if proposal["置信等级"] == "中"),
        key=lambda item: item["source_index"],
    )
    low = sorted(
        (proposal for proposal in effective if proposal["置信等级"] == "低"),
        key=lambda item: item["source_index"],
    )
    category_counts = Counter(
        category
        for proposal in effective
        for category in proposal["问题类别"]
    )

    return {
        "说明": [
            "本文件仅为SIZE_ITEMS标签清洗审核方案，未修改原训练集。",
            "重复描述组表示归一化后的同一描述出现至少两次；重复本身不等于错误，只有标签冲突或重复权重失衡才需要处理。",
            "高置信度建议可在人工确认后统一写回；中、低置信度建议必须确认尺寸属于产品主体，而不是技术要求或附加加工。",
            "同一source_index只出现一次，多个命中规则已合并为一个最终建议标签。",
        ],
        "源数据集": str(dataset_path),
        "训练集总条数": len(rows),
        "重复描述统计": duplicate_stats,
        "审核统计": {
            "建议修改行数": len(effective),
            "高置信度": len(high),
            "中置信度": len(medium),
            "低置信度": len(low),
            "分类命中统计": dict(sorted(category_counts.items())),
        },
        "重复尺寸冲突组": conflict_summaries,
        "待确认修改": {
            "高置信度尺寸标签清洗": high,
            "中置信度尺寸证据补全": medium,
            "低置信度待人工判断": low,
        },
    }


def _add_items(
    text: str,
    items: list[dict[str, str]],
    additions: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = deepcopy(items)
    for addition in additions:
        result = add_item_in_source_order(text, result, addition)
    return result


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("训练集顶层必须是JSON数组")
    review = build_review(rows, dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                **review["重复描述统计"],
                **review["审核统计"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
