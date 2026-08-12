#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/"
    "V2转换审核/02_V2已审核通过数据.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "apps/trainer/qwen3_fte/output/按8类拆分数据集/尺寸壁厚磅级/V2已划分"
)


@dataclass
class Group:
    skeleton: str
    indices: list[int]
    structures: Counter[str]
    expressions: Counter[str]
    actual_values: Counter[str]

    @property
    def size(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按描述表达骨架、输出结构和实际值拆分尺寸壁厚磅级 V2 数据集。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--attempts", type=int, default=120)
    parser.add_argument("--common-min-count", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组")
    for index, row in enumerate(data):
        if not isinstance(row, dict) or not isinstance(row.get("output"), dict):
            raise ValueError(f"第 {index + 1} 条不是有效 V2 样本")
    return data


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    replacements = {
        "×": "X",
        "*": "X",
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "“": '"',
        "”": '"',
        "′": "'",
        "″": '"',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def description_skeleton(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\d+(?:\.\d+)?(?:\s+\d+/\d+|/\d+)?", "#", text)
    text = re.sub(r"\s*([,;:/()\[\]{}\\X+\-=])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def output_structure_signature(output: dict[str, Any]) -> str:
    item_parts: list[str] = []
    items = output.get("ITEMS")
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scope = str(item.get("SCOPE") or "").upper() or "EMPTY"
        role = str(item.get("ROLE") or "").upper() or "EMPTY"
        sizes = item.get("SIZE") if isinstance(item.get("SIZE"), list) else []
        thicknesses = (
            item.get("THICKNESS") if isinstance(item.get("THICKNESS"), list) else []
        )
        size_types = ",".join(
            str(entry.get("type") or "").upper()
            for entry in sizes
            if isinstance(entry, dict)
        ) or "-"
        thickness_types = ",".join(
            str(entry.get("type") or "").upper()
            for entry in thicknesses
            if isinstance(entry, dict)
        ) or "-"
        item_parts.append(f"{scope}/{role}[S:{size_types};T:{thickness_types}]")
    length = "Y" if str(output.get("LENGTH") or "").strip() else "N"
    pressure = "Y" if str(output.get("PRESSURE") or "").strip() else "N"
    return " > ".join(item_parts or ["NO_ITEMS"]) + f" | L:{length};P:{pressure}"


def expression_tags(value: Any) -> set[str]:
    original = str(value or "")
    text = normalize_text(original)
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", original))
    has_en = bool(re.search(r"[A-Z]", text))
    tags = {
        "LANG:MIXED" if has_zh and has_en else "LANG:ZH" if has_zh else "LANG:EN",
        "LAYOUT:MULTILINE" if "\n" in original else "LAYOUT:SINGLELINE",
        "LAYOUT:COMPACT" if re.search(r"[A-Z]\d|\d[A-Z]", text) else "LAYOUT:SPACED",
    }
    patterns = {
        "SIZE:DN": r"\bDN\s*\d",
        "SIZE:NPS": r"\bNPS\s*\d",
        "SIZE:OD_SYMBOL": r"[ΦØ]\s*\d",
        "SIZE:INCH": r"\d\s*(?:\"|'{2}|IN\b)",
        "FORM:X_PAIR": r"\d\s*X\s*(?:DN|NPS|Φ|Ø)?\s*\d",
        "FORM:SLASH_PAIR": r"\d(?:\.\d+)?\s*/\s*\d",
        "THK:SCH": r"\b(?:SCH|SCHEDULE|S-)\s*(?:\d|STD|XS|XXS)",
        "THK:STD_XS": r"\b(?:STD|XS|XXS)\b",
        "THK:MM": r"\d(?:\.\d+)?\s*MM\b",
        "THK:ANCHOR": r"\b(?:THK|WT|T)\s*=|壁厚",
        "PRESSURE:PN": r"\bPN\s*\d",
        "PRESSURE:CLASS": r"\b(?:CL|CLASS)\s*\d|\d\s*(?:LB|LBS|#)\b",
        "LENGTH:ANCHOR": r"\bL\s*=|长度",
        "STRUCT:JACKET": r"JACKET|夹套|内管|外管",
        "STRUCT:LINING": r"LINED|LINING|衬里|衬层|钢衬",
        "STRUCT:BRANCH": r"OLET|支管|三通|TEE|开口焊|管嘴",
        "STRUCT:REDUCING": r"REDUC|异径|大小头|偏心|同心",
    }
    for tag, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            tags.add(tag)
    return tags


def normalized_value(value: Any) -> str:
    text = normalize_text(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")


def actual_value_features(output: dict[str, Any]) -> Counter[str]:
    features: Counter[str] = Counter()
    items = output.get("ITEMS")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in ("SIZE", "THICKNESS"):
                entries = item.get(field)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    value = normalized_value(entry.get("value"))
                    value_type = normalize_text(entry.get("type")) or "UNKNOWN"
                    if value:
                        features[f"{field}:{value_type}:{value}"] += 1
    for field in ("LENGTH", "PRESSURE"):
        value = normalized_value(output.get(field))
        if value:
            features[f"{field}:{value}"] += 1
    return features


def build_groups(rows: list[dict[str, Any]]) -> tuple[list[Group], Counter[str], Counter[str], Counter[str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    row_structures: list[str] = []
    row_expressions: list[set[str]] = []
    row_actuals: list[Counter[str]] = []
    total_structures: Counter[str] = Counter()
    total_expressions: Counter[str] = Counter()
    total_actuals: Counter[str] = Counter()

    for index, row in enumerate(rows):
        skeleton = description_skeleton(row.get("input"))
        structure = output_structure_signature(row["output"])
        expressions = expression_tags(row.get("input"))
        actuals = actual_value_features(row["output"])
        grouped_indices[skeleton].append(index)
        row_structures.append(structure)
        row_expressions.append(expressions)
        row_actuals.append(actuals)
        total_structures[structure] += 1
        total_expressions.update(expressions)
        total_actuals.update(actuals)

    groups: list[Group] = []
    for skeleton, indices in grouped_indices.items():
        structures: Counter[str] = Counter()
        expressions: Counter[str] = Counter()
        actuals: Counter[str] = Counter()
        for index in indices:
            structures[row_structures[index]] += 1
            expressions.update(row_expressions[index])
            actuals.update(row_actuals[index])
        groups.append(Group(skeleton, indices, structures, expressions, actuals))
    return groups, total_structures, total_expressions, total_actuals


def can_add_group(
    group: Group,
    selected_actuals: Counter[str],
    total_actuals: Counter[str],
) -> bool:
    return all(
        total_actuals[feature] - selected_actuals[feature] - count >= 1
        for feature, count in group.actual_values.items()
    )


def add_group(
    group: Group,
    structures: Counter[str],
    expressions: Counter[str],
    actuals: Counter[str],
) -> None:
    structures.update(group.structures)
    expressions.update(group.expressions)
    actuals.update(group.actual_values)


def remove_group(
    group: Group,
    structures: Counter[str],
    expressions: Counter[str],
    actuals: Counter[str],
) -> None:
    structures.subtract(group.structures)
    expressions.subtract(group.expressions)
    actuals.subtract(group.actual_values)


def distribution_metrics(
    total: Counter[str],
    selected: Counter[str],
    target_ratio: float,
    minimum: int,
    group_support: Counter[str],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = 0
    for feature, count in total.items():
        if count < minimum:
            continue
        selected_count = selected[feature]
        selected_ratio = selected_count / count
        deviation_pp = abs(selected_ratio - target_ratio) * 100
        weighted_sum += deviation_pp * count
        total_weight += count
        details.append(
            {
                "feature": feature,
                "total_count": count,
                "train_count": count - selected_count,
                "val_count": selected_count,
                "val_share_percent": round(selected_ratio * 100, 3),
                "deviation_percentage_points": round(deviation_pp, 3),
                "skeleton_group_count": group_support[feature],
                "splittable": group_support[feature] >= 2,
            }
        )
    details.sort(key=lambda item: (-item["deviation_percentage_points"], -item["total_count"], item["feature"]))
    return {
        "common_feature_count": len(details),
        "common_feature_missing_in_val": sum(item["val_count"] == 0 for item in details),
        "splittable_common_feature_missing_in_val": sum(
            item["val_count"] == 0 and item["splittable"] for item in details
        ),
        "inseparable_common_feature_count": sum(not item["splittable"] for item in details),
        "inseparable_common_features": [
            item for item in details if not item["splittable"]
        ],
        "weighted_mean_deviation_percentage_points": round(weighted_sum / total_weight, 3) if total_weight else 0.0,
        "max_deviation_percentage_points": details[0]["deviation_percentage_points"] if details else 0.0,
        "top_deviations": details[:30],
    }


def candidate_objective(
    total_rows: int,
    val_rows: int,
    target_ratio: float,
    total_structures: Counter[str],
    val_structures: Counter[str],
    total_expressions: Counter[str],
    val_expressions: Counter[str],
    total_actuals: Counter[str],
    val_actuals: Counter[str],
    common_minimum: int,
    structure_support: Counter[str],
    expression_support: Counter[str],
    actual_support: Counter[str],
) -> tuple[float, dict[str, Any]]:
    structure_metrics = distribution_metrics(
        total_structures, val_structures, target_ratio, common_minimum, structure_support
    )
    expression_metrics = distribution_metrics(
        total_expressions, val_expressions, target_ratio, common_minimum, expression_support
    )
    actual_metrics = distribution_metrics(
        total_actuals, val_actuals, target_ratio, common_minimum, actual_support
    )
    ratio_error_pp = abs(val_rows / total_rows - target_ratio) * 100
    missing_penalty = 10 * (
        structure_metrics["splittable_common_feature_missing_in_val"]
        + expression_metrics["splittable_common_feature_missing_in_val"]
        + actual_metrics["splittable_common_feature_missing_in_val"]
    )
    objective = (
        ratio_error_pp * 30
        + structure_metrics["weighted_mean_deviation_percentage_points"] * 4
        + expression_metrics["weighted_mean_deviation_percentage_points"] * 2
        + actual_metrics["weighted_mean_deviation_percentage_points"] * 2
        + missing_penalty
    )
    return objective, {
        "ratio_error_percentage_points": round(ratio_error_pp, 4),
        "structure": structure_metrics,
        "expression": expression_metrics,
        "actual_value": actual_metrics,
    }


def choose_validation_groups(
    groups: list[Group],
    total_rows: int,
    target_ratio: float,
    total_structures: Counter[str],
    total_expressions: Counter[str],
    total_actuals: Counter[str],
    seed: int,
    attempts: int,
    common_minimum: int,
) -> tuple[set[int], dict[str, Any]]:
    target_rows = round(total_rows * target_ratio)
    best_selected: set[int] | None = None
    best_summary: dict[str, Any] | None = None
    best_objective = float("inf")

    structure_buckets: dict[str, list[int]] = defaultdict(list)
    structure_support: Counter[str] = Counter()
    expression_support: Counter[str] = Counter()
    actual_support: Counter[str] = Counter()
    for group_index, group in enumerate(groups):
        dominant_structure = max(
            group.structures,
            key=lambda feature: (group.structures[feature], feature),
        )
        structure_buckets[dominant_structure].append(group_index)
        structure_support.update(group.structures.keys())
        expression_support.update(group.expressions.keys())
        actual_support.update(group.actual_values.keys())

    for attempt in range(attempts):
        rng = random.Random(seed + attempt * 104729)
        selected: set[int] = set()
        val_rows = 0
        val_structures: Counter[str] = Counter()
        val_expressions: Counter[str] = Counter()
        val_actuals: Counter[str] = Counter()

        # 先在每种主输出结构内部按骨架组选取接近目标比例的子集，避免
        # 某个大骨架被随机选中后将对应结构在验证集中的占比推到极端值。
        bucket_items = list(structure_buckets.items())
        rng.shuffle(bucket_items)
        bucket_items.sort(key=lambda item: sum(groups[index].size for index in item[1]))
        for _, bucket_group_indices in bucket_items:
            bucket_target = round(
                sum(groups[index].size for index in bucket_group_indices) * target_ratio
            )
            bucket_selected_rows = 0
            bucket_order = list(bucket_group_indices)
            rng.shuffle(bucket_order)
            for group_index in bucket_order:
                group = groups[group_index]
                current_distance = abs(bucket_target - bucket_selected_rows)
                new_distance = abs(bucket_target - (bucket_selected_rows + group.size))
                # 目标位于两个可选子集的正中间时允许选入较小骨架；例如
                # 总计 28 条、骨架为 22+6 时，验证集选 6 条比完全缺失更有代表性。
                if new_distance > current_distance:
                    continue
                if not can_add_group(group, val_actuals, total_actuals):
                    continue
                selected.add(group_index)
                bucket_selected_rows += group.size
                val_rows += group.size
                add_group(group, val_structures, val_expressions, val_actuals)

        order = list(range(len(groups)))
        rng.shuffle(order)
        remaining = [index for index in order if index not in selected]
        rng.shuffle(remaining)
        while val_rows < target_rows:
            candidates = [
                index
                for index in remaining[:800]
                if can_add_group(groups[index], val_actuals, total_actuals)
            ]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda index: (
                    abs(target_rows - (val_rows + groups[index].size)),
                    groups[index].size,
                    rng.random(),
                ),
            )
            group = groups[chosen]
            if abs(target_rows - (val_rows + group.size)) > abs(target_rows - val_rows) and val_rows >= target_rows * 0.995:
                break
            selected.add(chosen)
            remaining.remove(chosen)
            val_rows += group.size
            add_group(group, val_structures, val_expressions, val_actuals)

        # 主结构分层完成后，主动补齐仍缺失且由多个骨架承载的常见表达与
        # 实际值。随后再调平总行数，并保护这些特征不被移除至零。
        coverage_specs = (
            (total_structures, val_structures, structure_support, "structures"),
            (total_expressions, val_expressions, expression_support, "expressions"),
            (total_actuals, val_actuals, actual_support, "actual_values"),
        )
        for total_counter, val_counter, support_counter, group_attribute in coverage_specs:
            missing_features = [
                feature
                for feature, count in total_counter.items()
                if count >= common_minimum
                and support_counter[feature] >= 2
                and val_counter[feature] == 0
            ]
            missing_features.sort(key=lambda feature: (total_counter[feature], feature))
            for feature in missing_features:
                if val_counter[feature] > 0:
                    continue
                candidates = [
                    group_index
                    for group_index, group in enumerate(groups)
                    if group_index not in selected
                    and getattr(group, group_attribute)[feature] > 0
                    and can_add_group(group, val_actuals, total_actuals)
                ]
                if not candidates:
                    continue
                chosen = min(
                    candidates,
                    key=lambda group_index: (
                        groups[group_index].size,
                        getattr(groups[group_index], group_attribute)[feature],
                        groups[group_index].skeleton,
                    ),
                )
                group = groups[chosen]
                selected.add(chosen)
                val_rows += group.size
                add_group(group, val_structures, val_expressions, val_actuals)

        if val_rows > target_rows:
            removable = list(selected)
            rng.shuffle(removable)
            for group_index in removable:
                group = groups[group_index]
                if abs(target_rows - (val_rows - group.size)) >= abs(target_rows - val_rows):
                    continue
                loses_common_coverage = any(
                    total_structures[feature] >= common_minimum
                    and structure_support[feature] >= 2
                    and val_structures[feature] - count <= 0
                    for feature, count in group.structures.items()
                ) or any(
                    total_expressions[feature] >= common_minimum
                    and expression_support[feature] >= 2
                    and val_expressions[feature] - count <= 0
                    for feature, count in group.expressions.items()
                ) or any(
                    total_actuals[feature] >= common_minimum
                    and actual_support[feature] >= 2
                    and val_actuals[feature] - count <= 0
                    for feature, count in group.actual_values.items()
                )
                if loses_common_coverage:
                    continue
                selected.remove(group_index)
                val_rows -= group.size
                remove_group(group, val_structures, val_expressions, val_actuals)

        objective, metrics = candidate_objective(
            total_rows,
            val_rows,
            target_ratio,
            total_structures,
            val_structures,
            total_expressions,
            val_expressions,
            total_actuals,
            val_actuals,
            common_minimum,
            structure_support,
            expression_support,
            actual_support,
        )
        if objective < best_objective:
            best_objective = objective
            best_selected = selected
            best_summary = {
                "attempt": attempt + 1,
                "attempt_seed": seed + attempt * 104729,
                "objective": round(objective, 6),
                "val_rows": val_rows,
                "val_structures": val_structures,
                "val_expressions": val_expressions,
                "val_actuals": val_actuals,
                "metrics": metrics,
            }

    if best_selected is None or best_summary is None:
        raise RuntimeError("未生成可用划分")
    return best_selected, best_summary


def counter_prefix_summary(total: Counter[str], val: Counter[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "val": 0, "unique": 0})
    for feature, count in total.items():
        prefix = feature.split(":", 2)[0]
        grouped[prefix]["total"] += count
        grouped[prefix]["val"] += val[feature]
        grouped[prefix]["unique"] += 1
    return [
        {
            "field": prefix,
            "total_occurrences": values["total"],
            "train_occurrences": values["total"] - values["val"],
            "val_occurrences": values["val"],
            "unique_values": values["unique"],
        }
        for prefix, values in sorted(grouped.items())
    ]


def build_report(
    input_path: Path,
    rows: list[dict[str, Any]],
    groups: list[Group],
    selected_group_indices: set[int],
    summary: dict[str, Any],
    target_ratio: float,
    seed: int,
    attempts: int,
    common_minimum: int,
    total_actuals: Counter[str],
) -> dict[str, Any]:
    val_indices = {
        index
        for group_index in selected_group_indices
        for index in groups[group_index].indices
    }
    train_indices = set(range(len(rows))) - val_indices
    train_inputs = {normalize_text(rows[index].get("input")) for index in train_indices}
    val_inputs = {normalize_text(rows[index].get("input")) for index in val_indices}
    train_skeletons = {groups[index].skeleton for index in range(len(groups)) if index not in selected_group_indices}
    val_skeletons = {groups[index].skeleton for index in selected_group_indices}
    val_actuals: Counter[str] = summary["val_actuals"]
    missing_actuals = sorted(
        feature for feature, count in val_actuals.items() if total_actuals[feature] - count <= 0
    )
    metrics = summary["metrics"]
    actual_ratio = len(val_indices) / len(rows)
    checks = [
        {
            "check": "验证集比例与目标相差不超过0.25个百分点",
            "passed": metrics["ratio_error_percentage_points"] <= 0.25,
        },
        {"check": "描述表达骨架无跨集合泄漏", "passed": not (train_skeletons & val_skeletons)},
        {"check": "归一化原文无跨集合重复", "passed": not (train_inputs & val_inputs)},
        {"check": "验证集实际值均在训练集保留样本", "passed": not missing_actuals},
        {
            "check": "可分的常见输出结构在验证集无缺失",
            "passed": metrics["structure"]["splittable_common_feature_missing_in_val"] == 0,
        },
        {
            "check": "可分的常见表达标签在验证集无缺失",
            "passed": metrics["expression"]["splittable_common_feature_missing_in_val"] == 0,
        },
        {
            "check": "可分的常见实际值在验证集无缺失",
            "passed": metrics["actual_value"]["splittable_common_feature_missing_in_val"] == 0,
        },
        {
            "check": "三类特征加权平均偏差均不超过1.5个百分点",
            "passed": all(
                metrics[key]["weighted_mean_deviation_percentage_points"] <= 1.5
                for key in ("structure", "expression", "actual_value")
            ),
        },
    ]
    return {
        "source": str(input_path),
        "strategy": {
            "description_skeleton_grouped": True,
            "validation_actual_value_must_remain_in_train": True,
            "optimized_dimensions": ["输出结构", "描述表达形式", "尺寸壁厚磅级实际值"],
            "target_val_ratio": target_ratio,
            "common_feature_min_count": common_minimum,
            "base_seed": seed,
            "attempts": attempts,
            "selected_attempt": summary["attempt"],
            "selected_attempt_seed": summary["attempt_seed"],
            "objective": summary["objective"],
        },
        "counts": {
            "total_rows": len(rows),
            "train_rows": len(train_indices),
            "val_rows": len(val_indices),
            "actual_val_ratio_percent": round(actual_ratio * 100, 4),
            "total_skeleton_groups": len(groups),
            "train_skeleton_groups": len(train_skeletons),
            "val_skeleton_groups": len(val_skeletons),
            "skeleton_overlap": len(train_skeletons & val_skeletons),
            "normalized_input_overlap": len(train_inputs & val_inputs),
        },
        "quality": metrics,
        "actual_value_field_summary": counter_prefix_summary(total_actuals, val_actuals),
        "validation_values_missing_from_train": missing_actuals,
        "acceptance_checks": checks,
        "accepted": all(check["passed"] for check in checks),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio 必须在 0 和 0.5 之间")
    rows = load_rows(args.input)
    groups, total_structures, total_expressions, total_actuals = build_groups(rows)
    selected_groups, summary = choose_validation_groups(
        groups,
        len(rows),
        args.val_ratio,
        total_structures,
        total_expressions,
        total_actuals,
        args.seed,
        args.attempts,
        args.common_min_count,
    )
    val_indices = {
        index
        for group_index in selected_groups
        for index in groups[group_index].indices
    }
    train_rows = [row for index, row in enumerate(rows) if index not in val_indices]
    val_rows = [row for index, row in enumerate(rows) if index in val_indices]
    report = build_report(
        args.input,
        rows,
        groups,
        selected_groups,
        summary,
        args.val_ratio,
        args.seed,
        args.attempts,
        args.common_min_count,
        total_actuals,
    )

    train_path = args.output_dir / "尺寸壁厚磅级V2_train.json"
    val_path = args.output_dir / "尺寸壁厚磅级V2_val.json"
    report_path = args.output_dir / "尺寸壁厚磅级V2_划分报告.json"
    write_json(train_path, train_rows)
    write_json(val_path, val_rows)
    write_json(report_path, report)

    print(f"源数据: {len(rows)}")
    print(f"训练集: {len(train_rows)}")
    print(f"验证集: {len(val_rows)} ({len(val_rows) / len(rows):.2%})")
    print(f"表达骨架组: {len(groups)}")
    print(f"报告验收: {'通过' if report['accepted'] else '未通过'}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
