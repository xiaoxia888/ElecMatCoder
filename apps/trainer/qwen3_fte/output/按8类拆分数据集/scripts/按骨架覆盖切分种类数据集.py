#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "种类.json"
DEFAULT_TRAIN_OUTPUT = BASE_DIR / "种类_train.json"
DEFAULT_VAL_OUTPUT = BASE_DIR / "种类_val.json"
DEFAULT_REPORT_OUTPUT = BASE_DIR / "种类_signature_split_report.json"

TYPE_KEYWORDS = {
    "PIPE", "TUBE", "ELBOW", "TEE", "Y", "REDUCER", "ECC", "CONC", "CAP",
    "OLET", "SOCKOLET", "WELDOLET", "THREDOLET", "NIPOLET", "LATROLET",
    "FLANGE", "FLG", "WN", "SW", "SO", "LJ", "BL", "THD", "THRD", "BW",
    "BE", "PE", "PBE", "RF", "FF", "RTJ", "LR", "SR", "SMLS", "ERW",
    "SEAMLESS", "FORGED", "WELDED", "SPADE", "SPACER", "NIPPLE", "UNION",
    "COUPLING", "CROSS",
}

STOPWORDS = {
    "CS", "SS", "HT", "LT", "STD", "XS", "XXS", "FRP", "DARAKANE", "GALV",
    "GALVANIZED", "MAIN", "TYPE",
}

STANDARD_PREFIXES = (
    "GB", "HG", "SH", "NB", "ASME", "ASTM", "MSS", "DIN", "EN", "ISO",
    "JIS", "API", "AWWA", "BS", "ANSI",
)

CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


@dataclass
class TemplateGroup:
    signature_key: str
    signature_label: dict[str, Any]
    body: str
    template_key: str
    rows: list[dict[str, Any]]
    representative_text: str

    @property
    def size(self) -> int:
        return len(self.rows)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层不是 JSON 数组")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path} 中存在非对象样本")
    return data


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_output_value(sample: dict[str, Any]) -> dict[str, Any]:
    output = sample.get("output")
    return output if isinstance(output, dict) else {}


def get_type_value(sample: dict[str, Any]) -> dict[str, Any]:
    type_value = get_output_value(sample).get("TYPE")
    return type_value if isinstance(type_value, dict) else {}


def normalize_string_list(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    result = sorted({str(v).strip().upper() for v in values if str(v).strip()})
    return tuple(result)


def build_signature_parts(sample: dict[str, Any]) -> dict[str, Any]:
    output = get_output_value(sample)
    type_value = get_type_value(sample)

    geometry = type_value.get("GEOMETRY")
    angle = ""
    radius = ""
    if isinstance(geometry, dict):
        angle = str(geometry.get("ANGLE", "") or "").strip()
        radius = str(geometry.get("RADIUS", "") or "").strip()

    return {
        "CATEGORY": str(output.get("CATEGORY", "") or "").strip(),
        "BODY": str(type_value.get("BODY", "") or "").strip(),
        "ANGLE": angle,
        "RADIUS": radius,
        "FLANGE_STYLE": str(type_value.get("FLANGE_STYLE", "") or "").strip(),
        "MANU": list(normalize_string_list(type_value.get("MANU"))),
        "CONN": list(normalize_string_list(type_value.get("CONN"))),
        "SEAL": list(normalize_string_list(type_value.get("SEAL"))),
    }


def build_signature_key(sample: dict[str, Any]) -> str:
    parts = build_signature_parts(sample)
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def normalize_text_for_template(text: str) -> str:
    value = str(text or "").upper()
    value = value.replace("×", "X")
    value = value.replace('"', " INCH ")

    value = re.sub(
        r"\b(?:GB|HG|SH|NB|ASME|ASTM|MSS|DIN|EN|ISO|JIS|API|AWWA|BS|ANSI)"
        r"(?:\s*/?\s*[A-Z]*)?(?:[\s\-./()]*[A-Z0-9]+)+",
        " ",
        value,
        flags=re.IGNORECASE,
    )

    patterns = [
        r"\bDN\s*\d+(?:\.\d+)?\b",
        r"\bOD\s*\d+(?:\.\d+)?\b",
        r"\bSCH\s*\d+[A-Z]*\b",
        r"\bCL\s*\d+\b",
        r"\bPN\s*\d+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*MM\b",
        r"\b\d+(?:\.\d+)?\s*MPA\b",
        r"\b\d+(?:\.\d+)?\s*IN(?:CH)?\b",
        r"\b\d+(?:\.\d+)?D\b",
        r"\b\d+(?:\.\d+)?(?:X\d+(?:\.\d+)?)+\b",
        r"\b\d{4}\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)

    value = re.sub(r"[,:;|/()\-_=+*#]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_template_tokens(text: str) -> list[str]:
    normalized = normalize_text_for_template(text)
    tokens: list[str] = []
    for token in normalized.split():
        clean = token.strip().upper()
        if not clean or clean in STOPWORDS:
            continue
        if clean in TYPE_KEYWORDS:
            tokens.append(clean)
            continue
        if any(clean.startswith(prefix) for prefix in STANDARD_PREFIXES):
            continue
        if any(ch.isdigit() for ch in clean):
            continue
        if re.fullmatch(r"[A-Z]{1,2}", clean):
            continue
        if re.fullmatch(r"[A-Z]{3,}", clean):
            tokens.append(clean)
    tokens.extend(CHINESE_PATTERN.findall(normalized))

    uniq: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        uniq.append(token)
    return uniq


def build_template_key(sample: dict[str, Any]) -> str:
    return " ".join(extract_template_tokens(sample.get("input", "")))


def build_groups(rows: list[dict[str, Any]]) -> dict[str, list[TemplateGroup]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    signature_labels: dict[str, dict[str, Any]] = {}

    for row in rows:
        signature_key = build_signature_key(row)
        template_key = build_template_key(row)
        grouped[signature_key][template_key].append(row)
        signature_labels.setdefault(signature_key, build_signature_parts(row))

    result: dict[str, list[TemplateGroup]] = {}
    for signature_key, template_map in grouped.items():
        label = signature_labels[signature_key]
        body = str(label.get("BODY", "") or "").strip() or "未分类"
        groups: list[TemplateGroup] = []
        for template_key, items in template_map.items():
            groups.append(
                TemplateGroup(
                    signature_key=signature_key,
                    signature_label=label,
                    body=body,
                    template_key=template_key,
                    rows=items,
                    representative_text=str(items[0].get("input", "") or ""),
                )
            )
        result[signature_key] = groups
    return result


def pick_coverage_groups(
    grouped: dict[str, list[TemplateGroup]],
    rng: random.Random,
) -> tuple[list[TemplateGroup], dict[str, list[TemplateGroup]], dict[str, str]]:
    coverage_candidates: list[TemplateGroup] = []
    remaining_by_signature: dict[str, list[TemplateGroup]] = {}
    uncovered_reasons: dict[str, str] = {}

    for signature_key, groups in grouped.items():
        shuffled = list(groups)
        rng.shuffle(shuffled)
        sorted_groups = sorted(shuffled, key=lambda group: (group.size, group.template_key))
        remaining_by_signature[signature_key] = sorted_groups
        if len(sorted_groups) < 2:
            uncovered_reasons[signature_key] = "仅有一个模板组，无法同时保留训练集和验证集"
            continue
        chosen = sorted_groups[0]
        coverage_candidates.append(chosen)
        remaining_by_signature[signature_key] = sorted_groups[1:]

    coverage_candidates.sort(key=lambda group: (group.size, group.body, group.template_key))
    return coverage_candidates, remaining_by_signature, uncovered_reasons


def split_dataset(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped = build_groups(rows)
    rng = random.Random(seed)
    target_val_count = round(len(rows) * val_ratio) if rows else 0

    body_total_counter: Counter[str] = Counter()
    for row in rows:
        label = build_signature_parts(row)
        body = str(label.get("BODY", "") or "").strip() or "未分类"
        body_total_counter[body] += 1

    coverage_groups, remaining_by_signature, uncovered_reasons = pick_coverage_groups(grouped, rng)

    selected_val_groups: list[TemplateGroup] = []
    current_val_count = 0
    current_body_val_counter: Counter[str] = Counter()

    for group in coverage_groups:
        selected_val_groups.append(group)
        current_val_count += group.size
        current_body_val_counter[group.body] += group.size

    extra_candidates: list[TemplateGroup] = []
    for signature_key, groups in remaining_by_signature.items():
        if len(groups) < 1:
            continue
        if len(groups) == 1:
            continue
        extra_candidates.extend(groups[:-1])
        remaining_by_signature[signature_key] = groups[-1:]

    rng.shuffle(extra_candidates)

    body_target_counter = {
        body: round(count * val_ratio)
        for body, count in body_total_counter.items()
    }

    def candidate_sort_key(group: TemplateGroup) -> tuple[int, int, int, str]:
        body_gap = body_target_counter.get(group.body, 0) - current_body_val_counter.get(group.body, 0)
        priority = 0 if body_gap > 0 else 1
        return (priority, -body_gap, group.size, group.template_key)

    while current_val_count < target_val_count and extra_candidates:
        extra_candidates.sort(key=candidate_sort_key)
        chosen = extra_candidates.pop(0)
        selected_val_groups.append(chosen)
        current_val_count += chosen.size
        current_body_val_counter[chosen.body] += chosen.size

    selected_ids = {id(group) for group in selected_val_groups}
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    val_signature_keys: set[str] = set()

    for signature_key, groups in grouped.items():
        for group in groups:
            if id(group) in selected_ids:
                val_rows.extend(group.rows)
                val_signature_keys.add(signature_key)
            else:
                train_rows.extend(group.rows)

    splittable_signature_count = sum(1 for groups in grouped.values() if len(groups) >= 2)
    unsplittable_signature_count = len(grouped) - splittable_signature_count

    body_signature_total: Counter[str] = Counter()
    body_signature_val: Counter[str] = Counter()
    for signature_key, groups in grouped.items():
        body = groups[0].body if groups else "未分类"
        body_signature_total[body] += 1
        if signature_key in val_signature_keys:
            body_signature_val[body] += 1

    uncovered_signatures: list[dict[str, Any]] = []
    for signature_key, groups in grouped.items():
        if signature_key in val_signature_keys:
            continue
        uncovered_signatures.append(
            {
                "signature": groups[0].signature_label if groups else {},
                "sample_count": sum(group.size for group in groups),
                "template_group_count": len(groups),
                "reason": uncovered_reasons.get(signature_key, "未进入验证集"),
                "example_inputs": [group.representative_text for group in groups[:3]],
            }
        )

    uncovered_signatures.sort(
        key=lambda item: (
            item["template_group_count"],
            item["sample_count"],
            json.dumps(item["signature"], ensure_ascii=False, sort_keys=True),
        )
    )

    body_statistics = {}
    for body in sorted(body_total_counter):
        body_statistics[body] = {
            "sample_count": body_total_counter[body],
            "val_sample_count": current_body_val_counter.get(body, 0),
            "train_sample_count": body_total_counter[body] - current_body_val_counter.get(body, 0),
            "signature_count": body_signature_total.get(body, 0),
            "val_signature_count": body_signature_val.get(body, 0),
            "signature_coverage_ratio": round(
                body_signature_val.get(body, 0) / body_signature_total[body],
                6,
            ) if body_signature_total[body] else 0.0,
        }

    report = {
        "input_file": str(input_path),
        "total_sample_count": len(rows),
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "train_ratio": round((len(train_rows) / len(rows)) if rows else 0.0, 6),
        "val_ratio": round((len(val_rows) / len(rows)) if rows else 0.0, 6),
        "requested_val_ratio": val_ratio,
        "target_val_sample_count": target_val_count,
        "seed": seed,
        "signature_count": len(grouped),
        "splittable_signature_count": splittable_signature_count,
        "unsplittable_signature_count": unsplittable_signature_count,
        "val_signature_count": len(val_signature_keys),
        "overall_signature_coverage_ratio": round(
            (len(val_signature_keys) / len(grouped)) if grouped else 0.0,
            6,
        ),
        "splittable_signature_coverage_ratio": round(
            (len(val_signature_keys) / splittable_signature_count) if splittable_signature_count else 0.0,
            6,
        ),
        "body_statistics": body_statistics,
        "uncovered_signature_count": len(uncovered_signatures),
        "uncovered_signatures": uncovered_signatures,
    }

    return train_rows, val_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按种类骨架覆盖切分训练/验证集，优先保证验证集覆盖尽可能多的输出骨架，并避免模板组直接泄漏。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入种类数据集路径")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT, help="训练集输出路径")
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT, help="验证集输出路径")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT, help="报告输出路径")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="目标验证比例，默认 0.05")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_dataset(args.input)
    train_rows, val_rows, report = split_dataset(
        rows,
        input_path=args.input,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    dump_json(args.train_output, train_rows)
    dump_json(args.val_output, val_rows)
    dump_json(args.report_output, report)

    print(f"输入: {args.input}")
    print(f"训练集: {args.train_output} ({len(train_rows)} 条)")
    print(f"验证集: {args.val_output} ({len(val_rows)} 条)")
    print(f"报告: {args.report_output}")
    print(f"总骨架数: {report['signature_count']}")
    print(f"可拆骨架数: {report['splittable_signature_count']}")
    print(f"验证集覆盖骨架数: {report['val_signature_count']}")
    print(f"整体骨架覆盖率: {report['overall_signature_coverage_ratio']:.4f}")
    print(f"可拆骨架覆盖率: {report['splittable_signature_coverage_ratio']:.4f}")
    print(f"验证比例: {report['val_ratio']:.4f} (目标 {args.val_ratio:.4f})")


if __name__ == "__main__":
    main()
