#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RANDOM_SEED = 20250707

STD_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("STD:GBT", re.compile(r"\bGB/T\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?", re.IGNORECASE)),
    ("STD:HGT", re.compile(r"\bHG/T\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?", re.IGNORECASE)),
    ("STD:SHT", re.compile(r"\bSH/T\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?", re.IGNORECASE)),
    ("STD:SYT", re.compile(r"\bSY/T\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?", re.IGNORECASE)),
    ("STD:NBT", re.compile(r"\bNB/T\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?", re.IGNORECASE)),
    ("STD:JIS", re.compile(r"\bJIS\s*[A-Z]?\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:DIN", re.compile(r"\bDIN\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:ASME", re.compile(r"\bASME\s*[A-Z]?\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:ASTM", re.compile(r"\bASTM\s*[A-Z]?\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:API", re.compile(r"\bAPI\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:MSS", re.compile(r"\bMSS\s*[A-Z]?\s*\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:GB_RAW", re.compile(r"\bGB\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
    ("STD:HG_RAW", re.compile(r"\bHG\d+(?:\.\d+)*(?:-\d+)?[A-ZIVX]*", re.IGNORECASE)),
    ("STD:SH_RAW", re.compile(r"\bSH\d+(?:\.\d+)*(?:-\d+)?[A-Z]*", re.IGNORECASE)),
)

SKELETON_STD_PATTERNS: tuple[re.Pattern[str], ...] = tuple(pattern for _, pattern in STD_FAMILY_PATTERNS)

MATERIAL_FORM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("MAT_FORM:TP_PREFIX", re.compile(r"\bTP\s*3\d{2}[A-Z]?\b", re.IGNORECASE)),
    ("MAT_FORM:TP_SUFFIX", re.compile(r"\b3\d{2}[A-Z]?TP\b", re.IGNORECASE)),
    ("MAT_FORM:SF_PREFIX", re.compile(r"\bSF\.?\s*3\d{2}[A-Z]?\b", re.IGNORECASE)),
    ("MAT_FORM:SUS", re.compile(r"\bSUS(?:\.?F)?\.?\s*[A-Z]?\s*3\d{2}[A-Z]{0,2}\b", re.IGNORECASE)),
    ("MAT_FORM:S5", re.compile(r"\bS\d{5}\b", re.IGNORECASE)),
    ("MAT_FORM:HASH", re.compile(r"\b\d{1,2}#\b")),
    ("MAT_FORM:Q_GRADE", re.compile(r"\bQ\d{3,4}[A-Z]?\b", re.IGNORECASE)),
    ("MAT_FORM:CRNI", re.compile(r"\b0?\d{2}CR[0-9A-Z/\-]+\b", re.IGNORECASE)),
    ("MAT_FORM:N_NICKEL", re.compile(r"\bN\d{5}\b", re.IGNORECASE)),
    ("MAT_FORM:P_ALLOY", re.compile(r"\bP\d{2}\b", re.IGNORECASE)),
    ("MAT_FORM:WP_ALLOY", re.compile(r"\bWP\d+\b", re.IGNORECASE)),
    ("MAT_FORM:ASTM_GRADE", re.compile(r"\bA(?:105|106|182|234|312|333|335|358)\b", re.IGNORECASE)),
    ("MAT_FORM:L_GRADE", re.compile(r"\bL\d{3}[A-Z]?\b", re.IGNORECASE)),
    ("MAT_FORM:MONEL_CODE", re.compile(r"\bN0\d{4}\b", re.IGNORECASE)),
    ("MAT_FORM:POLYMER", re.compile(r"\b(?:PTFE|FRP|PVC|PP|EAA|PFA|FEP|PVDF)\b", re.IGNORECASE)),
)

SKELETON_MATERIAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(pattern for _, pattern in MATERIAL_FORM_PATTERNS)

DN_PATTERN = re.compile(r"\bDN\s*\d+(?:[X×*]\d+)*\b", re.IGNORECASE)
INCH_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\"")
NPS_PATTERN = re.compile(r"\bNPS\s*\d+(?:[X×*]\s*NPS?\s*\d+)*\b", re.IGNORECASE)
SCHEDULE_PATTERN = re.compile(r"\b(?:SCH|SCHEDULE)\s*[- ]?\s*(?:\d+(?:\.\d+)?S?|STD|XS|XXS)\b", re.IGNORECASE)
THICKNESS_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:MM)?\s*[+X×*]\s*\d+(?:\.\d+)?\s*(?:MM)?(?:\s*[+X×*]\s*\d+(?:\.\d+)?\s*(?:MM)?)?\b", re.IGNORECASE)
OD_SIZE_PATTERN = re.compile(r"[Φφ]\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)+(?:\(\d+(?:\.\d+)?\))?", re.IGNORECASE)
PLAIN_SIZE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*[X×*]\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)?(?:\(\d+(?:\.\d+)?\))?\b", re.IGNORECASE)
PRESSURE_PATTERN = re.compile(r"\b(?:PN\s*\d+(?:\.\d+)?|CL\s*\d+(?:\.\d+)?|CLASS\s*[A-Z0-9]+|LB\s*\d+|#\s*\d+)\b", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class GroupInfo:
    key: str
    indices: list[int]
    tags: set[str]

    @property
    def size(self) -> int:
        return len(self.indices)


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx + 1} 条不是对象")
        rows.append(item)
    return rows


def parse_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_text(text: str) -> str:
    return (
        str(text or "")
        .replace("，", ",")
        .replace("；", ";")
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace("“", '"')
        .replace("”", '"')
        .replace("×", "X")
        .replace("*", "X")
    )


def build_skeleton_key(text: str) -> str:
    normalized = normalize_text(text).upper()
    for pattern in SKELETON_STD_PATTERNS:
        normalized = pattern.sub(" STD ", normalized)
    for pattern in SKELETON_MATERIAL_PATTERNS:
        normalized = pattern.sub(" MAT ", normalized)
    normalized = DN_PATTERN.sub(" DN ", normalized)
    normalized = NPS_PATTERN.sub(" NPS ", normalized)
    normalized = INCH_PATTERN.sub(" INCH ", normalized)
    normalized = SCHEDULE_PATTERN.sub(" SCH ", normalized)
    normalized = PRESSURE_PATTERN.sub(" PRES ", normalized)
    normalized = OD_SIZE_PATTERN.sub(" SIZE ", normalized)
    normalized = THICKNESS_PATTERN.sub(" THK ", normalized)
    normalized = PLAIN_SIZE_PATTERN.sub(" SIZE ", normalized)
    normalized = PERCENT_PATTERN.sub(" PERCENT ", normalized)
    normalized = NUMBER_PATTERN.sub(" N ", normalized)
    normalized = re.sub(r"[/(),;:\-]+", " ", normalized)
    normalized = MULTI_SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def extract_case_tags(text: str, output_obj: dict[str, Any]) -> set[str]:
    normalized = normalize_text(text)
    upper_text = normalized.upper()
    tags: set[str] = set()

    for tag, pattern in STD_FAMILY_PATTERNS:
        if pattern.search(upper_text):
            tags.add(tag)

    for tag, pattern in MATERIAL_FORM_PATTERNS:
        if pattern.search(upper_text):
            tags.add(tag)

    if re.search(r"\b(?:GB|HG|SH|SY|NB|JIS|DIN|ASTM|ASME)[^ ,;]*\.\d", upper_text):
        tags.add("STD:DECIMAL")
    if re.search(r"\b(?:GB|HG|SH|SY|NB|JIS|DIN|ASTM|ASME)[^,;()]*\([^)]*\)", upper_text):
        tags.add("STD:APPENDIX")
    if "/" in upper_text:
        tags.add("TEXT:SLASH")
    if "+" in upper_text:
        tags.add("TEXT:PLUS")
    if " IN:" in f" {upper_text}" or " OUT:" in f" {upper_text}":
        tags.add("STRUCT:IN_OUT")
    if "LINED" in upper_text or "衬" in normalized:
        tags.add("STRUCT:LINED")
    if "夹套" in normalized or "JACKET" in upper_text:
        tags.add("STRUCT:JACKET")
    if "RT" in upper_text or "射线" in normalized:
        tags.add("PROC:RT")
    if "SMLS" in upper_text or "无缝" in normalized:
        tags.add("PROC:SMLS")
    if "WELDED" in upper_text or "焊接" in normalized or "SAWL" in upper_text or "ERW" in upper_text:
        tags.add("PROC:WELDED")

    material_items = output_obj.get("MATERIAL")
    if isinstance(material_items, list):
        if len(material_items) > 1:
            tags.add("OUT:MATERIAL_MULTI_ITEM")
        for item in material_items:
            if not isinstance(item, dict):
                continue
            value = str(item.get("VALUE") or "").strip()
            if "/" in value:
                tags.add("OUT:MATERIAL_SLASH")
            if "+" in value:
                tags.add("OUT:MATERIAL_PLUS")

    standard_items = output_obj.get("STANDARD")
    if isinstance(standard_items, list):
        if len(standard_items) > 1:
            tags.add("OUT:STANDARD_MULTI_ITEM")

    return tags


def build_group_signature(tags: set[str]) -> str:
    signature_tags = sorted(
        tag
        for tag in tags
        if tag.startswith("MAT_FORM:")
        or tag.startswith("STD:")
        or tag.startswith("STRUCT:")
        or tag in {"TEXT:SLASH", "TEXT:PLUS"}
    )
    return "|".join(signature_tags) if signature_tags else "NO_VARIANT_TAG"


def build_groups(rows: list[dict[str, Any]]) -> dict[str, GroupInfo]:
    groups: dict[str, GroupInfo] = {}
    for idx, row in enumerate(rows):
        text = str(row.get("input") or "").strip()
        output_obj = parse_output(row.get("output"))
        tags = extract_case_tags(text, output_obj)
        skeleton_key = build_skeleton_key(text)
        group_key = f"{skeleton_key} || {build_group_signature(tags)}"
        if group_key not in groups:
            groups[group_key] = GroupInfo(key=group_key, indices=[], tags=set())
        groups[group_key].indices.append(idx)
        groups[group_key].tags.update(tags)
    return groups


def pick_representative_group(
    group_keys: list[str],
    groups: dict[str, GroupInfo],
    selected: set[str],
) -> str | None:
    candidates = [key for key in group_keys if key not in selected]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda key: (
            groups[key].size,
            -len(groups[key].tags),
            key,
        ),
    )


def build_tag_to_groups(groups: dict[str, GroupInfo]) -> dict[str, list[str]]:
    tag_to_groups: dict[str, list[str]] = defaultdict(list)
    for key, group in groups.items():
        for tag in group.tags:
            tag_to_groups[tag].append(key)
    return tag_to_groups


def build_tag_val_caps(tag_to_groups: dict[str, list[str]], val_ratio: float) -> dict[str, int]:
    caps: dict[str, int] = {}
    for tag, group_keys in tag_to_groups.items():
        total_groups = len(group_keys)
        if total_groups <= 1:
            # 只有一个骨架组承载该情况时，默认保留给训练集。
            caps[tag] = 0
            continue
        proportional_cap = int(round(total_groups * val_ratio))
        proportional_cap = max(1, proportional_cap)
        caps[tag] = min(total_groups - 1, proportional_cap)
    return caps


def choose_val_groups(
    groups: dict[str, GroupInfo],
    val_target_rows: int,
    val_ratio: float,
) -> tuple[set[str], set[str], dict[str, list[str]], dict[str, int]]:
    selected: set[str] = set()
    covered_tags: set[str] = set()
    tag_to_groups = build_tag_to_groups(groups)
    tag_val_caps = build_tag_val_caps(tag_to_groups, val_ratio)
    selected_group_counts_by_tag: Counter[str] = Counter()

    def selected_rows() -> int:
        return sum(groups[key].size for key in selected)

    def can_select_group(group_key: str) -> bool:
        group = groups[group_key]
        for tag in group.tags:
            if selected_group_counts_by_tag[tag] + 1 > tag_val_caps[tag]:
                return False
        return True

    def add_group(group_key: str) -> None:
        selected.add(group_key)
        for tag in groups[group_key].tags:
            selected_group_counts_by_tag[tag] += 1
        covered_tags.update(groups[group_key].tags)

    tag_order = sorted(
        tag_to_groups,
        key=lambda tag: (
            len(tag_to_groups[tag]),
            min(groups[key].size for key in tag_to_groups[tag]),
            tag,
        ),
    )

    # 先覆盖稀有情况。
    hard_limit = max(val_target_rows, int(val_target_rows * 1.15))
    for tag in tag_order:
        if any(key in selected for key in tag_to_groups[tag]):
            covered_tags.add(tag)
            continue
        if tag_val_caps[tag] <= 0:
            continue
        candidates = [key for key in tag_to_groups[tag] if can_select_group(key)]
        chosen = pick_representative_group(candidates, groups, selected)
        if chosen is None:
            continue
        if selected and selected_rows() >= hard_limit and len(tag_to_groups[tag]) > 1:
            continue
        add_group(chosen)

    # 再补齐到目标比例，优先新骨架/新情况。
    all_keys = list(groups.keys())
    while selected_rows() < val_target_rows:
        remaining = [key for key in all_keys if key not in selected and can_select_group(key)]
        if not remaining:
            break

        best_key: str | None = None
        best_score: tuple[float, float, float, str] | None = None
        current_rows = selected_rows()
        for key in remaining:
            group = groups[key]
            unseen_tag_count = len(group.tags - covered_tags)
            rarity_score = sum(1.0 / len(tag_to_groups[tag]) for tag in group.tags if tag_to_groups[tag])
            overshoot_penalty = abs((current_rows + group.size) - val_target_rows) / max(val_target_rows, 1)
            score = (
                float(unseen_tag_count),
                rarity_score,
                -overshoot_penalty,
                key,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            break
        add_group(best_key)

    return selected, covered_tags, tag_to_groups, tag_val_caps


def resolve_output_paths(output_path: Path) -> tuple[Path, Path, Path]:
    if output_path.suffix.lower() == ".json":
        train_path = output_path.with_name(f"{output_path.stem}_train.json")
        val_path = output_path.with_name(f"{output_path.stem}_val.json")
        report_path = output_path.with_name(f"{output_path.stem}_split_report.json")
        return train_path, val_path, report_path

    output_path.mkdir(parents=True, exist_ok=True)
    return (
        output_path / "train.json",
        output_path / "val.json",
        output_path / "split_report.json",
    )


def build_report(
    rows: list[dict[str, Any]],
    groups: dict[str, GroupInfo],
    selected_val_groups: set[str],
    covered_tags: set[str],
    requested_ratio: float,
    tag_to_groups: dict[str, list[str]],
    tag_val_caps: dict[str, int],
) -> dict[str, Any]:
    val_indices: set[int] = set()
    for key in selected_val_groups:
        val_indices.update(groups[key].indices)

    total_rows = len(rows)
    val_rows = len(val_indices)
    train_rows = total_rows - val_rows

    tag_total = Counter()
    tag_train = Counter()
    tag_val = Counter()
    for key, group in groups.items():
        target_counter = tag_val if key in selected_val_groups else tag_train
        for tag in group.tags:
            tag_total[tag] += group.size
            target_counter[tag] += group.size

    missing_in_val = sorted(tag for tag in tag_total if tag_val[tag] == 0)
    missing_in_train = sorted(tag for tag in tag_total if tag_train[tag] == 0)
    single_group_reserved_tags = sorted(
        tag for tag, group_keys in tag_to_groups.items()
        if len(group_keys) == 1 and tag_val_caps.get(tag, 0) == 0
    )
    rare_tags = [
        {
            "tag": tag,
            "train_rows": tag_train[tag],
            "val_rows": tag_val[tag],
            "total_rows": tag_total[tag],
            "group_count": len(tag_to_groups.get(tag, [])),
            "val_group_cap": tag_val_caps.get(tag, 0),
        }
        for tag in sorted(tag_total, key=lambda item: (tag_total[item], item))[:50]
    ]

    return {
        "total_rows": total_rows,
        "total_groups": len(groups),
        "requested_val_ratio": requested_ratio,
        "actual_val_ratio": round(val_rows / total_rows, 6) if total_rows else 0.0,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_groups": len(groups) - len(selected_val_groups),
        "val_groups": len(selected_val_groups),
        "covered_case_tag_count": len(covered_tags),
        "missing_case_tags_in_val": missing_in_val,
        "missing_case_tags_in_train": missing_in_train,
        "single_group_case_tags_reserved_for_train": single_group_reserved_tags,
        "rare_case_tag_summary": rare_tags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按骨架 + 表达形态切分 LlamaFactory 训练/验证集，避免模板泄漏。"
    )
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出目录，或输出前缀 JSON 路径")
    parser.add_argument("--ratio", required=True, type=float, help="验证集比例，例如 0.1")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    ratio = float(args.ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError("--ratio 必须在 0 到 1 之间")

    rows = load_rows(input_path)
    groups = build_groups(rows)
    val_target_rows = max(1, int(round(len(rows) * ratio)))
    selected_val_groups, covered_tags, tag_to_groups, tag_val_caps = choose_val_groups(
        groups,
        val_target_rows,
        ratio,
    )

    val_indices: set[int] = set()
    for key in selected_val_groups:
        val_indices.update(groups[key].indices)

    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]

    train_path, val_path, report_path = resolve_output_paths(output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(
        rows,
        groups,
        selected_val_groups,
        covered_tags,
        ratio,
        tag_to_groups,
        tag_val_caps,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "train_output": str(train_path),
                "val_output": str(val_path),
                "report_output": str(report_path),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "requested_ratio": ratio,
                "actual_ratio": round(len(val_rows) / len(rows), 6) if rows else 0.0,
                "group_count": len(groups),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
