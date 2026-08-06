#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

"""按法兰、直管、管件分别构建训练集和验证集的工具。
python apps/trainer/qwen3_fte/src/build_type_train_val_datasets.py \
    --flange apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/法兰.json \
    --pipe apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/直管.json \
    --fitting apps/trainer/qwen3_fte/output/按8类拆分数据集/种类/管件.json \
    --raw-output-dir apps/trainer/qwen3_fte/output/按8类拆分数据集/type_raw \
    --llamafactory-output-dir apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/type_llamafactory \
    --val-ratio 0.1 \
    --test-ratio 0 \
    --seed 20260714
"""

PROJECT_ROOT = Path(__file__).resolve().parents[4]
QWEN_ROOT = PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte"
DEFAULT_PROMPT = QWEN_ROOT / "prompt" / "type_extraction_sft_instruction_v1.txt"
DEFAULT_SKELETON_DIR = QWEN_ROOT / "skeletons"
DEFAULT_SEED = 20260714
CATEGORIES = ("法兰", "直管", "管件")

STD_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:GB|HG|SH|NB|SY|JB|JIS|DIN|EN|ASTM|ASME|API|MSS)"
    r"\s*(?:/\s*T)?\s*[A-Z]*\s*\d+(?:\.\d+)*(?:-\d+)?(?:\([A-Z0-9IVX]+\))?",
    re.IGNORECASE,
)
DN_PATTERN = re.compile(r"\bDN\s*\d+(?:\s*[X×*]\s*(?:DN\s*)?\d+)*", re.IGNORECASE)
NPS_PATTERN = re.compile(r"\bNPS\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*(?:NPS\s*)?\d+(?:\.\d+)?)*", re.IGNORECASE)
INCH_PATTERN = re.compile(r"(?<![\d.])(?:\d+\s+)?\d+(?:\.\d+)?(?:/\d+)?\s*[\"″”]")
OD_PATTERN = re.compile(
    r"[Φφ]\s*\d+(?:\.\d+)?(?:\s*[X×*]\s*\d+(?:\.\d+)?)+(?:\s*MM)?",
    re.IGNORECASE,
)
SCHEDULE_PATTERN = re.compile(
    r"\b(?:SCH(?:EDULE)?\s*[-.]?\s*|S-)(?:\d+(?:\.\d+)?S?|STD|XS|XXS)"
    r"(?:\s*[X×*]\s*(?:(?:SCH(?:EDULE)?\s*[-.]?\s*|S-)?(?:\d+(?:\.\d+)?S?|STD|XS|XXS)))*",
    re.IGNORECASE,
)
PRESSURE_PATTERN = re.compile(
    r"\b(?:PN\s*\d+(?:\.\d+)?|CL(?:ASS)?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:LB|MPA))\b",
    re.IGNORECASE,
)
ANGLE_TEXT_PATTERN = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*(?:°|度|DEG(?:REE)?S?\b)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![A-Z])\d+(?:\.\d+)?(?![A-Z])", re.IGNORECASE)
MULTI_SPACE_PATTERN = re.compile(r"\s+")

FORM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FORM:FLG", re.compile(r"(?<![A-Z])FLG(?![A-Z])", re.IGNORECASE)),
    ("FORM:FLANGE", re.compile(r"\bFLANGE(?:D)?\b", re.IGNORECASE)),
    ("FORM:法兰", re.compile(r"法兰")),
    ("FORM:WN", re.compile(r"(?<![A-Z])WN(?![A-Z])|WELD\s*NECK|带颈对焊", re.IGNORECASE)),
    ("FORM:SO", re.compile(r"(?<![A-Z])S[O0](?![A-Z])|SLIP[- ]?[O0]N|带颈平焊", re.IGNORECASE)),
    ("FORM:SW", re.compile(r"(?<![A-Z])SW(?![A-Z])|SOCKET[- ]?WELD|承插焊", re.IGNORECASE)),
    ("FORM:BLIND", re.compile(r"\bBLIND\b|盲法兰|法兰盖|盲板", re.IGNORECASE)),
    ("FORM:LAP_JOINT", re.compile(r"LAP[- ]?JOINT|(?<![A-Z])LJ(?![A-Z])|松套|活套", re.IGNORECASE)),
    ("FORM:ELBOW", re.compile(r"\bELB(?:OW)?\b|弯头|弯管", re.IGNORECASE)),
    ("FORM:TEE", re.compile(r"\bTEE\b|三通", re.IGNORECASE)),
    ("FORM:REDUCER", re.compile(r"\bREDUCER\b|异径|大小头", re.IGNORECASE)),
    ("FORM:OLET", re.compile(r"\b[A-Z]*OLET\b|支管台|支管座", re.IGNORECASE)),
    ("FORM:PIPE", re.compile(r"\bPIPE\b|\bTUBE\b|管子|钢管|管道", re.IGNORECASE)),
    ("FORM:SMLS", re.compile(r"\bSMLS\b|\bSEAMLESS\b|无缝", re.IGNORECASE)),
    ("FORM:WELDED", re.compile(r"\bWELDED\b|\bEFW\b|\bERW\b|焊接|有缝", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Record:
    input: str
    output: dict[str, Any]
    category: str
    source: str
    source_index: int
    source_label: str = ""

    @property
    def is_augmented(self) -> bool:
        return self.source_label.strip().startswith("数据增强")


def load_json_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() == ".jsonl":
        data: list[Any] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} 顶层必须是数组")

    rows: list[dict[str, Any]] = []
    nested_arrays = 0

    def flatten(items: Iterable[Any]) -> None:
        nonlocal nested_arrays
        for item in items:
            if isinstance(item, list):
                nested_arrays += 1
                flatten(item)
            elif isinstance(item, dict):
                rows.append(item)
            else:
                raise ValueError(f"{path} 包含非对象元素: {type(item).__name__}")

    flatten(data)
    return rows, nested_arrays


def parse_output(value: Any, source: str, index: int) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} 第 {index + 1} 条 output JSON 解析失败: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} 第 {index + 1} 条 output 必须是对象或 JSON 字符串")
    return value


def load_skeletons(skeleton_dir: Path) -> dict[str, dict[str, Any]]:
    skeletons: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        path = skeleton_dir / f"{category}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"骨架必须是对象: {path}")
        skeletons[category] = value
    return skeletons


def copy_by_skeleton(template: Any, value: Any) -> Any:
    """只保留骨架字段，按骨架顺序递归补齐缺失值。"""
    if isinstance(template, dict):
        source = value if isinstance(value, dict) else {}
        return {key: copy_by_skeleton(default, source.get(key)) for key, default in template.items()}
    if isinstance(template, list):
        if value is None:
            return []
        if isinstance(value, list):
            return copy.deepcopy(value)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
    if isinstance(template, str):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return ""
        return str(value).strip()
    return copy.deepcopy(value if value is not None else template)


def normalize_output(output: dict[str, Any], category: str, skeletons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = copy_by_skeleton(skeletons[category], output)
    normalized["CATEGORY"] = category
    return normalized


def infer_category(output: dict[str, Any]) -> str:
    category = str(output.get("CATEGORY") or "").strip()
    if category in CATEGORIES:
        return category
    type_obj = output.get("TYPE") if isinstance(output.get("TYPE"), dict) else {}
    if "SEAL" in type_obj:
        return "法兰"
    if "GEOMETRY" in type_obj or "CONN" in type_obj:
        return "管件"
    if "FLANGE_STYLE" in type_obj or "MANU" in type_obj:
        return "直管"
    body = str(type_obj.get("BODY") or "")
    if "法兰" in body:
        return "法兰"
    if body == "直管":
        return "直管"
    raise ValueError(f"无法从验证样本 output 推断 CATEGORY: {output}")


def description_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return MULTI_SPACE_PATTERN.sub(" ", normalized).strip().casefold()


def output_key(output: dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_source_records(
    source_paths: dict[str, Path],
    skeletons: dict[str, dict[str, Any]],
) -> tuple[list[Record], dict[str, Any]]:
    records: list[Record] = []
    report: dict[str, Any] = {"files": {}, "category_overrides": []}
    for category in CATEGORIES:
        path = source_paths[category]
        rows, nested_arrays = load_json_rows(path)
        report["files"][category] = {
            "path": str(path),
            "loaded_rows": len(rows),
            "nested_arrays_flattened": nested_arrays,
            "category_override_count": 0,
            "augmented_rows": 0,
            "source_label_distribution": {},
        }
        source_labels: Counter[str] = Counter()
        for index, row in enumerate(rows):
            text = str(row.get("input") or "").strip()
            if not text:
                raise ValueError(f"{path} 第 {index + 1} 条缺少 input")
            output = parse_output(row.get("output"), str(path), index)
            source_category = str(output.get("CATEGORY") or "").strip()
            if source_category != category:
                report["files"][category]["category_override_count"] += 1
                if len(report["category_overrides"]) < 50:
                    report["category_overrides"].append({
                        "source": str(path),
                        "source_index": index,
                        "input": text,
                        "original_category": source_category,
                        "target_category": category,
                    })
            source_label = str(row.get("来源") or "").strip()
            source_labels[source_label or "<EMPTY>"] += 1
            if source_label.startswith("数据增强"):
                report["files"][category]["augmented_rows"] += 1
            records.append(Record(
                input=text,
                output=normalize_output(output, category, skeletons),
                category=category,
                source=str(path),
                source_index=index,
                source_label=source_label,
            ))
        report["files"][category]["source_label_distribution"] = dict(source_labels.most_common())
    return records, report


def load_validation_records(
    paths: list[Path],
    skeletons: dict[str, dict[str, Any]],
) -> tuple[list[Record], dict[str, Any]]:
    records: list[Record] = []
    files: list[dict[str, Any]] = []
    for path in paths:
        rows, nested_arrays = load_json_rows(path)
        files.append({"path": str(path), "loaded_rows": len(rows), "nested_arrays_flattened": nested_arrays})
        for index, row in enumerate(rows):
            text = str(row.get("input") or "").strip()
            if not text:
                raise ValueError(f"{path} 第 {index + 1} 条缺少 input")
            output = parse_output(row.get("output"), str(path), index)
            category = infer_category(output)
            records.append(Record(
                input=text,
                output=normalize_output(output, category, skeletons),
                category=category,
                source=str(path),
                source_index=index,
                source_label=str(row.get("来源") or "").strip(),
            ))
    return records, {"files": files}


def deduplicate_records(records: list[Record], *, reject_conflicts: bool) -> tuple[list[Record], dict[str, Any]]:
    by_description: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_description[description_key(record.input)].append(record)

    kept: list[Record] = []
    exact_duplicate_rows_removed = 0
    conflict_rows_removed = 0
    conflict_groups: list[dict[str, Any]] = []
    for grouped in by_description.values():
        labels = {output_key(record.output) for record in grouped}
        if len(labels) > 1:
            detail = {
                "input": grouped[0].input,
                "occurrences": len(grouped),
                "labels": [record.output for record in grouped],
                "sources": [f"{record.source}#{record.source_index}" for record in grouped],
            }
            if reject_conflicts:
                raise ValueError(f"指定验证集存在同描述冲突标签: {json.dumps(detail, ensure_ascii=False)}")
            conflict_groups.append(detail)
            conflict_rows_removed += len(grouped)
            continue
        # 同描述同标签时优先保留真实样本，避免增强标识覆盖原始来源。
        kept.append(next((record for record in grouped if not record.is_augmented), grouped[0]))
        exact_duplicate_rows_removed += len(grouped) - 1

    return kept, {
        "input_rows": len(records),
        "output_rows": len(kept),
        "rows_removed_total": len(records) - len(kept),
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "conflict_rows_removed": conflict_rows_removed,
        "conflict_group_count": len(conflict_groups),
        "conflict_groups": conflict_groups[:100],
    }


def surface_skeleton(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).upper()
    value = STD_PATTERN.sub(" STD ", value)
    value = DN_PATTERN.sub(" SIZE ", value)
    value = NPS_PATTERN.sub(" SIZE ", value)
    value = INCH_PATTERN.sub(" SIZE ", value)
    value = OD_PATTERN.sub(" SIZE ", value)
    value = SCHEDULE_PATTERN.sub(" THK ", value)
    value = PRESSURE_PATTERN.sub(" PRESS ", value)
    value = ANGLE_TEXT_PATTERN.sub(" ANGLE ", value)
    value = NUMBER_PATTERN.sub(" N ", value)
    value = re.sub(r"[,;:|/()\[\]{}]+", " ", value)
    return MULTI_SPACE_PATTERN.sub(" ", value).strip()


def expression_tags(record: Record) -> set[str]:
    tags = {f"CATEGORY:{record.category}"}
    type_obj = record.output.get("TYPE") if isinstance(record.output.get("TYPE"), dict) else {}
    body = str(type_obj.get("BODY") or "").strip()
    tags.add(f"BODY:{body or '<EMPTY>'}")

    geometry = type_obj.get("GEOMETRY") if isinstance(type_obj.get("GEOMETRY"), dict) else {}
    angle = str(geometry.get("ANGLE") or "").strip()
    radius = str(geometry.get("RADIUS") or "").strip()
    if angle:
        tags.add("ANGLE:PRESENT")
        if "." in angle:
            tags.add("ANGLE:DECIMAL")
        if angle not in {"45", "90"}:
            tags.add("ANGLE:UNCOMMON")
    if radius:
        tags.add(f"RADIUS:{radius}")

    flange_style = str(type_obj.get("FLANGE_STYLE") or "").strip()
    if flange_style:
        tags.add(f"FLANGE_STYLE:{flange_style}")
    for field in ("MANU", "CONN", "SEAL"):
        values = type_obj.get(field) if isinstance(type_obj.get(field), list) else []
        for value in values:
            if str(value).strip():
                tags.add(f"{field}:{str(value).strip()}")

    source = record.input
    for tag, pattern in FORM_PATTERNS:
        if pattern.search(source):
            tags.add(tag)
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", source))
    has_en = bool(re.search(r"[A-Za-z]", source))
    tags.add("LANG:MIXED" if has_zh and has_en else "LANG:ZH" if has_zh else "LANG:EN")
    return tags


def normalized_label_tags(record: Record) -> set[str]:
    """返回必须由训练集覆盖的归一化标签，不混入原文表达特征。"""
    type_obj = record.output.get("TYPE") if isinstance(record.output.get("TYPE"), dict) else {}
    tags = {
        f"CATEGORY:{record.category}",
        f"BODY:{str(type_obj.get('BODY') or '').strip() or '<EMPTY>'}",
        f"OUTPUT:{output_key(record.output)}",
    }
    geometry = type_obj.get("GEOMETRY") if isinstance(type_obj.get("GEOMETRY"), dict) else {}
    for field in ("ANGLE", "RADIUS"):
        value = str(geometry.get(field) or "").strip()
        if value:
            tags.add(f"GEOMETRY.{field}:{value}")
    flange_style = str(type_obj.get("FLANGE_STYLE") or "").strip()
    if flange_style:
        tags.add(f"FLANGE_STYLE:{flange_style}")
    for field in ("MANU", "CONN", "SEAL"):
        values = type_obj.get(field) if isinstance(type_obj.get(field), list) else []
        for value in values:
            normalized = str(value).strip()
            if normalized:
                tags.add(f"{field}:{normalized}")
    return tags


def validation_balance_tags(record: Record) -> set[str]:
    """用于验证集分层的标签，兼顾目标标签和原文表达形式。"""
    tags = {
        tag for tag in normalized_label_tags(record)
        if not tag.startswith("CATEGORY:")
    }
    tags.update(
        tag for tag in expression_tags(record)
        if tag.startswith(("FORM:", "LANG:"))
    )
    return tags


def body_value(record: Record) -> str:
    type_obj = record.output.get("TYPE") if isinstance(record.output.get("TYPE"), dict) else {}
    return str(type_obj.get("BODY") or "").strip() or "<EMPTY>"


def allocate_body_quotas(records: list[Record], target: int, ratio: float) -> dict[str, int]:
    """按 BODY 分配精确名额；稀有 BODY 优先在训练、验证两侧各保留样本。"""
    body_counts = Counter(body_value(record) for record in records)
    quotas = {body: 0 for body in body_counts}
    eligible = [body for body, count in body_counts.items() if count >= 2]

    if len(eligible) <= target:
        for body in eligible:
            quotas[body] = 1
    else:
        # 当 BODY 数比验证名额还多时，优先覆盖样本量较大的 BODY。
        ranked = sorted(eligible, key=lambda body: (-body_counts[body], body))
        for body in ranked[:target]:
            quotas[body] = 1

    while sum(quotas.values()) < target:
        candidates = [body for body, count in body_counts.items() if quotas[body] < count - 1]
        if not candidates:
            break
        # 选择距离比例目标最远的 BODY，确保大类不会被稀有标签挤占过多名额。
        chosen = max(
            candidates,
            key=lambda body: (
                body_counts[body] * ratio - quotas[body],
                body_counts[body],
                body,
            ),
        )
        quotas[chosen] += 1

    return quotas


def select_body_validation_indices(
    records: list[Record],
    indices: list[int],
    quota: int,
    seed: int,
) -> set[int]:
    """在一个 BODY 内按文本骨架和表达标签进行比例均衡抽样。"""
    if quota <= 0:
        return set()
    if quota >= len(indices):
        raise ValueError("验证配额必须至少给训练集保留一条 BODY 样本")

    tag_sets = {
        index: {
            tag for tag in expression_tags(records[index])
            if not tag.startswith(("CATEGORY:", "BODY:"))
        }
        for index in indices
    }
    skeletons = {index: surface_skeleton(records[index].input) for index in indices}
    tag_totals: Counter[str] = Counter()
    skeleton_totals: Counter[str] = Counter(skeletons.values())
    for tags in tag_sets.values():
        tag_totals.update(tags)

    tag_selected: Counter[str] = Counter()
    skeleton_selected: Counter[str] = Counter()
    selected: set[int] = set()
    rng = random.Random(seed)
    tie_breakers = {index: rng.random() for index in indices}
    selection_ratio = quota / len(indices)

    while len(selected) < quota:
        candidates = [index for index in indices if index not in selected]

        def score(index: int) -> tuple[float, float, float]:
            tags = tag_sets[index]
            tag_balance = sum(
                (tag_totals[tag] * selection_ratio - tag_selected[tag]) / tag_totals[tag]
                for tag in tags
            )
            rare_coverage = sum(
                1.0 / tag_totals[tag]
                for tag in tags
                if tag_selected[tag] == 0
            )
            skeleton = skeletons[index]
            skeleton_balance = (
                skeleton_totals[skeleton] * selection_ratio - skeleton_selected[skeleton]
            ) / skeleton_totals[skeleton]
            return (
                tag_balance + skeleton_balance + 0.5 * rare_coverage,
                -float(skeleton_selected[skeleton]),
                tie_breakers[index],
            )

        chosen = max(candidates, key=score)
        selected.add(chosen)
        tag_selected.update(tag_sets[chosen])
        skeleton_selected[skeletons[chosen]] += 1

    return selected


def choose_category_validation(
    records: list[Record],
    ratio: float,
    seed: int,
) -> tuple[set[int], dict[str, Any]]:
    if len(records) < 2:
        return set(), {"body_strata": len(records), "target_rows": 0, "selected_rows": 0}

    target = max(1, min(len(records) - 1, int(round(len(records) * ratio))))
    body_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        body_indices[body_value(record)].append(index)

    quotas = allocate_body_quotas(records, target, ratio)
    selected: set[int] = set()
    for offset, body in enumerate(sorted(body_indices)):
        selected.update(select_body_validation_indices(
            records,
            body_indices[body],
            quotas[body],
            seed + offset,
        ))

    if len(selected) != target:
        raise RuntimeError(f"BODY 分层切分数量异常: 目标 {target}, 实际 {len(selected)}")

    covered_tags: set[str] = set()
    for index in selected:
        covered_tags.update(expression_tags(records[index]))
    return selected, {
        "body_strata": len(body_indices),
        "target_rows": target,
        "selected_rows": len(selected),
        "body_quotas": [
            {
                "body": body,
                "total": len(body_indices[body]),
                "val": quotas[body],
                "ratio": round(quotas[body] / len(body_indices[body]), 6),
            }
            for body in sorted(body_indices)
        ],
        "covered_tags": sorted(covered_tags),
    }


def allocate_joint_body_quotas(
    records: list[Record],
    val_target: int,
    test_target: int,
    val_ratio: float,
    test_ratio: float,
    min_holdout_body_rows: int = 5,
) -> dict[str, dict[str, int]]:
    """联合分配验证/测试名额，避免先抽验证集后耗尽稀有 BODY。"""
    body_counts = Counter(body_value(record) for record in records)
    quotas = {body: {"val": 0, "test": 0} for body in body_counts}
    eligible = [body for body, count in body_counts.items() if count >= min_holdout_body_rows]

    def seed_one(split: str, target: int) -> None:
        if target <= 0:
            return
        ranked = sorted(eligible, key=lambda body: (-body_counts[body], body))
        for body in ranked[:target]:
            if sum(quotas[body].values()) < body_counts[body] - 1:
                quotas[body][split] = 1

    seed_one("val", val_target)
    seed_one("test", test_target)

    def current(split: str) -> int:
        return sum(value[split] for value in quotas.values())

    def add_one(split: str, target: int, ratio: float) -> bool:
        if current(split) >= target:
            return False
        candidates = [
            body for body in eligible
            if sum(quotas[body].values()) < body_counts[body] - 1
        ]
        if not candidates:
            raise ValueError(
                f"无法完成{split}集配额: 目标={target}, 已分配={current(split)}"
            )
        chosen = max(
            candidates,
            key=lambda body: (
                body_counts[body] * ratio - quotas[body][split],
                body_counts[body],
                body,
            ),
        )
        quotas[chosen][split] += 1
        return True

    while current("val") < val_target or current("test") < test_target:
        add_one("val", val_target, val_ratio)
        add_one("test", test_target, test_ratio)
    return quotas


def choose_category_holdouts(
    records: list[Record],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[int], set[int], dict[str, Any]]:
    real_indices = [index for index, record in enumerate(records) if not record.is_augmented]
    if len(real_indices) < 2:
        return set(), set(), {
            "body_strata": len({body_value(record) for record in records}),
            "real_rows": len(real_indices),
            "val_target_rows": 0,
            "test_target_rows": 0,
        }

    val_target = max(1, int(round(len(real_indices) * val_ratio))) if val_ratio else 0
    test_target = max(1, int(round(len(real_indices) * test_ratio))) if test_ratio else 0
    if val_target + test_target >= len(real_indices):
        raise ValueError("验证集与测试集数量之和必须小于真实样本数")

    # 骨架只由实际描述计算。归一化标签不参与分组，避免同类描述因标签差异被拆散。
    skeleton_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        skeleton_indices[surface_skeleton(record.input)].append(index)

    group_tags: dict[str, set[str]] = {}
    group_expression_tags: dict[str, set[str]] = {}
    group_balance_rows: dict[str, Counter[str]] = {}
    group_real_rows: dict[str, int] = {}
    forced_train_groups: set[str] = set()
    label_group_counts: Counter[str] = Counter()
    balance_row_counts: Counter[str] = Counter()
    for skeleton, indices in skeleton_indices.items():
        labels: set[str] = set()
        expressions: set[str] = set()
        balance_rows: Counter[str] = Counter()
        real_rows = 0
        for index in indices:
            record = records[index]
            record_labels = normalized_label_tags(record)
            labels.update(record_labels)
            expressions.update(expression_tags(record))
            if record.is_augmented:
                forced_train_groups.add(skeleton)
            else:
                real_rows += 1
                record_balance_tags = validation_balance_tags(record)
                balance_rows.update(sorted(record_balance_tags))
                balance_row_counts.update(sorted(record_balance_tags))
        group_tags[skeleton] = labels
        group_expression_tags[skeleton] = expressions
        group_balance_rows[skeleton] = balance_rows
        group_real_rows[skeleton] = real_rows
        label_group_counts.update(labels)

    # 记录每个标签尚留在训练侧的骨架数。任何标签至少保留一个训练骨架。
    remaining_train_label_groups = label_group_counts.copy()
    selected_groups: dict[str, set[str]] = {"val": set(), "test": set()}
    selected_rows: dict[str, int] = {"val": 0, "test": 0}
    selected_balance_rows: dict[str, Counter[str]] = {
        "val": Counter(),
        "test": Counter(),
    }
    selected_expression_tags: dict[str, set[str]] = {"val": set(), "test": set()}
    rng = random.Random(seed)
    tie_breakers = {skeleton: rng.random() for skeleton in skeleton_indices}

    def can_hold_out(skeleton: str, max_group_rows: int) -> bool:
        if skeleton in forced_train_groups:
            return False
        if not 0 < group_real_rows[skeleton] <= max_group_rows:
            return False
        if any(skeleton in groups for groups in selected_groups.values()):
            return False
        return all(remaining_train_label_groups[tag] > 1 for tag in group_tags[skeleton])

    def select_groups(split: str, target: int, ratio: float) -> None:
        if target <= 0:
            return
        # 单个骨架最多占验证集约 10%，避免少数批量模板支配验证结果。
        max_group_rows = max(8, int(round(target * 0.1)))

        candidate_group_counts: Counter[str] = Counter()
        for skeleton in skeleton_indices:
            if skeleton in forced_train_groups or not 0 < group_real_rows[skeleton] <= max_group_rows:
                continue
            candidate_group_counts.update(sorted(group_balance_rows[skeleton]))
        mandatory_prefixes = (
            "OUTPUT:", "BODY:", "MANU:", "CONN:", "SEAL:",
            "FLANGE_STYLE:", "GEOMETRY.",
        )
        common_tags = {
            tag for tag, total in balance_row_counts.items()
            if tag.startswith(mandatory_prefixes)
            and total >= 10
            and candidate_group_counts[tag] >= 2
        }

        def add_group(skeleton: str) -> None:
            selected_groups[split].add(skeleton)
            selected_rows[split] += group_real_rows[skeleton]
            selected_expression_tags[split].update(group_expression_tags[skeleton])
            selected_balance_rows[split].update(group_balance_rows[skeleton])
            for tag in group_tags[skeleton]:
                remaining_train_label_groups[tag] -= 1

        # 先覆盖样本充足的业务标签；优先选择能覆盖更多标签的较小骨架。
        while selected_rows[split] < target:
            covered_tags = set(selected_balance_rows[split])
            uncovered = common_tags - covered_tags
            if not uncovered:
                break
            remaining_rows = target - selected_rows[split]
            candidates = [
                skeleton for skeleton in skeleton_indices
                if can_hold_out(skeleton, max_group_rows)
                and group_real_rows[skeleton] <= remaining_rows
                and uncovered.intersection(group_balance_rows[skeleton])
            ]
            if not candidates:
                break

            def coverage_score(skeleton: str) -> tuple[float, float, float, float]:
                size = group_real_rows[skeleton]
                newly_covered = uncovered.intersection(group_balance_rows[skeleton])
                coverage_weight = sum(
                    1.0 / max(candidate_group_counts[tag], 1) ** 0.5
                    for tag in sorted(newly_covered)
                )
                overfill = 0.0
                for tag, count in sorted(group_balance_rows[skeleton].items()):
                    if tag.startswith("OUTPUT:") and balance_row_counts[tag] < 20:
                        continue
                    desired = balance_row_counts[tag] * ratio
                    after = selected_balance_rows[split][tag] + count
                    weight = 0.5 if tag.startswith("OUTPUT:") else 1.0
                    overfill += weight * max(0.0, after - desired) / max(desired, 1.0)
                return (
                    coverage_weight / max(size, 1) ** 0.5,
                    -overfill / max(len(group_balance_rows[skeleton]), 1),
                    -float(size),
                    tie_breakers[skeleton],
                )

            add_group(max(candidates, key=coverage_score))

        # 再按各标签距离目标比例的改善程度补足行数。
        while selected_rows[split] < target:
            remaining_rows = target - selected_rows[split]
            candidates = [
                skeleton for skeleton in skeleton_indices
                if can_hold_out(skeleton, max_group_rows)
                and group_real_rows[skeleton] <= remaining_rows
            ]
            if not candidates:
                break

            def score(skeleton: str) -> tuple[float, float, float, float]:
                size = group_real_rows[skeleton]
                distribution_gain = 0.0
                for tag, count in sorted(group_balance_rows[skeleton].items()):
                    if tag.startswith("OUTPUT:") and balance_row_counts[tag] < 20:
                        continue
                    desired = balance_row_counts[tag] * ratio
                    before = selected_balance_rows[split][tag]
                    after = before + count
                    weight = min(1.0, balance_row_counts[tag] / 20.0)
                    if tag.startswith("OUTPUT:"):
                        weight *= 0.5
                    distribution_gain += weight * (
                        abs(before - desired) - abs(after - desired)
                    ) / max(desired, 1.0)
                new_expression_count = sum(
                    1 for tag in group_expression_tags[skeleton]
                    if tag not in selected_expression_tags[split]
                )
                return (
                    distribution_gain / max(size, 1),
                    float(new_expression_count) / max(size, 1),
                    -float(size),
                    tie_breakers[skeleton],
                )

            add_group(max(candidates, key=score))

        selected_common_tags = common_tags.intersection(selected_balance_rows[split])
        split_selection_detail[split] = {
            "max_group_rows": max_group_rows,
            "common_tags_targeted": sorted(common_tags),
            "common_tags_missing": sorted(common_tags - selected_common_tags),
        }

    split_selection_detail: dict[str, dict[str, Any]] = {"val": {}, "test": {}}
    select_groups("val", val_target, val_ratio)
    select_groups("test", test_target, test_ratio)

    val_selected = {
        index
        for skeleton in selected_groups["val"]
        for index in skeleton_indices[skeleton]
        if not records[index].is_augmented
    }
    test_selected = {
        index
        for skeleton in selected_groups["test"]
        for index in skeleton_indices[skeleton]
        if not records[index].is_augmented
    }
    train_indices = set(range(len(records))) - val_selected - test_selected
    train_label_tags = set().union(
        *(normalized_label_tags(records[index]) for index in train_indices)
    ) if train_indices else set()
    holdout_label_tags = set().union(
        *(normalized_label_tags(records[index]) for index in val_selected | test_selected)
    ) if val_selected or test_selected else set()
    missing_train_labels = sorted(holdout_label_tags - train_label_tags)
    if missing_train_labels:
        raise RuntimeError(f"训练集缺失 {len(missing_train_labels)} 个验证标签")

    body_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if not record.is_augmented:
            body_indices[body_value(record)].append(index)
    covered = {"val": set(), "test": set()}
    for index in val_selected:
        covered["val"].update(expression_tags(records[index]))
    for index in test_selected:
        covered["test"].update(expression_tags(records[index]))
    return val_selected, test_selected, {
        "body_strata": len(body_indices),
        "real_rows": len(real_indices),
        "surface_skeleton_groups": len(skeleton_indices),
        "forced_train_skeleton_groups": len(forced_train_groups),
        "val_target_rows": val_target,
        "test_target_rows": test_target,
        "val_selected_rows": len(val_selected),
        "test_selected_rows": len(test_selected),
        "val_selected_skeleton_groups": len(selected_groups["val"]),
        "test_selected_skeleton_groups": len(selected_groups["test"]),
        "val_target_delta": len(val_selected) - val_target,
        "test_target_delta": len(test_selected) - test_target,
        "selection_constraints": split_selection_detail,
        "body_distribution": [
            {
                "body": body,
                "total": len(body_indices[body]),
                "train": sum(index in train_indices for index in body_indices[body]),
                "val": sum(index in val_selected for index in body_indices[body]),
                "test": sum(index in test_selected for index in body_indices[body]),
            }
            for body in sorted(body_indices)
        ],
        "covered_tags": {
            "val": sorted(covered["val"]),
            "test": sorted(covered["test"]),
        },
        "bodies_below_holdout_minimum": sorted(
            body for body, indices in body_indices.items() if len(indices) < 5
        ),
        "missing_normalized_labels_in_train": missing_train_labels,
    }


def split_automatically(
    records: list[Record],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Record], list[Record], list[Record], dict[str, Any]]:
    val_global_indices: set[int] = set()
    test_global_indices: set[int] = set()
    split_detail: dict[str, Any] = {}
    for offset, category in enumerate(CATEGORIES):
        category_pairs = [
            (index, record) for index, record in enumerate(records)
            if record.category == category
        ]
        local_records = [record for _, record in category_pairs]
        local_val, local_test, detail = choose_category_holdouts(
            local_records,
            val_ratio,
            test_ratio,
            seed + offset,
        )
        val_global_indices.update(category_pairs[index][0] for index in local_val)
        test_global_indices.update(category_pairs[index][0] for index in local_test)
        detail["augmented_rows_forced_to_train"] = sum(
            record.category == category and record.is_augmented for record in records
        )
        split_detail[category] = detail
    train = [
        record for index, record in enumerate(records)
        if index not in val_global_indices and index not in test_global_indices
    ]
    val = [record for index, record in enumerate(records) if index in val_global_indices]
    test = [record for index, record in enumerate(records) if index in test_global_indices]
    return train, val, test, split_detail


def count_tags(records: list[Record]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(expression_tags(record))
    return counter


def category_report(
    source: list[Record],
    train: list[Record],
    val: list[Record],
    test: list[Record],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in CATEGORIES:
        source_rows = [record for record in source if record.category == category]
        train_rows = [record for record in train if record.category == category]
        val_rows = [record for record in val if record.category == category]
        test_rows = [record for record in test if record.category == category]
        real_source_rows = [record for record in source_rows if not record.is_augmented]
        real_train_rows = [record for record in train_rows if not record.is_augmented]
        real_val_rows = [record for record in val_rows if not record.is_augmented]
        source_tags = count_tags(source_rows)
        train_tags = count_tags(train_rows)
        val_tags = count_tags(val_rows)
        test_tags = count_tags(test_rows)
        real_source_tags = count_tags(real_source_rows)
        real_train_tags = count_tags(real_train_rows)
        real_val_tags = count_tags(real_val_rows)
        body_tags = sorted(tag for tag in source_tags if tag.startswith("BODY:"))
        real_body_tags = sorted(tag for tag in real_source_tags if tag.startswith("BODY:"))
        form_tags = sorted(tag for tag in source_tags if tag.startswith("FORM:") or tag.startswith("LANG:"))
        result[category] = {
            "source_rows": len(source_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "actual_val_ratio": round(len(val_rows) / len(source_rows), 6) if source_rows else 0.0,
            "actual_test_ratio": round(len(test_rows) / len(source_rows), 6) if source_rows else 0.0,
            "real_source_rows": len(real_source_rows),
            "real_train_rows": len(real_train_rows),
            "real_val_rows": len(real_val_rows),
            "actual_val_ratio_of_real": round(
                len(real_val_rows) / len(real_source_rows), 6
            ) if real_source_rows else 0.0,
            "body_coverage": [
                {
                    "tag": tag,
                    "total": source_tags[tag],
                    "train": train_tags[tag],
                    "val": val_tags[tag],
                    "test": test_tags[tag],
                }
                for tag in body_tags
            ],
            "expression_coverage": [
                {
                    "tag": tag,
                    "total": source_tags[tag],
                    "train": train_tags[tag],
                    "val": val_tags[tag],
                    "test": test_tags[tag],
                }
                for tag in form_tags
            ],
            "missing_body_tags_in_val": [tag for tag in body_tags if val_tags[tag] == 0],
            "missing_body_tags_in_test": [tag for tag in body_tags if test_tags[tag] == 0],
            "missing_body_tags_in_train": [tag for tag in body_tags if train_tags[tag] == 0],
            "real_body_coverage": [
                {
                    "tag": tag,
                    "total": real_source_tags[tag],
                    "train": real_train_tags[tag],
                    "val": real_val_tags[tag],
                    "val_ratio": round(
                        real_val_tags[tag] / real_source_tags[tag], 6
                    ) if real_source_tags[tag] else 0.0,
                }
                for tag in real_body_tags
            ],
            "missing_real_body_tags_in_val": [
                tag for tag in real_body_tags if real_val_tags[tag] == 0
            ],
        }
    return result


def provenance_report(records: list[Record]) -> dict[str, Any]:
    source_labels = Counter(record.source_label or "<EMPTY>" for record in records)
    augmented_rows = sum(record.is_augmented for record in records)
    return {
        "rows": len(records),
        "original_rows": len(records) - augmented_rows,
        "augmented_rows": augmented_rows,
        "augmented_ratio": round(augmented_rows / len(records), 6) if records else 0.0,
        "source_label_distribution": dict(source_labels.most_common()),
    }


def split_overlap_report(left: list[Record], right: list[Record]) -> dict[str, Any]:
    left_descriptions = {description_key(record.input) for record in left}
    right_descriptions = {description_key(record.input) for record in right}
    description_overlap = left_descriptions & right_descriptions
    left_skeletons = {surface_skeleton(record.input) for record in left}
    right_skeletons = {surface_skeleton(record.input) for record in right}
    skeleton_overlap = left_skeletons & right_skeletons
    return {
        "description_overlap": len(description_overlap),
        "surface_skeleton_overlap": len(skeleton_overlap),
        "surface_skeleton_overlap_examples": sorted(skeleton_overlap)[:50],
    }


def split_quality_assessment(
    source: list[Record],
    val: list[Record],
    overlaps: dict[str, dict[str, Any]],
    split_detail: dict[str, Any],
    requested_val_ratio: float,
) -> dict[str, Any]:
    """按固定工程阈值评估自动拆分质量。"""
    categories: dict[str, Any] = {}
    for category in CATEGORIES:
        real_rows = [
            record for record in source
            if record.category == category and not record.is_augmented
        ]
        val_rows = [record for record in val if record.category == category]
        source_tags: Counter[str] = Counter()
        val_tags: Counter[str] = Counter()
        for record in real_rows:
            source_tags.update(
                tag for tag in normalized_label_tags(record)
                if not tag.startswith(("CATEGORY:", "OUTPUT:"))
            )
        for record in val_rows:
            val_tags.update(
                tag for tag in normalized_label_tags(record)
                if not tag.startswith(("CATEGORY:", "OUTPUT:"))
            )

        deviations_20 = [
            abs(val_tags[tag] / total - requested_val_ratio)
            for tag, total in source_tags.items()
            if total >= 20
        ]
        deviations_50 = [
            abs(val_tags[tag] / total - requested_val_ratio)
            for tag, total in source_tags.items()
            if total >= 50
        ]
        top_deviations = sorted(
            (
                {
                    "tag": tag,
                    "total": total,
                    "val": val_tags[tag],
                    "val_ratio": round(val_tags[tag] / total, 6),
                    "absolute_deviation": round(
                        abs(val_tags[tag] / total - requested_val_ratio), 6
                    ),
                }
                for tag, total in source_tags.items()
                if total >= 20
            ),
            key=lambda item: (-item["absolute_deviation"], item["tag"]),
        )
        skeleton_counts = Counter(surface_skeleton(record.input) for record in val_rows)
        max_skeleton_rows = max(skeleton_counts.values(), default=0)
        detail = split_detail.get(category) or {}
        common_missing = (
            detail.get("selection_constraints", {})
            .get("val", {})
            .get("common_tags_missing", [])
        )
        actual_ratio = len(val_rows) / len(real_rows) if real_rows else 0.0
        criteria = {
            "real_val_ratio_close_to_target": abs(actual_ratio - requested_val_ratio) <= 0.001,
            "common_labels_and_output_signatures_covered": not common_missing,
            "mean_deviation_for_labels_ge_20_within_2pct": (
                sum(deviations_20) / len(deviations_20) <= 0.02
                if deviations_20 else True
            ),
            "max_deviation_for_labels_ge_50_within_5pct": (
                max(deviations_50) <= 0.05 if deviations_50 else True
            ),
            "validation_skeletons_sufficiently_diverse": len(skeleton_counts) >= min(
                50, max(10, len(val_rows) // 4)
            ),
            "single_skeleton_share_within_10pct": (
                max_skeleton_rows / len(val_rows) <= 0.1 if val_rows else False
            ),
            "validation_labels_all_retained_in_train": not detail.get(
                "missing_normalized_labels_in_train"
            ),
        }
        categories[category] = {
            "status": "通过" if all(criteria.values()) else "不通过",
            "criteria": criteria,
            "real_rows": len(real_rows),
            "val_rows": len(val_rows),
            "actual_val_ratio": round(actual_ratio, 6),
            "validation_skeleton_groups": len(skeleton_counts),
            "max_single_skeleton_rows": max_skeleton_rows,
            "max_single_skeleton_share": round(
                max_skeleton_rows / len(val_rows), 6
            ) if val_rows else 0.0,
            "common_tags_missing": common_missing,
            "mean_absolute_deviation_labels_ge_20": round(
                sum(deviations_20) / len(deviations_20), 6
            ) if deviations_20 else 0.0,
            "max_absolute_deviation_labels_ge_50": round(
                max(deviations_50), 6
            ) if deviations_50 else 0.0,
            "largest_label_deviations_ge_20": top_deviations[:20],
        }

    overlap_ok = all(
        detail["description_overlap"] == 0
        and detail["surface_skeleton_overlap"] == 0
        for detail in overlaps.values()
    )
    overall_ok = overlap_ok and all(
        detail["status"] == "通过" for detail in categories.values()
    )
    return {
        "overall_status": "通过" if overall_ok else "不通过",
        "zero_description_and_skeleton_overlap": overlap_ok,
        "thresholds": {
            "val_ratio_absolute_tolerance": 0.001,
            "mean_label_deviation_min_count_20": 0.02,
            "max_label_deviation_min_count_50": 0.05,
            "max_single_skeleton_share": 0.1,
            "common_label_min_rows": 20,
        },
        "categories": categories,
    }


def raw_rows(records: list[Record]) -> list[dict[str, Any]]:
    return [{"input": record.input, "output": record.output} for record in records]


def llamafactory_rows(records: list[Record], instruction: str) -> list[dict[str, str]]:
    return [
        {
            "instruction": instruction,
            "input": record.input,
            "output": json.dumps(record.output, ensure_ascii=False),
        }
        for record in records
    ]


def conflict_review_rows(deduplication_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(deduplication_report.get("conflict_groups") or [], 1):
        labels = group.get("labels") or []
        sources = group.get("sources") or []
        candidates = []
        for index, output in enumerate(labels):
            candidates.append({
                "candidate_index": index + 1,
                "source": sources[index] if index < len(sources) else "",
                "output": output,
            })
        rows.append({
            "conflict_id": group_index,
            "input": group.get("input") or "",
            "occurrences": group.get("occurrences") or len(candidates),
            "candidates": candidates,
            "review_status": "待审核",
            "selected_candidate_index": None,
            "corrected_output": None,
            "review_note": "",
        })
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "读取法兰、直管、管件数据，按实际描述 skeleton 整组拆分，"
            "分别生成各类别训练集和验证集；增强数据只进入训练集。"
        )
    )
    parser.add_argument("--flange", required=True, help="法兰 input/output JSON")
    parser.add_argument("--pipe", required=True, help="直管 input/output JSON")
    parser.add_argument("--fitting", required=True, help="管件 input/output JSON")
    parser.add_argument("--raw-output-dir", required=True, help="原始 input/output 格式输出目录")
    parser.add_argument("--llamafactory-output-dir", required=True, help="LlamaFactory 格式输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="未指定验证集时的验证比例，默认 0.1")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="兼容参数，默认 0，不生成测试集",
    )
    parser.add_argument(
        "--val-file",
        action="append",
        default=[],
        help="指定验证集文件，可重复传入；支持原始格式和 LlamaFactory 格式",
    )
    parser.add_argument(
        "--test-file",
        action="append",
        default=[],
        help="指定固定测试集文件，可重复传入；测试集不参与训练和 checkpoint 选择",
    )
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="LlamaFactory instruction 提示词文件")
    parser.add_argument("--skeleton-dir", default=str(DEFAULT_SKELETON_DIR), help="直管/法兰/管件骨架目录")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="自动切分随机种子")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 0 到 1 之间")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio 必须在 0 到 1 之间")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio 与 --test-ratio 之和必须小于 1")

    prompt_path = Path(args.prompt).expanduser().resolve()
    instruction = prompt_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"提示词文件为空: {prompt_path}")
    skeleton_dir = Path(args.skeleton_dir).expanduser().resolve()
    skeletons = load_skeletons(skeleton_dir)

    source_paths = {
        "法兰": Path(args.flange).expanduser().resolve(),
        "直管": Path(args.pipe).expanduser().resolve(),
        "管件": Path(args.fitting).expanduser().resolve(),
    }
    source_loaded, source_load_report = load_source_records(source_paths, skeletons)
    source_records, source_dedup_report = deduplicate_records(source_loaded, reject_conflicts=False)

    explicit_val_paths = [Path(path).expanduser().resolve() for path in args.val_file]
    explicit_test_paths = [Path(path).expanduser().resolve() for path in args.test_file]
    explicit_report: dict[str, Any] = {}
    if explicit_val_paths or explicit_test_paths:
        val_records: list[Record] = []
        test_records: list[Record] = []
        val_report: dict[str, Any] = {}
        test_report: dict[str, Any] = {}
        if explicit_val_paths:
            val_loaded, val_load_report = load_validation_records(explicit_val_paths, skeletons)
            val_records, val_dedup_report = deduplicate_records(val_loaded, reject_conflicts=True)
            val_report = {"load": val_load_report, "deduplication": val_dedup_report}
        if explicit_test_paths:
            test_loaded, test_load_report = load_validation_records(explicit_test_paths, skeletons)
            test_records, test_dedup_report = deduplicate_records(test_loaded, reject_conflicts=True)
            test_report = {"load": test_load_report, "deduplication": test_dedup_report}

        if any(record.is_augmented for record in val_records + test_records):
            raise ValueError("固定验证集/测试集中不能包含带‘数据增强’标识的样本")

        val_keys = {description_key(record.input) for record in val_records}
        test_keys = {description_key(record.input) for record in test_records}
        explicit_overlap = val_keys & test_keys
        if explicit_overlap:
            raise ValueError(f"固定验证集与测试集存在 {len(explicit_overlap)} 条重复描述")

        holdout_keys = val_keys | test_keys
        source_keys = {description_key(record.input) for record in source_records}
        source_by_key = {description_key(record.input): record for record in source_records}
        augmented_holdout_matches = [
            key for key in holdout_keys
            if key in source_by_key and source_by_key[key].is_augmented
        ]
        if augmented_holdout_matches:
            raise ValueError(
                f"固定验证/测试集命中 {len(augmented_holdout_matches)} 条只来自增强数据的描述"
            )

        train_records = [
            record for record in source_records
            if description_key(record.input) not in holdout_keys
        ]
        removed = len(source_records) - len(train_records)
        split_mode = "explicit_holdout"
        split_detail: dict[str, Any] = {}
        explicit_report = {
            "validation": val_report,
            "test": test_report,
            "source_rows_removed_by_holdout_description": removed,
            "validation_rows_not_in_source": sum(
                1 for record in val_records
                if description_key(record.input) not in source_keys
            ),
            "test_rows_not_in_source": sum(
                1 for record in test_records
                if description_key(record.input) not in source_keys
            ),
        }
    else:
        train_records, val_records, test_records, split_detail = split_automatically(
            source_records, args.val_ratio, args.test_ratio, args.seed
        )
        split_mode = "automatic_stratified"

    automatic_mode = not (explicit_val_paths or explicit_test_paths)
    overlaps = {
        "train_val": split_overlap_report(train_records, val_records),
        "train_test": split_overlap_report(train_records, test_records),
        "val_test": split_overlap_report(val_records, test_records),
    }
    for pair, detail in overlaps.items():
        if detail["description_overlap"]:
            raise RuntimeError(f"{pair} 仍有 {detail['description_overlap']} 条重复描述")
        if automatic_mode and detail["surface_skeleton_overlap"]:
            raise RuntimeError(
                f"{pair} 仍有 {detail['surface_skeleton_overlap']} 个描述骨架跨集合"
            )
    if any(record.is_augmented for record in val_records + test_records):
        raise RuntimeError("验证集或测试集中仍存在增强样本")

    raw_dir = Path(args.raw_output_dir).expanduser().resolve()
    llama_dir = Path(args.llamafactory_output_dir).expanduser().resolve()
    conflict_review_path = raw_dir / "种类_冲突样本待审核.json"

    category_output_paths: dict[str, dict[str, str]] = {}
    split_records = {
        "train": train_records,
        "val": val_records,
    }
    if test_records:
        split_records["test"] = test_records
    for category in CATEGORIES:
        category_output_paths[category] = {}
        for split, records_for_split in split_records.items():
            category_records = [
                record for record in records_for_split if record.category == category
            ]
            raw_path = raw_dir / f"{category}_{split}.json"
            llama_path = llama_dir / f"{category}_{split}.json"
            write_json(raw_path, raw_rows(category_records))
            write_json(llama_path, llamafactory_rows(category_records, instruction))
            category_output_paths[category][f"raw_{split}"] = str(raw_path)
            category_output_paths[category][f"llamafactory_{split}"] = str(llama_path)
    write_json(conflict_review_path, conflict_review_rows(source_dedup_report))

    source_provenance = provenance_report(source_records)
    original_source_rows = source_provenance["original_rows"]
    category_summary = category_report(
        source_records, train_records, val_records, test_records
    )
    quality_assessment = (
        split_quality_assessment(
            source_records,
            val_records,
            overlaps,
            split_detail,
            args.val_ratio,
        )
        if automatic_mode
        else {}
    )
    raw_quality_report_path = raw_dir / "种类_split_quality_report.json"
    llama_quality_report_path = llama_dir / "种类_split_quality_report.json"
    report = {
        "mode": split_mode,
        "prompt_path": str(prompt_path),
        "skeleton_dir": str(skeleton_dir),
        "requested_val_ratio": args.val_ratio if automatic_mode else None,
        "requested_test_ratio": args.test_ratio if automatic_mode else None,
        "seed": args.seed if automatic_mode else None,
        "source_load": source_load_report,
        "source_deduplication": source_dedup_report,
        "source_provenance": source_provenance,
        "explicit_holdout": explicit_report,
        "automatic_split_detail": split_detail,
        "total_source_rows_after_dedup": len(source_records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "actual_val_ratio": round(len(val_records) / len(source_records), 6) if source_records else 0.0,
        "actual_test_ratio": round(len(test_records) / len(source_records), 6) if source_records else 0.0,
        "actual_val_ratio_of_original": round(len(val_records) / original_source_rows, 6)
        if original_source_rows else 0.0,
        "actual_test_ratio_of_original": round(len(test_records) / original_source_rows, 6)
        if original_source_rows else 0.0,
        "split_provenance": {
            "train": provenance_report(train_records),
            "val": provenance_report(val_records),
            "test": provenance_report(test_records),
        },
        "split_overlap": overlaps,
        "category_summary": category_summary,
        "quality_assessment": quality_assessment,
        "outputs": {
            "by_category": category_output_paths,
            "conflict_review": str(conflict_review_path),
            "raw_quality_report": str(raw_quality_report_path),
            "llamafactory_quality_report": str(llama_quality_report_path),
        },
    }
    write_json(raw_dir / "种类_split_report.json", report)
    write_json(llama_dir / "种类_split_report.json", report)
    if automatic_mode:
        write_json(raw_quality_report_path, quality_assessment)
        write_json(llama_quality_report_path, quality_assessment)
    summary = {
        "mode": split_mode,
        "quality_status": quality_assessment.get("overall_status") if automatic_mode else None,
        "source_rows": len(source_records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "actual_val_ratio": report["actual_val_ratio"],
        "actual_test_ratio": report["actual_test_ratio"],
        "actual_val_ratio_of_original": report["actual_val_ratio_of_original"],
        "actual_test_ratio_of_original": report["actual_test_ratio_of_original"],
        "augmented_rows": {
            split: report["split_provenance"][split]["augmented_rows"]
            for split in ("train", "val", "test")
        },
        "description_overlap": {
            pair: detail["description_overlap"] for pair, detail in overlaps.items()
        },
        "surface_skeleton_overlap": {
            pair: detail["surface_skeleton_overlap"] for pair, detail in overlaps.items()
        },
        "category_rows": {
            category: {
                "train": report["category_summary"][category]["train_rows"],
                "val": report["category_summary"][category]["val_rows"],
                "test": report["category_summary"][category]["test_rows"],
            }
            for category in CATEGORIES
        },
        "raw_report": str(raw_dir / "种类_split_report.json"),
        "llamafactory_report": str(llama_dir / "种类_split_report.json"),
        "raw_quality_report": str(raw_quality_report_path) if automatic_mode else None,
        "llamafactory_quality_report": (
            str(llama_quality_report_path) if automatic_mode else None
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if automatic_mode and quality_assessment["overall_status"] != "通过":
        raise RuntimeError(
            f"自动划分质量验收未通过，详见: {raw_quality_report_path}"
        )


if __name__ == "__main__":
    main()
