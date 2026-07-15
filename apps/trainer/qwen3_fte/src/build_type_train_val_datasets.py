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
        }
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
            records.append(Record(
                input=text,
                output=normalize_output(output, category, skeletons),
                category=category,
                source=str(path),
                source_index=index,
            ))
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
        kept.append(grouped[0])
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


def split_automatically(records: list[Record], ratio: float, seed: int) -> tuple[list[Record], list[Record], dict[str, Any]]:
    val_global_indices: set[int] = set()
    split_detail: dict[str, Any] = {}
    for offset, category in enumerate(CATEGORIES):
        category_pairs = [(index, record) for index, record in enumerate(records) if record.category == category]
        local_records = [record for _, record in category_pairs]
        local_val, detail = choose_category_validation(local_records, ratio, seed + offset)
        val_global_indices.update(category_pairs[index][0] for index in local_val)
        split_detail[category] = detail
    train = [record for index, record in enumerate(records) if index not in val_global_indices]
    val = [record for index, record in enumerate(records) if index in val_global_indices]
    return train, val, split_detail


def count_tags(records: list[Record]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(expression_tags(record))
    return counter


def category_report(source: list[Record], train: list[Record], val: list[Record]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in CATEGORIES:
        source_rows = [record for record in source if record.category == category]
        train_rows = [record for record in train if record.category == category]
        val_rows = [record for record in val if record.category == category]
        source_tags = count_tags(source_rows)
        train_tags = count_tags(train_rows)
        val_tags = count_tags(val_rows)
        body_tags = sorted(tag for tag in source_tags if tag.startswith("BODY:"))
        form_tags = sorted(tag for tag in source_tags if tag.startswith("FORM:") or tag.startswith("LANG:"))
        result[category] = {
            "source_rows": len(source_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "actual_val_ratio": round(len(val_rows) / len(source_rows), 6) if source_rows else 0.0,
            "body_coverage": [
                {"tag": tag, "total": source_tags[tag], "train": train_tags[tag], "val": val_tags[tag]}
                for tag in body_tags
            ],
            "expression_coverage": [
                {"tag": tag, "total": source_tags[tag], "train": train_tags[tag], "val": val_tags[tag]}
                for tag in form_tags
            ],
            "missing_body_tags_in_val": [tag for tag in body_tags if val_tags[tag] == 0],
            "missing_body_tags_in_train": [tag for tag in body_tags if train_tags[tag] == 0],
        }
    return result


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
        description="合并法兰、直管、管件数据，按 skeleton 统一结构并生成种类训练集和验证集。"
    )
    parser.add_argument("--flange", required=True, help="法兰 input/output JSON")
    parser.add_argument("--pipe", required=True, help="直管 input/output JSON")
    parser.add_argument("--fitting", required=True, help="管件 input/output JSON")
    parser.add_argument("--raw-output-dir", required=True, help="原始 input/output 格式输出目录")
    parser.add_argument("--llamafactory-output-dir", required=True, help="LlamaFactory 格式输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="未指定验证集时的验证比例，默认 0.1")
    parser.add_argument(
        "--val-file",
        action="append",
        default=[],
        help="指定验证集文件，可重复传入；支持原始格式和 LlamaFactory 格式",
    )
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="LlamaFactory instruction 提示词文件")
    parser.add_argument("--skeleton-dir", default=str(DEFAULT_SKELETON_DIR), help="直管/法兰/管件骨架目录")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="自动切分随机种子")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 0 到 1 之间")

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
    explicit_report: dict[str, Any] = {}
    if explicit_val_paths:
        val_loaded, explicit_load_report = load_validation_records(explicit_val_paths, skeletons)
        val_records, explicit_dedup_report = deduplicate_records(val_loaded, reject_conflicts=True)
        val_keys = {description_key(record.input) for record in val_records}
        source_keys = {description_key(record.input) for record in source_records}
        train_records = [record for record in source_records if description_key(record.input) not in val_keys]
        removed = len(source_records) - len(train_records)
        split_mode = "explicit_validation"
        split_detail: dict[str, Any] = {}
        explicit_report = {
            "load": explicit_load_report,
            "deduplication": explicit_dedup_report,
            "source_rows_removed_by_validation_description": removed,
            "validation_rows_not_in_source": sum(
                1 for record in val_records
                if description_key(record.input) not in source_keys
            ),
        }
    else:
        train_records, val_records, split_detail = split_automatically(
            source_records, args.val_ratio, args.seed
        )
        split_mode = "automatic_stratified"

    train_keys = {description_key(record.input) for record in train_records}
    val_keys = {description_key(record.input) for record in val_records}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"训练集和验证集仍有 {len(overlap)} 条重复描述")

    raw_dir = Path(args.raw_output_dir).expanduser().resolve()
    llama_dir = Path(args.llamafactory_output_dir).expanduser().resolve()
    raw_train_path = raw_dir / "种类_train.json"
    raw_val_path = raw_dir / "种类_val.json"
    llama_train_path = llama_dir / "种类_train.json"
    llama_val_path = llama_dir / "种类_val.json"
    conflict_review_path = raw_dir / "种类_冲突样本待审核.json"

    write_json(raw_train_path, raw_rows(train_records))
    write_json(raw_val_path, raw_rows(val_records))
    write_json(llama_train_path, llamafactory_rows(train_records, instruction))
    write_json(llama_val_path, llamafactory_rows(val_records, instruction))
    write_json(conflict_review_path, conflict_review_rows(source_dedup_report))

    report = {
        "mode": split_mode,
        "prompt_path": str(prompt_path),
        "skeleton_dir": str(skeleton_dir),
        "requested_val_ratio": args.val_ratio if not explicit_val_paths else None,
        "seed": args.seed if not explicit_val_paths else None,
        "source_load": source_load_report,
        "source_deduplication": source_dedup_report,
        "explicit_validation": explicit_report,
        "automatic_split_detail": split_detail,
        "total_source_rows_after_dedup": len(source_records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "actual_val_ratio": round(len(val_records) / (len(train_records) + len(val_records)), 6)
        if train_records or val_records else 0.0,
        "train_val_description_overlap": len(overlap),
        "category_summary": category_report(source_records, train_records, val_records),
        "outputs": {
            "raw_train": str(raw_train_path),
            "raw_val": str(raw_val_path),
            "llamafactory_train": str(llama_train_path),
            "llamafactory_val": str(llama_val_path),
            "conflict_review": str(conflict_review_path),
        },
    }
    write_json(raw_dir / "种类_split_report.json", report)
    write_json(llama_dir / "种类_split_report.json", report)
    summary = {
        "mode": split_mode,
        "source_rows": len(source_records),
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "actual_val_ratio": report["actual_val_ratio"],
        "train_val_description_overlap": len(overlap),
        "category_rows": {
            category: {
                "train": report["category_summary"][category]["train_rows"],
                "val": report["category_summary"][category]["val_rows"],
            }
            for category in CATEGORIES
        },
        "raw_report": str(raw_dir / "种类_split_report.json"),
        "llamafactory_report": str(llama_dir / "种类_split_report.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
