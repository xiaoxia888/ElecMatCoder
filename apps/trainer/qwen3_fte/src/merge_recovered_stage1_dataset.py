"""拆分恢复的一阶段数据，并与现有 LlamaFactory 数据集合并。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RECOVERED_PATH = PROJECT_ROOT / "runtime/artifacts/一阶段数据集_恢复.json"
DEFAULT_OLD_ROOT = (
    PROJECT_ROOT / "apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "apps/trainer/qwen3_fte/output/一阶段恢复合并数据集_20260911"
)

RESULT_COLUMNS = tuple(
    f"{field}_原始结果"
    for field in ("TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD")
)
TYPE_CATEGORIES = ("直管", "法兰", "管件")
TASKS = ("种类", "材质规范", "尺寸壁厚磅级")

OLD_FILES = {
    "种类": (
        "种类/0806/种类_train.json",
        "种类/0806/种类_val.json",
    ),
    "材质规范": (
        "材质规范/0814/材质规范_train.json",
        "材质规范/0814/材质规范_val.json",
    ),
    "尺寸壁厚磅级": (
        "尺寸壁厚磅级/0817/尺寸壁厚磅级_train.json",
        "尺寸壁厚磅级/0817/尺寸壁厚磅级_val.json",
    ),
}


@dataclass(frozen=True)
class PreparedRecovered:
    accepted: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    exact_duplicate_rows: int
    order_only_duplicate_rows: int
    order_only_duplicate_groups: int
    conflict_rows: int


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"数据集顶层必须是数组: {path}")
    return value


def _write_json_array(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for row in rows:
            if count:
                handle.write(",\n")
            handle.write("  ")
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            count += 1
        handle.write("\n]\n")
    return count


def _normalize_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_arrays(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_normalize_arrays(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    return value


def _parsed_output(candidate: Mapping[str, Any]) -> Any:
    raw_output = (candidate.get("record") or {}).get("output")
    if not isinstance(raw_output, str):
        raise ValueError("LlamaFactory 样本 output 必须是 JSON 字符串")
    return json.loads(raw_output)


def _raw_output_key(candidate: Mapping[str, Any]) -> str:
    return json.dumps(
        _parsed_output(candidate),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_output_key(candidate: Mapping[str, Any]) -> str:
    return json.dumps(
        _normalize_arrays(_parsed_output(candidate)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prepare_recovered(
    candidates: Sequence[dict[str, Any]], *, task: str
) -> PreparedRecovered:
    """按描述去重；数组仅换序视为同标签，真实冲突整组排除。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        input_text = str((candidate.get("record") or {}).get("input") or "")
        groups[input_text].append(candidate)

    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    exact_duplicate_rows = 0
    order_only_duplicate_rows = 0
    order_only_duplicate_groups = 0
    conflict_rows = 0

    for input_text, group in groups.items():
        semantic_keys = {_semantic_output_key(candidate) for candidate in group}
        if len(semantic_keys) > 1:
            conflict_rows += len(group)
            conflicts.append(
                {
                    "任务": task,
                    "材料描述": input_text,
                    "源序号": [str(item.get("source_sequence") or "") for item in group],
                    "分类": [str(item.get("category") or "") for item in group],
                    "候选标签": [_parsed_output(item) for item in group],
                    "处理结果": "整组暂不加入训练集，等待人工复核",
                }
            )
            continue

        accepted.append(group[0])
        if len(group) == 1:
            continue
        first_raw_key = _raw_output_key(group[0])
        has_order_only_difference = False
        for candidate in group[1:]:
            if _raw_output_key(candidate) == first_raw_key:
                exact_duplicate_rows += 1
            else:
                order_only_duplicate_rows += 1
                has_order_only_difference = True
        if has_order_only_difference:
            order_only_duplicate_groups += 1

    return PreparedRecovered(
        accepted=accepted,
        conflicts=conflicts,
        exact_duplicate_rows=exact_duplicate_rows,
        order_only_duplicate_rows=order_only_duplicate_rows,
        order_only_duplicate_groups=order_only_duplicate_groups,
        conflict_rows=conflict_rows,
    )


def merge_with_existing(
    old_records: Sequence[dict[str, Any]],
    recovered: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """旧数据优先，只有旧数据中不存在的描述才追加。"""
    old_inputs = {str(row.get("input") or "") for row in old_records}
    additions = [
        candidate["record"]
        for candidate in recovered
        if str(candidate["record"].get("input") or "") not in old_inputs
    ]
    skipped = len(recovered) - len(additions)
    return [*old_records, *additions], additions, skipped


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _build_candidates(
    recovered_rows: Sequence[dict[str, Any]],
    source_rows: Sequence[dict[str, str]],
    instructions: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if len(recovered_rows) != len(source_rows):
        raise ValueError(
            f"恢复数据与 CSV 有效行数量不一致: {len(recovered_rows)} != {len(source_rows)}"
        )

    result = {task: [] for task in TASKS}
    for sample, source in zip(recovered_rows, source_rows):
        original_input = str(sample.get("original_input") or "").strip()
        source_input = str(source.get("原始描述") or "").strip()
        if original_input != source_input:
            raise ValueError(
                f"恢复数据与 CSV 顺序不一致，CSV 序号={source.get('序号', '')}: "
                f"{original_input!r} != {source_input!r}"
            )
        category = str(source.get("分类") or "").strip()
        if category not in TYPE_CATEGORIES:
            raise ValueError(
                f"CSV 序号={source.get('序号', '')} 的分类不是直管/法兰/管件: {category!r}"
            )
        output = sample.get("output") or {}
        common = {
            "source_sequence": str(source.get("序号") or ""),
            "category": category,
        }
        task_outputs = {
            "种类": {"CATEGORY": category, "TYPE": output.get("TYPE") or {}},
            "材质规范": {
                "MATERIAL": output.get("MATERIAL") or [],
                "STANDARD": output.get("STANDARD") or [],
            },
            "尺寸壁厚磅级": {
                "ITEMS": output.get("ITEMS") or [],
                "LENGTH": output.get("LENGTH") or "",
                "PRESSURE": output.get("PRESSURE") or "",
            },
        }
        for task, task_output in task_outputs.items():
            result[task].append(
                {
                    **common,
                    "record": {
                        "instruction": instructions[task],
                        "input": str(sample.get("input") or ""),
                        "output": _compact_json(task_output),
                    },
                }
            )
    return result


def _load_source_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    all_rows = 0
    effective: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in RESULT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV 缺少必要列: {', '.join(missing)}")
        for row in reader:
            all_rows += 1
            if any(str(row.get(column) or "").strip() for column in RESULT_COLUMNS):
                effective.append(row)
    return effective, all_rows


def _load_old_records(old_root: Path) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    for task, relative_paths in OLD_FILES.items():
        rows: list[dict[str, Any]] = []
        for relative_path in relative_paths:
            rows.extend(_load_json_list(old_root / relative_path))
        if not rows:
            raise ValueError(f"旧{task}数据集为空，无法取得训练提示词")
        datasets[task] = rows
    return datasets


def _record_category(record: Mapping[str, Any]) -> str:
    try:
        output = json.loads(str(record.get("output") or "{}"))
    except json.JSONDecodeError:
        return ""
    return str(output.get("CATEGORY") or "")


def merge_recovered_stage1_dataset(
    *,
    recovered_path: Path,
    source_csv_path: Path,
    old_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    recovered_path = Path(recovered_path)
    source_csv_path = Path(source_csv_path)
    old_root = Path(old_root)
    output_dir = Path(output_dir)

    recovered_rows = _load_json_list(recovered_path)
    source_rows, csv_total_rows = _load_source_rows(source_csv_path)
    old = _load_old_records(old_root)
    instructions = {task: str(rows[0]["instruction"]) for task, rows in old.items()}
    candidates = _build_candidates(recovered_rows, source_rows, instructions)

    prepared = {
        task: prepare_recovered(task_candidates, task=task)
        for task, task_candidates in candidates.items()
    }
    merged: dict[str, list[dict[str, Any]]] = {}
    additions: dict[str, list[dict[str, Any]]] = {}
    task_stats: dict[str, dict[str, int]] = {}
    all_conflicts: list[dict[str, Any]] = []
    for task in TASKS:
        current = prepared[task]
        task_merged, task_additions, skipped_existing = merge_with_existing(
            old[task], current.accepted
        )
        merged[task] = task_merged
        additions[task] = task_additions
        all_conflicts.extend(current.conflicts)
        task_stats[task] = {
            "旧数据数量": len(old[task]),
            "旧数据唯一描述数": len({str(row.get("input") or "") for row in old[task]}),
            "恢复数据原始数量": len(candidates[task]),
            "完全相同重复跳过数": current.exact_duplicate_rows,
            "仅数组换序重复跳过数": current.order_only_duplicate_rows,
            "仅数组换序描述组数": current.order_only_duplicate_groups,
            "真实冲突描述组数": len(current.conflicts),
            "真实冲突排除行数": current.conflict_rows,
            "内部去重及排除冲突后数量": len(current.accepted),
            "与旧数据重复跳过数": skipped_existing,
            "新增数量": len(task_additions),
            "最终数量": len(task_merged),
        }

    split_dir = output_dir / "恢复数据拆分"
    accepted_type = prepared["种类"].accepted
    for category in TYPE_CATEGORIES:
        _write_json_array(
            split_dir / f"{category}.json",
            (
                candidate["record"]
                for candidate in accepted_type
                if candidate["category"] == category
            ),
        )
    _write_json_array(
        split_dir / "材质规范.json",
        (candidate["record"] for candidate in prepared["材质规范"].accepted),
    )
    _write_json_array(
        split_dir / "尺寸壁厚磅级.json",
        (candidate["record"] for candidate in prepared["尺寸壁厚磅级"].accepted),
    )
    for task in TASKS:
        _write_json_array(output_dir / f"{task}.json", merged[task])

    old_type_category_counts = Counter(_record_category(row) for row in old["种类"])
    added_type_category_counts = Counter(
        _record_category(row) for row in additions["种类"]
    )
    accepted_type_category_counts = Counter(
        candidate["category"] for candidate in prepared["种类"].accepted
    )
    source_type_category_counts = Counter(
        candidate["category"] for candidate in candidates["种类"]
    )
    type_category_stats = {
        category: {
            "旧数据数量": old_type_category_counts[category],
            "恢复数据原始数量": source_type_category_counts[category],
            "恢复数据去重且排除冲突后数量": accepted_type_category_counts[category],
            "与旧数据重复跳过数": (
                accepted_type_category_counts[category] - added_type_category_counts[category]
            ),
            "新增数量": added_type_category_counts[category],
            "最终数量": old_type_category_counts[category] + added_type_category_counts[category],
        }
        for category in TYPE_CATEGORIES
    }

    report: dict[str, Any] = {
        "恢复数据集": str(recovered_path.resolve()),
        "来源CSV": str(source_csv_path.resolve()),
        "旧数据集目录": str(old_root.resolve()),
        "输出目录": str(output_dir.resolve()),
        "CSV总行数": csv_total_rows,
        "CSV无结果跳过数": csv_total_rows - len(source_rows),
        "参与拆分的恢复数据数": len(recovered_rows),
        "去重规则": (
            "以格式化后的input精确匹配；旧数据优先；同描述同标签去重；"
            "所有数组仅顺序不同时视为同标签；真正冲突整组排除并进入待复核文件。"
        ),
        "旧数据版本": {
            task: list(relative_paths) for task, relative_paths in OLD_FILES.items()
        },
        "种类分项统计": type_category_stats,
        "任务统计": task_stats,
        "冲突文件": str((output_dir / "标签冲突待复核.json").resolve()),
        "冲突总组数": len(all_conflicts),
    }
    _write_json_array(output_dir / "标签冲突待复核.json", all_conflicts)
    (output_dir / "合并报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, default=DEFAULT_RECOVERED_PATH)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = merge_recovered_stage1_dataset(
        recovered_path=args.recovered,
        source_csv_path=args.source_csv,
        old_root=args.old_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "输出目录": report["输出目录"],
                "种类分项统计": report["种类分项统计"],
                "任务统计": report["任务统计"],
                "冲突总组数": report["冲突总组数"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
