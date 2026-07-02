#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过一阶段数据集获取种类数据集。

用途：
1. 读取一阶段结构化数据集（json / jsonl）
2. 读取 Excel 中的“材料描述 + 分类”映射
3. 按材料描述匹配分类
4. 输出“一个分类一个文件”的 TYPE 数据集

输出样例：
{
  "input": "...",
  "output": {
    "TYPE": {
      "BODY": "...",
      "GEOMETRY": {"ANGLE": "", "RADIUS": ""},
      "MANU": [],
      "CONN": [],
      "SEAL": [],
      "ENDS": []
    }
  }
}

示例：
python scripts/通过一阶段数据集获取种类数据集.py \
  --stage1-file /Users/guoxi/Downloads/stage1_dataset_2026-06-10.json \
  --excel-file /Users/guoxi/Downloads/1-111.xlsx \
  --output-dir outputs/type_dataset_by_category


python scripts/通过一阶段数据集获取种类数据集.py \
    --stage1-file data/pipe/llm_lora/ner_data_new_schema.json \
    --excel-file /path/to/材料分类.xlsx \
    --output-dir outputs/type_dataset_by_category \
    --desc-col 描述 \
    --category-col 分类
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_STAGE1_FILE = Path("data/pipe/llm_lora/ner_data_new_schema.json")
DEFAULT_OUTPUT_DIR = Path("outputs/type_dataset_by_category")
DEFAULT_DESC_COLUMNS = ("材料描述", "材料描述(多行)", "描述", "描述(多行)")
DEFAULT_CATEGORY_COLUMNS = ("分类",)


def _normalize_text(value: Any) -> str:
    """统一文本匹配口径，只做轻量归一，不改语义。"""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sanitize_filename(name: str) -> str:
    """把分类名转成安全文件名。"""
    text = str(name or "").strip() or "未分类"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """兼容 json 数组和 jsonl。"""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    data = json.loads(raw)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"不支持的 JSON 结构: {path}")


def _pick_first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    existing = set(columns)
    for name in candidates:
        if name in existing:
            return name
    raise KeyError(f"未找到列，候选为: {list(candidates)}")


def _build_desc_to_category_map(
    excel_path: Path,
    desc_col: str | None,
    category_col: str | None,
) -> Tuple[Dict[str, str], str, str]:
    """从 Excel 建立 描述 -> 分类 的映射。"""
    df = pd.read_excel(excel_path, dtype=object)

    resolved_desc_col = desc_col or _pick_first_existing(df.columns, DEFAULT_DESC_COLUMNS)
    resolved_category_col = category_col or _pick_first_existing(df.columns, DEFAULT_CATEGORY_COLUMNS)

    desc_to_category: Dict[str, str] = {}
    duplicate_conflicts: Dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        desc = _normalize_text(row.get(resolved_desc_col))
        category = _normalize_text(row.get(resolved_category_col))
        if not desc or not category:
            continue
        if desc in desc_to_category and desc_to_category[desc] != category:
            duplicate_conflicts[desc].update({desc_to_category[desc], category})
            continue
        desc_to_category[desc] = category

    if duplicate_conflicts:
        print(f"[警告] Excel 中存在同一描述对应多个分类，冲突描述数: {len(duplicate_conflicts)}")

    return desc_to_category, resolved_desc_col, resolved_category_col


def _extract_type_only_sample(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """从一阶段样本里抽出只保留 TYPE 的新样本。"""
    input_text = _normalize_text(row.get("input"))
    output_obj = row.get("output")
    if not input_text or not isinstance(output_obj, dict):
        return None
    type_obj = output_obj.get("TYPE")
    if not isinstance(type_obj, dict):
        return None
    return {
        "input": input_text,
        "output": {
            "TYPE": type_obj,
        },
    }


def split_stage1_type_dataset(
    stage1_rows: List[Dict[str, Any]],
    desc_to_category: Dict[str, str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """按分类切分 TYPE 数据集。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unmatched: List[Dict[str, Any]] = []

    for row in stage1_rows:
        sample = _extract_type_only_sample(row)
        if sample is None:
            continue
        input_text = sample["input"]
        category = desc_to_category.get(input_text)
        if not category:
            unmatched.append(sample)
            continue
        grouped[category].append(sample)

    return grouped, unmatched


def write_grouped_json(output_dir: Path, grouped: Dict[str, List[Dict[str, Any]]], unmatched: List[Dict[str, Any]]) -> None:
    """将分类结果写入目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for category, rows in grouped.items():
        file_name = _sanitize_filename(category) + ".json"
        output_path = output_dir / file_name
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if unmatched:
        unmatched_path = output_dir / "未匹配分类.json"
        unmatched_path.write_text(
            json.dumps(unmatched, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="通过一阶段数据集获取种类数据集")
    parser.add_argument("--stage1-file", type=Path, default=DEFAULT_STAGE1_FILE, help="一阶段数据集文件，支持 json/jsonl")
    parser.add_argument("--excel-file", type=Path, required=True, help="包含材料描述和分类的 Excel 文件")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--desc-col", type=str, default=None, help="Excel 中描述列名，不传则自动探测")
    parser.add_argument("--category-col", type=str, default=None, help="Excel 中分类列名，不传则自动探测")
    args = parser.parse_args()

    if not args.stage1_file.exists():
        raise FileNotFoundError(f"一阶段数据集不存在: {args.stage1_file}")
    if not args.excel_file.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {args.excel_file}")

    stage1_rows = _load_json_or_jsonl(args.stage1_file)
    desc_to_category, resolved_desc_col, resolved_category_col = _build_desc_to_category_map(
        excel_path=args.excel_file,
        desc_col=args.desc_col,
        category_col=args.category_col,
    )
    grouped, unmatched = split_stage1_type_dataset(stage1_rows, desc_to_category)
    write_grouped_json(args.output_dir, grouped, unmatched)

    total_output = sum(len(rows) for rows in grouped.values())
    print(f"[完成] 输入样本数: {len(stage1_rows)}")
    print(f"[完成] Excel 匹配列: 描述={resolved_desc_col}, 分类={resolved_category_col}")
    print(f"[完成] 匹配成功: {total_output}")
    print(f"[完成] 未匹配: {len(unmatched)}")
    print(f"[完成] 分类文件数: {len(grouped)}")
    print(f"[完成] 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
