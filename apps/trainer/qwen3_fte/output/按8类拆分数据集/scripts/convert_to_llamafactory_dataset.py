from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

"""

将 parser_train_new_schema.json 转换为 LlamaFactory 训练格式。

用法示例：
python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/材质规范.json

python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范_train.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/材质规范_train.json \
    --val-input apps/trainer/qwen3_fte/output/按8类拆分数据集/材质规范_val.json \
    --val-output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/材质规范_val.json



python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类_train.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_train.json

    
python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类0629.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_train.json \
    --val-input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类_val.json \
    --val-output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_val.json

python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类_val.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_val.json

python apps/trainer/qwen3_fte/output/按8类拆分数据集/scripts/convert_to_llamafactory_dataset.py \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_train.json \
    --val-input apps/trainer/qwen3_fte/output/按8类拆分数据集/种类_val.json \
    --val-output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/种类_val.json

python apps/trainer/qwen3_fte/src/convert_to_llamafactory_dataset.py \
    --mode keep \
    --skeleton 法兰 \
    --input apps/trainer/qwen3_fte/output/按8类拆分数据集/法兰.json \
    --output apps/trainer/qwen3_fte/output/按8类拆分llamafactory数据集/法兰.json

 mode raw|keep|strip  raw 模式（默认，保留原结构） keep 模式（保留完整骨架） strip 模式（省略空值）

python convert_to_llamafactory_dataset.py \
    --mode raw|keep|strip  raw 模式（默认，保留原结构） keep 模式（保留完整骨架） strip 模式（省略空值）

"""
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_INPUT = (
    PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte" / "output" 
    / "parser_train_new_schema.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte" / "output" 
    / "llamafactory_train.json"
)
DEFAULT_SKELETON_DIR = (
    PROJECT_ROOT / "apps" / "trainer" / "qwen3_fte" / "skeletons"
)
DEFAULT_SKELETON = "full"

INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)

MATERIAL_TEMPLATE: dict[str, Any] = {
    "ROLE": "",
    "VALUE": "",
    "SPECIAL_REQ": [],
}

STANDARD_TEMPLATE: dict[str, Any] = {
    "BODY": "",
    "GRADE": "",
    "METHOD": "",
    "APPENDIX": "",
}

ARRAY_ELEMENT_TEMPLATES: dict[str, dict[str, Any]] = {
    "MATERIAL": MATERIAL_TEMPLATE,
    "STANDARD": STANDARD_TEMPLATE,
}


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL 第 {line_no} 行不是对象")
                rows.append(item)
        return rows

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 顶层必须是数组")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON 数组元素必须全部是对象")
        return data

    raise ValueError(f"不支持的文件类型: {path.suffix}")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and all(_is_empty(v) for v in value.values()):
        return True
    return False


def strip_empty(obj: Any) -> Any:
    """递归移除空值字段（空字符串、空数组、全空子对象）。"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v_clean = strip_empty(v)
            if not _is_empty(v_clean):
                cleaned[k] = v_clean
        return cleaned
    if isinstance(obj, list):
        return [strip_empty(item) for item in obj]
    return obj


def _deep_merge(skeleton: Any, data: Any) -> Any:
    """将 data 合并到 skeleton 中，skeleton 提供缺失字段的默认值。"""
    if isinstance(skeleton, dict) and isinstance(data, dict):
        merged = {}
        for k in skeleton:
            if k in data:
                merged[k] = _deep_merge(skeleton[k], data[k])
            else:
                merged[k] = _deep_copy(skeleton[k])
        for k in data:
            if k not in skeleton:
                merged[k] = data[k]
        return merged
    if data is not None:
        return data
    return _deep_copy(skeleton)


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(item) for item in obj]
    return obj


def _apply_array_templates(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """对数组字段中的每个元素，用对应模板补齐缺失的 key。"""
    result = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if path in ARRAY_ELEMENT_TEMPLATES and isinstance(v, list):
            tpl = ARRAY_ELEMENT_TEMPLATES[path]
            result[k] = [_deep_merge(tpl, elem) if isinstance(elem, dict) else elem
                         for elem in v]
        elif isinstance(v, dict):
            result[k] = _apply_array_templates(v, path)
        else:
            result[k] = v
    return result


def load_skeleton(skeleton: str | None, skeleton_dir: Path) -> dict[str, Any]:
    if not skeleton:
        skeleton = DEFAULT_SKELETON
    candidate = Path(skeleton)
    if candidate.suffix.lower() == ".json" or "/" in skeleton or "\\" in skeleton:
        path = candidate
    else:
        path = skeleton_dir / f"{skeleton}.json"
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve() if not path.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"骨架文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"骨架文件不是 JSON 对象: {path}")
    return data


def keep_all(obj: dict[str, Any], skeleton: dict[str, Any]) -> dict[str, Any]:
    """用指定骨架补齐所有缺失字段，保留空值。"""
    merged = _deep_merge(skeleton, obj)
    return _apply_array_templates(merged)


def process_output(output_value: Any, mode: str, skeleton: dict[str, Any] | None) -> str:
    if isinstance(output_value, str):
        output_value = json.loads(output_value)

    if not isinstance(output_value, dict):
        raise ValueError(f"output 字段类型不支持: {type(output_value).__name__}")

    if mode == "raw":
        pass
    elif mode == "strip":
        output_value = strip_empty(output_value)
    elif mode == "keep":
        if skeleton is None:
            raise ValueError("keep 模式必须提供骨架")
        output_value = keep_all(output_value, skeleton)

    return json.dumps(output_value, ensure_ascii=False)


def convert_record(item: dict[str, Any], mode: str, skeleton: dict[str, Any] | None) -> dict[str, str]:
    if "input" not in item:
        raise ValueError("样本缺少 input 字段")
    if "output" not in item:
        raise ValueError("样本缺少 output 字段")

    return {
        "instruction": INSTRUCTION,
        "input": str(item["input"]).strip(),
        "output": process_output(item["output"], mode, skeleton),
    }


def normalize_input_text(item: dict[str, Any]) -> str:
    if "input" not in item:
        raise ValueError("样本缺少 input 字段")
    return str(item["input"]).strip()


def filter_train_rows_by_val(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    val_inputs = {normalize_input_text(item) for item in val_rows}
    filtered_rows = [item for item in train_rows if normalize_input_text(item) not in val_inputs]
    skipped_count = len(train_rows) - len(filtered_rows)
    return filtered_rows, skipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将标注数据集转换为 LLaMA-Factory 训练格式"
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=DEFAULT_INPUT,
        help="输入数据集路径（.json 或 .jsonl）",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="输出 LLaMA-Factory 格式文件路径",
    )
    parser.add_argument(
        "--val-input", type=Path,
        help="可选：验证集源文件路径（.json 或 .jsonl）。指定后，训练集会跳过其中出现过的 input，同时该验证集也会被转换。",
    )
    parser.add_argument(
        "--val-output", type=Path,
        help="可选：验证集输出路径。指定 --val-input 时建议同时指定；若未指定，则默认在 train 输出同目录下生成同名 *_val.json 文件。",
    )
    parser.add_argument(
        "-m", "--mode", choices=["raw", "strip", "keep"], default="raw",
        help="空值处理模式: raw=保留原结构（默认）, strip=省略空值, keep=保留完整骨架",
    )
    parser.add_argument(
        "--skeleton", default=DEFAULT_SKELETON,
        help="骨架名称或骨架 JSON 路径。名称会到 skeleton-dir 下查找同名 .json 文件",
    )
    parser.add_argument(
        "--skeleton-dir", type=Path, default=DEFAULT_SKELETON_DIR,
        help="骨架目录，默认 apps/trainer/qwen3_fte/skeletons",
    )
    args = parser.parse_args()

    rows = load_records(args.input)
    original_train_count = len(rows)
    val_rows: list[dict[str, Any]] = []
    skipped_count = 0
    if args.val_input:
        val_rows = load_records(args.val_input)
        rows, skipped_count = filter_train_rows_by_val(rows, val_rows)

    skeleton = load_skeleton(args.skeleton, args.skeleton_dir) if args.mode == "keep" else None
    converted = [convert_record(item, args.mode, skeleton) for item in rows]
    converted_val = [convert_record(item, args.mode, skeleton) for item in val_rows] if val_rows else []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    actual_val_output: Path | None = None
    if args.val_input:
        actual_val_output = args.val_output
        if actual_val_output is None:
            stem = args.output.stem
            if stem.endswith("_train"):
                val_stem = stem[:-6] + "_val"
            else:
                val_stem = stem + "_val"
            actual_val_output = args.output.with_name(f"{val_stem}{args.output.suffix}")
        actual_val_output.parent.mkdir(parents=True, exist_ok=True)
        actual_val_output.write_text(
            json.dumps(converted_val, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    mode_text = {
        "raw": "保留原结构",
        "strip": "省略空值",
        "keep": "保留完整骨架",
    }[args.mode]
    print(f"模式:   {args.mode} ({mode_text})")
    if args.mode == "keep":
        print(f"骨架:   {args.skeleton}")
        print(f"骨架目录: {args.skeleton_dir}")
    print(f"输入:   {args.input}")
    print(f"输出:   {args.output}")
    print(f"样本数: {len(converted)}")
    if args.val_input:
        ratio_vs_original = (len(converted_val) / original_train_count * 100) if original_train_count else 0.0
        ratio_vs_final_train = (len(converted_val) / len(converted) * 100) if converted else 0.0
        print(f"验证集输入: {args.val_input}")
        print(f"验证集输出: {actual_val_output}")
        print(f"验证集样本数: {len(converted_val)}")
        print(f"训练集因验证集去重跳过: {skipped_count}")
        print(f"验证集占原始训练集比例: {ratio_vs_original:.2f}%")
        print(f"验证集占最终训练集比例: {ratio_vs_final_train:.2f}%")

    if converted:
        sample = json.loads(converted[0]["output"])
        print(f"\n=== 训练集第 1 条样本 output 预览 ===")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    else:
        print("\n训练集为空，未输出样本预览。")

    if converted_val:
        val_sample = json.loads(converted_val[0]["output"])
        print(f"\n=== 验证集第 1 条样本 output 预览 ===")
        print(json.dumps(val_sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
