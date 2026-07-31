from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

"""
评测 LoRA 微调模型的结构化抽取效果。

支持两种模式：
 1. 交互模式：直接输入描述，实时查看模型输出
 2. 批量模式：输入 test_set.json，逐条评测并生成报告
 3. Excel 预测模式：读取材料描述列，保留原列并追加模型预测结果

用法示例：
  # 交互模式
  python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-4B-Instruct-2507 \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-2500-材质规范

  # 编码模型交互模式
  python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --task code \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-4B-Instruct-2507 \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-1200-编码

python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
  --task structural \
  --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-4B-Instruct-2507 \
  --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-4000-尺寸


python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --task type \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-8B \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-2400-种类

  # 新结构材质规范模型交互模式
  python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --task material_standard \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-4B-Instruct-2507 \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-材质规范v3

  # 编码模型批量评测
  python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --task code \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-8B \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-编码 \
      --test-file /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/按8类拆分数据集/all_fields_single_field_train.json \
      --max-samples 100 \
      --report-file /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/eval/code_eval_report.json \
      --detail-file /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/eval/code_eval_details.jsonl

  python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-4B-Instruct-2507 \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-1200-编码

    python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
      --base-model /Users/guoxi/.cache/huggingface/hub/Qwen3-8B \
      --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-2000-种类

  # 批量验证
python evaluate_llamafactory_model.py \
    --base-model /Users/guoxi/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-1400 \
    --test-file ../output/test_set.json \
    --report-file ../output/eval/eval_report.json \
    --detail-file ../output/eval/eval_details.jsonl \
    --max-samples 10

# 批量预测
python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
    --base-model /Users/guoxi/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
    --lora /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/model/checkpoint-1500 \
    --test-file /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/工作簿2_test_file.json \
    --predict \
    --predict-output /Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/trainer/qwen3_fte/output/工作簿2_predicted.json

# Excel 批量预测
python apps/trainer/qwen3_fte/src/evaluate_llamafactory_model.py \
    --task type \
    --base-model /path/to/base-model \
    --lora /path/to/checkpoint \
    --excel-file /path/to/input.xlsx \
    --input-column 材料描述 \
    --excel-output /path/to/output.xlsx

  # 不加载 LoRA（测试底座模型）
  python evaluate_llamafactory_model.py \
      --base-model /path/to/Qwen3-4B
"""


EXTRACT_INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)
STRUCTURAL_INSTRUCTION = (
    "你是工业管道材料描述结构化抽取助手。请从材料描述中抽取尺寸、长度、壁厚和磅级信息，"
    "并输出严格 JSON。输出字段只能包含 SIZE_ITEMS、LENGTH、THICKNESS_ITEMS、PRESSURE。"
    "LENGTH 统一转换为毫米单位，SIZE_ITEMS 和 THICKNESS_ITEMS 按原文顺序输出，不要解释，"
    "描述中同时存在SCHEDULE和MM类型壁厚，需要按照描述原文顺序同时提取出，"
    "不要输出 JSON 以外的内容。"
)
CODE_INSTRUCTION = "你是工业管道材料字段编码助手。请根据字段类型和原始字段值，输出唯一的标准化编码。只输出编码，不要解释。"

DEFAULT_INSTRUCTIONS = {
    "extract": EXTRACT_INSTRUCTION,
    "structural": STRUCTURAL_INSTRUCTION,
    "code": CODE_INSTRUCTION,
}

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt"
TASK_INSTRUCTION_PATHS = {
    "type": PROMPT_DIR / "type_extraction_sft_instruction_v1.txt",
    "material_standard": PROMPT_DIR / "material_standard_extraction_sft_instruction_v3.txt",
}
JSON_TASKS = frozenset({"extract", "type", "structural", "material_standard"})

MATERIAL_V3_PARTS = frozenset({"BODY", "LINING", "INNER_PIPE", "OUTER_PIPE", "FLANGE"})
MATERIAL_V3_SPECIAL_REQS = frozenset(
    {"NACE", "GALVANIZED", "ANTI-H2S", "ANTI-HIC", "ANTI-SCC", "CE", "3PE", "4PE", "PE", "EP"}
)

CODE_FIELDS = ("TYPE", "SIZE", "THICKNESS", "PRESSURE", "MATERIAL", "STANDARD")


def resolve_instruction(task: str, override: str = "") -> tuple[str, Path | None]:
    """按任务读取 system instruction，命令行显式覆盖始终优先。"""
    custom_instruction = str(override or "").strip()
    if custom_instruction:
        return custom_instruction, None

    instruction_path = TASK_INSTRUCTION_PATHS.get(task)
    if instruction_path is not None:
        if not instruction_path.is_file():
            raise FileNotFoundError(f"{task} 提示词文件不存在: {instruction_path}")
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"{task} 提示词文件为空: {instruction_path}")
        return instruction, instruction_path

    return DEFAULT_INSTRUCTIONS[task], None


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────

def load_model(base_model: str, lora_path: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载底座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    print(f"模型加载完成, 设备: {model.device}\n")
    return model, tokenizer


# ──────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────

def build_messages(input_text: str, instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]


def generate(
    model, tokenizer, input_text: str,
    *,
    instruction: str,
    max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
) -> str:
    import torch

    # 保持与 LlamaFactory qwen3_nothink 训练模板一致。
    # 这里不能直接用 tokenizer 自带 chat_template，也不再向 prompt 注入 /no_think，
    # 否则都会偏离训练分布，LoRA 命中会明显变差。
    text = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    bad_words_ids = []
    for marker in ("<think>", "</think>"):
        token_ids = tokenizer.encode(marker, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids or None,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json_output(raw: str) -> dict | None:
    """尝试从模型输出中提取 JSON 对象。"""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def clean_text_output(raw: str) -> str:
    """清理编码模型输出，保留一行最终编码文本。"""
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()


def normalize_code_input(input_text: str, default_field: str = "TYPE") -> str:
    """编码模型训练时使用单字段输入；交互输入原始值时自动补齐字段类型。"""
    text = str(input_text or "").strip()
    if not text:
        return text

    has_field_type = re.search(r"字段类型\s*[:：]", text)
    has_raw_value = re.search(r"原始值\s*[:：]", text)
    if has_field_type and has_raw_value:
        return text

    field_pattern = "|".join(CODE_FIELDS)
    match = re.match(rf"^\s*({field_pattern})\s*[:：]\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        field = match.group(1).upper()
        value = match.group(2).strip()
        return f"字段类型: {field}\n原始值: {value}"

    return f"字段类型: {default_field.upper()}\n原始值: {text}"


def expected_to_text(expected: Any) -> str:
    if isinstance(expected, str):
        return clean_text_output(expected)
    return json.dumps(expected, ensure_ascii=False, sort_keys=True)


def expected_to_json(expected: Any) -> dict | None:
    if isinstance(expected, dict):
        return expected
    if isinstance(expected, str):
        return parse_json_output(expected)
    return None


def validate_material_standard_v3(output: Any) -> list[str]:
    """校验材质规范 v3 的固定输出结构，不对模型结果做静默修复。"""
    errors: list[str] = []
    if not isinstance(output, dict):
        return ["顶层必须是 JSON 对象"]

    expected_root_keys = {"MATERIAL", "STANDARD"}
    actual_root_keys = set(output)
    missing_root_keys = sorted(expected_root_keys - actual_root_keys)
    extra_root_keys = sorted(actual_root_keys - expected_root_keys)
    if missing_root_keys:
        errors.append(f"顶层缺少字段: {', '.join(missing_root_keys)}")
    if extra_root_keys:
        errors.append(f"顶层存在 v3 未定义字段: {', '.join(extra_root_keys)}")

    materials = output.get("MATERIAL")
    if not isinstance(materials, list):
        errors.append("MATERIAL 必须是数组")
    else:
        expected_material_keys = {"PART", "VALUE", "SPECIAL_REQ"}
        for index, item in enumerate(materials):
            path = f"MATERIAL[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{path} 必须是对象")
                continue
            item_keys = set(item)
            if item_keys != expected_material_keys:
                missing = sorted(expected_material_keys - item_keys)
                extra = sorted(item_keys - expected_material_keys)
                if missing:
                    errors.append(f"{path} 缺少字段: {', '.join(missing)}")
                if extra:
                    errors.append(f"{path} 存在 v3 未定义字段: {', '.join(extra)}")
            part = item.get("PART")
            if part not in MATERIAL_V3_PARTS:
                errors.append(f"{path}.PART 非法: {part!r}")
            if not isinstance(item.get("VALUE"), str):
                errors.append(f"{path}.VALUE 必须是字符串")
            special_reqs = item.get("SPECIAL_REQ")
            if not isinstance(special_reqs, list):
                errors.append(f"{path}.SPECIAL_REQ 必须是数组")
            else:
                invalid_reqs = [
                    req
                    for req in special_reqs
                    if not isinstance(req, str) or req not in MATERIAL_V3_SPECIAL_REQS
                ]
                if invalid_reqs:
                    errors.append(f"{path}.SPECIAL_REQ 存在非法值: {invalid_reqs!r}")

    standards = output.get("STANDARD")
    if not isinstance(standards, list):
        errors.append("STANDARD 必须是数组")
    else:
        for index, item in enumerate(standards):
            path = f"STANDARD[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{path} 必须是对象")
            elif set(item) != {"BODY"}:
                errors.append(f"{path} 只能包含 BODY 字段")
            elif not isinstance(item.get("BODY"), str):
                errors.append(f"{path}.BODY 必须是字符串")

    return errors


def validate_task_output(task: str, output: Any) -> list[str]:
    """返回任务结构错误；空数组表示结构合法。"""
    if task == "material_standard":
        return validate_material_standard_v3(output)
    return []


# ──────────────────────────────────────────────
# 评测指标
# ──────────────────────────────────────────────

def _canonical_scalar(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value) if value is not None else ""


def _flatten(obj: Any, prefix: str = "", *, preserve_list_order: bool = False) -> dict[str, Any]:
    """将嵌套 JSON 拍平为 dot-path → value 的映射。"""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            result.update(_flatten(v, f"{prefix}{k}.", preserve_list_order=preserve_list_order))
    elif isinstance(obj, list):
        items = [_canonical_scalar(x) for x in obj]
        result[prefix.rstrip(".")] = items if preserve_list_order else sorted(items)
    else:
        result[prefix.rstrip(".")] = str(obj) if obj is not None else ""
    return result


def compare_record(expected: dict, predicted: dict, *, preserve_list_order: bool = False) -> dict:
    """逐字段对比 expected vs predicted，返回详细结果。"""
    exp_flat = _flatten(expected, preserve_list_order=preserve_list_order)
    pred_flat = _flatten(predicted, preserve_list_order=preserve_list_order)

    all_keys = set(exp_flat.keys()) | set(pred_flat.keys())
    # 过滤掉全空的字段
    all_keys = {
        k for k in all_keys
        if exp_flat.get(k, "") not in ("", []) or pred_flat.get(k, "") not in ("", [])
    }

    correct = 0
    wrong = 0
    missing = 0
    extra = 0
    details = {}

    for key in sorted(all_keys):
        exp_val = exp_flat.get(key, "")
        pred_val = pred_flat.get(key, "")
        if exp_val == pred_val:
            correct += 1
            details[key] = {"status": "correct", "expected": exp_val, "predicted": pred_val}
        elif key not in pred_flat or pred_val in ("", []):
            missing += 1
            details[key] = {"status": "missing", "expected": exp_val, "predicted": pred_val}
        elif key not in exp_flat or exp_val in ("", []):
            extra += 1
            details[key] = {"status": "extra", "expected": exp_val, "predicted": pred_val}
        else:
            wrong += 1
            details[key] = {"status": "wrong", "expected": exp_val, "predicted": pred_val}

    total = correct + wrong + missing + extra
    accuracy = correct / total if total > 0 else 0

    return {
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "extra": extra,
        "total_fields": total,
        "field_accuracy": round(accuracy, 4),
        "details": details,
    }


def aggregate_report(results: list[dict]) -> dict:
    """汇总所有记录的评测结果。"""
    total_records = len(results)
    json_parse_fail = sum(1 for r in results if r.get("json_parse_fail"))
    schema_fail = sum(1 for r in results if r.get("schema_errors"))
    evaluated = [r for r in results if not r.get("json_parse_fail")]

    if not evaluated:
        return {
            "total_records": total_records,
            "json_parse_fail": json_parse_fail,
            "schema_fail": schema_fail,
            "field_accuracy": 0,
        }

    total_correct = sum(r["comparison"]["correct"] for r in evaluated)
    total_fields = sum(r["comparison"]["total_fields"] for r in evaluated)
    total_wrong = sum(r["comparison"]["wrong"] for r in evaluated)
    total_missing = sum(r["comparison"]["missing"] for r in evaluated)
    total_extra = sum(r["comparison"]["extra"] for r in evaluated)
    perfect = sum(1 for r in evaluated if r["comparison"]["field_accuracy"] == 1.0)

    field_accuracies = [r["comparison"]["field_accuracy"] for r in evaluated]
    avg_accuracy = sum(field_accuracies) / len(field_accuracies)

    # 按字段名统计错误率
    field_errors: dict[str, dict[str, int]] = {}
    for r in evaluated:
        for key, detail in r["comparison"]["details"].items():
            if key not in field_errors:
                field_errors[key] = {"correct": 0, "wrong": 0, "missing": 0, "extra": 0, "total": 0}
            field_errors[key][detail["status"]] += 1
            field_errors[key]["total"] += 1

    top_error_fields = sorted(
        [
            (k, v["wrong"] + v["missing"], v["total"],
             round((v["wrong"] + v["missing"]) / v["total"], 4) if v["total"] > 0 else 0)
            for k, v in field_errors.items()
        ],
        key=lambda x: -x[1],
    )[:20]

    return {
        "total_records": total_records,
        "json_parse_fail": json_parse_fail,
        "schema_fail": schema_fail,
        "evaluated_records": len(evaluated),
        "perfect_records": perfect,
        "perfect_rate": round(perfect / len(evaluated), 4),
        "avg_field_accuracy": round(avg_accuracy, 4),
        "total_fields": total_fields,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "total_missing": total_missing,
        "total_extra": total_extra,
        "overall_field_accuracy": round(total_correct / total_fields, 4) if total_fields else 0,
        "top_error_fields": [
            {"field": f, "errors": e, "total": t, "error_rate": r}
            for f, e, t, r in top_error_fields if e > 0
        ],
    }


# ──────────────────────────────────────────────
# 交互模式
# ──────────────────────────────────────────────

def interactive_mode(
    model,
    tokenizer,
    *,
    task: str,
    instruction: str,
    code_field: str = "TYPE",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> None:
    print("=" * 60)
    print("交互模式 — 输入内容，查看模型输出")
    print(f"任务类型: {task}")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n描述> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        model_input = normalize_code_input(user_input, code_field) if task == "code" else user_input

        t0 = time.time()
        raw_output = generate(
            model,
            tokenizer,
            model_input,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - t0

        print(f"\n耗时: {elapsed:.2f}s")
        if task in JSON_TASKS:
            parsed = parse_json_output(raw_output)
            if parsed is not None:
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
                schema_errors = validate_task_output(task, parsed)
                if schema_errors:
                    print("\n[v3 结构校验失败]")
                    for error in schema_errors:
                        print(f"- {error}")
            else:
                print(f"[JSON 解析失败] 原始输出:\n{raw_output}")
        else:
            parsed_text = clean_text_output(raw_output)
            if model_input != user_input:
                print(f"实际输入:\n{model_input}")
            print(parsed_text)


def aggregate_code_report(results: list[dict]) -> dict:
    total_records = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    wrong_results = [r for r in results if not r.get("correct")]

    field_stats: dict[str, dict[str, int]] = {}
    for r in results:
        field = str(r.get("field") or "UNKNOWN")
        if field not in field_stats:
            field_stats[field] = {"total": 0, "correct": 0, "wrong": 0}
        field_stats[field]["total"] += 1
        if r.get("correct"):
            field_stats[field]["correct"] += 1
        else:
            field_stats[field]["wrong"] += 1

    by_field = []
    for field, stat in sorted(field_stats.items()):
        total = stat["total"]
        by_field.append({
            "field": field,
            "total": total,
            "correct": stat["correct"],
            "wrong": stat["wrong"],
            "accuracy": round(stat["correct"] / total, 4) if total else 0,
        })

    return {
        "total_records": total_records,
        "correct": correct,
        "wrong": len(wrong_results),
        "accuracy": round(correct / total_records, 4) if total_records else 0,
        "by_field": by_field,
        "wrong_examples": wrong_results[:50],
    }


def extract_field_from_input(input_text: str) -> str:
    match = re.search(r"字段类型\s*[:：]\s*([A-Za-z_]+)", str(input_text or ""))
    return match.group(1).upper() if match else ""


def print_code_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("编码模型评测报告")
    print("=" * 60)
    print(f"  总样本:      {report['total_records']}")
    print(f"  正确:        {report['correct']}")
    print(f"  错误:        {report['wrong']}")
    print(f"  准确率:      {report['accuracy']:.1%}")

    if report.get("by_field"):
        print("\n  分字段准确率:")
        for item in report["by_field"]:
            print(
                f"    {item['field']}: {item['accuracy']:.1%} "
                f"({item['correct']}/{item['total']})"
            )


def batch_code_mode(
    model, tokenizer,
    test_file: Path,
    report_file: Path | None,
    detail_file: Path | None,
    max_samples: int | None = None,
    *,
    instruction: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> None:
    data = json.loads(test_file.read_text(encoding="utf-8"))
    if max_samples:
        data = data[:max_samples]

    print(f"编码模型批量评测: {len(data)} 条样本\n")

    results = []
    for i, rec in enumerate(data):
        input_text = rec["input"]
        expected = expected_to_text(rec["output"])
        sample_instruction = str(rec.get("instruction") or instruction).strip() or instruction

        t0 = time.time()
        raw_output = generate(
            model,
            tokenizer,
            input_text,
            instruction=sample_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - t0
        predicted = clean_text_output(raw_output)
        correct = predicted == expected

        result = {
            "index": i,
            "field": extract_field_from_input(input_text),
            "input": input_text,
            "expected": expected,
            "predicted": predicted,
            "raw_output": raw_output,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)

        status = "OK" if correct else "WRONG"
        progress = f"[{i+1}/{len(data)}]"
        print(f"  {progress} {status}  ({elapsed:.1f}s)  {input_text[:60]}")

    report = aggregate_code_report(results)
    print_code_report(report)

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已保存: {report_file}")

    if detail_file:
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        with detail_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"详细结果已保存: {detail_file}")

# ──────────────────────────────────────────────
# 批量评测模式
# ──────────────────────────────────────────────

def batch_mode(
    model, tokenizer,
    test_file: Path,
    report_file: Path | None,
    detail_file: Path | None,
    max_samples: int | None = None,
    *,
    task: str,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    preserve_list_order: bool = False,
) -> None:
    data = json.loads(test_file.read_text(encoding="utf-8"))
    if max_samples:
        data = data[:max_samples]

    print(f"批量评测: {len(data)} 条样本\n")

    results = []
    for i, rec in enumerate(data):
        input_text = rec["input"]
        expected = expected_to_json(rec["output"])
        sample_instruction = str(rec.get("instruction") or instruction).strip() or instruction

        if expected is None:
            raise ValueError(f"第 {i + 1} 条样本 output 不是 JSON 对象或 JSON 字符串")

        t0 = time.time()
        raw_output = generate(
            model,
            tokenizer,
            input_text,
            instruction=sample_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - t0

        predicted = parse_json_output(raw_output)

        result: dict[str, Any] = {
            "index": i,
            "input": input_text,
            "elapsed_s": round(elapsed, 2),
        }

        if predicted is None:
            result["json_parse_fail"] = True
            result["raw_output"] = raw_output
            status = "PARSE_FAIL"
        else:
            result["json_parse_fail"] = False
            schema_errors = validate_task_output(task, predicted)
            result["schema_errors"] = schema_errors
            result["comparison"] = compare_record(
                expected,
                predicted,
                preserve_list_order=preserve_list_order,
            )
            acc = result["comparison"]["field_accuracy"]
            status = "SCHEMA_FAIL" if schema_errors else f"ACC={acc:.0%}"
            if acc < 1.0:
                result["expected"] = expected
                result["predicted"] = predicted

        progress = f"[{i+1}/{len(data)}]"
        print(f"  {progress} {status}  ({elapsed:.1f}s)  {input_text[:60]}")

        results.append(result)

    # 生成汇总报告
    report = aggregate_report(results)

    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)
    print(f"  总样本:          {report['total_records']}")
    print(f"  JSON解析失败:    {report['json_parse_fail']}")
    if task == "material_standard":
        print(f"  v3结构校验失败: {report.get('schema_fail', 0)}")
    print(f"  完美匹配:        {report.get('perfect_records', 0)} / {report.get('evaluated_records', 0)}"
          f" ({report.get('perfect_rate', 0):.1%})")
    print(f"  平均字段准确率:  {report.get('avg_field_accuracy', 0):.1%}")
    print(f"  总字段准确率:    {report.get('overall_field_accuracy', 0):.1%}"
          f" ({report.get('total_correct', 0)}/{report.get('total_fields', 0)})")

    if report.get("top_error_fields"):
        print(f"\n  错误最多的字段:")
        for item in report["top_error_fields"][:10]:
            print(f"    {item['field']}: {item['errors']}/{item['total']}"
                  f" (错误率 {item['error_rate']:.1%})")

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已保存: {report_file}")

    if detail_file:
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        with detail_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"详细结果已保存: {detail_file}")


# ──────────────────────────────────────────────
# 预测标注模式（仅生成预测，不做评测）
# ──────────────────────────────────────────────

def predict_mode(
    model, tokenizer,
    input_file: Path,
    output_file: Path,
    max_samples: int | None = None,
    *,
    task: str,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> None:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if max_samples:
        data = data[:max_samples]

    print(f"预测标注: {len(data)} 条样本\n")

    results = []
    fail_count = 0
    schema_fail_count = 0
    for i, rec in enumerate(data):
        input_text = rec["input"]
        sample_instruction = str(rec.get("instruction") or instruction).strip() or instruction

        t0 = time.time()
        raw_output = generate(
            model,
            tokenizer,
            input_text,
            instruction=sample_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.time() - t0

        if task in JSON_TASKS:
            predicted = parse_json_output(raw_output)
            if predicted is None:
                fail_count += 1
                status = "PARSE_FAIL"
                result = {
                    "input": input_text,
                    "output": {},
                    "_raw": raw_output,
                }
            else:
                schema_errors = validate_task_output(task, predicted)
                if schema_errors:
                    schema_fail_count += 1
                    status = "SCHEMA_FAIL"
                else:
                    status = "OK"
                result = {
                    "input": input_text,
                    "output": predicted,
                }
                if schema_errors:
                    result["_schema_errors"] = schema_errors
        else:
            predicted = clean_text_output(raw_output)
            status = "OK"
            result = {
                "input": input_text,
                "output": predicted,
                "_raw": raw_output,
            }

        # 保留 _meta 信息
        if "_meta" in rec:
            result["_meta"] = rec["_meta"]

        progress = f"[{i+1}/{len(data)}]"
        print(f"  {progress} {status}  ({elapsed:.1f}s)  {input_text[:60]}")

        results.append(result)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n完成: {len(results)} 条, JSON解析失败: {fail_count}, "
        f"结构校验失败: {schema_fail_count}"
    )
    print(f"输出: {output_file}")


def _excel_value(value: Any) -> str:
    """将结构化字段转换为适合写入 Excel 的文本。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _type_excel_columns(predicted: dict[str, Any]) -> dict[str, Any]:
    """展开种类模型结果，便于在 Excel 中直接筛选和比对。"""
    type_result = predicted.get("TYPE")
    if not isinstance(type_result, dict):
        type_result = {}
    geometry = type_result.get("GEOMETRY")
    if not isinstance(geometry, dict):
        geometry = {}

    return {
        "TYPE_CATEGORY": _excel_value(predicted.get("CATEGORY")),
        "TYPE_BODY": _excel_value(type_result.get("BODY")),
        "TYPE_ANGLE": _excel_value(geometry.get("ANGLE")),
        "TYPE_RADIUS": _excel_value(geometry.get("RADIUS")),
        "TYPE_FLANGE_STYLE": _excel_value(type_result.get("FLANGE_STYLE")),
        "TYPE_MANU": _excel_value(type_result.get("MANU")),
        "TYPE_CONN": _excel_value(type_result.get("CONN")),
        "TYPE_SEAL": _excel_value(type_result.get("SEAL")),
    }


def excel_predict_mode(
    model,
    tokenizer,
    *,
    input_file: Path,
    output_file: Path,
    input_column: str,
    sheet_name: str | int = 0,
    max_samples: int | None = None,
    task: str,
    instruction: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> None:
    """读取 Excel 描述列进行预测，并将结果追加到原始列后。"""
    if not input_file.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {input_file}")

    data = pd.read_excel(input_file, sheet_name=sheet_name, dtype=object)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Excel 工作表读取失败，请检查 --excel-sheet")
    if input_column not in data.columns:
        columns = ", ".join(str(column) for column in data.columns)
        raise ValueError(
            f"Excel 中不存在描述列 {input_column!r}。现有列: {columns}"
        )
    if max_samples is not None:
        data = data.head(max_samples).copy()
    else:
        data = data.copy()

    result_columns = [
        "TYPE_模型结果",
        "TYPE_CATEGORY",
        "TYPE_BODY",
        "TYPE_ANGLE",
        "TYPE_RADIUS",
        "TYPE_FLANGE_STYLE",
        "TYPE_MANU",
        "TYPE_CONN",
        "TYPE_SEAL",
        "模型原始输出",
        "模型解析状态",
        "模型耗时_秒",
    ]
    for column in result_columns:
        data[column] = ""

    print(
        f"Excel 预测: {len(data)} 行 | 工作表: {sheet_name!r} | "
        f"描述列: {input_column!r}\n"
    )
    ok_count = 0
    fail_count = 0
    skipped_count = 0

    for position, (row_index, row) in enumerate(data.iterrows(), start=1):
        raw_input = row.get(input_column)
        input_text = "" if pd.isna(raw_input) else str(raw_input).strip()
        if not input_text:
            data.at[row_index, "模型解析状态"] = "SKIPPED_EMPTY"
            skipped_count += 1
            print(f"  [{position}/{len(data)}] SKIPPED_EMPTY")
            continue

        t0 = time.time()
        try:
            raw_output = generate(
                model,
                tokenizer,
                input_text,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.time() - t0
            data.at[row_index, "模型原始输出"] = raw_output
            data.at[row_index, "模型耗时_秒"] = round(elapsed, 3)

            if task in JSON_TASKS:
                predicted = parse_json_output(raw_output)
                if predicted is None:
                    status = "PARSE_FAIL"
                    fail_count += 1
                else:
                    schema_errors = validate_task_output(task, predicted)
                    if schema_errors:
                        status = "SCHEMA_FAIL: " + " | ".join(schema_errors)
                        fail_count += 1
                    else:
                        status = "OK"
                        ok_count += 1
                    data.at[row_index, "TYPE_模型结果"] = json.dumps(
                        predicted, ensure_ascii=False, separators=(",", ":")
                    )
                    if task == "type":
                        for column, value in _type_excel_columns(predicted).items():
                            data.at[row_index, column] = value
            else:
                predicted_text = clean_text_output(raw_output)
                data.at[row_index, "TYPE_模型结果"] = predicted_text
                status = "OK"
                ok_count += 1

            data.at[row_index, "模型解析状态"] = status
        except Exception as exc:
            elapsed = time.time() - t0
            status = "ERROR"
            fail_count += 1
            data.at[row_index, "模型解析状态"] = f"ERROR: {exc}"
            data.at[row_index, "模型耗时_秒"] = round(elapsed, 3)

        print(f"  [{position}/{len(data)}] {status}  ({elapsed:.1f}s)  {input_text[:60]}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(output_file, index=False, engine="openpyxl")
    print(
        f"\n完成: {len(data)} 行, 成功: {ok_count}, "
        f"失败: {fail_count}, 空描述跳过: {skipped_count}"
    )
    print(f"输出: {output_file}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="评测 LoRA 微调模型，支持结构化抽取模型和字段编码模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model", required=True,
        help="底座模型路径（如 /path/to/Qwen3-4B）",
    )
    parser.add_argument(
        "--lora", default=None,
        help="LoRA 权重路径（如 /path/to/checkpoint-260）。不指定则使用底座模型。",
    )
    parser.add_argument(
        "--task",
        choices=["extract", "type", "material_standard", "structural", "code"],
        default="extract",
        help=(
            "评测任务类型。extract=通用结构化抽取；type=种类抽取（自动读取种类提示词文件）；"
            "material_standard=v3 材质规范抽取（自动读取 v3 提示词并校验固定结构）；"
            "structural=尺寸/长度/壁厚/磅级抽取；code=字段编码纯文本"
        ),
    )
    parser.add_argument(
        "--code-field",
        choices=CODE_FIELDS,
        default="TYPE",
        help="code 交互模式下，直接输入原始值时自动使用的字段类型，默认 TYPE。",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="覆盖默认 system instruction。不传时按 task 使用默认 instruction；样本内 instruction 优先级更高。",
    )
    parser.add_argument(
        "--test-file", type=Path, default=None,
        help="测试集 JSON 文件路径。不指定则进入交互模式。",
    )
    parser.add_argument(
        "--excel-file", type=Path, default=None,
        help="Excel 批量预测输入路径。设置后直接进入预测模式。",
    )
    parser.add_argument(
        "--input-column", default="材料描述",
        help="Excel 中的材料描述列名，默认: 材料描述。",
    )
    parser.add_argument(
        "--excel-sheet", default=0,
        help="Excel 工作表名称，默认读取第一张工作表。",
    )
    parser.add_argument(
        "--excel-output", type=Path, default=None,
        help="Excel 预测输出路径，默认生成 <输入文件名>_predicted.xlsx。",
    )
    parser.add_argument(
        "--report-file", type=Path, default=None,
        help="评测报告输出路径（JSON 格式）",
    )
    parser.add_argument(
        "--detail-file", type=Path, default=None,
        help="逐条评测详情输出路径（JSONL 格式）",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="最多评测条数（用于快速测试）",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="预测模式：仅生成模型预测结果，不做评测对比。",
    )
    parser.add_argument(
        "--predict-output", type=Path, default=None,
        help="预测模式的输出文件路径",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="最大生成 token 数。extract 默认 512，code 默认 64。",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="解码温度，默认 0.0（与 MLX 服务一致，贪心）",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="top_p，默认 1.0（与 MLX 服务一致）",
    )

    args = parser.parse_args()
    if args.test_file and args.excel_file:
        parser.error("--test-file 和 --excel-file 不能同时使用")
    instruction, instruction_path = resolve_instruction(args.task, args.instruction)
    if instruction_path:
        print(f"{args.task} 模型提示词: {instruction_path}")
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 64 if args.task == "code" else 512

    model, tokenizer = load_model(args.base_model, args.lora)

    if args.excel_file:
        excel_sheet: str | int = args.excel_sheet
        if isinstance(excel_sheet, str) and excel_sheet.isdigit():
            excel_sheet = int(excel_sheet)
        excel_output = args.excel_output or args.excel_file.with_name(
            args.excel_file.stem + "_predicted.xlsx"
        )
        excel_predict_mode(
            model,
            tokenizer,
            input_file=args.excel_file,
            output_file=excel_output,
            input_column=args.input_column,
            sheet_name=excel_sheet,
            max_samples=args.max_samples,
            task=args.task,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.predict and args.test_file:
        out = args.predict_output or args.test_file.with_name(
            args.test_file.stem + "_predicted.json"
        )
        predict_mode(
            model, tokenizer,
            input_file=args.test_file,
            output_file=out,
            max_samples=args.max_samples,
            task=args.task,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.test_file:
        if args.task == "code":
            batch_code_mode(
                model, tokenizer,
                test_file=args.test_file,
                report_file=args.report_file,
                detail_file=args.detail_file,
                max_samples=args.max_samples,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            batch_mode(
                model, tokenizer,
                test_file=args.test_file,
                report_file=args.report_file,
                detail_file=args.detail_file,
                max_samples=args.max_samples,
                task=args.task,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                preserve_list_order=args.task in {"structural", "material_standard"},
            )
    else:
        interactive_mode(
            model,
            tokenizer,
            task=args.task,
            instruction=instruction,
            code_field=args.code_field,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
