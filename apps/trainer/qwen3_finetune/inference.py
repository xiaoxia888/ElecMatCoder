# -*- coding: utf-8 -*-
"""
Qwen3-4B 微调模型推理 & 评估脚本

支持三种模式:
    1. 交互式推理 (默认)
    2. 批量评估 (--eval)
    3. 单条推理 (--text)

使用方法:
    # 交互式
    python -m apps.trainer.qwen3_finetune.inference --model_path outputs/qwen3_finetune/run_xxx/final

    # 单条
    python -m apps.trainer.qwen3_finetune.inference --model_path outputs/qwen3_finetune/run_xxx/final \
        --text "90度弯头 DN50 SCH40 A234 WPB ASME B16.9"

    # 批量评估
    python -m apps.trainer.qwen3_finetune.inference --model_path outputs/qwen3_finetune/run_xxx/final \
        --eval --eval_file data/pipe/qwen3_finetune/val.jsonl --eval_samples 100

    # 使用合并后的模型
    python -m apps.trainer.qwen3_finetune.inference --model_path outputs/qwen3_finetune/merged --merged
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.llm_ner.predictor import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class PipeExtractor:
    """管道材料信息提取器"""

    def __init__(self, model_path: str, merged: bool = False, device: str = "auto"):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        is_mps = (
            device == "mps"
            or (device == "auto" and torch.backends.mps.is_available())
        )
        dtype = torch.float16 if is_mps else torch.bfloat16

        logger.info(f"加载 Tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"加载模型 (merged={merged}, dtype={dtype}, mps={is_mps})...")
        if merged:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
        else:
            base_config_path = self.model_path / "adapter_config.json"
            if base_config_path.exists():
                with open(base_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B")
            else:
                base_model_name = "Qwen/Qwen3-4B"

            logger.info(f"  基座模型: {base_model_name}")
            # MPS: 先在 CPU 上加载并合并 LoRA，再转到 MPS
            load_device = "cpu" if is_mps else device
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map=load_device,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model = self.model.merge_and_unload()
            if is_mps:
                logger.info("  合并完成，转移到 MPS...")
                self.model = self.model.to("mps")

        self.model.eval()
        logger.info("模型加载完成")

    def extract(self, text: str, max_new_tokens: int = 256, temperature: float = 0.1) -> dict:
        """从管道材料描述中提取结构化信息"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else 1.0,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return self._parse_response(response)

    def extract_with_timing(self, text: str, **kwargs) -> tuple[dict, float, str]:
        """提取并返回 (结果, 耗时秒, 原始响应)"""
        start = time.perf_counter()
        result = self.extract(text, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed, result.get("_raw", "")

    def _parse_response(self, response: str) -> dict:
        """解析模型输出的 JSON"""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                result["_raw"] = response
                return result
        except json.JSONDecodeError:
            pass

        # 尝试提取 {} 块
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            try:
                result = json.loads(cleaned[start_idx:end_idx + 1])
                if isinstance(result, dict):
                    result["_raw"] = response
                    return result
            except json.JSONDecodeError:
                pass

        return {"_parse_error": True, "_raw": response}


def _iter_output_fields(payload: dict) -> list[str]:
    """按模型输出动态遍历字段，忽略内部元字段。"""
    return [field for field in payload.keys() if not str(field).startswith("_")]


def evaluate(extractor: PipeExtractor, eval_file: str, max_samples: int = None):
    """批量评估模型性能"""
    eval_path = Path(eval_file)
    if not eval_path.is_absolute():
        eval_path = PROJECT_ROOT / eval_path

    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if max_samples and max_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    logger.info(f"评估样本数: {len(samples)}")

    total_fields = 0
    correct_fields = 0
    field_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    parse_errors = 0
    latencies = []
    error_cases = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        input_text = messages[1]["content"]
        expected_str = messages[2]["content"]

        try:
            expected = json.loads(expected_str)
        except json.JSONDecodeError:
            continue

        start = time.perf_counter()
        predicted = extractor.extract(input_text)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        if predicted.get("_parse_error"):
            parse_errors += 1
            error_cases.append({
                "input": input_text,
                "expected": expected,
                "raw_output": predicted.get("_raw", ""),
            })
            for field in _iter_output_fields(expected):
                if field in expected:
                    field_stats[field]["fn"] += 1
                    total_fields += 1
            continue

        compare_fields = set(_iter_output_fields(expected)) | set(_iter_output_fields(predicted))
        for field in compare_fields:
            exp_val = expected.get(field)
            pred_val = predicted.get(field)

            if exp_val is not None:
                total_fields += 1
                if pred_val is not None:
                    exp_norm = _normalize(exp_val)
                    pred_norm = _normalize(pred_val)
                    if exp_norm == pred_norm:
                        correct_fields += 1
                        field_stats[field]["tp"] += 1
                    else:
                        field_stats[field]["fp"] += 1
                        field_stats[field]["fn"] += 1
                        error_cases.append({
                            "input": input_text,
                            "field": field,
                            "expected": exp_val,
                            "predicted": pred_val,
                        })
                else:
                    field_stats[field]["fn"] += 1
            elif pred_val is not None:
                field_stats[field]["fp"] += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  进度: {i+1}/{len(samples)}, 平均延迟: {sum(latencies)/len(latencies):.2f}s")

    # 汇总结果
    accuracy = correct_fields / total_fields if total_fields > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"样本数:        {len(samples)}")
    print(f"字段总准确率:  {accuracy:.2%} ({correct_fields}/{total_fields})")
    print(f"JSON解析失败:  {parse_errors} ({parse_errors/len(samples)*100:.1f}%)")
    print(f"平均延迟:      {avg_latency:.2f}s")
    print(f"P95 延迟:      {p95_latency:.2f}s")

    print(f"\n{'字段':<18} {'精确率':>8} {'召回率':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)
    for field in sorted(field_stats.keys()):
        stats = field_stats[field]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{field:<18} {precision:>8.2%} {recall:>8.2%} {f1:>8.2%} {tp:>6} {fp:>6} {fn:>6}")

    results = {
        "accuracy": accuracy,
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "parse_errors": parse_errors,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "error_cases": error_cases[:50],
    }

    output_path = PROJECT_ROOT / "outputs" / "qwen3_finetune" / "eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细报告已保存: {output_path}")

    return results


def _normalize(val) -> str:
    """标准化值用于比较"""
    if isinstance(val, list):
        normalized_items = []
        for v in val:
            if isinstance(v, dict):
                normalized_items.append(json.dumps(v, ensure_ascii=False, sort_keys=True))
            else:
                normalized_items.append(str(v).strip().upper())
        return json.dumps(sorted(normalized_items), ensure_ascii=False)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, sort_keys=True)
    return str(val).strip().upper()


def interactive(extractor: PipeExtractor):
    """交互式推理"""
    print("\n" + "=" * 60)
    print("管道材料信息提取 - 交互模式")
    print("输入管道材料描述，按回车提取。输入 'q' 退出。")
    print("=" * 60)

    while True:
        try:
            text = input("\n输入> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue

        result, elapsed, raw = extractor.extract_with_timing(text)

        if result.get("_parse_error"):
            print(f"  [解析失败] 原始输出: {raw}")
        else:
            display = {k: v for k, v in result.items() if not k.startswith("_")}
            print(f"  结果: {json.dumps(display, ensure_ascii=False, indent=2)}")
        print(f"  耗时: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B 推理/评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径 (LoRA adapter 或 merged)")
    parser.add_argument("--merged", action="store_true", help="是否为合并后的模型")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--text", type=str, default=None, help="单条推理文本")
    parser.add_argument("--eval", action="store_true", help="批量评估模式")
    parser.add_argument("--eval_file", type=str, default="data/pipe/qwen3_finetune/val.jsonl")
    parser.add_argument("--eval_samples", type=int, default=None, help="评估样本数 (None=全部)")

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    extractor = PipeExtractor(args.model_path, merged=args.merged, device=args.device)

    if args.text:
        result, elapsed, raw = extractor.extract_with_timing(
            args.text, temperature=args.temperature, max_new_tokens=args.max_new_tokens
        )
        if result.get("_parse_error"):
            print(f"解析失败，原始输出: {raw}")
        else:
            display = {k: v for k, v in result.items() if not k.startswith("_")}
            print(json.dumps(display, ensure_ascii=False, indent=2))
        print(f"耗时: {elapsed:.2f}s")

    elif args.eval:
        evaluate(extractor, args.eval_file, args.eval_samples)

    else:
        interactive(extractor)


if __name__ == "__main__":
    main()
