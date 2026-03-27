# -*- coding: utf-8 -*-
"""
Qwen3-4B LoRA 微调训练脚本

管道材料描述 → 结构化 JSON 提取

使用方法:
    # 1. 先准备数据
    python -m apps.trainer.qwen3_finetune.prepare_data

    # 2. 开始训练 (使用默认配置)
    python -m apps.trainer.qwen3_finetune.train

    # 3. 指定配置 / 覆盖参数
    python -m apps.trainer.qwen3_finetune.train --config apps/trainer/qwen3_finetune/config.yaml
    python -m apps.trainer.qwen3_finetune.train --epochs 5 --lr 1e-4 --batch_size 4
"""

import os
import sys
import json
import argparse
import inspect
import logging
from pathlib import Path
from datetime import datetime

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.qwen3_finetune.training_curves import export_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_sft_config_kwargs(eval_dataset, model_cfg: dict, train_cfg: dict, run_dir: Path) -> dict:
    kwargs = {
        "output_dir": str(run_dir),
        "num_train_epochs": train_cfg["num_train_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.05),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "eval_steps": train_cfg.get("eval_steps", 200),
        "save_steps": train_cfg.get("save_steps", 200),
        "save_total_limit": train_cfg.get("save_total_limit", 3),
        "eval_strategy": train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        "bf16": train_cfg.get("bf16", True),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 4),
        "report_to": train_cfg.get("report_to", "none"),
        "seed": train_cfg.get("seed", 42),
        "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
        "max_seq_length": int(model_cfg.get("max_seq_length", 512)),
        "dataset_text_field": "text",
        "packing": False,
    }

    supported = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    if "max_seq_length" not in supported and "max_length" in supported:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        logger.warning(f"SFTConfig 不支持以下参数，已自动忽略: {dropped}")
    return filtered


class CompletionOnlyCollator:
    """
    兼容任意 TRL 版本的 completion-only collator。
    仅对 assistant 起始标记之后的 token 计算 loss，避免依赖 trl 内部 API 变动。
    """

    def __init__(self, tokenizer, response_template_ids: list[int]):
        self.tokenizer = tokenizer
        self.response_template_ids = response_template_ids
        self.pad_token_id = tokenizer.pad_token_id

    @staticmethod
    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.tolist()
        return list(x)

    @staticmethod
    def _find_subseq(sequence: list[int], subseq: list[int]) -> int:
        if not subseq or len(subseq) > len(sequence):
            return -1
        n, m = len(sequence), len(subseq)
        for i in range(n - m + 1):
            if sequence[i:i + m] == subseq:
                return i
        return -1

    def __call__(self, features: list[dict]) -> dict:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        max_len = 0

        for feat in features:
            ids = self._to_list(feat["input_ids"])
            mask = self._to_list(feat.get("attention_mask", [1] * len(ids)))
            max_len = max(max_len, len(ids))

            # 默认不计算 loss
            labels = [-100] * len(ids)
            pos = self._find_subseq(ids, self.response_template_ids)
            if pos != -1:
                start = pos + len(self.response_template_ids)
                for i in range(start, len(ids)):
                    if mask[i] == 1:
                        labels[i] = ids[i]

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labels)

        # 手工 padding，避免依赖外部 collator API
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            if pad_len > 0:
                input_ids_list[i].extend([self.pad_token_id] * pad_len)
                attention_mask_list[i].extend([0] * pad_len)
                labels_list[i].extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def load_config(config_path: str | None = None) -> dict:
    default_config = Path(__file__).parent / "config.yaml"
    cfg_path = Path(config_path) if config_path else default_config
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(file_path: Path, tokenizer) -> Dataset:
    """加载数据，使用 apply_chat_template 正确处理特殊 token"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            text = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append({"text": text})
    logger.info(f"加载 {len(texts)} 条样本: {file_path}")
    return Dataset.from_list(texts)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B LoRA 微调")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--no_quantize", action="store_true", help="不使用4bit量化")
    parser.add_argument("--resume_from", type=str, default=None, help="从checkpoint恢复")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    if args.lr:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size:
        train_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum:
        train_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.lora_r:
        lora_cfg["r"] = args.lora_r
        lora_cfg["lora_alpha"] = args.lora_r * 2

    output_dir = PROJECT_ROOT / train_cfg["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"配置已保存: {config_save_path}")

    # =========== 加载 Tokenizer ===========
    logger.info(f"加载 Tokenizer: {model_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # =========== 加载模型 ===========
    logger.info(f"加载模型: {model_cfg['name_or_path']}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
    }

    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl and attn_impl != "flash_attention_2":
        model_kwargs["attn_implementation"] = attn_impl

    use_quantize = not args.no_quantize
    if use_quantize:
        logger.info("启用 4-bit QLoRA 量化")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        logger.info("全精度加载 (不使用量化)")

    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("使用 Flash Attention 2")
        except ImportError:
            logger.warning("flash_attn 未安装, 回退到 sdpa")
            model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        **model_kwargs,
    )

    if use_quantize:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
        )

    model.config.use_cache = False

    # =========== LoRA 配置 ===========
    logger.info(f"应用 LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias=lora_cfg.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # =========== 加载数据 ===========
    train_path = PROJECT_ROOT / data_cfg["train_file"]
    val_path = PROJECT_ROOT / data_cfg["val_file"]

    if not train_path.exists():
        logger.error(f"训练数据不存在: {train_path}")
        logger.error("请先运行: python -m apps.trainer.qwen3_finetune.prepare_data")
        sys.exit(1)

    train_dataset = load_dataset_from_jsonl(train_path, tokenizer)
    eval_dataset = load_dataset_from_jsonl(val_path, tokenizer) if val_path.exists() else None

    # =========== 数据统计 & 调试 ===========
    sample_text = train_dataset[0]["text"]
    sample_tokens = tokenizer(sample_text, return_tensors="pt")
    logger.info(f"样本 token 长度示例: {sample_tokens['input_ids'].shape[1]}")
    logger.info(f"样本文本前 200 字符: {sample_text[:200]}")

    all_lengths = []
    for i in range(min(500, len(train_dataset))):
        tokens = tokenizer(train_dataset[i]["text"], return_tensors="pt")
        all_lengths.append(tokens["input_ids"].shape[1])

    import statistics
    max_seq_length = int(model_cfg.get("max_seq_length", 512))
    over_limit = [length for length in all_lengths if length > max_seq_length]
    over_limit_ratio = (len(over_limit) / len(all_lengths)) if all_lengths else 0.0
    logger.info(
        f"Token 长度统计 (前 {len(all_lengths)} 条): "
        f"mean={statistics.mean(all_lengths):.0f}, "
        f"max={max(all_lengths)}, "
        f"p95={sorted(all_lengths)[int(len(all_lengths)*0.95)]}"
    )
    logger.info(
        f"max_seq_length={max_seq_length}, "
        f"超长样本={len(over_limit)}/{len(all_lengths)} "
        f"({over_limit_ratio:.1%})"
    )
    if over_limit:
        logger.warning(
            f"存在长度超过 max_seq_length 的样本，最长 {max(over_limit)} token，"
            "训练时会被截断。"
        )

    # =========== Completion-only Data Collator ===========
    # 找到 apply_chat_template 生成的 assistant 响应起始标记
    # 用单条样本 debug 确认 response_template token 能被正确匹配
    probe_msgs = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "OK"},
    ]
    probe_text = tokenizer.apply_chat_template(probe_msgs, tokenize=False, add_generation_prompt=False)
    logger.info(f"Chat template probe: {repr(probe_text)}")

    # 从 probe 文本中提取 assistant 起始标记
    assistant_marker = "<|im_start|>assistant\n"
    marker_pos = probe_text.find(assistant_marker)
    if marker_pos == -1:
        assistant_marker = "<|im_start|>assistant"
        marker_pos = probe_text.find(assistant_marker)
    logger.info(f"Response template: {repr(assistant_marker)}, found at pos {marker_pos}")

    probe_ids = tokenizer.encode(probe_text, add_special_tokens=False)
    template_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    logger.info(f"Template token IDs: {template_ids}")
    logger.info(f"Full probe IDs: {probe_ids}")

    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template_ids=template_ids,
    )

    # 验证: collator 能否正确 mask
    test_batch = tokenizer([probe_text], return_tensors="pt", padding=True)
    test_result = collator([{k: v[0] for k, v in test_batch.items()}])
    n_valid = (test_result["labels"][0] != -100).sum().item()
    logger.info(f"Collator 验证: {n_valid} 个 token 参与 loss (应 > 0)")
    if n_valid == 0:
        logger.warning("Collator 未能匹配 response template! 回退到全序列 loss")
        collator = None

    # =========== 训练参数 ===========
    sft_config = SFTConfig(
        **_build_sft_config_kwargs(
            eval_dataset=eval_dataset,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            run_dir=run_dir,
        )
    )

    # =========== 训练 ===========
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    if collator is not None:
        trainer_kwargs["data_collator"] = collator
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  模型: {model_cfg['name_or_path']}")
    logger.info(f"  LoRA r: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    logger.info(f"  训练集: {len(train_dataset)} 条")
    logger.info(f"  验证集: {len(eval_dataset) if eval_dataset else 0} 条")
    logger.info(f"  Epochs: {train_cfg['num_train_epochs']}")
    logger.info(f"  Batch size: {train_cfg['per_device_train_batch_size']}")
    logger.info(f"  Gradient accum: {train_cfg['gradient_accumulation_steps']}")
    effective_bs = train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']
    logger.info(f"  有效 batch size: {effective_bs}")
    logger.info(f"  学习率: {train_cfg['learning_rate']}")
    logger.info(f"  输出目录: {run_dir}")
    logger.info("=" * 60)

    if args.resume_from:
        logger.info(f"从 checkpoint 恢复: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # =========== 保存 ===========
    logger.info("保存 LoRA 权重...")
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    try:
        curve_artifacts = export_training_curves(trainer.state.log_history, run_dir)
        logger.info("训练曲线已导出:")
        for name, path in curve_artifacts.items():
            logger.info(f"  {name}: {path}")
    except Exception as exc:
        logger.warning(f"训练曲线导出失败，已跳过，不影响模型保存: {exc}")

    if eval_dataset:
        logger.info("最终评估...")
        eval_result = trainer.evaluate()
        logger.info(f"验证集 loss: {eval_result.get('eval_loss', 'N/A'):.4f}")
        eval_save = run_dir / "eval_results.json"
        with open(eval_save, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)

    logger.info(f"训练完成! 模型已保存到: {final_dir}")
    logger.info(f"下一步: python -m apps.trainer.qwen3_finetune.inference --model_path {final_dir}")


if __name__ == "__main__":
    main()
