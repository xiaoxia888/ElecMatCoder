# -*- coding: utf-8 -*-
"""
Qwen3/Qwen3.5 二阶段 DPO 训练脚本

用途:
    在一阶段 SFT 之后，继续进行偏好优化，重点压制:
    - 幻觉
    - 边界缺失
    - schema 不稳定
    - STANDARD_* 拆分和绑定错误

推荐做法:
    1. 先完成一阶段 SFT
    2. 将 SFT merged 模型路径写入 config_dpo.yaml 的 model.name_or_path
    3. 准备 prompt/chosen/rejected 偏好数据
    4. 运行本脚本进行 DPO 微调

示例:
    python -m apps.trainer.qwen3_finetune.train_dpo \
        --config apps/trainer/qwen3_finetune/config_dpo.yaml

    python -m apps.trainer.qwen3_finetune.train_dpo \
        --config apps/trainer/qwen3_finetune/config_dpo.yaml \
        --beta 0.2 \
        --lr 5e-6
"""

import argparse
import inspect
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_ner.predictor import SYSTEM_PROMPT as NER_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> dict:
    default_config = Path(__file__).parent / "config_dpo.yaml"
    cfg_path = Path(config_path) if config_path else default_config
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_optional_path(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _resolve_model_name_or_path(value: str) -> str:
    """
    兼容两类输入:
    - HuggingFace 模型名，如 Qwen/Qwen3-4B
    - 相对/绝对本地模型路径
    """
    if not value:
        return value
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = PROJECT_ROOT / value
    if candidate.exists():
        return str(candidate)
    return value


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> tuple[dict, dict]:
    """按目标可调用对象签名过滤 kwargs，返回 (supported, unsupported)。"""
    try:
        sig = inspect.signature(callable_obj)
        supported_names = set(sig.parameters.keys())
    except (TypeError, ValueError):
        return dict(kwargs), {}

    supported = {}
    unsupported = {}
    for key, value in kwargs.items():
        if key in supported_names:
            supported[key] = value
        else:
            unsupported[key] = value
    return supported, unsupported


def _json_to_text(value, append_eos_token: bool, eos_token: str | None) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    if append_eos_token and eos_token and not text.endswith(eos_token):
        text += eos_token
    return text


def _build_prompt_text(
    tokenizer,
    system_prompt: str,
    user_text: str,
    use_chat_template: bool = True,
) -> str:
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n用户输入:\n{user_text}\n\n请直接输出 JSON："


def load_preference_dataset(
    file_path: Path,
    tokenizer,
    system_prompt: str,
    append_eos_token: bool = True,
    use_chat_template: bool = True,
) -> Dataset:
    """
    支持两种原始 JSONL 格式:

    1) 推荐格式:
        {"input": "...", "chosen": {...}, "rejected": {...}, "meta": {...}}

    2) 已展开格式:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            if "prompt" in sample:
                prompt = str(sample["prompt"])
            elif "input" in sample:
                prompt = _build_prompt_text(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    user_text=str(sample["input"]),
                    use_chat_template=use_chat_template,
                )
            else:
                raise ValueError(f"{file_path} 第 {line_num} 行缺少 prompt 或 input 字段")

            if "chosen" not in sample or "rejected" not in sample:
                raise ValueError(f"{file_path} 第 {line_num} 行缺少 chosen/rejected 字段")

            chosen = _json_to_text(
                sample["chosen"],
                append_eos_token=append_eos_token,
                eos_token=tokenizer.eos_token,
            )
            rejected = _json_to_text(
                sample["rejected"],
                append_eos_token=append_eos_token,
                eos_token=tokenizer.eos_token,
            )

            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    logger.info(f"加载 {len(rows)} 条偏好样本: {file_path}")
    return Dataset.from_list(rows)


def _load_raw_preference_rows(file_path: Path) -> list[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix.lower() == ".json":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"{file_path} 应为数组 JSON")
            rows.extend(payload)
        else:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{file_path} 第 {line_num} 行 JSON 解析失败: {exc}") from exc
    return rows


def _write_jsonl_rows(file_path: Path, rows: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_preference_split(data_cfg: dict, run_dir: Path) -> tuple[Path, Path]:
    train_file = _normalize_optional_path(data_cfg.get("train_file"))
    val_file = _normalize_optional_path(data_cfg.get("val_file"))

    if train_file:
        train_path = _resolve_path(train_file)
        val_path = _resolve_path(val_file) if val_file else run_dir / "auto_val.jsonl"
        return train_path, val_path

    preference_source = _normalize_optional_path(data_cfg.get("preference_source"))
    if not preference_source:
        raise ValueError("DPO 配置缺少 train_file，且未提供 preference_source")

    source_path = _resolve_path(preference_source)
    if not source_path.exists():
        raise FileNotFoundError(f"DPO 偏好源文件不存在: {source_path}")

    rows = _load_raw_preference_rows(source_path)
    if len(rows) < 2:
        raise ValueError(f"DPO 偏好源文件样本过少: {source_path}")

    train_ratio = float(data_cfg.get("train_ratio", 0.9))
    train_ratio = max(0.5, min(0.99, train_ratio))
    seed = int(data_cfg.get("split_seed", 42))

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    split_idx = max(1, min(len(shuffled) - 1, split_idx))
    train_rows = shuffled[:split_idx]
    val_rows = shuffled[split_idx:]

    train_path = run_dir / "auto_train.jsonl"
    val_path = run_dir / "auto_val.jsonl"
    _write_jsonl_rows(train_path, train_rows)
    _write_jsonl_rows(val_path, val_rows)

    logger.info(f"未指定 train/val，已从 {source_path} 自动切分偏好数据")
    logger.info(f"  train: {train_path} ({len(train_rows)} 条)")
    logger.info(f"  val:   {val_path} ({len(val_rows)} 条)")
    return train_path, val_path


def build_model_kwargs(model_cfg: dict, use_quantize: bool) -> dict:
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
    }

    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl and attn_impl != "flash_attention_2":
        model_kwargs["attn_implementation"] = attn_impl

    if use_quantize:
        logger.info("启用 4-bit QLoRA 量化")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("使用 Flash Attention 2")
        except ImportError:
            logger.warning("flash_attn 未安装, 回退到 sdpa")
            model_kwargs["attn_implementation"] = "sdpa"

    return model_kwargs


def main():
    parser = argparse.ArgumentParser(description="Qwen3/Qwen3.5 二阶段 DPO 微调")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--no_quantize", action="store_true", help="不使用4bit量化")
    parser.add_argument("--resume_from", type=str, default=None, help="从 checkpoint 恢复")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    dpo_cfg = cfg["dpo"]

    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    if args.lr:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size:
        train_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum:
        train_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.beta is not None:
        dpo_cfg["beta"] = args.beta
    if args.lora_r:
        lora_cfg["r"] = args.lora_r
        lora_cfg["lora_alpha"] = args.lora_r * 2

    output_dir = PROJECT_ROOT / train_cfg["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"配置已保存: {run_dir / 'config.yaml'}")

    model_name_or_path = _resolve_model_name_or_path(str(model_cfg["name_or_path"]))
    ref_model_name_or_path = _resolve_model_name_or_path(
        str(model_cfg.get("reference_name_or_path", model_name_or_path))
    )
    system_prompt = data_cfg.get("system_prompt") or NER_SYSTEM_PROMPT

    logger.info(f"加载 Tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_quantize = not args.no_quantize
    model_kwargs = build_model_kwargs(model_cfg, use_quantize=use_quantize)

    logger.info(f"加载可训练模型: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if use_quantize:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )
    model.config.use_cache = False

    logger.info(f"应用 DPO LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
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

    logger.info(f"加载参考模型: {ref_model_name_or_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name_or_path, **model_kwargs)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_path, val_path = prepare_preference_split(data_cfg, run_dir)
    if not train_path.exists():
        logger.error(f"DPO 训练数据不存在: {train_path}")
        logger.error("请先准备偏好数据，或使用 data/pipe/dpo_demo/ 下的示例数据")
        sys.exit(1)

    train_dataset = load_preference_dataset(
        file_path=train_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        append_eos_token=data_cfg.get("append_eos_token", True),
        use_chat_template=data_cfg.get("use_chat_template", True),
    )
    eval_dataset = None
    if val_path.exists():
        eval_dataset = load_preference_dataset(
            file_path=val_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            append_eos_token=data_cfg.get("append_eos_token", True),
            use_chat_template=data_cfg.get("use_chat_template", True),
        )

    sample = train_dataset[0]
    sample_total = sample["prompt"] + sample["chosen"]
    sample_tokens = tokenizer(sample_total, return_tensors="pt")
    logger.info(f"样本 token 长度示例: {sample_tokens['input_ids'].shape[1]}")
    logger.info(f"样本 prompt 前 180 字符: {sample['prompt'][:180]}")
    logger.info(f"样本 chosen 前 180 字符: {sample['chosen'][:180]}")
    logger.info(f"样本 rejected 前 180 字符: {sample['rejected'][:180]}")

    dpo_arg_candidates = {
        "output_dir": str(run_dir),
        "num_train_epochs": train_cfg["num_train_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 2),
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.05),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "eval_steps": train_cfg.get("eval_steps", 100),
        "save_steps": train_cfg.get("save_steps", 100),
        "save_total_limit": train_cfg.get("save_total_limit", 3),
        "eval_strategy": train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        "bf16": train_cfg.get("bf16", True),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 2),
        "report_to": train_cfg.get("report_to", "none"),
        "seed": train_cfg.get("seed", 42),
        "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
        "remove_unused_columns": False,
        "beta": dpo_cfg.get("beta", 0.1),
        "loss_type": dpo_cfg.get("loss_type", "sigmoid"),
        "max_length": model_cfg.get("max_seq_length", 768),
        "max_prompt_length": dpo_cfg.get("max_prompt_length", 512),
        "max_completion_length": dpo_cfg.get("max_completion_length", 256),
    }
    supported_dpo_args, unsupported_dpo_args = _filter_supported_kwargs(DPOConfig.__init__, dpo_arg_candidates)
    if unsupported_dpo_args:
        logger.info(f"DPOConfig 不支持这些参数，稍后尝试传给 DPOTrainer: {sorted(unsupported_dpo_args.keys())}")
    dpo_args = DPOConfig(**supported_dpo_args)

    trainer_candidates = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        **unsupported_dpo_args,
    }
    supported_trainer_args, still_unsupported = _filter_supported_kwargs(DPOTrainer.__init__, trainer_candidates)
    if still_unsupported:
        logger.warning(f"当前 DPOTrainer 也不支持这些参数，将忽略: {sorted(still_unsupported.keys())}")

    trainer = DPOTrainer(**supported_trainer_args)

    logger.info("=" * 60)
    logger.info("开始 DPO 训练")
    logger.info(f"  训练模型: {model_name_or_path}")
    logger.info(f"  参考模型: {ref_model_name_or_path}")
    logger.info(f"  LoRA r: {lora_cfg['r']}, alpha: {lora_cfg['lora_alpha']}")
    logger.info(f"  训练集: {len(train_dataset)} 条")
    logger.info(f"  验证集: {len(eval_dataset) if eval_dataset else 0} 条")
    logger.info(f"  Epochs: {train_cfg['num_train_epochs']}")
    logger.info(f"  Batch size: {train_cfg['per_device_train_batch_size']}")
    logger.info(f"  Gradient accum: {train_cfg['gradient_accumulation_steps']}")
    logger.info(
        f"  有效 batch size: "
        f"{train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}"
    )
    logger.info(f"  学习率: {train_cfg['learning_rate']}")
    logger.info(f"  Beta: {dpo_cfg.get('beta', 0.1)}")
    logger.info(f"  输出目录: {run_dir}")
    logger.info("=" * 60)

    if args.resume_from:
        logger.info(f"从 checkpoint 恢复: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    logger.info("保存 DPO LoRA 权重...")
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if eval_dataset:
        logger.info("最终评估...")
        eval_result = trainer.evaluate()
        eval_save = run_dir / "eval_results.json"
        with open(eval_save, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
        logger.info(f"验证结果已保存: {eval_save}")

    logger.info(f"DPO 训练完成! 模型已保存到: {final_dir}")


if __name__ == "__main__":
    main()
