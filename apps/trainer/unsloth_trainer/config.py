from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent


def _detect_project_root(package_dir: Path = PACKAGE_DIR) -> Path:
    """Support both the repository layout and /home/waas/unsloth_trainer."""
    for candidate in package_dir.parents:
        if (candidate / "apps" / "trainer" / "unsloth_trainer").is_dir():
            return candidate
    return package_dir.parent


PROJECT_ROOT = _detect_project_root()
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"


DEFAULTS: dict[str, Any] = {
    "runtime": {
        "studio_disabled": True,
        "compile_disable": True,
        "disable_fast_generation": True,
        "disable_auto_updates": True,
    },
    "model": {
        "max_seq_length": 1024,
        "dtype": None,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
        "fast_inference": False,
        "offload_embedding": False,
        "trust_remote_code": True,
    },
    "lora": {
        "enabled": True,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_gradient_checkpointing": "unsloth",
        "use_rslora": False,
        "random_state": 2026,
    },
    "data": {
        "validation_file": None,
        "format": "alpaca",
        "num_proc": 8,
        "columns": {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
            "messages": "messages",
            "text": "text",
        },
        "system_prompt": None,
        "input_separator": "\n",
        "enable_thinking": False,
        "train_on_responses_only": True,
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "training": {
        "output_dir": "outputs/unsloth_qwen35",
        "append_timestamp": True,
        "num_train_epochs": 2.0,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2.0e-5,
        "weight_decay": 0.01,
        "warmup_fraction": 0.03,
        "warmup_steps": "auto",
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "bf16": True,
        "fp16": False,
        "logging_steps": 20,
        "logging_dir": "tensorboard",
        "logging_nan_inf_filter": False,
        "eval_strategy": "steps",
        "eval_steps": 200,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "max_grad_norm": 1.0,
        "packing": False,
        "dataloader_num_workers": 4,
        "seed": 2026,
        "data_seed": 2026,
        "report_to": "tensorboard",
        "resume_from_checkpoint": "auto",
    },
    "save": {
        "adapter_subdir": "final_adapter",
        "plot_training_curves": True,
        "merged_16bit": False,
        "merged_subdir": "merged_16bit",
        "gguf": False,
        "gguf_subdir": "gguf",
        "gguf_quantization_method": "q4_k_m",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(value: str | Path | None, config_path: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    # Repository-relative paths make the same YAML portable between workstations.
    project_path = PROJECT_ROOT / path
    config_path_candidate = config_path.parent / path
    if project_path.exists() or not config_path_candidate.exists():
        return project_path
    return config_path_candidate


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 顶层必须是对象: {path}")
    return raw


def _apply_finetuning_method(config: dict[str, Any], method: Any) -> None:
    """Translate one user-facing method into the low-level Unsloth switches."""
    if method is None:
        return
    if not isinstance(method, str):
        raise ValueError("finetuning_method 必须是 lora、qlora 或 full")

    normalized = method.strip().lower()
    if normalized not in {"lora", "qlora", "full"}:
        raise ValueError(
            f"finetuning_method 不支持 {method!r}，只能是 lora、qlora 或 full"
        )

    model = config["model"]
    lora = config["lora"]
    training = config["training"]

    # Qwen3.5 的 LoRA 和全参训练默认使用 BF16；QLoRA 仅将
    # 冻结的基座权重改为 4-bit，计算精度仍保持 BF16。
    model["load_in_4bit"] = normalized == "qlora"
    model["load_in_8bit"] = False
    model["load_in_16bit"] = normalized != "qlora"
    model["full_finetuning"] = normalized == "full"
    lora["enabled"] = normalized != "full"
    training["bf16"] = True
    training["fp16"] = False
    config["finetuning_method"] = normalized


def _apply_simple_fields(config: dict[str, Any], task_config: dict[str, Any]) -> None:
    """Map the small user-facing config surface to the internal schema."""
    mappings = {
        "base_model": ("model", "name_or_path"),
        "train_dataset": ("data", "train_file"),
        "validation_dataset": ("data", "validation_file"),
        "output_dir": ("training", "output_dir"),
        # Optional advanced overrides remain flat and discoverable when needed.
        "epochs": ("training", "num_train_epochs"),
        "learning_rate": ("training", "learning_rate"),
        "max_seq_length": ("model", "max_seq_length"),
        "batch_size": ("training", "per_device_train_batch_size"),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
    }
    for source, (section, target) in mappings.items():
        if source in task_config:
            config[section][target] = deepcopy(task_config[source])


def load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    # config.yaml contains stable defaults. A task config only needs to override
    # the model, dataset, output path, and hyperparameters that change per run.
    config = _deep_merge(DEFAULTS, _read_yaml(DEFAULT_CONFIG_PATH))
    if path != DEFAULT_CONFIG_PATH.resolve():
        task_config = _read_yaml(path)
        config = _deep_merge(config, task_config)
        _apply_simple_fields(config, task_config)
        # Only an explicitly supplied convenience field is authoritative.
        # Legacy hardware-named configs keep their original low-level switches.
        _apply_finetuning_method(config, task_config.get("finetuning_method"))
    validate_config(config, path)
    return config, path


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    for section in ("runtime", "model", "lora", "data", "training", "save"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"配置项 {section} 必须是对象")

    model = config["model"]
    data = config["data"]
    training = config["training"]
    lora = config["lora"]

    if not model.get("name_or_path"):
        raise ValueError("model.name_or_path 不能为空")
    if int(model["max_seq_length"]) <= 0:
        raise ValueError("model.max_seq_length 必须大于 0")
    if sum(bool(model.get(key)) for key in ("load_in_4bit", "load_in_8bit", "load_in_16bit")) > 1:
        raise ValueError("load_in_4bit/load_in_8bit/load_in_16bit 最多只能启用一个")
    if model.get("full_finetuning") and lora.get("enabled"):
        raise ValueError("全参数训练与 LoRA 不能同时启用")
    if not data.get("train_file"):
        raise ValueError("data.train_file 不能为空")
    if data.get("format") not in {"alpaca", "chatml", "text"}:
        raise ValueError("data.format 只能是 alpaca、chatml 或 text")
    if bool(training.get("bf16")) and bool(training.get("fp16")):
        raise ValueError("training.bf16 与 training.fp16 不能同时启用")

    positive_keys = (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
    )
    for key in positive_keys:
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} 必须大于 0")
    if float(training["learning_rate"]) < 0:
        raise ValueError("training.learning_rate 不能小于 0")
    warmup_steps = training.get("warmup_steps", "auto")
    if warmup_steps != "auto" and int(warmup_steps) < 0:
        raise ValueError("training.warmup_steps 必须为 auto 或非负整数")
    warmup_fraction = float(training.get("warmup_fraction", 0.0))
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("training.warmup_fraction 必须在 0 到 1 之间")
    if lora.get("enabled") and int(lora["r"]) <= 0:
        raise ValueError("lora.r 必须大于 0")

    resume = training.get("resume_from_checkpoint")
    if isinstance(resume, str) and resume.lower() in {"true", "latest"}:
        training["resume_from_checkpoint"] = "auto"

    validation_path = resolve_path(data.get("validation_file"), config_path)
    if training.get("load_best_model_at_end") and validation_path is None:
        raise ValueError("load_best_model_at_end=true 时必须配置 data.validation_file")
    if training.get("load_best_model_at_end") and training.get("eval_strategy") == "no":
        raise ValueError("load_best_model_at_end=true 时 eval_strategy 不能为 no")
    if training.get("load_best_model_at_end") and training.get("save_strategy") != training.get("eval_strategy"):
        raise ValueError("加载最佳模型时 save_strategy 必须与 eval_strategy 相同")
