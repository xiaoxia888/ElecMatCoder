from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT, load_config, resolve_path
from .data import load_and_format_datasets, validate_data_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


_RUNTIME_ENV = {
    "studio_disabled": "UNSLOTH_STUDIO_DISABLED",
    "compile_disable": "UNSLOTH_COMPILE_DISABLE",
    "disable_fast_generation": "UNSLOTH_DISABLE_FAST_GENERATION",
    "disable_auto_updates": "UNSLOTH_DISABLE_AUTO_UPDATES",
}


def _apply_runtime_config(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    enabled = []
    for config_key, env_name in _RUNTIME_ENV.items():
        value = bool(runtime.get(config_key, False))
        os.environ[env_name] = "1" if value else "0"
        if value:
            enabled.append(env_name)
    LOGGER.info("Unsloth 运行时开关: %s", ", ".join(enabled) or "无")


def _inspect_local_model(model_path: Path) -> None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"本地模型缺少 config.json: {config_path}")

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = model_config.get("architectures") or []
    text_config = model_config.get("text_config") or model_config
    LOGGER.info(
        "模型预检: model_type=%s, architectures=%s, layers=%s, hidden_size=%s",
        model_config.get("model_type"),
        ",".join(architectures) or "未知",
        text_config.get("num_hidden_layers", "未知"),
        text_config.get("hidden_size", "未知"),
    )
    if model_config.get("model_type") != "qwen3_5":
        LOGGER.warning("当前模型不是 qwen3_5，实际 model_type=%s", model_config.get("model_type"))

    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"本地模型目录没有找到权重文件: {model_path}")


def _assert_no_meta_parameters(model: Any) -> None:
    """Stop before SFTTrainer if loading left placeholder-only tensors."""
    meta_parameters = [
        name
        for name, parameter in model.named_parameters()
        if getattr(parameter, "is_meta", False)
    ]
    if not meta_parameters:
        return
    preview = ", ".join(meta_parameters[:8])
    suffix = "..." if len(meta_parameters) > 8 else ""
    raise RuntimeError(
        "模型加载后仍有 meta tensor，无法安全创建 SFTTrainer。"
        f"共 {len(meta_parameters)} 个，示例: {preview}{suffix}。"
        "请确认 model.offload_embedding=false，并在无残留训练进程的干净环境中重试；"
        "不要使用 to_empty()，否则会得到未初始化权重。"
    )


def _supported_kwargs(callable_object: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return kwargs
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        LOGGER.warning("当前库版本不支持这些参数，已忽略: %s", ", ".join(dropped))
    return supported


def _resolve_warmup_steps(config: dict[str, Any], train_examples: int) -> int:
    """Resolve the custom auto warmup without passing deprecated warmup_ratio."""
    cfg = config["training"]
    configured = cfg.get("warmup_steps", "auto")
    if configured != "auto":
        return int(configured)

    max_steps = int(cfg["max_steps"])
    if max_steps > 0:
        total_steps = max_steps
    else:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        effective_batch = (
            int(cfg["per_device_train_batch_size"])
            * int(cfg["gradient_accumulation_steps"])
            * world_size
        )
        total_steps = math.ceil(
            train_examples * float(cfg["num_train_epochs"]) / effective_batch
        )

    fraction = float(cfg.get("warmup_fraction", 0.0))
    warmup_steps = math.ceil(total_steps * fraction)
    LOGGER.info(
        "自动 warmup: total_steps=%s, fraction=%s, warmup_steps=%s",
        total_steps,
        fraction,
        warmup_steps,
    )
    return warmup_steps


def _build_training_arguments(SFTConfig: Any, config: dict[str, Any], run_dir: Path, has_eval: bool) -> Any:
    model_cfg = config["model"]
    cfg = config["training"]
    kwargs = {
        "output_dir": str(run_dir),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "max_steps": int(cfg["max_steps"]),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "optim": cfg["optim"],
        "bf16": bool(cfg["bf16"]),
        "fp16": bool(cfg["fp16"]),
        "logging_steps": int(cfg["logging_steps"]),
        "logging_nan_inf_filter": bool(cfg["logging_nan_inf_filter"]),
        "eval_strategy": cfg["eval_strategy"] if has_eval else "no",
        "eval_steps": int(cfg["eval_steps"]),
        "save_strategy": cfg["save_strategy"],
        "save_steps": int(cfg["save_steps"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "load_best_model_at_end": bool(cfg["load_best_model_at_end"] and has_eval),
        "metric_for_best_model": cfg["metric_for_best_model"],
        "greater_is_better": bool(cfg["greater_is_better"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "packing": bool(cfg["packing"]),
        "dataloader_num_workers": int(cfg["dataloader_num_workers"]),
        "dataset_text_field": "text",
        "dataset_num_proc": int(config["data"]["num_proc"]),
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["data_seed"]),
        "report_to": cfg["report_to"],
    }

    parameters = inspect.signature(SFTConfig).parameters
    length_key = "max_length" if "max_length" in parameters else "max_seq_length"
    kwargs[length_key] = int(model_cfg["max_seq_length"])
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return SFTConfig(**_supported_kwargs(SFTConfig, kwargs))


def _create_run_dir(config: dict[str, Any]) -> Path:
    configured = Path(config["training"]["output_dir"]).expanduser()
    output_root = configured if configured.is_absolute() else PROJECT_ROOT / configured
    if config["training"].get("append_timestamp", True):
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = output_root / run_name
        suffix = 1
        while run_dir.exists():
            run_dir = output_root / f"{run_name}_{suffix:02d}"
            suffix += 1
    else:
        run_dir = output_root
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _configured_output_root(config: dict[str, Any]) -> Path:
    configured = Path(config["training"]["output_dir"]).expanduser()
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _latest_run_dir(config: dict[str, Any]) -> Path | None:
    output_root = _configured_output_root(config)
    if not output_root.is_dir():
        return None
    if not config["training"].get("append_timestamp", True):
        return output_root
    run_dirs = sorted(
        (path for path in output_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    return run_dirs[-1] if run_dirs else None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_valid_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").is_file()
    ]
    return max(checkpoints, key=_checkpoint_step) if checkpoints else None


def _completed_adapter(run_dir: Path, adapter_subdir: str) -> Path | None:
    adapter_dir = run_dir / adapter_subdir
    if (run_dir / "run_summary.json").is_file() and (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir
    return None


def _prepare_run_dir(config: dict[str, Any], config_path: Path) -> tuple[Path, str | None]:
    resume = config["training"].get("resume_from_checkpoint")
    auto_resume = resume is True or (isinstance(resume, str) and resume.lower() == "auto")
    if auto_resume:
        previous_run = _latest_run_dir(config)
        if previous_run is not None:
            completed_adapter = _completed_adapter(previous_run, config["save"]["adapter_subdir"])
            if completed_adapter is not None:
                raise RuntimeError(
                    "检测到该输出目录中的训练已经完成。\n"
                    f"推荐 LoRA: {completed_adapter}\n"
                    "如需开始新实验，请修改 output_dir，或将 resume_from_checkpoint 设为 null。"
                )
            checkpoint = _latest_valid_checkpoint(previous_run)
            if checkpoint is not None:
                LOGGER.info("自动恢复最新 checkpoint: %s", checkpoint)
                return previous_run, str(checkpoint)
        run_dir = _create_run_dir(config)
        LOGGER.info("未发现可恢复 checkpoint，开始新训练: %s", run_dir)
        return run_dir, None

    if resume not in (None, False, ""):
        checkpoint = resolve_path(str(resume), config_path)
        if checkpoint is None or not (checkpoint / "trainer_state.json").is_file():
            raise FileNotFoundError(f"无效的恢复 checkpoint: {checkpoint}")
        LOGGER.info("使用指定 checkpoint 恢复: %s", checkpoint)
        return checkpoint.parent, str(checkpoint)

    run_dir = _create_run_dir(config)
    LOGGER.info("已禁用自动恢复，开始新训练: %s", run_dir)
    return run_dir, None


def _validate_response_labels(trainer: Any) -> None:
    sample = trainer.train_dataset[0]
    labels = sample.get("labels")
    if labels is None:
        LOGGER.warning("无法检查 response-only 标签：训练集没有 labels 字段")
        return
    trainable = sum(int(value != -100) for value in labels)
    if trainable == 0:
        raise RuntimeError("response-only 掩码后没有可训练 token，请检查 instruction_part/response_part")
    LOGGER.info("response-only 标签检查通过，首条样本可训练 token=%d", trainable)


def _moving_average(
    points: list[tuple[int, float]], window: int = 5
) -> list[tuple[int, float]]:
    """Smooth logged metrics without introducing a numpy dependency."""
    if len(points) < 3:
        return points.copy()
    window = max(2, min(window, len(points)))
    smoothed: list[tuple[int, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        if len(values) > window:
            values.pop(0)
        smoothed.append((step, sum(values) / len(values)))
    return smoothed


def _use_log_scale(series: list[list[tuple[int, float]]]) -> bool:
    positive_values = [
        value
        for points in series
        for _, value in points
        if math.isfinite(value) and value > 0
    ]
    return bool(positive_values) and max(positive_values) / min(positive_values) >= 50


def _plot_points(axis: Any, points: list[tuple[int, float]], **kwargs: Any) -> None:
    if points:
        axis.plot(
            [step for step, _ in points],
            [value for _, value in points],
            **kwargs,
        )


def _style_loss_axis(axis: Any, title: str, *, logarithmic: bool = False) -> None:
    axis.set_title(title)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss" + (" (log scale)" if logarithmic else ""))
    if logarithmic:
        axis.set_yscale("log")
    axis.grid(True, which="both" if logarithmic else "major", alpha=0.25)


def _save_training_curves(
    log_history: list[dict[str, Any]], run_dir: Path
) -> dict[str, str]:
    """Export separate train/eval plots plus a backwards-compatible overview."""
    train_points = [
        (int(item["step"]), float(item["loss"]))
        for item in log_history
        if "step" in item and "loss" in item
    ]
    eval_points = [
        (int(item["step"]), float(item["eval_loss"]))
        for item in log_history
        if "step" in item and "eval_loss" in item
    ]
    learning_rate_points = [
        (int(item["step"]), float(item["learning_rate"]))
        for item in log_history
        if "step" in item and "learning_rate" in item
    ]
    if not train_points and not eval_points:
        LOGGER.warning("没有找到 loss 日志，跳过训练曲线")
        return {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("未安装 matplotlib，跳过训练曲线")
        return {}

    artifacts: dict[str, str] = {}
    train_smoothed = _moving_average(train_points)
    full_log_scale = _use_log_scale([train_points, eval_points])

    if train_points:
        figure, axis = plt.subplots(figsize=(11, 6.5))
        _plot_points(
            axis,
            train_points,
            color="#90caf9",
            linewidth=1,
            alpha=0.55,
            label="train loss (raw)",
        )
        _plot_points(
            axis,
            train_smoothed,
            color="#1565c0",
            linewidth=2.3,
            label="train loss (moving average)",
        )
        _style_loss_axis(axis, "Training loss", logarithmic=_use_log_scale([train_points]))
        axis.legend()
        figure.tight_layout()
        train_path = run_dir / "training_loss.png"
        figure.savefig(train_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        artifacts["train_loss"] = str(train_path)

    if eval_points:
        figure, axis = plt.subplots(figsize=(10, 6.2))
        _plot_points(
            axis,
            eval_points,
            color="#ef6c00",
            marker="o",
            markersize=6,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="eval loss (measured steps only)",
        )
        best_step, best_loss = min(eval_points, key=lambda point: point[1])
        axis.scatter(
            [best_step],
            [best_loss],
            color="#c62828",
            marker="*",
            s=220,
            zorder=4,
            label=f"best: {best_loss:.6g} @ step {best_step}",
        )
        # Label every point for short runs; sample labels on evaluation-heavy runs.
        label_stride = max(1, math.ceil(len(eval_points) / 10))
        for index, (step, loss) in enumerate(eval_points):
            if index % label_stride == 0 or index == len(eval_points) - 1 or step == best_step:
                axis.annotate(
                    f"{loss:.6g}",
                    (step, loss),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#9a4b00",
                )
        _style_loss_axis(axis, "Validation loss — markers are actual evaluations")
        axis.legend()
        figure.tight_layout()
        eval_path = run_dir / "validation_loss.png"
        figure.savefig(eval_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        artifacts["eval_loss"] = str(eval_path)

    figure, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    overview_axis, detail_axis = axes
    _plot_points(
        overview_axis,
        train_points,
        color="#90caf9",
        linewidth=0.9,
        alpha=0.45,
        label="train loss (raw)",
    )
    _plot_points(
        overview_axis,
        train_smoothed,
        color="#1565c0",
        linewidth=2.2,
        label="train loss (moving average)",
    )
    _plot_points(
        overview_axis,
        eval_points,
        color="#ef6c00",
        marker="o",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="eval loss (measured steps only)",
    )
    _style_loss_axis(
        overview_axis,
        "Full training run",
        logarithmic=full_log_scale,
    )
    overview_axis.legend()

    detail_start = eval_points[0][0] if eval_points else train_points[len(train_points) // 2][0]
    detail_train = [point for point in train_points if point[0] >= detail_start]
    detail_smoothed = [point for point in train_smoothed if point[0] >= detail_start]
    _plot_points(
        detail_axis,
        detail_train,
        color="#90caf9",
        linewidth=0.9,
        alpha=0.45,
        label="train loss (raw)",
    )
    _plot_points(
        detail_axis,
        detail_smoothed,
        color="#1565c0",
        linewidth=2.2,
        label="train loss (moving average)",
    )
    _plot_points(
        detail_axis,
        eval_points,
        color="#ef6c00",
        marker="o",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="eval loss (measured steps only)",
    )
    if eval_points:
        best_step, best_loss = min(eval_points, key=lambda point: point[1])
        detail_axis.scatter(
            [best_step], [best_loss], color="#c62828", marker="*", s=180, zorder=4
        )
    _style_loss_axis(detail_axis, f"Late-stage detail (step {detail_start} onward)")
    detail_axis.legend()
    overview_path = run_dir / "training_curves.png"
    figure.savefig(overview_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    artifacts["overview"] = str(overview_path)

    if learning_rate_points:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        _plot_points(
            axis,
            learning_rate_points,
            color="#6a1b9a",
            linewidth=2,
            label="learning rate",
        )
        axis.set_title("Learning-rate schedule")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Learning rate")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        lr_path = run_dir / "learning_rate.png"
        figure.savefig(lr_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        artifacts["learning_rate"] = str(lr_path)

    LOGGER.info("训练图表已保存: %s", ", ".join(artifacts.values()))
    return artifacts


def run(config_path: str | Path | None, validate_only: bool = False) -> Path | None:
    config, loaded_path = load_config(config_path)
    _apply_runtime_config(config)
    data_cfg = config["data"]
    train_path = resolve_path(data_cfg["train_file"], loaded_path)
    validation_path = resolve_path(data_cfg.get("validation_file"), loaded_path)
    assert train_path is not None

    train_sample = validate_data_file(train_path, data_cfg)
    if validation_path is not None:
        validate_data_file(validation_path, data_cfg)
    LOGGER.info("配置有效: %s", loaded_path)
    if config.get("finetuning_method"):
        LOGGER.info("微调方式: %s", str(config["finetuning_method"]).upper())
    LOGGER.info("训练集: %s", train_path)
    LOGGER.info("验证集: %s", validation_path or "未配置")
    LOGGER.info("首条样本字段: %s", ", ".join(train_sample))
    training_cfg = config["training"]
    effective_batch = int(training_cfg["per_device_train_batch_size"]) * int(
        training_cfg["gradient_accumulation_steps"]
    )
    LOGGER.info(
        "训练批次: physical=%s, accumulation=%s, effective=%s",
        training_cfg["per_device_train_batch_size"],
        training_cfg["gradient_accumulation_steps"],
        effective_batch,
    )
    if validate_only:
        return None

    # Unsloth must be imported before Transformers/TRL so its patches are installed.
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model_cfg = config["model"]
    model_name_or_path = model_cfg["name_or_path"]
    local_model_path = resolve_path(model_name_or_path, loaded_path)
    if local_model_path is not None and local_model_path.exists():
        _inspect_local_model(local_model_path)
        model_name_or_path = str(local_model_path)
    LOGGER.info("加载基座模型: %s", model_name_or_path)
    model_kwargs = {
        "model_name": model_name_or_path,
        "max_seq_length": int(model_cfg["max_seq_length"]),
        "dtype": model_cfg.get("dtype"),
        "load_in_4bit": bool(model_cfg["load_in_4bit"]),
        "load_in_8bit": bool(model_cfg["load_in_8bit"]),
        "load_in_16bit": bool(model_cfg["load_in_16bit"]),
        "full_finetuning": bool(model_cfg["full_finetuning"]),
        "fast_inference": bool(model_cfg["fast_inference"]),
        "offload_embedding": bool(model_cfg.get("offload_embedding", False)),
        "trust_remote_code": bool(model_cfg["trust_remote_code"]),
    }
    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    lora_cfg = config["lora"]
    if lora_cfg["enabled"]:
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg["r"]),
            target_modules=lora_cfg["target_modules"],
            lora_alpha=int(lora_cfg["lora_alpha"]),
            lora_dropout=float(lora_cfg["lora_dropout"]),
            bias=lora_cfg["bias"],
            use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
            random_state=int(lora_cfg["random_state"]),
            use_rslora=bool(lora_cfg["use_rslora"]),
            max_seq_length=int(model_cfg["max_seq_length"]),
        )
    _assert_no_meta_parameters(model)

    train_dataset, eval_dataset = load_and_format_datasets(
        train_path, validation_path, tokenizer, data_cfg
    )
    config["training"]["warmup_steps"] = _resolve_warmup_steps(
        config, len(train_dataset)
    )
    run_dir, resume_checkpoint = _prepare_run_dir(config, loaded_path)
    report_targets = config["training"]["report_to"]
    if isinstance(report_targets, str):
        report_targets = [report_targets]
    tensorboard_dir = run_dir / config["training"]["logging_dir"]
    if "tensorboard" in report_targets:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_dir)
        LOGGER.info("TensorBoard 实时日志: %s", tensorboard_dir)
    (run_dir / "source_config.yaml").write_text(
        loaded_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": _build_training_arguments(SFTConfig, config, run_dir, eval_dataset is not None),
    }
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    if data_cfg.get("train_on_responses_only"):
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part=data_cfg.get("instruction_part"),
            response_part=data_cfg.get("response_part"),
        )
        _validate_response_labels(trainer)

    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    save_cfg = config["save"]
    adapter_dir = run_dir / save_cfg["adapter_subdir"]
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    LOGGER.info("LoRA adapter 已保存: %s", adapter_dir)

    curve_artifacts: dict[str, str] = {}
    if save_cfg.get("plot_training_curves"):
        try:
            curve_artifacts = _save_training_curves(trainer.state.log_history, run_dir)
        except Exception:
            # Plotting is post-processing and must never discard a completed adapter.
            LOGGER.exception("训练已完成，但生成训练图表失败")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    best_info = {
        "selection_metric": config["training"]["metric_for_best_model"],
        "greater_is_better": config["training"]["greater_is_better"],
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint,
        "recommended_lora_path": str(adapter_dir),
        "recommended_lora_source": (
            "best_checkpoint" if best_checkpoint else "last_training_step"
        ),
    }
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps(best_info, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if best_checkpoint:
        LOGGER.info("最佳 checkpoint: %s (%s=%s)", best_checkpoint, best_info["selection_metric"], best_metric)
    else:
        LOGGER.warning("本次训练没有产生可比较的评估 checkpoint，final_adapter 来自最后训练步")

    if save_cfg.get("merged_16bit"):
        merged_dir = run_dir / save_cfg["merged_subdir"]
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        LOGGER.info("16-bit 合并模型已保存: %s", merged_dir)
    if save_cfg.get("gguf"):
        gguf_dir = run_dir / save_cfg["gguf_subdir"]
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=save_cfg["gguf_quantization_method"],
        )
        LOGGER.info("GGUF 已保存: %s", gguf_dir)

    summary = {
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter_dir),
        "best_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        # Keep the old field for existing consumers and expose every new image separately.
        "training_curves": curve_artifacts.get("overview"),
        "training_curve_artifacts": curve_artifacts,
        "tensorboard_dir": str(tensorboard_dir) if "tensorboard" in report_targets else None,
        "metrics": result.metrics,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML 驱动的 Unsloth SFT 训练入口")
    parser.add_argument("--config", required=True, help="train_config 下的任务覆盖配置")
    parser.add_argument("--validate-only", action="store_true", help="只校验配置与数据，不加载模型")
    args = parser.parse_args()
    run(args.config, args.validate_only)


if __name__ == "__main__":
    main()
