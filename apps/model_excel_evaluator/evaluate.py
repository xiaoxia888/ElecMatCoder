from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
DEFAULT_CONFIG = APP_DIR / "config.yaml"
DEFAULT_FULL_CONFIG = APP_DIR / "config.full.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("model_excel_evaluator")

TASK_ALIASES = {
    "type": "type",
    "种类": "type",
    "size": "size",
    "尺寸": "size",
    "thickness": "thickness",
    "壁厚": "thickness",
    "pressure": "pressure",
    "压力": "pressure",
    "磅级": "pressure",
    "material": "material",
    "材质": "material",
    "standard": "standard",
    "规范": "standard",
    "code": "code",
    "编码": "code",
    "full": "full",
    "完整": "full",
    "完整编码": "full",
}

FULL_STAGE_DEFAULTS = ("type", "material", "size")
LOCAL_TRANSFORMERS_BACKEND = "local_transformers"
MANAGED_VLLM_BACKEND = "managed_vllm"


@dataclass
class EncodedValue:
    code: str = ""
    coder_value: str = ""
    source: str = "规则编码"
    error: str = ""


@dataclass
class CoderRequest:
    model_name: str
    row_index: int
    field: str
    value: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比多个 Qwen/LoRA 模型的字段抽取与最终编码准确率")
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML 配置文件；默认按 task 选择 config.yaml 或 config.full.yaml",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="type/种类、size/尺寸、材质、规范、壁厚、磅级、code/直接编码或 full/完整编码",
    )
    parser.add_argument("--input", required=True, type=Path, help="包含材料描述和正确代码的 Excel")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output-dir", type=Path, help="结果目录；包含 Excel 和多张独立分析图")
    output_group.add_argument("--output", type=Path, help="兼容旧用法：直接指定输出 Excel 路径")
    parser.add_argument("--prompt", type=Path, help="覆盖当前任务的提示词路径")
    parser.add_argument("--models", nargs="+", help="要测试的模型名称；默认测试 YAML 中全部模型")
    parser.add_argument("--groups", nargs="+", help="full 模式要测试的组合组名；默认测试 groups 中全部组")
    parser.add_argument("--base-model", action="append", default=[], metavar="NAME=PATH", help="覆盖模型底座；full 用 NAME:TASK=PATH")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH", help="覆盖模型 LoRA；full 用 NAME:TASK=PATH")
    parser.add_argument("--sheet", help="输入工作表名称或从 0 开始的序号")
    parser.add_argument("--description-column", help="材料描述列名")
    parser.add_argument("--truth-column", help="正确代码列名")
    parser.add_argument("--limit", type=int, help="仅测试前 N 条，便于冒烟验证")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖结果目录中的已有文件")
    parser.add_argument("--validate-only", action="store_true", help="只校验配置、模型和输入文件")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def normalize_task(value: str) -> str:
    task = TASK_ALIASES.get(str(value or "").strip().lower())
    if not task:
        raise ValueError(f"不支持的任务: {value}；可选值: {', '.join(TASK_ALIASES)}")
    return task


def load_yaml(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML 顶层必须是对象")
    data["_config_path"] = config_path
    return data


def resolve_path(value: Any, config_path: Path, *, required: bool = False) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("必填路径为空")
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def parse_named_overrides(values: Iterable[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{option} 格式必须是 NAME=PATH，收到: {raw}")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"{option} 格式必须是 NAME=PATH，收到: {raw}")
        result[name] = path
    return result


def resolve_adapter(model_config: dict[str, Any], task: str) -> str:
    adapters = model_config.get("adapters") or {}
    task_value = adapters.get(task) if isinstance(adapters, dict) else ""
    if not task_value and isinstance(adapters, dict):
        shared_task = {"thickness": "size", "pressure": "size", "standard": "material"}.get(task)
        task_value = adapters.get(shared_task) if shared_task else ""
    return str(task_value or model_config.get("adapter_path") or model_config.get("weight_path") or "").strip()


def resolve_full_stages(task_config: dict[str, Any]) -> tuple[str, ...]:
    raw_stages = task_config.get("stages") or FULL_STAGE_DEFAULTS
    if not isinstance(raw_stages, (list, tuple)):
        raise ValueError("tasks.full.stages 必须是列表")
    stages = tuple(normalize_task(value) for value in raw_stages)
    if set(stages) != set(FULL_STAGE_DEFAULTS) or len(stages) != len(FULL_STAGE_DEFAULTS):
        expected = ", ".join(FULL_STAGE_DEFAULTS)
        raise ValueError(f"tasks.full.stages 必须且只能包含: {expected}")
    return stages


def resolve_full_adapter(model_config: dict[str, Any], stage: str) -> str:
    adapters = model_config.get("adapters") or {}
    if not isinstance(adapters, dict):
        return ""
    return str(adapters.get(stage) or "").strip()


def select_task_model_config(model_config: dict[str, Any], task: str, model_name: str) -> dict[str, Any]:
    routes = model_config.get("routes")
    if routes is None:
        return dict(model_config)
    if not isinstance(routes, dict):
        raise ValueError(f"models.{model_name}.routes 必须是对象")
    route_task = task_route_name(task, routes)
    route = routes.get(route_task)
    if not isinstance(route, dict):
        raise ValueError(f"models.{model_name} 没有配置 {task} 任务的 route")
    merged = {key: value for key, value in model_config.items() if key != "routes"}
    merged.update(route)
    return merged


def task_route_name(task: str, routes: dict[str, Any]) -> str:
    if task in routes:
        return task
    return {"thickness": "size", "pressure": "size", "standard": "material"}.get(task, task)


def model_supports_task(model_config: Any, task: str) -> bool:
    if not isinstance(model_config, dict):
        return False
    routes = model_config.get("routes")
    if routes is None:
        return True
    return isinstance(routes, dict) and task_route_name(task, routes) in routes


def normalize_stage_backend(value: Any) -> str:
    backend = str(value or LOCAL_TRANSFORMERS_BACKEND).strip().lower()
    aliases = {
        "transformers": LOCAL_TRANSFORMERS_BACKEND,
        "local": LOCAL_TRANSFORMERS_BACKEND,
        "vllm": MANAGED_VLLM_BACKEND,
        "vllm_service": MANAGED_VLLM_BACKEND,
    }
    backend = aliases.get(backend, backend)
    if backend not in {LOCAL_TRANSFORMERS_BACKEND, MANAGED_VLLM_BACKEND}:
        raise ValueError(
            f"不支持的模型后端: {value}；可选值: {LOCAL_TRANSFORMERS_BACKEND}, {MANAGED_VLLM_BACKEND}"
        )
    return backend


def prepare_stage_model(
    *,
    name: str,
    item: dict[str, Any],
    config_path: Path,
    base_override: str = "",
    adapter_override: str = "",
    default_adapter: str = "",
) -> dict[str, Any]:
    backend = normalize_stage_backend(item.get("backend"))
    if backend == MANAGED_VLLM_BACKEND:
        if base_override or adapter_override:
            raise ValueError(
                f"{name} 使用 managed_vllm，模型路径由 service_config 管理，不能使用 --base-model/--adapter 覆盖"
            )
        service_config = resolve_path(item.get("service_config"), config_path, required=True)
        served_model = str(item.get("served_model") or item.get("model") or "").strip()
        if not served_model:
            raise ValueError(f"{name} 使用 managed_vllm 时必须配置 served_model")
        return {
            "name": name,
            "backend": backend,
            "service_config": service_config,
            "served_model": served_model,
            "max_new_tokens": item.get("max_new_tokens"),
        }

    base_raw = base_override or item.get("base_model_path")
    adapter_raw = adapter_override or item.get("adapter_path") or default_adapter
    return {
        "name": name,
        "backend": backend,
        "base_model_path": resolve_path(base_raw, config_path, required=True),
        "adapter_path": resolve_path(adapter_raw, config_path, required=True),
    }


def validate_base_model_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label}底座模型目录不存在: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label}底座模型缺少 config.json: {path}")


def validate_adapter_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label}LoRA 权重目录不存在: {path}")
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{label}LoRA 缺少 adapter_config.json: {path}")
    weight_names = ("adapter_model.safetensors", "adapter_model.bin")
    if not any((path / name).is_file() for name in weight_names):
        raise FileNotFoundError(f"{label}LoRA 缺少 adapter_model.safetensors 或 adapter_model.bin: {path}")


def validate_managed_vllm_model(model: dict[str, Any], label: str) -> None:
    from apps.vllm_service.config import load_config as load_vllm_config

    service_config = model["service_config"]
    if not service_config.is_file():
        raise FileNotFoundError(f"{label}vLLM 配置不存在: {service_config}")
    deployment = load_vllm_config(service_config)
    served_model = str(model["served_model"])
    if served_model not in deployment.models:
        available = ", ".join(deployment.models)
        raise ValueError(
            f"{label}served_model={served_model} 未在 {service_config.name} 注册；可选: {available}"
        )
    for engine_name, engine in deployment.engines.items():
        validate_base_model_dir(Path(engine.model_path), f"{label}vLLM/{engine_name} ")
        for adapter_name, adapter_path in engine.lora_modules.items():
            validate_adapter_dir(Path(adapter_path), f"{label}vLLM/{engine_name}/{adapter_name} ")


def validate_stage_model(model: dict[str, Any], label: str) -> None:
    if model.get("backend") == MANAGED_VLLM_BACKEND:
        validate_managed_vllm_model(model, label)
    else:
        validate_base_model_dir(model["base_model_path"], label)
        validate_adapter_dir(model["adapter_path"], label)


def prepare_runtime(args: argparse.Namespace) -> dict[str, Any]:
    task = normalize_task(args.task)
    selected_config_path = args.config or (DEFAULT_FULL_CONFIG if task == "full" else DEFAULT_CONFIG)
    config = load_yaml(selected_config_path)
    config_path = config["_config_path"]
    task_config = (config.get("tasks") or {}).get(task)
    if not isinstance(task_config, dict):
        raise ValueError(f"配置中缺少 tasks.{task}")

    requested_groups = getattr(args, "groups", None)
    if task != "full" and requested_groups:
        raise ValueError("--groups 只适用于 full/完整编码任务")
    if task == "full":
        # groups 是完整流水线的推荐写法；models 作为旧配置兼容入口。
        models_config = config.get("groups") or config.get("models") or {}
    else:
        models_config = config.get("models") or {}
    if not isinstance(models_config, dict) or not models_config:
        key = "groups（或兼容的 models）" if task == "full" else "models"
        raise ValueError(f"配置中 {key} 不能为空")
    if task == "full":
        selected_names = requested_groups or args.models or list(models_config)
    elif args.models:
        selected_names = args.models
    else:
        selected_names = [
            name for name, item in models_config.items()
            if model_supports_task(item, task)
        ]
        if not selected_names:
            raise ValueError(f"配置中没有支持 {task} 任务的模型 route")
    unknown = [name for name in selected_names if name not in models_config]
    if unknown:
        label = "组合组" if task == "full" else "模型"
        raise ValueError(f"配置中不存在{label}: {', '.join(unknown)}")

    base_overrides = parse_named_overrides(args.base_model, "--base-model")
    adapter_overrides = parse_named_overrides(args.adapter, "--adapter")
    allowed_adapter_override_names = set(selected_names)
    allowed_base_override_names = set(selected_names)
    if task == "full":
        allowed_adapter_override_names.update(
            f"{name}:{stage}" for name in selected_names for stage in FULL_STAGE_DEFAULTS
        )
        allowed_base_override_names.update(
            f"{name}:{stage}" for name in selected_names for stage in FULL_STAGE_DEFAULTS
        )
    unknown_overrides = set(base_overrides) - allowed_base_override_names
    unknown_overrides |= set(adapter_overrides) - allowed_adapter_override_names
    if unknown_overrides:
        raise ValueError(f"覆盖参数包含未选中的模型: {', '.join(sorted(unknown_overrides))}")

    full_stages = resolve_full_stages(task_config) if task == "full" else ()
    if task == "full" and args.prompt:
        raise ValueError("full 任务包含三个提示词，请在 tasks.type/material/size.prompt_path 中分别配置")
    if task == "full" and any(name in adapter_overrides or name in base_overrides for name in selected_names):
        raise ValueError("full 任务覆盖路径时请使用 NAME:TASK=PATH，例如 方案A:size=/path/to/model")

    selected_models = []
    for name in selected_names:
        item = dict(models_config[name] or {})
        if task == "full":
            configured_stages = item.get("stages") or {}
            if not isinstance(configured_stages, dict):
                raise ValueError(f"models.{name}.stages 必须是对象")
            stage_models = {}
            for stage in full_stages:
                stage_item = configured_stages.get(stage) or {}
                if not isinstance(stage_item, dict):
                    raise ValueError(f"models.{name}.stages.{stage} 必须是对象")
                # 兼容旧 full 配置在组级提供公共 base_model_path/adapters。
                if item.get("base_model_path") and not stage_item.get("base_model_path"):
                    stage_item = {**stage_item, "base_model_path": item.get("base_model_path")}
                stage_models[stage] = prepare_stage_model(
                    name=f"{name}/{stage}",
                    item=stage_item,
                    config_path=config_path,
                    base_override=str(base_overrides.get(f"{name}:{stage}") or ""),
                    adapter_override=str(adapter_overrides.get(f"{name}:{stage}") or ""),
                    default_adapter=str(resolve_full_adapter(item, stage) or ""),
                )
            selected_models.append({"name": name, "stages": stage_models})
        else:
            item = select_task_model_config(item, task, name)
            adapter_raw = adapter_overrides.get(name) or resolve_adapter(item, task)
            selected_models.append(
                prepare_stage_model(
                    name=name,
                    item=item,
                    config_path=config_path,
                    base_override=str(base_overrides.get(name) or ""),
                    adapter_override=str(adapter_overrides.get(name) or ""),
                    default_adapter=str(adapter_raw or ""),
                )
            )

    if task != "full" and args.prompt and any(
        model.get("backend") == MANAGED_VLLM_BACKEND for model in selected_models
    ):
        raise ValueError(
            "managed_vllm 的提示词由 service_config 管理，不能使用 --prompt 覆盖；请修改对应 service.eval.*.yaml"
        )

    if task == "full":
        prompt_paths = {}
        for stage in full_stages:
            stage_config = (config.get("tasks") or {}).get(stage)
            if not isinstance(stage_config, dict):
                raise ValueError(f"配置中缺少 tasks.{stage}")
            prompt_paths[stage] = resolve_path(stage_config.get("prompt_path"), config_path, required=True)
        prompt_path = None
    else:
        prompt_path = args.prompt.expanduser().resolve() if args.prompt else resolve_path(
            task_config.get("prompt_path"), config_path, required=True
        )
        prompt_paths = {task: prompt_path}
    input_path = args.input.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_dir = output_path.parent
    else:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else APP_DIR / "outputs" / f"模型对比_{task}_{timestamp}"
        )
        output_path = output_dir / "评测结果.xlsx"
    image_paths = {
        "accuracy": output_dir / "准确率对比.png",
        "latency": output_dir / "耗时对比.png",
        "disagreement": output_dir / "模型差异.png",
    }
    return {
        "config": config,
        "config_path": config_path,
        "task": task,
        "task_config": task_config,
        "task_field": str(task_config.get("field") or task).upper(),
        "models": selected_models,
        "prompt_path": prompt_path,
        "prompt_paths": prompt_paths,
        "full_stages": full_stages,
        "input_path": input_path,
        "output_dir": output_dir,
        "output_path": output_path,
        "image_paths": image_paths,
    }


def validate_runtime(runtime: dict[str, Any], args: argparse.Namespace) -> tuple[str, str, Any]:
    if not runtime["input_path"].is_file():
        raise FileNotFoundError(f"文件不存在: {runtime['input_path']}")
    for prompt_path in runtime["prompt_paths"].values():
        if not prompt_path.is_file():
            raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
        if not prompt_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"提示词文件为空: {prompt_path}")
    for model in runtime["models"]:
        if runtime["task"] == "full":
            for stage, stage_model in model["stages"].items():
                validate_stage_model(stage_model, f"{model['name']}/{stage} ")
        else:
            validate_stage_model(model, f"{model['name']} ")

    coder = runtime["config"].get("coder") or {}
    fallback_fields = {str(value).upper() for value in coder.get("fallback_fields") or []}
    coder_needed = runtime["task"] == "full" or runtime["task_field"] in fallback_fields
    if coder.get("enabled", True) and coder_needed and fallback_fields:
        backend = str(coder.get("backend") or "local_transformers")
        if backend == "local_transformers":
            coder_base = resolve_path(coder.get("base_model_path"), runtime["config_path"], required=True)
            coder_adapter = resolve_path(coder.get("adapter_path"), runtime["config_path"], required=True)
            coder_prompt = resolve_path(coder.get("prompt_path"), runtime["config_path"], required=True)
            validate_base_model_dir(coder_base, "coder ")
            validate_adapter_dir(coder_adapter, "coder ")
            if not coder_prompt.is_file():
                raise FileNotFoundError(f"coder 提示词不存在: {coder_prompt}")
        elif backend == MANAGED_VLLM_BACKEND:
            coder_model = {
                "backend": MANAGED_VLLM_BACKEND,
                "service_config": resolve_path(
                    coder.get("service_config"), runtime["config_path"], required=True
                ),
                "served_model": str(coder.get("served_model") or "coder"),
            }
            validate_managed_vllm_model(coder_model, "coder ")
        elif backend != "mlx_service":
            raise ValueError("coder.backend 只能是 local_transformers、managed_vllm 或 mlx_service")

    excel_config = runtime["config"].get("excel") or {}
    description_column = args.description_column or excel_config.get("description_column") or "材料描述"
    truth_column = (
        args.truth_column
        or runtime["task_config"].get("truth_column")
        or excel_config.get("truth_column")
        or "正确代码"
    )
    sheet = args.sheet if args.sheet is not None else excel_config.get("sheet_name", 0)
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    frame = pd.read_excel(runtime["input_path"], sheet_name=sheet, dtype=str).fillna("")
    missing = [column for column in (description_column, truth_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Excel 缺少列: {', '.join(missing)}；实际列: {', '.join(map(str, frame.columns))}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        frame = frame.head(args.limit).copy()
    if frame.empty:
        raise ValueError("输入 Excel 没有可测试的数据")
    return str(description_column), str(truth_column), frame


def parse_json_output(raw: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def clean_code_output(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def serialize_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return str(value or "").strip()


class ManagedVLLMService:
    """由评测进程托管的 apps.vllm_service.launch 子进程。"""

    def __init__(self, service_config: Path, settings: dict[str, Any] | None = None):
        from apps.vllm_service.config import load_config as load_vllm_config

        self.service_config = service_config.expanduser().resolve()
        self.settings = settings or {}
        self.deployment = load_vllm_config(self.service_config)
        self.service_url = str(
            self.settings.get("service_url")
            or f"http://127.0.0.1:{self.deployment.gateway.port}"
        ).rstrip("/")
        self.process: subprocess.Popen[str] | None = None
        self.load_seconds = 0.0

    def _health(self, timeout: float = 3.0) -> dict[str, Any] | None:
        request = urllib.request.Request(f"{self.service_url}/health")
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                if not 200 <= response.status < 300:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else None
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ):
            return None

    def start(self) -> "ManagedVLLMService":
        existing = self._health(timeout=1.0)
        if existing is not None:
            raise RuntimeError(
                f"{self.service_url} 已存在服务。managed_vllm 不会接管未知进程，请先关闭占用端口的服务"
            )
        command = [
            sys.executable,
            "-m",
            "apps.vllm_service.launch",
            "--config",
            str(self.service_config),
            "--log-level",
            str(self.settings.get("log_level") or "info"),
        ]
        LOGGER.info("自动启动 vLLM: %s", self.service_config)
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            start_new_session=True,
        )
        try:
            timeout = float(
                self.settings.get("startup_timeout")
                or self.deployment.gateway.startup_timeout_seconds + 60
            )
            interval = max(0.5, float(self.settings.get("health_interval", 2.0)))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                code = self.process.poll()
                if code is not None:
                    raise RuntimeError(f"vLLM 启动器提前退出，退出码: {code}，配置: {self.service_config}")
                health = self._health(timeout=min(5.0, interval + 1.0))
                if health and health.get("ok"):
                    self.load_seconds = time.perf_counter() - started
                    LOGGER.info("vLLM 已就绪，用时 %.1f 秒: %s", self.load_seconds, self.service_url)
                    return self
                time.sleep(interval)
            raise TimeoutError(f"等待 vLLM 服务启动超时（{timeout:.0f}秒）: {self.service_config}")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        LOGGER.info("停止 vLLM 服务并释放显存: %s", self.service_config.name)
        process.terminate()
        timeout = max(5.0, float(self.settings.get("shutdown_timeout", 60)))
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            LOGGER.warning("vLLM 未在 %.0f 秒内退出，强制终止进程组", timeout)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, PermissionError):
                process.kill()
            process.wait(timeout=10)
        cooldown = max(0.0, float(self.settings.get("shutdown_cooldown", 2.0)))
        if cooldown:
            time.sleep(cooldown)

    def __enter__(self) -> "ManagedVLLMService":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.stop()


_VLLM_HTTP_LOCAL = threading.local()


def _vllm_http_session():
    import requests

    session = getattr(_VLLM_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _VLLM_HTTP_LOCAL.session = session
    return session


def _post_vllm_predict(
    *,
    service: ManagedVLLMService,
    served_model: str,
    text: str,
    max_new_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = _vllm_http_session().post(
                f"{service.service_url}/predict",
                headers={"Connection": "keep-alive"},
                json={
                    "model": served_model,
                    "text": text,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return {
                "raw": str(payload.get("raw_response") or ""),
                "parsed": payload.get("parsed_json"),
                "stage1_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            last_error = exc
            if service.process is not None and service.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM 服务在推理期间退出，退出码: {service.process.returncode}"
                ) from exc
            if attempt < retries:
                time.sleep(min(5.0, 0.5 * (2 ** attempt)))
    raise RuntimeError(f"vLLM 请求失败，已重试 {retries} 次: {last_error}") from last_error


def run_vllm_stage1(
    model_config: dict[str, Any],
    descriptions: list[str],
    settings: dict[str, Any],
    service: ManagedVLLMService,
) -> list[dict[str, Any]]:
    vllm_settings = settings.get("_managed_vllm") or {}
    concurrency = max(1, int(vllm_settings.get("concurrency", 16)))
    timeout = float(vllm_settings.get("request_timeout", 600))
    retries = max(0, int(vllm_settings.get("retries", 2)))
    progress_every = max(1, int(vllm_settings.get("progress_every", 100)))
    max_new_tokens = int(
        model_config.get("max_new_tokens")
        or settings.get("max_new_tokens", 256)
    )
    LOGGER.info(
        "调用 vLLM 模型 %s（route=%s），concurrency=%d",
        model_config["name"],
        model_config["served_model"],
        concurrency,
    )
    records: list[dict[str, Any] | None] = [None] * len(descriptions)
    completed = 0
    next_index = 0

    def submit(pool: ThreadPoolExecutor, index: int):
        return pool.submit(
            _post_vllm_predict,
            service=service,
            served_model=model_config["served_model"],
            text=descriptions[index],
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            retries=retries,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}
        while next_index < len(descriptions) and len(pending) < concurrency:
            future = submit(pool, next_index)
            pending[future] = next_index
            next_index += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                records[index] = future.result()
                completed += 1
                if completed % progress_every == 0 or completed == len(descriptions):
                    LOGGER.info("%s: %d/%d", model_config["name"], completed, len(descriptions))
                if next_index < len(descriptions):
                    replacement = submit(pool, next_index)
                    pending[replacement] = next_index
                    next_index += 1
    if any(record is None for record in records):
        raise RuntimeError("vLLM 推理结果数量与输入不一致")
    return [record for record in records if record is not None]


class TransformersRunner:
    def __init__(self, base_model_path: Path, adapter_path: Path | None, settings: dict[str, Any]):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.settings = settings
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config_file = self.base_model_path / "config.json"
        model_type = ""
        if config_file.is_file():
            model_type = str(json.loads(config_file.read_text(encoding="utf-8")).get("model_type") or "")
        if model_type == "qwen3_5":
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise RuntimeError(
                    f"Transformers {transformers.__version__} 不支持 Qwen3.5，请安装项目要求的 5.3+ 版本"
                ) from exc
            model_class = AutoModelForMultimodalLM
        else:
            model_class = AutoModelForCausalLM

        dtype_name = str(self.settings.get("dtype") or "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": self.settings.get("device_map", "auto"),
            "trust_remote_code": bool(self.settings.get("trust_remote_code", True)),
        }
        if self.settings.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=load_kwargs["trust_remote_code"],
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = model_class.from_pretrained(self.base_model_path, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.model = model_class.from_pretrained(self.base_model_path, **load_kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def _device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _render(instruction: str, user_text: str) -> str:
        return (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(self, user_texts: list[str], instruction: str, *, max_new_tokens: int) -> tuple[list[str], float]:
        import torch

        prompts = [self._render(instruction, text) for text in user_texts]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.settings.get("max_input_tokens", 2048)),
        ).to(self._device())
        temperature = float(self.settings.get("temperature", 0.0))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=float(self.settings.get("top_p", 1.0)))
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prompt_size = inputs["input_ids"].shape[1]
        decoded = [
            self.tokenizer.decode(row[prompt_size:], skip_special_tokens=True).strip()
            for row in outputs
        ]
        return decoded, elapsed

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


class TaskEncoder:
    def __init__(self, task: str):
        from src.encoder.processors import (
            get_material_encoder,
            get_pressure_processor,
            get_size_processor,
            get_standard_processor,
            get_thickness_processor,
            get_type_encoder,
        )

        self.task = task
        self.type_encoder = get_type_encoder()
        self.material_encoder = get_material_encoder()
        self.size_processor = get_size_processor()
        self.thickness_processor = get_thickness_processor()
        self.pressure_processor = get_pressure_processor()
        self.standard_processor = get_standard_processor()

    @staticmethod
    def _type_coder_value(value: Any) -> str:
        if not isinstance(value, dict):
            return serialize_value(value)
        geometry = value.get("GEOMETRY") if isinstance(value.get("GEOMETRY"), dict) else {}
        order = ("BODY", "ANGLE", "RADIUS", "FLANGE_STYLE", "CONN", "SEAL", "MANU")
        source = dict(value)
        source["ANGLE"] = geometry.get("ANGLE") or source.get("ANGLE")
        source["RADIUS"] = geometry.get("RADIUS") or source.get("RADIUS")
        payload = {key: source[key] for key in order if source.get(key) not in (None, "", [])}
        return serialize_value(payload)

    @staticmethod
    def _ordered_structural_items(value: dict[str, Any]) -> list[dict[str, Any]]:
        order = {"SINGLE": 0, "MAIN": 10, "END_A": 10, "BRANCH": 20, "END_B": 20}
        indexed = [(index, item) for index, item in enumerate(value.get("ITEMS") or []) if isinstance(item, dict)]
        indexed.sort(key=lambda pair: (order.get(str(pair[1].get("ROLE") or ""), 99), pair[0]))
        return [item for _, item in indexed]

    def _structural(self, parsed: dict[str, Any]) -> dict[str, Any]:
        from src.llm_ner.structural_field_output_normalizer import StructuralFieldOutputNormalizer

        return StructuralFieldOutputNormalizer.normalize(parsed)

    def encode(self, parsed: dict[str, Any] | None, raw: str) -> EncodedValue:
        if self.task == "code":
            code = clean_code_output(raw)
            return EncodedValue(code=code, source="模型直接编码", error="" if code else "编码输出为空")
        if parsed is None:
            return EncodedValue(error="一阶段 JSON 解析失败")
        try:
            return getattr(self, f"_encode_{self.task}")(parsed)
        except Exception as exc:
            return EncodedValue(error=f"规则编码异常: {exc}")

    def _encode_type(self, parsed: dict[str, Any]) -> EncodedValue:
        value = parsed.get("TYPE", parsed)
        result = self.type_encoder.encode(value) if isinstance(value, dict) else None
        code = str(getattr(result, "code", "") or "")
        return EncodedValue(code=code, coder_value=self._type_coder_value(value), error="" if code else "TYPE 规则未解析")

    def _encode_size(self, parsed: dict[str, Any]) -> EncodedValue:
        structural = self._structural(parsed)
        if structural.get("_schema_version") == "v2":
            codes: list[str] = []
            coder_values: list[str] = []
            for item in self._ordered_structural_items(structural):
                size_items = item.get("SIZE") or []
                if not size_items:
                    continue
                value: dict[str, Any] = {"_ITEMS": size_items}
                mm_values = [
                    str(entry.get("value") or "")
                    for entry in item.get("THICKNESS") or []
                    if isinstance(entry, dict) and str(entry.get("type") or "").upper() == "MM"
                ]
                if mm_values:
                    value["_THICKNESS_MM_CONTEXT"] = mm_values
                code = self.size_processor.process(value)
                if code and code not in codes:
                    codes.append(code)
                coder_values.extend(serialize_value(entry) for entry in size_items)
            length = str(structural.get("LENGTH") or "").strip()
            if length:
                length_code = self.size_processor.extract_length_prefix(
                    {"_ITEMS": [{"type": "LENGTH", "value": length}]}
                )
                if length_code:
                    if codes:
                        codes[-1] = f"{codes[-1]}{length_code}"
                    else:
                        codes.append(length_code)
            final = "x".join(codes)
            return EncodedValue(code=final, coder_value=" x ".join(coder_values), error="" if final else "SIZE 规则未解析")
        value = structural.get("SIZE") or {"_ITEMS": structural.get("SIZE_ITEMS") or []}
        code = self.size_processor.process(value)
        return EncodedValue(code=code, coder_value=serialize_value(value), error="" if code else "SIZE 规则未解析")

    def _encode_thickness(self, parsed: dict[str, Any]) -> EncodedValue:
        structural = self._structural(parsed)
        values: list[Any] = []
        if structural.get("_schema_version") == "v2":
            values = [item.get("THICKNESS") for item in self._ordered_structural_items(structural) if item.get("THICKNESS")]
        else:
            values = [structural.get("THICKNESS") or {"_ITEMS": structural.get("THICKNESS_ITEMS") or []}]
        codes = []
        for value in values:
            structured = {"_ITEMS": value} if isinstance(value, list) else value
            code = self.thickness_processor.process(structured)
            if code and code not in codes:
                codes.append(code)
        final = "X".join(codes)
        return EncodedValue(code=final, coder_value=" X ".join(map(serialize_value, values)), error="" if final else "THICKNESS 规则未解析")

    def _encode_pressure(self, parsed: dict[str, Any]) -> EncodedValue:
        structural = self._structural(parsed)
        value = str(structural.get("PRESSURE") or parsed.get("PRESSURE") or "").strip()
        code = self.pressure_processor.process(value)
        return EncodedValue(code=code, coder_value=value, error="" if code else "PRESSURE 规则未解析")

    def _encode_material(self, parsed: dict[str, Any]) -> EncodedValue:
        values = parsed.get("MATERIAL") or []
        if isinstance(values, dict):
            values = [values]
        codes, fallback_values = [], []
        for item in values if isinstance(values, list) else []:
            if not isinstance(item, dict):
                fallback_values.append(str(item))
                continue
            result = self.material_encoder.encode(item)
            if result.code:
                codes.append(result.code)
            else:
                fallback_values.append(" ".join([
                    str(item.get("VALUE") or "").strip(),
                    *[str(value).strip() for value in item.get("SPECIAL_REQ") or []],
                ]).strip())
        final = self.material_encoder.apply_composition_override("".join(codes))
        return EncodedValue(
            code=final if not fallback_values else "",
            coder_value=" / ".join(fallback_values or [serialize_value(values)]),
            error="" if final and not fallback_values else "MATERIAL 规则未完全解析",
        )

    def _encode_standard(self, parsed: dict[str, Any]) -> EncodedValue:
        values = parsed.get("STANDARD") or []
        if isinstance(values, dict):
            values = [values]
        bodies = [
            str(item.get("BODY") or "").strip() if isinstance(item, dict) else str(item).strip()
            for item in values if item not in (None, "")
        ]
        bodies = [value for value in bodies if value]
        code = self.standard_processor.process_multi(bodies)
        return EncodedValue(code=code, coder_value=" / ".join(bodies), error="" if code else "STANDARD 规则未解析")


def normalize_code(value: Any, comparison: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if comparison.get("remove_whitespace", True):
        text = re.sub(r"\s+", "", text)
    return text.upper() if comparison.get("ignore_case", True) else text


def run_stage1(
    model_config: dict[str, Any],
    descriptions: list[str],
    instruction: str,
    settings: dict[str, Any],
    managed_service: ManagedVLLMService | None = None,
) -> tuple[list[dict[str, Any]], float]:
    if model_config.get("backend") == MANAGED_VLLM_BACKEND:
        if managed_service is not None:
            return run_vllm_stage1(model_config, descriptions, settings, managed_service), 0.0
        vllm_settings = settings.get("_managed_vllm") or {}
        with ManagedVLLMService(model_config["service_config"], vllm_settings) as service:
            records = run_vllm_stage1(model_config, descriptions, settings, service)
            return records, service.load_seconds

    batch_size = max(1, int(settings.get("batch_size", 1)))
    LOGGER.info("加载一阶段模型 %s，GPU batch_size=%d", model_config["name"], batch_size)
    runner = TransformersRunner(model_config["base_model_path"], model_config["adapter_path"], settings)
    load_started = time.perf_counter()
    runner.load()
    load_seconds = time.perf_counter() - load_started
    records: list[dict[str, Any]] = []
    try:
        for start in range(0, len(descriptions), batch_size):
            batch = descriptions[start:start + batch_size]
            outputs, elapsed = runner.generate(
                batch,
                instruction,
                max_new_tokens=int(settings.get("max_new_tokens", 512)),
            )
            per_item = elapsed / max(1, len(batch))
            for raw in outputs:
                records.append({"raw": raw, "parsed": parse_json_output(raw), "stage1_seconds": per_item})
            LOGGER.info("%s: %d/%d", model_config["name"], min(start + len(batch), len(descriptions)), len(descriptions))
    finally:
        runner.close()
    return records, load_seconds


def stage_model_signature(model_config: dict[str, Any], instruction: str = "") -> tuple[str, ...]:
    if model_config.get("backend") == MANAGED_VLLM_BACKEND:
        return (
            MANAGED_VLLM_BACKEND,
            str(model_config["service_config"]),
            str(model_config["served_model"]),
        )
    return (
        LOCAL_TRANSFORMERS_BACKEND,
        str(model_config["base_model_path"]),
        str(model_config["adapter_path"]),
        instruction,
    )


def run_stage1_jobs(
    jobs: dict[tuple[str, ...], tuple[dict[str, Any], str]],
    descriptions: list[str],
    settings: dict[str, Any],
) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], dict[tuple[str, ...], float]]:
    """执行去重后的一阶段任务；共享 service_config 的 vLLM route 共用一次服务启动。"""
    records_by_job: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    load_by_job: dict[tuple[str, ...], float] = {}
    managed_groups: dict[Path, list[tuple[tuple[str, ...], dict[str, Any], str]]] = {}

    for key, (model_config, instruction) in jobs.items():
        if model_config.get("backend") == MANAGED_VLLM_BACKEND:
            managed_groups.setdefault(model_config["service_config"], []).append(
                (key, model_config, instruction)
            )
        else:
            records, load_seconds = run_stage1(model_config, descriptions, instruction, settings)
            records_by_job[key] = records
            load_by_job[key] = load_seconds

    vllm_settings = settings.get("_managed_vllm") or {}
    for service_config, service_jobs in managed_groups.items():
        with ManagedVLLMService(service_config, vllm_settings) as service:
            for index, (key, model_config, instruction) in enumerate(service_jobs):
                records, _ = run_stage1(
                    model_config,
                    descriptions,
                    instruction,
                    settings,
                    managed_service=service,
                )
                records_by_job[key] = records
                load_by_job[key] = service.load_seconds if index == 0 else 0.0

    return records_by_job, load_by_job


def run_local_coder(
    requests: list[CoderRequest],
    coder_config: dict[str, Any],
    settings: dict[str, Any],
    config_path: Path,
) -> list[tuple[str, float]]:
    base = resolve_path(coder_config.get("base_model_path"), config_path, required=True)
    adapter = resolve_path(coder_config.get("adapter_path"), config_path, required=True)
    prompt = resolve_path(coder_config.get("prompt_path"), config_path, required=True)
    instruction = prompt.read_text(encoding="utf-8").strip()
    runner = TransformersRunner(base, adapter, settings)
    runner.load()
    results: list[tuple[str, float]] = []
    batch_size = max(1, int(settings.get("batch_size", 1)))
    try:
        inputs = [
            f"字段类型: {request.field}\n{'规范值' if request.field == 'TYPE' else '原始值'}: {request.value}"
            for request in requests
        ]
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            outputs, elapsed = runner.generate(
                batch,
                instruction,
                max_new_tokens=int(settings.get("coder_max_new_tokens", 64)),
            )
            per_item = elapsed / max(1, len(batch))
            results.extend((clean_code_output(raw), per_item) for raw in outputs)
    finally:
        runner.close()
    return results


def run_service_coder(requests: list[CoderRequest], coder_config: dict[str, Any]) -> list[tuple[str, float]]:
    from src.llm_ner.predictor import Qwen3Predictor

    predictor = Qwen3Predictor(
        model_name=str(coder_config.get("model_name") or "coder"),
        backend="mlx_service",
        service_url=str(coder_config.get("service_url") or "http://127.0.0.1:8200"),
        ollama_temperature=0.0,
        ollama_logprobs_enabled=False,
        request_timeout=int(coder_config.get("timeout", 300)),
        stage2_system_prompt=None,
    )
    results = []
    for request in requests:
        started = time.perf_counter()
        codes, _ = predictor.encode_with_confidence({request.field: request.value})
        results.append((str(codes.get(request.field) or "").strip(), time.perf_counter() - started))
    return results


def run_managed_vllm_coder(
    requests: list[CoderRequest],
    coder_config: dict[str, Any],
    settings: dict[str, Any],
    config_path: Path,
) -> list[tuple[str, float]]:
    service_config = resolve_path(coder_config.get("service_config"), config_path, required=True)
    served_model = str(coder_config.get("served_model") or "coder").strip()
    vllm_settings = settings.get("_managed_vllm") or {}
    concurrency = max(1, int(vllm_settings.get("concurrency", 16)))
    timeout = float(vllm_settings.get("request_timeout", 600))
    retries = max(0, int(vllm_settings.get("retries", 2)))
    max_new_tokens = int(coder_config.get("max_new_tokens") or settings.get("coder_max_new_tokens", 64))
    results: list[tuple[str, float] | None] = [None] * len(requests)

    def coder_text(request: CoderRequest) -> str:
        value_label = "规范值" if request.field == "TYPE" else "原始值"
        return f"字段类型: {request.field}\n{value_label}: {request.value}"

    with ManagedVLLMService(service_config, vllm_settings) as service:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {}
            next_index = 0

            def submit(index: int):
                return pool.submit(
                    _post_vllm_predict,
                    service=service,
                    served_model=served_model,
                    text=coder_text(requests[index]),
                    max_new_tokens=max_new_tokens,
                    timeout=timeout,
                    retries=retries,
                )

            while next_index < len(requests) and len(pending) < concurrency:
                future = submit(next_index)
                pending[future] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    record = future.result()
                    results[index] = (clean_code_output(record["raw"]), float(record["stage1_seconds"]))
                    if next_index < len(requests):
                        replacement = submit(next_index)
                        pending[replacement] = next_index
                        next_index += 1
    if any(result is None for result in results):
        raise RuntimeError("vLLM coder 结果数量与请求不一致")
    return [result for result in results if result is not None]


class _NullCoderBridge:
    """保留正式编码器的结构化 TYPE/材质规则流程，但不调用模型兜底。"""

    elapsed_seconds = 0.0
    load_seconds = 0.0

    @staticmethod
    def encode_with_confidence(entities: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        return {}, {field: None for field in entities}

    @staticmethod
    def close() -> None:
        return None


class _LocalCoderBridge:
    """将本地 Transformers coder 适配成 LlmPipeEncoder 所需的接口。"""

    def __init__(self, coder_config: dict[str, Any], settings: dict[str, Any], config_path: Path):
        base = resolve_path(coder_config.get("base_model_path"), config_path, required=True)
        adapter = resolve_path(coder_config.get("adapter_path"), config_path, required=True)
        prompt = resolve_path(coder_config.get("prompt_path"), config_path, required=True)
        self.instruction = prompt.read_text(encoding="utf-8").strip()
        self.settings = settings
        self.runner = TransformersRunner(base, adapter, settings)
        started = time.perf_counter()
        self.runner.load()
        self.load_seconds = time.perf_counter() - started
        self.elapsed_seconds = 0.0

    def encode_with_confidence(self, entities: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        codes: dict[str, str] = {}
        confidences: dict[str, Any] = {}
        for field, value in entities.items():
            if value in (None, "", [], {}):
                continue
            value_label = "规范值" if str(field).upper() == "TYPE" else "原始值"
            outputs, elapsed = self.runner.generate(
                [f"字段类型: {field}\n{value_label}: {value}"],
                self.instruction,
                max_new_tokens=int(self.settings.get("coder_max_new_tokens", 64)),
            )
            self.elapsed_seconds += elapsed
            codes[str(field)] = clean_code_output(outputs[0] if outputs else "")
            # TransformersRunner 当前不返回 token logprob，不伪造置信度。
            confidences[str(field)] = None
        return codes, confidences

    def close(self) -> None:
        self.runner.close()


class _ServiceCoderBridge:
    def __init__(self, coder_config: dict[str, Any]):
        from src.llm_ner.predictor import Qwen3Predictor

        self.predictor = Qwen3Predictor(
            model_name=str(coder_config.get("model_name") or "coder"),
            backend="mlx_service",
            service_url=str(coder_config.get("service_url") or "http://127.0.0.1:8200"),
            ollama_temperature=0.0,
            ollama_logprobs_enabled=False,
            request_timeout=int(coder_config.get("timeout", 300)),
            stage2_system_prompt=None,
        )
        self.elapsed_seconds = 0.0
        self.load_seconds = 0.0

    def encode_with_confidence(self, entities: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        result = self.predictor.encode_with_confidence(entities)
        self.elapsed_seconds += time.perf_counter() - started
        return result

    @staticmethod
    def close() -> None:
        return None


class _ManagedVLLMCoderBridge:
    def __init__(self, coder_config: dict[str, Any], settings: dict[str, Any], config_path: Path):
        self.service_config = resolve_path(coder_config.get("service_config"), config_path, required=True)
        self.served_model = str(coder_config.get("served_model") or "coder").strip()
        self.vllm_settings = settings.get("_managed_vllm") or {}
        self.timeout = float(self.vllm_settings.get("request_timeout", 600))
        self.retries = max(0, int(self.vllm_settings.get("retries", 2)))
        self.max_new_tokens = int(
            coder_config.get("max_new_tokens") or settings.get("coder_max_new_tokens", 64)
        )
        self.service = ManagedVLLMService(self.service_config, self.vllm_settings).start()
        self.elapsed_seconds = 0.0
        self.load_seconds = self.service.load_seconds

    def encode_with_confidence(self, entities: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        codes: dict[str, str] = {}
        confidences: dict[str, Any] = {}
        for field, value in entities.items():
            if value in (None, "", [], {}):
                continue
            value_label = "规范值" if str(field).upper() == "TYPE" else "原始值"
            record = _post_vllm_predict(
                service=self.service,
                served_model=self.served_model,
                text=f"字段类型: {field}\n{value_label}: {value}",
                max_new_tokens=self.max_new_tokens,
                timeout=self.timeout,
                retries=self.retries,
            )
            self.elapsed_seconds += float(record["stage1_seconds"])
            codes[str(field)] = clean_code_output(record["raw"])
            confidences[str(field)] = None
        return codes, confidences

    def close(self) -> None:
        self.service.stop()


class FullEncoderSession:
    """在评测器配置下复用生产 LlmPipeEncoder 的完整规则与组装流程。"""

    def __init__(self, coder_config: dict[str, Any], settings: dict[str, Any], config_path: Path):
        from src.encoder.pipe_encoder import PipeEncoderBase
        from src.encoder.pipe_encoder_llm import LlmPipeEncoder
        from src.encoder.processors import get_material_encoder, get_type_encoder

        fallback_fields = {
            str(value).strip().upper()
            for value in coder_config.get("fallback_fields") or []
            if str(value).strip()
        }

        # 先初始化规则编码器，再启动可能长期占用 GPU 的 coder 服务；初始化失败时不会遗留进程。
        self.encoder = LlmPipeEncoder.__new__(LlmPipeEncoder)
        PipeEncoderBase.__init__(self.encoder)
        self.encoder.type_encoder = get_type_encoder()
        self.encoder.material_encoder = get_material_encoder()
        self.encoder._last_type_encode_meta = {}

        if coder_config.get("enabled", True) and fallback_fields:
            backend = str(coder_config.get("backend") or "local_transformers")
            if backend == "mlx_service":
                self.bridge = _ServiceCoderBridge(coder_config)
            elif backend == MANAGED_VLLM_BACKEND:
                self.bridge = _ManagedVLLMCoderBridge(coder_config, settings, config_path)
            elif backend == "local_transformers":
                self.bridge = _LocalCoderBridge(coder_config, settings, config_path)
            else:
                raise ValueError("coder.backend 只能是 local_transformers、managed_vllm 或 mlx_service")
        else:
            self.bridge = _NullCoderBridge()
            fallback_fields = set()

        # 绕过 LlmPipeEncoder.__init__ 中的固定平台服务配置，改用评测器 coder 配置。
        self.encoder.llm_encoder = self.bridge
        self.encoder.backend = str(coder_config.get("backend") or "disabled")
        self.encoder.fallback_fields = fallback_fields

    def close(self) -> None:
        self.bridge.close()


def merge_full_stage_records(stage_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """将三个一阶段 LoRA 输出合并成 PipeEncoder 的标准实体结构。"""
    from src.domain.structural_v2 import STRUCTURAL_V2_FIELD
    from src.llm_ner.structural_field_output_normalizer import StructuralFieldOutputNormalizer

    entities: dict[str, Any] = {}
    raw_outputs: dict[str, str] = {}
    errors: list[str] = []
    material_category = ""
    total_seconds = 0.0

    for stage in FULL_STAGE_DEFAULTS:
        record = stage_records.get(stage) or {}
        raw_outputs[stage] = str(record.get("raw") or "")
        total_seconds += float(record.get("stage1_seconds") or 0.0)
        parsed = record.get("parsed")
        if not isinstance(parsed, dict):
            errors.append(f"{stage} 一阶段 JSON 解析失败")
            continue

        if stage == "type":
            material_category = str(parsed.get("CATEGORY") or parsed.get("category") or "").strip()
            type_value = parsed.get("TYPE")
            if type_value not in (None, "", [], {}):
                entities["TYPE"] = copy.deepcopy(type_value)
        elif stage == "material":
            for field in ("MATERIAL", "STANDARD"):
                value = parsed.get(field)
                if value not in (None, "", [], {}):
                    entities[field] = copy.deepcopy(value)
        elif stage == "size":
            structural = StructuralFieldOutputNormalizer.normalize(parsed)
            if structural.get("_schema_version") == "v2":
                entities[STRUCTURAL_V2_FIELD] = structural
            else:
                for field in ("SIZE", "THICKNESS", "PRESSURE"):
                    value = structural.get(field)
                    if isinstance(value, dict) and not (value.get("_ITEMS") or []):
                        continue
                    if value not in (None, "", [], {}):
                        entities[field] = copy.deepcopy(value)

    return {
        "raw": serialize_value(raw_outputs),
        "parsed": entities,
        "stage1_seconds": total_seconds,
        "material_category": material_category,
        "errors": errors,
    }


def _full_result_used_coder(result: Any) -> bool:
    for field_result in (getattr(result, "fields", {}) or {}).values():
        meta = getattr(field_result, "encode_confidence_v2", {}) or {}
        if isinstance(meta, dict) and str(meta.get("source") or "") == "llm_fallback":
            return True
        for item in getattr(field_result, "detail_items", []) or []:
            item_meta = item.get("encode_meta") if isinstance(item, dict) else None
            if isinstance(item_meta, dict) and str(item_meta.get("source") or "") == "llm_fallback":
                return True
    return False


def run_full_evaluation(
    runtime: dict[str, Any],
    descriptions: list[str],
    settings: dict[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[EncodedValue]],
    dict[str, float],
    dict[tuple[str, int], float],
    dict[str, list[str]],
]:
    stage1_by_model: dict[str, list[dict[str, Any]]] = {}
    encoded_by_model: dict[str, list[EncodedValue]] = {}
    model_load_seconds: dict[str, float] = {}
    stage2_elapsed: dict[tuple[str, int], float] = {}
    full_details: dict[str, list[str]] = {}
    instructions = {
        stage: path.read_text(encoding="utf-8").strip()
        for stage, path in runtime["prompt_paths"].items()
    }

    jobs: dict[tuple[str, ...], tuple[dict[str, Any], str]] = {}
    keys_by_group: dict[str, dict[str, tuple[str, ...]]] = {}
    for model in runtime["models"]:
        group_keys = {}
        for stage in runtime["full_stages"]:
            stage_model = model["stages"][stage]
            key = stage_model_signature(stage_model, instructions[stage])
            jobs.setdefault(key, (stage_model, instructions[stage]))
            group_keys[stage] = key
        keys_by_group[model["name"]] = group_keys

    cached_records, cached_load_seconds = run_stage1_jobs(jobs, descriptions, settings)

    for model in runtime["models"]:
        records_by_stage = {
            stage: cached_records[keys_by_group[model["name"]][stage]]
            for stage in runtime["full_stages"]
        }
        stage1_by_model[model["name"]] = [
            merge_full_stage_records({stage: records_by_stage[stage][index] for stage in runtime["full_stages"]})
            for index in range(len(descriptions))
        ]
        model_load_seconds[model["name"]] = sum(
            cached_load_seconds[keys_by_group[model["name"]][stage]]
            for stage in runtime["full_stages"]
        )

    session = FullEncoderSession(runtime["config"].get("coder") or {}, settings, runtime["config_path"])
    try:
        for model in runtime["models"]:
            name = model["name"]
            encoded_values: list[EncodedValue] = []
            detail_values: list[str] = []
            for index, (description, record) in enumerate(zip(descriptions, stage1_by_model[name])):
                started = time.perf_counter()
                try:
                    entities = copy.deepcopy(record["parsed"])
                    raw_snapshot = copy.deepcopy(entities)
                    result = session.encoder.encode(
                        entities,
                        description,
                        stage1_raw_snapshot=raw_snapshot,
                        material_category=record["material_category"],
                    )
                    elapsed = time.perf_counter() - started
                    errors = [*record["errors"], *(getattr(result, "errors", []) or [])]
                    used_coder = _full_result_used_coder(result)
                    encoded_values.append(
                        EncodedValue(
                            code=str(result.final_code or ""),
                            source="完整编码器+coder兜底" if used_coder else "完整编码器",
                            error="; ".join(str(value) for value in errors if str(value)),
                        )
                    )
                    detail_values.append(serialize_value(result.to_dict()))
                except Exception as exc:
                    elapsed = time.perf_counter() - started
                    encoded_values.append(EncodedValue(error=f"完整编码异常: {exc}"))
                    detail_values.append("")
                stage2_elapsed[(name, index)] = elapsed
            encoded_by_model[name] = encoded_values
            full_details[name] = detail_values
    finally:
        session.close()

    return stage1_by_model, encoded_by_model, model_load_seconds, stage2_elapsed, full_details


def pairwise_stats(
    detail: pd.DataFrame,
    model_names: list[str],
    comparison: dict[str, Any] | None = None,
) -> pd.DataFrame:
    comparison = comparison or {}
    rows = []
    total = len(detail)
    for left, right in itertools.combinations(model_names, 2):
        left_code = detail[f"{left}_最终编码"].map(lambda value: normalize_code(value, comparison))
        right_code = detail[f"{right}_最终编码"].map(lambda value: normalize_code(value, comparison))
        left_ok = detail[f"{left}_是否正确"].astype(bool)
        right_ok = detail[f"{right}_是否正确"].astype(bool)
        disagree = left_code != right_code
        rows.append({
            "模型A": left,
            "模型B": right,
            "样本数": total,
            "预测不一致数": int(disagree.sum()),
            "预测不一致率": float(disagree.mean()) if total else 0.0,
            "两者都正确": int((left_ok & right_ok).sum()),
            "仅模型A正确": int((left_ok & ~right_ok).sum()),
            "仅模型B正确": int((~left_ok & right_ok).sum()),
            "两者都错误": int((~left_ok & ~right_ok).sum()),
        })
    return pd.DataFrame(rows)


def build_summary(
    detail: pd.DataFrame,
    model_names: list[str],
    model_load_seconds: dict[str, float] | None = None,
) -> pd.DataFrame:
    model_load_seconds = model_load_seconds or {}
    rows = []
    for name in model_names:
        latency = pd.to_numeric(detail[f"{name}_总耗时秒"], errors="coerce").fillna(0.0)
        correct = detail[f"{name}_是否正确"].astype(bool)
        errors = detail[f"{name}_错误"].fillna("").astype(str).str.len() > 0
        fallback = detail[f"{name}_编码来源"].fillna("").astype(str).str.contains("coder兜底", regex=False)
        rows.append({
            "模型": name,
            "样本数": len(detail),
            "正确数": int(correct.sum()),
            "错误数": int((~correct).sum()),
            "准确率": float(correct.mean()) if len(detail) else 0.0,
            "coder兜底数": int(fallback.sum()),
            "异常数": int(errors.sum()),
            "模型加载秒": float(model_load_seconds.get(name, 0.0)),
            "总耗时秒": float(latency.sum()),
            "平均耗时秒": float(latency.mean()) if len(latency) else 0.0,
            "P50耗时秒": float(latency.quantile(0.5)) if len(latency) else 0.0,
            "P95耗时秒": float(latency.quantile(0.95)) if len(latency) else 0.0,
        })
    return pd.DataFrame(rows)


def draw_analysis(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    image_paths: dict[str, Path],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = summary["模型"].tolist()
    for path in image_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    accuracy = summary["准确率"].to_numpy(dtype=float)
    axis.bar(names, accuracy, color="#1f9d8a")
    axis.set_ylim(0, 1.05)
    axis.set_title("Final-code accuracy")
    axis.set_ylabel("Accuracy")
    for index, value in enumerate(accuracy):
        axis.text(index, value + 0.02, f"{value:.2%}", ha="center")
    figure.tight_layout()
    figure.savefig(image_paths["accuracy"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    avg = summary["平均耗时秒"].to_numpy(dtype=float)
    p95 = summary["P95耗时秒"].to_numpy(dtype=float)
    x = np.arange(len(names))
    width = 0.36
    axis.bar(x - width / 2, avg, width, label="Average", color="#e88d3d")
    axis.bar(x + width / 2, p95, width, label="P95", color="#355f8d")
    axis.set_xticks(x, names)
    axis.set_title("Latency per sample")
    axis.set_ylabel("Seconds")
    axis.legend()
    figure.tight_layout()
    figure.savefig(image_paths["latency"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    matrix = np.zeros((len(names), len(names)), dtype=float)
    lookup = {(row["模型A"], row["模型B"]): row["预测不一致率"] for _, row in pairwise.iterrows()}
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i != j:
                matrix[i, j] = lookup.get((left, right), lookup.get((right, left), 0.0))
    image = axis.imshow(matrix, vmin=0, vmax=max(0.01, float(matrix.max())), cmap="YlOrRd")
    axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Pairwise disagreement")
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(image_paths["disagreement"], dpi=180, bbox_inches="tight")
    plt.close(figure)


def export_excel(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    runtime_info: dict[str, Any],
    output_path: Path,
    image_paths: dict[str, Path],
) -> None:
    from openpyxl.drawing.image import Image
    from openpyxl.styles import Alignment, Font, PatternFill

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_rows = pd.DataFrame([{"配置项": key, "值": serialize_value(value)} for key, value in runtime_info.items()])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        detail.to_excel(writer, index=False, sheet_name="评测明细")
        summary.to_excel(writer, index=False, sheet_name="模型汇总")
        pairwise.to_excel(writer, index=False, sheet_name="模型差异")
        config_rows.to_excel(writer, index=False, sheet_name="运行配置")
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F5B66")
                cell.alignment = Alignment(horizontal="center")
            for column_cells in sheet.columns:
                width = min(50, max(10, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                sheet.column_dimensions[column_cells[0].column_letter].width = width
        summary_sheet = writer.book["模型汇总"]
        anchors = {"accuracy": "M2", "latency": "M31", "disagreement": "M60"}
        for name, image_path in image_paths.items():
            if not image_path.is_file():
                continue
            chart = Image(str(image_path))
            chart.width = 720
            chart.height = 480
            summary_sheet.add_image(chart, anchors[name])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    runtime = prepare_runtime(args)
    description_column, truth_column, detail = validate_runtime(runtime, args)
    LOGGER.info("配置有效，任务=%s，样本=%d，模型=%s", runtime["task"], len(detail), ", ".join(m["name"] for m in runtime["models"]))
    if args.validate_only:
        print("配置、模型路径、提示词和 Excel 校验通过")
        return

    artifacts = [runtime["output_path"], *runtime["image_paths"].values()]
    existing = [path for path in artifacts if path.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(f"结果文件已存在，请更换 --output-dir 或增加 --overwrite:\n{paths}")

    settings = dict(runtime["config"].get("inference") or {})
    settings["_managed_vllm"] = dict(runtime["config"].get("vllm") or {})
    comparison = runtime["config"].get("comparison") or {}
    descriptions = detail[description_column].astype(str).tolist()
    truth = detail[truth_column].astype(str).tolist()
    coder_config = runtime["config"].get("coder") or {}
    full_details: dict[str, list[str]] = {}

    if runtime["task"] == "full":
        (
            stage1_by_model,
            encoded_by_model,
            model_load_seconds,
            coder_elapsed,
            full_details,
        ) = run_full_evaluation(runtime, descriptions, settings)
    else:
        instruction = runtime["prompt_path"].read_text(encoding="utf-8").strip()
        encoder = TaskEncoder(runtime["task"])
        encoded_by_model = {}
        stage1_by_model = {}
        model_load_seconds = {}
        coder_requests: list[CoderRequest] = []
        fallback_fields = {str(value).upper() for value in coder_config.get("fallback_fields") or []}

        jobs = {}
        keys_by_name = {}
        for model in runtime["models"]:
            key = stage_model_signature(model, instruction)
            jobs.setdefault(key, (model, instruction))
            keys_by_name[model["name"]] = key
        cached_records, cached_load_seconds = run_stage1_jobs(jobs, descriptions, settings)

        for model in runtime["models"]:
            key = keys_by_name[model["name"]]
            stage1 = cached_records[key]
            load_seconds = cached_load_seconds[key]
            encoded = [encoder.encode(record["parsed"], record["raw"]) for record in stage1]
            stage1_by_model[model["name"]] = stage1
            model_load_seconds[model["name"]] = load_seconds
            encoded_by_model[model["name"]] = encoded
            if coder_config.get("enabled", True) and runtime["task_field"] in fallback_fields:
                for index, value in enumerate(encoded):
                    if not value.code and value.coder_value:
                        coder_requests.append(CoderRequest(model["name"], index, runtime["task_field"], value.coder_value))

        coder_elapsed = {}
        if coder_requests:
            LOGGER.info("规则未解析 %d 项，启动二阶段 coder 兜底", len(coder_requests))
            backend = str(coder_config.get("backend") or "local_transformers")
            if backend == "mlx_service":
                coder_results = run_service_coder(coder_requests, coder_config)
            elif backend == MANAGED_VLLM_BACKEND:
                coder_results = run_managed_vllm_coder(
                    coder_requests,
                    coder_config,
                    settings,
                    runtime["config_path"],
                )
            else:
                coder_results = run_local_coder(coder_requests, coder_config, settings, runtime["config_path"])
            for request, (code, elapsed) in zip(coder_requests, coder_results):
                target = encoded_by_model[request.model_name][request.row_index]
                if code:
                    target.code = code
                    target.source = "coder兜底"
                    target.error = ""
                else:
                    target.error = f"{target.error}; coder兜底输出为空".strip("; ")
                coder_elapsed[(request.model_name, request.row_index)] = elapsed

    model_names = [model["name"] for model in runtime["models"]]
    for name in model_names:
        records = stage1_by_model[name]
        encoded = encoded_by_model[name]
        detail[f"{name}_一阶段原始输出"] = [record["raw"] for record in records]
        detail[f"{name}_一阶段识别结果"] = [
            serialize_value(record["parsed"]) if record["parsed"] is not None else record["raw"]
            for record in records
        ]
        if runtime["task"] == "full":
            detail[f"{name}_完整编码明细"] = full_details[name]
        detail[f"{name}_最终编码"] = [value.code for value in encoded]
        detail[f"{name}_编码来源"] = [value.source for value in encoded]
        detail[f"{name}_是否正确"] = [
            normalize_code(value.code, comparison) == normalize_code(expected, comparison)
            for value, expected in zip(encoded, truth)
        ]
        detail[f"{name}_一阶段耗时秒"] = [round(record["stage1_seconds"], 6) for record in records]
        detail[f"{name}_二阶段耗时秒"] = [round(coder_elapsed.get((name, index), 0.0), 6) for index in range(len(detail))]
        detail[f"{name}_总耗时秒"] = [
            round(record["stage1_seconds"] + coder_elapsed.get((name, index), 0.0), 6)
            for index, record in enumerate(records)
        ]
        detail[f"{name}_错误"] = [value.error for value in encoded]

    summary = build_summary(detail, model_names, model_load_seconds)
    pairwise = pairwise_stats(detail, model_names, comparison)
    draw_analysis(summary, pairwise, runtime["image_paths"])
    export_excel(
        detail,
        summary,
        pairwise,
        {
            "任务": runtime["task"],
            "配置文件": runtime["config_path"],
            "输入Excel": runtime["input_path"],
            "输出目录": runtime["output_dir"],
            "输出Excel": runtime["output_path"],
            "提示词": runtime["prompt_paths"],
            "真值列": truth_column,
            "样本数": len(detail),
            "模型": model_names,
            "coder启用": bool(coder_config.get("enabled", True)),
            "coder后端": coder_config.get("backend", "local_transformers"),
            "生成时间": datetime.now().isoformat(timespec="seconds"),
        },
        runtime["output_path"],
        runtime["image_paths"],
    )
    print(summary.to_string(index=False))
    print(f"\n结果目录: {runtime['output_dir']}")
    print(f"Excel: {runtime['output_path']}")
    for image_path in runtime["image_paths"].values():
        print(f"分析图: {image_path}")


if __name__ == "__main__":
    main()
