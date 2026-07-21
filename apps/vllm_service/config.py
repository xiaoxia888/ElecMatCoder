# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("service.yaml")

PROFILE_ALLOWED_TOP_LEVEL = {"gateway", "engines"}
PROFILE_ALLOWED_GATEWAY_FIELDS = {
    "request_timeout_seconds",
    "startup_timeout_seconds",
    "health_check_interval_seconds",
}
PROFILE_ALLOWED_ENGINE_FIELDS = {
    "cuda_visible_devices",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "max_num_seqs",
    "enforce_eager",
    "enable_prefix_caching",
    "quantization",
    "kv_cache_dtype",
    "cpu_offload_gb",
    "environment",
    "extra_args",
}
PROFILE_REQUIRED_ENGINE_FIELDS = {
    "cuda_visible_devices",
    "dtype",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "max_num_seqs",
}


def _expand(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(value or "").strip()))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GatewaySpec:
    host: str = "0.0.0.0"
    port: int = 8200
    request_timeout_seconds: int = 300
    startup_timeout_seconds: int = 900
    health_check_interval_seconds: float = 2.0


@dataclass(frozen=True)
class EngineSpec:
    name: str
    host: str
    port: int
    model_path: str
    served_model_name: str
    cuda_visible_devices: str
    dtype: str = "float16"
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 16
    trust_remote_code: bool = True
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    quantization: str = ""
    kv_cache_dtype: str = "auto"
    cpu_offload_gb: float = 0.0
    api_key: str = ""
    max_loras: int = 1
    max_cpu_loras: int = 1
    max_lora_rank: int = 16
    lora_modules: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class ModelRoute:
    name: str
    engine: str
    upstream_model: str
    instruction: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeploymentConfig:
    gateway: GatewaySpec
    engines: dict[str, EngineSpec]
    models: dict[str, ModelRoute]
    profile_name: str
    profile_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 对象: {path}")
    return data


def _resolve_profile_path(config_path: Path, profile: str | Path) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(profile).strip()))
    if not raw:
        raise ValueError("profile 不能为空")

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parent != Path("."):
        return (config_path.parent / candidate).resolve()

    filename = candidate.name if candidate.suffix else f"{candidate.name}.yaml"
    return (config_path.parent / "profiles" / filename).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_profile(data: dict[str, Any], base: dict[str, Any], path: Path) -> None:
    unknown_top_level = set(data) - PROFILE_ALLOWED_TOP_LEVEL
    if unknown_top_level:
        raise ValueError(
            f"硬件Profile不允许配置这些顶层字段: {sorted(unknown_top_level)} ({path})"
        )

    raw_gateway = data.get("gateway") or {}
    if not isinstance(raw_gateway, dict):
        raise ValueError(f"Profile gateway 必须是对象: {path}")
    unknown_gateway = set(raw_gateway) - PROFILE_ALLOWED_GATEWAY_FIELDS
    if unknown_gateway:
        raise ValueError(
            f"硬件Profile不允许覆盖这些gateway字段: {sorted(unknown_gateway)} ({path})"
        )

    raw_engines = data.get("engines") or {}
    if not isinstance(raw_engines, dict):
        raise ValueError(f"Profile engines 必须是对象: {path}")
    base_engines = base.get("engines") or {}
    unknown_engines = set(raw_engines) - set(base_engines)
    if unknown_engines:
        raise ValueError(f"硬件Profile引用了未知engine: {sorted(unknown_engines)} ({path})")
    missing_engines = set(base_engines) - set(raw_engines)
    if missing_engines:
        raise ValueError(f"硬件Profile缺少engine配置: {sorted(missing_engines)} ({path})")
    for name, row in raw_engines.items():
        if not isinstance(row, dict):
            raise ValueError(f"Profile engine {name} 必须是对象: {path}")
        unknown_fields = set(row) - PROFILE_ALLOWED_ENGINE_FIELDS
        if unknown_fields:
            raise ValueError(
                f"硬件Profile不允许覆盖engine {name}字段: "
                f"{sorted(unknown_fields)} ({path})"
            )
        missing_fields = PROFILE_REQUIRED_ENGINE_FIELDS - set(row)
        if missing_fields:
            raise ValueError(
                f"硬件Profile的engine {name}缺少必填字段: "
                f"{sorted(missing_fields)} ({path})"
            )


def load_config(path: Path, profile: str | Path | None = None) -> DeploymentConfig:
    path = path.expanduser().resolve()
    base_data = _load_yaml(path)
    selected_profile = profile if profile is not None else base_data.get("profile")
    if selected_profile is None or not str(selected_profile).strip():
        raise ValueError(f"service.yaml必须配置顶层profile字段: {path}")
    profile_path = _resolve_profile_path(path, selected_profile)
    if not profile_path.is_file():
        raise FileNotFoundError(f"硬件Profile不存在: {profile_path}")
    profile_data = _load_yaml(profile_path)
    _validate_profile(profile_data, base_data, profile_path)
    data = _deep_merge(base_data, profile_data)

    raw_gateway = data.get("gateway") or {}
    gateway = GatewaySpec(
        host=str(raw_gateway.get("host", "0.0.0.0")),
        port=int(raw_gateway.get("port", 8200)),
        request_timeout_seconds=int(raw_gateway.get("request_timeout_seconds", 300)),
        startup_timeout_seconds=int(raw_gateway.get("startup_timeout_seconds", 900)),
        health_check_interval_seconds=float(raw_gateway.get("health_check_interval_seconds", 2.0)),
    )

    engines: dict[str, EngineSpec] = {}
    for name, row in (data.get("engines") or {}).items():
        if not isinstance(row, dict):
            raise ValueError(f"engine {name} 配置必须是对象")
        raw_environment = row.get("environment") or {}
        if not isinstance(raw_environment, dict):
            raise ValueError(f"engine {name} environment 配置必须是对象")
        lora_modules = {
            str(alias).strip(): _expand(adapter_path)
            for alias, adapter_path in (row.get("lora_modules") or {}).items()
            if str(alias).strip() and _expand(adapter_path)
        }
        engine = EngineSpec(
            name=str(name),
            host=str(row.get("host", "127.0.0.1")),
            port=int(row["port"]),
            model_path=_expand(row["model_path"]),
            served_model_name=str(row.get("served_model_name") or f"{name}-base"),
            cuda_visible_devices=str(row.get("cuda_visible_devices", "0")),
            dtype=str(row.get("dtype", "float16")),
            max_model_len=int(row.get("max_model_len", 1024)),
            gpu_memory_utilization=float(row.get("gpu_memory_utilization", 0.9)),
            tensor_parallel_size=int(row.get("tensor_parallel_size", 1)),
            max_num_seqs=int(row.get("max_num_seqs", 16)),
            trust_remote_code=_as_bool(row.get("trust_remote_code"), True),
            enforce_eager=_as_bool(row.get("enforce_eager"), False),
            enable_prefix_caching=_as_bool(row.get("enable_prefix_caching"), True),
            quantization=str(row.get("quantization") or "").strip(),
            kv_cache_dtype=str(row.get("kv_cache_dtype", "auto")),
            cpu_offload_gb=float(row.get("cpu_offload_gb", 0.0)),
            api_key=str(row.get("api_key") or ""),
            max_loras=int(row.get("max_loras", max(1, len(lora_modules)))),
            max_cpu_loras=int(row.get("max_cpu_loras", max(1, len(lora_modules)))),
            max_lora_rank=int(row.get("max_lora_rank", 16)),
            lora_modules=lora_modules,
            environment={
                str(key).strip(): os.path.expandvars(os.path.expanduser(str(value)))
                for key, value in raw_environment.items()
                if str(key).strip()
            },
            extra_args=tuple(str(value) for value in (row.get("extra_args") or [])),
        )
        if not engine.model_path:
            raise ValueError(f"engine {name} 缺少 model_path")
        if not 0 < engine.gpu_memory_utilization <= 1:
            raise ValueError(f"engine {name} gpu_memory_utilization 必须在 (0, 1] 范围")
        if engine.lora_modules and engine.max_cpu_loras < engine.max_loras:
            raise ValueError(f"engine {name} max_cpu_loras 必须大于等于 max_loras")
        engines[engine.name] = engine

    if not engines:
        raise ValueError("配置文件至少需要一个 engine")

    utilization_by_devices: dict[tuple[str, ...], float] = {}
    for engine in engines.values():
        devices = tuple(
            sorted(
                device.strip()
                for device in engine.cuda_visible_devices.split(",")
                if device.strip()
            )
        )
        if not devices:
            continue
        utilization_by_devices[devices] = (
            utilization_by_devices.get(devices, 0.0) + engine.gpu_memory_utilization
        )
    for devices, utilization in utilization_by_devices.items():
        if utilization > 1.0 + 1e-9:
            raise ValueError(
                f"共享GPU {','.join(devices)} 的gpu_memory_utilization总和不能超过1，"
                f"当前为{utilization:.4f}"
            )

    models: dict[str, ModelRoute] = {}
    for name, row in (data.get("models") or {}).items():
        if not isinstance(row, dict):
            raise ValueError(f"model {name} 配置必须是对象")
        engine_name = str(row.get("engine") or "").strip()
        if engine_name not in engines:
            raise ValueError(f"model {name} 引用了不存在的 engine: {engine_name}")
        upstream_model = str(row.get("upstream_model") or name).strip()
        available_models = {
            engines[engine_name].served_model_name,
            *engines[engine_name].lora_modules.keys(),
        }
        if upstream_model not in available_models:
            raise ValueError(
                f"model {name} 的 upstream_model={upstream_model} 未在 engine {engine_name} 注册"
            )
        models[str(name)] = ModelRoute(
            name=str(name),
            engine=engine_name,
            upstream_model=upstream_model,
            instruction=str(row.get("instruction") or ""),
            max_tokens=int(row.get("max_tokens", 512)),
            temperature=float(row.get("temperature", 0.0)),
            top_p=float(row.get("top_p", 1.0)),
            chat_template_kwargs=dict(row.get("chat_template_kwargs") or {}),
            extra_body=dict(row.get("extra_body") or {}),
        )

    if not models:
        raise ValueError("配置文件至少需要一个 model 路由")

    return DeploymentConfig(
        gateway=gateway,
        engines=engines,
        models=models,
        profile_name=profile_path.stem,
        profile_path=profile_path,
    )


def build_engine_command(engine: EngineSpec) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        engine.model_path,
        "--host",
        engine.host,
        "--port",
        str(engine.port),
        "--served-model-name",
        engine.served_model_name,
        "--dtype",
        engine.dtype,
        "--max-model-len",
        str(engine.max_model_len),
        "--gpu-memory-utilization",
        str(engine.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(engine.tensor_parallel_size),
        "--max-num-seqs",
        str(engine.max_num_seqs),
        "--kv-cache-dtype",
        engine.kv_cache_dtype,
        "--cpu-offload-gb",
        str(engine.cpu_offload_gb),
    ]
    if engine.trust_remote_code:
        command.append("--trust-remote-code")
    if engine.enforce_eager:
        command.append("--enforce-eager")
    if engine.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if engine.quantization:
        command.extend(["--quantization", engine.quantization])
    if engine.api_key:
        command.extend(["--api-key", engine.api_key])
    if engine.lora_modules:
        command.extend(
            [
                "--enable-lora",
                "--max-loras",
                str(engine.max_loras),
                "--max-cpu-loras",
                str(engine.max_cpu_loras),
                "--max-lora-rank",
                str(engine.max_lora_rank),
                "--lora-modules",
                *[f"{name}={path}" for name, path in engine.lora_modules.items()],
            ]
        )
    command.extend(engine.extra_args)
    return command
