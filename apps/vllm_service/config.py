# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("service.yaml")


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


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 对象: {path}")
    return data


def load_config(path: Path) -> DeploymentConfig:
    path = path.expanduser().resolve()
    data = _load_yaml(path)

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

    return DeploymentConfig(gateway=gateway, engines=engines, models=models)


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
