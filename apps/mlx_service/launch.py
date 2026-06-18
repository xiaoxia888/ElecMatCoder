# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
DEFAULT_SERVICE_CONFIG = Path(__file__).resolve().with_name("service.yaml")


@dataclass
class WorkerLaunchSpec:
    name: str
    host: str
    port: int
    models: list[str]


@dataclass
class GatewayLaunchSpec:
    host: str
    port: int


@dataclass
class SingleLaunchSpec:
    host: str
    port: int
    models: list[str]


@dataclass
class LaunchConfig:
    mode: str
    single: SingleLaunchSpec
    workers: list[WorkerLaunchSpec]
    gateway: GatewayLaunchSpec


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 对象: {path}")
    return data


def _iter_worker_rows(raw_workers: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(raw_workers, dict):
        return [(str(name), row or {}) for name, row in raw_workers.items()]
    if isinstance(raw_workers, list):
        rows: list[tuple[str, dict[str, Any]]] = []
        for row in raw_workers:
            if isinstance(row, dict):
                rows.append((str(row.get("name", "") or ""), row))
        return rows
    return []


def load_launch_config(path: Path) -> LaunchConfig:
    data = _load_yaml(path)
    mode = str((data.get("deployment") or {}).get("mode", "single") or "single").strip().lower()
    if mode not in {"single", "gateway_serial", "gateway_parallel"}:
        raise ValueError(f"deployment.mode 只支持 single|gateway_serial|gateway_parallel，当前为: {mode}")

    raw_single = data.get("single") or {}
    single_models = [
        str(model).strip()
        for model in (raw_single.get("models") or list((data.get("models") or {}).keys()))
        if str(model).strip()
    ]
    single = SingleLaunchSpec(
        host=str(raw_single.get("host", "0.0.0.0")),
        port=int(raw_single.get("port", 8200)),
        models=single_models,
    )

    raw_workers = data.get("workers") or []
    workers: list[WorkerLaunchSpec] = []
    for name, row in _iter_worker_rows(raw_workers):
        if not name:
            raise ValueError("worker 未配置 name")
        models = [str(model).strip() for model in (row.get("models") or []) if str(model).strip()]
        if not models:
            raise ValueError(f"worker {name} 未配置 models")
        workers.append(
            WorkerLaunchSpec(
                name=name,
                host=str(row.get("host", "0.0.0.0")),
                port=int(row["port"]),
                models=models,
            )
        )

    raw_gateway = data.get("gateway") or {}
    if not raw_gateway:
        raise ValueError("service.yaml 必须配置 gateway")
    gateway = GatewayLaunchSpec(
        host=str(raw_gateway.get("host", "0.0.0.0")),
        port=int(raw_gateway.get("port", 8200)),
    )
    return LaunchConfig(mode=mode, single=single, workers=workers, gateway=gateway)


def _spawn_server(
    *,
    name: str,
    config: Path,
    host: str,
    port: int,
    models: list[str],
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "apps.mlx_service.server",
        "--config",
        str(config),
        "--host",
        host,
        "--port",
        str(port),
        "--models",
        *models,
    ]
    logger.info("[MLX Launch] 启动 %s port=%s models=%s", name, port, ",".join(models))
    return subprocess.Popen(cmd)


def _spawn_gateway(gateway: GatewayLaunchSpec, config: Path) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "apps.mlx_service.gateway",
        "--config",
        str(config),
        "--host",
        gateway.host,
        "--port",
        str(gateway.port),
    ]
    logger.info("[MLX Launch] 启动 gateway port=%s", gateway.port)
    return subprocess.Popen(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX service 一键启动器")
    parser.add_argument("--config", type=Path, default=DEFAULT_SERVICE_CONFIG, help="统一服务配置 YAML")
    parser.add_argument("--registry", type=Path, default=None, help="兼容旧参数：等同于 --config")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = (args.registry or args.config).expanduser()
    launch_config = load_launch_config(config_path)
    processes: list[subprocess.Popen[str]] = []

    def terminate_all() -> None:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 8
        for proc in reversed(processes):
            if proc.poll() is None:
                remaining = max(0.0, deadline - time.time())
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        logger.info("[MLX Launch] 收到信号 %s，开始停止全部子进程", signum)
        terminate_all()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if launch_config.mode == "single":
            processes.append(
                _spawn_server(
                    name="single_server",
                    config=config_path,
                    host=launch_config.single.host,
                    port=launch_config.single.port,
                    models=launch_config.single.models,
                )
            )
        else:
            for worker in launch_config.workers:
                processes.append(
                    _spawn_server(
                        name=f"worker={worker.name}",
                        config=config_path,
                        host=worker.host,
                        port=worker.port,
                        models=worker.models,
                    )
                )
            time.sleep(1.0)
            processes.append(_spawn_gateway(launch_config.gateway, config_path))

        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    logger.error("[MLX Launch] 子进程异常退出 pid=%s code=%s", proc.pid, code)
                    terminate_all()
                    return code or 1
            time.sleep(1.0)
    finally:
        terminate_all()


if __name__ == "__main__":
    raise SystemExit(main())
