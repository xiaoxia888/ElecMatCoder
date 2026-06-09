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


@dataclass
class WorkerLaunchSpec:
    name: str
    registry: Path
    host: str
    port: int
    max_loaded_models: int
    idle_timeout_seconds: int


@dataclass
class GatewayLaunchSpec:
    registry: Path
    host: str
    port: int


def load_cluster_registry(path: Path) -> tuple[list[WorkerLaunchSpec], GatewayLaunchSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_workers = data.get("workers") or []
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ValueError("cluster registry 必须配置 workers 列表")

    workers: list[WorkerLaunchSpec] = []
    for row in raw_workers:
        workers.append(
            WorkerLaunchSpec(
                name=str(row["name"]),
                registry=Path(str(row["registry"])).expanduser(),
                host=str(row.get("host", "0.0.0.0")),
                port=int(row["port"]),
                max_loaded_models=int(row.get("max_loaded_models", 2)),
                idle_timeout_seconds=int(row.get("idle_timeout_seconds", 1800)),
            )
        )

    raw_gateway = data.get("gateway") or {}
    if not raw_gateway:
        raise ValueError("cluster registry 必须配置 gateway")
    gateway = GatewayLaunchSpec(
        registry=Path(str(raw_gateway["registry"])).expanduser(),
        host=str(raw_gateway.get("host", "0.0.0.0")),
        port=int(raw_gateway.get("port", 8200)),
    )
    return workers, gateway


def _spawn_worker(worker: WorkerLaunchSpec) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "apps.mlx_service.server",
        "--registry",
        str(worker.registry),
        "--host",
        worker.host,
        "--port",
        str(worker.port),
        "--max-loaded-models",
        str(worker.max_loaded_models),
        "--idle-timeout-seconds",
        str(worker.idle_timeout_seconds),
    ]
    logger.info("[MLX Launch] 启动 worker=%s port=%s", worker.name, worker.port)
    return subprocess.Popen(cmd)


def _spawn_gateway(gateway: GatewayLaunchSpec) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "apps.mlx_service.gateway",
        "--registry",
        str(gateway.registry),
        "--host",
        gateway.host,
        "--port",
        str(gateway.port),
    ]
    logger.info("[MLX Launch] 启动 gateway port=%s", gateway.port)
    return subprocess.Popen(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX service 一键启动器")
    parser.add_argument("--registry", type=Path, required=True, help="cluster registry YAML")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    workers, gateway = load_cluster_registry(args.registry)
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
        for worker in workers:
            processes.append(_spawn_worker(worker))
        time.sleep(1.0)
        processes.append(_spawn_gateway(gateway))

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
