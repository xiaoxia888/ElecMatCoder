# -*- coding: utf-8 -*-
from __future__ import annotations

"""
python -m vllm_service.launch \
    --config vllm_service/service.yaml
"""
import argparse
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, EngineSpec, build_engine_command, load_config


logger = logging.getLogger(__name__)


def _gateway_module_name() -> str:
    return f"{__package__}.gateway" if __package__ else "apps.vllm_service.gateway"


def _engine_env(engine: EngineSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.update(engine.environment)
    if engine.cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = engine.cuda_visible_devices.strip()
    return env


def _format_command(engine: EngineSpec) -> str:
    environment = dict(engine.environment)
    if engine.cuda_visible_devices.strip():
        environment["CUDA_VISIBLE_DEVICES"] = engine.cuda_visible_devices.strip()
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(environment.items())
    )
    if prefix:
        prefix += " "
    return prefix + shlex.join(build_engine_command(engine))


def _is_engine_ready(engine: EngineSpec) -> bool:
    request = urllib.request.Request(
        f"{engine.base_url}/health",
        headers={"Authorization": f"Bearer {engine.api_key}"} if engine.api_key else {},
    )
    try:
        # Engine endpoints are always local. Ignore HTTP(S)_PROXY from cloud
        # environments or localhost health checks may never reach vLLM.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _wait_for_engines(
    processes: dict[str, subprocess.Popen[str]],
    engines: dict[str, EngineSpec],
    *,
    timeout_seconds: int,
    interval_seconds: float,
) -> None:
    pending = set(engines)
    deadline = time.time() + timeout_seconds
    while pending:
        for name in list(pending):
            process = processes[name]
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"vLLM engine {name} 启动失败，退出码: {code}")
            if _is_engine_ready(engines[name]):
                pending.remove(name)
                logger.info("[vLLM Launch] engine=%s 已就绪", name)
        if not pending:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"等待 vLLM engines 超时: {sorted(pending)}")
        logger.info("[vLLM Launch] 等待 engines: %s", ", ".join(sorted(pending)))
        time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM 多基座、多LoRA一键启动器")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--profile", default=None, help="临时覆盖service.yaml中的硬件Profile"
    )
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置并打印启动命令")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, profile=args.profile)

    if args.dry_run:
        output = {
            "profile": {
                "name": config.profile_name,
                "path": str(config.profile_path),
            },
            "engines": {name: _format_command(engine) for name, engine in config.engines.items()},
            "gateway": shlex.join(
                [
                    sys.executable,
                    "-m",
                    _gateway_module_name(),
                    "--config",
                    str(config_path),
                    "--profile",
                    str(config.profile_path),
                ]
            ),
            "models": {
                name: {"engine": route.engine, "upstream_model": route.upstream_model}
                for name, route in config.models.items()
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0

    processes: dict[str, subprocess.Popen[str]] = {}

    def terminate_all() -> None:
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 15
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    process.kill()

    def handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        logger.info("[vLLM Launch] 收到信号 %s，停止全部服务", signum)
        terminate_all()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for name, engine in config.engines.items():
            logger.info("[vLLM Launch] 启动 engine=%s: %s", name, _format_command(engine))
            processes[name] = subprocess.Popen(
                build_engine_command(engine),
                env=_engine_env(engine),
                text=True,
            )
            if not args.skip_health_check:
                # Sequential initialization avoids competing memory probes and
                # transient allocation spikes when engines share one GPU.
                _wait_for_engines(
                    {name: processes[name]},
                    {name: engine},
                    timeout_seconds=config.gateway.startup_timeout_seconds,
                    interval_seconds=config.gateway.health_check_interval_seconds,
                )

        gateway_command = [
            sys.executable,
            "-m",
            _gateway_module_name(),
            "--config",
            str(config_path),
            "--profile",
            str(config.profile_path),
            "--log-level",
            args.log_level,
        ]
        logger.info("[vLLM Launch] 启动统一网关 port=%s", config.gateway.port)
        processes["gateway"] = subprocess.Popen(gateway_command, text=True)

        while True:
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    logger.error("[vLLM Launch] 子进程退出 name=%s code=%s", name, code)
                    return code or 1
            time.sleep(1)
    finally:
        terminate_all()


if __name__ == "__main__":
    raise SystemExit(main())
