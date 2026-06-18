# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)

DEFAULT_TEXTS = [
    "90度弯头, 20 NB/T47008, FTE,CL 2000, SH/T3410, Galvanized , DN25",
    "TEE,RED SMLS BW A234 WPB ASME B16.9 High pressure service SCH160xSCH160 DN100x80",
    "SPECTACLE BLANK CL300(PN50) RF NB/T 47008 20 ENR STD 40T018 DN250",
    "45度法兰弯头, PTFElined GB/T 8163-20, RF, PN16, HG/T20538, SMLS , DN100, 4.0 mm",
]


@dataclass(frozen=True)
class MemorySample:
    timestamp: float
    rss_mb: float
    rss_percent: float
    pids: list[int]


def _read_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text_file:
        path = Path(args.text_file).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                texts.append(value)
    texts.extend([value.strip() for value in args.text if value.strip()])
    return texts or DEFAULT_TEXTS


def _physical_memory_mb() -> float:
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(output) / 1024 / 1024
    except Exception:
        return 0.0


def _parse_ps_rows(output: str) -> list[tuple[int, float, str]]:
    rows: list[tuple[int, float, str]] = []
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_mb = int(parts[1]) / 1024
        except ValueError:
            continue
        rows.append((pid, rss_mb, parts[2]))
    return rows


def discover_mlx_pids() -> list[int]:
    current_pid = str(subprocess.os.getpid())
    try:
        output = subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True)
    except Exception:
        return []

    pids: list[int] = []
    for pid, _rss_mb, command in _parse_ps_rows(output):
        if str(pid) == current_pid:
            continue
        if "apps.mlx_service" not in command:
            continue
        if "测试MLX部署模式性能.py" in command:
            continue
        pids.append(pid)
    return sorted(set(pids))


def _rss_for_pids(pids: list[int]) -> tuple[float, list[int]]:
    if not pids:
        return 0.0, []
    wanted = {str(pid) for pid in pids}
    try:
        output = subprocess.check_output(["ps", "-axo", "pid,rss,command"], text=True)
    except Exception:
        return 0.0, []

    total = 0.0
    alive: list[int] = []
    for pid, rss_mb, _command in _parse_ps_rows(output):
        if str(pid) in wanted:
            total += rss_mb
            alive.append(pid)
    return total, sorted(alive)


class MemorySampler:
    def __init__(self, pids: list[int], interval_seconds: float):
        self.pids = pids
        self.interval_seconds = interval_seconds
        self.total_memory_mb = _physical_memory_mb()
        self.samples: list[MemorySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._sample_once()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        rss_mb, alive_pids = _rss_for_pids(self.pids)
        rss_percent = (rss_mb / self.total_memory_mb * 100) if self.total_memory_mb else 0.0
        self.samples.append(
            MemorySample(
                timestamp=time.time(),
                rss_mb=round(rss_mb, 2),
                rss_percent=round(rss_percent, 3),
                pids=alive_pids,
            )
        )

    def summary(self) -> dict[str, Any]:
        rss_values = [sample.rss_mb for sample in self.samples]
        percent_values = [sample.rss_percent for sample in self.samples]
        if not rss_values:
            return {
                "pids": self.pids,
                "sample_count": 0,
                "note": "未采集到内存数据",
            }
        return {
            "pids": self.pids,
            "physical_memory_mb": round(self.total_memory_mb, 2),
            "sample_count": len(self.samples),
            "rss_mb": {
                "start": rss_values[0],
                "end": rss_values[-1],
                "min": min(rss_values),
                "max": max(rss_values),
                "mean": round(statistics.mean(rss_values), 2),
            },
            "rss_percent": {
                "start": percent_values[0],
                "end": percent_values[-1],
                "min": min(percent_values),
                "max": max(percent_values),
                "mean": round(statistics.mean(percent_values), 3),
            },
        }


def call_predict(
    *,
    service_url: str,
    model: str,
    text: str,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    resp = requests.post(
        f"{service_url.rstrip('/')}/predict",
        json={
            "model": model,
            "text": text,
            "instruction": instruction,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=timeout,
    )
    wall_seconds = time.perf_counter() - started
    resp.raise_for_status()
    payload = resp.json()
    return {
        "model": model,
        "wall_seconds": round(wall_seconds, 4),
        "service_elapsed_seconds": payload.get("elapsed_seconds"),
        "json_parse_ok": payload.get("json_parse_ok"),
        "worker": payload.get("_worker"),
        "concurrency_mode": payload.get("_concurrency_mode"),
        "loaded_models": payload.get("loaded_models"),
        "raw_preview": str(payload.get("raw_response", ""))[:160],
    }


def _make_jobs(models: list[str], texts: list[str], repeat: int) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for _ in range(repeat):
        for text in texts:
            for model in models:
                jobs.append((model, text))
    return jobs


def _summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(item["wall_seconds"]) for item in calls]
    service_values = [
        float(item["service_elapsed_seconds"])
        for item in calls
        if item.get("service_elapsed_seconds") is not None
    ]
    return {
        "count": len(calls),
        "wall_seconds": {
            "sum": round(sum(walls), 4),
            "mean": round(statistics.mean(walls), 4) if walls else 0,
            "median": round(statistics.median(walls), 4) if walls else 0,
            "min": round(min(walls), 4) if walls else 0,
            "max": round(max(walls), 4) if walls else 0,
        },
        "service_elapsed_seconds": {
            "sum": round(sum(service_values), 4),
            "mean": round(statistics.mean(service_values), 4) if service_values else 0,
        },
        "json_parse_ok_count": sum(1 for item in calls if item.get("json_parse_ok")),
    }


def run_sequential(args: argparse.Namespace, jobs: list[tuple[str, str]]) -> dict[str, Any]:
    started = time.perf_counter()
    calls = [
        call_predict(
            service_url=args.service_url,
            model=model,
            text=text,
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )
        for model, text in jobs
    ]
    total = time.perf_counter() - started
    return {
        "mode": "sequential",
        "total_wall_seconds": round(total, 4),
        "summary": _summarize_calls(calls),
        "calls": calls,
    }


def run_concurrent(args: argparse.Namespace, jobs: list[tuple[str, str]]) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                call_predict,
                service_url=args.service_url,
                model=model,
                text=text,
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )
            for model, text in jobs
        ]
        for future in as_completed(futures):
            calls.append(future.result())
    total = time.perf_counter() - started
    return {
        "mode": "concurrent",
        "concurrency": args.concurrency,
        "total_wall_seconds": round(total, 4),
        "summary": _summarize_calls(calls),
        "calls": calls,
    }


def get_service_info(service_url: str, timeout: int) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for path in ("health", "models"):
        try:
            resp = requests.get(f"{service_url.rstrip('/')}/{path}", timeout=min(timeout, 20))
            resp.raise_for_status()
            info[path] = resp.json()
        except Exception as exc:
            info[path] = {"ok": False, "error": str(exc)}
    return info


def maybe_unload_all(service_url: str, timeout: int) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{service_url.rstrip('/')}/models/unload_all",
            timeout=min(timeout, 60),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 MLX 单进程 / gateway 串行 / gateway 并行三类部署模式的耗时和内存"
    )
    parser.add_argument("--case-name", required=True, help="本次测试名称，例如 single-server / gateway-serial / gateway-parallel")
    parser.add_argument("--service-url", default="http://127.0.0.1:8200")
    parser.add_argument("--models", nargs="+", default=["type", "material-standard"])
    parser.add_argument("--text", action="append", default=[], help="可重复传入测试描述")
    parser.add_argument("--text-file", help="每行一条测试描述")
    parser.add_argument("--repeat", type=int, default=1, help="每条描述重复轮数")
    parser.add_argument("--concurrency", type=int, default=2, help="并发测试线程数")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--warmup", action="store_true", help="正式测试前先预热每个模型一次")
    parser.add_argument("--unload-before", action="store_true", help="测试前调用 /models/unload_all，测试冷启动加载成本")
    parser.add_argument("--pid", action="append", type=int, default=[], help="手动指定要统计内存的服务 PID，可重复")
    parser.add_argument("--memory-interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="保存 JSON 结果路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texts = _read_texts(args)
    jobs = _make_jobs(args.models, texts, args.repeat)
    pids = sorted(set(args.pid or discover_mlx_pids()))

    if args.unload_before:
        maybe_unload_all(args.service_url, args.timeout)

    if args.warmup:
        for model in args.models:
            call_predict(
                service_url=args.service_url,
                model=model,
                text=texts[0],
                instruction=args.instruction,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
            )

    before = get_service_info(args.service_url, args.timeout)
    sampler = MemorySampler(pids=pids, interval_seconds=args.memory_interval)
    sampler.start()
    started = time.perf_counter()
    try:
        sequential = run_sequential(args, jobs)
        concurrent = run_concurrent(args, jobs)
    finally:
        total_seconds = time.perf_counter() - started
        sampler.stop()
    after = get_service_info(args.service_url, args.timeout)

    seq_total = sequential["total_wall_seconds"]
    con_total = concurrent["total_wall_seconds"]
    result = {
        "case_name": args.case_name,
        "service_url": args.service_url,
        "models": args.models,
        "text_count": len(texts),
        "repeat": args.repeat,
        "job_count": len(jobs),
        "total_script_seconds": round(total_seconds, 4),
        "speedup_concurrent_vs_sequential": round(seq_total / con_total, 4) if con_total else None,
        "process_discovery": {
            "pids": pids,
            "note": "如果这里为空或不准，请用 --pid 手动指定 server/gateway/worker 的 PID",
        },
        "memory": sampler.summary(),
        "service_before": before,
        "service_after": after,
        "sequential": sequential,
        "concurrent": concurrent,
    }

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().write_text(output_text + "\n", encoding="utf-8")
        print(f"结果已保存: {args.output}")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
