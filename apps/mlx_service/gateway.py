# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model: str = Field(..., description="模型名，对应 worker 中注册的 name")
    text: str = Field(..., description="原始描述")
    instruction: str | None = Field(default=None, description="覆盖默认 instruction")
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class WorkerSpec:
    name: str
    base_url: str
    models: list[str]
    timeout_seconds: int


@dataclass
class GatewayConfig:
    concurrency_mode: str


class GatewayRouter:
    def __init__(self, workers: dict[str, WorkerSpec], config: GatewayConfig):
        self.workers = workers
        self.config = config
        self.model_to_worker: dict[str, WorkerSpec] = {}
        self._serial_lock = threading.Lock()
        for worker in workers.values():
            for model in worker.models:
                if model in self.model_to_worker:
                    raise ValueError(f"模型重复注册到多个 worker: {model}")
                self.model_to_worker[model] = worker

    def _get_worker(self, model: str) -> WorkerSpec:
        worker = self.model_to_worker.get(model)
        if worker is None:
            raise KeyError(f"未注册模型: {model}")
        return worker

    def health(self) -> dict[str, Any]:
        workers = {}
        overall_ok = True
        for name, worker in self.workers.items():
            try:
                resp = requests.get(
                    f"{worker.base_url.rstrip('/')}/health",
                    timeout=min(worker.timeout_seconds, 10),
                )
                resp.raise_for_status()
                payload = resp.json()
                workers[name] = {
                    "ok": bool(payload.get("ok", True)),
                    "base_url": worker.base_url,
                    "models": worker.models,
                    "health": payload,
                }
            except Exception as exc:
                overall_ok = False
                workers[name] = {
                    "ok": False,
                    "base_url": worker.base_url,
                    "models": worker.models,
                    "error": str(exc),
                }
        return {
            "ok": overall_ok,
            "concurrency_mode": self.config.concurrency_mode,
            "workers": workers,
            "registered_models": sorted(self.model_to_worker.keys()),
        }

    def list_models(self) -> dict[str, Any]:
        workers = {}
        for name, worker in self.workers.items():
            try:
                resp = requests.get(
                    f"{worker.base_url.rstrip('/')}/models",
                    timeout=min(worker.timeout_seconds, 10),
                )
                resp.raise_for_status()
                payload = resp.json()
                workers[name] = {
                    "ok": True,
                    "base_url": worker.base_url,
                    "models": worker.models,
                    "payload": payload,
                }
            except Exception as exc:
                workers[name] = {
                    "ok": False,
                    "base_url": worker.base_url,
                    "models": worker.models,
                    "error": str(exc),
                }
        return {
            "concurrency_mode": self.config.concurrency_mode,
            "routing": {
                model: worker.name for model, worker in sorted(self.model_to_worker.items())
            },
            "workers": workers,
        }

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        worker = self._get_worker(request.model)
        def _forward() -> dict[str, Any]:
            resp = requests.post(
                f"{worker.base_url.rstrip('/')}/predict",
                json=request.model_dump(),
                timeout=worker.timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            payload["_worker"] = worker.name
            payload["_concurrency_mode"] = self.config.concurrency_mode
            return payload

        if self.config.concurrency_mode == "serial":
            with self._serial_lock:
                return _forward()
        return _forward()

    def unload_model(self, model: str) -> dict[str, Any]:
        worker = self._get_worker(model)
        resp = requests.post(
            f"{worker.base_url.rstrip('/')}/models/{model}/unload",
            timeout=min(worker.timeout_seconds, 30),
        )
        resp.raise_for_status()
        payload = resp.json()
        payload["_worker"] = worker.name
        return payload

    def unload_all(self) -> dict[str, Any]:
        results = {}
        total = 0
        for name, worker in self.workers.items():
            try:
                resp = requests.post(
                    f"{worker.base_url.rstrip('/')}/models/unload_all",
                    timeout=min(worker.timeout_seconds, 60),
                )
                resp.raise_for_status()
                payload = resp.json()
                results[name] = payload
                total += int(payload.get("unloaded", 0) or 0)
            except Exception as exc:
                results[name] = {"ok": False, "error": str(exc)}
        return {"ok": True, "unloaded": total, "workers": results}


def load_gateway_registry(path: Path) -> tuple[dict[str, WorkerSpec], GatewayConfig]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_workers = data.get("workers") or {}
    workers: dict[str, WorkerSpec] = {}
    for name, row in raw_workers.items():
        models = [str(model).strip() for model in (row.get("models") or []) if str(model).strip()]
        if not models:
            raise ValueError(f"worker {name} 未配置 models")
        base_url = str(row.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"worker {name} 未配置 base_url")
        workers[name] = WorkerSpec(
            name=str(name),
            base_url=base_url,
            models=models,
            timeout_seconds=int(row.get("timeout_seconds", 300)),
        )
    if not workers:
        raise ValueError("gateway registry 未配置任何 workers")
    raw_gateway = data.get("gateway") or {}
    concurrency_mode = str(raw_gateway.get("concurrency_mode", "serial")).strip().lower()
    if concurrency_mode not in {"serial", "parallel"}:
        raise ValueError(f"gateway.concurrency_mode 只支持 serial|parallel，当前为: {concurrency_mode}")
    return workers, GatewayConfig(concurrency_mode=concurrency_mode)


def build_app(router: GatewayRouter) -> FastAPI:
    app = FastAPI(title="MLX Service Gateway", version="0.1.0")

    @app.get("/health")
    def health():
        return router.health()

    @app.get("/models")
    def list_models():
        return router.list_models()

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            return router.predict(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            status_code = exc.response.status_code if exc.response is not None else 500
            raise HTTPException(status_code=status_code, detail=detail) from exc
        except Exception as exc:
            logger.exception("[MLX Gateway] 预测失败")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/models/{name}/unload")
    def unload_model(name: str):
        try:
            return router.unload_model(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            status_code = exc.response.status_code if exc.response is not None else 500
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @app.post("/models/unload_all")
    def unload_all():
        return router.unload_all()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX 多 worker 网关服务")
    parser.add_argument("--registry", type=Path, required=True, help="gateway registry YAML")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    workers, config = load_gateway_registry(args.registry)
    router = GatewayRouter(workers, config)
    app = build_app(router)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
