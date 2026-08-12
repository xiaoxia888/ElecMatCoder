# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG_PATH, DeploymentConfig, EngineSpec, load_config


logger = logging.getLogger(__name__)


def parse_json_output(raw: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


class PredictRequest(BaseModel):
    model: str = Field(..., description="对外模型名")
    text: str = Field(..., description="原始描述")
    instruction: Optional[str] = Field(default=None, description="临时覆盖配置中的提示词")
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    include_logprobs: bool = False


class VLLMGateway:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self.client = httpx.AsyncClient(timeout=self.config.gateway.request_timeout_seconds)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    @staticmethod
    def _headers(engine: EngineSpec) -> dict[str, str]:
        if not engine.api_key:
            return {}
        return {"Authorization": f"Bearer {engine.api_key}"}

    async def _engine_get(self, engine: EngineSpec, path: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("HTTP client 尚未启动")
        started = time.perf_counter()
        try:
            response = await self.client.get(
                f"{engine.base_url}{path}",
                headers=self._headers(engine),
                timeout=min(20, self.config.gateway.request_timeout_seconds),
            )
            response.raise_for_status()
            try:
                payload: Any = response.json()
            except json.JSONDecodeError:
                payload = response.text
            return {
                "ok": True,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "payload": payload,
            }
        except Exception as exc:
            return {
                "ok": False,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "error": str(exc),
            }

    async def health(self) -> dict[str, Any]:
        names = list(self.config.engines)
        checks = await asyncio.gather(
            *[self._engine_get(self.config.engines[name], "/health") for name in names]
        )
        engines = dict(zip(names, checks))
        return {
            "ok": all(item.get("ok") for item in checks),
            "backend": "vllm",
            "profile": self.config.profile_name,
            "registered_models": len(self.config.models),
            "engines": engines,
        }

    async def models(self) -> dict[str, Any]:
        names = list(self.config.engines)
        checks = await asyncio.gather(
            *[self._engine_get(self.config.engines[name], "/v1/models") for name in names]
        )
        return {
            "registered": [
                {
                    "name": route.name,
                    "engine": route.engine,
                    "upstream_model": route.upstream_model,
                    "prompt_file": route.prompt_file,
                    "max_tokens": route.max_tokens,
                }
                for route in self.config.models.values()
            ],
            "engines": dict(zip(names, checks)),
        }

    async def predict(self, request: PredictRequest) -> dict[str, Any]:
        route = self.config.models.get(request.model)
        if route is None:
            raise KeyError(f"未注册模型: {request.model}")
        engine = self.config.engines[route.engine]
        if self.client is None:
            raise RuntimeError("HTTP client 尚未启动")

        instruction = route.instruction if request.instruction is None else request.instruction
        requested_max_tokens = (
            route.max_tokens if request.max_new_tokens is None else request.max_new_tokens
        )
        # Route limits are hard safety limits. A stale client must not be able to
        # consume the model's context window by requesting a larger completion.
        max_tokens = min(route.max_tokens, requested_max_tokens)
        temperature = route.temperature if request.temperature is None else request.temperature
        top_p = route.top_p if request.top_p is None else request.top_p
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": request.text})

        body: dict[str, Any] = {
            "model": route.upstream_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if request.include_logprobs:
            body["logprobs"] = True
        if route.chat_template_kwargs:
            body["chat_template_kwargs"] = route.chat_template_kwargs
        body.update(route.extra_body)

        started = time.perf_counter()
        response = await self.client.post(
            f"{engine.base_url}/v1/chat/completions",
            json=body,
            headers=self._headers(engine),
        )
        elapsed = time.perf_counter() - started
        if response.is_error:
            raise RuntimeError(
                f"vLLM engine={engine.name} status={response.status_code}: {response.text[:1000]}"
            )
        payload = response.json()
        choices = payload.get("choices") or []
        raw = ""
        if choices:
            message = choices[0].get("message") or {}
            raw = str(message.get("content") or "").strip()
        token_logprobs: list[dict[str, Any]] = []
        if request.include_logprobs and choices:
            cursor = 0
            content_items = ((choices[0].get("logprobs") or {}).get("content") or [])
            for item in content_items:
                if not isinstance(item, dict) or item.get("logprob") is None:
                    continue
                token = str(item.get("token") or "")
                raw_bytes = item.get("bytes")
                if isinstance(raw_bytes, list):
                    try:
                        token = bytes(int(value) for value in raw_bytes).decode("utf-8")
                    except (TypeError, ValueError, UnicodeDecodeError):
                        pass
                token_logprobs.append(
                    {
                        "token": token,
                        "logprob": float(item["logprob"]),
                        "start": cursor,
                        "end": cursor + len(token),
                    }
                )
                cursor += len(token)
            if token_logprobs:
                unstripped = "".join(str(item["token"]) for item in token_logprobs)
                left = len(unstripped) - len(unstripped.lstrip())
                adjusted = []
                for record in token_logprobs:
                    start = max(0, int(record["start"]) - left)
                    end = min(len(raw), int(record["end"]) - left)
                    if end > start:
                        adjusted.append({**record, "token": raw[start:end], "start": start, "end": end})
                token_logprobs = adjusted
        parsed = parse_json_output(raw)
        return {
            "model": request.model,
            "engine": engine.name,
            "upstream_model": route.upstream_model,
            "elapsed_seconds": round(elapsed, 4),
            "instruction": instruction,
            "prompt": messages,
            "raw_response": raw,
            "parsed_json": parsed,
            "json_parse_ok": parsed is not None,
            "usage": payload.get("usage") or {},
            "token_logprobs": token_logprobs,
        }


def build_app(config: DeploymentConfig) -> FastAPI:
    gateway = VLLMGateway(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await gateway.start()
        try:
            yield
        finally:
            await gateway.close()

    app = FastAPI(title="vLLM Multi-LoRA Gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        result = await gateway.health()
        if not result["ok"]:
            return result
        return result

    @app.get("/models")
    async def models():
        return await gateway.models()

    @app.post("/predict")
    async def predict(request: PredictRequest):
        try:
            return await gateway.predict(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[vLLM Gateway] 预测失败")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM 多基座、多LoRA统一网关")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--profile", default=None, help="临时覆盖service.yaml中的硬件Profile"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config(args.config, profile=args.profile)
    app = build_app(config)
    uvicorn.run(
        app,
        host=args.host or config.gateway.host,
        port=args.port or config.gateway.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
