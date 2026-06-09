# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "你是一个工业管道材料结构化信息提取助手。"
    "请从材料描述中提取结构化信息，并以 JSON 格式返回。"
)


def _build_raw_chatml_prompt(instruction: str, input_text: str) -> str:
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_json_output(raw: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


@dataclass
class ModelSpec:
    name: str
    model_path: str
    instruction: str
    max_tokens: int
    temperature: float
    top_p: float
    trust_remote_code: bool


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    loaded_at: float
    last_used_at: float
    hits: int = 0
    in_flight: int = 0


class PredictRequest(BaseModel):
    model: str = Field(..., description="模型名，对应注册表中的 name")
    text: str = Field(..., description="原始描述")
    instruction: Optional[str] = Field(default=None, description="覆盖默认 instruction")
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class MLXModelManager:
    def __init__(
        self,
        specs: dict[str, ModelSpec],
        *,
        max_loaded_models: int,
        idle_timeout_seconds: int,
    ):
        self.specs = specs
        self.max_loaded_models = max_loaded_models
        self.idle_timeout_seconds = idle_timeout_seconds
        self.loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._generation_locks: dict[str, asyncio.Lock] = {}

    def list_registered(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "model_path": spec.model_path,
                "max_tokens": spec.max_tokens,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
                "trust_remote_code": spec.trust_remote_code,
            }
            for spec in self.specs.values()
        ]

    def list_loaded(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.spec.name,
                "loaded_at": item.loaded_at,
                "last_used_at": item.last_used_at,
                "hits": item.hits,
                "in_flight": item.in_flight,
            }
            for item in sorted(self.loaded.values(), key=lambda x: x.last_used_at, reverse=True)
        ]

    def unload(self, name: str) -> bool:
        item = self.loaded.pop(name, None)
        if item is None:
            return False
        if item.in_flight > 0:
            self.loaded[name] = item
            return False
        del item.model
        del item.tokenizer
        self._generation_locks.pop(name, None)
        gc.collect()
        logger.info("[MLX Service] 已卸载模型: %s", name)
        return True

    def unload_all(self) -> int:
        names = list(self.loaded.keys())
        count = 0
        for name in names:
            if self.unload(name):
                count += 1
        return count

    def _evict_idle_models(self, now: float) -> None:
        if self.idle_timeout_seconds <= 0:
            return
        idle_names = [
            name
            for name, item in self.loaded.items()
            if item.in_flight == 0 and (now - item.last_used_at) >= self.idle_timeout_seconds
        ]
        for name in idle_names:
            self.unload(name)

    def _evict_lru_if_needed(self) -> None:
        while len(self.loaded) >= self.max_loaded_models and self.loaded:
            evictable = [item for item in self.loaded.values() if item.in_flight == 0]
            if not evictable:
                break
            oldest = min(evictable, key=lambda x: x.last_used_at)
            self.unload(oldest.spec.name)

    @staticmethod
    def _load_model_sync(spec: ModelSpec) -> LoadedModel:
        from mlx_lm.utils import load

        model, tokenizer = load(
            spec.model_path,
            tokenizer_config={"trust_remote_code": spec.trust_remote_code},
        )
        now = time.time()
        logger.info("[MLX Service] 已加载模型: %s", spec.name)
        return LoadedModel(
            spec=spec,
            model=model,
            tokenizer=tokenizer,
            loaded_at=now,
            last_used_at=now,
        )

    async def get_or_load(self, name: str) -> LoadedModel:
        async with self._lock:
            spec = self.specs.get(name)
            if spec is None:
                raise KeyError(f"未注册模型: {name}")

            now = time.time()
            self._evict_idle_models(now)

            if name in self.loaded:
                item = self.loaded[name]
                item.last_used_at = now
                item.hits += 1
                item.in_flight += 1
                return item

            self._evict_lru_if_needed()
            item = await asyncio.to_thread(self._load_model_sync, spec)
            item.hits += 1
            item.in_flight += 1
            self.loaded[name] = item
            self._generation_locks.setdefault(name, asyncio.Lock())
            return item

    async def release(self, name: str) -> None:
        async with self._lock:
            item = self.loaded.get(name)
            if item is None:
                return
            item.in_flight = max(0, item.in_flight - 1)
            item.last_used_at = time.time()

    def _get_generation_lock(self, name: str) -> asyncio.Lock:
        lock = self._generation_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._generation_locks[name] = lock
        return lock

    @staticmethod
    def _predict_sync(
        item: LoadedModel,
        *,
        text: str,
        instruction: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        from mlx_lm.generate import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        prompt = _build_raw_chatml_prompt(instruction, text)
        think_token_ids = {}
        for marker in ("<think>", "</think>"):
            token_ids = item.tokenizer.encode(marker, add_special_tokens=False)
            if len(token_ids) == 1:
                think_token_ids[token_ids[0]] = -1e9

        logits_processors = (
            make_logits_processors(logit_bias=think_token_ids)
            if think_token_ids
            else None
        )
        started = time.perf_counter()
        raw = generate(
            item.model,
            item.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=int(max_new_tokens),
            sampler=make_sampler(
                temp=float(temperature),
                top_p=float(top_p),
            ),
            logits_processors=logits_processors,
        ).strip()
        elapsed = time.perf_counter() - started
        parsed = _parse_json_output(raw)
        return {
            "model": item.spec.name,
            "elapsed_seconds": round(elapsed, 4),
            "instruction": instruction,
            "prompt": prompt,
            "raw_response": raw,
            "parsed_json": parsed,
            "json_parse_ok": parsed is not None,
        }

    async def predict(self, request: PredictRequest) -> dict[str, Any]:
        item = await self.get_or_load(request.model)
        spec = item.spec

        instruction = request.instruction or spec.instruction
        max_new_tokens = request.max_new_tokens or spec.max_tokens
        temperature = spec.temperature if request.temperature is None else request.temperature
        top_p = spec.top_p if request.top_p is None else request.top_p

        try:
            async with self._get_generation_lock(request.model):
                result = await asyncio.to_thread(
                    self._predict_sync,
                    item,
                    text=request.text,
                    instruction=instruction,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            result["loaded_models"] = len(self.loaded)
            return result
        finally:
            await self.release(request.model)


def load_registry(path: Path) -> dict[str, ModelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_models = data.get("models") or {}
    specs: dict[str, ModelSpec] = {}
    for name, row in raw_models.items():
        specs[name] = ModelSpec(
            name=name,
            model_path=str(Path(str(row["model_path"])).expanduser()),
            instruction=str(row.get("instruction", DEFAULT_INSTRUCTION)),
            max_tokens=int(row.get("max_tokens", 512)),
            temperature=float(row.get("temperature", 0.0)),
            top_p=float(row.get("top_p", 1.0)),
            trust_remote_code=bool(row.get("trust_remote_code", True)),
        )
    return specs


def build_app(manager: MLXModelManager) -> FastAPI:
    app = FastAPI(title="MLX Lazy Model Service", version="0.1.0")

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "registered_models": len(manager.specs),
            "loaded_models": len(manager.loaded),
        }

    @app.get("/models")
    async def list_models():
        return {
            "registered": manager.list_registered(),
            "loaded": manager.list_loaded(),
        }

    @app.post("/predict")
    async def predict(request: PredictRequest):
        try:
            return await manager.predict(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[MLX Service] 预测失败")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/models/{name}/unload")
    async def unload_model(name: str):
        async with manager._lock:
            ok = manager.unload(name)
        return {"ok": ok, "name": name}

    @app.post("/models/unload_all")
    async def unload_all():
        async with manager._lock:
            count = manager.unload_all()
        return {"ok": True, "unloaded": count}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX 多模型懒加载服务")
    parser.add_argument("--registry", type=Path, required=True, help="模型注册表 YAML")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--max-loaded-models", type=int, default=2, help="最多同时保留几个已加载模型")
    parser.add_argument("--idle-timeout-seconds", type=int, default=1800, help="模型空闲多久后自动释放")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    specs = load_registry(args.registry)
    manager = MLXModelManager(
        specs,
        max_loaded_models=args.max_loaded_models,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    app = build_app(manager)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
