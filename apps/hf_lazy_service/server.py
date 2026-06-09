# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
# max-loaded-models 常驻进程

python -m apps.hf_lazy_service.server \
    --registry apps/hf_lazy_service/models.mac.yaml \
    --host 0.0.0.0 \
    --port 8100 \
    --device auto \
    --max-loaded-models 2 \
    --idle-timeout-seconds 1800
"""

logger = logging.getLogger(__name__)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str, dtype_name: str):
    if dtype_name != "auto":
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"不支持的 dtype: {dtype_name}") from exc
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _clear_device_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _prepare_compatible_tokenizer_path(model_path: Path) -> Path:
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return model_path

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except Exception:
        return model_path

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return model_path

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_lazy_tokenizer_"))
    for filename in _TOKENIZER_FILES:
        src = model_path / filename
        if src.exists():
            shutil.copy2(src, temp_dir / filename)

    patched_config = dict(tokenizer_config)
    patched_config.pop("extra_special_tokens", None)
    (temp_dir / "tokenizer_config.json").write_text(
        json.dumps(patched_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "[HF Lazy Service] 检测到不兼容的 extra_special_tokens(list)，已生成兼容 tokenizer 目录: %s",
        temp_dir,
    )
    return temp_dir


@dataclass
class ModelSpec:
    name: str
    model_path: Path
    device: str
    dtype_name: str
    instruction: str
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class LoadedModel:
    spec: ModelSpec
    tokenizer: Any
    model: Any
    tokenizer_path: Path
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


class LazyModelManager:
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
        self._lock = threading.RLock()

    def list_registered(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "model_path": str(spec.model_path),
                "device": spec.device,
                "dtype": spec.dtype_name,
                "max_new_tokens": spec.max_new_tokens,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
            }
            for spec in self.specs.values()
        ]

    def list_loaded(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": item.spec.name,
                    "device": item.spec.device,
                    "loaded_at": item.loaded_at,
                    "last_used_at": item.last_used_at,
                    "hits": item.hits,
                    "in_flight": item.in_flight,
                }
                for item in sorted(self.loaded.values(), key=lambda x: x.last_used_at, reverse=True)
            ]

    def unload(self, name: str) -> bool:
        with self._lock:
            item = self.loaded.pop(name, None)
            if item is None:
                return False
            if item.in_flight > 0:
                self.loaded[name] = item
                return False
            del item.model
            del item.tokenizer
            if item.tokenizer_path != item.spec.model_path:
                shutil.rmtree(item.tokenizer_path, ignore_errors=True)
            gc.collect()
            _clear_device_cache(item.spec.device)
            logger.info("[HF Lazy Service] 已卸载模型: %s", name)
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

    def _load_model(self, spec: ModelSpec) -> LoadedModel:
        tokenizer_path = _prepare_compatible_tokenizer_path(spec.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = _resolve_device(spec.device)
        dtype = _resolve_dtype(device, spec.dtype_name)

        model = AutoModelForCausalLM.from_pretrained(
            str(spec.model_path),
            dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        now = time.time()
        logger.info(
            "[HF Lazy Service] 已加载模型: %s, device=%s, dtype=%s",
            spec.name,
            device,
            dtype,
        )
        return LoadedModel(
            spec=ModelSpec(
                name=spec.name,
                model_path=spec.model_path,
                device=device,
                dtype_name=spec.dtype_name,
                instruction=spec.instruction,
                max_new_tokens=spec.max_new_tokens,
                temperature=spec.temperature,
                top_p=spec.top_p,
            ),
            tokenizer=tokenizer,
            model=model,
            tokenizer_path=tokenizer_path,
            loaded_at=now,
            last_used_at=now,
            hits=0,
        )

    def get_or_load(self, name: str) -> LoadedModel:
        with self._lock:
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
            item = self._load_model(spec)
            item.hits += 1
            item.in_flight += 1
            self.loaded[name] = item
            return item

    def release(self, name: str) -> None:
        with self._lock:
            item = self.loaded.get(name)
            if item is None:
                return
            item.in_flight = max(0, item.in_flight - 1)
            item.last_used_at = time.time()

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        item = self.get_or_load(request.model)
        spec = item.spec

        instruction = request.instruction or spec.instruction
        max_new_tokens = request.max_new_tokens or spec.max_new_tokens
        temperature = spec.temperature if request.temperature is None else request.temperature
        top_p = spec.top_p if request.top_p is None else request.top_p

        prompt = _build_raw_chatml_prompt(instruction, request.text)
        inputs = item.tokenizer(prompt, return_tensors="pt").to(item.model.device)

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0,
            "pad_token_id": item.tokenizer.eos_token_id,
        }
        if float(temperature) > 0:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)

        try:
            started = time.perf_counter()
            bad_words_ids = []
            for marker in ("<think>", "</think>"):
                token_ids = item.tokenizer.encode(marker, add_special_tokens=False)
                if token_ids:
                    bad_words_ids.append(token_ids)
            if bad_words_ids:
                generate_kwargs["bad_words_ids"] = bad_words_ids

            with torch.no_grad():
                outputs = item.model.generate(**inputs, **generate_kwargs)
            elapsed = time.perf_counter() - started

            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            raw = item.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            parsed = _parse_json_output(raw)

            return {
                "model": request.model,
                "device": item.spec.device,
                "elapsed_seconds": round(elapsed, 4),
                "instruction": instruction,
                "prompt": prompt,
                "raw_response": raw,
                "parsed_json": parsed,
                "json_parse_ok": parsed is not None,
                "loaded_models": len(self.loaded),
            }
        finally:
            self.release(request.model)


def load_registry(path: Path, default_device: str, default_dtype: str) -> dict[str, ModelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_models = data.get("models") or {}
    specs: dict[str, ModelSpec] = {}
    for name, row in raw_models.items():
        model_path = Path(str(row["model_path"])).expanduser()
        specs[name] = ModelSpec(
            name=name,
            model_path=model_path,
            device=str(row.get("device", default_device)),
            dtype_name=str(row.get("dtype", default_dtype)),
            instruction=str(row.get("instruction", DEFAULT_INSTRUCTION)),
            max_new_tokens=int(row.get("max_new_tokens", 512)),
            temperature=float(row.get("temperature", 0.0)),
            top_p=float(row.get("top_p", 1.0)),
        )
    return specs


def build_app(manager: LazyModelManager) -> FastAPI:
    app = FastAPI(title="HF Lazy Model Service", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "registered_models": len(manager.specs),
            "loaded_models": len(manager.loaded),
        }

    @app.get("/models")
    def list_models():
        return {
            "registered": manager.list_registered(),
            "loaded": manager.list_loaded(),
        }

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            return manager.predict(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("[HF Lazy Service] 预测失败")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/models/{name}/unload")
    def unload_model(name: str):
        ok = manager.unload(name)
        return {"ok": ok, "name": name}

    @app.post("/models/unload_all")
    def unload_all():
        count = manager.unload_all()
        return {"ok": True, "unloaded": count}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF 多模型懒加载服务")
    parser.add_argument("--registry", type=Path, required=True, help="模型注册表 YAML")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="auto", help="默认设备: auto/cpu/mps/cuda")
    parser.add_argument("--dtype", default="auto", help="默认 dtype: auto/float16/bfloat16/float32")
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
    specs = load_registry(args.registry, args.device, args.dtype)
    manager = LazyModelManager(
        specs,
        max_loaded_models=args.max_loaded_models,
        idle_timeout_seconds=args.idle_timeout_seconds,
    )
    app = build_app(manager)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
