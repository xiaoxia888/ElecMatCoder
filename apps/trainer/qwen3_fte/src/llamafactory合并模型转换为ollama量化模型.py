#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LlamaFactory merge 后的 HuggingFace 模型目录导入 Ollama。

支持两种模式：
1. gguf: HF -> GGUF(F16) -> llama-quantize -> ollama create
   适合 Linux / 云端，也适合本地 mac。若你希望本地与远程行为尽量一致，建议两边都用这个模式。
2. experimental: ollama create --experimental --quantize
   直接基于 HuggingFace 目录导入，适合作为本地便捷模式。

默认 mode=auto：
- 若检测到 llama.cpp，可优先走 gguf
- 否则回退到 experimental


python apps/trainer/qwen3_fte/src/llamafactory合并模型转换为ollama量化模型.py \
    --model-dir /Users/guoxi/Desktop/workspace/NJNCC/python_code/LlamaFactory/saves/qwen3-4b-base/lora/train_2027-04-11/contract_llm \
    --ollama-name contract-llm-all:q4 \
    --mode gguf \
    --quantize Q4_K_M \
    --llama-cpp-dir /path/to/llama.cpp \
    --execute

python apps/trainer/qwen3_fte/src/llamafactory合并模型转换为ollama量化模型.py \
    --model-dir /workspace/model/lora/qwen3-4b-merge/train_2027-04-11/contract_llm \
    --ollama-name contract-llm-all:q4 \
    --mode gguf \
    --quantize Q4_K_M \
    --llama-cpp-dir /workspace/llama.cpp \
    --execute


python apps/trainer/qwen3_fte/src/llamafactory合并模型转换为ollama量化模型.py \
    --model-dir /workspace/model/lora/qwen3-4b-merge/train_2027-04-11/contract_llm \
    --ollama-name contract-llm-fittings:q4\
    --mode gguf \
    --quantize Q4_K_M \
    --llama-cpp-dir /Users/guoxi/llama.cpp \
    --execute

  如果只是本地快速试：

  python apps/trainer/qwen3_fte/src/llamafactory合并模型转换为ollama量化模型.py \
    --model-dir /Users/guoxi/Desktop/workspace/NJNCC/python_code/LlamaFactory/saves/qwen3-4b-base/lora/pipe \
    --ollama-name contract-llm-pipe:q4 \
    --mode experimental \
    --quantize Q4_K_M \
    --execute


"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    PROJECT_ROOT = Path.cwd()

# 这里不是做“通用聊天助手”模板，而是尽量贴近 HF 直推时的单轮 chatml 入口。
#
# 关键点：
# 1. 兼容 Ollama /api/chat 传入的 .Messages
# 2. 显式保留 assistant 起始位
# 3. 使用 .Response 占位，避免 Ollama 在生成阶段自行补壳
DEFAULT_TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "system" }}<|im_start|>system
{{ .Content }}<|im_end|>
{{- else if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
{{- else if eq .Role "assistant" }}{{ .Content }}<|im_end|>
{{- end }}{{- end }}{{ .Response }}"""

# 默认不注入通用 SYSTEM。
# 平台与评测脚本都会通过 messages 显式传入任务专用 system prompt，
# 这里保空，避免把模型包装成“普通聊天助手”。
DEFAULT_SYSTEM = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 LlamaFactory merge 模型到 Ollama")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/qwen3/contract_llm"),
        help="merge 后的 HuggingFace 模型目录",
    )
    parser.add_argument(
        "--ollama-name",
        required=True,
        help="创建到 Ollama 的模型名，例如 contract-llm:q4",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "gguf", "experimental"],
        help="导入模式。若追求本地与远程一致，明确使用 gguf。",
    )
    parser.add_argument(
        "--quantize",
        default="Q4_K_M",
        help="量化等级，例如 Q4_K_M / Q5_K_M / Q8_0",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="",
        help="llama.cpp 目录。gguf 模式需要。",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="执行 convert_hf_to_gguf.py 的 Python",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="写入 Modelfile 的 num_ctx",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="写入 Modelfile 的默认 SYSTEM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="写入 Modelfile 的 temperature；结构化抽取建议 0.0",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="写入 Modelfile 的 top_p；结构化抽取建议 1.0",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="写入 Modelfile 的 num_predict",
    )
    parser.add_argument(
        "--modelfile-name",
        default="Modelfile",
        help="生成的 Modelfile 文件名",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="直接执行导入；不加时只生成文件与命令预览",
    )
    parser.add_argument(
        "--skip-ollama-create",
        action="store_true",
        help="gguf 模式下只导出/量化，不执行 ollama create",
    )
    return parser.parse_args()


def ensure_ollama_available() -> None:
    if shutil.which("ollama") is None:
        raise RuntimeError("未找到 ollama 命令，请先安装 Ollama。")


def validate_model_dir(model_dir: Path) -> dict:
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"不是目录: {model_dir}")

    required_files = ["config.json", "tokenizer.json"]
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"缺少必要文件: {', '.join(missing)}")

    safetensors = sorted(model_dir.glob("model*.safetensors"))
    if not safetensors:
        raise FileNotFoundError("未找到 safetensors 权重文件。")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "").strip()
    if model_type != "qwen3":
        raise ValueError(f"当前脚本按 qwen3 写 Modelfile，检测到 model_type={model_type!r}")

    return {
        "model_type": model_type,
        "num_safetensors": len(safetensors),
        "has_existing_modelfile": (model_dir / "Modelfile").exists(),
        "config": config,
    }


def _detect_llama_cpp_dir(explicit: str) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            Path("/workspace/llama.cpp"),
            PROJECT_ROOT / "llama.cpp",
            Path.home() / "llama.cpp",
            Path.home() / "workspace" / "llama.cpp",
        ]
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved.exists():
            return resolved
    return None


def _find_convert_script(llama_cpp_dir: Path) -> Path:
    path = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not path.exists():
        raise FileNotFoundError(f"在 {llama_cpp_dir} 下未找到 convert_hf_to_gguf.py")
    return path


def _find_quantize_binary(llama_cpp_dir: Path) -> Path:
    for rel in ("build/bin/llama-quantize", "build/bin/quantize"):
        path = llama_cpp_dir / rel
        if path.exists():
            return path
    raise FileNotFoundError(
        f"在 {llama_cpp_dir} 下未找到 llama-quantize，可先执行 cmake --build build --target llama-quantize -j"
    )


def _write_modelfile(
    *,
    modelfile_path: Path,
    from_value: str,
    num_ctx: int,
    system: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    include_template: bool = True,
    include_system: bool = True,
    stop_tokens: Optional[list[str]] = None,
) -> Path:
    lines = [f"FROM {from_value}", ""]
    if include_template:
        lines.extend([f'TEMPLATE """{DEFAULT_TEMPLATE}"""', ""])
    if include_system:
        lines.extend([f'SYSTEM """{system}"""', ""])
    for stop in (stop_tokens or []):
        lines.append(f'PARAMETER stop "{stop}"')
    lines.extend(
        [
            f"PARAMETER num_ctx {num_ctx}",
            f"PARAMETER temperature {temperature}",
            f"PARAMETER top_p {top_p}",
            f"PARAMETER num_predict {num_predict}",
        ]
    )
    content = "\n".join(lines) + "\n"
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def shell_join(parts: list[str]) -> str:
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


def resolve_mode(args: argparse.Namespace, llama_cpp_dir: Optional[Path]) -> str:
    if args.mode != "auto":
        return args.mode
    return "gguf" if llama_cpp_dir is not None else "experimental"


def build_experimental_command(modelfile_path: Path, ollama_name: str, quantize: str) -> list[str]:
    return [
        "ollama",
        "create",
        ollama_name,
        "--experimental",
        "--quantize",
        quantize,
        "--file",
        str(modelfile_path),
    ]


def build_gguf_paths(model_dir: Path, quantize: str) -> tuple[Path, Path, Path]:
    gguf_dir = model_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    fp16_gguf = gguf_dir / "model-f16.gguf"
    quant_suffix = quantize.lower().replace("_", "")
    quant_gguf = gguf_dir / f"model-{quant_suffix}.gguf"
    modelfile = model_dir / "Modelfile"
    return fp16_gguf, quant_gguf, modelfile


def main() -> int:
    args = parse_args()
    ensure_ollama_available()

    info = validate_model_dir(args.model_dir)
    llama_cpp_dir = _detect_llama_cpp_dir(args.llama_cpp_dir)
    mode = resolve_mode(args, llama_cpp_dir)

    print(f"模型目录: {args.model_dir}")
    print(f"model_type: {info['model_type']}")
    print(f"safetensors 分片数: {info['num_safetensors']}")
    print(f"已有 Modelfile: {info['has_existing_modelfile']}")
    print(f"导入模式: {mode}")

    if mode == "experimental":
        modelfile_path = _write_modelfile(
            modelfile_path=args.model_dir / args.modelfile_name,
            from_value=".",
            num_ctx=args.num_ctx,
            system=args.system,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            include_template=True,
            include_system=True,
            stop_tokens=["<|im_end|>", "<|im_start|>"],
        )
        command = build_experimental_command(modelfile_path, args.ollama_name, args.quantize)
        print(f"Modelfile 已写入: {modelfile_path}")
        print("\n建议执行命令:")
        print(shell_join(command))
        if not args.execute:
            print("\n未执行导入。若确认无误，追加 --execute 直接导入。")
            return 0

        print("\n开始执行 ollama create ...")
        proc = subprocess.run(command, cwd=str(args.model_dir), check=False)
        if proc.returncode != 0:
            print(f"\nollama create 失败，退出码: {proc.returncode}", file=sys.stderr)
            return proc.returncode

        print(f"\nOllama 模型创建完成: {args.ollama_name}")
        return 0

    if llama_cpp_dir is None:
        raise FileNotFoundError("gguf 模式需要 llama.cpp，但未检测到。请通过 --llama-cpp-dir 显式指定。")

    convert_script = _find_convert_script(llama_cpp_dir)
    quantize_bin = _find_quantize_binary(llama_cpp_dir)
    fp16_gguf, quant_gguf, modelfile_path = build_gguf_paths(args.model_dir, args.quantize)
    _write_modelfile(
        modelfile_path=modelfile_path,
        from_value=str(quant_gguf),
        num_ctx=args.num_ctx,
        system=args.system,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.num_predict,
        include_template=True,
        include_system=False,
        stop_tokens=["<|im_end|>"],
    )

    convert_cmd = [
        args.python_bin,
        str(convert_script),
        str(args.model_dir),
        "--outfile",
        str(fp16_gguf),
        "--outtype",
        "f16",
    ]
    quantize_cmd = [
        str(quantize_bin),
        str(fp16_gguf),
        str(quant_gguf),
        args.quantize,
    ]
    ollama_cmd = [
        "ollama",
        "create",
        args.ollama_name,
        "-f",
        str(modelfile_path),
    ]

    print(f"llama.cpp: {llama_cpp_dir}")
    print(f"GGUF 输出目录: {fp16_gguf.parent}")
    print(f"Modelfile 已写入: {modelfile_path}")
    print("\n将执行以下步骤：")
    print("1. HF -> GGUF(F16)")
    print("2. GGUF(F16) -> 量化 GGUF")
    if not args.skip_ollama_create:
        print("3. ollama create 注册模型")

    print("\n命令预览：")
    print(shell_join(convert_cmd))
    print(shell_join(quantize_cmd))
    if not args.skip_ollama_create:
        print(shell_join(ollama_cmd))

    if not args.execute:
        print("\n未执行。加 --execute 后开始实际导出。")
        return 0

    _run(convert_cmd)
    _run(quantize_cmd)
    if not args.skip_ollama_create:
        _run(ollama_cmd)

    print("\n完成!")
    print(f"量化模型: {quant_gguf}")
    if not args.skip_ollama_create:
        print(f"Ollama 模型: {args.ollama_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
