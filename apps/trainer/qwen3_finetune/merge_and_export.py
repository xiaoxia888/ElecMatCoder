# -*- coding: utf-8 -*-
"""
LoRA 权重合并 & GGUF 导出

训练完成后，将 LoRA adapter 合并回基座模型，并可选导出 GGUF 用于 Ollama 部署。

使用方法:
    # 合并并创建 Modelfile（默认行为）
    python -m apps.trainer.qwen3_finetune.merge_and_export \
        --adapter_path outputs/qwen3_finetune/run_xxx/final \
        --output_dir outputs/qwen3_finetune/merged
"""

import sys
import json
import argparse
import logging
import tempfile
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.llm_ner.predictor import SYSTEM_PROMPT


def _prepare_tokenizer_source(adapter_path: Path) -> Path:
    """
    兼容旧版 tokenizer_config.json：
    某些训练产物中的 extra_special_tokens 被保存成 list，
    但新版本 transformers 在 Qwen tokenizer 初始化时要求 dict。
    这里在临时目录中做一次轻量修复，避免直接改原产物。
    """
    tokenizer_cfg = adapter_path / "tokenizer_config.json"
    if not tokenizer_cfg.exists():
        return adapter_path

    try:
        data = json.loads(tokenizer_cfg.read_text(encoding="utf-8"))
    except Exception:
        return adapter_path

    extra_special_tokens = data.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return adapter_path

    temp_dir = Path(tempfile.mkdtemp(prefix="qwen3_tokenizer_fix_"))
    for item in adapter_path.iterdir():
        target = temp_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    fixed_cfg = temp_dir / "tokenizer_config.json"
    fixed_data = json.loads(fixed_cfg.read_text(encoding="utf-8"))
    fixed_data["extra_special_tokens"] = {}
    fixed_cfg.write_text(json.dumps(fixed_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.warning(
        "检测到 tokenizer_config.json 中 extra_special_tokens 为 list，"
        "已在临时目录中修正为 dict 后加载 tokenizer。"
    )
    return temp_dir


def merge_lora(adapter_path: Path, output_dir: Path):
    """合并 LoRA 权重到基座模型"""
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B")
    else:
        base_model_name = "Qwen/Qwen3-4B"

    logger.info(f"基座模型: {base_model_name}")
    logger.info(f"LoRA adapter: {adapter_path}")

    logger.info("加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("加载 Tokenizer...")
    tokenizer_source = _prepare_tokenizer_source(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), trust_remote_code=True,
    )

    logger.info("合并 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"保存合并后的模型: {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("合并完成!")
    return output_dir


def create_modelfile(output_dir: Path, gguf_filename: str = "model-q4km.gguf"):
    """创建 Ollama Modelfile（不内嵌 SYSTEM，由推理代码动态发送）"""
    gguf_path = output_dir / "gguf" / gguf_filename

    content = f'''FROM {gguf_path}

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER num_predict 256
'''

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Modelfile 已创建: {modelfile_path}")
    return modelfile_path


def print_gguf_guide(merged_dir: Path, quantize: str = "Q4_K_M"):
    """打印手动 GGUF 导出指南"""
    gguf_dir = merged_dir / "gguf"
    print("\n" + "=" * 60)
    print("GGUF 导出指南 (手动执行)")
    print("=" * 60)
    print(f"""
# 1. 安装 llama.cpp (如未安装)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
make -j

# 2. HF → GGUF FP16
python convert_hf_to_gguf.py {merged_dir} \\
    --outfile {gguf_dir}/model-fp16.gguf --outtype f16

# 3. 量化
./build/bin/llama-quantize \\
    {gguf_dir}/model-fp16.gguf \\
    {gguf_dir}/model-q4km.gguf {quantize}

# 4. Ollama 部署
ollama create qwen3-pipe -f {merged_dir}/Modelfile
ollama run qwen3-pipe
""")


def main():
    parser = argparse.ArgumentParser(description="LoRA 合并 & 导出")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="LoRA adapter 路径")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3_finetune/merged")
    parser.add_argument("--no_modelfile", action="store_true",
                        help="不创建 Ollama Modelfile（默认会创建）")
    parser.add_argument("--quantize", type=str, default="Q4_K_M")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    merged_dir = merge_lora(adapter_path, output_dir)

    if not args.no_modelfile:
        gguf_name = f"model-{args.quantize.lower().replace('_', '')}.gguf"
        create_modelfile(merged_dir, gguf_name)

    print_gguf_guide(merged_dir, args.quantize)

    print("\n完成!")
    print(f"  合并模型路径: {merged_dir}")


if __name__ == "__main__":
    main()
