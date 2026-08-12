from __future__ import annotations

import argparse
from pathlib import Path


def load_model(base_model: str, lora_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载底座: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel

        print(f"加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    print(f"模型设备: {model.device}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    instruction: str,
    user_text: str,
    max_new_tokens: int,
    show_prompt: bool,
    enable_thinking: bool,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_text},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    if show_prompt:
        print("\n===== 实际模型输入 =====")
        print(rendered_prompt)
        print("===== 输入结束 =====\n")

    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    if enable_thinking:
        generation_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
    else:
        generation_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Qwen 官方 apply_chat_template 测试 qwen3_nothink 模型",
    )
    parser.add_argument("--base-model", required=True, help="底座模型路径")
    parser.add_argument("--lora", help="可选的 LoRA checkpoint 路径")
    parser.add_argument("--prompt-file", required=True, type=Path, help="system 提示词文件")
    parser.add_argument("--text", help="单条描述；不传则进入交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否开启 Qwen 思考模式；默认关闭。可使用 --enable-thinking 开启，"
            "或使用 --no-enable-thinking 显式关闭"
        ),
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="打印 apply_chat_template 生成的完整模型输入",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_path = args.prompt_file.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")
    instruction = prompt_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"提示词文件为空: {prompt_path}")

    model, tokenizer = load_model(args.base_model, args.lora)

    if args.text:
        result = generate(
            model,
            tokenizer,
            instruction,
            args.text,
            args.max_new_tokens,
            args.show_prompt,
            args.enable_thinking,
        )
        print(result)
        return

    print("\n交互模式：输入材料描述，输入 quit 或 exit 退出。")
    while True:
        try:
            user_text = input("\n描述> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        result = generate(
            model,
            tokenizer,
            instruction,
            user_text,
            args.max_new_tokens,
            args.show_prompt,
            args.enable_thinking,
        )
        print(result)


if __name__ == "__main__":
    main()
