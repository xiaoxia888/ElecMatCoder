# -*- coding: utf-8 -*-
"""
从训练 run 目录中的最新 checkpoint 恢复一个可直接使用的 final adapter 目录。

适用场景：
- 训练已经跑完，但在保存 final 之前因画图/依赖报错中断
- 不想重新训练，只想复用最新 checkpoint
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path_str


def _find_latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"未找到 checkpoint-* 目录: {run_dir}")
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _ensure_tokenizer(adapter_dir: Path) -> None:
    if (adapter_dir / "tokenizer_config.json").exists():
        return

    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return

    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(adapter_dir))


def main():
    parser = argparse.ArgumentParser(description="从最新 checkpoint 恢复 final adapter")
    parser.add_argument("--run_dir", required=True, help="训练输出目录 run_xxx")
    parser.add_argument(
        "--output_dir",
        default="",
        help="恢复后的输出目录，默认写入 run_dir/final_recovered",
    )
    args = parser.parse_args()

    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir) if args.output_dir else run_dir / "final_recovered"

    latest_ckpt = _find_latest_checkpoint(run_dir)
    _copy_tree(latest_ckpt, output_dir)
    _ensure_tokenizer(output_dir)

    print(f"latest_checkpoint: {latest_ckpt}")
    print(f"recovered_final:   {output_dir}")
    if (run_dir / "final").exists():
        print("note: run_dir/final 已存在，你可以直接优先使用 final")
    else:
        print("note: 这是从最新 checkpoint 恢复的近似 final，不一定等同于训练最后一步权重")


if __name__ == "__main__":
    main()
