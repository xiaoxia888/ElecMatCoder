# -*- coding: utf-8 -*-
"""
离线导出训练曲线。

用途：
1. 训练时因缺少 matplotlib 等原因未成功出图
2. 不想重新训练，只想从已有 run/checkpoint 补画曲线

优先读取：
- trainer_state.json
- training_log_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.trainer.qwen3_finetune.training_curves import export_training_curves


def _load_log_history(run_dir: Path) -> list[dict]:
    candidates = []
    direct_state = run_dir / "trainer_state.json"
    if direct_state.exists():
        candidates.append(direct_state)

    direct_history = run_dir / "training_log_history.json"
    if direct_history.exists():
        candidates.append(direct_history)

    checkpoints = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        candidates.extend(reversed(checkpoints))

    for path in candidates:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("log_history"), list):
            return payload["log_history"]
        if isinstance(payload, list):
            return payload

    raise FileNotFoundError(f"未找到可用日志文件: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="离线导出训练曲线")
    parser.add_argument("--run_dir", required=True, help="训练输出目录 run_xxx")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir

    log_history = _load_log_history(run_dir)
    artifacts = export_training_curves(log_history, run_dir)
    print("训练曲线已导出:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
