# -*- coding: utf-8 -*-
"""
一阶段语义解析器专用训练入口。

仅训练:
原始描述 -> {mentions, semantics, decisions}

二阶段编码训练保留原有入口，不从这里进入。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.trainer.qwen3_finetune.train_qwen3 import main as train_qwen3_main


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="apps/trainer/qwen3_finetune/config_semantic_parser.yaml")
    known, remaining = parser.parse_known_args()

    default_config = Path(known.config)
    forwarded = ["--config", str(default_config), *remaining]

    import sys

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *forwarded]
        train_qwen3_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

