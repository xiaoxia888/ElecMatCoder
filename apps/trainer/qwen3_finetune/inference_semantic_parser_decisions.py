# -*- coding: utf-8 -*-
"""
一阶段语义解析器专用推理入口（仅输出 decisions）。
用于测试只生成最终结构化决策结果时的速度与效果。
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import apps.trainer.qwen3_finetune.inference as base_inference
from src.llm_ner.prompts import get_stage1_decisions_only_prompt

base_inference.SYSTEM_PROMPT = get_stage1_decisions_only_prompt()


def main() -> None:
    base_inference.main()


if __name__ == "__main__":
    main()
