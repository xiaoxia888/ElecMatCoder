# -*- coding: utf-8 -*-
"""
一阶段语义解析器专用推理入口。
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import apps.trainer.qwen3_finetune.inference as base_inference
from src.llm_ner.prompts import SEMANTIC_PARSER_SYSTEM_PROMPT

base_inference.SYSTEM_PROMPT = SEMANTIC_PARSER_SYSTEM_PROMPT


def main() -> None:
    base_inference.main()


if __name__ == "__main__":
    main()

