# -*- coding: utf-8 -*-


"""
python scripts/查看尺寸壁厚磅级一阶段正则提取.py 'SPECTACLE BLANK CL300(PN50) RF NB/T 47008 20 ENR STD 40T018 DN250'
"""


from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoder.processors.pressure_processor import PressureProcessor
from src.encoder.processors.rule_extraction import (
    _augment_size_result_with_size_pair_echo,
    _classify_od_pair_decisions,
    build_structured_rule_entities,
    extract_size_and_thickness_by_rules,
)
from src.encoder.processors.size_processor import SizeProcessor
from src.encoder.processors.thickness_processor import ThicknessProcessor
from src.tokenizer_utils.preprocessor import TextPreprocessor


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _run_rule_pipeline(text: str) -> dict[str, Any]:
    preprocessor = TextPreprocessor()
    size_processor = SizeProcessor()
    thickness_processor = ThicknessProcessor(enable_rule_layered=False)
    pressure_processor = PressureProcessor()

    processed_text = preprocessor.process(text)

    size_result_before = size_processor.extract_by_rules(processed_text)
    od_pair_decisions = _classify_od_pair_decisions(processed_text, size_result_before, size_processor)
    size_result_before = _augment_size_result_with_size_pair_echo(size_result_before, processed_text, od_pair_decisions)

    thickness_blocked_spans = list(getattr(size_result_before, "consumed_spans", []) or size_result_before.matched_spans)
    for decision in od_pair_decisions:
        if decision.action == "treat_as_size_pair" and decision.second_value_span not in thickness_blocked_spans:
            thickness_blocked_spans.append(decision.second_value_span)
        elif decision.action == "keep_as_thickness" and decision.second_value_span in thickness_blocked_spans:
            thickness_blocked_spans.remove(decision.second_value_span)

    thickness_result_before = thickness_processor.extract_by_rules(
        processed_text,
        size_context=size_result_before,
        blocked_spans=thickness_blocked_spans,
    )

    pressure_blocked_spans = thickness_blocked_spans + list(
        getattr(thickness_result_before, "consumed_spans", []) or thickness_result_before.matched_spans
    )
    pressure_result_before = pressure_processor.extract_by_rules(
        processed_text,
        blocked_spans=pressure_blocked_spans,
    )

    unified_result = extract_size_and_thickness_by_rules(
        processed_text,
        size_processor=size_processor,
        thickness_processor=thickness_processor,
        pressure_processor=pressure_processor,
    )

    return {
        "原始描述": text,
        "预处理后描述": processed_text,
        "OD二元组判定": [
            {
                "pair_span": list(item.pair_span),
                "second_value_span": list(item.second_value_span),
                "action": item.action,
                "second_value": item.second_value,
            }
            for item in od_pair_decisions
        ],
        "单处理器直出（调试用）": {
            "尺寸": asdict(size_result_before),
            "壁厚": asdict(thickness_result_before),
            "磅级": asdict(pressure_result_before),
        },
        "统一规则入口结果（与平台一致）": {
            "尺寸": asdict(unified_result.size),
            "壁厚": asdict(unified_result.thickness),
            "磅级": asdict(unified_result.pressure),
        },
        "最终结构化结果": build_structured_rule_entities(unified_result, original_text=processed_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="查看尺寸/壁厚/磅级一阶段正则提取结果")
    parser.add_argument("text", nargs="?", help="待分析的材料描述")
    parser.add_argument("--text", dest="text_flag", help="待分析的材料描述")
    args = parser.parse_args()

    text = args.text_flag or args.text
    if not text:
        raise SystemExit("请提供待分析的描述文本。示例：python scripts/查看尺寸壁厚磅级一阶段正则提取.py 'DN50 SCH40'")

    result = _run_rule_pipeline(text)
    print(_json(result))


if __name__ == "__main__":
    main()
