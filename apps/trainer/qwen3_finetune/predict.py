# -*- coding: utf-8 -*-
"""
平台同链路预测测试

使用与平台一致的预处理、NER 和编码逻辑，便于离线复现平台结果。
"""

import os
import sys
import argparse
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from apps.platform.server import (
    _preprocessor,
    _serialize_pipe_entity_for_encode,
    _append_pipe_entity,
    _build_pipe_entities_for_encode,
    _convert_pipe_result,
    get_ner_predictor,
    get_pipe_encoder,
)


def _collect_platform_predict(text: str, preprocess: bool = True) -> dict:
    processed_text = _preprocessor.process(text) if preprocess else text
    predictor = get_ner_predictor()
    result = predictor.predict(processed_text)

    entities = _build_pipe_entities_for_encode(result)
    extract_confidence = {}
    for e in result.get("entities", []):
        field = e["type"]
        conf = e.get("confidence")
        prev_conf = extract_confidence.get(field)
        if field in extract_confidence:
            if isinstance(prev_conf, list):
                prev_conf.append(conf)
            else:
                extract_confidence[field] = [prev_conf, conf]
        else:
            extract_confidence[field] = conf

    return {
        "processed_text": processed_text,
        "entities": entities,
        "extract_confidence": extract_confidence,
        "raw_entities": result.get("entities", []),
        "model_output": result.get("model_output", {}),
        "model_output_raw": result.get("model_output_raw", {}),
        "model_output_hybrid": result.get("model_output_hybrid", result.get("model_output", {})),
        "decision_log": result.get("decision_log", {}),
        "model_raw_response": result.get("model_raw_response", ""),
    }


def _strip_meta_fields(data):
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def run(text, do_encode=False, preprocess=True):
    print(f"\n{'='*60}")
    print(f"输入: {text}")

    # ── 第一步：分词 ──
    print(f"\n【第一步：分词】")
    t0 = time.perf_counter()
    predict_result = _collect_platform_predict(text, preprocess=preprocess)
    t_ner = time.perf_counter() - t0

    if predict_result["processed_text"] != text:
        print(f"  预处理文本         {predict_result['processed_text']}")

    entities = predict_result["entities"]
    raw_entities = predict_result["raw_entities"]
    model_output_raw = _strip_meta_fields(predict_result.get("model_output_raw", {}))
    model_output_hybrid = _strip_meta_fields(predict_result.get("model_output_hybrid", {}))
    decision_log = predict_result.get("decision_log", {}) or {}

    if model_output_raw:
        print("  模型原始结构化结果（raw）")
        print(json.dumps(model_output_raw, ensure_ascii=False, indent=2))
    else:
        print("  模型原始结构化结果（raw）  {}")

    if model_output_hybrid:
        print("  Hybrid结构化结果（canonical）")
        print(json.dumps(model_output_hybrid, ensure_ascii=False, indent=2))
    else:
        print("  Hybrid结构化结果（canonical）  {}")

    if decision_log:
        print("  决策日志（decision_log）")
        print(json.dumps(decision_log, ensure_ascii=False, indent=2))

    if entities:
        print("  平台聚合后结果")
        print(json.dumps(entities, ensure_ascii=False, indent=2))
    else:
        print("  平台聚合后结果  {}")

    if raw_entities:
        print("  打平实体列表")
        for e in raw_entities:
            label = e["type"]
            subtype = e.get("subtype")
            if subtype:
                label = f"{label}.{subtype}"
            print(f"  {label:18s}  {e['value']}")
    else:
        print("  未识别到实体")
    print(f"  耗时: {t_ner:.2f}s")

    if not do_encode or not entities:
        return

    # ── 第二步：编码 ──
    print(f"\n【第二步：编码】")
    t1 = time.perf_counter()
    encoder = get_pipe_encoder()
    encoded = encoder.encode(entities, text, predict_result["extract_confidence"])
    converted = _convert_pipe_result(encoded)
    t_enc = time.perf_counter() - t1

    if converted["fields"]:
        for field, info in converted["fields"].items():
            print(f"  {field:18s}  {info['original_value']} → {info['code']}")
    else:
        print("  编码失败")
    if converted.get("warnings"):
        for warning in converted["warnings"]:
            print(f"  警告                {warning}")
    print(f"  耗时: {t_enc:.2f}s")
    print(f"  总耗时: {t_ner + t_enc:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="平台同链路预测测试")
    parser.add_argument("task", choices=["ner", "all"], help="ner=只测分词, all=分词+编码")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None, help="批量输入（每行一条描述）")
    parser.add_argument("--no_preprocess", action="store_true", help="关闭平台同款预处理")
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("请提供 --text 或 --file")

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts.extend([l.strip() for l in f if l.strip()])

    for t in texts:
        run(t, do_encode=(args.task == "all"), preprocess=(not args.no_preprocess))


if __name__ == "__main__":
    main()
