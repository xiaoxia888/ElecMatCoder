# -*- coding: utf-8 -*-
"""
Qwen3 微调数据准备（统一入口）

将 NER 源数据 + 编码源数据 转换为 ChatML 训练格式。

源数据格式 (JSON):
  NER:    [{"input": "描述文本", "output": {
              "TYPE": {"BODY": "...", "CONN": "...", "ENDS": "...", "SEAL": "...", "MANU": "..."},
              "SIZE": {"DN": [...], "OD": [...], "INCH": [...], "LENGTH": [...]},
              "MATERIAL": {"RELATION": "...", "ITEMS": [...]},
              ...
          }}, ...]
  编码:   [{"input": {"TYPE": "...", ...}, "output": {"TYPE": "...", ...}}, ...]

说明:
  - 本脚本对 NER 样本采用“原样透传”策略，不改写 `output` 结构。
  - 因此一阶段 schema 的变化主要由 `src.llm_ner.predictor.SYSTEM_PROMPT`
    和 `data/pipe/llm_lora/ner_data_new_schema.json` 决定。
  - 当前编码样本仍沿用二阶段既有输入格式，不在本脚本内做结构迁移。

输出格式 (JSONL, ChatML):
  {"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}

用法:
  # 默认：合并 NER + 编码 → data/pipe/qwen3_mixed/
  python prepare_training_data.py

  # 只生成 NER 训练数据
  python prepare_training_data.py --ner_only

  # 指定路径
  python prepare_training_data.py --ner_data path/to/ner.json --encoding_data path/to/enc.json --output_dir path/to/out
"""

import json
import random
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_ner.predictor import SYSTEM_PROMPT as NER_SYSTEM_PROMPT
from src.llm_ner.predictor import ENCODING_SYSTEM_PROMPT
from src.tokenizer_utils.preprocessor import TextPreprocessor
from apps.trainer.qwen3_finetune.augment_ner_glue import augment_ner_samples_inplace


def load_json_or_jsonl(path: Path) -> list:
    """自动识别 JSON / JSONL 格式加载"""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def to_chatml(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert_ner(samples: list, preprocessor: TextPreprocessor | None = None) -> list:
    """NER 源数据 → ChatML。保持新 schema 的 output 原样写入 assistant。"""
    results = []
    for s in samples:
        user = s["input"]
        if preprocessor is not None:
            user = preprocessor.process(user)
        output = s["output"]
        assistant = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
        results.append(to_chatml(NER_SYSTEM_PROMPT, user, assistant))
    return results


def convert_encoding(samples: list) -> list:
    """编码源数据 → ChatML。当前仍按二阶段既有字符串输入 schema 处理。"""
    results = []
    for s in samples:
        inp = s["input"]
        output = s["output"]
        user = json.dumps(inp, ensure_ascii=False) if isinstance(inp, dict) else str(inp)
        assistant = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
        results.append(to_chatml(ENCODING_SYSTEM_PROMPT, user, assistant))
    return results


def write_jsonl(path: Path, data: list):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedupe_samples(samples: list) -> tuple[list, int]:
    """按 input+output 精确去重，保留首次出现。"""
    seen = set()
    deduped = []
    removed = 0
    for s in samples:
        key = json.dumps(
            {"input": s.get("input"), "output": s.get("output")},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(s)
    return deduped, removed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 微调数据准备：源数据 → ChatML 训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python prepare_training_data.py\n"
               "  python prepare_training_data.py --ner_only\n"
               "  python prepare_training_data.py --ner_data my_ner.json --output_dir my_output/\n"
    )
    parser.add_argument("--ner_data", type=str,
                        default="data/pipe/llm_lora/ner_data_new_schema.json",
                        help="NER 源数据路径 (JSON/JSONL)")
    parser.add_argument("--encoding_data", type=str,
                        default="data/pipe/llm_lora/encoding_data.json",
                        help="编码源数据路径 (JSON/JSONL)")
    parser.add_argument("--output_dir", type=str,
                        default="data/pipe/qwen3_mixed",
                        help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例 (默认 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--ner_only", action="store_true",
                        help="只生成 NER 训练数据")
    parser.add_argument("--encoding_only", action="store_true",
                        help="只生成编码训练数据")
    parser.add_argument("--augment", action="store_true",
                        help="启用 NER 粘连数据增强（替换式，不改变数据量）")
    parser.add_argument("--glue_prob", type=float, default=0.3,
                        help="粘连增强替换概率 (默认 0.3，即 30%% 可粘连样本被替换)")
    parser.add_argument("--no_dedupe", action="store_true",
                        help="关闭 input+output 精确去重（默认开启）")
    args = parser.parse_args()

    random.seed(args.seed)
    preprocessor = TextPreprocessor()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    all_chatml = []

    if not args.encoding_only:
        ner_path = resolve(args.ner_data)
        if ner_path.exists():
            ner_samples = load_json_or_jsonl(ner_path)
            print(f"NER 数据:  {len(ner_samples):>6} 条  ← {ner_path}")

            if not args.no_dedupe:
                ner_samples, removed = dedupe_samples(ner_samples)
                if removed:
                    print(f"  精确去重: 删除 {removed} 条重复样本")

            if args.augment:
                ner_samples, aug_stats = augment_ner_samples_inplace(
                    ner_samples,
                    glue_prob=args.glue_prob,
                    seed=args.seed,
                )
                print(f"  粘连增强: 可粘连 {aug_stats['glueable']} 条, "
                      f"替换 {aug_stats['replaced']} 条 (prob={args.glue_prob})")
                if aug_stats['pair_counts']:
                    for pair, cnt in sorted(aug_stats['pair_counts'].items(),
                                            key=lambda x: -x[1]):
                        print(f"    {pair}: {cnt}")

            ner_chatml = convert_ner(ner_samples, preprocessor=preprocessor)
            all_chatml.extend(ner_chatml)
        else:
            print(f"⚠ NER 数据文件不存在: {ner_path}")

    if not args.ner_only:
        enc_path = resolve(args.encoding_data)
        if enc_path.exists():
            enc_samples = load_json_or_jsonl(enc_path)
            if not args.no_dedupe:
                enc_samples, removed = dedupe_samples(enc_samples)
                if removed:
                    print(f"  编码数据去重: 删除 {removed} 条重复样本")
            enc_chatml = convert_encoding(enc_samples)
            all_chatml.extend(enc_chatml)
            print(f"编码数据:  {len(enc_samples):>6} 条  ← {enc_path}")
        else:
            print(f"⚠ 编码数据文件不存在: {enc_path}")

    if not all_chatml:
        print("错误: 没有加载到任何数据")
        sys.exit(1)

    random.shuffle(all_chatml)
    val_count = int(len(all_chatml) * args.val_ratio)
    val_data = all_chatml[:val_count]
    train_data = all_chatml[val_count:]

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    write_jsonl(train_path, train_data)
    write_jsonl(val_path, val_data)

    print(f"\n{'='*50}")
    print(f"合计:      {len(all_chatml):>6} 条")
    print(f"训练集:    {len(train_data):>6} 条  → {train_path}")
    print(f"验证集:    {len(val_data):>6} 条  → {val_path}")
    print(f"{'='*50}")

    print(f"\n--- 训练样本示例 ---")
    sample = train_data[0]
    for msg in sample["messages"]:
        content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
        print(f"  [{msg['role']}] {content}")


if __name__ == "__main__":
    main()
