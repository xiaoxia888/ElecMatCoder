#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TYPE Seq2Seq 模型评估脚本

使用方法:
    python apps/trainer/type_seq2seq/evaluate.py --model outputs/type_seq2seq/final_model
    python apps/trainer/type_seq2seq/evaluate.py --model outputs/type_seq2seq/final_model --test_file data/seq2seq/test.jsonl
"""

import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def preprocess_text(text: str) -> str:
    """预处理输入文本"""
    processed = re.sub(r'[|/\\,;]', ' ', text)
    processed = re.sub(r'\s+', ' ', processed).strip()
    return processed


def load_data(file_path: str) -> List[Dict]:
    """加载JSONL数据"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def evaluate_model(
    model,
    tokenizer,
    test_samples: List[Dict],
    device: str,
    num_beams: int = 4,
    max_length: int = 16
) -> Dict:
    """评估模型"""
    model.eval()
    
    correct = 0
    total = len(test_samples)
    results = []
    
    for sample in test_samples:
        input_text = preprocess_text(sample['input'])
        expected = sample['output']
        
        # 生成
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )
        
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        results.append({
            'input': sample['input'],
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='评估 TYPE Seq2Seq 模型')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='data/seq2seq/type_seq2seq_train.jsonl',
        help='测试数据文件'
    )
    parser.add_argument('--num_beams', type=int, default=4, help='Beam search 宽度')
    parser.add_argument('--show_errors', action='store_true', help='显示错误样本')
    parser.add_argument('--limit', type=int, help='限制评估样本数')
    
    args = parser.parse_args()
    
    # 设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = PROJECT_ROOT / args.model
    print(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    
    # 加载测试数据
    test_file = PROJECT_ROOT / args.test_file
    print(f"加载测试数据: {test_file}")
    
    test_samples = load_data(test_file)
    if args.limit:
        test_samples = test_samples[:args.limit]
    
    print(f"  样本数: {len(test_samples)}")
    
    # 评估
    print("\n评估中...")
    eval_results = evaluate_model(
        model, tokenizer, test_samples, device, args.num_beams
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"精确匹配率: {eval_results['accuracy']:.2%}")
    print(f"正确数: {eval_results['correct']}/{eval_results['total']}")
    
    # 显示错误样本
    if args.show_errors:
        errors = [r for r in eval_results['results'] if not r['correct']]
        if errors:
            print(f"\n错误样本 ({len(errors)}个):")
            for e in errors[:20]:
                print(f"  输入: {e['input']}")
                print(f"    期望: {e['expected']}")
                print(f"    预测: {e['predicted']}")
                print()
            
            if len(errors) > 20:
                print(f"  ... 共 {len(errors)} 个错误")
    
    # 交互测试
    print("\n" + "=" * 60)
    print("交互测试 (输入 'quit' 退出)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n输入类型名称: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            processed = preprocess_text(user_input)
            inputs = tokenizer(processed, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=16,
                    num_beams=args.num_beams
                )
            
            code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"  → 编码: {code}")
            
        except KeyboardInterrupt:
            break
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()
