#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIO 格式转 Span 格式数据转换脚本

将 BIO 标注格式转换为 GlobalPointer 所需的 Span 格式
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def bio_to_spans(text: str, labels: List[str]) -> List[Dict]:
    """
    将 BIO 标签转换为 Span 格式
    
    Args:
        text: 原始文本
        labels: BIO 标签列表
        
    Returns:
        实体列表 [{"start": int, "end": int, "type": str, "text": str}, ...]
    """
    if len(text) != len(labels):
        raise ValueError(f"文本长度 ({len(text)}) 与标签长度 ({len(labels)}) 不匹配")
    
    entities = []
    current_entity = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            # 保存之前的实体
            if current_entity is not None:
                current_entity['end'] = i
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
            
            # 开始新实体
            entity_type = label[2:]
            current_entity = {
                'start': i,
                'end': i + 1,
                'type': entity_type
            }
        elif label.startswith('I-'):
            entity_type = label[2:]
            if current_entity is not None and current_entity['type'] == entity_type:
                # 继续当前实体
                current_entity['end'] = i + 1
            else:
                # I 标签但没有对应的 B 标签，作为新实体开始
                if current_entity is not None:
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                    entities.append(current_entity)
                current_entity = {
                    'start': i,
                    'end': i + 1,
                    'type': entity_type
                }
        else:  # O 标签
            if current_entity is not None:
                current_entity['end'] = i
                current_entity['text'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = None
    
    # 处理最后一个实体
    if current_entity is not None:
        current_entity['end'] = len(text)
        current_entity['text'] = text[current_entity['start']:current_entity['end']]
        entities.append(current_entity)
    
    return entities


def convert_file(input_path: str, output_path: str) -> Tuple[int, int, Dict[str, int]]:
    """
    转换单个文件
    
    Args:
        input_path: 输入文件路径 (BIO 格式 JSONL)
        output_path: 输出文件路径 (Span 格式 JSONL)
        
    Returns:
        (成功数, 失败数, 实体类型统计)
    """
    success_count = 0
    error_count = 0
    entity_type_counts = defaultdict(int)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                text = data['text']
                labels = data['ner_labels']
                
                # 转换为 Span 格式
                entities = bio_to_spans(text, labels)
                
                # 统计实体类型
                for entity in entities:
                    entity_type_counts[entity['type']] += 1
                
                # 输出新格式
                output_data = {
                    'text': text,
                    'entities': entities
                }
                
                # 保留其他字段
                if 'type_class' in data:
                    output_data['type_class'] = data['type_class']
                
                fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                success_count += 1
                
            except Exception as e:
                print(f"  行 {line_num} 转换失败: {e}")
                error_count += 1
    
    return success_count, error_count, dict(entity_type_counts)


def main():
    parser = argparse.ArgumentParser(description='BIO 转 Span 格式')
    parser.add_argument('--input', '-i', type=str, 
                        default='data/pipe/raw/总数据_enhanced.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default='data/globalpointer/train.jsonl',
                        help='输出文件路径')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("BIO → Span 格式转换")
    print("=" * 50)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    
    # 先读取所有数据
    all_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    text = data['text']
                    labels = data['ner_labels']
                    entities = bio_to_spans(text, labels)
                    
                    output_data = {
                        'text': text,
                        'entities': entities
                    }
                    if 'type_class' in data:
                        output_data['type_class'] = data['type_class']
                    
                    all_data.append(output_data)
                except Exception as e:
                    pass
    
    print(f"总样本数: {len(all_data)}")
    
    # 划分训练集和验证集
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    val_size = int(len(all_data) * args.val_split)
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]
    
    # 保存训练集
    train_path = output_path
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"训练集: {len(train_data)} 条 -> {train_path}")
    
    # 保存验证集
    val_path = output_path.parent / 'val.jsonl'
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"验证集: {len(val_data)} 条 -> {val_path}")
    
    # 统计实体类型
    entity_counts = defaultdict(int)
    for item in all_data:
        for entity in item['entities']:
            entity_counts[entity['type']] += 1
    
    print(f"\n实体类型统计:")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    
    # 保存标签文件
    labels_path = output_path.parent / 'labels.json'
    labels = sorted(entity_counts.keys())
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"\n标签文件: {labels_path}")
    
    print("\n转换完成!")


if __name__ == '__main__':
    main()
