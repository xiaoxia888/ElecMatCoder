#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GlobalPointer NER 预测脚本

使用方法:
    python apps/trainer/globalpointer_ner/predict.py "90度弯头 DN50 S30408"
    python apps/trainer/globalpointer_ner/predict.py  # 交互模式
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.globalpointer_ner.model import create_model, GlobalPointerForNER


class GlobalPointerPredictor:
    """GlobalPointer NER 预测器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        threshold: float = 0.0
    ):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            device: 设备 ('auto', 'cuda', 'mps', 'cpu')
            threshold: 预测阈值
        """
        self.threshold = threshold
        
        # 设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载配置
        config_path = Path(model_path) / 'config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.labels = self.config['labels']
        self.label2id = self.config['label2id']
        self.max_len = self.config['max_len']
        
        print(f"标签: {self.labels}")
        
        # 加载 tokenizer
        print(f"加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建模型
        print(f"加载模型...")
        self.model = create_model(
            encoder_path=self.config['encoder_path'],
            labels=self.labels,
            head_size=self.config.get('head_size', 64),
            max_len=self.max_len,
            dropout=0.0
        )
        
        # 加载权重
        state_dict = torch.load(
            Path(model_path) / 'pytorch_model.bin',
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成!")
    
    def predict(self, text: str) -> Dict:
        """
        预测单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果，包含 tokens 和 entities
        """
        # Tokenize（字符级别）
        tokens = list(text)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        word_ids = encoding.word_ids()
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 解码
            predictions = self.model.decode(
                outputs['logits'], 
                attention_mask, 
                self.threshold
            )[0]
        
        # 将 token 位置转换回字符位置
        entities = []
        for pred in predictions:
            start_token = pred['start']
            end_token = pred['end'] - 1  # 转为闭区间
            
            # 找到对应的字符位置
            start_char = None
            end_char = None
            
            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if i == start_token:
                    start_char = word_id
                if i == end_token:
                    end_char = word_id + 1  # 转为左闭右开
            
            if start_char is not None and end_char is not None:
                entity_text = text[start_char:end_char]
                entities.append({
                    'start': start_char,
                    'end': end_char,
                    'type': pred['type'],
                    'text': entity_text,
                    'score': pred['score']
                })
        
        # 转换为与 BERT NER 兼容的 tokens 格式
        bio_tokens = []
        char_labels = ['O'] * len(text)
        
        # 按 start 位置排序，处理重叠
        entities = sorted(entities, key=lambda x: (x['start'], -x['score']))
        
        # 标记每个字符的标签
        used_positions = set()
        filtered_entities = []
        
        for entity in entities:
            # 检查是否与已标记的位置重叠
            positions = set(range(entity['start'], entity['end']))
            if positions & used_positions:
                continue  # 跳过重叠实体
            
            filtered_entities.append(entity)
            used_positions.update(positions)
            
            # 标记 BIO
            for i in range(entity['start'], entity['end']):
                if i == entity['start']:
                    char_labels[i] = f"B-{entity['type']}"
                else:
                    char_labels[i] = f"I-{entity['type']}"
        
        # 构建 tokens 列表
        for i, char in enumerate(text):
            bio_tokens.append({
                'word': char,
                'tag': char_labels[i],
                'confidence': 1.0  # GlobalPointer 不直接提供字符级置信度
            })
        
        return {
            'text': text,
            'tokens': bio_tokens,
            'entities': filtered_entities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        return [self.predict(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description='GlobalPointer NER 预测')
    parser.add_argument('text', nargs='?', type=str, help='输入文本')
    parser.add_argument('--model', type=str, 
                        default='outputs/globalpointer_ner/best_model',
                        help='模型路径')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='预测阈值')
    args = parser.parse_args()
    
    # 初始化预测器
    model_path = PROJECT_ROOT / args.model
    predictor = GlobalPointerPredictor(
        model_path=str(model_path),
        threshold=args.threshold
    )
    
    if args.text:
        # 命令行模式
        result = predictor.predict(args.text)
        print(f"\n输入: {result['text']}")
        print(f"\n识别到的实体:")
        for entity in result['entities']:
            print(f"  [{entity['type']}] {entity['text']} "
                  f"(位置: {entity['start']}-{entity['end']}, 得分: {entity['score']:.4f})")
    else:
        # 交互模式
        print("\n" + "=" * 50)
        print("GlobalPointer NER 预测 (输入 'q' 退出)")
        print("=" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                if not text:
                    continue
                
                result = predictor.predict(text)
                
                print(f"\n识别到的实体:")
                for entity in result['entities']:
                    print(f"  [{entity['type']}] {entity['text']} "
                          f"(位置: {entity['start']}-{entity['end']}, 得分: {entity['score']:.4f})")
                
                if not result['entities']:
                    print("  (未识别到实体)")
                    
            except KeyboardInterrupt:
                break
        
        print("\n预测结束!")


if __name__ == '__main__':
    main()
