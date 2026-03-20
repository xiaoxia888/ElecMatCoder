#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GlobalPointer NER 训练脚本

使用方法:
    python apps/trainer/globalpointer_ner/train.py
    python apps/trainer/globalpointer_ner/train.py --epochs 10 --batch_size 16
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.trainer.globalpointer_ner.model import create_model, GlobalPointerForNER


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    encoder_path: str = "hfl/chinese-roberta-wwm-ext"
    head_size: int = 64
    max_len: int = 256
    dropout: float = 0.1
    
    # 训练配置
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 2e-5
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 数据配置
    train_file: str = "data/globalpointer/train.jsonl"
    val_file: str = "data/globalpointer/val.jsonl"
    labels_file: str = "data/globalpointer/labels.json"
    
    # 输出配置
    output_dir: str = "outputs/globalpointer_ner"
    save_steps: int = 500
    eval_steps: int = 500


class GlobalPointerDataset(Dataset):
    """GlobalPointer NER 数据集"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        label2id: Dict[str, int],
        max_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.num_labels = len(label2id)
        
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        entities = sample['entities']
        
        # Tokenize（字符级别）
        # 使用 is_split_into_words=True 保持字符对齐
        tokens = list(text)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids))
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = token_type_ids.squeeze(0)
        else:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 构建 word_ids 映射（token 到 word 的映射）
        word_ids = encoding.word_ids()
        
        # 构建标签矩阵 [num_labels, max_len, max_len]
        labels = torch.zeros(self.num_labels, self.max_len, self.max_len)
        
        for entity in entities:
            start_char = entity['start']
            end_char = entity['end'] - 1  # 转为闭区间
            entity_type = entity['type']
            
            if entity_type not in self.label2id:
                continue
            
            label_id = self.label2id[entity_type]
            
            # 找到对应的 token 位置
            start_token = None
            end_token = None
            
            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id == start_char and start_token is None:
                    start_token = i
                if word_id == end_char:
                    end_token = i
            
            if start_token is not None and end_token is not None:
                if start_token < self.max_len and end_token < self.max_len:
                    labels[label_id, start_token, end_token] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


def compute_f1(
    predictions: List[List[Dict]], 
    references: List[List[Dict]]
) -> Dict[str, float]:
    """
    计算 F1 分数
    
    Args:
        predictions: 预测的实体列表
        references: 真实的实体列表
        
    Returns:
        包含 precision, recall, f1 的字典
    """
    tp = 0
    pred_count = 0
    true_count = 0
    
    for pred_entities, true_entities in zip(predictions, references):
        # 转换为集合进行比较 (start, end, type)
        pred_set = {(e['start'], e['end'], e['type']) for e in pred_entities}
        true_set = {(e['start'], e['end'], e['type']) for e in true_entities}
        
        tp += len(pred_set & true_set)
        pred_count += len(pred_set)
        true_count += len(true_set)
    
    precision = tp / pred_count if pred_count > 0 else 0
    recall = tp / true_count if true_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(
    model: GlobalPointerForNER,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    max_grad_norm: float = 1.0
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 更新参数
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(
    model: GlobalPointerForNER,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
            
            # 解码预测结果
            predictions = model.decode(outputs['logits'], attention_mask, threshold)
            all_predictions.extend(predictions)
            
            # 解码真实标签
            batch_size = labels.shape[0]
            for i in range(batch_size):
                entities = []
                for label_id in range(model.num_labels):
                    indices = torch.where(labels[i, label_id] == 1)
                    for start, end in zip(indices[0].tolist(), indices[1].tolist()):
                        entities.append({
                            'start': start,
                            'end': end + 1,
                            'type': model.id2label[label_id]
                        })
                all_references.append(entities)
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_f1(all_predictions, all_references)
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='GlobalPointer NER 训练')
    parser.add_argument('--encoder', type=str, default='hfl/chinese-roberta-wwm-ext',
                        help='预训练编码器路径')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--max_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--output_dir', type=str, default='outputs/globalpointer_ner',
                        help='输出目录')
    parser.add_argument('--train_file', type=str, default='data/globalpointer/train.jsonl',
                        help='训练数据')
    parser.add_argument('--val_file', type=str, default='data/globalpointer/val.jsonl',
                        help='验证数据')
    parser.add_argument('--labels_file', type=str, default='data/globalpointer/labels.json',
                        help='标签文件')
    args = parser.parse_args()
    
    # 设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("=" * 60)
    print("GlobalPointer NER 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"编码器: {args.encoder}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"最大长度: {args.max_len}")
    
    # 加载标签
    labels_path = PROJECT_ROOT / args.labels_file
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"\n标签数量: {len(labels)}")
    print(f"标签: {labels}")
    
    # 加载 tokenizer
    print(f"\n加载 tokenizer: {args.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    
    # 创建模型
    print(f"创建模型...")
    model = create_model(
        encoder_path=args.encoder,
        labels=labels,
        head_size=64,
        max_len=args.max_len,
        dropout=0.1
    )
    model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 加载数据
    print(f"\n加载数据...")
    train_dataset = GlobalPointerDataset(
        file_path=str(PROJECT_ROOT / args.train_file),
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=args.max_len
    )
    val_dataset = GlobalPointerDataset(
        file_path=str(PROJECT_ROOT / args.val_file),
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=args.max_len
    )
    
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # 优化器（分层学习率）
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.learning_rate},
        {'params': head_params, 'lr': args.learning_rate * 10}
    ], weight_decay=0.01)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        print(f"  训练损失: {train_loss:.4f}")
        
        # 评估
        val_loss, metrics = evaluate(model, val_loader, device)
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"  ✓ 新最佳 F1: {best_f1:.4f}")
            
            # 保存模型
            model_path = output_dir / 'best_model'
            model_path.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), model_path / 'pytorch_model.bin')
            tokenizer.save_pretrained(model_path)
            
            # 保存配置
            config = {
                'encoder_path': args.encoder,
                'labels': labels,
                'label2id': label2id,
                'head_size': 64,
                'max_len': args.max_len,
                'best_f1': best_f1
            }
            with open(model_path / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳 F1: {best_f1:.4f}")
    print(f"模型保存到: {output_dir / 'best_model'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
