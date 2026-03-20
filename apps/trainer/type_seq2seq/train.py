#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TYPE Seq2Seq 编码生成模型训练脚本

使用方法:
    python apps/trainer/type_seq2seq/train.py
    python apps/trainer/type_seq2seq/train.py --config src/seq2seq/config/training.yml
    python apps/trainer/type_seq2seq/train.py --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import json
import yaml
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.seq2seq.model import preprocess_text


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(file_path: str) -> List[Dict]:
    """加载JSONL数据"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def prepare_dataset(
    samples: List[Dict],
    tokenizer,
    max_input_length: int = 64,
    max_output_length: int = 16
) -> HFDataset:
    """准备 HuggingFace Dataset"""
    
    def preprocess_function(examples):
        # 预处理输入
        inputs = [preprocess_text(inp) for inp in examples['input']]
        
        # Tokenize 输入
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize 输出（作为 labels）
        # 注：新版 transformers 已移除 as_target_tokenizer()，直接使用 tokenizer
        labels = tokenizer(
            text_target=examples['output'],
            max_length=max_output_length,
            truncation=True,
            padding='max_length'
        )
        
        # 将 padding token 替换为 -100
        labels_ids = []
        for label in labels['input_ids']:
            labels_ids.append([
                -100 if token == tokenizer.pad_token_id else token 
                for token in label
            ])
        
        model_inputs['labels'] = labels_ids
        return model_inputs
    
    # 转换为 HuggingFace Dataset
    dataset = HFDataset.from_dict({
        'input': [s['input'] for s in samples],
        'output': [s['output'] for s in samples]
    })
    
    # 应用预处理
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['input', 'output']
    )
    
    return dataset


def compute_metrics(eval_preds, tokenizer):
    """计算评估指标"""
    predictions, labels = eval_preds
    
    # 解码预测结果
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 处理 labels（将 -100 替换回 pad_token_id）
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算精确匹配
    exact_match = sum(
        pred.strip() == label.strip() 
        for pred, label in zip(decoded_preds, decoded_labels)
    ) / len(decoded_preds)
    
    return {
        'exact_match': exact_match
    }


def main():
    parser = argparse.ArgumentParser(description='训练 TYPE Seq2Seq 编码生成模型')
    parser.add_argument(
        '--config', 
        type=str, 
        default='src/seq2seq/config/training.yml',
        help='配置文件路径'
    )
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--model', type=str, help='预训练模型')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = PROJECT_ROOT / args.config
    print(f"加载配置: {config_path}")
    config = load_config(config_path)
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.model:
        config['model']['pretrained_model'] = args.model
    if args.output_dir:
        config['model']['output_dir'] = args.output_dir
    
    # 设置随机种子
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 设备
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_config
    print(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    model_name = config['model']['pretrained_model']
    print(f"加载预训练模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 加载数据
    train_file = PROJECT_ROOT / config['data']['train_file']
    print(f"加载训练数据: {train_file}")
    train_samples = load_data(train_file)
    print(f"  样本数: {len(train_samples)}")
    
    # 划分验证集
    val_file = config['data'].get('val_file')
    if val_file and (PROJECT_ROOT / val_file).exists():
        val_samples = load_data(PROJECT_ROOT / val_file)
    else:
        val_split = config['data'].get('val_split', 0.1)
        random.shuffle(train_samples)
        split_idx = int(len(train_samples) * (1 - val_split))
        val_samples = train_samples[split_idx:]
        train_samples = train_samples[:split_idx]
    
    print(f"  训练集: {len(train_samples)}")
    print(f"  验证集: {len(val_samples)}")
    
    # 准备数据集
    max_input_length = config['data'].get('max_input_length', 64)
    max_output_length = config['data'].get('max_output_length', 16)
    
    train_dataset = prepare_dataset(
        train_samples, tokenizer, max_input_length, max_output_length
    )
    val_dataset = prepare_dataset(
        val_samples, tokenizer, max_input_length, max_output_length
    )
    
    # 输出目录
    output_dir = PROJECT_ROOT / config['model']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练参数
    training_config = config['training']
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(training_config['epochs']),
        per_device_train_batch_size=int(training_config['batch_size']),
        per_device_eval_batch_size=int(training_config['batch_size']),
        learning_rate=float(training_config['learning_rate']),
        warmup_ratio=float(training_config.get('warmup_ratio', 0.1)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        
        # 评估和保存
        eval_strategy='epoch',
        save_strategy=training_config.get('save_strategy', 'epoch'),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model='exact_match',
        greater_is_better=True,
        
        # 生成配置
        predict_with_generate=True,
        generation_max_length=config['generation'].get('max_length', 16),
        generation_num_beams=config['generation'].get('num_beams', 4),
        
        # 日志
        logging_dir=str(output_dir / 'logs'),
        logging_steps=10,
        report_to='none',
        
        # 其他
        seed=seed,
        fp16=device == 'cuda',  # GPU 使用混合精度
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    # 早停回调
    early_stopping = config['training'].get('early_stopping', {})
    callbacks = []
    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping.get('patience', 5)
            )
        )
    
    # 创建 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # 新版 transformers 使用 processing_class 替代 tokenizer
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=callbacks
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    trainer.train()
    
    # 保存最终模型
    final_model_dir = output_dir / 'final_model'
    print(f"\n保存最终模型至: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # 评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    
    eval_results = trainer.evaluate()
    print(f"验证集精确匹配率: {eval_results['eval_exact_match']:.2%}")
    
    # 保存评估结果
    with open(output_dir / 'eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    # 测试一些样本
    print("\n" + "=" * 60)
    print("测试样本")
    print("=" * 60)
    
    test_inputs = [
        "无缝钢管",
        "管子",
        "PIPE",
        "45度弯头",
        "90°长半径弯头",
        "偏心异径管 REDUCER ECC",
        "不锈钢管件同心异径管",
    ]
    
    model.eval()
    for inp in test_inputs:
        processed = preprocess_text(inp)
        inputs = tokenizer(processed, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=16,
                num_beams=4
            )
        
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {inp} → {code}")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
