"""
TYPE 字段分类模型训练脚本
使用 xlm-roberta-base 进行分类
"""
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(args):
    """训练分类模型"""
    logger.info(f"开始训练 TYPE 分类模型")
    logger.info(f"模型: {args.model_name}")
    logger.info(f"数据目录: {args.data_dir}")
    
    # 加载标签映射
    data_dir = Path(args.data_dir)
    with open(data_dir / "labels.json", 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    
    label2id = label_info['label2id']
    id2label = {int(k): v for k, v in label_info['id2label'].items()}
    num_labels = label_info['num_labels']
    
    logger.info(f"类别数: {num_labels}")
    
    # 加载 tokenizer 和模型
    logger.info("加载模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    
    # 加载数据集
    dataset = load_dataset('json', data_files={
        'train': str(data_dir / "train.jsonl"),
        'validation': str(data_dir / "val.jsonl")
    })
    
    # 预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            max_length=args.max_length,
            truncation=True,
            padding='max_length'
        )
    
    # 预处理数据
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['text', 'label_name']
    )
    
    logger.info(f"训练集大小: {len(tokenized_dataset['train'])}")
    logger.info(f"验证集大小: {len(tokenized_dataset['validation'])}")
    
    # 训练参数
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # 保存标签映射
    with open(final_model_dir / "labels.json", 'w', encoding='utf-8') as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"模型已保存到: {final_model_dir}")
    
    # 评估
    logger.info("评估模型...")
    eval_results = trainer.evaluate()
    logger.info(f"评估结果: {eval_results}")
    
    # 保存评估结果
    with open(output_dir / "eval_results.json", 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    return trainer, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 TYPE 分类模型")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="预训练模型名称")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=64, help="最大序列长度")
    
    args = parser.parse_args()
    train(args)
