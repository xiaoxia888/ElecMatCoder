# -*- coding: utf-8 -*-
"""
微调语义模型 - 让模型更好地理解材料编码的相似性

使用方法:
    python apps/trainer/finetune_semantic.py

可调参数:
    --epochs: 训练轮数 (默认 10)
    --batch_size: 批次大小 (默认 16)
    --lr: 学习率 (默认 2e-5)
    --warmup_ratio: 预热比例 (默认 0.1)

依赖:
    pip install sentence-transformers torch scikit-learn
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.training_args import BatchSamplers
from torch.utils.data import DataLoader
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='微调语义模型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率 (默认: 2e-5)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热步数占总步数的比例 (默认: 0.1)')
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='训练数据目录 (默认: data/finetune)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='模型输出目录 (默认: models/semantic_model_finetuned)')
    parser.add_argument('--base_model', type=str, default=None,
                        help='基础模型路径 (默认: models/semantic_model)')
    
    # 其他参数
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='每多少步评估一次 (默认: 100)')
    parser.add_argument('--save_best', action='store_true', default=True,
                        help='只保存最佳模型')
    
    return parser.parse_args()


def load_pairs(file_path: Path):
    """加载样本对数据（返回 InputExample 列表）"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            examples.append(InputExample(
                texts=[data['sentence1'], data['sentence2']],
                label=float(data['label'])
            ))
    return examples


def load_pairs_as_dataset(file_path: Path) -> Dataset:
    """加载样本对数据（返回 HuggingFace Dataset）"""
    data = {'sentence1': [], 'sentence2': [], 'label': []}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data['sentence1'].append(item['sentence1'])
            data['sentence2'].append(item['sentence2'])
            data['label'].append(float(item['label']))
    return Dataset.from_dict(data)


def create_evaluator(val_pairs_path: Path):
    """创建评估器"""
    sentences1 = []
    sentences2 = []
    labels = []
    
    with open(val_pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            sentences1.append(data['sentence1'])
            sentences2.append(data['sentence2'])
            labels.append(float(data['label']))
    
    return evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, labels,
        name='material-similarity',
        show_progress_bar=True
    )


def test_model(model, test_cases=None):
    """测试模型效果"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if test_cases is None:
        test_cases = [
            # (查询, 应该匹配的标准名, 不应该匹配的)
            ("S30408", "304", "2205"),
            ("S31603", "316L", "304L"),
            ("ASTM A182 F304", "304", "316"),
            ("022Cr17Ni12Mo2", "316L", "304L"),
            ("TP304L", "304L", "304"),
            ("S30408Ⅱ", "304", "316"),
            ("ASTMA403GRADEWP304-S", "304", "304L"),
        ]
    
    logger.info("\n" + "="*60)
    logger.info("模型效果测试")
    logger.info("="*60)
    
    correct = 0
    total = len(test_cases)
    
    for query, should_match, should_not_match in test_cases:
        embeddings = model.encode([query, should_match, should_not_match])
        
        sim_correct = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_wrong = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        is_correct = sim_correct > sim_wrong
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        
        logger.info(f"{status} '{query}':")
        logger.info(f"   vs '{should_match}': {sim_correct:.4f}")
        logger.info(f"   vs '{should_not_match}': {sim_wrong:.4f}")
    
    accuracy = correct / total * 100
    logger.info(f"\n测试准确率: {correct}/{total} = {accuracy:.1f}%")
    return accuracy


def main():
    args = parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "finetune"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "semantic_model_finetuned"
    
    # 基础模型路径
    if args.base_model:
        base_model_path = args.base_model
    else:
        local_model = project_root / "models" / "semantic_model"
        if local_model.exists():
            base_model_path = str(local_model)
        else:
            base_model_path = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # 检查训练数据 - 优先使用包含 TYPE 的数据
    train_pairs_path = data_dir / "train_pairs_with_type.jsonl"
    val_pairs_path = data_dir / "val_pairs_with_type.jsonl"
    
    # 如果新数据不存在，回退到旧数据
    if not train_pairs_path.exists():
        train_pairs_path = data_dir / "train_pairs.jsonl"
        val_pairs_path = data_dir / "val_pairs.jsonl"
    
    if not train_pairs_path.exists():
        logger.error(f"训练数据不存在: {train_pairs_path}")
        logger.error("请先运行: python scripts/generate_finetune_data.py")
        return
    
    # 打印配置
    logger.info("="*60)
    logger.info("训练配置")
    logger.info("="*60)
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"训练数据: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"预热比例: {args.warmup_ratio}")
    logger.info("="*60)
    
    # 加载模型
    logger.info("加载基础模型...")
    model = SentenceTransformer(base_model_path)
    
    # 测试微调前的效果
    logger.info("\n>>> 微调前的模型效果:")
    test_model(model)
    
    # 加载训练数据
    logger.info("\n加载训练数据...")
    train_dataset = load_pairs_as_dataset(train_pairs_path)
    logger.info(f"加载了 {len(train_dataset)} 个训练样本")
    
    # 加载验证数据
    eval_dataset = None
    if val_pairs_path.exists():
        eval_dataset = load_pairs_as_dataset(val_pairs_path)
        logger.info(f"加载了 {len(eval_dataset)} 个验证样本")
    
    # 使用 CosineSimilarityLoss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 创建评估器（用于最终评估）
    evaluator = None
    if val_pairs_path.exists():
        evaluator = create_evaluator(val_pairs_path)
    
    # 计算训练步数
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练参数 - 使用新的 Trainer API 获得详细日志
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        
        # 日志设置 - 打印损失、学习率
        logging_strategy="steps",
        logging_steps=10,  # 每10步打印一次
        
        # 评估设置
        eval_strategy="epoch",  # 每个 epoch 评估一次
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 其他
        fp16=torch.cuda.is_available(),  # GPU 时使用混合精度
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    logger.info(f"\n开始训练...")
    logger.info(f"  - 每 epoch 步数: {steps_per_epoch}")
    logger.info(f"  - 总步数: {total_steps}")
    logger.info(f"  - 预热步数: {warmup_steps}")
    logger.info(f"  - 日志间隔: 每 10 步")
    
    # 创建 Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    model.save(str(output_dir))
    
    logger.info(f"\n训练完成！模型保存到: {output_dir}")
    
    # 测试微调后的效果
    logger.info("\n>>> 微调后的模型效果:")
    finetuned_model = SentenceTransformer(str(output_dir))
    test_model(finetuned_model)
    
    logger.info("\n" + "="*60)
    logger.info("下一步: 更新配置使用微调后的模型")
    logger.info("修改 src/config/platform_config.yaml:")
    logger.info("  semantic_model:")
    logger.info("    local_model_path: models/semantic_model_finetuned")
    logger.info("="*60)


if __name__ == "__main__":
    main()
