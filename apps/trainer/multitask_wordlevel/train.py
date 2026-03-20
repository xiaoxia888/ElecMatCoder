"""
词级别多任务BERT模型训练脚本

与原 train_multitask.py 的关键区别：
1. 使用词级别分词（直接传入文本）而非字符级别（拆成字符列表）
2. 使用 offset_mapping 动态对齐标签
3. 不需要添加 [SPACE] 特殊 token

优势：
- Token 数量更少，训练更快
- 保留词级别语义，避免字符级别的误识别（如 "screw" 中的 "SC" 不会被误识别）
- 数据集格式不变，只修改处理逻辑

使用方法:
    # 使用默认配置训练
    python -m apps.trainer.multitask_wordlevel.train --data_file data/pipe/raw/总数据_enhanced.jsonl
    
    # 指定配置文件
    python -m apps.trainer.multitask_wordlevel.train --config src/bert_ner/config/training.yml
    
    # 命令行覆盖参数
    python -m apps.trainer.multitask_wordlevel.train --epochs 20 --batch_size 32
"""

import os
import sys
import json
import argparse
import logging
import random
from datetime import datetime
from typing import List, Dict, Tuple
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from src.bert_ner.bert_multitask import BertMultiTaskModel, create_multitask_model
from src.bert_ner.losses import DiceLoss, FocalLoss, LabelSmoothingLoss

# 【关键改动】使用词级别数据集
from .dataset import WordLevelMultiTaskDataset


def split_data(data: List[dict], train_ratio=0.8, dev_ratio=0.1, seed=42) -> Tuple[List, List, List]:
    """将数据划分为 train/dev/test"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def preprocess_sample(sample: dict) -> dict:
    """
    预处理单条样本：
    1. 全角罗马数字转半角（并扩展标签）
    2. 转换为小写
    """
    ROMAN_FULL_TO_HALF = {
        'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
        'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
        'Ⅺ': 'XI', 'Ⅻ': 'XII',
    }
    
    text = sample['text']
    ner_labels = sample['ner_labels']
    type_entity = sample.get('type_entity', '')
    
    new_text = []
    new_labels = []
    for char, label in zip(text, ner_labels):
        if char in ROMAN_FULL_TO_HALF:
            half = ROMAN_FULL_TO_HALF[char]
            new_text.append(half)
            for i, c in enumerate(half):
                if i == 0:
                    new_labels.append(label)
                else:
                    if label.startswith('B-'):
                        new_labels.append('I-' + label[2:])
                    else:
                        new_labels.append(label)
        else:
            new_text.append(char)
            new_labels.append(label)
    text = ''.join(new_text)
    ner_labels = new_labels
    
    text = text.lower()
    if type_entity:
        type_entity = type_entity.lower()
    
    result = sample.copy()
    result['text'] = text
    result['ner_labels'] = ner_labels
    if type_entity:
        result['type_entity'] = type_entity
    
    return result


def prepare_data(data_file: str, output_dir: str, seed: int = 42) -> str:
    """准备训练数据：从单个JSONL文件划分为train/dev/test"""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print("预处理数据（全角罗马数字转半角、小写化）...")
    data = [preprocess_sample(sample) for sample in data]
    
    train_data, dev_data, test_data = split_data(data, seed=seed)
    
    data_dir = os.path.join(output_dir, 'prepared_data')
    os.makedirs(data_dir, exist_ok=True)
    
    for name, subset in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        path = os.path.join(data_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据划分完成:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(dev_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    return data_dir


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str):
    """配置日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str) -> str:
    """获取设备"""
    if device_config == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_config


def plot_training_curves(history: dict, output_dir: str):
    """绘制训练曲线"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress (Word-Level)', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_ner_loss'], 'g-', label='NER Loss', linewidth=2)
    ax2.plot(epochs, history['train_type_loss'], 'b-', label='TYPE Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Task-specific Loss (Train)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(epochs, [acc * 100 for acc in history['val_ner_acc']], 'g-', label='NER Acc', linewidth=2)
    ax3.plot(epochs, [acc * 100 for acc in history['val_type_acc']], 'b-', label='TYPE Acc', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Validation Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    if history['best_epoch'] > 0:
        ax3.axvline(x=history['best_epoch'], color='orange', linestyle='--', alpha=0.7)
    
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['learning_rates'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_training_summary(history: dict, test_metrics: dict, output_dir: str, config: dict):
    """打印并保存训练总结"""
    summary = []
    summary.append("=" * 70)
    summary.append("              词级别多任务模型 - 训练总结报告")
    summary.append("=" * 70)
    summary.append(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"输出目录: {output_dir}")
    summary.append("")
    
    summary.append("-" * 70)
    summary.append("训练配置")
    summary.append("-" * 70)
    summary.append(f"  数据目录: {config['data']['data_dir']}")
    summary.append(f"  BERT模型: {config['model']['bert_model']}")
    summary.append(f"  分词方式: 词级别 (offset_mapping)")
    summary.append(f"  总轮数: {len(history['train_loss'])}")
    summary.append(f"  批次大小: {config['training']['batch_size']}")
    summary.append(f"  学习率: {config['training']['learning_rate']}")
    summary.append(f"  最佳轮次: {history['best_epoch']}")
    summary.append("")
    
    summary.append("-" * 70)
    summary.append("最终训练损失")
    summary.append("-" * 70)
    summary.append(f"  总损失: {history['train_loss'][-1]:.4f}")
    summary.append(f"  NER损失: {history['train_ner_loss'][-1]:.4f}")
    summary.append(f"  TYPE损失: {history['train_type_loss'][-1]:.4f}")
    summary.append("")
    
    summary.append("-" * 70)
    summary.append(f"最佳验证结果 (Epoch {history['best_epoch']})")
    summary.append("-" * 70)
    best_idx = history['best_epoch'] - 1
    summary.append(f"  NER准确率: {history['val_ner_acc'][best_idx]*100:.2f}%")
    summary.append(f"  TYPE准确率: {history['val_type_acc'][best_idx]*100:.2f}%")
    summary.append(f"  平均准确率: {history['best_val_acc']*100:.2f}%")
    summary.append("")
    
    summary.append("-" * 70)
    summary.append("测试集结果")
    summary.append("-" * 70)
    summary.append(f"  NER准确率: {test_metrics['ner_accuracy']*100:.2f}%")
    summary.append(f"  TYPE准确率: {test_metrics['type_accuracy']*100:.2f}%")
    avg_test = (test_metrics['ner_accuracy'] + test_metrics['type_accuracy']) / 2
    summary.append(f"  平均准确率: {avg_test*100:.2f}%")
    summary.append("")
    
    summary.append("-" * 70)
    summary.append("输出文件")
    summary.append("-" * 70)
    summary.append(f"  最佳模型: {os.path.join(output_dir, 'best_model')}")
    summary.append(f"  训练曲线: {os.path.join(output_dir, 'training_curves.png')}")
    summary.append("=" * 70)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, 
                dice_loss_fn=None, dice_weight=0.5, accumulation_steps=1) -> dict:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_ner_loss = 0
    total_type_loss = 0
    total_dice_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ner_labels = batch['ner_labels'].to(device)
        type_labels = batch['type_labels'].to(device)
        type_entity_mask = batch['type_entity_mask'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_labels=ner_labels,
            type_labels=type_labels,
            type_entity_mask=type_entity_mask
        )
        
        loss = outputs['loss']
        
        dice_loss_val = 0
        if dice_loss_fn is not None and 'ner_emissions' in outputs:
            dice_loss = dice_loss_fn(
                outputs['ner_emissions'], 
                ner_labels, 
                attention_mask
            )
            loss = loss + dice_weight * dice_loss
            dice_loss_val = dice_loss.item()
            total_dice_loss += dice_loss_val
        
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        if 'ner_loss' in outputs:
            total_ner_loss += outputs['ner_loss'].item()
        if 'type_loss' in outputs:
            total_type_loss += outputs['type_loss'].item()
        
        current_lr = scheduler.get_last_lr()[0]
        
        postfix = {
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'ner': f'{outputs.get("ner_loss", torch.tensor(0)).item():.4f}',
            'type': f'{outputs.get("type_loss", torch.tensor(0)).item():.4f}',
            'lr': f'{current_lr:.2e}'
        }
        if dice_loss_fn is not None:
            postfix['dice'] = f'{dice_loss_val:.4f}'
        
        progress_bar.set_postfix(postfix)
    
    n = len(dataloader)
    result = {
        'loss': total_loss / n,
        'ner_loss': total_ner_loss / n,
        'type_loss': total_type_loss / n,
        'learning_rate': scheduler.get_last_lr()[0]
    }
    if dice_loss_fn is not None:
        result['dice_loss'] = total_dice_loss / n
    
    return result


def extract_entities(labels: List[int], id2label: dict) -> set:
    """从标签序列中提取实体"""
    entities = set()
    current_type = None
    start = None
    
    for i, label_id in enumerate(labels):
        label = id2label.get(label_id, 'O')
        
        if label.startswith('B-'):
            if current_type is not None:
                entities.add((current_type, start, i))
            current_type = label[2:]
            start = i
        elif label.startswith('I-'):
            if current_type != label[2:]:
                if current_type is not None:
                    entities.add((current_type, start, i))
                current_type = None
                start = None
        else:
            if current_type is not None:
                entities.add((current_type, start, i))
            current_type = None
            start = None
    
    if current_type is not None:
        entities.add((current_type, start, len(labels)))
    
    return entities


def evaluate(model, dataloader, device, id2label: dict = None) -> dict:
    """评估模型"""
    model.eval()
    
    total_loss = 0
    ner_correct = 0
    ner_total = 0
    type_correct = 0
    type_total = 0
    
    all_pred_entities = 0
    all_true_entities = 0
    all_correct_entities = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            type_labels = batch['type_labels'].to(device)
            type_entity_mask = batch['type_entity_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                type_labels=type_labels,
                type_entity_mask=type_entity_mask
            )
            
            total_loss += outputs['loss'].item()
            
            ner_preds = model.decode_ner(input_ids, attention_mask)
            
            for i, (pred, label, att_mask) in enumerate(zip(ner_preds, ner_labels, attention_mask)):
                valid_mask = (label != -100)
                pred_tensor = torch.tensor(pred, device=device)
                
                seq_len = len(pred)
                label_seq = label[:seq_len]
                valid_in_seq = valid_mask[:seq_len]
                
                if valid_in_seq.sum() > 0:
                    ner_correct += (pred_tensor[valid_in_seq] == label_seq[valid_in_seq]).sum().item()
                    ner_total += valid_in_seq.sum().item()
                
                if id2label is not None:
                    valid_pred = [pred[j] for j in range(seq_len) if valid_in_seq[j]]
                    valid_label = [label_seq[j].item() for j in range(seq_len) if valid_in_seq[j]]
                    
                    pred_entities = extract_entities(valid_pred, id2label)
                    true_entities = extract_entities(valid_label, id2label)
                    
                    all_pred_entities += len(pred_entities)
                    all_true_entities += len(true_entities)
                    all_correct_entities += len(pred_entities & true_entities)
            
            if 'type_logits' in outputs:
                type_preds = outputs['type_logits'].argmax(dim=-1)
                type_correct += (type_preds == type_labels).sum().item()
                type_total += type_labels.size(0)
    
    n = len(dataloader)
    
    precision = all_correct_entities / all_pred_entities if all_pred_entities > 0 else 0
    recall = all_correct_entities / all_true_entities if all_true_entities > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / n,
        'ner_accuracy': ner_correct / ner_total if ner_total > 0 else 0,
        'ner_precision': precision,
        'ner_recall': recall,
        'ner_f1': f1,
        'type_accuracy': type_correct / type_total if type_total > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='训练词级别多任务BERT模型')
    parser.add_argument('--config', type=str, default='src/bert_ner/config/training.yml',
                        help='配置文件路径')
    parser.add_argument('--data_file', type=str, help='标注平台导出的JSONL文件')
    parser.add_argument('--data_dir', type=str, help='已划分好的数据目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--bert_model', type=str, help='BERT模型')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', type=str, help='设备')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    args = parser.parse_args()
    
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    
    # 命令行参数覆盖
    if args.output_dir:
        config['model']['output_dir'] = args.output_dir
    else:
        # 默认输出到 wordlevel 子目录
        config['model']['output_dir'] = config['model']['output_dir'].replace('multitask', 'multitask_wordlevel')
    
    if args.bert_model:
        config['model']['bert_model'] = args.bert_model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    set_seed(config['seed'])
    device = get_device(config['device'])
    
    output_dir = os.path.join(PROJECT_ROOT, config['model']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"使用设备: {device}")
    logger.info(f"分词方式: 词级别 (offset_mapping)")
    
    # 确定数据目录
    if args.data_file:
        data_file = os.path.join(PROJECT_ROOT, args.data_file) if not os.path.isabs(args.data_file) else args.data_file
        logger.info(f"从JSONL文件划分数据: {data_file}")
        data_dir = prepare_data(data_file, output_dir, config['seed'])
    elif args.data_dir:
        data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    else:
        data_dir = os.path.join(PROJECT_ROOT, config['data']['data_dir'])
    
    logger.info(f"数据目录: {data_dir}")
    
    # 保存配置
    config['tokenization'] = 'word_level'  # 标记分词方式
    with open(os.path.join(output_dir, 'config.yml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    # 标签映射
    ner_labels = config['ner_labels']
    ner_label2id = {label: i for i, label in enumerate(ner_labels)}
    ner_id2label = {i: label for label, i in ner_label2id.items()}
    
    type_labels = config.get('type_labels', ['管子', '管件', '法兰', '螺栓', '阀门', '垫片'])
    type_label2id = {label: i for i, label in enumerate(type_labels)}
    type_id2label = {i: label for label, i in type_label2id.items()}
    
    logger.info(f"NER标签数: {len(ner_label2id)}")
    logger.info(f"TYPE分类数: {len(type_label2id)}")
    
    # 保存标签映射
    label_map = {
        'ner_labels': ner_labels,
        'ner_label2id': ner_label2id,
        'ner_id2label': {str(k): v for k, v in ner_id2label.items()},
        'type_labels': type_labels,
        'type_label2id': type_label2id,
        'type_id2label': {str(k): v for k, v in type_id2label.items()},
        'tokenization': 'word_level'  # 标记分词方式
    }
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    # 【关键改动】加载 tokenizer（不添加 [SPACE] token）
    bert_model = config['model']['bert_model']
    logger.info(f"加载tokenizer: {bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    logger.info(f"词表大小: {len(tokenizer)}")
    logger.info("【词级别分词】不需要添加 [SPACE] 特殊 token")
    
    # 【关键改动】使用词级别数据集
    max_length = config['data']['max_length']
    augment_enabled = config['data'].get('augment', True)
    augment_prob = config['data'].get('augment_prob', 0.3)
    logger.info("加载数据集（词级别）...")
    
    train_dataset = WordLevelMultiTaskDataset(
        os.path.join(data_dir, 'train.jsonl'),
        tokenizer, ner_label2id, type_label2id,
        max_length=max_length,
        augment=augment_enabled,
        augment_prob=augment_prob
    )
    if augment_enabled:
        logger.info(f"✅ 训练集启用数据增强 (概率={augment_prob})")
    
    dev_dataset = WordLevelMultiTaskDataset(
        os.path.join(data_dir, 'dev.jsonl'),
        tokenizer, ner_label2id, type_label2id,
        max_length=max_length
    )
    test_dataset = WordLevelMultiTaskDataset(
        os.path.join(data_dir, 'test.jsonl'),
        tokenizer, ner_label2id, type_label2id,
        max_length=max_length
    )
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(dev_dataset)}, 测试集: {len(test_dataset)}")
    
    # DataLoader
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型（不需要调整 embedding 大小）
    logger.info("创建多任务模型...")
    
    model = create_multitask_model(
        bert_model,
        num_ner_labels=len(ner_label2id),
        num_type_labels=len(type_label2id)
    )
    # 【注意】词级别不需要 resize_token_embeddings，因为没有添加新 token
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    # 优化器和调度器
    epochs = int(config['training']['epochs'])
    lr = float(config['training']['learning_rate'])
    crf_lr_mult = float(config['training'].get('crf_lr_multiplier', 50))
    classifier_lr_mult = float(config['training'].get('classifier_lr_multiplier', 10))
    warmup_ratio = float(config['training']['warmup_ratio'])
    weight_decay = float(config['training']['weight_decay'])
    
    bert_params = []
    crf_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        elif 'crf' in name:
            crf_params.append(param)
        else:
            classifier_params.append(param)
    
    crf_lr = lr * crf_lr_mult
    classifier_lr = lr * classifier_lr_mult
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': crf_params, 'lr': crf_lr, 'weight_decay': 0.0},
        {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': weight_decay}
    ])
    
    logger.info(f"差分学习率: BERT={lr:.2e}, CRF={crf_lr:.2e}, Classifier={classifier_lr:.2e}")
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    if accumulation_steps > 1:
        logger.info(f"启用梯度累积: {accumulation_steps} 步")
    
    use_dice_loss = config['training'].get('use_dice_loss', False)
    dice_weight = config['training'].get('dice_weight', 0.5)
    
    dice_loss_fn = None
    if use_dice_loss:
        dice_loss_fn = DiceLoss()
        logger.info(f"启用 Dice Loss, 权重={dice_weight}")
    
    early_stopping_config = config['training'].get('early_stopping', {})
    patience = early_stopping_config.get('patience', 3)
    min_delta = early_stopping_config.get('min_delta', 0.001)
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_ner_loss': [],
        'train_type_loss': [],
        'train_dice_loss': [],
        'val_loss': [],
        'val_ner_acc': [],
        'val_ner_f1': [],
        'val_type_acc': [],
        'learning_rates': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{epochs}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            dice_loss_fn=dice_loss_fn, dice_weight=dice_weight,
            accumulation_steps=accumulation_steps
        )
        logger.info(f"训练损失: {train_metrics['loss']:.4f}")
        
        dev_metrics = evaluate(model, dev_loader, device, id2label=ner_id2label)
        logger.info(f"验证损失: {dev_metrics['loss']:.4f}")
        logger.info(f"  NER Token准确率: {dev_metrics['ner_accuracy']*100:.2f}%")
        logger.info(f"  NER 实体级 F1: {dev_metrics['ner_f1']*100:.2f}%")
        logger.info(f"  TYPE准确率: {dev_metrics['type_accuracy']*100:.2f}%")
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_ner_loss'].append(train_metrics['ner_loss'])
        history['train_type_loss'].append(train_metrics['type_loss'])
        history['train_dice_loss'].append(train_metrics.get('dice_loss', 0))
        history['val_loss'].append(dev_metrics['loss'])
        history['val_ner_acc'].append(dev_metrics['ner_accuracy'])
        history['val_ner_f1'].append(dev_metrics['ner_f1'])
        history['val_type_acc'].append(dev_metrics['type_accuracy'])
        history['learning_rates'].append(train_metrics['learning_rate'])
        
        avg_acc = dev_metrics['ner_f1'] * 0.7 + dev_metrics['type_accuracy'] * 0.3
        
        improved = avg_acc > history['best_val_acc'] + min_delta
        if improved:
            history['best_val_acc'] = avg_acc
            history['best_epoch'] = epoch
            patience_counter = 0
            
            model_path = os.path.join(output_dir, 'best_model')
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            with open(os.path.join(model_path, 'ner_labels.json'), 'w', encoding='utf-8') as f:
                json.dump({'label2id': ner_label2id, 'id2label': ner_id2label}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(model_path, 'type_labels.json'), 'w', encoding='utf-8') as f:
                json.dump({'label2id': type_label2id, 'id2label': type_id2label}, f, ensure_ascii=False, indent=2)
            
            # 【重要】保存分词方式标记
            with open(os.path.join(model_path, 'tokenization_info.json'), 'w', encoding='utf-8') as f:
                json.dump({'method': 'word_level', 'use_offset_mapping': True}, f)
            
            logger.info(f"✅ 保存最佳模型 (epoch {epoch}, avg_acc={avg_acc*100:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"  早停计数: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f"\n验证集性能连续 {patience} 轮未提升，触发早停")
                break
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    plot_training_curves(history, output_dir)
    
    # 测试集评估
    logger.info(f"\n{'='*60}")
    logger.info("在测试集上评估最佳模型...")
    
    best_model_path = os.path.join(output_dir, 'best_model')
    best_config = AutoConfig.from_pretrained(best_model_path)
    best_config.num_labels = len(ner_label2id)
    
    model = BertMultiTaskModel.from_pretrained(
        best_model_path,
        config=best_config,
        num_type_labels=len(type_label2id)
    )
    model.to(device)
    
    test_metrics = evaluate(model, test_loader, device, id2label=ner_id2label)
    
    logger.info(f"测试集 NER 实体级 F1: {test_metrics['ner_f1']*100:.2f}%")
    
    print_training_summary(history, test_metrics, output_dir, config)
    
    logger.info(f"\n✅ 词级别训练完成！所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()