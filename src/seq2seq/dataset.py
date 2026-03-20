# ============================================
# Seq2Seq TYPE编码数据集
# ============================================

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset


def preprocess_text(text: str) -> str:
    """预处理输入文本：替换符号为空格"""
    processed = re.sub(r'[|/\\,;]', ' ', text)
    processed = re.sub(r'\s+', ' ', processed).strip()
    return processed


class TypeEncoderDataset(Dataset):
    """TYPE编码生成数据集"""
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_input_length: int = 64,
        max_output_length: int = 16,
        preprocess: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.preprocess = preprocess
        
        # 加载数据
        self.samples = self._load_data(data_file)
    
    def _load_data(self, data_file: str) -> List[Dict]:
        """加载JSONL数据"""
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    samples.append(sample)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 获取输入和输出
        input_text = sample['input']
        output_text = sample['output']
        
        # 预处理输入
        if self.preprocess:
            input_text = preprocess_text(input_text)
        
        # 编码输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码输出（作为labels）
        with self.tokenizer.as_target_tokenizer():
            output_encoding = self.tokenizer(
                output_text,
                max_length=self.max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # 将padding token替换为-100（忽略loss计算）
        labels = output_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def create_dataloaders(
    train_file: str,
    tokenizer,
    batch_size: int = 16,
    val_file: Optional[str] = None,
    val_split: float = 0.1,
    max_input_length: int = 64,
    max_output_length: int = 16,
    seed: int = 42
):
    """创建训练和验证数据加载器"""
    from torch.utils.data import DataLoader, random_split
    
    # 加载训练数据
    train_dataset = TypeEncoderDataset(
        train_file,
        tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length
    )
    
    # 验证集
    if val_file and Path(val_file).exists():
        val_dataset = TypeEncoderDataset(
            val_file,
            tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length
        )
    else:
        # 从训练集划分验证集
        torch.manual_seed(seed)
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size]
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
