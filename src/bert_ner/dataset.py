"""
NER数据集处理模块
"""

import os
import random
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset


def load_bio_data(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """
    加载BIO格式数据
    
    Args:
        file_path: BIO文件路径
        
    Returns:
        [(chars, tags), ...] 列表，每个元素是一个句子
    """
    sentences = []
    chars, tags = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行表示句子结束
                if chars:
                    sentences.append((chars, tags))
                    chars, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    chars.append(parts[0])
                    tags.append(parts[1])
                elif len(parts) == 1:
                    chars.append(parts[0])
                    tags.append('O')
    
    # 处理最后一个句子
    if chars:
        sentences.append((chars, tags))
    
    return sentences


def save_bio_data(sentences: List[Tuple[List[str], List[str]]], file_path: str):
    """
    保存BIO格式数据
    
    Args:
        sentences: [(chars, tags), ...] 列表
        file_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for chars, tags in sentences:
            for char, tag in zip(chars, tags):
                f.write(f"{char} {tag}\n")
            f.write("\n")


def split_data(
    sentences: List[Tuple[List[str], List[str]]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True
) -> Tuple[List, List, List]:
    """
    划分数据集
    
    Args:
        sentences: 句子列表
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        shuffle: 是否打乱
        
    Returns:
        (train_data, dev_data, test_data)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-5
    
    if shuffle:
        random.seed(seed)
        sentences = sentences.copy()
        random.shuffle(sentences)
    
    n = len(sentences)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    
    train_data = sentences[:train_end]
    dev_data = sentences[train_end:dev_end]
    test_data = sentences[dev_end:]
    
    return train_data, dev_data, test_data


class NERDataset(Dataset):
    """
    NER数据集类，用于PyTorch DataLoader
    """
    
    def __init__(
        self,
        sentences: List[Tuple[List[str], List[str]]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128
    ):
        """
        Args:
            sentences: [(chars, tags), ...] 列表
            tokenizer: BERT tokenizer
            label2id: 标签到ID的映射
            max_length: 最大序列长度
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        chars, tags = self.sentences[idx]
        
        # 转换为BERT输入格式
        # 对于中文，每个字符就是一个token
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,  # 输入已经是分词后的
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签
        # 需要考虑[CLS]和[SEP]以及padding
        label_ids = []
        word_ids = encoding.word_ids()  # 获取每个token对应的原始word索引
        
        for word_id in word_ids:
            if word_id is None:
                # [CLS], [SEP], [PAD] 使用-100（PyTorch会忽略）
                label_ids.append(-100)
            else:
                # 映射标签
                tag = tags[word_id] if word_id < len(tags) else 'O'
                label_ids.append(self.label2id.get(tag, self.label2id['O']))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def create_dataloader(
    dataset: NERDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
):
    """创建DataLoader"""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

