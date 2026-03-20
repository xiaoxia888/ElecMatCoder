"""
词级别多任务学习数据集

关键改动：使用 offset_mapping 将字符级别标签动态对齐到词级别 tokens
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedTokenizerFast


class WordLevelMultiTaskDataset(Dataset):
    """
    词级别多任务数据集
    
    数据格式 (JSONL) - 与原格式完全相同:
    {
        "text": "管子;A182 F321;GB/T3087;DN15",
        "ner_labels": ["B-TYPE", "I-TYPE", "O", ...],  # 字符级别标签
        "type_class": "管子"
    }
    
    关键区别：
    - 原方法：将文本拆成字符列表，用 is_split_into_words=True
    - 新方法：直接传入文本，用 offset_mapping 对齐标签
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        ner_label2id: Dict[str, int],
        type_label2id: Dict[str, int],
        max_length: int = 128,
        augment: bool = False,
        augment_prob: float = 0.3
    ):
        """
        Args:
            data_path: JSONL数据文件路径
            tokenizer: tokenizer（需要支持 return_offsets_mapping）
            ner_label2id: NER标签到ID的映射
            type_label2id: TYPE分类标签到ID的映射
            max_length: 最大序列长度
            augment: 是否启用数据增强
            augment_prob: 数据增强概率
        """
        import random
        self.random = random
        
        self.tokenizer = tokenizer
        self.ner_label2id = ner_label2id
        self.type_label2id = type_label2id
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        
        # 加载数据
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        text = sample['text']
        ner_labels = sample['ner_labels'].copy()  # 复制以避免修改原数据
        type_class = sample.get('type_class', '')
        type_entity = sample.get('type_entity', '')
        
        # ========== 数据增强：随机移除空格 ==========
        if self.augment and self.random.random() < self.augment_prob:
            new_text = []
            new_labels = []
            for i, (char, label) in enumerate(zip(text, ner_labels)):
                if char.isspace() and self.random.random() < 0.7:
                    continue
                new_text.append(char)
                new_labels.append(label)
            text = ''.join(new_text)
            ner_labels = new_labels
        # ============================================
        
        # 【关键改动】词级别分词 + offset_mapping
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # 关键：获取每个token对应的原文位置
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()  # [(start, end), ...]
        
        # 【关键改动】使用 offset_mapping 对齐标签
        aligned_ner_labels = self._align_labels_by_offset(ner_labels, offset_mapping)
        
        # 创建TYPE实体位置掩码
        type_entity_mask = self._create_entity_mask_by_offset(text, type_entity, offset_mapping)
        
        # TYPE分类标签
        type_label = self.type_label2id.get(type_class, 0) if type_class else 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': torch.tensor(aligned_ner_labels, dtype=torch.long),
            'type_labels': torch.tensor(type_label, dtype=torch.long),
            'type_entity_mask': torch.tensor(type_entity_mask, dtype=torch.float),
        }
    
    def _align_labels_by_offset(
        self,
        char_labels: List[str],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        """
        使用 offset_mapping 将字符级标签对齐到 token 级
        
        策略：每个 token 取其覆盖的第一个字符的标签
        
        Args:
            char_labels: 字符级 BIO 标签列表
            offset_mapping: 每个 token 对应的 (start, end) 位置
        
        Returns:
            token 级标签 ID 列表
        """
        aligned_labels = []
        
        for start, end in offset_mapping:
            if start == end:
                # 特殊 token（如 <s>, </s>, <pad>）
                aligned_labels.append(-100)
            elif start < len(char_labels):
                # 取该 token 覆盖的第一个字符的标签
                label = char_labels[start]
                aligned_labels.append(self.ner_label2id.get(label, self.ner_label2id.get('O', 0)))
            else:
                # 超出范围，忽略
                aligned_labels.append(-100)
        
        return aligned_labels
    
    def _create_entity_mask_by_offset(
        self,
        text: str,
        entity: str,
        offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        """
        使用 offset_mapping 创建实体位置掩码
        
        Args:
            text: 原始文本
            entity: 实体文本
            offset_mapping: 每个 token 的位置映射
        
        Returns:
            掩码列表，1 表示实体位置
        """
        mask = [0] * len(offset_mapping)
        
        if not entity:
            return mask
        
        # 找到实体在文本中的位置
        entity_start = text.find(entity)
        if entity_start == -1:
            return mask
        
        entity_end = entity_start + len(entity)
        
        # 标记与实体有交集的 token
        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # 特殊 token
                continue
            # 检查 token 是否与实体区间有交集
            if start < entity_end and end > entity_start:
                mask[i] = 1
        
        return mask


def load_label_maps(label_map_path: str) -> tuple:
    """
    加载标签映射
    
    Returns:
        (type_label2id, type_id2label)
    """
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    type_labels = label_map.get('type_labels', [])
    
    type_label2id = {label: i for i, label in enumerate(type_labels)}
    type_id2label = {i: label for label, i in type_label2id.items()}
    
    return type_label2id, type_id2label
