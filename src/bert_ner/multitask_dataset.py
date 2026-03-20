"""
多任务学习数据集

支持同时加载NER标签和TYPE分类标签
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from transformers import BertTokenizerFast


class MultiTaskDataset(Dataset):
    """
    多任务数据集
    
    数据格式 (JSONL):
    {
        "text": "管子;A182 F321;GB/T3087;DN15",
        "ner_labels": ["B-TYPE", "I-TYPE", "O", ...],
        "type_class": "管子"  # 大类：管子/管件/阀门/垫片/螺栓
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizerFast,
        ner_label2id: Dict[str, int],
        type_label2id: Dict[str, int],
        max_length: int = 128,
        augment: bool = False,
        augment_prob: float = 0.3
    ):
        """
        Args:
            data_path: JSONL数据文件路径
            tokenizer: BERT tokenizer
            ner_label2id: NER标签到ID的映射
            type_label2id: TYPE分类标签到ID的映射
            max_length: 最大序列长度
            augment: 是否启用数据增强（训练时启用，验证/测试时关闭）
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
        type_class = sample.get('type_class', '')  # 大类标签
        type_entity = sample.get('type_entity', '')  # TYPE实体文本（可选，用于定位）
        
        # 注意：全角罗马数字转半角、小写化已在 prepare_data 时完成
        # 训练数据已经是预处理后的格式
        
        # ========== 数据增强：随机移除空格 ==========
        # 让模型对有无空格的输入都能正确识别
        if self.augment and self.random.random() < self.augment_prob:
            new_text = []
            new_labels = []
            for i, (char, label) in enumerate(zip(text, ner_labels)):
                # 70%概率移除空格（只移除空格，不移除其他字符）
                if char.isspace() and self.random.random() < 0.7:
                    continue  # 跳过这个空格及其标签
                new_text.append(char)
                new_labels.append(label)
            text = ''.join(new_text)
            ner_labels = new_labels
        # ============================================
        
        # 按字符分词（与预测时保持一致！）
        # 将空格替换为 [SPACE]，这样 tokenizer 不会跳过空格
        chars = []
        for char in text:
            if char.isspace():
                chars.append('[SPACE]')  # 用特殊 token 替代空格
            else:
                chars.append(char)
        
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,  # 关键：按字符分词
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        word_ids = encoding.word_ids()  # 获取每个 token 对应的字符索引
        
        # 对齐NER标签到tokenized序列
        aligned_ner_labels = self._align_labels_by_word_ids(ner_labels, word_ids)
        
        # 创建TYPE实体位置掩码（可选）
        type_entity_mask = self._create_entity_mask_by_word_ids(text, type_entity, word_ids)
        
        # TYPE分类标签
        type_label = self.type_label2id.get(type_class, 0) if type_class else 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ner_labels': torch.tensor(aligned_ner_labels, dtype=torch.long),
            'type_labels': torch.tensor(type_label, dtype=torch.long),
            'type_entity_mask': torch.tensor(type_entity_mask, dtype=torch.float),
        }
    
    def _align_labels_by_word_ids(
        self,
        char_labels: List[str],
        word_ids: List[Optional[int]]
    ) -> List[int]:
        """
        使用 word_ids 将字符级标签对齐到 token 级
        
        Args:
            char_labels: 字符级 BIO 标签
            word_ids: 每个 token 对应的字符索引（None 表示特殊 token）
        
        Returns:
            token 级标签 ID 列表
        """
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:  # 特殊 token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_id < len(char_labels):
                label = char_labels[word_id]
                aligned_labels.append(self.ner_label2id.get(label, self.ner_label2id.get('O', 0)))
            else:
                aligned_labels.append(-100)
        
        return aligned_labels
    
    def _create_entity_mask_by_word_ids(
        self,
        text: str,
        entity: str,
        word_ids: List[Optional[int]]
    ) -> List[int]:
        """
        使用 word_ids 创建实体位置掩码
        
        Args:
            text: 原始文本
            entity: 实体文本
            word_ids: 每个 token 对应的字符索引
        
        Returns:
            掩码列表，1 表示实体位置
        """
        mask = [0] * len(word_ids)
        
        if not entity:
            return mask
        
        # 找到实体在文本中的位置
        entity_start = text.find(entity)
        if entity_start == -1:
            return mask
        
        entity_end = entity_start + len(entity)
        
        # 标记对应的 token 位置
        for i, word_id in enumerate(word_ids):
            if word_id is None:  # 特殊 token
                continue
            # 如果字符在实体范围内
            if entity_start <= word_id < entity_end:
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
