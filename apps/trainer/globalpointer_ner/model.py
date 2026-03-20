#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GlobalPointer 模型实现

GlobalPointer 通过预测实体的起始和结束位置来进行 NER，
相比 BIO+CRF 更加鲁棒，不易出现边界截断问题。

参考论文: https://arxiv.org/abs/2208.03054
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple, List, Dict


class SinusoidalPositionEncoding(nn.Module):
    """
    旋转位置编码 (RoPE)
    用于在 GlobalPointer 中引入相对位置信息
    """
    
    def __init__(self, max_len: int, hidden_size: int):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * 
                            (-torch.log(torch.tensor(10000.0)) / hidden_size))
        
        sinusoid = torch.zeros(max_len, hidden_size)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('sinusoid', sinusoid)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        return self.sinusoid[:seq_len]


class GlobalPointer(nn.Module):
    """
    GlobalPointer 模型
    
    对于每种实体类型，预测所有可能的 (start, end) 位置对的得分
    """
    
    def __init__(
        self,
        encoder_path: str,
        num_labels: int,
        head_size: int = 64,
        max_len: int = 512,
        rope: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            encoder_path: 预训练模型路径
            num_labels: 实体类型数量
            head_size: 每个 head 的维度
            max_len: 最大序列长度
            rope: 是否使用旋转位置编码
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.head_size = head_size
        self.rope = rope
        
        # 加载预训练编码器
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.hidden_size = self.encoder.config.hidden_size
        
        # GlobalPointer 头
        # 对于每种实体类型，需要 2 * head_size 维度（分别用于 start 和 end）
        self.dense = nn.Linear(self.hidden_size, num_labels * head_size * 2)
        self.dropout = nn.Dropout(dropout)
        
        # 旋转位置编码
        if rope:
            self.position_encoding = SinusoidalPositionEncoding(max_len, head_size)
    
    def apply_rotary_position_embeddings(
        self, 
        qw: torch.Tensor, 
        kw: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码
        
        Args:
            qw: query 向量 [batch, num_labels, seq_len, head_size]
            kw: key 向量 [batch, num_labels, seq_len, head_size]
            seq_len: 序列长度
            
        Returns:
            应用位置编码后的 qw, kw
        """
        # 获取位置编码
        pos_emb = self.position_encoding(seq_len)  # [seq_len, head_size]
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # [seq_len, head_size]
        sin_pos = pos_emb[..., 0::2].repeat_interleave(2, dim=-1)  # [seq_len, head_size]
        
        # 扩展维度以匹配 qw, kw
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_size]
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_size]
        
        # 应用旋转
        # q2, k2 是交替取负后的版本
        qw2 = torch.stack([-qw[..., 1::2], qw[..., 0::2]], dim=-1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        
        kw2 = torch.stack([-kw[..., 1::2], kw[..., 0::2]], dim=-1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        
        return qw, kw
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            token_type_ids: token 类型 IDs（可选）
            labels: 标签 [batch, num_labels, seq_len, seq_len]（训练时）
            
        Returns:
            包含 logits 和可选 loss 的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 编码器输出
        encoder_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            encoder_kwargs['token_type_ids'] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)
        
        # 投影到 GlobalPointer 空间
        # [batch, seq_len, num_labels * head_size * 2]
        logits = self.dense(hidden_states)
        
        # 重塑为 [batch, seq_len, num_labels, head_size, 2]
        logits = logits.view(batch_size, seq_len, self.num_labels, self.head_size, 2)
        
        # 分离 query (start) 和 key (end) 向量
        # qw: [batch, seq_len, num_labels, head_size] -> [batch, num_labels, seq_len, head_size]
        qw = logits[..., 0].permute(0, 2, 1, 3)
        kw = logits[..., 1].permute(0, 2, 1, 3)
        
        # 应用旋转位置编码
        if self.rope:
            qw, kw = self.apply_rotary_position_embeddings(qw, kw, seq_len)
        
        # 计算 span 得分: [batch, num_labels, seq_len, seq_len]
        # qw: [batch, num_labels, seq_len, head_size]
        # kw: [batch, num_labels, seq_len, head_size]
        # logits[b, l, i, j] 表示第 b 个样本、第 l 类实体、(start=i, end=j) 的得分
        logits = torch.einsum('blid,bljd->blij', qw, kw) / (self.head_size ** 0.5)
        
        # 创建上三角掩码（start <= end，即 i <= j）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device, dtype=logits.dtype), diagonal=0)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # attention_mask 扩展: 有效位置为 1，padding 位置为 0
        # attn_mask[b, 1, i, j] = attention_mask[b, i] * attention_mask[b, j]
        attn_mask_2d = attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(-2)  # [batch, seq_len, seq_len]
        attn_mask_2d = attn_mask_2d.unsqueeze(1).to(logits.dtype)  # [batch, 1, seq_len, seq_len]
        
        # 组合掩码
        combined_mask = mask * attn_mask_2d
        
        # 应用掩码（将无效位置设为很小的负数）
        logits = logits * combined_mask - (1 - combined_mask) * 1e12
        
        result = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            loss = self.compute_loss(logits, labels, combined_mask)
            result['loss'] = loss
        
        return result
    
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多标签分类损失
        
        使用 GlobalPointer 论文中的多标签分类损失：
        对于每个实体类型，将 span 预测看作多标签分类问题
        """
        batch_size, num_labels, seq_len, _ = logits.shape
        
        # 展平后两个维度
        logits = logits.view(batch_size, num_labels, -1)  # [batch, num_labels, seq_len*seq_len]
        labels = labels.view(batch_size, num_labels, -1)  # [batch, num_labels, seq_len*seq_len]
        mask = mask.view(batch_size, 1, -1).expand_as(labels)  # [batch, num_labels, seq_len*seq_len]
        
        # 将无效位置的 logits 设为很小的值（但不要太极端）
        logits = logits * mask + (1 - mask) * (-1e4)
        
        # 多标签交叉熵损失
        # 使用 sigmoid + binary cross entropy
        # 但为了处理类别不平衡，使用 focal loss 思想
        
        # 方法：使用 softmax 版本的多标签损失
        # 参考 GlobalPointer 原始实现
        
        # 正例位置
        y_true = labels.float()
        
        # 计算损失：log(1 + sum(exp(负例得分))) + log(1 + sum(exp(-正例得分)))
        # 等价于：让正例得分 > 0，负例得分 < 0
        
        # 正例的得分应该大于 0
        pos_logits = logits * y_true  # 只保留正例位置的得分
        # 负例的得分应该小于 0  
        neg_logits = logits * (1 - y_true) * mask  # 只保留负例位置的得分
        
        # 使用 logsumexp 计算损失（数值稳定版本）
        # log(1 + sum(exp(neg))) = logsumexp([0, neg_1, neg_2, ...])
        # log(1 + sum(exp(-pos))) = logsumexp([0, -pos_1, -pos_2, ...])
        
        # 添加 0 基准
        zeros = torch.zeros(batch_size, num_labels, 1, device=logits.device, dtype=logits.dtype)
        
        # 负例部分：将正例位置设为很小的值，然后 logsumexp
        neg_for_loss = torch.where(y_true > 0, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype), logits)
        neg_for_loss = torch.where(mask > 0, neg_for_loss, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))
        neg_for_loss = torch.cat([zeros, neg_for_loss], dim=-1)
        neg_loss = torch.logsumexp(neg_for_loss, dim=-1)
        
        # 正例部分：将负例位置设为很小的值，然后 logsumexp(-x)
        pos_for_loss = torch.where(y_true > 0, -logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))
        pos_for_loss = torch.cat([zeros, pos_for_loss], dim=-1)
        pos_loss = torch.logsumexp(pos_for_loss, dim=-1)
        
        loss = (neg_loss + pos_loss).mean()
        
        return loss


class GlobalPointerForNER(nn.Module):
    """
    用于 NER 任务的 GlobalPointer 封装
    """
    
    def __init__(
        self,
        encoder_path: str,
        label2id: Dict[str, int],
        head_size: int = 64,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.num_labels = len(label2id)
        
        self.model = GlobalPointer(
            encoder_path=encoder_path,
            num_labels=self.num_labels,
            head_size=head_size,
            max_len=max_len,
            dropout=dropout
        )
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def decode(
        self, 
        logits: torch.Tensor, 
        attention_mask: torch.Tensor,
        threshold: float = 0.0
    ) -> List[List[Dict]]:
        """
        解码预测结果
        
        Args:
            logits: 模型输出 [batch, num_labels, seq_len, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            threshold: 阈值
            
        Returns:
            每个样本的实体列表
        """
        batch_size = logits.shape[0]
        results = []
        
        for i in range(batch_size):
            entities = []
            seq_len = attention_mask[i].sum().item()
            
            for label_id in range(self.num_labels):
                # 获取该类型的 span 得分
                scores = logits[i, label_id, :seq_len, :seq_len]
                
                # 找出得分大于阈值的 span
                indices = torch.where(scores > threshold)
                
                for start, end in zip(indices[0].tolist(), indices[1].tolist()):
                    if start <= end:  # 有效 span
                        entities.append({
                            'start': start,
                            'end': end + 1,  # 转为左闭右开
                            'type': self.id2label[label_id],
                            'score': scores[start, end].item()
                        })
            
            # 按得分排序
            entities = sorted(entities, key=lambda x: -x['score'])
            results.append(entities)
        
        return results


def create_model(
    encoder_path: str,
    labels: List[str],
    head_size: int = 64,
    max_len: int = 512,
    dropout: float = 0.1
) -> GlobalPointerForNER:
    """
    创建 GlobalPointer NER 模型
    
    Args:
        encoder_path: 预训练编码器路径
        labels: 标签列表
        head_size: GlobalPointer head 维度
        max_len: 最大序列长度
        dropout: Dropout 比例
        
    Returns:
        GlobalPointerForNER 模型
    """
    label2id = {label: i for i, label in enumerate(labels)}
    
    model = GlobalPointerForNER(
        encoder_path=encoder_path,
        label2id=label2id,
        head_size=head_size,
        max_len=max_len,
        dropout=dropout
    )
    
    return model
