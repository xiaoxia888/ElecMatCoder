#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER 损失函数

包含多种处理类别不平衡的损失函数：
- Dice Loss: 缓解类别不平衡
- Focal Loss: 关注难分类样本
- Label Smoothing: 防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss 用于处理类别不平衡
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    对于 NER 任务，可以缓解 O 标签过多的问题
    
    注意：使用 scatter 替代 one_hot，更省内存
    """
    
    def __init__(self, smooth: float = 1.0, square_denominator: bool = True):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_labels]
            labels: [batch, seq_len]
            mask: [batch, seq_len] 有效位置为 True/1
            
        Returns:
            Dice loss
        """
        num_labels = logits.shape[-1]
        batch_size, seq_len = labels.shape
        
        # 处理 -100 标签（忽略）
        labels_clean = labels.clone()
        invalid_mask = (labels == -100)
        labels_clean[invalid_mask] = 0
        
        # 有效位置 mask
        valid_mask = ~invalid_mask
        if mask is not None:
            valid_mask = valid_mask & mask.bool()
        
        # Softmax 获取概率
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, num_labels]
        
        # 使用 gather 获取每个位置真实标签的概率（避免 one-hot）
        # [batch, seq_len, 1]
        true_probs = probs.gather(dim=-1, index=labels_clean.unsqueeze(-1))
        
        # 计算每个类别的 Dice 系数
        # 使用 scatter_add 累计每个类别的概率和
        dice_loss = 0.0
        
        for c in range(num_labels):
            # 该类别的位置
            class_mask = (labels_clean == c) & valid_mask
            
            if class_mask.sum() == 0:
                continue
            
            # 该类别的预测概率
            p_c = probs[:, :, c]  # [batch, seq_len]
            
            # 交集：预测为类别c且真实为类别c的概率
            intersection = (p_c * class_mask.float()).sum()
            
            # 并集
            if self.square_denominator:
                union = (p_c ** 2).sum() + class_mask.float().sum()
            else:
                union = p_c.sum() + class_mask.float().sum()
            
            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice_c)
        
        return dice_loss / num_labels


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理难分类样本
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    当 gamma > 0 时，减少易分类样本的权重，使模型更关注难分类样本
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_labels]
            labels: [batch, seq_len]
            mask: [batch, seq_len]
            
        Returns:
            Focal loss
        """
        num_labels = logits.shape[-1]
        
        # 处理 -100 标签
        valid_mask = (labels != -100)
        if mask is not None:
            valid_mask = valid_mask & mask.bool()
        
        labels_clean = labels.clone()
        labels_clean[~valid_mask] = 0
        
        # 计算 softmax 概率
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, num_labels]
        
        # 获取真实类别的概率
        # [batch, seq_len]
        p_t = probs.gather(dim=-1, index=labels_clean.unsqueeze(-1)).squeeze(-1)
        
        # 计算 focal 权重
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算 cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, num_labels), 
            labels_clean.view(-1), 
            reduction='none'
        ).view_as(labels_clean)
        
        # 应用 focal 权重
        loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels_clean]
            loss = alpha_t * loss
        
        # 只计算有效位置的损失
        loss = loss * valid_mask.float()
        
        if self.reduction == 'mean':
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失
    
    将 hard label 转换为 soft label，防止模型过度自信
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_labels]
            labels: [batch, seq_len]
            mask: [batch, seq_len]
        """
        num_labels = logits.shape[-1]
        
        # 处理 -100 标签
        valid_mask = (labels != -100)
        if mask is not None:
            valid_mask = valid_mask & mask.bool()
        
        labels_clean = labels.clone()
        labels_clean[~valid_mask] = 0
        
        # 创建 soft labels
        # [batch, seq_len, num_labels]
        soft_labels = torch.full_like(logits, self.smoothing / (num_labels - 1))
        soft_labels.scatter_(-1, labels_clean.unsqueeze(-1), 1.0 - self.smoothing)
        
        # 计算 KL 散度损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1)
        
        # 只计算有效位置
        loss = loss * valid_mask.float()
        
        if self.reduction == 'mean':
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合 CRF Loss + Dice Loss + Focal Loss
    """
    
    def __init__(
        self,
        crf_weight: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.crf_weight = crf_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        if dice_weight > 0:
            self.dice_loss = DiceLoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        
        if label_smoothing > 0:
            self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        else:
            self.label_smoothing_loss = None
        
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        crf_loss: torch.Tensor,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            crf_loss: CRF 层计算的损失
            emissions: 分类器输出 [batch, seq_len, num_labels]
            labels: 标签 [batch, seq_len]
            mask: 掩码 [batch, seq_len]
            
        Returns:
            组合损失
        """
        total_loss = self.crf_weight * crf_loss
        
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(emissions, labels, mask)
            total_loss = total_loss + self.dice_weight * dice_loss
        
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(emissions, labels, mask)
            total_loss = total_loss + self.focal_weight * focal_loss
        
        if self.label_smoothing > 0 and self.label_smoothing_loss is not None:
            ls_loss = self.label_smoothing_loss(emissions, labels, mask)
            total_loss = total_loss + 0.1 * ls_loss
        
        return total_loss


def compute_class_weights(label_counts: dict, num_labels: int, method: str = 'inverse') -> torch.Tensor:
    """
    计算类别权重
    
    Args:
        label_counts: 标签计数字典 {label_id: count}
        num_labels: 标签总数
        method: 权重计算方法
            - 'inverse': 逆频率
            - 'sqrt_inverse': 逆频率的平方根
            - 'effective': 有效样本数方法
            
    Returns:
        类别权重张量 [num_labels]
    """
    weights = torch.ones(num_labels)
    
    total = sum(label_counts.values())
    
    for label_id, count in label_counts.items():
        if label_id < num_labels and count > 0:
            if method == 'inverse':
                weights[label_id] = total / (num_labels * count)
            elif method == 'sqrt_inverse':
                weights[label_id] = (total / (num_labels * count)) ** 0.5
            elif method == 'effective':
                beta = 0.9999
                effective_num = 1.0 - beta ** count
                weights[label_id] = (1.0 - beta) / effective_num
    
    # 归一化
    weights = weights / weights.sum() * num_labels
    
    return weights
