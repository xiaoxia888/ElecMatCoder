"""
多任务BERT模型：NER + TYPE分类

结构：
    共享编码器 (BERT/XLM-RoBERTa/其他)
        ├── NER头 (CRF): 序列标注，提取实体边界
        └── TYPE分类头: 判断材料大类（管子/管件/阀门/垫片/螺栓）
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from torchcrf import CRF
from typing import Dict, List, Optional, Tuple


class MultiTaskModel(PreTrainedModel):
    """
    多任务模型（支持BERT、XLM-RoBERTa等）
    
    同时完成：
    1. NER: 序列标注（Encoder + CRF）
    2. TYPE分类: 判断材料属于哪个大类（管子/管件/阀门/垫片/螺栓）
    """
    
    # 支持自动配置
    config_class = AutoConfig
    base_model_prefix = "encoder"
    
    def __init__(self, config, num_type_labels: int = 0):
        super().__init__(config)
        
        # 保存配置
        self.num_ner_labels = config.num_labels  # NER标签数量 (BIO标签)
        self.num_type_labels = num_type_labels    # TYPE分类标签数量（5类）
        config.num_type_labels = num_type_labels
        
        # 共享的编码器（自动适配 BERT/XLM-RoBERTa 等）
        self.encoder = AutoModel.from_config(config, add_pooling_layer=True)
        
        # dropout 比例（不同模型配置名可能不同）
        dropout_prob = getattr(config, 'hidden_dropout_prob', 
                              getattr(config, 'dropout', 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        
        # NER分类器：单层线性层
        self.ner_classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # CRF层
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        # TYPE分类头（判断大类：管子/管件/阀门/垫片/螺栓）
        if num_type_labels > 0:
            self.type_classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(config.hidden_size, num_type_labels)
            )
        else:
            self.type_classifier = None
        
        # 初始化权重
        self.post_init()
        
        # 手动初始化 CRF 参数
        self._init_crf_weights()
    
    def _init_crf_weights(self):
        """初始化 CRF 转移矩阵（零初始化）"""
        nn.init.zeros_(self.crf.transitions)
        nn.init.zeros_(self.crf.start_transitions)
        nn.init.zeros_(self.crf.end_transitions)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        ner_labels: Optional[torch.Tensor] = None,
        type_labels: Optional[torch.Tensor] = None,
        type_entity_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            ner_labels: NER标签 [batch_size, seq_len]
            type_labels: TYPE分类标签 [batch_size]
            type_entity_mask: TYPE实体位置掩码 [batch_size, seq_len]
        
        Returns:
            包含loss和logits的字典
        """
        # 编码器前向传播（XLM-RoBERTa 不使用 token_type_ids）
        encoder_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        # 只有 BERT 类模型才使用 token_type_ids
        if token_type_ids is not None and hasattr(self.config, 'type_vocab_size') and self.config.type_vocab_size > 0:
            encoder_kwargs['token_type_ids'] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        result = {}
        total_loss = 0.0
        
        # ========== NER任务 ==========
        ner_emissions = self.ner_classifier(sequence_output)
        result['ner_emissions'] = ner_emissions
        
        # CRF mask
        mask = attention_mask.bool()
        
        if ner_labels is not None:
            # 将 -100 替换为有效标签（0=O）
            labels_for_crf = ner_labels.clone()
            labels_for_crf[ner_labels == -100] = 0
            
            ner_loss = -self.crf(ner_emissions, labels_for_crf, mask=mask, reduction='mean')
            result['ner_loss'] = ner_loss
            total_loss += ner_loss
        else:
            # 解码NER预测
            ner_predictions = self.crf.decode(ner_emissions, mask=mask)
            result['ner_predictions'] = ner_predictions
        
        # ========== TYPE分类任务 ==========
        if self.type_classifier is not None:
            if type_entity_mask is not None:
                type_repr = self._masked_mean_pooling(sequence_output, type_entity_mask)
            else:
                type_repr = pooled_output
            
            type_logits = self.type_classifier(type_repr)
            result['type_logits'] = type_logits
            
            if type_labels is not None:
                type_loss = nn.CrossEntropyLoss()(type_logits, type_labels)
                result['type_loss'] = type_loss
                total_loss += type_loss
        
        # 总损失
        if ner_labels is not None or type_labels is not None:
            result['loss'] = total_loss
        
        return result
    
    def _masked_mean_pooling(
        self,
        sequence_output: torch.Tensor,
        entity_mask: torch.Tensor
    ) -> torch.Tensor:
        """对实体位置进行平均池化"""
        mask_expanded = entity_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
    
    def decode_ner(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """只进行NER解码"""
        with torch.no_grad():
            encoder_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if token_type_ids is not None and hasattr(self.config, 'type_vocab_size') and self.config.type_vocab_size > 0:
                encoder_kwargs['token_type_ids'] = token_type_ids
            
            outputs = self.encoder(**encoder_kwargs)
            sequence_output = outputs.last_hidden_state
            emissions = self.ner_classifier(sequence_output)
            
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
        return predictions
    
    def decode_ner_with_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        o_bias: float = 0.0
    ) -> tuple:
        """
        NER解码并返回置信度
        
        Args:
            o_bias: O标签的偏置值，正值使模型更倾向于预测O
        
        Returns:
            predictions: 预测的标签ID列表
            confidences: 每个token的置信度列表
        """
        with torch.no_grad():
            encoder_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if token_type_ids is not None and hasattr(self.config, 'type_vocab_size') and self.config.type_vocab_size > 0:
                encoder_kwargs['token_type_ids'] = token_type_ids
            
            outputs = self.encoder(**encoder_kwargs)
            sequence_output = outputs.last_hidden_state
            emissions = self.ner_classifier(sequence_output)
            
            # 给O标签加偏置
            if o_bias != 0.0:
                emissions[:, :, 0] += o_bias
            
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            
            # 计算置信度
            probs = torch.softmax(emissions, dim=-1)
            confidences = []
            for batch_idx, pred_seq in enumerate(predictions):
                seq_conf = []
                for pos, label_id in enumerate(pred_seq):
                    conf = probs[batch_idx, pos, label_id].item()
                    seq_conf.append(conf)
                confidences.append(seq_conf)
        
        return predictions, confidences
    
    def classify_type(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        type_entity_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """只进行TYPE分类"""
        if self.type_classifier is None:
            raise ValueError("TYPE分类头未初始化")
        
        with torch.no_grad():
            encoder_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if token_type_ids is not None and hasattr(self.config, 'type_vocab_size') and self.config.type_vocab_size > 0:
                encoder_kwargs['token_type_ids'] = token_type_ids
            
            outputs = self.encoder(**encoder_kwargs)
            
            if type_entity_mask is not None:
                repr = self._masked_mean_pooling(outputs.last_hidden_state, type_entity_mask)
            else:
                repr = outputs.pooler_output
            
            logits = self.type_classifier(repr)
            return torch.argmax(logits, dim=-1)


def create_multitask_model(
    model_name: str,
    num_ner_labels: int,
    num_type_labels: int = 0
) -> MultiTaskModel:
    """
    创建多任务模型（支持 BERT、XLM-RoBERTa 等）
    
    Args:
        model_name: 预训练模型名称或路径
        num_ner_labels: NER标签数量
        num_type_labels: TYPE分类标签数量
    """
    import logging
    logger = logging.getLogger(__name__)
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_ner_labels
    
    logger.info(f"加载预训练模型: {model_name} (类型: {config.model_type})")
    
    model = MultiTaskModel(config, num_type_labels=num_type_labels)
    
    # 加载预训练权重到 encoder
    pretrained_model = AutoModel.from_pretrained(model_name, add_pooling_layer=True)
    model.encoder.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    logger.info(f"成功加载预训练权重到 encoder")
    
    return model


# 保持向后兼容的别名
BertMultiTaskModel = MultiTaskModel
