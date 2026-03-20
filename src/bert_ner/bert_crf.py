"""
BERT + CRF 命名实体识别模型
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF


class BertCRFModel(BertPreTrainedModel):
    """
    BERT + CRF 模型用于序列标注
    
    结构：
    - BERT编码器：将输入token编码为上下文表示
    - Dropout层：防止过拟合
    - 线性分类层：将BERT输出映射到标签空间
    - CRF层：建模标签转移约束，确保输出合法的BIO序列
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # CRF层
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_emissions=False
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型IDs [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]，-100表示忽略
            return_emissions: 是否返回emission分数
            
        Returns:
            如果有labels：返回loss
            否则：返回预测的标签序列
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # 映射到标签空间
        emissions = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        if return_emissions:
            return emissions
        
        # 创建有效位置的掩码（排除padding和特殊token）
        # 对于CRF，我们需要将attention_mask转换为bool类型
        mask = attention_mask.bool()
        
        if labels is not None:
            # 训练模式：计算CRF损失
            # 需要处理-100标签（将其替换为0，但通过mask忽略）
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            
            # CRF负对数似然损失
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            return {'loss': loss, 'emissions': emissions}
        else:
            # 推理模式：使用Viterbi解码
            predictions = self.crf.decode(emissions, mask=mask)
            return {'predictions': predictions, 'emissions': emissions}
    
    def decode(self, input_ids, attention_mask, token_type_ids=None):
        """
        解码预测
        
        Returns:
            预测的标签序列列表
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        return outputs['predictions']


def create_model(config, pretrained_model_name: str = None):
    """
    创建BERT-CRF模型
    
    Args:
        config: NERConfig配置对象
        pretrained_model_name: 预训练模型名称或路径
        
    Returns:
        BertCRFModel实例
    """
    from transformers import BertConfig
    
    model_name = pretrained_model_name or config.bert_model
    
    # 加载BERT配置并添加标签信息
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.num_labels = config.num_labels
    bert_config.label2id = config.label2id()
    bert_config.id2label = config.id2label()
    
    # 创建模型
    model = BertCRFModel.from_pretrained(
        model_name,
        config=bert_config,
        ignore_mismatched_sizes=True
    )
    
    return model

