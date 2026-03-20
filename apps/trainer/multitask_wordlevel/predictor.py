"""
词级别多任务模型预测器

关键改动：使用 offset_mapping 将 token 预测映射回字符级别
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional, Tuple

# 设置确定性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WordLevelPipePredictor:
    """
    词级别管道材料多任务预测器
    
    与原 PipePredictor 的区别：
    - 使用词级别分词（直接传入文本）
    - 使用 offset_mapping 映射预测结果回字符位置
    """
    
    def __init__(self, model_path: str, device: str = 'auto', o_bias: float = 0.0):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            device: 设备（cuda, mps, cpu, auto）
            o_bias: O标签偏置
        """
        from transformers import AutoTokenizer, AutoConfig
        
        # 动态导入模型类
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.bert_ner.bert_multitask import MultiTaskModel
        
        self.o_bias = o_bias
        
        # 自动选择设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[词级别预测器] 使用设备: {self.device}")
        
        # 加载NER标签配置
        ner_labels_path = os.path.join(model_path, 'ner_labels.json')
        if os.path.exists(ner_labels_path):
            with open(ner_labels_path, 'r', encoding='utf-8') as f:
                ner_config = json.load(f)
                self.ner_id2label = {int(k): v for k, v in ner_config['id2label'].items()}
                self.ner_label2id = ner_config['label2id']
        else:
            self.ner_labels = [
                'O',
                'B-TYPE', 'I-TYPE',
                'B-MATERIAL', 'I-MATERIAL',
                'B-SIZE', 'I-SIZE',
                'B-THICKNESS', 'I-THICKNESS',
                'B-PRESSURE', 'I-PRESSURE',
                'B-STANDARD', 'I-STANDARD',
                'B-CONN', 'I-CONN',
                'B-MANU', 'I-MANU'
            ]
            self.ner_id2label = {i: label for i, label in enumerate(self.ner_labels)}
            self.ner_label2id = {label: i for i, label in enumerate(self.ner_labels)}
        
        # 加载TYPE分类标签
        type_labels_path = os.path.join(model_path, 'type_labels.json')
        if os.path.exists(type_labels_path):
            with open(type_labels_path, 'r', encoding='utf-8') as f:
                type_config = json.load(f)
                self.type_id2label = {int(k): v for k, v in type_config['id2label'].items()}
                self.type_label2id = type_config['label2id']
        else:
            self.type_labels = ['管子', '管件', '法兰', '螺栓', '阀门', '垫片']
            self.type_id2label = {i: label for i, label in enumerate(self.type_labels)}
            self.type_label2id = {label: i for i, label in enumerate(self.type_labels)}
        
        # 加载模型和tokenizer
        print(f"[词级别预测器] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = len(self.ner_label2id)
        
        self.model = MultiTaskModel.from_pretrained(
            model_path,
            config=config,
            num_type_labels=len(self.type_label2id)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = 256
        print(f"[词级别预测器] 模型加载完成！NER标签数: {len(self.ner_label2id)}, TYPE分类数: {len(self.type_label2id)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        预测文本
        
        Args:
            text: 输入文本
            
        Returns:
            {
                'text': 原始文本,
                'tokens': 分词结果列表,
                'entities': 实体列表,
                'type_class': 材料大类
            }
        """
        if not text or not text.strip():
            return {'text': text, 'tokens': [], 'entities': [], 'type_class': None}
        
        original_text = text
        
        # 转换为小写进行预测
        text_lower = text.lower()
        
        # 【关键改动】词级别分词 + offset_mapping
        encoding = self.tokenizer(
            text_lower,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # 关键
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0].tolist()  # [(start, end), ...]
        
        # NER预测
        with torch.no_grad():
            ner_predictions, ner_confidences = self.model.decode_ner_with_confidence(
                input_ids, attention_mask, o_bias=self.o_bias
            )
        
        predictions = ner_predictions[0]  # 第一个样本
        confidences = ner_confidences[0]
        
        # 【关键改动】直接在 token 级别用 BIO 规则提取实体
        # 不映射回字符级别，避免空格导致实体断开
        entities, tokens = self._extract_entities_from_tokens(
            predictions, confidences, offset_mapping, original_text
        )
        
        # 后处理：过滤无效识别
        entities = [e for e in entities if not (e['type'] == 'STANDARD' and len(e['text']) < 3)]
        
        # 获取 TYPE 实体
        type_entity = None
        for entity in entities:
            if entity['type'] == 'TYPE':
                type_entity = entity['text']
                break
        
        # TYPE分类
        type_class = None
        if self.type_label2id:
            type_class = self._classify_type(text_lower, type_entity)
        
        return {
            'text': original_text,
            'tokens': tokens,
            'entities': entities,
            'type_class': type_class
        }
    
    def _extract_entities_from_tokens(
        self,
        token_predictions: List[int],
        token_confidences: List[float],
        offset_mapping: List[Tuple[int, int]],
        original_text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        直接在 token 级别用 BIO 规则提取实体
        
        关键：用 BIO 标签判断实体边界，用 offset_mapping 获取文本范围
        这样可以正确处理 "GB/T 12459" 这种空格分隔但属于同一实体的情况
        
        Args:
            token_predictions: token 级预测 ID 列表
            token_confidences: token 级置信度列表
            offset_mapping: 每个 token 的 (start, end) 位置
            original_text: 原始文本
        
        Returns:
            (entities, tokens): 实体列表和分词结果
        """
        entities = []
        tokens = []
        
        # 当前正在构建的实体
        current_entity_type = None
        current_entity_start = None
        current_entity_end = None
        current_entity_confs = []
        
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue  # 跳过特殊 token（<s>, </s>, <pad>）
            
            if token_idx >= len(token_predictions):
                continue
            
            pred_id = token_predictions[token_idx]
            conf = token_confidences[token_idx] if token_idx < len(token_confidences) else 1.0
            label = self.ner_id2label.get(pred_id, 'O')
            
            # 解析 BIO 标签
            if label.startswith('B-'):
                entity_type = label[2:]
                is_begin = True
            elif label.startswith('I-'):
                entity_type = label[2:]
                is_begin = False
            else:
                entity_type = 'O'
                is_begin = True
            
            # 记录 token 信息
            token_text = original_text[start:end]
            tokens.append({
                'word': token_text,
                'start': start,
                'end': end,
                'tag': entity_type,
                'confidence': round(conf, 4)
            })
            
            # BIO 规则处理实体
            if entity_type == 'O':
                # 遇到 O，结束当前实体
                if current_entity_type is not None:
                    self._save_entity(entities, current_entity_type, 
                                     current_entity_start, current_entity_end,
                                     current_entity_confs, original_text)
                    current_entity_type = None
                    current_entity_start = None
                    current_entity_end = None
                    current_entity_confs = []
            elif is_begin:
                # 遇到 B-X
                # 【关键修复】如果新 B 标签的 start 和当前实体 start 相同（重叠 token），
                # 不要创建新实体，而是继续当前实体（扩展 end）
                if current_entity_type is not None and current_entity_start == start:
                    # 同一位置的重复 B 标签，忽略，只扩展 end
                    if end > current_entity_end:
                        current_entity_end = end
                    current_entity_confs.append(conf)
                else:
                    # 正常情况：结束当前实体，开始新实体
                    if current_entity_type is not None:
                        self._save_entity(entities, current_entity_type,
                                         current_entity_start, current_entity_end,
                                         current_entity_confs, original_text)
                    current_entity_type = entity_type
                    current_entity_start = start
                    current_entity_end = end
                    current_entity_confs = [conf]
            else:
                # 遇到 I-X
                if current_entity_type == entity_type:
                    # 同类型，继续当前实体
                    current_entity_end = end
                    current_entity_confs.append(conf)
                else:
                    # 类型不匹配，结束当前实体，开始新实体
                    if current_entity_type is not None:
                        self._save_entity(entities, current_entity_type,
                                         current_entity_start, current_entity_end,
                                         current_entity_confs, original_text)
                    current_entity_type = entity_type
                    current_entity_start = start
                    current_entity_end = end
                    current_entity_confs = [conf]
        
        # 处理最后一个实体
        if current_entity_type is not None:
            self._save_entity(entities, current_entity_type,
                             current_entity_start, current_entity_end,
                             current_entity_confs, original_text)
        
        return entities, tokens
    
    def _save_entity(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
        start: int,
        end: int,
        confidences: List[float],
        original_text: str
    ):
        """保存实体到列表"""
        if start is None or end is None:
            return
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
        entity_text = original_text[start:end]  # 直接用 offset 获取原文，包含空格！
        
        entities.append({
            'text': entity_text,
            'type': entity_type,
            'start': start,
            'end': end,
            'confidence': round(avg_conf, 4)
        })
    
    def _classify_type(self, text: str, type_entity: Optional[str]) -> Optional[str]:
        """
        对文本进行 TYPE 分类
        """
        if not self.type_label2id:
            return None
        
        # 【改动】词级别分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0].tolist()
        
        # 如果有 TYPE 实体，使用实体掩码
        entity_mask_tensor = None
        if type_entity:
            entity_mask = self._create_entity_mask(text, type_entity, offset_mapping)
            entity_mask_tensor = torch.tensor([entity_mask], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            pred_id = self.model.classify_type(input_ids, attention_mask, entity_mask_tensor)
        
        return self.type_id2label.get(pred_id.item(), None)
    
    def _create_entity_mask(
        self,
        text: str,
        entity: str,
        offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        """创建实体位置掩码"""
        mask = [0] * len(offset_mapping)
        
        if not entity:
            return mask
        
        start = text.find(entity)
        if start == -1:
            return mask
        
        end = start + len(entity)
        
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            if tok_start < end and tok_end > start:
                mask[i] = 1
        
        return mask
