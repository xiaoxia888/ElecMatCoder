"""
多任务模型预测器 - 用于API服务

支持：
1. NER: 序列标注（提取实体）
2. TYPE分类: 判断材料大类（管子/管件/法兰/螺栓/阀门/垫片）
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional

# 设置确定性，避免MPS等后端的不稳定性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PipePredictor:
    """
    管道材料多任务预测器
    
    同时完成NER和TYPE分类
    """
    
    def __init__(self, model_path: str, device: str = 'auto', o_bias: float = 0.0):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径（包含模型文件和配置）
            device: 设备（cuda, mps, cpu, auto）
            o_bias: O标签偏置，正值使模型更倾向于预测O（推荐1.0-3.0）
        """
        from transformers import AutoTokenizer, AutoConfig
        from .bert_multitask import MultiTaskModel
        
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
        
        print(f"[管道预测器] 使用设备: {self.device}")
        
        # 加载NER标签配置
        ner_labels_path = os.path.join(model_path, 'ner_labels.json')
        if os.path.exists(ner_labels_path):
            with open(ner_labels_path, 'r', encoding='utf-8') as f:
                ner_config = json.load(f)
                self.ner_id2label = {int(k): v for k, v in ner_config['id2label'].items()}
                self.ner_label2id = ner_config['label2id']
        else:
            # 默认标签（包含 CONN 和 MANU）
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
        
        # 加载TYPE分类标签（材料大类）
        type_labels_path = os.path.join(model_path, 'type_labels.json')
        if os.path.exists(type_labels_path):
            with open(type_labels_path, 'r', encoding='utf-8') as f:
                type_config = json.load(f)
                self.type_id2label = {int(k): v for k, v in type_config['id2label'].items()}
                self.type_label2id = type_config['label2id']
        else:
            # 默认分类
            self.type_labels = ['管子', '管件', '法兰', '螺栓', '阀门', '垫片']
            self.type_id2label = {i: label for i, label in enumerate(self.type_labels)}
            self.type_label2id = {label: i for i, label in enumerate(self.type_labels)}
        
        # 加载模型和tokenizer
        print(f"[管道预测器] 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载配置
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = len(self.ner_label2id)
        
        # 加载模型
        self.model = MultiTaskModel.from_pretrained(
            model_path,
            config=config,
            num_type_labels=len(self.type_label2id)
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = 256  # 增加以支持较长文本
        print(f"[管道预测器] 模型加载完成！NER标签数: {len(self.ner_label2id)}, TYPE分类数: {len(self.type_label2id)}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        预测文本，返回NER结果和TYPE分类结果
        
        Args:
            text: 输入文本
            
        Returns:
            {
                'text': 原始文本,
                'tokens': 分词结果列表,
                'entities': 实体列表,
                'type_class': 材料大类（管子/管件/法兰/螺栓/阀门/垫片）
            }
        """
        if not text or not text.strip():
            return {'text': text, 'tokens': [], 'entities': [], 'type_class': None}
        
        # 保存原始文本和字符（用于输出，保留原始大小写）
        original_text = text
        original_chars = list(text)
        
        # 转换为小写进行模型预测（解决大小写敏感问题）
        # 模型训练时也应使用小写数据
        text_lower = text.lower()
        lower_chars = list(text_lower)
        
        # 将空格替换为 [SPACE]，这样 tokenizer 不会跳过它
        # 记录哪些位置是空格
        space_positions = set()
        processed_chars = []
        for i, char in enumerate(lower_chars):  # 使用小写字符
            if char.isspace():
                space_positions.add(i)
                processed_chars.append('[SPACE]')  # 用特殊 token 替代空格
            else:
                processed_chars.append(char)
        
        # 编码（使用处理后的字符）
        encoding = self.tokenizer(
            processed_chars,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # NER预测（带置信度和O标签偏置）
        with torch.no_grad():
            ner_predictions, ner_confidences = self.model.decode_ner_with_confidence(
                input_ids, attention_mask, o_bias=self.o_bias
            )
        
        predictions = ner_predictions[0]  # 第一个样本
        confidences = ner_confidences[0]  # 第一个样本的置信度
        
        # 获取word_ids映射
        word_ids = encoding.word_ids()
        
        # 将预测映射回原始字符
        char_labels = []
        last_word_id = None
        for idx, (pred_id, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is not None and word_id < len(processed_chars) and word_id != last_word_id:
                label = self.ner_id2label.get(pred_id, 'O')
                conf = confidences[idx] if idx < len(confidences) else 1.0
                # 使用原始字符（空格而非下划线）
                original_char = original_chars[word_id]
                char_labels.append((word_id, original_char, label, conf))
                last_word_id = word_id
        
        # 现在所有字符都应该有预测了（包括空格位置的下划线）
        full_char_labels = char_labels

        # 合并连续相同标签的字符为词
        tokens = []
        current_word = []
        current_tag = None
        current_start = 0
        current_confs = []  # 当前词的置信度列表
        
        for idx, char, tag, conf in full_char_labels:
            if tag.startswith('B-'):
                actual_tag = tag[2:]
                is_begin = True
            elif tag.startswith('I-'):
                actual_tag = tag[2:]
                is_begin = False
            else:
                actual_tag = 'O'
                is_begin = True
            
            # 只有在以下情况才开始新词：
            # 1. 遇到 B- 标签
            # 2. 标签类型发生变化
            # 3. 如果当前是 O 标签，每个字符都应该独立（如空格、分号）
            should_start_new = (
                is_begin or
                actual_tag != current_tag or
                actual_tag == 'O'
            )
            
            if should_start_new and current_word:
                # 计算词的平均置信度
                avg_conf = sum(current_confs) / len(current_confs) if current_confs else 1.0
                tokens.append({
                    'word': ''.join(current_word),
                    'start': current_start,
                    'end': current_start + len(current_word),
                    'tag': current_tag,
                    'confidence': round(avg_conf, 4)
                })
                current_word = []
                current_confs = []
            
            if not current_word:
                current_start = idx
            
            current_word.append(char)
            current_tag = actual_tag
            current_confs.append(conf)
        
        if current_word:
            avg_conf = sum(current_confs) / len(current_confs) if current_confs else 1.0
            word = ''.join(current_word)
            tokens.append({
                'word': word,
                'start': current_start,
                'end': current_start + len(current_word),
                'tag': current_tag,
                'confidence': round(avg_conf, 4)
            })
        
        # # 后处理：过滤无效的STANDARD识别（长度过短）
        for token in tokens:
            if token['tag'] == 'STANDARD' and len(token['word']) < 3:
                token['tag'] = 'O'
        
        # 提取实体
        entities = []
        type_entity = None
        
        for token in tokens:
            if token['tag'] != 'O':
                entity = {
                    'text': token['word'],
                    'type': token['tag'],
                    'start': token['start'],
                    'end': token['end'],
                    'confidence': token.get('confidence', 1.0)
                }
                entities.append(entity)
                
                if token['tag'] == 'TYPE' and type_entity is None:
                    type_entity = token['word']
        
        # TYPE分类（判断材料大类）
        type_class = None
        if self.type_label2id:
            type_class = self._classify_type(text, type_entity)
        
        return {
            'text': text,
            'tokens': tokens,
            'entities': entities,
            'type_class': type_class
        }
    
    def _classify_type(self, text: str, type_entity: Optional[str]) -> Optional[str]:
        """
        对整个文本进行TYPE分类（判断材料大类）
        
        Args:
            text: 输入文本
            type_entity: TYPE实体文本（可选，用于创建实体掩码）
        
        Returns:
            材料大类（管子/管件/法兰/螺栓/阀门/垫片）
        """
        if not self.type_label2id:
            return None
        
        # 转换为小写（与 predict 方法保持一致）
        text_lower = text.lower()
        
        # 将空格替换为 [SPACE]（与训练时保持一致）
        processed_chars = []
        for char in text_lower:  # 使用小写字符
            if char.isspace():
                processed_chars.append('[SPACE]')
            else:
                processed_chars.append(char)
        
        encoding = self.tokenizer(
            processed_chars,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 如果有TYPE实体，使用实体掩码；否则使用[CLS]
        entity_mask_tensor = None
        if type_entity:
            entity_mask = self._create_entity_mask(text, type_entity)
            entity_mask_tensor = torch.tensor([entity_mask], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            pred_id = self.model.classify_type(input_ids, attention_mask, entity_mask_tensor)
        
        return self.type_id2label.get(pred_id.item(), None)
    
    def _create_entity_mask(self, text: str, entity: str) -> List[int]:
        """创建实体位置掩码"""
        mask = [0] * self.max_length
        
        if not entity:
            return mask
        
        start = text.find(entity)
        if start == -1:
            return mask
        
        end = start + len(entity)
        
        # 标记对应位置（+1 因为有[CLS]）
        for i in range(start, min(end, self.max_length - 2)):
            mask[i + 1] = 1
        
        return mask


# 兼容旧版本的NERPredictor
class NERPredictor:
    """
    NER预测器（兼容旧版本）
    
    用于加载旧版本的BERT+CRF模型
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        from transformers import BertTokenizerFast
        from .bert_crf import BertCRFModel
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[NER预测器] 使用设备: {self.device}")
        
        config_path = os.path.join(model_path, 'ner_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.ner_config = json.load(f)
        else:
            self.ner_config = {
                'labels': [
                    'O',
                    'B-NAME', 'I-NAME',
                    'B-MATERIAL', 'I-MATERIAL',
                    'B-TYPE', 'I-TYPE',
                    'B-SPEC', 'I-SPEC',
                ],
                'max_seq_length': 128
            }
        
        self.labels = self.ner_config['labels']
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.max_length = self.ner_config.get('max_seq_length', 128)
        
        print(f"[NER预测器] 加载模型: {model_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertCRFModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[NER预测器] 模型加载完成！")
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """预测单个文本，返回分词结果"""
        if not text or not text.strip():
            return []
        
        original_chars = list(text)
        
        # 转换为小写进行模型预测（解决大小写敏感问题）
        text_lower = text.lower()
        lower_chars = list(text_lower)
        
        # 将空格替换为 [SPACE]（与训练时保持一致）
        chars = []
        for char in lower_chars:  # 使用小写字符
            if char.isspace():
                chars.append('[SPACE]')
            else:
                chars.append(char)
        
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        predictions = outputs['predictions'][0]
        word_ids = encoding.word_ids()
        
        char_labels_dict = {}
        last_word_id = None
        for pred_id, word_id in zip(predictions, word_ids):
            if word_id is not None and word_id < len(original_chars) and word_id != last_word_id:
                label = self.id2label[pred_id]
                char_labels_dict[word_id] = (word_id, original_chars[word_id], label)
                last_word_id = word_id
        
        char_labels = []
        for i, char in enumerate(original_chars):
            if i in char_labels_dict:
                char_labels.append(char_labels_dict[i])
            else:
                char_labels.append((i, char, 'O'))
        
        tokens = []
        current_word = []
        current_tag = None
        current_start = 0
        
        for idx, char, tag in char_labels:
            if char.isspace():
                actual_tag = 'O'
                is_begin = True
            elif tag.startswith('B-'):
                actual_tag = tag[2:]
                is_begin = True
            elif tag.startswith('I-'):
                actual_tag = tag[2:]
                is_begin = False
            else:
                actual_tag = 'O'
                is_begin = True
            
            should_start_new = (
                is_begin or
                actual_tag != current_tag or
                (actual_tag == 'O' and current_tag == 'O')
            )
            
            if should_start_new and current_word:
                tokens.append({
                    'word': ''.join(current_word),
                    'start': current_start,
                    'end': current_start + len(current_word),
                    'tag': current_tag
                })
                current_word = []
            
            if not current_word:
                current_start = idx
            
            current_word.append(char)
            current_tag = actual_tag
        
        if current_word:
            tokens.append({
                'word': ''.join(current_word),
                'start': current_start,
                'end': current_start + len(current_word),
                'tag': current_tag
            })
        
        return tokens
    
    def predict_raw(self, text: str) -> Dict[str, Any]:
        """预测并返回详细结果"""
        tokens = self.predict(text)
        
        entities = []
        for token in tokens:
            if token['tag'] != 'O':
                entities.append({
                    'text': token['word'],
                    'type': token['tag'],
                    'start': token['start'],
                    'end': token['end']
                })
        
        return {
            'text': text,
            'tokens': tokens,
            'entities': entities
        }
