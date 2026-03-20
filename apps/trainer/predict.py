"""
多任务NER模型预测脚本

使用方法:
    python -m apps.trainer.predict --model_path outputs/pipe_multitask_v9/best_model --text "承插焊法兰, S30408 NB/T47010, RF, CL 600, DN15"

参数:
    --model_path: 训练好的模型路径
    --text: 要预测的文本
    --file: 包含多行文本的文件
"""

import os
import sys
import argparse
import json
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from src.bert_ner.bert_multitask import MultiTaskModel


class NERPredictor:
    """多任务NER预测器"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            device: 设备（cuda, mps, cpu, auto）
        """
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
        
        print(f"使用设备: {self.device}")
        
        # 加载 NER 标签配置
        ner_labels_path = os.path.join(model_path, 'ner_labels.json')
        if os.path.exists(ner_labels_path):
            with open(ner_labels_path, 'r', encoding='utf-8') as f:
                ner_config = json.load(f)
                self.ner_id2label = {int(k): v for k, v in ner_config['id2label'].items()}
                self.ner_label2id = ner_config['label2id']
        else:
            raise FileNotFoundError(f"NER标签配置不存在: {ner_labels_path}")
        
        # 加载 TYPE 标签配置
        type_labels_path = os.path.join(model_path, 'type_labels.json')
        if os.path.exists(type_labels_path):
            with open(type_labels_path, 'r', encoding='utf-8') as f:
                type_config = json.load(f)
                self.type_id2label = {int(k): v for k, v in type_config['id2label'].items()}
                self.type_label2id = type_config['label2id']
        else:
            self.type_id2label = {}
            self.type_label2id = {}
        
        # 加载模型和tokenizer
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultiTaskModel.from_pretrained(model_path, num_type_labels=len(self.type_id2label))
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成！")
        print(f"NER标签数: {len(self.ner_id2label)}")
        print(f"TYPE标签数: {len(self.type_id2label)}")
    
    def predict(self, text: str) -> dict:
        """
        预测单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果，包含实体列表和类型分类
        """
        # 将文本转换为字符列表（与训练时保持一致！）
        # 关键：将空格替换为 [SPACE] 特殊 token
        chars = []
        original_chars = list(text)  # 保存原始字符用于输出
        for char in text:
            if char.isspace():
                chars.append('[SPACE]')  # 与训练时相同的处理
            else:
                chars.append(char)
        
        # 编码
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # NER 预测
        ner_predictions = outputs['ner_predictions'][0]  # 第一个样本
        
        # TYPE 预测
        type_class = None
        if 'type_logits' in outputs and self.type_id2label:
            type_logits = outputs['type_logits']
            type_pred = type_logits.argmax(dim=-1).item()
            type_class = self.type_id2label.get(type_pred, "未知")
        
        # 获取word_ids映射
        word_ids = encoding.word_ids()
        
        # 将预测映射回原始字符（去重：一个字符可能对应多个subword）
        unique_labels = []
        prev_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id and word_id < len(original_chars):
                if i < len(ner_predictions):
                    label = self.ner_id2label.get(ner_predictions[i], 'O')
                    unique_labels.append((original_chars[word_id], label))
                prev_word_id = word_id
        
        # 提取实体
        entities = self._extract_entities(unique_labels)
        
        return {
            'text': text,
            'char_labels': unique_labels,
            'entities': entities,
            'type_class': type_class
        }
    
    def _extract_entities(self, char_labels: list) -> list:
        """从字符级标签中提取实体"""
        entities = []
        current_entity = []
        current_type = None
        start_pos = 0
        
        for i, (char, tag) in enumerate(char_labels):
            if tag.startswith('B-'):
                # 保存上一个实体
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': start_pos,
                        'end': i
                    })
                # 开始新实体
                current_type = tag[2:]
                current_entity = [char]
                start_pos = i
            elif tag.startswith('I-') and current_type == tag[2:]:
                # 继续当前实体
                current_entity.append(char)
            else:
                # O标签或不匹配的I标签
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type,
                        'start': start_pos,
                        'end': i
                    })
                current_entity = []
                current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append({
                'text': ''.join(current_entity),
                'type': current_type,
                'start': start_pos,
                'end': len(char_labels)
            })
        
        return entities
    
    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        return [self.predict(text) for text in texts]


def format_result(result: dict) -> str:
    """格式化预测结果"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"原文: {result['text']}")
    lines.append("-" * 60)
    
    if result.get('type_class'):
        lines.append(f"类型分类: {result['type_class']}")
    
    if result['entities']:
        lines.append("\n识别的实体:")
        for entity in result['entities']:
            lines.append(f"  [{entity['type']:15s}] {entity['text']}")
    else:
        lines.append("未识别到实体")
    
    # 显示字符级标注（简化版）
    lines.append("\n字符级标注 (前50字符):")
    bio_parts = []
    for char, tag in result['char_labels'][:50]:
        if tag == 'O':
            bio_parts.append(char)
        else:
            bio_parts.append(f"[{char}/{tag}]")
    bio_str = "".join(bio_parts)
    if len(result['char_labels']) > 50:
        bio_str += " ..."
    lines.append(f"  {bio_str}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='管道材料NER预测')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径（best_model 目录）')
    parser.add_argument('--text', type=str, default=None,
                        help='要预测的文本')
    parser.add_argument('--file', type=str, default=None,
                        help='包含多行文本的文件')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: cuda, mps, cpu, auto')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（JSON格式）')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("请提供 --text 或 --file 参数")
    
    # 检查模型路径
    model_path = args.model_path
    
    # 优先检查 best_model 子目录
    best_model_path = os.path.join(model_path, 'best_model')
    if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
        model_path = best_model_path
        print(f"使用模型目录: {model_path}")
    elif not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 初始化预测器
    predictor = NERPredictor(model_path, args.device)
    
    # 收集要预测的文本
    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    
    # 预测
    results = []
    for text in texts:
        result = predictor.predict(text)
        results.append(result)
        print(format_result(result))
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
