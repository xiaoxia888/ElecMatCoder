"""
管道材料实体分类器

使用训练好的BERT分类器将实体文本转换为标准编码
"""

import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, Optional, Tuple


class EntityClassifier:
    """实体分类器：将实体文本转换为标准编码"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: 模型路径
            device: 设备 (auto, cuda, mps, cpu)
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
        
        print(f"[EntityClassifier] 设备: {self.device}")
        
        # 加载模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载标签映射
        label_map_path = os.path.join(model_path, "label_map.json")
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        self.label2id = label_map['label2id']
        self.labels = label_map['labels']
        
        print(f"[EntityClassifier] 类别数: {len(self.labels)}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        预测单个实体的编码
        
        Args:
            text: 实体文本（如 "A182 F321"）
            
        Returns:
            (编码, 置信度) 如 ("321", 0.95)
        """
        # 编码
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            pred_id = logits.argmax(dim=-1).item()
            confidence = probs[0, pred_id].item()
        
        code = self.id2label[pred_id]
        return code, confidence
    
    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        results = []
        for text in texts:
            code, conf = self.predict(text)
            results.append({
                'text': text,
                'code': code,
                'confidence': conf
            })
        return results


class PipeEntityNormalizer:
    """
    管道实体归一化器
    
    组合多个分类器，将NER提取的实体转换为标准编码
    """
    
    def __init__(
        self,
        type_model_path: str = None,
        material_model_path: str = None,
        device: str = 'auto'
    ):
        """
        Args:
            type_model_path: 种类分类器路径
            material_model_path: 材质分类器路径
        """
        self.classifiers = {}
        
        # 默认路径
        base_path = "models/pipe_classifier"
        
        if type_model_path or os.path.exists(os.path.join(base_path, "type/final")):
            path = type_model_path or os.path.join(base_path, "type/final")
            print(f"加载种类分类器: {path}")
            self.classifiers['TYPE'] = EntityClassifier(path, device)
        
        if material_model_path or os.path.exists(os.path.join(base_path, "material/final")):
            path = material_model_path or os.path.join(base_path, "material/final")
            print(f"加载材质分类器: {path}")
            self.classifiers['MATERIAL'] = EntityClassifier(path, device)
    
    def normalize(self, entity_type: str, text: str) -> Dict:
        """
        归一化单个实体
        
        Args:
            entity_type: 实体类型 (TYPE, MATERIAL, SIZE, etc.)
            text: 实体文本
            
        Returns:
            {
                'code': '321',
                'confidence': 0.95,
                'method': 'classifier' | 'rule'
            }
        """
        # 使用分类器
        if entity_type in self.classifiers:
            code, confidence = self.classifiers[entity_type].predict(text)
            return {
                'code': code,
                'confidence': confidence,
                'method': 'classifier'
            }
        
        # 规则转换
        if entity_type == 'STANDARD':
            code = self._convert_standard(text)
            return {'code': code, 'confidence': 1.0, 'method': 'rule'}
        
        if entity_type == 'SIZE':
            code = self._convert_size(text)
            return {'code': code, 'confidence': 1.0, 'method': 'rule'}
        
        if entity_type == 'THICKNESS':
            code = self._convert_thickness(text)
            return {'code': code, 'confidence': 1.0, 'method': 'rule'}
        
        if entity_type == 'PRESSURE':
            code = self._convert_pressure(text)
            return {'code': code, 'confidence': 1.0, 'method': 'rule'}
        
        # 未知类型，返回原值
        return {'code': text, 'confidence': 0.0, 'method': 'unknown'}
    
    def _convert_standard(self, text: str) -> str:
        """标准号转换：GB/T3087 → GBT3087"""
        return text.replace("/", "").replace(" ", "").replace("-", "").upper()
    
    def _convert_size(self, text: str) -> str:
        """尺寸转换：DN15 → 15"""
        import re
        # 去掉 DN 前缀
        text = re.sub(r'^DN', '', text, flags=re.IGNORECASE)
        # 去掉 φ 或 Φ 前缀
        text = re.sub(r'^[φΦ]', '', text)
        return text.strip()
    
    def _convert_thickness(self, text: str) -> str:
        """壁厚转换：4mm → 4MM"""
        return text.upper()
    
    def _convert_pressure(self, text: str) -> str:
        """压力转换：CL3000 → C3000"""
        import re
        text = text.upper()
        text = re.sub(r'^CL(?=\d)', 'C', text)
        text = re.sub(r'^CLASS', 'C', text)
        return text


# 测试代码
if __name__ == "__main__":
    # 测试分类器
    normalizer = PipeEntityNormalizer()
    
    # 测试材质
    test_materials = [
        "A182 F321",
        "TP321",
        "ASTM A182 GRADE F321",
        "S30408",
        "304",
        "316L",
    ]
    
    print("\n=== 材质分类测试 ===")
    for material in test_materials:
        result = normalizer.normalize('MATERIAL', material)
        print(f"{material:30s} → {result['code']:10s} (置信度: {result['confidence']:.3f})")
    
    # 测试种类
    test_types = [
        "管子",
        "弯头",
        "TEE(BW)",
        "等径三通",
        "straight tee",
    ]
    
    print("\n=== 种类分类测试 ===")
    for t in test_types:
        result = normalizer.normalize('TYPE', t)
        print(f"{t:30s} → {result['code']:10s} (置信度: {result['confidence']:.3f})")
    
    # 测试规则转换
    print("\n=== 规则转换测试 ===")
    print(f"GB/T3087 → {normalizer.normalize('STANDARD', 'GB/T3087')['code']}")
    print(f"DN100 → {normalizer.normalize('SIZE', 'DN100')['code']}")
    print(f"CL3000 → {normalizer.normalize('PRESSURE', 'CL3000')['code']}")
    print(f"4mm → {normalizer.normalize('THICKNESS', '4mm')['code']}")



