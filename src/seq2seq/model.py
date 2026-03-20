# ============================================
# Seq2Seq TYPE编码生成模型
# ============================================

import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)
from typing import Optional, List


def preprocess_text(text: str) -> str:
    """预处理输入文本：清理多余空格，统一转小写"""
    processed = re.sub(r'\s+', ' ', text).strip()
    processed = processed.lower()  # 统一转小写
    return processed


class TypeEncoder:
    """TYPE编码生成器"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 16
    ):
        """
        Args:
            model_path: 模型路径（预训练模型名或本地路径）
            device: 设备 (cuda, mps, cpu, None=auto)
            num_beams: beam search 宽度
            max_length: 生成最大长度
        """
        self.model_path = model_path
        self.num_beams = num_beams
        self.max_length = max_length
        
        # 自动选择设备
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, type_name: str, preprocess: bool = True) -> str:
        """
        将类型名称编码为类型编码
        
        Args:
            type_name: 类型名称，如 "45度弯头"、"无缝钢管"
            preprocess: 是否预处理（替换符号）
            
        Returns:
            类型编码，如 "45EL"、"P"
        """
        if preprocess:
            type_name = preprocess_text(type_name)
        
        # 编码输入
        inputs = self.tokenizer(
            type_name,
            return_tensors='pt',
            max_length=64,
            truncation=True
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # 解码
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code.strip()
    
    def encode_batch(
        self, 
        type_names: List[str], 
        preprocess: bool = True
    ) -> List[str]:
        """
        批量编码
        
        Args:
            type_names: 类型名称列表
            preprocess: 是否预处理
            
        Returns:
            编码列表
        """
        if preprocess:
            type_names = [preprocess_text(name) for name in type_names]
        
        # 编码输入
        inputs = self.tokenizer(
            type_names,
            return_tensors='pt',
            max_length=64,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # 解码
        codes = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [code.strip() for code in codes]
    
    def save(self, output_dir: str):
        """保存模型"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'TypeEncoder':
        """从预训练模型加载"""
        return cls(model_path, **kwargs)
