# -*- coding: utf-8 -*-
"""
Seq2Seq 编码器
用于 TYPE 和 MATERIAL 的生成式编码
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch

from src.seq2seq.model import preprocess_text

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqResult:
    """Seq2Seq 编码结果"""
    original: str           # 原始输入
    code: str               # 生成的编码
    confidence: float = 1.0  # 置信度
    

class Seq2SeqEncoder:
    """
    Seq2Seq 编码器
    
    使用训练好的 mT5 模型将 TYPE/MATERIAL 文本转换为编码
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        初始化编码器
        
        Args:
            model_path: 模型路径
            device: 设备 (auto/cpu/cuda/mps)
        """
        if self._initialized:
            return
            
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = model_path
        self._device_preference = device
        self._initialized = True
    
    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            # 确定模型路径
            if self.model_path:
                model_path = Path(self.model_path)
            else:
                # 默认路径
                model_path = Path(__file__).parent.parent.parent / "outputs" / "type_seq2seq" / "final_model"
            
            if not model_path.exists():
                logger.warning(f"Seq2Seq 模型不存在: {model_path}")
                return
            
            logger.info(f"加载 Seq2Seq 模型: {model_path}")
            
            # 确定设备
            if self._device_preference == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self._device_preference
            
            logger.info(f"使用设备: {self.device}")
            
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Seq2Seq 模型加载完成")
            
        except Exception as e:
            logger.error(f"加载 Seq2Seq 模型失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        self._load_model()
        return self.model is not None and self.tokenizer is not None
    
    def _encode_single(self, text: str) -> Seq2SeqResult:
        """
        编码单个文本（内部方法）
        
        Args:
            text: 输入文本
            
        Returns:
            编码结果
        """
        try:
            # 预处理：统一转小写
            processed_text = preprocess_text(text)
            
            # 编码输入
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 生成编码，同时获取分数
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=16,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            # 解码输出
            generated = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            code = generated.strip()
            
            # 计算置信度：从 scores 计算每个 token 的概率
            confidence = self._compute_confidence(outputs)
            
            return Seq2SeqResult(
                original=text,
                code=code,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Seq2Seq 编码失败: {e}")
            return Seq2SeqResult(original=text, code="", confidence=0.0)
    
    def encode(self, text: str, max_length: int = 32) -> Seq2SeqResult:
        """
        编码文本
        
        Args:
            text: 输入文本 (TYPE 或 MATERIAL 描述)
            max_length: 最大生成长度
            
        Returns:
            编码结果
        """
        if not text or not text.strip():
            return Seq2SeqResult(original=text, code="", confidence=0.0)
        
        self._load_model()
        
        if not self.is_available():
            logger.warning("Seq2Seq 模型不可用，返回空编码")
            return Seq2SeqResult(original=text, code="", confidence=0.0)
        
        # 特殊处理：如果包含 "/"，按 "/" 分割分别识别再拼接
        # 例如：S31608/S31603 -> 316/316L
        if '/' in text:
            parts = text.split('/')
            # 过滤空白部分
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                results = [self._encode_single(part) for part in parts]
                codes = [r.code for r in results]
                # 计算平均置信度
                avg_confidence = sum(r.confidence for r in results) / len(results)
                # 去重：如果所有编码相同，只保留一个
                unique_codes = list(dict.fromkeys(codes))  # 保持顺序的去重
                return Seq2SeqResult(
                    original=text,
                    code='/'.join(unique_codes),
                    confidence=avg_confidence
                )
        
        return self._encode_single(text)
    
    def _compute_confidence(self, outputs) -> float:
        """
        从生成的 scores 计算置信度
        
        Args:
            outputs: model.generate() 的输出
            
        Returns:
            置信度 (0-1)
        """
        import torch.nn.functional as F
        
        if not hasattr(outputs, 'scores') or not outputs.scores:
            return 1.0
        
        # 获取生成的 token ids（跳过开始符号）
        generated_ids = outputs.sequences[0].tolist()[1:]
        
        total_log_prob = 0.0
        valid_tokens = 0
        
        for step_idx, score in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            
            token_id = generated_ids[step_idx]
            
            # 跳过特殊 token
            if token_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                continue
            
            # 计算 log softmax，获取选中 token 的 log 概率
            log_probs = F.log_softmax(score[0], dim=-1)
            token_log_prob = log_probs[token_id].item()
            
            total_log_prob += token_log_prob
            valid_tokens += 1
        
        if valid_tokens == 0:
            return 1.0
        
        # 计算平均 log 概率，转换为概率
        avg_log_prob = total_log_prob / valid_tokens
        avg_prob = math.exp(avg_log_prob)
        
        return round(avg_prob, 4)


# 全局实例
_seq2seq_encoder = None


def get_seq2seq_encoder(model_path: str = None, device: str = "auto") -> Seq2SeqEncoder:
    """获取 Seq2Seq 编码器实例"""
    global _seq2seq_encoder
    if _seq2seq_encoder is None:
        _seq2seq_encoder = Seq2SeqEncoder(model_path, device)
    return _seq2seq_encoder
