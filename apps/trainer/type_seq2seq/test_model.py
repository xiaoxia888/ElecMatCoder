#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TYPE Seq2Seq 模型测试脚本

使用方法:
    python apps/trainer/type_seq2seq/test_model.py "无缝钢管"
    python apps/trainer/type_seq2seq/test_model.py "45度弯头"
    python apps/trainer/type_seq2seq/test_model.py  # 交互模式
"""

import sys
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 模型路径
MODEL_PATH = PROJECT_ROOT / "outputs/type_seq2seq_v2/final_model"


def preprocess_text(text: str) -> str:
    """预处理输入文本：清理多余空格（保留 / 分隔符）"""
    # 只清理多余空格，不替换 / 等分隔符
    processed = re.sub(r'\s+', ' ', text).strip()
    return processed


class TypeSeq2SeqTester:
    """TYPE Seq2Seq 模型测试器"""
    
    def __init__(self, model_path: str = None):
        model_path = model_path or str(MODEL_PATH)
        
        print(f"加载模型: {model_path}")
        
        # 设备
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成!\n")
    
    def _predict_single(self, text: str, num_beams: int = 4) -> str:
        """单个文本的预测（内部方法）"""
        # 预处理
        processed = preprocess_text(text)
        
        # 编码
        inputs = self.tokenizer(
            processed,
            return_tensors='pt',
            max_length=64,
            truncation=True
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=16,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # 解码
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code.strip()
    
    def predict(self, text: str, num_beams: int = 4) -> str:
        """
        预测类型编码
        
        Args:
            text: 类型名称，如 "45度弯头"、"无缝钢管"
            num_beams: beam search 宽度
            
        Returns:
            类型编码，如 "45EL"、"P"
        """
        # 特殊处理：如果包含 "/"，按 "/" 分割分别识别再拼接
        # 例如：S31608/S31603 -> 316/316L
        if '/' in text:
            parts = text.split('/')
            # 过滤空白部分
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                codes = [self._predict_single(part, num_beams) for part in parts]
                return '/'.join(codes)
        
        return self._predict_single(text, num_beams)
    
    def test(self, text: str) -> None:
        """测试并打印结果"""
        code = self.predict(text)
        print(f"  输入: {text}")
        print(f"  编码: {code}")
        print()


def main():
    tester = TypeSeq2SeqTester()
    
    if len(sys.argv) > 1:
        # 命令行参数模式
        for text in sys.argv[1:]:
            tester.test(text)
    else:
        # 交互模式
        print("=" * 50)
        print("TYPE 编码测试 (输入 'q' 退出)")
        print("=" * 50)
        
        while True:
            try:
                text = input("\n请输入类型名称: ").strip()
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                if not text:
                    continue
                tester.test(text)
            except KeyboardInterrupt:
                break
        
        print("\n测试结束!")


if __name__ == '__main__':
    main()
