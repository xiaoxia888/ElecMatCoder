"""
TYPE 字段分类模型评估脚本
测试模型的准确率和泛化能力
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TypeClassifier:
    """TYPE 编码分类器"""
    
    def __init__(self, model_dir: str):
        """
        初始化分类器
        
        Args:
            model_dir: 模型目录
        """
        logger.info(f"加载模型: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 加载标签映射
        with open(Path(model_dir) / "labels.json", 'r', encoding='utf-8') as f:
            label_info = json.load(f)
        
        self.id2label = {int(k): v for k, v in label_info['id2label'].items()}
        self.label2id = label_info['label2id']
        
        logger.info(f"加载 {len(self.id2label)} 个类别")
    
    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        预测 TYPE 编码
        
        Args:
            text: 输入文本
            top_k: 返回前 k 个候选
            
        Returns:
            预测结果字典
        """
        inputs = self.tokenizer(
            text,
            max_length=64,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        # 获取 top-k 预测
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(self.id2label)))
        
        candidates = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            candidates.append({
                'label': self.id2label[idx],
                'probability': float(prob)
            })
        
        best = candidates[0]
        
        return {
            'input': text,
            'prediction': best['label'],
            'probability': best['probability'],
            'candidates': candidates
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        return [self.predict(text) for text in texts]


def evaluate_on_dataset(classifier: TypeClassifier, data_file: str) -> Dict:
    """
    在数据集上评估模型
    """
    # 加载数据
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    logger.info(f"评估 {len(samples)} 条数据")
    
    # 评估
    correct = 0
    errors = []
    
    for sample in samples:
        text = sample['text']
        expected = sample['label_name']
        
        result = classifier.predict(text)
        prediction = result['prediction']
        
        if prediction == expected:
            correct += 1
        else:
            errors.append({
                'input': text,
                'expected': expected,
                'prediction': prediction,
                'probability': result['probability'],
                'candidates': result['candidates'][:3]
            })
    
    accuracy = correct / len(samples)
    
    logger.info(f"准确率: {accuracy:.4f} ({correct}/{len(samples)})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(samples),
        'errors': errors[:20]
    }


def test_generalization(classifier: TypeClassifier):
    """
    测试模型泛化能力
    """
    logger.info("测试泛化能力...")
    
    # 测试用例
    test_cases = [
        # 管子变体
        ("管子", "P"),
        ("钢管", "P"),
        ("无缝钢管", "P"),
        ("不锈钢管", "P"),
        ("无缝不锈钢管", "P"),
        ("焊接钢管", "P"),
        
        # 弯头变体
        ("90度弯头", "90EL"),
        ("90°弯头", "90EL"),
        ("九十度弯头", "90EL"),
        ("直角弯头", "90EL"),
        
        # 法兰变体
        ("法兰", "F"),
        ("对焊法兰", "F"),
        ("焊接法兰", "F"),
        ("WN法兰", "F"),
        
        # 三通变体
        ("三通", "T"),
        ("等径三通", "T"),
        ("直通", "T"),
        
        # 半管接头变体
        ("半管接头", "HC"),
        ("HALF CPLG", "HC"),
        ("半接头", "HC"),
        
        # 偏心异径管变体
        ("偏心异径管", "RE"),
        ("偏心大小头", "RE"),
        ("ECC REDUCER", "RE"),
        ("REDUCER ECC", "RE"),
    ]
    
    correct = 0
    results = []
    
    for input_text, expected in test_cases:
        result = classifier.predict(input_text)
        prediction = result['prediction']
        probability = result['probability']
        is_correct = prediction == expected
        
        if is_correct:
            correct += 1
        
        results.append({
            'input': input_text,
            'expected': expected,
            'prediction': prediction,
            'probability': probability,
            'correct': is_correct
        })
        
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} '{input_text}' → '{prediction}' ({probability:.2%}) (期望: '{expected}')")
    
    accuracy = correct / len(test_cases)
    logger.info(f"\n泛化测试准确率: {accuracy:.4f} ({correct}/{len(test_cases)})")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_cases),
        'results': results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 TYPE 分类模型")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    parser.add_argument("--data_file", type=str, help="评估数据文件")
    parser.add_argument("--test_generalization", action="store_true", help="测试泛化能力")
    
    args = parser.parse_args()
    
    # 初始化分类器
    classifier = TypeClassifier(args.model_dir)
    
    # 在数据集上评估
    if args.data_file:
        results = evaluate_on_dataset(classifier, args.data_file)
        
        output_file = Path(args.model_dir) / "eval_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_file}")
    
    # 测试泛化能力
    if args.test_generalization:
        gen_results = test_generalization(classifier)
        
        output_file = Path(args.model_dir) / "generalization_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gen_results, f, ensure_ascii=False, indent=2)
        logger.info(f"泛化测试结果已保存到: {output_file}")
