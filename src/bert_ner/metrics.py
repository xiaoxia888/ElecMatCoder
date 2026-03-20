"""
NER评估指标
"""

from typing import List, Dict
from collections import defaultdict


def compute_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str]
) -> Dict:
    """
    计算NER评估指标
    
    Args:
        predictions: 预测的标签ID列表
        labels: 真实的标签ID列表
        id2label: ID到标签的映射
        
    Returns:
        包含precision, recall, f1的字典
    """
    # 将ID转换为标签
    pred_labels = [[id2label[p] for p in pred] for pred in predictions]
    true_labels = [[id2label[l] for l in label] for label in labels]
    
    # 提取实体
    pred_entities = [extract_entities(pred) for pred in pred_labels]
    true_entities = [extract_entities(true) for true in true_labels]
    
    # 统计
    total_pred = 0
    total_true = 0
    total_correct = 0
    
    # 按实体类型统计
    type_stats = defaultdict(lambda: {'pred': 0, 'true': 0, 'correct': 0})
    
    for pred, true in zip(pred_entities, true_entities):
        pred_set = set(pred)
        true_set = set(true)
        
        total_pred += len(pred_set)
        total_true += len(true_set)
        total_correct += len(pred_set & true_set)
        
        # 按类型统计
        for entity in pred_set:
            entity_type = entity[0]
            type_stats[entity_type]['pred'] += 1
            if entity in true_set:
                type_stats[entity_type]['correct'] += 1
        
        for entity in true_set:
            entity_type = entity[0]
            type_stats[entity_type]['true'] += 1
    
    # 计算整体指标
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_pred': total_pred,
        'total_true': total_true,
        'total_correct': total_correct,
        'type_stats': dict(type_stats)
    }


def extract_entities(tags: List[str]) -> List[tuple]:
    """
    从BIO标签序列中提取实体
    
    Args:
        tags: 标签序列，如 ['O', 'B-NAME', 'I-NAME', 'O']
        
    Returns:
        实体列表，每个元素是 (entity_type, start, end)
    """
    entities = []
    current_entity = None
    current_type = None
    start = None
    
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # 保存上一个实体
            if current_entity is not None:
                entities.append((current_type, start, i))
            # 开始新实体
            current_type = tag[2:]
            current_entity = True
            start = i
        elif tag.startswith('I-'):
            tag_type = tag[2:]
            if current_type != tag_type:
                # I标签与当前实体类型不匹配，结束当前实体
                if current_entity is not None:
                    entities.append((current_type, start, i))
                current_entity = None
                current_type = None
                start = None
        else:
            # O标签，结束当前实体
            if current_entity is not None:
                entities.append((current_type, start, i))
            current_entity = None
            current_type = None
            start = None
    
    # 处理最后一个实体
    if current_entity is not None:
        entities.append((current_type, start, len(tags)))
    
    return entities


def classification_report(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str]
) -> str:
    """
    生成详细的分类报告
    
    Returns:
        格式化的报告字符串
    """
    metrics = compute_metrics(predictions, labels, id2label)
    
    lines = []
    lines.append("=" * 60)
    lines.append("NER 评估报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"整体指标:")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall:    {metrics['recall']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1']:.4f}")
    lines.append("")
    lines.append(f"实体统计:")
    lines.append(f"  预测实体数: {metrics['total_pred']}")
    lines.append(f"  真实实体数: {metrics['total_true']}")
    lines.append(f"  正确实体数: {metrics['total_correct']}")
    lines.append("")
    
    if metrics['type_stats']:
        lines.append("各类型详细指标:")
        lines.append("-" * 60)
        lines.append(f"{'类型':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 60)
        
        for entity_type, stats in sorted(metrics['type_stats'].items()):
            p = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
            r = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            lines.append(f"{entity_type:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f} {stats['true']:>10}")
        
        lines.append("-" * 60)
    
    return "\n".join(lines)

