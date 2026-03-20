"""
词级别多任务NER训练和预测模块

使用词级别分词（而非字符级别），通过 offset_mapping 动态对齐标签
优势：
1. Token数量更少，训练更快
2. 保留词级别语义，避免字符级别的误识别
3. 数据集格式不变，只修改处理逻辑
"""

from .dataset import WordLevelMultiTaskDataset
from .predictor import WordLevelPipePredictor

__all__ = ['WordLevelMultiTaskDataset', 'WordLevelPipePredictor']
