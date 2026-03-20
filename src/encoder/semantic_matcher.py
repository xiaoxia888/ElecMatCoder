# -*- coding: utf-8 -*-
"""
语义相似度匹配器
使用 Sentence-Transformers + Faiss 实现高效的语义相似度搜索
"""

import logging
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from ..config import get_semantic_config

logger = logging.getLogger(__name__)

# 延迟导入，避免启动时加载大型模型
_sentence_transformer = None
_faiss = None


def _get_sentence_transformer():
    """延迟加载 sentence-transformers"""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
    return _sentence_transformer


def _get_faiss():
    """延迟加载 faiss"""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            logger.warning("Faiss 未安装，使用 numpy 进行相似度计算（较慢）")
            _faiss = None
    return _faiss


@dataclass
class MatchResult:
    """匹配结果"""
    matched_name: str           # 匹配到的标准名称
    code: str                   # 对应的编码
    similarity: float           # 相似度分数 (0-1)
    is_exact_match: bool        # 是否精确匹配
    need_review: bool           # 是否需要人工审核
    candidates: List[Tuple[str, str, float]] = None  # 候选列表 [(name, code, score), ...]


class SemanticMatcher:
    """
    语义相似度匹配器
    
    支持两种匹配模式：
    1. 精确匹配：直接查找映射表
    2. 语义匹配：使用向量相似度找最接近的标准名称
    """
    
    def __init__(self):
        """初始化匹配器"""
        # 配置目录
        self.config_dir = Path(__file__).parent / "config"
        
        # 加载映射表
        self.mapping = self._load_mapping()
        
        # 模型和索引（延迟初始化）
        self._model = None
        self._indexes: Dict[str, Any] = {}  # {field_type: faiss_index}
        self._embeddings: Dict[str, Dict[str, np.ndarray]] = {}  # {field_type: {name: embedding}}
        self._name_list: Dict[str, List[str]] = {}  # {field_type: [name1, name2, ...]}
        
        # 从平台配置中读取语义匹配参数
        semantic_config = get_semantic_config()
        self.threshold = semantic_config.get('similarity_threshold', 0.9)
        self.model_name = semantic_config.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.local_model_path = semantic_config.get('local_model_path', '')
        self.top_k = semantic_config.get('top_k', 5)
        self.cache_embeddings = semantic_config.get('cache_embeddings', True)
        self.offline_mode = semantic_config.get('offline_mode', False)
        
        # 设置离线模式
        if self.offline_mode:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    def _load_mapping(self) -> dict:
        """加载映射表"""
        mapping_path = self.config_dir / "pipe_code_mapping.yaml"
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载映射表失败: {e}")
            return {}
    
    def _get_model(self):
        """获取或初始化模型"""
        if self._model is None:
            SentenceTransformer = _get_sentence_transformer()
            
            # 优先使用本地模型路径
            model_path = self.model_name
            if self.local_model_path:
                # 计算项目根目录
                project_root = Path(__file__).parent.parent.parent
                local_path = project_root / self.local_model_path
                if local_path.exists():
                    model_path = str(local_path)
                    logger.info(f"正在从本地加载语义模型: {local_path}")
                else:
                    logger.warning(f"本地模型路径不存在: {local_path}，将从 HuggingFace 下载")
                    logger.info(f"正在加载语义模型: {self.model_name}")
            else:
                logger.info(f"正在加载语义模型: {self.model_name}")
            
            self._model = SentenceTransformer(model_path)
            logger.info("语义模型加载完成")
        return self._model
    
    def _build_index(self, field_type: str):
        """
        为指定字段类型构建向量索引
        
        Args:
            field_type: 字段类型 (TYPE, MATERIAL 等)
        """
        if field_type in self._indexes:
            return
        
        mapping_data = self.mapping.get(field_type, {})
        if not mapping_data or field_type.startswith('_'):
            return
        
        # 收集所有需要编码的文本
        texts = []
        names = []
        
        for name, info in mapping_data.items():
            if isinstance(info, dict):
                # 新格式: {code: "X", aliases: [...]}
                texts.append(name)
                names.append(name)
                
                aliases = info.get('aliases', [])
                for alias in aliases:
                    texts.append(alias)
                    names.append(name)  # 别名指向标准名称
            else:
                # 旧格式: 直接是编码
                texts.append(name)
                names.append(name)
        
        if not texts:
            return
        
        # 计算向量
        model = self._get_model()
        logger.info(f"正在为 {field_type} 构建向量索引 ({len(texts)} 条)")
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # 存储名称映射
        self._name_list[field_type] = names
        
        # 尝试使用 Faiss
        faiss = _get_faiss()
        if faiss is not None:
            # 创建 Faiss 索引（使用内积，因为向量已归一化）
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype(np.float32))
            self._indexes[field_type] = index
            logger.info(f"{field_type} Faiss 索引构建完成")
        else:
            # 使用 numpy
            self._embeddings[field_type] = {
                '_matrix': embeddings,
                '_names': names
            }
            logger.info(f"{field_type} numpy 索引构建完成")
    
    def exact_match(self, field_type: str, value: str) -> Optional[MatchResult]:
        """
        精确匹配
        
        Args:
            field_type: 字段类型
            value: 待匹配的值
            
        Returns:
            匹配结果，未找到返回 None
        """
        mapping_data = self.mapping.get(field_type, {})
        if not mapping_data:
            return None
        
        # 先尝试直接匹配
        if value in mapping_data:
            info = mapping_data[value]
            if isinstance(info, dict):
                code = str(info.get('code', ''))
            else:
                code = str(info)
            
            return MatchResult(
                matched_name=value,
                code=code,
                similarity=1.0,
                is_exact_match=True,
                need_review=False
            )
        
        # 尝试匹配别名
        value_lower = value.lower()
        for name, info in mapping_data.items():
            if isinstance(info, dict):
                aliases = info.get('aliases', [])
                for alias in aliases:
                    if str(alias).lower() == value_lower:
                        return MatchResult(
                            matched_name=name,
                            code=str(info.get('code', '')),
                            similarity=1.0,
                            is_exact_match=True,
                            need_review=False
                        )
        
        return None
    
    def _preprocess_for_matching(self, field_type: str, value: str) -> str:
        """
        预处理待匹配的值
        
        针对 TYPE 字段，将符号替换为空格，便于匹配中英文混合格式
        如: 同心异径管|Concentric Reducer → 同心异径管 Concentric Reducer
        
        Args:
            field_type: 字段类型
            value: 原始值
            
        Returns:
            预处理后的值
        """
        import re
        
        if field_type == 'TYPE':
            # 将常见分隔符替换为空格
            # | / \ , ; 等
            processed = re.sub(r'[|/\\,;]', ' ', value)
            # 合并多个空格为一个
            processed = re.sub(r'\s+', ' ', processed).strip()
            return processed
        
        return value
    
    def semantic_match(
        self, 
        field_type: str, 
        value: str,
        threshold: float = None
    ) -> MatchResult:
        """
        语义相似度匹配
        
        Args:
            field_type: 字段类型
            value: 待匹配的值
            threshold: 相似度阈值（可选，默认使用配置）
            
        Returns:
            匹配结果
        """
        threshold = threshold if threshold is not None else self.threshold
        
        # 先尝试精确匹配
        exact_result = self.exact_match(field_type, value)
        if exact_result:
            return exact_result
        
        # 构建索引（如果尚未构建）
        self._build_index(field_type)
        
        # 检查是否有索引数据
        if field_type not in self._indexes and field_type not in self._embeddings:
            return MatchResult(
                matched_name=value,
                code="",
                similarity=0.0,
                is_exact_match=False,
                need_review=True,
                candidates=[]
            )
        
        # 预处理查询值（TYPE 字段将符号替换为空格）
        processed_value = self._preprocess_for_matching(field_type, value)
        
        # 计算查询向量
        model = self._get_model()
        query_embedding = model.encode([processed_value], convert_to_numpy=True, normalize_embeddings=True)
        
        # 搜索最相似的项
        faiss = _get_faiss()
        if faiss is not None and field_type in self._indexes:
            # 使用 Faiss 搜索
            index = self._indexes[field_type]
            scores, indices = index.search(query_embedding.astype(np.float32), self.top_k)
            
            names = self._name_list[field_type]
            candidates = []
            seen = set()
            
            for score, idx in zip(scores[0], indices[0]):
                name = names[idx]
                if name in seen:
                    continue
                seen.add(name)
                
                info = self.mapping[field_type].get(name, {})
                code = str(info.get('code', '')) if isinstance(info, dict) else str(info)
                candidates.append((name, code, float(score)))
        else:
            # 使用 numpy
            embeddings_data = self._embeddings.get(field_type, {})
            matrix = embeddings_data.get('_matrix')
            names = embeddings_data.get('_names', [])
            
            if matrix is None:
                return MatchResult(
                    matched_name=value,
                    code="",
                    similarity=0.0,
                    is_exact_match=False,
                    need_review=True,
                    candidates=[]
                )
            
            # 计算余弦相似度
            scores = np.dot(matrix, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            
            candidates = []
            seen = set()
            for idx in top_indices:
                name = names[idx]
                if name in seen:
                    continue
                seen.add(name)
                
                info = self.mapping[field_type].get(name, {})
                code = str(info.get('code', '')) if isinstance(info, dict) else str(info)
                candidates.append((name, code, float(scores[idx])))
        
        # 获取最佳匹配
        if candidates:
            best_name, best_code, best_score = candidates[0]
            need_review = best_score < threshold
            
            return MatchResult(
                matched_name=best_name,
                code=best_code,
                similarity=best_score,
                is_exact_match=False,
                need_review=need_review,
                candidates=candidates
            )
        
        return MatchResult(
            matched_name=value,
            code="",
            similarity=0.0,
            is_exact_match=False,
            need_review=True,
            candidates=[]
        )
    
    def match(
        self, 
        field_type: str, 
        value: str, 
        use_semantic: bool = True,
        threshold: float = None
    ) -> MatchResult:
        """
        统一匹配接口
        
        Args:
            field_type: 字段类型
            value: 待匹配的值
            use_semantic: 是否使用语义匹配
            threshold: 相似度阈值
            
        Returns:
            匹配结果
        """
        if not value or not value.strip():
            return MatchResult(
                matched_name="",
                code="",
                similarity=0.0,
                is_exact_match=False,
                need_review=True,
                candidates=[]
            )
        
        value = value.strip()
        
        if use_semantic:
            return self.semantic_match(field_type, value, threshold)
        else:
            result = self.exact_match(field_type, value)
            if result:
                return result
            else:
                return MatchResult(
                    matched_name=value,
                    code="",
                    similarity=0.0,
                    is_exact_match=False,
                    need_review=True,
                    candidates=[]
                )
    
    def reload_mapping(self):
        """重新加载映射表（用于热更新）"""
        self.mapping = self._load_mapping()
        # 清空索引，下次匹配时重建
        self._indexes.clear()
        self._embeddings.clear()
        self._name_list.clear()
        logger.info("映射表已重新加载")
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold
    
    def set_threshold(self, threshold: float):
        """设置阈值"""
        self.threshold = threshold


# 单例模式
_matcher_instance: Optional[SemanticMatcher] = None


def get_semantic_matcher() -> SemanticMatcher:
    """获取语义匹配器单例"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = SemanticMatcher()
    return _matcher_instance
