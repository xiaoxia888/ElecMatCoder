# ============================================
# Seq2Seq TYPE编码预测器
# ============================================

from typing import Optional, List, Dict
from .model import TypeEncoder, preprocess_text


class TypeEncoderPredictor:
    """TYPE编码预测器（封装，便于在编码流程中使用）"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 16,
        fallback_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备
            num_beams: beam search 宽度
            max_length: 生成最大长度
            fallback_mapping: 回退映射表（模型生成失败时使用）
        """
        self.encoder = TypeEncoder(
            model_path,
            device=device,
            num_beams=num_beams,
            max_length=max_length
        )
        self.fallback_mapping = fallback_mapping or {}
        
        # 缓存已生成的编码
        self._cache: Dict[str, str] = {}
    
    def predict(self, type_name: str, use_cache: bool = True) -> str:
        """
        预测类型编码
        
        Args:
            type_name: 类型名称
            use_cache: 是否使用缓存
            
        Returns:
            类型编码
        """
        # 预处理
        processed = preprocess_text(type_name)
        
        # 检查缓存
        if use_cache and processed in self._cache:
            return self._cache[processed]
        
        # 先检查回退映射表（精确匹配）
        if processed in self.fallback_mapping:
            code = self.fallback_mapping[processed]
            self._cache[processed] = code
            return code
        
        # 使用模型生成
        try:
            code = self.encoder.encode(processed, preprocess=False)
            
            # 验证生成的编码（应该是大写字母和数字）
            if code and code.replace('-', '').replace('_', '').isalnum():
                self._cache[processed] = code
                return code
            
        except Exception as e:
            print(f"模型生成失败: {e}")
        
        # 生成失败，返回空
        return ''
    
    def predict_batch(
        self, 
        type_names: List[str], 
        use_cache: bool = True
    ) -> List[str]:
        """批量预测"""
        results = []
        uncached = []
        uncached_indices = []
        
        for i, name in enumerate(type_names):
            processed = preprocess_text(name)
            
            if use_cache and processed in self._cache:
                results.append(self._cache[processed])
            elif processed in self.fallback_mapping:
                code = self.fallback_mapping[processed]
                self._cache[processed] = code
                results.append(code)
            else:
                results.append(None)  # 占位
                uncached.append(processed)
                uncached_indices.append(i)
        
        # 批量生成未缓存的
        if uncached:
            codes = self.encoder.encode_batch(uncached, preprocess=False)
            for idx, code in zip(uncached_indices, codes):
                processed = preprocess_text(type_names[idx])
                if code and code.replace('-', '').replace('_', '').isalnum():
                    self._cache[processed] = code
                    results[idx] = code
                else:
                    results[idx] = ''
        
        return results
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
