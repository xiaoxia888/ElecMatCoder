# -*- coding: utf-8 -*-
"""
编码映射管理器
负责管理和维护材料编码映射表

设计原则：
1. 手动配置和自动缓存分离存储
2. 实时读取，不缓存在内存中
3. 只写入自动缓存文件，不修改手动配置
"""

import os
import yaml
import logging
from typing import Dict, Optional, List
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_DIR = Path(__file__).parent / "config"
MANUAL_MAPPING_FILE = CONFIG_DIR / "code_mapping.yaml"    # 用户手动维护
AUTO_CACHE_FILE = CONFIG_DIR / "auto_cache.yaml"          # 系统自动维护


class CodeMappingManager:
    """
    编码映射管理器
    
    特性：
    1. 双文件分离：手动配置(只读) + 自动缓存(读写)
    2. 实时读取：每次查询都从文件读取最新内容
    3. 线程安全：使用锁保护文件写入
    """
    
    def __init__(
        self, 
        manual_config_path: str = None,
        auto_cache_path: str = None
    ):
        """
        初始化映射管理器
        
        Args:
            manual_config_path: 手动配置文件路径
            auto_cache_path: 自动缓存文件路径
        """
        self.manual_config_path = Path(manual_config_path) if manual_config_path else MANUAL_MAPPING_FILE
        self.auto_cache_path = Path(auto_cache_path) if auto_cache_path else AUTO_CACHE_FILE
        self._write_lock = Lock()
        
        # 确保自动缓存文件存在
        self._ensure_auto_cache_exists()
    
    def _ensure_auto_cache_exists(self):
        """确保自动缓存文件存在"""
        if not self.auto_cache_path.exists():
            self.auto_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_auto_cache(self._get_empty_cache())
    
    def _get_empty_cache(self) -> dict:
        """获取空的缓存结构"""
        return {
            'name': {},
            'material': {},
            'type': {},
            'classification_cache': {
                'name': {},
                'material': {},
                'type': {}
            }
        }
    
    def _load_yaml(self, path: Path) -> dict:
        """加载YAML文件"""
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f) or {}
                # logger.debug(f"加载YAML {path.name}: 包含 {len(content.keys())} 个顶层键")
                return content
        except Exception as e:
            logger.error(f"加载YAML失败 {path}: {e}")
            return {}

    def _load_manual_config(self) -> dict:
        """实时加载手动配置文件"""
        return self._load_yaml(self.manual_config_path)

    def _load_auto_cache(self) -> dict:
        """实时加载自动缓存文件"""
        return self._load_yaml(self.auto_cache_path)

    def _save_auto_cache(self, data: dict):
        """保存自动缓存文件（线程安全）"""
        with self._write_lock:
            try:
                self.auto_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.auto_cache_path, 'w', encoding='utf-8') as f:
                    # 添加文件头注释
                    f.write("# ============================================================\n")
                    f.write("# 系统自动缓存 (由系统自动维护，请勿手动修改)\n")
                    f.write("# ============================================================\n\n")
                    yaml.dump(
                        data, 
                        f, 
                        allow_unicode=True, 
                        default_flow_style=False,
                        sort_keys=False
                    )
                logger.debug(f"保存自动缓存: {self.auto_cache_path}")
            except Exception as e:
                logger.error(f"保存自动缓存失败: {e}")

    def _get_merged_data(self) -> dict:
        """
        实时获取合并后的数据（手动配置 + 自动缓存）
        手动配置优先级更高
        """
        manual = self._load_manual_config()
        auto = self._load_auto_cache()
        
        # 记录加载情况
        manual_names = manual.get('name', {})
        if len(manual_names) <= 1:
            logger.warning(f"[映射加载] 警告：手动映射(name)条数异常: {len(manual_names)} 条")
        else:
            logger.info(f"[映射加载] 手动映射(name): {len(manual_names)} 条")

        # 合并数据，手动配置优先
        merged = {
            'name': {**auto.get('name', {}), **manual.get('name', {})},
            'material': {**auto.get('material', {}), **manual.get('material', {})},
            'type': {**auto.get('type', {}), **manual.get('type', {})},
            'classification_cache': {
                'name': {
                    **auto.get('classification_cache', {}).get('name', {}),
                    **manual.get('classification_cache', {}).get('name', {})
                },
                'material': {
                    **auto.get('classification_cache', {}).get('material', {}),
                    **manual.get('classification_cache', {}).get('material', {})
                },
                'type': {
                    **auto.get('classification_cache', {}).get('type', {}),
                    **manual.get('classification_cache', {}).get('type', {})
                },
            },
            'inference_rules': manual.get('inference_rules', {}),
            'defaults': manual.get('defaults', {})
        }
        
        return merged
    
    # ==================== 编码查询（实时读取） ====================
    
    def get_code(self, entity_type: str, category: str) -> Optional[str]:
        """
        获取标准大类的编码（实时读取）
        
        Args:
            entity_type: 实体类型 (name, material, type)
            category: 标准大类名称
            
        Returns:
            编码，如果不存在返回None
        """
        data = self._get_merged_data()
        mapping = data.get(entity_type, {})
        return mapping.get(category)
    
    def get_all_codes(self, entity_type: str) -> Dict[str, str]:
        """获取某类型的所有编码映射（实时读取）"""
        data = self._get_merged_data()
        return data.get(entity_type, {}).copy()
    
    def get_all_categories(self, entity_type: str) -> List[str]:
        """获取某类型的所有标准大类（实时读取）"""
        data = self._get_merged_data()
        return list(data.get(entity_type, {}).keys())
    
    def get_all_known_terms(self, entity_type: str) -> List[str]:
        """获取某类型的所有已知词汇，包括标准大类和缓存的变体（实时读取）"""
        data = self._get_merged_data()
        # 标准大类
        categories = list(data.get(entity_type, {}).keys())
        # 缓存中的变体名
        cached_terms = list(data.get('classification_cache', {}).get(entity_type, {}).keys())
        return list(set(categories + cached_terms))
    
    def add_code(self, entity_type: str, category: str, code: str):
        """
        添加新的编码映射（写入自动缓存）
        
        Args:
            entity_type: 实体类型
            category: 标准大类
            code: 编码
        """
        auto_cache = self._load_auto_cache()
        
        if entity_type not in auto_cache:
            auto_cache[entity_type] = {}
        
        auto_cache[entity_type][category] = code
        logger.info(f"[自动缓存] 添加编码: {entity_type}.{category} -> {code}")
        
        self._save_auto_cache(auto_cache)
    
    def code_exists(self, entity_type: str, code: str) -> bool:
        """检查编码是否已被使用（实时读取）"""
        data = self._get_merged_data()
        mapping = data.get(entity_type, {})
        return code in mapping.values()
    
    # ==================== 分类缓存 ====================
    
    def get_cached_category(self, entity_type: str, variant: str) -> Optional[str]:
        """从缓存中获取变体对应的标准大类（实时读取）"""
        data = self._get_merged_data()
        cache = data.get('classification_cache', {}).get(entity_type, {})
        return cache.get(variant)
    
    def add_to_cache(self, entity_type: str, variant: str, category: str):
        """
        添加变体到分类缓存（写入自动缓存）
        
        Args:
            entity_type: 实体类型
            variant: 变体名称
            category: 标准大类
        """
        auto_cache = self._load_auto_cache()
        
        if 'classification_cache' not in auto_cache:
            auto_cache['classification_cache'] = {}
        if entity_type not in auto_cache['classification_cache']:
            auto_cache['classification_cache'][entity_type] = {}
        
        auto_cache['classification_cache'][entity_type][variant] = category
        logger.info(f"[自动缓存] 添加分类: {entity_type}.{variant} -> {category}")
        
        self._save_auto_cache(auto_cache)
    
    # ==================== 推断规则（只读） ====================
    
    def get_inference_keys(self, rule_type: str) -> list:
        """获取推断规则的所有键（实时读取）
        
        Args:
            rule_type: 规则类型，如 'name_to_type', 'name_to_material'
        
        Returns:
            规则键列表，按长度降序排列（优先匹配更长的关键词）
        """
        data = self._get_merged_data()
        rules = data.get('inference_rules', {}).get(rule_type, {})
        # 按长度降序排序，确保优先匹配更长的关键词（如"电信桥架"优先于"桥架"）
        return sorted(rules.keys(), key=len, reverse=True)
    
    def get_inferred_type(self, name: str) -> Optional[str]:
        """根据名称推断类型（实时读取）"""
        data = self._get_merged_data()
        rules = data.get('inference_rules', {}).get('name_to_type', {})
        return rules.get(name)
    
    def get_inferred_material(self, name: str) -> Optional[str]:
        """根据名称推断材质（实时读取）"""
        data = self._get_merged_data()
        rules = data.get('inference_rules', {}).get('name_to_material', {})
        return rules.get(name)
    
    # ==================== 默认值（只读） ====================
    
    def get_default(self, entity_type: str) -> Optional[str]:
        """获取默认值（实时读取）"""
        data = self._get_merged_data()
        return data.get('defaults', {}).get(entity_type)
    
    # ==================== 工具方法 ====================
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息（实时读取）"""
        data = self._get_merged_data()
        manual = self._load_manual_config()
        auto = self._load_auto_cache()
        
        return {
            'total_name_codes': len(data.get('name', {})),
            'total_material_codes': len(data.get('material', {})),
            'total_type_codes': len(data.get('type', {})),
            'manual_name_codes': len(manual.get('name', {})),
            'manual_material_codes': len(manual.get('material', {})),
            'auto_name_codes': len(auto.get('name', {})),
            'auto_material_codes': len(auto.get('material', {})),
            'auto_name_cache': len(auto.get('classification_cache', {}).get('name', {})),
            'auto_material_cache': len(auto.get('classification_cache', {}).get('material', {})),
        }
    
    def clear_auto_cache(self):
        """清空自动缓存"""
        self._save_auto_cache(self._get_empty_cache())
        logger.info("已清空自动缓存")


# 全局单例
_manager_instance = None


def get_mapping_manager() -> CodeMappingManager:
    """获取映射管理器单例"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CodeMappingManager()
    return _manager_instance
