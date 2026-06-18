"""
配置模块
"""

import os
import importlib.util
import copy
from pathlib import Path
from typing import Dict, Any
import yaml

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置文件目录
CONFIG_DIR = Path(__file__).parent

# 配置缓存
_config_cache: Dict[str, Any] = {}


def _merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(filepath: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        filepath: 配置文件路径（相对于 CONFIG_DIR 或绝对路径）
        use_cache: 是否使用缓存
        
    Returns:
        配置字典
    """
    # 处理路径
    if os.path.isabs(filepath):
        config_path = Path(filepath)
    else:
        config_path = CONFIG_DIR / filepath
    
    cache_key = str(config_path)
    
    if use_cache and cache_key in _config_cache:
        return _config_cache[cache_key]
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    if use_cache:
        _config_cache[cache_key] = config
    
    return config


def get_platform_config() -> Dict[str, Any]:
    """获取平台配置"""
    base_config = load_yaml_config("platform_config.yaml")
    config_file = os.environ.get("PLATFORM_CONFIG", "").strip()
    platform_env = os.environ.get("PLATFORM_ENV", "").strip()

    if not config_file and platform_env:
        config_file = f"platform_config.{platform_env}.yaml"
    if not config_file:
        return base_config

    override_config = load_yaml_config(config_file)
    return _merge_nested_dict(base_config, override_config)


def get_semantic_config() -> Dict[str, Any]:
    """获取语义匹配配置"""
    config = get_platform_config()
    return config.get("semantic_matching", {})


def get_ner_config() -> Dict[str, Any]:
    """获取NER识别配置"""
    config = get_platform_config()
    return config.get("ner", {})


def reload_config():
    """重新加载所有配置（清除缓存）"""
    global _config_cache
    _config_cache = {}


# 直接导入 label_config.py，避免循环导入
_label_config_path = os.path.join(PROJECT_ROOT, 'src', 'bert_ner', 'config', 'label_config.py')
_spec = importlib.util.spec_from_file_location('label_config', _label_config_path)
_label_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_label_config)

# 导出 label_config
get_platform = _label_config.get_platform
get_platform_labels = _label_config.get_platform_labels
get_bio_labels = _label_config.get_bio_labels
get_all_platforms = _label_config.get_all_platforms
list_platforms = _label_config.list_platforms
get_normalization_labels = _label_config.get_normalization_labels
add_normalization_label = _label_config.add_normalization_label
PLATFORMS = _label_config.PLATFORMS
CABLE_PLATFORM = _label_config.CABLE_PLATFORM
PIPE_PLATFORM = _label_config.PIPE_PLATFORM
NORMALIZATION_LABELS = _label_config.NORMALIZATION_LABELS
LabelInfo = _label_config.LabelInfo
PlatformConfig = _label_config.PlatformConfig

__all__ = [
    # 标签配置
    'get_platform',
    'get_platform_labels',
    'get_bio_labels',
    'get_all_platforms',
    'list_platforms',
    'get_normalization_labels',
    'add_normalization_label',
    'PLATFORMS',
    'CABLE_PLATFORM',
    'PIPE_PLATFORM',
    'NORMALIZATION_LABELS',
    'LabelInfo',
    'PlatformConfig',
    'PROJECT_ROOT',
    # 平台配置
    'CONFIG_DIR',
    'load_yaml_config',
    'get_platform_config',
    'get_semantic_config',
    'get_ner_config',
    'reload_config',
]
