"""
标签配置模块

统一管理不同平台的标签定义
- 前端标注使用
- 后端API返回
- 模型训练使用
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LabelInfo:
    """单个标签信息"""
    value: str          # 标签值，如 "NAME", "TYPE"
    label: str          # 显示名称，如 "名称", "种类"
    color: str          # 颜色，如 "#1565c0"
    key: str            # 快捷键，如 "1"
    description: str = ""  # 描述说明


@dataclass
class PlatformConfig:
    """平台配置"""
    name: str                    # 平台名称
    display_name: str            # 显示名称
    icon: str                    # 图标
    labels: List[LabelInfo]      # 标签列表
    
    def get_label_values(self) -> List[str]:
        """获取所有标签值（不含O）"""
        return [l.value for l in self.labels if l.value != 'O']
    
    def get_bio_labels(self) -> List[str]:
        """获取BIO格式的标签列表（用于训练）"""
        bio_labels = ['O']
        for label in self.labels:
            if label.value != 'O':
                bio_labels.append(f'B-{label.value}')
                bio_labels.append(f'I-{label.value}')
        return bio_labels
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于API返回）"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'icon': self.icon,
            'labels': [
                {
                    'value': l.value,
                    'label': l.label,
                    'color': l.color,
                    'key': l.key,
                    'description': l.description
                }
                for l in self.labels
            ]
        }


# ==================== 平台配置定义 ====================

# 电缆/桥架平台
CABLE_PLATFORM = PlatformConfig(
    name='cable',
    display_name='电缆/桥架',
    icon='🔌',
    labels=[
        LabelInfo('O', '无标签', '#3a3a4a', '0', '不属于任何实体'),
        LabelInfo('NAME', '名称', '#1565c0', '1', '材料的通用名称，如电力电缆、控制电缆、桥架'),
        LabelInfo('MATERIAL', '材质', '#7b1fa2', '2', '材料的主要构成材质，如铜芯、铝芯、铝合金'),
        LabelInfo('TYPE', '类型', '#c62828', '3', '电缆类型或桥架结构类型，如阻燃、耐火、梯级式'),
        LabelInfo('ARMOR', '铠装', '#ef6c00', '4', '电缆铠装结构，如钢带铠装、22、32'),
        LabelInfo('FEATURE', '特征', '#2e7d32', '5', '材料特性、绝缘/护套材料、型号代码，如YJV、KVV'),
        LabelInfo('VOLTAGE', '电压', '#00838f', '6', '额定电压等级，如0.6/1kV、8.7/15KV'),
        LabelInfo('SPEC', '规格', '#ad1457', '7', '尺寸、芯数、截面积等参数，如3×95、600×150'),
    ]
)

# 管道平台
PIPE_PLATFORM = PlatformConfig(
    name='pipe',
    display_name='管道',
    icon='🔧',
    labels=[
        LabelInfo('O', '无标签', '#3a3a4a', '0', '不属于任何实体'),
        LabelInfo('TYPE', '种类', '#1565c0', '1', '产品类型，如法兰、弯头、三通、WN-RF'),
        LabelInfo('MATERIAL', '材质', '#7b1fa2', '2', '材料材质，如2205、304、A182 F51、碳钢'),
        LabelInfo('SIZE', '尺寸', '#c62828', '3', '尺寸规格，如DN100、φ108、4寸'),
        LabelInfo('THICKNESS', '壁厚', '#ef6c00', '4', '壁厚，如4mm、Sch40、SCH80'),
        LabelInfo('PRESSURE', '磅级', '#2e7d32', '5', '压力等级，如PN16、Class150、150LB'),
        LabelInfo('STANDARD', '规范', '#00838f', '6', '标准规范，如GB/T、HG/T、NB/T、ASME'),
        LabelInfo('CONN', '连接方式', '#9c27b0', '7', '连接方式，如BW、SW、对焊、螺纹连接、法兰连接'),
        LabelInfo('MANU', '制造工艺', '#ff9800', '8', '制造工艺，如焊接、无缝、锻制、SMLS'),
    ]
)


# 管道TYPE分类标签（用于分类头）
PIPE_TYPE_CLASSES = [
    '管子',    # 管子/直管/钢管等
    '管件',    # 弯头/三通/异径管等
    '法兰',    # 各类法兰
    '螺栓',    # 螺栓/螺母/紧固件
    '阀门',    # 各类阀门
    '垫片',    # 垫片/密封件
]


# 平台注册表
PLATFORMS: Dict[str, PlatformConfig] = {
    'cable': CABLE_PLATFORM,
    'pipe': PIPE_PLATFORM,
}


# ==================== 标准名标签配置 ====================

# 管道平台标准名标签
PIPE_NORMALIZATION_LABELS: Dict[str, List[Dict[str, str]]] = {
    'TYPE': [
        {'value': '法兰', 'label': '法兰'},
        {'value': '弯头', 'label': '弯头'},
        {'value': '三通', 'label': '三通'},
        {'value': '四通', 'label': '四通'},
        {'value': '异径管', 'label': '异径管'},
        {'value': '管帽', 'label': '管帽'},
        {'value': '阀门', 'label': '阀门'},
        {'value': '直管', 'label': '直管'},
    ],
    'MATERIAL': [
        {'value': '碳钢', 'label': '碳钢'},
        {'value': '不锈钢304', 'label': '不锈钢304'},
        {'value': '不锈钢316', 'label': '不锈钢316'},
        {'value': '双相钢2205', 'label': '双相钢2205'},
        {'value': '合金钢', 'label': '合金钢'},
        {'value': '铸铁', 'label': '铸铁'},
        {'value': 'PVC', 'label': 'PVC'},
        {'value': 'PPR', 'label': 'PPR'},
    ],
    'PRESSURE': [
        {'value': 'PN10', 'label': 'PN10'},
        {'value': 'PN16', 'label': 'PN16'},
        {'value': 'PN25', 'label': 'PN25'},
        {'value': 'PN40', 'label': 'PN40'},
        {'value': 'Class150', 'label': 'Class150'},
        {'value': 'Class300', 'label': 'Class300'},
        {'value': 'Class600', 'label': 'Class600'},
    ],
    'STANDARD': [
        {'value': 'GB/T', 'label': 'GB/T 国标'},
        {'value': 'HG/T', 'label': 'HG/T 化工标准'},
        {'value': 'NB/T', 'label': 'NB/T 能源标准'},
        {'value': 'ASME', 'label': 'ASME 美标'},
        {'value': 'DIN', 'label': 'DIN 德标'},
        {'value': 'EN', 'label': 'EN 欧标'},
        {'value': 'JIS', 'label': 'JIS 日标'},
    ],
    'WELD': [
        {'value': 'BW', 'label': 'BW 对焊'},
        {'value': 'SW', 'label': 'SW 承插焊'},
        {'value': 'TW', 'label': 'TW 螺纹'},
        {'value': 'FL', 'label': 'FL 法兰连接'},
    ],
    # SIZE 和 THICKNESS 通常不需要归一化，保留原值
}

# 电缆/桥架平台归一化标签
CABLE_NORMALIZATION_LABELS: Dict[str, List[Dict[str, str]]] = {
    'NAME': [
        {'value': '桥架', 'label': '桥架'},
        {'value': '弯通', 'label': '弯通'},
        {'value': '三通', 'label': '三通'},
        {'value': '四通', 'label': '四通'},
        {'value': '直通', 'label': '直通'},
        {'value': '线槽', 'label': '线槽'},
        {'value': '槽盒', 'label': '槽盒'},
    ],
    'MATERIAL': [
        {'value': '铝合金', 'label': '铝合金'},
        {'value': '镀锌钢', 'label': '镀锌钢'},
        {'value': '不锈钢', 'label': '不锈钢'},
        {'value': '玻璃钢', 'label': '玻璃钢'},
        {'value': '复合型', 'label': '复合型'},
        {'value': 'PVC', 'label': 'PVC'},
    ],
    'TYPE': [
        {'value': '梯式', 'label': '梯式'},
        {'value': '槽式', 'label': '槽式'},
        {'value': '托盘式', 'label': '托盘式'},
        {'value': '梯级式', 'label': '梯级式'},
    ],
}

# 归一化标签注册表
NORMALIZATION_LABELS: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    'cable': CABLE_NORMALIZATION_LABELS,
    'pipe': PIPE_NORMALIZATION_LABELS,
}


# ==================== 公共接口 ====================

def get_platform(platform_name: str) -> PlatformConfig:
    """获取平台配置"""
    if platform_name not in PLATFORMS:
        raise ValueError(f"未知平台: {platform_name}，可用平台: {list(PLATFORMS.keys())}")
    return PLATFORMS[platform_name]


def get_platform_labels(platform_name: str) -> List[Dict[str, Any]]:
    """获取平台标签列表（用于API返回）"""
    platform = get_platform(platform_name)
    return [
        {
            'value': l.value,
            'label': f'{l.value} - {l.label}',
            'color': l.color,
            'key': l.key,
        }
        for l in platform.labels
    ]


def get_bio_labels(platform_name: str) -> List[str]:
    """获取BIO格式的标签列表（用于训练）"""
    platform = get_platform(platform_name)
    return platform.get_bio_labels()


def get_all_platforms() -> List[Dict[str, Any]]:
    """获取所有平台列表"""
    return [
        {
            'name': p.name,
            'display_name': p.display_name,
            'icon': p.icon,
        }
        for p in PLATFORMS.values()
    ]


def list_platforms() -> List[str]:
    """列出所有平台名称"""
    return list(PLATFORMS.keys())


def get_normalization_labels(platform_name: str, entity_type: str = None) -> Dict[str, List[Dict[str, str]]]:
    """
    获取标准名映射标签配置
    
    Args:
        platform_name: 平台名称 (cable, pipe)
        entity_type: 实体类型，如 'TYPE', 'MATERIAL'。为None时返回所有类型
    
    Returns:
        标准名映射字典
    """
    platform_labels = NORMALIZATION_LABELS.get(platform_name, {})
    
    if entity_type:
        return {entity_type: platform_labels.get(entity_type, [])}
    
    return platform_labels


def add_normalization_label(platform_name: str, entity_type: str, label: str) -> bool:
    """
    动态添加标准名标签
    
    Args:
        platform_name: 平台名称
        entity_type: 实体类型
        label: 新标准名值
    
    Returns:
        是否添加成功
    """
    if platform_name not in NORMALIZATION_LABELS:
        return False
    
    if entity_type not in NORMALIZATION_LABELS[platform_name]:
        NORMALIZATION_LABELS[platform_name][entity_type] = []
    
    # 检查是否已存在
    existing = [l['value'] for l in NORMALIZATION_LABELS[platform_name][entity_type]]
    if label not in existing:
        NORMALIZATION_LABELS[platform_name][entity_type].append({
            'value': label,
            'label': label
        })
        return True
    
    return False

