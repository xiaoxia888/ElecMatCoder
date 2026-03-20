"""
尺寸处理器

处理规则：
1. 解析尺寸值，返回数值列表
2. 保持原始出现顺序
3. 相同尺寸去重
4. 支持 DN 格式、NPS 格式、φ 格式、其他格式
"""
import re
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizeResult:
    """尺寸解析结果"""
    values: List[float]  # 数值列表（按原始出现顺序去重）
    original: str  # 原始值
    format_type: str  # 格式类型：dn, phi, other


class SizeProcessor:
    """
    尺寸处理器
    
    处理规则：
    1. 解析尺寸值，返回数值列表
    2. 保持原始出现顺序
    3. 相同尺寸去重
    4. 支持 DN 格式、NPS 格式、φ 格式、其他格式
    """
    
    # 管外径 -> 公称直径 映射表（类变量，延迟加载）
    _od_to_dn_mapping: dict = None
    _instance: 'SizeProcessor' = None
    _nps_to_dn_mapping: dict = None
    STRUCTURED_SUBTYPES = ("DN", "OD", "INCH")
    
    def __init__(self):
        self._load_od_mapping()
        self._load_nps_mapping()
    
    @classmethod
    def _load_od_mapping(cls):
        """从 Excel 加载管外径到公称直径的映射"""
        if cls._od_to_dn_mapping is not None:
            return
        
        cls._od_to_dn_mapping = {}
        
        try:
            import pandas as pd
            excel_path = Path(__file__).parent.parent / "config" / "壁厚对照汇总表.xlsx"
            
            if excel_path.exists():
                df = pd.read_excel(excel_path, header=1)
                
                for _, row in df.iterrows():
                    try:
                        od = float(row['管外径'])
                        dn = int(float(row['公称直径']))
                        cls._od_to_dn_mapping[od] = dn
                    except (ValueError, TypeError, KeyError):
                        continue
                
                logger.info(f"[SizeProcessor] 加载管外径映射: {len(cls._od_to_dn_mapping)} 条")
            else:
                logger.warning(f"[SizeProcessor] 管外径映射文件不存在: {excel_path}")
        except Exception as e:
            logger.warning(f"[SizeProcessor] 加载管外径映射失败: {e}")
    
    @classmethod
    def _load_nps_mapping(cls):
        """从 encoder_config.yaml 加载 NPS→DN 映射"""
        if cls._nps_to_dn_mapping is not None:
            return
        
        cls._nps_to_dn_mapping = {}
        try:
            config_path = Path(__file__).parent.parent / "config" / "encoder_config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                nps_config = config.get('size_processing', {}).get('nps_to_dn', {})
                cls._nps_to_dn_mapping = {str(k): int(v) for k, v in nps_config.items()}
                logger.info(f"[SizeProcessor] 加载NPS映射: {len(cls._nps_to_dn_mapping)} 条")
            else:
                logger.warning(f"[SizeProcessor] 配置文件不存在: {config_path}")
        except Exception as e:
            logger.warning(f"[SizeProcessor] 加载NPS映射失败: {e}")
    
    def _nps_to_dn(self, nps_str: str) -> Optional[int]:
        """将 NPS/英制尺寸字符串转换为 DN 公称直径"""
        normalized = self._normalize_nps_token(nps_str)
        if not normalized:
            return None
        mapped = self._nps_to_dn_mapping.get(normalized)
        if mapped is not None:
            return mapped

        # 兜底规则：当英制尺寸未配置且为纯数字且 > 10 时，按 inch * 25 转换为 DN
        # 例如：11" -> DN275, NPS13 -> DN325
        if re.fullmatch(r'\d+(?:\.\d+)?', normalized):
            inch_val = float(normalized)
            if inch_val > 10:
                return int(round(inch_val * 25))

        return None

    def _normalize_nps_token(self, token: str) -> str:
        """
        归一化英制尺寸 token，支持：
        - 1/2, 1/2", 1/2”
        - 1-1/4, 1 1/4, 1-1/4", NPS1 1/4
        - NPS4, 4", 4”
        """
        if not token:
            return ""
        t = str(token).strip()
        t = re.sub(r'(?i)^NPS\s*', '', t)
        # 统一引号（英寸）
        t = t.replace('”', '"').replace('“', '"').replace('″', '"')
        t = re.sub(r'["\']', '', t).strip()
        # 统一混合分数写法：1 1/4 -> 1-1/4
        t = re.sub(r'\s+', ' ', t)
        if re.fullmatch(r'\d+\s+\d+/\d+', t):
            t = t.replace(' ', '-')
        # 去掉连字符周围空格：1 - 1/4 -> 1-1/4
        t = re.sub(r'\s*-\s*', '-', t)

        # 小数英寸写法归一化：1.5 -> 1-1/2, 1.25 -> 1-1/4, 1.75 -> 1-3/4
        # 这样可命中配置里的 NPS 映射（通常使用分数/混合分数表达）
        if re.fullmatch(r'\d+\.\d+', t):
            try:
                x = float(t)
                int_part = int(x)
                frac = round(x - int_part, 6)
                frac_map = {
                    0.125: "1/8",
                    0.25: "1/4",
                    0.375: "3/8",
                    0.5: "1/2",
                    0.625: "5/8",
                    0.75: "3/4",
                    0.875: "7/8",
                }
                frac_token = frac_map.get(frac)
                if frac_token:
                    if int_part == 0:
                        t = frac_token
                    else:
                        t = f"{int_part}-{frac_token}"
            except Exception:
                pass
        return t
    
    def _od_to_dn(self, od_value: float) -> Optional[int]:
        """将管外径转换为公称直径"""
        if not self._od_to_dn_mapping:
            return None
        
        # 精确匹配
        if od_value in self._od_to_dn_mapping:
            return self._od_to_dn_mapping[od_value]
        
        # 容差匹配（±0.5mm）
        for od, dn in self._od_to_dn_mapping.items():
            if abs(od - od_value) <= 0.5:
                return dn
        
        return None

    @staticmethod
    def _extract_numeric_value(value: Any) -> Optional[float]:
        """从 DN/OD 原始值中提取数字部分。"""
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _collect_structured_values(self, value: Dict[str, Any]) -> List[float]:
        """
        处理新 schema 的 SIZE 结构：
        - DN 直接取值
        - OD / INCH 先转换为 DN
        - 最终统一按原始出现顺序去重
        """
        all_values: List[float] = []

        for item in self._ensure_list(value.get("DN")):
            dn = self._extract_numeric_value(item)
            if dn is not None:
                all_values.append(float(dn))

        for item in self._ensure_list(value.get("OD")):
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od)
            if dn is not None:
                all_values.append(float(dn))

        for item in self._ensure_list(value.get("INCH")):
            raw = item.get("value") if isinstance(item, dict) else item
            dn = self._nps_to_dn(str(raw).strip()) if raw not in (None, "") else None
            if dn is not None:
                all_values.append(float(dn))

        return self._sort_sizes(all_values)

    def _analyze_structured_values(self, value: Dict[str, Any]) -> Tuple[List[float], bool]:
        """
        分析结构化 SIZE 值。

        规则：
        1. 若显式给出了 DN，则编码仅使用 DN。
        2. 若同时存在 OD/INCH，先换算成 DN；只要换算结果不在显式 DN 集合中，则标记待审。
        3. 若没有 DN，则按现有逻辑将 OD/INCH 全部换算为 DN 后按原始顺序去重编码。
        """
        dn_values: List[float] = []
        converted_values: List[float] = []

        for item in self._ensure_list(value.get("DN")):
            dn = self._extract_numeric_value(item)
            if dn is not None:
                dn_values.append(float(dn))

        for item in self._ensure_list(value.get("OD")):
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od)
            if dn is not None:
                converted_values.append(float(dn))

        for item in self._ensure_list(value.get("INCH")):
            raw = item.get("value") if isinstance(item, dict) else item
            dn = self._nps_to_dn(str(raw).strip()) if raw not in (None, "") else None
            if dn is not None:
                converted_values.append(float(dn))

        explicit_dn = self._sort_sizes(dn_values)
        if explicit_dn:
            explicit_dn_set = set(explicit_dn)
            need_review = any(value not in explicit_dn_set for value in converted_values)
            return explicit_dn, need_review

        merged = self._sort_sizes(converted_values)
        return merged, False
    
    def parse(self, value: str) -> SizeResult:
        """
        解析尺寸值，返回数值列表
        
        Args:
            value: 尺寸值（如 "DN250xDN300", "φ60.3Xφ21.3"）
            
        Returns:
            SizeResult: 解析结果，values 按原始出现顺序去重
        """
        if not value:
            return SizeResult(values=[], original=value, format_type='other')
        
        value = value.strip()
        normalized_value = value.replace('”', '"').replace('“', '"').replace('″', '"')
        
        # ========== DN 格式 ==========
        # DN25XDN15, DN30XDN30, DN250xDN300
        dn_cross_pattern = r'DN\s*(\d+(?:\.\d+)?)\s*[xX*×/]\s*DN?\s*(\d+(?:\.\d+)?)'
        dn_cross_match = re.search(dn_cross_pattern, value, re.IGNORECASE)
        if dn_cross_match:
            size1, size2 = float(dn_cross_match.group(1)), float(dn_cross_match.group(2))
            values = self._sort_sizes([size1, size2])
            return SizeResult(values=values, original=value, format_type='dn')
        
        # DN150X20, DN300/20
        dn_num_pattern = r'DN\s*(\d+(?:\.\d+)?)\s*[xX*×/]\s*(\d+(?:\.\d+)?)'
        dn_num_match = re.search(dn_num_pattern, value, re.IGNORECASE)
        if dn_num_match:
            size1, size2 = float(dn_num_match.group(1)), float(dn_num_match.group(2))
            values = self._sort_sizes([size1, size2])
            return SizeResult(values=values, original=value, format_type='dn')
        
        # 单一 DN 值: DN100
        dn_pattern = r'DN\s*(\d+(?:\.\d+)?)'
        dn_match = re.search(dn_pattern, value, re.IGNORECASE)
        if dn_match:
            size = float(dn_match.group(1))
            return SizeResult(values=[size], original=value, format_type='dn')
        
        # ========== NPS 格式：转换为公称直径 ==========
        # NPS14×1/2, NPS14xNPS1/2, NPS1/2xNPS1/4, NPS1/2×NPS1/4
        nps_cross = re.search(
            r'NPS\s*(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\s*["]?\s*[xX*×]\s*(?:NPS\s*)?(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\s*["]?',
            normalized_value, re.IGNORECASE
        )
        if nps_cross:
            dn1 = self._nps_to_dn(nps_cross.group(1))
            dn2 = self._nps_to_dn(nps_cross.group(2))
            if dn1 is not None and dn2 is not None:
                values = self._sort_sizes([float(dn1), float(dn2)])
                return SizeResult(values=values, original=value, format_type='dn')
        
        # NPS14, NPS1/2
        nps_single = re.search(r'NPS\s*(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\b', normalized_value, re.IGNORECASE)
        if nps_single:
            dn = self._nps_to_dn(nps_single.group(1))
            if dn is not None:
                return SizeResult(values=[float(dn)], original=value, format_type='dn')
        
        # 含分数的异径 (无NPS前缀): 1/2×1/4, 14×1/2, 1/2×14
        frac_cross = re.search(r'(\d+(?:/\d+)?)\s*[xX*×]\s*(\d+(?:/\d+)?)', value)
        if frac_cross:
            v1, v2 = frac_cross.group(1), frac_cross.group(2)
            if '/' in v1 or '/' in v2:
                dn1 = self._nps_to_dn(v1)
                dn2 = self._nps_to_dn(v2)
                if dn1 is not None and dn2 is not None:
                    values = self._sort_sizes([float(dn1), float(dn2)])
                    return SizeResult(values=values, original=value, format_type='dn')
        
        # 纯分数单值: 1/2, 3/4
        frac_single = re.match(r'^(\d+/\d+)$', value.strip())
        if frac_single:
            dn = self._nps_to_dn(frac_single.group(1))
            if dn is not None:
                return SizeResult(values=[float(dn)], original=value, format_type='dn')

        # ========== 英制引号格式（" / ” / ″）：统一按英制处理 ==========
        # 例如：1/2", 1 1/4, 1-1/4", NPS4×1 1/2, NPS4×NPS1 1/2
        # 支持小数英寸：1.5", 2.25"
        imperial_token = r'(\d+\s*[-\s]\s*\d+/\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?)'
        imperial_cross = re.search(
            rf'(?<![A-Za-z0-9])(?:NPS\s*)?{imperial_token}\s*["]?\s*[xX*×]\s*(?:NPS\s*)?{imperial_token}\s*["]?(?![A-Za-z0-9])',
            normalized_value,
            re.IGNORECASE
        )
        if imperial_cross:
            dn1 = self._nps_to_dn(imperial_cross.group(1))
            dn2 = self._nps_to_dn(imperial_cross.group(2))
            if dn1 is not None and dn2 is not None:
                values = self._sort_sizes([float(dn1), float(dn2)])
                return SizeResult(values=values, original=value, format_type='dn')

        # 单值英制（有 NPS 前缀，或包含英寸引号）
        has_inch_quote = any(ch in normalized_value for ch in ['"'])
        imperial_single = re.search(
            r'(?<![A-Za-z0-9])(?:NPS\s*)?(\d+(?:\s*[-\s]\s*\d+/\d+|\s*/\s*\d+|\.\d+)?)\s*["]?(?![A-Za-z0-9/])',
            normalized_value,
            re.IGNORECASE
        )
        if imperial_single and (has_inch_quote or re.search(r'(?i)\bNPS\b', normalized_value)):
            dn = self._nps_to_dn(imperial_single.group(1))
            if dn is not None:
                return SizeResult(values=[float(dn)], original=value, format_type='dn')

        # 无引号、无NPS前缀的混合分数/分数（如 1 1/4, 1-1/4, 1/2）
        bare_imperial = re.match(r'^\s*(\d+\s*[-\s]\s*\d+/\d+|\d+\s*/\s*\d+)\s*$', normalized_value)
        if bare_imperial:
            dn = self._nps_to_dn(bare_imperial.group(1))
            if dn is not None:
                return SizeResult(values=[float(dn)], original=value, format_type='dn')
        
        # ========== φ 格式：转换为公称直径 ==========
        # φ60.3Xφ21.3
        phi_cross_pattern = r'[φΦ](\d+(?:\.\d+)?)\s*[xX*×]\s*[φΦ](\d+(?:\.\d+)?)'
        phi_cross_match = re.search(phi_cross_pattern, value)
        if phi_cross_match:
            od1, od2 = float(phi_cross_match.group(1)), float(phi_cross_match.group(2))
            
            # 尝试转换为公称直径
            dn1 = self._od_to_dn(od1)
            dn2 = self._od_to_dn(od2)
            
            if dn1 is not None and dn2 is not None:
                values = self._sort_sizes([float(dn1), float(dn2)])
            else:
                # 无法转换，使用原值
                values = self._sort_sizes([od1, od2])
            
            return SizeResult(values=values, original=value, format_type='phi')
        
        # 单一 φ 值: φ60.3
        phi_pattern = r'[φΦ](\d+(?:\.\d+)?)'
        phi_match = re.search(phi_pattern, value)
        if phi_match:
            od = float(phi_match.group(1))
            dn = self._od_to_dn(od)
            size = float(dn) if dn is not None else od
            return SizeResult(values=[size], original=value, format_type='phi')
        
        # ========== 其他格式 ==========
        # NNmmxNNmm: 150mmx100mm
        mm_cross_pattern = r'(\d+(?:\.\d+)?)\s*mm\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*mm'
        mm_cross_match = re.search(mm_cross_pattern, value, re.IGNORECASE)
        if mm_cross_match:
            size1, size2 = float(mm_cross_match.group(1)), float(mm_cross_match.group(2))
            values = self._sort_sizes([size1, size2])
            return SizeResult(values=values, original=value, format_type='other')
        
        # NNxNN: 150x100
        cross_pattern = r'(\d+(?:\.\d+)?)\s*[xX*×]\s*(\d+(?:\.\d+)?)'
        cross_match = re.search(cross_pattern, value)
        if cross_match:
            size1, size2 = float(cross_match.group(1)), float(cross_match.group(2))
            values = self._sort_sizes([size1, size2])
            return SizeResult(values=values, original=value, format_type='other')
        
        # 带单位的单一值: 150mm
        unit_pattern = r'(\d+(?:\.\d+)?)\s*(mm|MM|cm|CM|m|M|in|inch|寸)'
        unit_match = re.search(unit_pattern, value)
        if unit_match:
            size = float(unit_match.group(1))
            return SizeResult(values=[size], original=value, format_type='other')
        
        # 纯数字
        number_pattern = r'(\d+(?:\.\d+)?)'
        number_match = re.search(number_pattern, value)
        if number_match:
            size = float(number_match.group(1))
            return SizeResult(values=[size], original=value, format_type='other')
        
        return SizeResult(values=[], original=value, format_type='other')
    
    def _sort_sizes(self, sizes: List[float]) -> List[float]:
        """按原始出现顺序去重，不再按大小排序。"""
        unique: List[float] = []
        for size in sizes:
            if size not in unique:
                unique.append(size)
        return unique
    
    def format_code(self, values: List[float]) -> str:
        """
        将数值列表格式化为编码字符串
        
        Args:
            values: 数值列表（按原始出现顺序去重）
            
        Returns:
            编码字符串，如 "300x250" 或 "100"
        """
        if not values:
            return ""
        
        # 去重
        unique = []
        for v in values:
            if v not in unique:
                unique.append(v)
        
        # 格式化：整数不带小数点
        formatted = []
        for v in unique:
            if v == int(v):
                formatted.append(str(int(v)))
            else:
                formatted.append(str(v))
        
        if len(formatted) == 1:
            return formatted[0]
        
        return 'x'.join(formatted)
    
    def process(self, value: Any) -> str:
        """
        处理单个尺寸值，返回编码字符串
        
        Args:
            value: 尺寸值
            
        Returns:
            编码字符串
        """
        if isinstance(value, dict):
            values, _ = self._analyze_structured_values(value)
            return self.format_code(values)
        if isinstance(value, list):
            return self.process_multi(value)

        result = self.parse(str(value))
        return self.format_code(result.values)
    
    def process_multi(self, values: Any) -> str:
        """
        处理多个尺寸值
        
        规则：
        1. 优先选择 DN 格式
        2. 合并所有数值
        3. 按原始顺序去重
        4. 格式化输出
        
        Args:
            values: 尺寸值列表
            
        Returns:
            编码字符串
        """
        if not values:
            return ""

        if isinstance(values, dict):
            return self.process(values)

        if not isinstance(values, list):
            return self.process(values)

        # 新 schema：若包含结构化 dict，则按显式 DN 优先、其他尺寸换算校验的逻辑处理
        if any(isinstance(v, dict) for v in values):
            all_values: List[float] = []
            for v in values:
                if not v:
                    continue
                if isinstance(v, dict):
                    normalized_values, _ = self._analyze_structured_values(v)
                    all_values.extend(normalized_values)
                else:
                    all_values.extend(self.parse(str(v).strip()).values)
            return self.format_code(self._sort_sizes(all_values))
        
        # 分离不同格式
        dn_results = []
        phi_results = []
        other_results = []
        
        for v in values:
            if not v or not v.strip():
                continue
            result = self.parse(v.strip())
            if result.format_type == 'dn':
                dn_results.append(result)
            elif result.format_type == 'phi':
                phi_results.append(result)
            else:
                other_results.append(result)
        
        # 优先级：DN > other > phi
        if dn_results:
            all_values = []
            for r in dn_results:
                all_values.extend(r.values)
            sorted_values = self._sort_sizes(all_values)
            return self.format_code(sorted_values)
        
        if other_results:
            return self.format_code(other_results[0].values)
        
        if phi_results:
            all_values = []
            for r in phi_results:
                all_values.extend(r.values)
            sorted_values = self._sort_sizes(all_values)
            return self.format_code(sorted_values)
        
        return ""

    def process_multi_with_review(self, values: Any) -> Tuple[str, bool]:
        """
        返回 SIZE 编码和是否需要审核。
        """
        if not values:
            return "", False

        if isinstance(values, dict):
            normalized_values, need_review = self._analyze_structured_values(values)
            return self.format_code(normalized_values), need_review

        if not isinstance(values, list):
            return self.process(values), False

        if any(isinstance(v, dict) for v in values):
            all_values: List[float] = []
            need_review = False
            for v in values:
                if not v:
                    continue
                if isinstance(v, dict):
                    normalized_values, item_review = self._analyze_structured_values(v)
                    all_values.extend(normalized_values)
                    need_review = need_review or item_review
                else:
                    all_values.extend(self.parse(str(v).strip()).values)
            return self.format_code(self._sort_sizes(all_values)), need_review

        return self.process_multi(values), False


# 单例
_processor_instance: Optional[SizeProcessor] = None


def get_size_processor() -> SizeProcessor:
    """获取尺寸处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = SizeProcessor()
    return _processor_instance


# 测试代码
if __name__ == '__main__':
    processor = SizeProcessor()
    
    test_cases = [
        'DN250xDN300',  # 期望: 250x300
        'DN300xDN250',  # 期望: 300x250
        'DN100',        # 期望: 100
        'DN100xDN100',  # 期望: 100（去重）
        'φ60.3Xφ21.3',  # 期望: 转换为公称直径
        '150mmx100mm',  # 期望: 150x100
        '100x150',      # 期望: 100x150
    ]
    
    nps_test_cases = [
        ('NPS14×1/2',       '350x15'),
        ('NPS1/2xNPS1/4',   '15x8'),
        ('NPS1/2×NPS1/4',   '15x8'),
        ('NPS14xNPS1/2',    '350x15'),
        ('1 1/4',           '32'),
        ('NPS1/2',          '15'),
        ('NPS3/4',          '20'),
        ('1/2x1/4',         '15x8'),
        ('1/2×1/4',         '15x8'),
        ('14×1/2',          '350x15'),
        ('1/2×14',          '15x350'),
        ('1/2',             '15'),
        ('3/4',             '20'),
        ('1-1/4',           '32'),
    ]
    
    print("=== 尺寸处理测试 ===")
    for case in test_cases:
        result = processor.parse(case)
        code = processor.format_code(result.values)
        print(f"输入: {case}")
        print(f"  解析: {result.values}  编码: {code}")
        print("-" * 40)
    
    print("\n=== NPS 格式测试 ===")
    for case, expected in nps_test_cases:
        result = processor.parse(case)
        code = processor.format_code(result.values)
        status = '✓' if code == expected else '✗'
        print(f"{status} 输入: {case}")
        print(f"  解析: {result.values}  编码: {code}  期望: {expected}")
        if code != expected:
            print(f"  *** 不匹配 ***")
        print("-" * 40)
