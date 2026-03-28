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
                df = pd.read_excel(excel_path, header=0)
                
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

    @staticmethod
    def _normalize_number_text(num: Any) -> str:
        try:
            value = float(str(num))
        except Exception:
            return str(num).strip()
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _extract_item_start(item: Any) -> Optional[int]:
        if not isinstance(item, dict):
            return None
        start = item.get("start")
        if start is None:
            return None
        try:
            return int(start)
        except (TypeError, ValueError):
            return None

    def _find_size_position(self, subtype: str, item: Any, original_text: str) -> int:
        if not original_text:
            return -1

        if isinstance(item, dict):
            raw = item.get("value")
        else:
            raw = item
        raw_text = str(raw or "").strip()
        if not raw_text:
            return -1

        text = original_text.upper()
        raw_upper = raw_text.upper()
        direct = text.find(raw_upper)
        if direct >= 0:
            return direct

        if subtype == "DN":
            m = re.search(r'(\d+(?:\.\d+)?)', raw_text)
            if not m:
                return -1
            num = m.group(1)
            for pattern in (
                rf'DN\s*{re.escape(num)}\b',
                rf'(?<![A-Z]){re.escape(num)}(?![A-Z])',
            ):
                hit = re.search(pattern, text, re.IGNORECASE)
                if hit:
                    return hit.start()
            return -1

        if subtype == "OD":
            m = re.search(r'(\d+(?:\.\d+)?)', raw_text)
            if not m:
                return -1
            num = m.group(1)
            for pattern in (
                rf'外径\s*{re.escape(num)}(?:MM)?',
                rf'[Φφ]\s*{re.escape(num)}\b',
                rf'OD\s*{re.escape(num)}\b',
                rf'(?<![A-Z0-9]){re.escape(num)}MM(?![A-Z0-9])',
            ):
                hit = re.search(pattern, text, re.IGNORECASE)
                if hit:
                    return hit.start()
            return -1

        if subtype == "INCH":
            normalized = self._normalize_nps_token(raw_text)
            variants = [raw_text, normalized]
            if normalized and "-" in normalized:
                variants.append(normalized.replace("-", " "))
                decimal_variant = self._mixed_fraction_to_decimal(normalized)
                if decimal_variant:
                    variants.append(decimal_variant)
            if normalized:
                variants.append(f'NPS{normalized}')
                variants.append(f'NPS {normalized}')
                if "-" in normalized:
                    spaced = normalized.replace("-", " ")
                    variants.append(f'NPS{spaced}')
                    variants.append(f'NPS {spaced}')
                    decimal_variant = self._mixed_fraction_to_decimal(normalized)
                    if decimal_variant:
                        variants.append(f'NPS{decimal_variant}')
                        variants.append(f'NPS {decimal_variant}')
            for variant in variants:
                variant = str(variant or "").strip()
                if not variant:
                    continue
                pos = text.find(variant.upper())
                if pos >= 0:
                    return pos
            return -1

        return -1

    @staticmethod
    def _mixed_fraction_to_decimal(token: str) -> str:
        """
        2-1/2 -> 2.5
        1-1/4 -> 1.25
        3/4   -> 0.75
        """
        t = str(token or "").strip()
        if not t:
            return ""
        try:
            if "-" in t:
                integer, frac = t.split("-", 1)
                num, den = frac.split("/", 1)
                value = float(integer) + (float(num) / float(den))
            elif "/" in t:
                num, den = t.split("/", 1)
                value = float(num) / float(den)
            else:
                return ""
            return str(value).rstrip('0').rstrip('.')
        except Exception:
            return ""

    def _sort_structured_items(self, subtype: str, items: List[Any], original_text: str = "") -> List[Any]:
        if len(items) <= 1:
            return items

        indexed = []
        for idx, item in enumerate(items):
            start = self._extract_item_start(item)
            if start is None:
                pos = self._find_size_position(subtype, item, original_text)
            else:
                pos = start
            indexed.append((idx, pos, item))

        indexed.sort(key=lambda x: (x[1] < 0, x[1] if x[1] >= 0 else 10**9, x[0]))
        return [item for _, _, item in indexed]

    def _build_positioned_size_items(self, value: Dict[str, Any], original_text: str = "") -> List[Dict[str, Any]]:
        """
        构建带位置的尺寸对象列表。

        规则：
        1. 若显式存在 DN，则只保留 DN 参与编码/壁厚换算
        2. 若无 DN，则使用可转换为 DN 的 OD / INCH，并按原文位置排序
        3. 若无 DN 且 OD 无法转换，则保留 OD 原值仅用于尺寸编码回退
        """
        dn_items: List[Dict[str, Any]] = []
        converted_items: List[Dict[str, Any]] = []
        od_fallback_items: List[Dict[str, Any]] = []

        for item in self._sort_structured_items("DN", self._ensure_list(value.get("DN")), original_text):
            raw = item.get("value") if isinstance(item, dict) else item
            start = self._extract_item_start(item)
            if start is None:
                start = self._find_size_position("DN", item, original_text)
            dn = self._extract_numeric_value(item)
            if dn is None:
                continue
            dn_items.append({
                "subtype": "DN",
                "raw": str(raw or "").strip(),
                "start": start,
                "encode_value": float(dn),
                "dn_value": float(dn),
            })

        for item in self._sort_structured_items("OD", self._ensure_list(value.get("OD")), original_text):
            raw = item.get("value") if isinstance(item, dict) else item
            start = self._extract_item_start(item)
            if start is None:
                start = self._find_size_position("OD", item, original_text)
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od)
            if dn is not None:
                converted_items.append({
                    "subtype": "OD",
                    "raw": str(raw or "").strip(),
                    "start": start,
                    "encode_value": float(dn),
                    "dn_value": float(dn),
                })
            else:
                od_fallback_items.append({
                    "subtype": "OD",
                    "raw": str(raw or "").strip(),
                    "start": start,
                    "encode_value": float(od),
                    "dn_value": None,
                })

        for item in self._sort_structured_items("INCH", self._ensure_list(value.get("INCH")), original_text):
            raw = item.get("value") if isinstance(item, dict) else item
            start = self._extract_item_start(item)
            if start is None:
                start = self._find_size_position("INCH", item, original_text)
            raw_text = str(raw or "").strip()
            if not raw_text:
                continue
            dn = self._nps_to_dn(raw_text)
            if dn is not None:
                converted_items.append({
                    "subtype": "INCH",
                    "raw": raw_text,
                    "start": start,
                    "encode_value": float(dn),
                    "dn_value": float(dn),
                })

        if dn_items:
            return sorted(dn_items, key=lambda x: (x["start"] < 0, x["start"] if x["start"] >= 0 else 10**9))
        if converted_items:
            return sorted(converted_items, key=lambda x: (x["start"] < 0, x["start"] if x["start"] >= 0 else 10**9))
        if od_fallback_items:
            return sorted(od_fallback_items, key=lambda x: (x["start"] < 0, x["start"] if x["start"] >= 0 else 10**9))
        return []

    def _extract_length_prefix(self, value: Any, original_text: str = "") -> str:
        """
        从 SIZE 中提取 LENGTH，并格式化为编码前缀 `L数字`。
        不参与 DN/OD/INCH 的尺寸计算，仅用于最终 SIZE code 前缀。
        """
        if not value:
            return ""

        def _raw_item_text(item: Any) -> str:
            if isinstance(item, dict):
                return str(item.get("value") or "").strip()
            return str(item or "").strip()

        def _from_text(text: str) -> str:
            s = str(text or "").strip().upper()
            if not s:
                return ""
            m = re.search(r'(?:L|LEN|LENGTH)\s*=?\s*(\d+(?:\.\d+)?)(?:\s*(?:MM|CM|M))?', s)
            if not m:
                return ""
            return f"L{self._normalize_number_text(m.group(1))}"

        if isinstance(value, dict):
            items = self._sort_structured_items("LENGTH", self._ensure_list(value.get("LENGTH")), original_text)
            for item in items:
                prefix = _from_text(_raw_item_text(item))
                if prefix:
                    return prefix
            return ""

        if isinstance(value, list):
            for item in value:
                prefix = self._extract_length_prefix(item, original_text=original_text)
                if prefix:
                    return prefix
            return ""

        return _from_text(str(value))

    def extract_length_prefix(self, value: Any, original_text: str = "") -> str:
        """公开方法：提取 SIZE.LENGTH 对应的编码前缀 `L数字`。"""
        return self._extract_length_prefix(value, original_text=original_text)

    def _prepend_length_prefix(self, code: str, length_prefix: str) -> str:
        if not length_prefix:
            return code or ""
        if not code:
            return length_prefix
        return f"{length_prefix}{code}"

    def _collect_structured_values(self, value: Dict[str, Any], original_text: str = "") -> List[float]:
        """
        处理新 schema 的 SIZE 结构：
        - DN 直接取值
        - OD / INCH 先转换为 DN
        - 最终统一按原始出现顺序去重
        """
        return self._sort_sizes([float(item["encode_value"]) for item in self._build_positioned_size_items(value, original_text=original_text)])

    def _analyze_structured_values(self, value: Dict[str, Any], original_text: str = "") -> Tuple[List[float], bool]:
        """
        分析结构化 SIZE 值。

        规则：
        1. 若显式给出了 DN，则编码仅使用 DN。
        2. 若同时存在 OD/INCH，先换算成 DN；只要换算结果不在显式 DN 集合中，则标记待审。
        3. 若没有 DN，则按现有逻辑将 OD/INCH 全部换算为 DN 后按原始顺序去重编码。
        """
        explicit_dn_values: List[float] = []
        converted_values: List[float] = []
        od_fallback_values: List[float] = []

        for item in self._sort_structured_items("DN", self._ensure_list(value.get("DN")), original_text):
            dn = self._extract_numeric_value(item)
            if dn is not None:
                explicit_dn_values.append(float(dn))

        for item in self._sort_structured_items("OD", self._ensure_list(value.get("OD")), original_text):
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od)
            if dn is not None:
                converted_values.append(float(dn))
            else:
                od_fallback_values.append(float(od))

        for item in self._sort_structured_items("INCH", self._ensure_list(value.get("INCH")), original_text):
            raw = item.get("value") if isinstance(item, dict) else item
            dn = self._nps_to_dn(str(raw).strip()) if raw not in (None, "") else None
            if dn is not None:
                converted_values.append(float(dn))

        explicit_dn = self._sort_sizes(explicit_dn_values)
        if explicit_dn:
            explicit_dn_set = set(explicit_dn)
            need_review = any(v not in explicit_dn_set for v in converted_values) or bool(od_fallback_values)
            return explicit_dn, need_review

        merged = self._sort_sizes([float(item["encode_value"]) for item in self._build_positioned_size_items(value, original_text=original_text)])
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
    
    def process(self, value: Any, original_text: str = "") -> str:
        """
        处理单个尺寸值，返回编码字符串
        
        Args:
            value: 尺寸值
            
        Returns:
            编码字符串
        """
        length_prefix = self._extract_length_prefix(value, original_text=original_text)
        if isinstance(value, dict):
            values, _ = self._analyze_structured_values(value, original_text=original_text)
            return self._prepend_length_prefix(self.format_code(values), length_prefix)
        if isinstance(value, list):
            return self.process_multi(value, original_text=original_text)

        result = self.parse(str(value))
        return self._prepend_length_prefix(self.format_code(result.values), length_prefix)
    
    def process_multi(self, values: Any, original_text: str = "") -> str:
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

        length_prefix = self._extract_length_prefix(values, original_text=original_text)

        if isinstance(values, dict):
            return self.process(values, original_text=original_text)

        if not isinstance(values, list):
            return self.process(values, original_text=original_text)

        # 新 schema：若包含结构化 dict，则按显式 DN 优先、其他尺寸换算校验的逻辑处理
        if any(isinstance(v, dict) for v in values):
            all_values: List[float] = []
            for v in values:
                if not v:
                    continue
                if isinstance(v, dict):
                    normalized_values, _ = self._analyze_structured_values(v, original_text=original_text)
                    all_values.extend(normalized_values)
                else:
                    all_values.extend(self.parse(str(v).strip()).values)
            return self._prepend_length_prefix(self.format_code(self._sort_sizes(all_values)), length_prefix)
        
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
            return self._prepend_length_prefix(self.format_code(sorted_values), length_prefix)
        
        if other_results:
            return self._prepend_length_prefix(self.format_code(other_results[0].values), length_prefix)
        
        if phi_results:
            all_values = []
            for r in phi_results:
                all_values.extend(r.values)
            sorted_values = self._sort_sizes(all_values)
            return self._prepend_length_prefix(self.format_code(sorted_values), length_prefix)

        return length_prefix

    def process_multi_with_review(self, values: Any, original_text: str = "") -> Tuple[str, bool]:
        """
        返回 SIZE 编码和是否需要审核。
        """
        if not values:
            return "", False

        length_prefix = self._extract_length_prefix(values, original_text=original_text)

        if isinstance(values, dict):
            normalized_values, need_review = self._analyze_structured_values(values, original_text=original_text)
            return self._prepend_length_prefix(self.format_code(normalized_values), length_prefix), need_review

        if not isinstance(values, list):
            return self.process(values), False

        if any(isinstance(v, dict) for v in values):
            all_values: List[float] = []
            need_review = False
            for v in values:
                if not v:
                    continue
                if isinstance(v, dict):
                    normalized_values, item_review = self._analyze_structured_values(v, original_text=original_text)
                    all_values.extend(normalized_values)
                    need_review = need_review or item_review
                else:
                    all_values.extend(self.parse(str(v).strip()).values)
            return self._prepend_length_prefix(self.format_code(self._sort_sizes(all_values)), length_prefix), need_review

        return self.process_multi(values, original_text=original_text), False

    def extract_dn_values(self, value: Any, original_text: str = "") -> List[str]:
        """
        提取用于后续按位处理的 DN 列表。
        保持原始出现顺序去重，返回纯数字字符串列表，如 ['300', '200']。
        """
        if not value:
            return []

        if isinstance(value, dict):
            positioned_items = self._build_positioned_size_items(value, original_text=original_text)
            normalized_values = [
                float(item["dn_value"]) for item in positioned_items
                if item.get("dn_value") is not None
            ]
        elif isinstance(value, list):
            normalized_values = []
            for item in value:
                if isinstance(item, dict):
                    positioned_items = self._build_positioned_size_items(item, original_text=original_text)
                    normalized_values.extend(
                        float(entry["dn_value"]) for entry in positioned_items
                        if entry.get("dn_value") is not None
                    )
                else:
                    parsed = self.parse(str(item))
                    if parsed.format_type == 'dn':
                        normalized_values.extend(parsed.values)
        else:
            parsed = self.parse(str(value))
            normalized_values = parsed.values if parsed.format_type == 'dn' else []

        result: List[str] = []
        for num in normalized_values:
            if num == int(num):
                result.append(str(int(num)))
            else:
                result.append(str(num).rstrip('0').rstrip('.'))
        return result


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
