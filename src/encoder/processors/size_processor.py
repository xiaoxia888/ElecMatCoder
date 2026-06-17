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

from .combo_fallback_extractor import ComboFallbackExtractor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SizeResult:
    """尺寸解析结果"""
    values: List[float]  # 数值列表（按原始出现顺序去重）
    original: str  # 原始值
    format_type: str  # 格式类型：dn, phi, other


@dataclass
class RuleSizeExtraction:
    """基于确定性规则的尺寸抽取结果。"""
    dn: List[str]
    od: List[str]
    inch: List[str]
    length: List[str]
    size_code: str
    matched_texts: List[str]
    matched_spans: List[Tuple[int, int]] = field(default_factory=list)
    consumed_spans: List[Tuple[int, int]] = field(default_factory=list)
    ordered_items: List[Dict[str, str]] = field(default_factory=list)


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
    _od_candidate_rows: dict = None
    _instance: 'SizeProcessor' = None
    _nps_to_dn_mapping: dict = None
    _common_dn_values: set = None
    STRUCTURED_SUBTYPES = ("DN", "OD", "INCH")
    
    def __init__(
        self,
        od_mapping_path: Optional[Union[str, Path]] = None,
        nps_config_path: Optional[Union[str, Path]] = None,
    ):
        self.od_mapping_path = Path(od_mapping_path) if od_mapping_path else Path(__file__).parent.parent / "config" / "壁厚对照汇总表.xlsx"
        self.nps_config_path = Path(nps_config_path) if nps_config_path else Path(__file__).parent.parent / "config" / "encoder_config.yaml"
        self._od_to_dn_mapping = {}
        self._od_candidate_rows = {}
        self._nps_to_dn_mapping = {}
        self._common_dn_values = set()
        self._load_od_mapping_for_instance()
        self._load_nps_mapping_for_instance()

    def _load_od_mapping_for_instance(self) -> None:
        """实例化加载 OD -> DN 映射。"""
        try:
            import pandas as pd

            if not self.od_mapping_path.exists():
                logger.warning(f"[SizeProcessor] 管外径映射文件不存在: {self.od_mapping_path}")
                return

            df = pd.read_excel(self.od_mapping_path, header=0)
            for _, row in df.iterrows():
                try:
                    od = float(row['管外径'])
                    dn = int(float(row['公称直径']))
                    thickness_code = str(row.get('壁厚号') or '').strip()
                    thickness_mm = None
                    raw_mm = row.get('壁厚')
                    if raw_mm not in (None, ''):
                        try:
                            thickness_mm = float(raw_mm)
                        except (ValueError, TypeError):
                            thickness_mm = None
                    self._od_to_dn_mapping[od] = dn
                    self._od_candidate_rows.setdefault(od, []).append({
                        'od': od,
                        'dn': dn,
                        'thickness_code': thickness_code,
                        'thickness_mm': thickness_mm,
                    })
                except (ValueError, TypeError, KeyError):
                    continue
        except Exception as e:
            logger.warning(f"[SizeProcessor] 加载管外径映射失败: {e}")

    def _load_nps_mapping_for_instance(self) -> None:
        """实例化加载 NPS -> DN 映射。"""
        try:
            if not self.nps_config_path.exists():
                logger.warning(f"[SizeProcessor] 配置文件不存在: {self.nps_config_path}")
                return
            with open(self.nps_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            size_config = config.get('size_processing', {})
            nps_config = size_config.get('nps_to_dn', {})
            self._nps_to_dn_mapping = {str(k): int(v) for k, v in nps_config.items()}
            self._common_dn_values = {
                int(v) for v in size_config.get('common_dn_values', [])
                if str(v).strip()
            }
        except Exception as e:
            logger.warning(f"[SizeProcessor] 加载NPS映射失败: {e}")
    
    @classmethod
    def _load_od_mapping(cls):
        """从 Excel 加载管外径到公称直径的映射"""
        if cls._od_to_dn_mapping is not None:
            return
        
        cls._od_to_dn_mapping = {}
        cls._od_candidate_rows = {}
        
        try:
            import pandas as pd
            excel_path = Path(__file__).parent.parent / "config" / "壁厚对照汇总表.xlsx"
            
            if excel_path.exists():
                df = pd.read_excel(excel_path, header=0)
                
                for _, row in df.iterrows():
                    try:
                        od = float(row['管外径'])
                        dn = int(float(row['公称直径']))
                        thickness_code = str(row.get('壁厚号') or '').strip()
                        thickness_mm = None
                        raw_mm = row.get('壁厚')
                        if raw_mm not in (None, ''):
                            try:
                                thickness_mm = float(raw_mm)
                            except (ValueError, TypeError):
                                thickness_mm = None
                        cls._od_to_dn_mapping[od] = dn
                        cls._od_candidate_rows.setdefault(od, []).append({
                            'od': od,
                            'dn': dn,
                            'thickness_code': thickness_code,
                            'thickness_mm': thickness_mm,
                        })
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
                size_config = config.get('size_processing', {})
                nps_config = size_config.get('nps_to_dn', {})
                cls._nps_to_dn_mapping = {str(k): int(v) for k, v in nps_config.items()}
                cls._common_dn_values = {
                    int(v) for v in size_config.get('common_dn_values', [])
                    if str(v).strip()
                }
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

        # 兜底规则：当英制尺寸未配置且为纯数字时，按 inch * 25 转换为 DN。
        # 例如：9" -> DN225, 11" -> DN275, NPS13 -> DN325
        # 分数/混合分数会优先在 _normalize_nps_token 中归一后走映射表。
        if re.fullmatch(r'\d+(?:\.\d+)?', normalized):
            inch_val = float(normalized)
            if inch_val > 0:
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
        # 小数点周围脏空格：2 .5 -> 2.5
        t = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', t)
        # 混合分数另一种脏写法：1.1/2 -> 1-1/2
        t = re.sub(r'^(\d+)\.(\d+/\d+)$', r'\1-\2', t)
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

    def _is_common_dn_integer(self, token: Any) -> bool:
        text = self._normalize_number_text(token)
        if not re.fullmatch(r'\d+', text):
            return False
        return int(text) in self._common_dn_values
    
    def _resolve_candidate_rows(self, rows: List[Dict[str, Any]], thickness_mm: Optional[float] = None) -> Optional[int]:
        """在同一 OD 对应多条候选记录时确定 DN。

        同一外径在对照表里常常同时映射到多个公称直径（如 OD720 → DN700/DN800），
        壁厚也无法唯一区分。此时按物理规则选择：取「小于该外径且最接近外径」的公称直径
        （DN 通常略小于 OD）；若候选 DN 全部不小于外径，则退而取最小的 DN。
        """
        if not rows:
            return None
        distinct_dns = sorted({int(row['dn']) for row in rows})
        if len(distinct_dns) == 1:
            return distinct_dns[0]
        od_value = float(rows[0].get('od'))
        below = [dn for dn in distinct_dns if dn < od_value]
        if below:
            return max(below)
        return min(distinct_dns)

    def _od_to_dn(self, od_value: float, thickness_mm: Optional[float] = None) -> Optional[int]:
        """
        将管外径转换为公称直径。

        规则：
        1. 先精确匹配映射表
        2. 再做小容差匹配（±0.5mm）
        3. 若同一 OD 命中多条记录（对应多个不同 DN），取「小于该外径且最接近外径」的 DN
           （DN 通常略小于 OD；候选 DN 全部不小于外径时退而取最小 DN）
        4. 若仍未命中，则按工程直径规则，取「小于等于当前 OD 数值」且最接近的 DN
        5. 若不存在满足条件的 DN，则返回 None，由上层决定是否回退原始 OD
        """
        if not self._od_to_dn_mapping:
            return None
        
        # 精确匹配
        if od_value in self._od_candidate_rows:
            return self._resolve_candidate_rows(self._od_candidate_rows[od_value], thickness_mm)
        
        # 容差匹配（±0.5mm）
        for od, rows in self._od_candidate_rows.items():
            if abs(od - od_value) <= 0.5:
                return self._resolve_candidate_rows(rows, thickness_mm)

        # 工程直径回退：按 DN 数值比较，取不大于当前 OD 且最接近的 DN
        dn_candidates = [int(dn) for dn in self._od_to_dn_mapping.values() if float(dn) <= od_value]
        if dn_candidates:
            return max(dn_candidates)

        return None

    @staticmethod
    def _extract_mm_context(value: Dict[str, Any]) -> List[float]:
        """提取挂在 SIZE 结构上的显式 mm 壁厚上下文。"""
        raw = value.get('_THICKNESS_MM_CONTEXT')
        if not isinstance(raw, list):
            return []
        result: List[float] = []
        for item in raw:
            try:
                result.append(float(item))
            except (ValueError, TypeError):
                continue
        return result

    @staticmethod
    def _pick_mm_for_index(mm_values: List[float], index: int) -> Optional[float]:
        if not mm_values:
            return None
        if len(mm_values) == 1:
            return float(mm_values[0])
        if 0 <= index < len(mm_values):
            return float(mm_values[index])
        return float(mm_values[-1])

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

    def _normalize_length_value(self, raw_value: Any, unit: str = "") -> str:
        """
        长度归一：
        1. 无单位：按毫米默认值直接保留，不做换算
        2. MM/毫米：直接保留
        3. CM/厘米：转毫米
        4. M/米：转毫米
        """
        text = self._normalize_number_text(raw_value)
        try:
            value = float(text)
        except Exception:
            return str(raw_value).strip()

        unit_u = str(unit or "").strip().upper()
        if unit_u in ("", "MM", "毫米"):
            return self._normalize_number_text(value)
        if unit_u in ("CM", "厘米"):
            return self._normalize_number_text(value * 10)
        if unit_u in ("M", "米"):
            return self._normalize_number_text(value * 1000)
        return self._normalize_number_text(value)

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

    def _build_positioned_size_items(self, value: Dict[str, Any], original_text: str = "") -> List[Dict[str, Any]]:
        """
        构建带位置的尺寸对象列表。

        规则：
        1. 若显式存在 DN，则只保留 DN 参与编码/壁厚换算
        2. 若无 DN，则使用可转换为 DN 的 OD / INCH，并按原文位置排序
        3. OD 优先按映射表、公差或向下最近工程直径转换；仅在完全找不到更小映射外径时才保留 OD 原值回退
        """
        dn_items: List[Dict[str, Any]] = []
        converted_items: List[Dict[str, Any]] = []
        od_fallback_items: List[Dict[str, Any]] = []
        mm_context = self._extract_mm_context(value)

        for item in self._ensure_list(value.get("DN")):
            raw = item.get("value") if isinstance(item, dict) else item
            raw_text = str(raw or "").strip()
            dn = self._extract_numeric_value(item)
            if dn is None:
                continue
            dn_items.append({
                "subtype": "DN",
                "raw": raw_text,
                "encode_value": float(dn),
                "dn_value": float(dn),
            })

        for idx, item in enumerate(self._ensure_list(value.get("OD"))):
            raw = item.get("value") if isinstance(item, dict) else item
            raw_text = str(raw or "").strip()
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od, thickness_mm=self._pick_mm_for_index(mm_context, idx))
            if dn is not None:
                converted_items.append({
                    "subtype": "OD",
                    "raw": raw_text,
                    "encode_value": float(dn),
                    "dn_value": float(dn),
                })
            else:
                od_fallback_items.append({
                    "subtype": "OD",
                    "raw": raw_text,
                    "encode_value": float(od),
                    "dn_value": None,
                })

        for item in self._ensure_list(value.get("INCH")):
            raw = item.get("value") if isinstance(item, dict) else item
            raw_text = str(raw or "").strip()
            if not raw_text:
                continue
            dn = self._nps_to_dn(raw_text)
            if dn is not None:
                converted_items.append({
                    "subtype": "INCH",
                    "raw": raw_text,
                    "encode_value": float(dn),
                    "dn_value": float(dn),
                })

        if dn_items:
            return dn_items
        if converted_items:
            return converted_items
        if od_fallback_items:
            return od_fallback_items
        return []

    def _extract_length_prefix(self, value: Any, original_text: str = "") -> str:
        """
        从 SIZE 中提取 LENGTH，并格式化为编码片段 `L数字`。
        不参与 DN/OD/INCH 的尺寸计算，仅用于最终 SIZE code 拼接。
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
            # 区间长度（如 1000~2000mm / 1000-2000 / 1000至2000）取较大端作为长度
            range_m = re.search(
                r'(\d+(?:\.\d+)?)\s*(?:MM|CM|M|毫米|厘米|米)?\s*[~～\-至到]\s*(\d+(?:\.\d+)?)\s*(MM|CM|M|毫米|厘米|米)?',
                s,
            )
            if range_m:
                bigger = max(float(range_m.group(1)), float(range_m.group(2)))
                return f"L{self._normalize_length_value(bigger, range_m.group(3) or '')}"
            m = re.search(r'(?:L|LEN|LENGTH)\s*=?\s*(\d+(?:\.\d+)?)(?:\s*(MM|CM|M|毫米|厘米|米))?', s)
            if m:
                return f"L{self._normalize_length_value(m.group(1), m.group(2) or '')}"

            # 结构化 SIZE.LENGTH 常常只有纯长度值，如 `1000mm` / `1000`
            # 这类值在字段层面已经被识别为 LENGTH，可直接转成编码片段。
            m = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*(MM|CM|M|毫米|厘米|米))?', s)
            if not m:
                return ""
            return f"L{self._normalize_length_value(m.group(1), m.group(2) or '')}"

        if isinstance(value, dict):
            items = self._ensure_list(value.get("LENGTH"))
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
        """公开方法：提取 SIZE.LENGTH 对应的编码片段 `L数字`。"""
        return self._extract_length_prefix(value, original_text=original_text)

    def _append_length_suffix(self, code: str, length_token: str) -> str:
        if not length_token:
            return code or ""
        if not code:
            return length_token
        return f"{code}{length_token}"

    def _collect_structured_values(self, value: Dict[str, Any], original_text: str = "") -> List[float]:
        """
        处理新 schema 的 SIZE 结构：
        - DN 直接取值
        - OD / INCH 先转换为 DN
        - 最终统一按原始出现顺序去重
        """
        ordered_values = self._collect_ordered_item_values(value)
        if ordered_values:
            return self._sort_sizes(ordered_values)
        return self._sort_sizes([float(item["encode_value"]) for item in self._build_positioned_size_items(value, original_text=original_text)])

    def _analyze_structured_values(self, value: Dict[str, Any], original_text: str = "") -> Tuple[List[float], bool]:
        """
        分析结构化 SIZE 值。

        规则：
        1. 若显式给出了 DN，则编码仅使用 DN。
        2. 若同时存在 OD/INCH，先换算成 DN；只要换算结果不在显式 DN 集合中，则标记待审。
        3. 若没有 DN，则优先按映射、公差或向下最近工程直径将 OD/INCH 换算为 DN 后编码。
        """
        ordered_values = self._collect_ordered_item_values(value)
        if ordered_values:
            return self._sort_sizes(ordered_values), False

        explicit_dn_values: List[float] = []
        converted_values: List[float] = []
        od_fallback_values: List[float] = []
        mm_context = self._extract_mm_context(value)

        for item in self._ensure_list(value.get("DN")):
            dn = self._extract_numeric_value(item)
            if dn is not None:
                explicit_dn_values.append(float(dn))

        for idx, item in enumerate(self._ensure_list(value.get("OD"))):
            od = self._extract_numeric_value(item)
            if od is None:
                continue
            dn = self._od_to_dn(od, thickness_mm=self._pick_mm_for_index(mm_context, idx))
            if dn is not None:
                converted_values.append(float(dn))
            else:
                od_fallback_values.append(float(od))

        for item in self._ensure_list(value.get("INCH")):
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

    def _collect_ordered_item_values(self, value: Dict[str, Any]) -> List[float]:
        items = value.get("_ITEMS")
        if not isinstance(items, list):
            return []

        has_explicit_dn = any(
            isinstance(item, dict)
            and str(item.get("type", "")).strip().upper() == "DN"
            and item.get("value") not in (None, "")
            for item in items
        )

        result: List[float] = []
        mm_context = self._extract_mm_context(value)
        od_idx = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            subtype = str(item.get("type", "")).strip().upper()
            raw_value = item.get("value")
            if raw_value in (None, ""):
                continue

            if has_explicit_dn and subtype != "DN":
                continue

            encoded: Optional[float] = None
            if subtype == "DN":
                encoded = self._extract_numeric_value(raw_value)
            elif subtype == "OD":
                od = self._extract_numeric_value(raw_value)
                if od is not None:
                    dn = self._od_to_dn(od, thickness_mm=self._pick_mm_for_index(mm_context, od_idx))
                    encoded = float(dn) if dn is not None else float(od)
                    od_idx += 1
            elif subtype == "INCH":
                dn = self._nps_to_dn(str(raw_value).strip())
                if dn is not None:
                    encoded = float(dn)
            elif subtype == "LENGTH":
                continue

            if encoded is None:
                continue
            result.append(float(encoded))
        return result
    
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

        # 单值英制（有 NPS 前缀，或包含英寸引号 / in / inch）
        has_inch_quote = any(ch in normalized_value for ch in ['"'])
        has_inch_word = bool(re.search(r'(?i)\bIN(?:CH)?\b', normalized_value))
        imperial_single = re.search(
            r'(?<![A-Za-z0-9])(?:NPS\s*)?(\d+(?:\s*[-\s]\s*\d+/\d+|\s*/\s*\d+|\.\d+)?)\s*["]?(?![A-Za-z0-9/])',
            normalized_value,
            re.IGNORECASE
        )
        if imperial_single and (has_inch_quote or has_inch_word or re.search(r'(?i)\bNPS\b', normalized_value)):
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
            return self._append_length_suffix(self.format_code(values), length_prefix)
        if isinstance(value, list):
            return self.process_multi(value, original_text=original_text)

        result = self.parse(str(value))
        return self._append_length_suffix(self.format_code(result.values), length_prefix)

    def extract_by_rules(self, text: str) -> RuleSizeExtraction:
        """
        基于显式锚点和高频稳定结构做尺寸抽取。

        规则：
        1. 显式 DN/NPS/英制/OD/φ/Φ/D/LENGTH 直接走规则
        2. `DNx整数` 视为双 DN
        3. `DN-整数` 若第二段在常见 DN 表中，也视为双 DN
        4. `DN-小数` 仅保留首个显式 DN，第二段留给壁厚规则
        3. `DNx小数` 仅保留首个显式 DN，第二段不进入尺寸
        4. `OD/φ/Φ/D x 小数` 视为外径+壁厚，尺寸只保留第一段
        5. 不处理无锚点裸数字 / 裸 AxB
        """
        source = str(text or "")
        normalized = source.replace('”', '"').replace('“', '"').replace('″', '"')
        normalized = self._normalize_section_labels(normalized)
        normalized = self._normalize_glued_dn_mm(normalized)

        # 0) 只有当脏小数串实际落在尺寸片段里，才让尺寸规则失效。
        # 例如：
        # - OD108.73.2      -> 影响尺寸，应失效
        # - DN250×6.31.5... -> DN250 本身未受影响，不应整条尺寸失效
        if self._has_malformed_size_fragment(normalized):
            return RuleSizeExtraction(
                dn=[],
                od=[],
                inch=[],
                length=[],
                size_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )

        # 0. 显式复杂复合规格直接留给大模型，例如：
        # DN114.3x114.3x60.3x6.02 / D323.5x219.1x114.3x8.74
        # 这类四段及以上结构当前不进入规则路径。
        if self._has_complex_composite_size(normalized) and not self._has_phi_dual_od_dual_thk_structure(normalized):
            return RuleSizeExtraction(
                dn=[],
                od=[],
                inch=[],
                length=[],
                size_code="",
                matched_texts=[],
                matched_spans=[],
                ordered_items=[],
            )
        explicit_result = self._extract_explicit_size_rules(normalized)
        dn_values = explicit_result["dn"]
        od_values = explicit_result["od"]
        inch_values = explicit_result["inch"]
        length_values = explicit_result["length"]
        matched_texts = explicit_result["matched_texts"]
        matched_spans = explicit_result["matched_spans"]
        ordered_items = explicit_result["ordered_items"]

        self._apply_bare_size_fallback(
            normalized=normalized,
            dn_values=dn_values,
            od_values=od_values,
            inch_values=inch_values,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
            ordered_items=ordered_items,
        )
        sorted_ordered_items = sorted(ordered_items, key=lambda x: (x["span"][0], x["span"][1]))
        dn_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "DN"]
        od_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "OD"]
        inch_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "INCH"]
        length_values = [str(item["value"]) for item in sorted_ordered_items if str(item.get("type") or "").upper() == "LENGTH"]
        consumed_spans = self._derive_consumed_spans_from_ordered_items(normalized, ordered_items)

        # 5.5 主体规格使用校验：
        # 若描述中存在裸复合主体规格，但规则没有拿到显式 DN，
        # 说明规则结果只是从别的弱片段（如尾部 10"、单独 OD）捞出来的，
        # 这类结果整体作废，交给大模型。
        if self._has_unconsumed_bare_complex_spec(normalized) and not dn_values and not od_values:
            return RuleSizeExtraction(
                dn=[],
                od=[],
                inch=[],
                length=[],
                size_code="",
                matched_texts=[],
                matched_spans=[],
                consumed_spans=[],
                ordered_items=[],
            )

        # 6. 生成规则尺寸编码
        code_values: List[str] = []
        if dn_values:
            code_values = dn_values
        elif od_values:
            converted: List[str] = []
            for item in od_values:
                numeric = self._extract_numeric_value(item)
                if numeric is None:
                    continue
                mapped = self._od_to_dn(float(numeric))
                if mapped is not None:
                    converted.append(self._normalize_number_text(mapped))
                else:
                    converted.append(self._normalize_number_text(item))
            code_values = converted
        elif inch_values:
            converted = []
            for item in inch_values:
                mapped = self._nps_to_dn(item)
                if mapped is not None:
                    converted.append(self._normalize_number_text(mapped))
            code_values = converted

        size_code = self.format_code([float(v) for v in code_values if re.fullmatch(r'\d+(?:\.\d+)?', v)]) if code_values else ""
        if length_values:
            # 编码取区间大端（_extract_length_prefix 内部已处理区间 → L{大端}）
            size_code = self._append_length_suffix(size_code, self._extract_length_prefix(length_values[0]))

        result = RuleSizeExtraction(
            dn=dn_values,
            od=od_values,
            inch=inch_values,
            length=length_values,
            size_code=size_code,
            matched_texts=matched_texts,
            matched_spans=matched_spans,
            consumed_spans=consumed_spans,
            ordered_items=[
                {"type": item["type"], "value": item["value"]}
                for item in sorted_ordered_items
            ],
        )
        return result

    @staticmethod
    def _derive_consumed_spans_from_ordered_items(text: str, ordered_items: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        consumed_spans: List[Tuple[int, int]] = []
        cursor_by_span: Dict[Tuple[int, int], int] = {}
        numeric_token_re = re.compile(r'\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?')
        # 长度允许区间写法（1000~2000），消费 span 需覆盖整个区间，避免大端被当成未消费残留
        length_token_re = re.compile(r'\d+(?:\.\d+)?(?:\s*[~～\-至到]\s*\d+(?:\.\d+)?)?')
        schedule_like_re = re.compile(r'(?i)SCH\s*\.?\s*(?:\d+S?|STD|XS|XXS)|S-(?:\d+S?|STD|XS|XXS)|S\d+S?|\d+S|STD|XS|XXS')

        for item in ordered_items:
            span = item.get("span")
            if not span or not isinstance(span, tuple) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            if start < 0 or end > len(text) or start >= end:
                continue
            full_text = text[start:end]
            cursor = cursor_by_span.get((start, end), 0)
            search_text = full_text[cursor:]
            item_type = str(item.get("type") or "").upper()
            token_match = None
            if item_type == "LENGTH":
                token_match = length_token_re.search(search_text)
            elif item_type in {"DN", "OD", "MM", "PRESSURE", "INCH"}:
                token_match = numeric_token_re.search(search_text)
            elif item_type == "SCHEDULE":
                token_match = schedule_like_re.search(search_text)
            if not token_match:
                continue
            token_start = start + cursor + token_match.start()
            token_end = start + cursor + token_match.end()
            candidate = (token_start, token_end)
            if candidate not in consumed_spans:
                consumed_spans.append(candidate)
            cursor_by_span[(start, end)] = cursor + token_match.end()

        return consumed_spans

    def _extract_explicit_size_rules(self, normalized: str) -> Dict[str, Any]:
        normalized = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', normalized)
        normalized = re.sub(r'(?<!\d)(\d+)\.(\d+/\d+)(?=\s*")', r'\1-\2', normalized)
        imperial_token = r'(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)'

        dn_values: List[str] = []
        od_values: List[str] = []
        inch_values: List[str] = []
        length_values: List[str] = []
        matched_texts: List[str] = []
        matched_spans: List[Tuple[int, int]] = []
        ordered_items: List[Dict[str, Any]] = []

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        def _add_ordered_item(item_type: str, value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        def _is_astm_d_context(start: int) -> bool:
            prefix = normalized[max(0, start - 12):start]
            return bool(re.search(r'(?i)ASTM\s*$', prefix))

        mm_length_candidates: List[str] = []
        generic_length_candidates: List[str] = []
        consumed_length_spans: List[Tuple[int, int]] = []

        def _overlaps_consumed_length(span: Tuple[int, int]) -> bool:
            for start, end in consumed_length_spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        def _overlaps_spans(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
            for start, end in spans:
                if span[0] < end and start < span[1]:
                    return True
            return False

        cn_range_pattern = re.compile(
            r'长度\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:MM|mm)?\s*[~\-]\s*(\d+(?:\.\d+)?)\s*(?:MM|mm)?',
            re.IGNORECASE,
        )
        for m in cn_range_pattern.finditer(normalized):
            span = (m.start(), m.end())
            # 区间长度按原样保留（如 1000~2000），编码阶段再取大端；
            # 消费 span 覆盖整个区间，避免大端被残留数字校验误判为未消费规格
            interval = f"{self._normalize_number_text(m.group(1))}~{self._normalize_number_text(m.group(2))}"
            mm_length_candidates.append(interval)
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)(\d+(?:\.\d+)?)\s*MM\s*LENGTH\b', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            mm_length_candidates.append(self._normalize_number_text(m.group(1)))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'长度\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:MM|mm)', normalized, re.IGNORECASE):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            mm_length_candidates.append(self._normalize_number_text(m.group(1)))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)\bLENGTH\s*[:=]?\s*(\d+(?:\.\d+)?)\s*MM', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            mm_length_candidates.append(self._normalize_number_text(m.group(1)))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)\bLENGTH\s*[:=]?\s*(\d+(?:\.\d+)?)(?:\s*(MM|CM|M|毫米|厘米|米))?\b', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            generic_length_candidates.append(self._normalize_length_value(m.group(1), m.group(2) or ''))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)\bCUT\s*[-]?\s*TO\s*(\d+(?:\.\d+)?)\b', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            mm_length_candidates.append(self._normalize_number_text(m.group(1)))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)(?<![A-Z0-9])L\s*=\s*(\d+(?:\.\d+)?)\s*MM', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            mm_length_candidates.append(self._normalize_number_text(m.group(1)))
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        for m in re.finditer(r'(?i)(?<![A-Z0-9])L\s*=\s*(\d+(?:\.\d+)?)\s*(CM|M|米)?\b', normalized):
            span = (m.start(), m.end())
            if _overlaps_consumed_length(span):
                continue
            value = self._normalize_length_value(m.group(1), m.group(2) or '')
            generic_length_candidates.append(value)
            consumed_length_spans.append(span)
            _record(m.group(0), span)
        if mm_length_candidates:
            _add_unique(length_values, mm_length_candidates[0])
            if consumed_length_spans:
                _add_ordered_item("LENGTH", mm_length_candidates[0], consumed_length_spans[0])
        elif generic_length_candidates:
            _add_unique(length_values, generic_length_candidates[0])
            if consumed_length_spans:
                _add_ordered_item("LENGTH", generic_length_candidates[0], consumed_length_spans[0])

        dn_pair_mm_pair_pattern = re.compile(
            r'(?i)(?<![A-Z0-9])DN\s*(\d+(?:\.\d+)?)\s*[xX×*/]\s*DN\s*'
            r'(' + '|'.join(map(re.escape, sorted((str(v) for v in self._common_dn_values), key=len, reverse=True))) + r')'
            r'(?=\d+\.\d+\s*MM(?:\b|\s*[xX×/,;)]))'
        ) if self._common_dn_values else None
        consumed_pair_spans: List[Tuple[int, int]] = []
        if dn_pair_mm_pair_pattern:
            for m in dn_pair_mm_pair_pattern.finditer(normalized):
                span = (m.start(), m.end())
                first_value = self._normalize_number_text(m.group(1))
                second_value = self._normalize_number_text(m.group(2))
                _add_unique(dn_values, first_value)
                _add_unique(dn_values, second_value)
                _add_ordered_item("DN", first_value, m.span(1))
                _add_ordered_item("DN", second_value, m.span(2))
                consumed_pair_spans.append(span)
                _record(m.group(0), span)

        dn_dash_pair_pattern = re.compile(
            rf'(?i)(?<![A-Z0-9])DN\s*(\d+(?:\.\d+)?)\s*-\s*(\d+\.\d+|{"|".join(map(re.escape, sorted((str(v) for v in self._common_dn_values), key=len, reverse=True))) if self._common_dn_values else r"\\d+"})'
            r'(?!\s*(?:MM|毫米))'
        )
        for m in dn_dash_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            first = m.group(1)
            second = m.group(2)
            if any(start <= span[0] and span[1] <= end for start, end in consumed_pair_spans):
                continue
            first_value = self._normalize_number_text(first)
            _add_unique(dn_values, first_value)
            _add_ordered_item("DN", first_value, m.span(1))
            if "." not in second and self._is_common_dn_integer(second):
                second_value = self._normalize_number_text(second)
                _add_unique(dn_values, second_value)
                _add_ordered_item("DN", second_value, m.span(2))
            consumed_pair_spans.append(span)
            _record(m.group(0), span)

        dn_pair_pattern = re.compile(
            rf'(?i)(?<![A-Z0-9])DN\s*(\d+(?:\.\d+)?)\s*[xX×*/]\s*(?:DN\s*)?({"|".join(map(re.escape, sorted((str(v) for v in self._common_dn_values), key=len, reverse=True))) if self._common_dn_values else r"\\d+"})(?!\.\d)(?!\s*(?:MM|毫米))'
        )
        for m in dn_pair_pattern.finditer(normalized):
            first, second = m.group(1), m.group(2)
            span = (m.start(), m.end())
            if any(start <= span[0] and span[1] <= end for start, end in consumed_pair_spans):
                continue
            first_value = self._normalize_number_text(first)
            _add_unique(dn_values, first_value)
            _add_ordered_item("DN", first_value, m.span(1))
            if '.' not in second:
                second_value = self._normalize_number_text(second)
                _add_unique(dn_values, second_value)
                _add_ordered_item("DN", second_value, m.span(2))
            consumed_pair_spans.append(span)
            _record(m.group(0), span)

        dn_single_pattern = re.compile(
            rf'(?i)(?<![A-Z0-9])DN\s*({"|".join(map(re.escape, sorted((str(v) for v in self._common_dn_values), key=len, reverse=True))) if self._common_dn_values else r"\\d+"})(?!\.\d)'
        )
        for m in dn_single_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if any(start <= span[0] and span[1] <= end for start, end in consumed_pair_spans):
                continue
            dn_value = self._normalize_number_text(m.group(1))
            _add_unique(dn_values, dn_value)
            _add_ordered_item("DN", dn_value, span)
            _record(m.group(0), span)

        has_explicit_dn_anchor = bool(dn_values)

        consumed_d_spans: List[Tuple[int, int]] = []
        if not has_explicit_dn_anchor:
            d_pair_pattern = re.compile(
                r'(?i)\bD\s*(\d+(?:\.\d+)?)\s*[xX×]\s*\bD\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\b'
            )
            for m in d_pair_pattern.finditer(normalized):
                first, second = m.group(1), m.group(2)
                span = (m.start(), m.end())
                if _is_astm_d_context(m.start()):
                    continue
                if self._is_common_dn_integer(first) and self._is_common_dn_integer(second):
                    first_value = self._normalize_number_text(first)
                    second_value = self._normalize_number_text(second)
                    _add_unique(dn_values, first_value)
                    _add_unique(dn_values, second_value)
                    _add_ordered_item("DN", first_value, m.span(1))
                    _add_ordered_item("DN", second_value, m.span(2))
                    consumed_d_spans.append(span)
                    _record(m.group(0), span)

        # D数字x数字 / D数字xD数字：
        # 若前两段都在 common DN 表中，则默认 D 表示 DN，而不是 OD。
        consumed_d_dn_pair_spans: List[Tuple[int, int]] = []
        if not has_explicit_dn_anchor:
            d_dn_pair_pattern = re.compile(
                r'(?i)\bD\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(?:\bD\s*)?(\d+(?:\.\d+)?)\b'
            )
            for m in d_dn_pair_pattern.finditer(normalized):
                first, second = m.group(1), m.group(2)
                span = (m.start(), m.end())
                if _is_astm_d_context(m.start()):
                    continue
                if _overlaps_spans(span, consumed_d_spans):
                    continue
                if self._is_common_dn_integer(first) and self._is_common_dn_integer(second):
                    first_value = self._normalize_number_text(first)
                    second_value = self._normalize_number_text(second)
                    _add_unique(dn_values, first_value)
                    _add_unique(dn_values, second_value)
                    _add_ordered_item("DN", first_value, m.span(1))
                    _add_ordered_item("DN", second_value, m.span(2))
                    consumed_d_dn_pair_spans.append(span)
                    _record(m.group(0), span)

        # 结构族：ΦA×B/T1×T2 或 ΦA×ΦB/T1×T2
        # 前两段按双外径处理，后两段交给壁厚规则。
        phi_dual_od_dual_thk_pattern = re.compile(
            r'(?i)[φΦФф]\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(?:[φΦФф]\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)'
        )
        consumed_phi_dual_spans: List[Tuple[int, int]] = []
        for m in phi_dual_od_dual_thk_pattern.finditer(normalized):
            span = (m.start(), m.end())
            first_value = self._normalize_number_text(m.group(1))
            second_value = self._normalize_number_text(m.group(2))
            _add_unique(od_values, first_value)
            _add_unique(od_values, second_value)
            _add_ordered_item("OD", first_value, m.span(1))
            _add_ordered_item("OD", second_value, m.span(2))
            consumed_phi_dual_spans.append(span)
            _record(m.group(0), span)

        od_double_head_three_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(?:\bOD|[φΦФф]|\bD)\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\b'
        )
        od_three_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф]|\bD)\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\b'
        )
        consumed_od_spans: List[Tuple[int, int]] = []
        for m in od_double_head_three_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _is_astm_d_context(m.start()):
                continue
            if _overlaps_spans(span, consumed_phi_dual_spans):
                continue
            if _overlaps_spans(span, consumed_d_spans):
                continue
            if _overlaps_spans(span, consumed_d_dn_pair_spans):
                continue
            first_value = self._normalize_number_text(m.group(1))
            second_value = self._normalize_number_text(m.group(2))
            _add_unique(od_values, first_value)
            _add_unique(od_values, second_value)
            _add_ordered_item("OD", first_value, m.span(1))
            _add_ordered_item("OD", second_value, m.span(2))
            consumed_od_spans.append(span)
            _record(m.group(0), span)
        for m in od_three_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _is_astm_d_context(m.start()):
                continue
            if _overlaps_spans(span, consumed_phi_dual_spans):
                continue
            if _overlaps_spans(span, consumed_d_spans):
                continue
            if _overlaps_spans(span, consumed_d_dn_pair_spans):
                continue
            if _overlaps_spans(span, consumed_od_spans):
                continue
            first_value = self._normalize_number_text(m.group(1))
            second_value = self._normalize_number_text(m.group(2))
            _add_unique(od_values, first_value)
            _add_unique(od_values, second_value)
            _add_ordered_item("OD", first_value, m.span(1))
            _add_ordered_item("OD", second_value, m.span(2))
            consumed_od_spans.append(span)
            _record(m.group(0), span)

        if not has_explicit_dn_anchor:
            d_od_pair_pattern = re.compile(
                r'(?i)\bD\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(?:\bD\s*)?(\d+(?:\.\d+)?)\s*(?:MM)?(?!\s*[xX×]\s*\d)'
            )
            for m in d_od_pair_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _is_astm_d_context(m.start()):
                    continue
                if _overlaps_spans(span, consumed_d_spans):
                    continue
                if _overlaps_spans(span, consumed_d_dn_pair_spans):
                    continue
                if _overlaps_spans(span, consumed_od_spans):
                    continue
                od_value = self._normalize_number_text(m.group(1))
                _add_unique(od_values, od_value)
                _add_ordered_item("OD", od_value, span)
                consumed_od_spans.append(span)
                _record(m.group(0), span)

        od_pair_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф])\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:MM)?(?!\s*[xX×]\s*\d)'
        )
        for m in od_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _is_astm_d_context(m.start()):
                continue
            if _overlaps_spans(span, consumed_phi_dual_spans):
                continue
            if _overlaps_spans(span, consumed_d_spans):
                continue
            if _overlaps_spans(span, consumed_d_dn_pair_spans):
                continue
            if _overlaps_spans(span, consumed_od_spans):
                continue
            od_value = self._normalize_number_text(m.group(1))
            _add_unique(od_values, od_value)
            _add_ordered_item("OD", od_value, span)
            consumed_od_spans.append(span)
            _record(m.group(0), span)

        if not has_explicit_dn_anchor:
            d_od_schedule_pattern = re.compile(
                r'(?i)\bD\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(?:SCH[.\s]*\d+S?|SCH[.\s]*(?:STD|XS|XXS)|STD|XS|XXS|S-\d+S?|S-\d+)'
            )
            for m in d_od_schedule_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _is_astm_d_context(m.start()):
                    continue
                if _overlaps_spans(span, consumed_d_spans):
                    continue
                if _overlaps_spans(span, consumed_d_dn_pair_spans):
                    continue
                if _overlaps_spans(span, consumed_od_spans):
                    continue
                od_value = self._normalize_number_text(m.group(1))
                _add_unique(od_values, od_value)
                _add_ordered_item("OD", od_value, span)
                consumed_od_spans.append(span)
                _record(m.group(0), span)

        od_schedule_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф])\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(?:SCH[.\s]*\d+S?|SCH[.\s]*(?:STD|XS|XXS)|STD|XS|XXS|S-\d+S?|S-\d+)'
        )
        for m in od_schedule_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _is_astm_d_context(m.start()):
                continue
            if _overlaps_spans(span, consumed_d_spans):
                continue
            if _overlaps_spans(span, consumed_d_dn_pair_spans):
                continue
            if _overlaps_spans(span, consumed_od_spans):
                continue
            od_value = self._normalize_number_text(m.group(1))
            _add_unique(od_values, od_value)
            _add_ordered_item("OD", od_value, span)
            consumed_od_spans.append(span)
            _record(m.group(0), span)

        consumed_d_single_spans: List[Tuple[int, int]] = []
        if not has_explicit_dn_anchor:
            d_single_pattern = re.compile(r'(?i)\bD\s*(\d+(?:\.\d+)?)\b')
            for m in d_single_pattern.finditer(normalized):
                span = (m.start(), m.end())
                if _is_astm_d_context(m.start()):
                    continue
                if _overlaps_spans(span, consumed_d_spans):
                    continue
                if _overlaps_spans(span, consumed_d_dn_pair_spans):
                    continue
                if _overlaps_spans(span, consumed_od_spans):
                    continue
                value = m.group(1)
                if self._is_common_dn_integer(value):
                    dn_value = self._normalize_number_text(value)
                    _add_unique(dn_values, dn_value)
                    _add_ordered_item("DN", dn_value, span)
                else:
                    od_value = self._normalize_number_text(value)
                    _add_unique(od_values, od_value)
                    _add_ordered_item("OD", od_value, span)
                consumed_d_single_spans.append(span)
                _record(m.group(0), span)

        # 单值外径兜底：
        # 1. 允许常规边界结尾：Φ89; / OD60.3 空格 / 句末
        # 2. 允许后面紧跟 `X/×` 再接另一段显式外径锚点：
        #    这样即便双外径规则漏掉，`Φ133XΦ89` 也能分别兜底出 Φ133 与 Φ89，
        #    不会只剩下右半段。
        od_single_pattern = re.compile(
            r'(?i)(?:\bOD|[φΦФф])\s*(\d+(?:\.\d+)?)(?=\b|\s*[xX×]\s*(?:\bOD|[φΦФф]))'
        )
        for m in od_single_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if _is_astm_d_context(m.start()):
                continue
            if _overlaps_spans(span, consumed_phi_dual_spans):
                continue
            if _overlaps_spans(span, consumed_d_single_spans):
                continue
            if _overlaps_spans(span, consumed_d_spans):
                continue
            if _overlaps_spans(span, consumed_d_dn_pair_spans):
                continue
            if _overlaps_spans(span, consumed_od_spans):
                continue
            od_value = self._normalize_number_text(m.group(1))
            _add_unique(od_values, od_value)
            _add_ordered_item("OD", od_value, span)
            _record(m.group(0), span)

        inch_suffix_pattern = r'(?:["”″]|\'\s*\'|\bIN(?:CH)?\b)'

        inch_pair_pattern = re.compile(
            r'(?<![A-Za-z0-9])'
            rf'(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\s*{inch_suffix_pattern}?\s*[xX×*]\s*'
            rf'(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\s*{inch_suffix_pattern}'
            r'(?![A-Za-z0-9])'
            , re.IGNORECASE
        )
        consumed_inch_spans: List[Tuple[int, int]] = []
        for m in inch_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            first_value = self._normalize_nps_token(m.group(1))
            second_value = self._normalize_nps_token(m.group(2))
            _add_unique(inch_values, first_value)
            _add_unique(inch_values, second_value)
            _add_ordered_item("INCH", first_value, m.span(1))
            _add_ordered_item("INCH", second_value, m.span(2))
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        nps_pair_pattern = re.compile(
            r'(?i)(?<![A-Z0-9])NPS\s*(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)\s*(?:["])?\s*[xX×*]\s*(?:NPS\s*)?(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)'
        )
        for m in nps_pair_pattern.finditer(normalized):
            span = (m.start(), m.end())
            first_value = self._normalize_nps_token(m.group(1))
            second_value = self._normalize_nps_token(m.group(2))
            _add_unique(inch_values, first_value)
            _add_unique(inch_values, second_value)
            _add_ordered_item("INCH", first_value, m.span(1))
            _add_ordered_item("INCH", second_value, m.span(2))
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        for m in re.finditer(r'(?i)(?<![A-Z0-9])NPS\s*(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)', normalized):
            span = (m.start(), m.end())
            if any(start <= span[0] and span[1] <= end for start, end in consumed_inch_spans):
                continue
            inch_value = self._normalize_nps_token(m.group(1))
            _add_unique(inch_values, inch_value)
            _add_ordered_item("INCH", inch_value, span)
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        size_labeled_inch_pattern = re.compile(rf'(?i)\bSIZE\s*{imperial_token}\s*{inch_suffix_pattern}')
        for m in size_labeled_inch_pattern.finditer(normalized):
            span = (m.start(), m.end())
            inch_value = self._normalize_nps_token(m.group(1))
            _add_unique(inch_values, inch_value)
            _add_ordered_item("INCH", inch_value, span)
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        # 英寸锚点后允许紧跟壁厚、材质或中文描述，例如：
        # 10"SCH20、3"PIPE、3"管
        inch_quote_pattern = re.compile(rf'(?<![A-Za-z0-9./-]){imperial_token}\s*{inch_suffix_pattern}')
        for m in inch_quote_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if m.start() > 0 and normalized[m.start() - 1] in {'/', '-', '.'}:
                continue
            if any(start <= span[0] and span[1] <= end for start, end in consumed_inch_spans):
                continue
            prefix = normalized[max(0, m.start() - 16):m.start()]
            if re.search(r'(?i)NPS\s*\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?\s*$', prefix):
                continue
            inch_value = self._normalize_nps_token(m.group(1))
            _add_unique(inch_values, inch_value)
            _add_ordered_item("INCH", inch_value, span)
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        inch_word_pattern = re.compile(rf'(?i)(?<![A-Za-z0-9./-]){imperial_token}\s*(?:INCH|IN)\b')
        for m in inch_word_pattern.finditer(normalized):
            span = (m.start(), m.end())
            if any(start <= span[0] and span[1] <= end for start, end in consumed_inch_spans):
                continue
            prefix = normalized[max(0, m.start() - 16):m.start()]
            if re.search(r'(?i)NPS\s*\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?\s*$', prefix):
                continue
            inch_value = self._normalize_nps_token(m.group(1))
            _add_unique(inch_values, inch_value)
            _add_ordered_item("INCH", inch_value, span)
            consumed_inch_spans.append(span)
            _record(m.group(0), span)

        return {
            "dn": dn_values,
            "od": od_values,
            "inch": inch_values,
            "length": length_values,
            "matched_texts": matched_texts,
            "matched_spans": matched_spans,
            "ordered_items": ordered_items,
        }

    @staticmethod
    def _has_phi_dual_od_dual_thk_structure(text: str) -> bool:
        pattern = re.compile(
            r'(?i)[φΦФф]\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(?:[φΦФф]\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)'
        )
        for m in pattern.finditer(text):
            return True
        return False

    def _apply_bare_size_fallback(
        self,
        normalized: str,
        dn_values: List[str],
        od_values: List[str],
        inch_values: List[str],
        matched_texts: List[str],
        matched_spans: List[Tuple[int, int]],
        ordered_items: List[Dict[str, Any]],
    ) -> None:
        if dn_values or od_values or inch_values:
            return

        def _add_unique(items: List[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        def _record(match_text: str, span: Optional[Tuple[int, int]] = None) -> None:
            mt = str(match_text or "").strip()
            if mt and mt not in matched_texts:
                matched_texts.append(mt)
            if span and span not in matched_spans:
                matched_spans.append(span)

        def _add_ordered_item(item_type: str, value: str, span: Tuple[int, int]) -> None:
            candidate = {"type": item_type, "value": str(value), "span": span}
            if candidate not in ordered_items:
                ordered_items.append(candidate)

        flange_spec_blocks = self._extract_flange_spec_dn_pressure_blocks(normalized)
        for block in flange_spec_blocks:
            dn_value = self._normalize_number_text(block["dn"])
            _add_unique(dn_values, dn_value)
            _add_ordered_item("DN", dn_value, block["size_span"])
            _record(block["raw"], block["full_span"])

        bare_blocks = self._extract_bare_od_thickness_blocks(normalized)
        for block in bare_blocks:
            od_value = self._normalize_number_text(block["od"])
            _add_unique(od_values, od_value)
            _add_ordered_item("OD", od_value, block["span"])
            _record(block["raw"], block["span"])

        if dn_values or od_values or inch_values:
            return

        tail_dn_block = self._extract_tail_dn_fallback_block(normalized)
        if tail_dn_block:
            dn_value = self._normalize_number_text(tail_dn_block["dn"])
            _add_unique(dn_values, dn_value)
            _add_ordered_item("DN", dn_value, tail_dn_block["span"])
            _record(tail_dn_block["raw"], tail_dn_block["span"])

    @staticmethod
    def _has_complex_composite_size(text: str) -> bool:
        """
        对复合尺寸规格，三段及以上暂不走规则，直接留给大模型。
        例如：
        - DN114.3x114.3x60.3
        - DN114.3x114.3x60.3x6.02
        - D323.5x219.1x8.74
        - D323.5x219.1x114.3x8.74
        """
        patterns = [
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+\.\d+'),
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?(?:\s*[xX×*]\s*\d+(?:\.\d+)?){2,}'),
            # DN 复合尺寸中，第二个 DN 后若直接连小数厚度，整体交给大模型。
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+(?:\.\d+)?\s*[xX×*/]\s*DN\s*\d+(?=\d+\.\d+\s*(?:MM|毫米))'),
            re.compile(r'(?i)(?:\bD|[φΦФф]|(?:\bOD))\s*\d+(?:\.\d+)?(?:\s*[xX×]\s*\d+(?:\.\d+)?){2,}'),
            re.compile(r'(?i)(?:\bD|[φΦФф]|(?:\bOD))\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?'),
        ]
        return any(pattern.search(text) for pattern in patterns)

    @staticmethod
    def _has_malformed_size_fragment(text: str) -> bool:
        """
        识别真正影响尺寸解释的脏小数片段。

        仅当脏小数链出现在尺寸锚点片段里时才失效，例如：
        - OD108.73.2
        - φ108.73.2
        - D108.73.2
        - DN108.73.2

        不因为后续别的脏串就整条尺寸失效，例如：
        - DN250×6.31.5D12459
        这里 DN250 仍然可安全提取。
        """
        patterns = (
            re.compile(r'(?i)(?:\bOD|[φΦФфØø]|\bD)\s*\d+\.\d+\.\d+'),
            re.compile(r'(?i)(?<![A-Z0-9])DN\s*\d+\.\d+\.\d+'),
        )
        return any(p.search(text) for p in patterns)

    def _has_unconsumed_bare_complex_spec(self, text: str) -> bool:
        """
        裸复合主体规格存在，但当前规则若没拿到显式 DN，就不应仅靠别的弱尺寸片段出结果。
        例如：
        - 273x8.0/295x3.6;10"
        - 273x8.0/295x3.6;OD100
        """
        return bool(self._extract_bare_od_thickness_blocks(text))

    @staticmethod
    def _extract_bare_od_thickness_blocks(text: str) -> List[Dict[str, Any]]:
        """
        识别裸规格块：ODxTHK / ODxTHKBW / 多段串接中的单块。

        只负责识别“一个块”，不直接决定是否交给大模型。
        例：
        - 48x2.8
        - 610x9.53-323.9x9.53BW
        - 273x8.0；10"
        """
        pattern = re.compile(
            r'(?<![A-Za-z0-9])'
            r'(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+\.\d+)'
            r'(?!\s*")\s*(?:MM|毫米)?'
            r'(?=$|[^0-9])',
            re.IGNORECASE,
        )
        blocks: List[Dict[str, Any]] = []
        for m in pattern.finditer(text):
            # 不能从前一个小数的尾部起跳：
            # 168.3x7.11 不应截成 3x7.11
            if m.start() >= 2 and text[m.start() - 1] == '.' and text[m.start() - 2].isdigit():
                continue
            try:
                od = float(m.group(1))
                thk = float(m.group(2))
            except ValueError:
                continue
            blocks.append({
                "od": od,
                "thickness": thk,
                "raw": m.group(0).strip(),
                "span": (m.start(), m.end()),
            })
        return blocks

    def _extract_flange_spec_dn_pressure_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        识别法兰型号规格中的兜底组合，例如：
        - SO100(B)-10RF
        - SO100(B)- 10 RF

        语义：
        - 第一段数字兜底为 DN100
        - 第二段数字留给 pressure_processor 兜底为 PN10

        只在尺寸规则未命中任何显式结果时参与。
        """
        blocks: List[Dict[str, Any]] = []
        for combo in ComboFallbackExtractor.extract_flange_spec_rf_combos(
            text,
            allow_rf_right_glue=False,
        ):
            dn_raw = self._normalize_number_text(combo.first_value)
            try:
                dn_numeric = int(float(dn_raw))
            except Exception:
                continue
            if self._common_dn_values and dn_numeric not in self._common_dn_values:
                continue
            blocks.append({
                "raw": combo.raw,
                "dn": dn_raw,
                "pn": self._normalize_number_text(combo.second_value),
                "full_span": combo.full_span,
                "size_span": combo.first_span,
                "pressure_span": combo.second_span,
            })
        return blocks

    def _extract_tail_dn_fallback_block(self, text: str) -> Optional[Dict[str, Any]]:
        """
        最低优先级兜底：
        若描述末尾存在一个裸整数，且该整数属于 common_dn_values，
        则把它视为一个疑似 DN。

        例：
        - PIPE / WELDED / ASME B36.19M SCH5S / A312 TP304 / BE / POLISHING 50
        """
        if not self._common_dn_values:
            return None

        match = re.search(r'(?<![A-Za-z0-9./])(\d+)\s*$', str(text or ""))
        if not match:
            return None

        raw_value = self._normalize_number_text(match.group(1))
        try:
            numeric_value = int(float(raw_value))
        except Exception:
            return None

        if numeric_value not in self._common_dn_values:
            return None

        return {
            "raw": match.group(0).strip(),
            "dn": raw_value,
            "span": match.span(1),
        }

    @staticmethod
    def _normalize_section_labels(text: str) -> str:
        """
        处理历史表中常见的编号字段标签粘连：
        - DN50X253.连接方式 -> DN50X25 3.连接方式
        - 2.规格:DN50X253.连接方式 -> 2.规格:DN50X25 3.连接方式

        只在 `数字.` 后面紧跟中文/英文字段标签并带 `:`/`：` 时切开，
        不影响 B36.10 这类规范写法。
        """
        text = re.sub(
            r'(?<=[A-Za-z0-9])([1-9])\.(?=[\u4e00-\u9fffA-Za-z][^:：]{0,20}[:：])',
            r' \1.',
            text,
        )
        text = re.sub(r'(?<=[A-Za-z0-9.])(?=DN\s*\d)', ' ', text, flags=re.IGNORECASE)
        return text

    def _normalize_glued_dn_mm(self, text: str) -> str:
        """
        切开 `DN1506.3mm` 这类 `DN + 壁厚` 粘连：
        - DN200XDN1506.3mmX7.1mm -> DN200XDN150 6.3mmX7.1mm
        - DN150×DN40S-10S×SCH40S -> DN150×DN40 S-10S×SCH40S
        仅当 `DN<常见公称直径>` 后面紧跟 `小数mm` 时生效。
        """
        if not self._common_dn_values:
            return text
        dn_tokens = sorted((str(v) for v in self._common_dn_values), key=len, reverse=True)
        decimal_mm_pattern = re.compile(
            rf'(?i)(DN\s*)({"|".join(map(re.escape, dn_tokens))})(?=(\d+\.\d+\s*MM(?:\b|\s*[xX×/,;)])))'
        )
        text = decimal_mm_pattern.sub(r'\1\2 ', text)
        glued_schedule_pattern = re.compile(
            rf'(?i)(DN\s*)({"|".join(map(re.escape, dn_tokens))})(?=((?:S-\d+S?|SCH\d+S?|\d+S)\b))'
        )
        return glued_schedule_pattern.sub(r'\1\2 ', text)
    
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
            return self._append_length_suffix(self.format_code(self._sort_sizes(all_values)), length_prefix)
        
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
            return self._append_length_suffix(self.format_code(sorted_values), length_prefix)
        
        if other_results:
            return self._append_length_suffix(self.format_code(other_results[0].values), length_prefix)
        
        if phi_results:
            all_values = []
            for r in phi_results:
                all_values.extend(r.values)
            sorted_values = self._sort_sizes(all_values)
            return self._append_length_suffix(self.format_code(sorted_values), length_prefix)

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
            return self._append_length_suffix(self.format_code(normalized_values), length_prefix), need_review

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
            return self._append_length_suffix(self.format_code(self._sort_sizes(all_values)), length_prefix), need_review

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
        imperial_token = r'(\d+(?:\.\d+)?(?:[-\s]\d+/\d+|/\d+)?)'
