"""
壁厚对照表处理器。

根据以下信息查表获取毫米壁厚：
- 标准（推荐传结构化 STANDARD 对象数组）
- 公称直径 DN（可传 DN20 / 20 / [300, 200] / '300X200'）
- 壁厚号（可传 SCH80 / S80 / XS / 3MM / 'S30XS40' / ['S30', 'S40']）

查表优先级：
1. 先通过标准映射得到 5 个目标标准之一
2. 先按“标准简写”匹配表中记录
3. 若简写路径未命中，再按“适用标准”候选回退
4. 若仍未命中，保持原值

返回毫米值时默认不带单位，例如：'3.91'。
后续若编码流程需要，可由调用方自行拼接为 '3.91MM'。
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .size_processor import get_size_processor
from .thickness_processor import get_thickness_processor
from .standard_target_mapper import map_standard_to_target

logger = logging.getLogger(__name__)


class ThicknessTableProcessor:
    _table_df: Optional[pd.DataFrame] = None
    _instance: Optional["ThicknessTableProcessor"] = None

    # 目标标准 -> 适用标准回退候选（按优先级）
    APPLICABLE_STANDARD_FALLBACKS: Dict[str, List[str]] = {
        "AB3619": ["ASME B36.19-2022", "ANSI B36.10 B36.19"],
        "AB3610": ["ASME B36.10-2022", "ANSI B36.10 B36.19"],
        "HGT20553Ia": ["HG/T 20553(Ia)"],
        "HGT20553II": ["HG/T 20553(II)"],
        "SHT3405": ["SH/T 3405-2017"],
    }

    THICKNESS_TOKEN_RE = re.compile(r'(XXS|XS|STD|S\d+(?:\.\d+)?S?|\d+(?:\.\d+)?MM)', re.IGNORECASE)
    EXPLICIT_SEPARATOR_RE = re.compile(r'[×*/]|(?<!X)X(?!S)', re.IGNORECASE)

    def __init__(self):
        self._load_table()
        self.size_processor = get_size_processor()
        self.thickness_processor = get_thickness_processor()

    @classmethod
    def _load_table(cls) -> None:
        if cls._table_df is not None:
            return

        excel_path = Path(__file__).parent.parent / 'config' / '壁厚对照汇总表.xlsx'
        df = pd.read_excel(excel_path, header=0)
        df = df.copy()
        df['适用标准'] = df['适用标准'].fillna('').astype(str).str.strip()
        df['标准简写'] = df['标准简写'].fillna('').astype(str).str.strip()
        df['公称直径'] = df['公称直径'].apply(cls._normalize_dn_cell)
        df['壁厚号'] = df['壁厚号'].fillna('').astype(str).map(cls._normalize_lookup_code)
        df['壁厚'] = df['壁厚'].apply(cls._normalize_mm_cell)
        df['数据状态'] = df['数据状态'].fillna('').astype(str).str.strip()
        # 只保留已生效和空状态
        df = df[df['数据状态'].isin(['', '已生效'])]
        cls._table_df = df
        logger.info('[ThicknessTableProcessor] 加载壁厚对照表: %s 条', len(df))

    def map_standard_to_target(self, standards: Any) -> str:
        return map_standard_to_target(standards)

    def lookup_mm(self, standards: Any, dn: Any, thickness_code: Any) -> str:
        """单值查表：返回毫米值字符串；查不到返回空字符串。"""
        target = self.map_standard_to_target(standards)
        if not target:
            return ''

        dn_key = self._normalize_dn_value(dn)
        code_key = self._normalize_lookup_code(thickness_code)
        if not dn_key or not code_key:
            return ''

        row = self._lookup_row(target, dn_key, code_key)
        return row['壁厚'] if row is not None and row.get('壁厚') else ''

    def convert_to_mm_parts(self, standards: Any, dn_values: Any, thickness_values: Any, original_text: str = "") -> List[str]:
        """
        组合转换：返回按位置对齐后的毫米值列表。
        未命中则保留原壁厚片段。
        """
        details = self.convert_to_mm_details(
            standards,
            dn_values,
            thickness_values,
            original_text=original_text,
        )
        if not details:
            return []
        return [detail['converted'] or detail['source'] for detail in details]

    def convert_to_mm_details(self, standards: Any, dn_values: Any, thickness_values: Any, original_text: str = "") -> List[Dict[str, str]]:
        """
        组合转换明细：
        返回按位置对齐后的换算结果，包含 source / converted / dn / target_standard。
        未命中时 converted 为空，调用方可决定是否保留原值。
        """
        target = self.map_standard_to_target(standards)
        if not target:
            return []

        dn_items = self.build_dn_items(dn_values, original_text=original_text)
        thickness_items = self.build_thickness_items(thickness_values, original_text=original_text)

        if not thickness_items:
            return []
        if not dn_items:
            return []

        results: List[Dict[str, str]] = []

        if len(dn_items) == 1:
            pairs = [(dn_items[0], part) for part in thickness_items]
        elif len(dn_items) == len(thickness_items):
            pairs = zip(dn_items, thickness_items)
        else:
            logger.info(
                "[ThicknessTableProcessor] 跳过壁厚毫米换算：DN数量(%s) 与壁厚数量(%s) 不满足 1->N 或 N->N",
                len(dn_items),
                len(thickness_items),
            )
            return []

        for dn_item, thickness_item in pairs:
            dn_part = str(dn_item.get('value') or '').strip()
            thickness_part = str(thickness_item.get('normalized') or '').strip()
            mm = self.lookup_mm(standards, dn_part, thickness_part)
            results.append({
                'source': thickness_part,
                'converted': str(mm or '').strip(),
                'dn': dn_part,
                'source_type': str(thickness_item.get('type') or '').strip().upper(),
                'target_standard': target,
            })

        return results

    def convert_to_mm(self, standards: Any, dn_values: Any, thickness_values: Any, original_text: str = "") -> str:
        parts = self.convert_to_mm_parts(standards, dn_values, thickness_values, original_text=original_text)
        return 'X'.join(parts)

    def _lookup_row(self, target: str, dn_key: str, code_key: str) -> Optional[Dict[str, str]]:
        df = self._table_df
        if df is None or df.empty:
            return None

        # 1) 先按标准简写查
        sub = df[(df['标准简写'] == target) & (df['公称直径'] == dn_key) & (df['壁厚号'] == code_key)]
        if not sub.empty:
            return sub.iloc[0].to_dict()

        # 2) 简写没命中，再按适用标准回退（适配 ANSI 等空简写场景）
        for applicable in self.APPLICABLE_STANDARD_FALLBACKS.get(target, []):
            sub = df[(df['适用标准'] == applicable) & (df['公称直径'] == dn_key) & (df['壁厚号'] == code_key)]
            if not sub.empty:
                return sub.iloc[0].to_dict()

        return None

    @staticmethod
    def _normalize_dn_cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ''
        text = str(value).strip()
        m = re.search(r'(\d+(?:\.\d+)?)', text)
        if not m:
            return ''
        num = float(m.group(1))
        if num.is_integer():
            return str(int(num))
        return str(num).rstrip('0').rstrip('.')

    @staticmethod
    def _normalize_mm_cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ''
        text = str(value).strip()
        if not text:
            return ''
        try:
            num = float(text)
            if num.is_integer():
                return str(int(num))
            return str(num).rstrip('0').rstrip('.')
        except ValueError:
            return text

    @classmethod
    def _normalize_dn_value(cls, value: Any) -> str:
        if isinstance(value, dict):
            value = value.get('value')
        return cls._normalize_dn_cell(value)

    def build_dn_items(self, value: Any, original_text: str = "") -> List[Dict[str, str]]:
        dn_parts = self.size_processor.extract_dn_values(value, original_text=original_text)
        result: List[Dict[str, str]] = []
        seen: set[str] = set()
        for dn in dn_parts:
            normalized = self._normalize_dn_value(dn)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append({"type": "DN", "value": normalized})
        return result

    def build_thickness_items(self, value: Any, original_text: str = "") -> List[Dict[str, str]]:
        if isinstance(value, dict):
            items = value.get("_ITEMS")
            if isinstance(items, list) and items:
                normalized_items = self._normalize_thickness_item_list(items)
                if normalized_items:
                    return normalized_items
            normalized_parts = self._normalize_thickness_parts(value, original_text=original_text)
        else:
            normalized_parts = self._normalize_thickness_parts(value, original_text=original_text)

        result: List[Dict[str, str]] = []
        for part in normalized_parts:
            normalized = str(part or '').strip().upper()
            if not normalized:
                continue
            result.append({
                "type": self._infer_thickness_item_type(normalized),
                "value": normalized,
                "normalized": normalized,
            })
        return result

    def _normalize_thickness_item_list(self, items: List[Any]) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subtype = str(item.get("type") or "").strip().upper()
            raw_value = item.get("value")
            normalized = self._normalize_thickness_item(subtype, raw_value)
            if not normalized:
                continue
            result.append({
                "type": subtype or self._infer_thickness_item_type(normalized),
                "value": str(raw_value or "").strip(),
                "normalized": normalized,
            })
        return result

    @classmethod
    def _normalize_thickness_item(cls, subtype: str, raw_value: Any) -> str:
        text = str(raw_value or "").strip()
        if not text:
            return ""
        if subtype == "MM":
            mm_match = re.search(r'(\d+(?:\.\d+)?)', text.upper())
            if not mm_match:
                return ""
            return f"{cls._normalize_mm_cell(mm_match.group(1))}MM"
        if subtype in {"SCHEDULE", "SERIES"}:
            return cls._normalize_lookup_code(text)
        if subtype == "INCH":
            return re.sub(r'\s+', '', text.upper())
        if subtype == "BWG":
            return text.upper()
        return cls._normalize_lookup_code(text)

    @staticmethod
    def _infer_thickness_item_type(normalized: str) -> str:
        upper = str(normalized or "").strip().upper()
        if not upper:
            return ""
        if upper.endswith("MM"):
            return "MM"
        if upper in {"XS", "XXS", "STD"} or upper.startswith("SCH"):
            return "SCHEDULE"
        return "UNKNOWN"

    def _normalize_thickness_parts(self, value: Any, original_text: str = "") -> List[str]:
        if isinstance(value, dict):
            normalized = self.thickness_processor.process(value, original_text=original_text)
            return [p for p in re.split(r'[xX×*/]+', normalized.replace(' ', '')) if p]
        items = self._coerce_list(value)
        parts: List[str] = []
        for item in items:
            if isinstance(item, dict):
                item = item.get('value')
            if item in (None, ''):
                continue
            text = str(item).strip()
            if not text:
                continue
            normalized = self._normalize_single_thickness_or_group(text)
            if isinstance(normalized, list):
                parts.extend(normalized)
            elif normalized:
                parts.append(normalized)
        return parts

    @classmethod
    def _normalize_single_thickness_or_group(cls, text: str) -> List[str] | str:
        s = str(text).strip().upper()
        s = s.replace('”', '"').replace('“', '"').replace('″', '"')
        s = re.sub(r'\s+', '', s)
        s = s.replace('SCH.', 'SCH').replace('SCH ', 'SCH')

        # 结构化 list 是主路径。字符串仅做兜底：
        # 1) 显式分隔符（如 SCH40/SCH80, 4MM*3MM）优先按分隔符拆
        # 2) 紧凑编码（如 S30XS40, S30XXS）交给智能分词，避免把 XS/XXS 中的 X 当分隔符
        if cls.EXPLICIT_SEPARATOR_RE.search(s):
            raw_parts = [
                part for part in re.split(r'\s*[×*/]\s*|\s*(?<!X)X(?!S)\s*', s) if part
            ]
            normalized_parts = [
                cls._normalize_lookup_code(part) for part in raw_parts if cls._normalize_lookup_code(part)
            ]
            if len(normalized_parts) > 1:
                return normalized_parts

        split_parts = cls._split_thickness_parts(s)
        return [cls._normalize_lookup_code(part) for part in split_parts if cls._normalize_lookup_code(part)]

    @classmethod
    def _split_thickness_parts(cls, normalized: str) -> List[str]:
        if not normalized:
            return []

        s = normalized.strip().upper()
        parts: List[str] = []
        i = 0
        n = len(s)

        while i < n:
            m = cls.THICKNESS_TOKEN_RE.match(s, i)
            if not m:
                return [s]
            parts.append(m.group(1).upper())
            i = m.end()
            if i >= n:
                break
            if s[i] in ('X', '×', '*', '/'):
                i += 1
                continue
            return [s]

        return parts if len(parts) > 1 else [s]

    @classmethod
    def _normalize_lookup_code(cls, value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ''
        text = str(value).strip().upper()
        if not text:
            return ''

        text = text.replace('SCH.', 'SCH').replace('SCH ', 'SCH')
        text = text.replace('THK=', '').replace('T=', '')
        text = re.sub(r'\s+', '', text)

        if text in {'XS', 'XXS', 'STD'}:
            return text

        mm_match = re.fullmatch(r'(\d+(?:\.\d+)?)MM', text)
        if mm_match:
            num = cls._normalize_mm_cell(mm_match.group(1))
            return f'{num}MM'

        s_match = re.fullmatch(r'S(\d+(?:\.\d+)?)(S?)', text)
        if s_match:
            num = cls._normalize_mm_cell(s_match.group(1))
            tail = s_match.group(2)
            return f'SCH{num}{tail}'

        sch_match = re.fullmatch(r'SCH(\d+(?:\.\d+)?)(S?)', text)
        if sch_match:
            num = cls._normalize_mm_cell(sch_match.group(1))
            tail = sch_match.group(2)
            return f'SCH{num}{tail}'

        if text.startswith('T='):
            # 已在上面去掉，这里保留兼容
            text = text[2:]

        raw_mm = re.fullmatch(r'(\d+(?:\.\d+)?)', text)
        if raw_mm:
            return f'{cls._normalize_mm_cell(raw_mm.group(1))}MM'

        return text

    @staticmethod
    def _coerce_list(value: Any) -> List[Any]:
        if value in (None, ''):
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def is_schedule_like(value: Any) -> bool:
        text = str(value or "").strip().upper()
        return bool(text) and (text in {"XS", "XXS", "STD"} or text.startswith("SCH"))

    @staticmethod
    def mm_values_equivalent(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
        def _to_number(raw: Any) -> Optional[float]:
            text = str(raw or "").strip().upper()
            if text.endswith("MM"):
                text = text[:-2]
            if not text:
                return None
            try:
                return float(text)
            except (TypeError, ValueError):
                return None

        left_num = _to_number(left)
        right_num = _to_number(right)
        if left_num is None or right_num is None:
            return False
        return math.isclose(left_num, right_num, rel_tol=tolerance, abs_tol=tolerance)


@lru_cache(maxsize=1)
def get_thickness_table_processor() -> ThicknessTableProcessor:
    return ThicknessTableProcessor()


def convert_thickness_to_mm(standards: Any, dn_values: Any, thickness_values: Any, original_text: str = "") -> str:
    """便捷函数：按标准 + DN + 壁厚号查表换算为毫米壁厚。"""
    return get_thickness_table_processor().convert_to_mm(standards, dn_values, thickness_values, original_text=original_text)
