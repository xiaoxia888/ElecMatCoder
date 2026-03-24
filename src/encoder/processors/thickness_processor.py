"""
壁厚处理器
负责壁厚/壁厚规格的格式化和编码转换

处理规则：
1. SCH 格式：SCH80 → S80, SCH40S → S40S
2. S- 格式：S-80 → S80, S-10S → S10S  
3. 特殊值：SCHXXS → XXS, SCHSTD → STD
4. mm 格式：8.1mm → 8.1MM, T=3.0mm → 3MM
5. 异径处理：SCH80XSCH80 → S80 (相同合并), SCH80XSCH40 → S80XS40
6. 分隔符统一：x, *, ×, /, , → X
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional


class ThicknessProcessor:
    """壁厚处理器"""
    STRUCTURED_SUBTYPE_ORDER = ("MM", "INCH", "SCHEDULE", "SERIES", "BWG")
    
    # 分隔符正则（用于异径规格）
    SEPARATOR_PATTERN = re.compile(r'\s*[xX*×/,]\s*')
    
    # SCH 格式正则（注意：XXS 要放在 XS 前面，避免 XS 先匹配）
    # 支持 SCH40S, SCH.40S, SCH 40S 等格式
    SCH_PATTERN = re.compile(r'SCH[.\s]*(XXS|XS|STD|\d+S?)', re.IGNORECASE)
    
    # S- 格式正则
    S_DASH_PATTERN = re.compile(r'S-(XXS|XS|STD|\d+S?)', re.IGNORECASE)
    
    # mm 格式正则（数字+mm）
    MM_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE)
    
    # T=/THK= 前缀正则
    PREFIX_PATTERN = re.compile(r'^(?:T|THK)\s*=\s*', re.IGNORECASE)
    
    # (L)/(S)/L/S 后缀正则（大小端标识）
    SUFFIX_PATTERN = re.compile(r'\s*\([LS]\)|\s+[LS](?=\s|$|[xX*×/,])', re.IGNORECASE)
    
    # 特殊值（不带数字的）
    SPECIAL_VALUES = {'XXS', 'XS', 'STD'}

    # 来源于现有映射表的少量下划线脏样本别名。
    # 这些写法语义不一致，逐条归一比套一条通用规则更安全。
    UNDERSCORE_ALIASES = {
        'SCH20_S': 'SCH20S',
        'SCH40S_S': 'SCH40S/S',
        'SCH40_S': 'SCH40/S',
        'SCH40_STD': 'SCH40/STD',
        'STD_STD': 'STD/STD',
        'STD_SCH2': 'STD/SCH2',
        'SCH20_STD': 'STD/SCH20',
        'XS_XS': 'XS/XS',
    }

    def process(self, value: Any, original_text: str = "") -> str:
        """
        处理壁厚值，返回标准编码
        
        Args:
            value: 原始壁厚描述，如 "SCH40S", "S-10S x S-40S", "THK=6.3mmX6.3mm"
            
        Returns:
            标准编码，如 "S40S", "S10SXS40S", "6.3MM"
        """
        if not value:
            return ""

        if isinstance(value, dict):
            return self._process_structured(value, original_text=original_text)

        if isinstance(value, list):
            merged_parts: List[str] = []
            for item in value:
                normalized = self.process(item, original_text=original_text)
                if not normalized:
                    continue
                for part in normalized.split('X'):
                    part = part.strip()
                    if part and part not in merged_parts:
                        merged_parts.append(part)
            return 'X'.join(merged_parts)
        
        value = str(value).strip()
        
        # 1. 预处理
        value = self._preprocess(value)
        
        # 2. 分割异径规格
        parts = self._split_parts(value)
        
        # 3. 转换每个部分
        converted = [self._convert_single(p) for p in parts]
        converted = [c for c in converted if c]  # 过滤空值
        
        if not converted:
            return value.upper()  # 无法识别时返回大写原值
        
        # 4. 合并结果
        return self._merge_parts(converted)

    def _process_structured(self, value: Dict[str, Any], original_text: str = "") -> str:
        parts: List[str] = []

        ordered_keys = [k for k in self.STRUCTURED_SUBTYPE_ORDER if k in value]
        extra_keys = [k for k in value.keys() if k not in ordered_keys]

        for subtype in ordered_keys + extra_keys:
            raw_items = value.get(subtype)
            if raw_items in (None, "", []):
                continue
            if not isinstance(raw_items, list):
                raw_items = [raw_items]

            for item in raw_items:
                normalized = self._normalize_structured_part(str(subtype).upper(), item)
                if normalized and normalized not in parts:
                    parts.append(normalized)

        parts = self._reorder_parts_by_original_text(parts, original_text)
        return 'X'.join(parts)

    def _normalize_structured_part(self, subtype: str, item: Any) -> str:
        if isinstance(item, dict):
            item = item.get("value")
        if item in (None, ""):
            return ""

        text = str(item).strip()
        if not text:
            return ""

        if subtype == "MM":
            # 兼容 LLM 子类型误标：如 MM: SCH20 / MM: STD。
            # 这类值应走通用壁厚规则，不能强行转成 20MM。
            upper = text.upper()
            if (
                "SCH" in upper
                or upper in self.SPECIAL_VALUES
                or re.search(r'(^|[^A-Z])\d+(?:\.\d+)?S($|[^A-Z])', upper)
                or upper.startswith("S-")
                or re.match(r'^S\d', upper)
            ):
                return self.process(text)

            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if not match:
                return ""
            return f"{self._normalize_number(match.group(1))}MM"

        if subtype == "INCH":
            normalized = text.replace('”', '"').replace('“', '"').replace('″', '"')
            normalized = re.sub(r'\s+', '', normalized)
            return normalized.upper()

        if subtype in ("SCHEDULE", "SERIES"):
            return self.process(text)

        if subtype == "BWG":
            return text.upper()

        return self.process(text)

    def _reorder_parts_by_original_text(self, parts: List[str], original_text: str) -> List[str]:
        """
        按原始描述中的出现顺序重排壁厚片段。
        匹配失败的片段保持原顺序并放在后面。
        """
        if not original_text or len(parts) <= 1:
            return parts

        indexed = []
        for idx, part in enumerate(parts):
            pos = self._find_part_pos_in_text(part, original_text)
            indexed.append((idx, part, pos))

        indexed.sort(key=lambda x: (x[2] < 0, x[2] if x[2] >= 0 else 10**9, x[0]))
        return [item[1] for item in indexed]

    def _find_part_pos_in_text(self, part: str, original_text: str) -> int:
        if not part or not original_text:
            return -1

        text = original_text.upper()
        p = part.strip().upper()

        # 直接命中
        direct = text.find(p)
        if direct >= 0:
            return direct

        # 特殊值
        if p in self.SPECIAL_VALUES:
            for cand in (p, f"S{p}", f"S-{p}", f"SCH{p}", f"SCH {p}", f"SCH.{p}"):
                pos = text.find(cand)
                if pos >= 0:
                    return pos
            return -1

        # S80 / S80S / S6.3
        m_s = re.match(r'^S(\d+(?:\.\d+)?)(S?)$', p)
        if m_s:
            num = m_s.group(1)
            tail = m_s.group(2)
            candidates = [
                f"SCH{num}{tail}",
                f"SCH {num}{tail}",
                f"SCH.{num}{tail}",
                f"S-{num}{tail}",
                f"S{num}{tail}",
                f"{num}{tail}",
            ]
            for cand in candidates:
                pos = text.find(cand)
                if pos >= 0:
                    return pos
            return -1

        # 10MM / 6.3MM
        m_mm = re.match(r'^(\d+(?:\.\d+)?)MM$', p)
        if m_mm:
            num = m_mm.group(1)
            for cand in (f"{num}MM", f"{num} MM", num):
                pos = text.find(cand)
                if pos >= 0:
                    return pos
            return -1

        return -1
    
    def _preprocess(self, value: str) -> str:
        """
        预处理：去除前缀、后缀、无效字符等
        """
        result = value

        result = self._normalize_known_aliases(result)

        # 先去掉裸露的大小端标记（如 "SCH40 L x SCH80 S"）。
        # 这一类 L/S 只是位置标识，不是壁厚值本体；必须在删空格前处理，
        # 否则会被粘成 "SCH40LxSCH80S"，后续无法区分。
        result = self.SUFFIX_PATTERN.sub('', result)
        
        # 第一步：去除所有空格
        result = result.replace(' ', '')

        # 去掉 II-/Ⅱ- 前缀（只表示系列，不是壁厚本体）。
        result = re.sub(r'(?i)(?:^|(?<=[xX*×/,]))(?:II|Ⅱ)-', '', result)
        
        # 去除所有位置的 T=/THK= 前缀（开头和分隔符后的）
        # 匹配：T=, THK=, T =, THK =（在开头或分隔符后）
        result = re.sub(r'(?:^|(?<=[xX*×/,]))(?:T|THK)\s*=\s*', '', result, flags=re.IGNORECASE)
        
        # 容错处理：去除单独的 = 号（NER 可能漏掉 T，只识别出 =4.0mm）
        # 匹配开头或分隔符后的 = 号（后面跟数字）
        result = re.sub(r'(?:^|(?<=[xX*×/,]))=\s*(?=\d)', '', result, flags=re.IGNORECASE)
        
        # 处理 Thk 格式：thk3X6 → 3X6, thk3Xthk6 → 3X6
        result = re.sub(r'(?:^|(?<=[xX*×/,]))(?:THK|Thk|thk)', '', result, flags=re.IGNORECASE)
        
        # 去除 (L)/(S) 后缀（此时空格已去除，只匹配括号形式）
        result = re.sub(r'\([LS]\)', '', result, flags=re.IGNORECASE)
        
        # 去除无效字符（如单引号、反引号等）
        result = re.sub(r"[''`]", '', result)
        
        return result

    def _normalize_known_aliases(self, value: str) -> str:
        """处理映射表中已知的少量脏别名。"""
        compact = re.sub(r'\s+', '', value or '').upper()
        return self.UNDERSCORE_ALIASES.get(compact, value)
    
    def _split_parts(self, value: str) -> List[str]:
        """
        按分隔符分割异径规格
        
        注意：需要智能分割，避免把 XXS 中的 X 当作分隔符
        分隔符特征：X 两边都有数字或壁厚值
        """
        # 特殊值直接返回，不分割
        if value.upper() in self.SPECIAL_VALUES:
            return [value]
        
        # 先检查是否是 SCH 开头的特殊值（如 SCHXXS, SCHXS）
        sch_special = re.match(r'^SCH\s*(XXS|XS|STD)$', value, re.IGNORECASE)
        if sch_special:
            return [value]
        
        # 智能分割：只在有效分隔位置分割
        # 有效分隔：分隔符前后都是完整的壁厚值
        # 模式：值 + 分隔符 + 值
        # 值的模式：SCH\d+S?|S-?\d+S?|S?-?XXS|S?-?XS|STD|\d+(\.\d+)?\s*mm|\d+(\.\d+)?
        
        # 使用更精确的分割模式
        # 匹配异径分隔符（前后有数字或字母）
        split_pattern = re.compile(
            r'(?<=[0-9SsDdMmXx])\s*[xX*×/,]\s*(?=[0-9SsTsHhCcXx])',
            re.IGNORECASE
        )
        
        parts = split_pattern.split(value)
        # 过滤空字符串
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 1:
            concatenated_parts = self._split_concatenated_tokens(value)
            if len(concatenated_parts) > 1:
                parts = concatenated_parts
        
        # 处理省略前缀的情况：
        # SCH10/10 → SCH10/SCH10, S10/10 → S10/S10, S-10/10 → S-10/S-10
        # 如果第一个部分有壁厚前缀，后续纯数字部分应继承该前缀
        if len(parts) > 1:
            first_part = parts[0]
            
            # 匹配各种壁厚前缀格式
            # SCH10, SCH10S, S10, S10S, S-10, S-10S
            prefix_match = re.match(r'^(SCH|S-?)\s*(\d+S?)$', first_part, re.IGNORECASE)
            if prefix_match:
                prefix = prefix_match.group(1)  # SCH, S, S-
                for i in range(1, len(parts)):
                    # 如果是纯数字（可能带S后缀），补上前缀
                    if re.match(r'^(\d+S?)$', parts[i], re.IGNORECASE):
                        parts[i] = f"{prefix}{parts[i]}"
        
        return parts

    def _split_concatenated_tokens(self, value: str) -> List[str]:
        """
        尝试拆分无显式分隔符的连续壁厚 token。

        例如：
        - SCH20SCH40 -> ["SCH20", "SCH40"]
        - SCH20SCH40S -> ["SCH20", "SCH40S"]
        - S40S20 -> ["S40", "S20"]
        - 40S20S -> ["40S", "20S"]
        - S40SS20S -> ["S40S", "S20S"]

        仅当整串可以被完整切分为 2 段及以上合法 token 时才生效，
        避免把普通单值误拆。
        """
        if not value:
            return [value]

        upper = value.upper()
        if len(upper) < 4:
            return [value]

        @lru_cache(maxsize=None)
        def dfs(pos: int):
            if pos == len(upper):
                return []

            for candidate in self._iter_concatenated_candidates(upper, pos):
                rest = dfs(pos + len(candidate))
                if rest is not None:
                    return [candidate] + rest
            return None

        parts = dfs(0)
        return parts if parts and len(parts) > 1 else [value]

    def _iter_concatenated_candidates(self, value: str, pos: int) -> List[str]:
        """返回当前位置可能的连续壁厚 token，按更长优先。"""
        candidates: List[str] = []
        rest = value[pos:]

        if rest.startswith('SCH'):
            body = rest[3:]
            for special in ('XXS', 'XS', 'STD'):
                token = f'SCH{special}'
                if rest.startswith(token):
                    candidates.append(token)
            num_match = re.match(r'SCH(\d+(?:\.\d+)?)', rest)
            if num_match:
                base = num_match.group(0)
                if rest.startswith(base + 'S'):
                    candidates.append(base + 'S')
                candidates.append(base)

        if rest.startswith('S-'):
            for special in ('XXS', 'XS', 'STD'):
                token = f'S-{special}'
                if rest.startswith(token):
                    candidates.append(token)
            num_match = re.match(r'S-(\d+(?:\.\d+)?)', rest)
            if num_match:
                base = num_match.group(0)
                if rest.startswith(base + 'S'):
                    candidates.append(base + 'S')
                candidates.append(base)

        if rest.startswith('S'):
            num_match = re.match(r'S(\d+(?:\.\d+)?)', rest)
            if num_match:
                base = num_match.group(0)
                if rest.startswith(base + 'S'):
                    candidates.append(base + 'S')
                candidates.append(base)

        num_s_match = re.match(r'(\d+(?:\.\d+)?)S', rest)
        if num_s_match:
            candidates.append(num_s_match.group(0))

        mm_match = re.match(r'(\d+(?:\.\d+)?)MM', rest)
        if mm_match:
            candidates.append(mm_match.group(0))

        for special in ('XXS', 'XS', 'STD'):
            if rest.startswith(special):
                candidates.append(special)

        # 去重并按长度倒序，优先尝试更完整 token，再回溯处理歧义情况。
        unique = []
        seen = set()
        for token in sorted(candidates, key=len, reverse=True):
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique
    
    def _convert_single(self, value: str) -> str:
        """
        转换单个壁厚值
        """
        if not value:
            return ""
        
        value = value.strip()
        upper_value = value.upper()
        
        # 检查是否是特殊值（XXS, XS, STD）- 精确匹配
        if upper_value in self.SPECIAL_VALUES:
            return upper_value
        
        # 尝试 SCH 格式：SCH80 → S80, SCH40S → S40S, SCHXXS → XXS
        sch_match = self.SCH_PATTERN.match(value)
        if sch_match:
            suffix = sch_match.group(1).upper()
            # 特殊值不加 S 前缀
            if suffix in self.SPECIAL_VALUES:
                return suffix
            return f"S{suffix}"
        
        # 尝试 S- 格式：S-80 → S80, S-10S → S10S, S-XS → XS
        s_dash_match = self.S_DASH_PATTERN.match(value)
        if s_dash_match:
            suffix = s_dash_match.group(1).upper()
            if suffix in self.SPECIAL_VALUES:
                return suffix
            return f"S{suffix}"
        
        # 尝试 S 格式（已经是 S 开头）：S80 → S80, S10S → S10S, S6.3 → S6.3
        s_match = re.match(r'^S(\d+(?:\.\d+)?S?)', value, re.IGNORECASE)
        if s_match:
            suffix = s_match.group(1).upper()
            # 如果是纯数字（可能有小数），保留
            return f"S{suffix}"
        
        # 尝试纯数字+S 格式：40S → S40S, 10S → S10S, 80S → S80S
        num_s_match = re.match(r'^(\d+)(S)$', value, re.IGNORECASE)
        if num_s_match:
            num = num_s_match.group(1)
            return f"S{num}S"
        
        # 尝试 mm 格式：8.1mm → 8.1MM, 22.0mm → 22MM
        mm_match = self.MM_PATTERN.match(value)
        if mm_match:
            num = mm_match.group(1)
            # 去除小数点后的 .0
            num = self._normalize_number(num)
            return f"{num}MM"
        
        # 尝试 Thk 格式：Thk4.0 → 4MM, THK=6.3 → 6.3MM
        thk_match = re.match(r'^THK\s*[=]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?$', value, re.IGNORECASE)
        if thk_match:
            num = self._normalize_number(thk_match.group(1))
            return f"{num}MM"
        
        # 尝试纯数字格式：6.3 → 6.3MM
        num_match = re.match(r'^(\d+(?:\.\d+)?)$', value)
        if num_match:
            num = self._normalize_number(num_match.group(1))
            return f"{num}MM"
        
        # 无法识别，返回大写原值
        return upper_value
    
    def _normalize_number(self, num_str: str) -> str:
        """
        规范化数字：去除尾部的 .0
        """
        try:
            num = float(num_str)
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return num_str
    
    def _merge_parts(self, parts: List[str]) -> str:
        """
        合并异径规格部分
        
        规则：
        - 如果两部分相同，只输出一个
        - 如果不同，用 X 连接
        """
        if not parts:
            return ""
        
        if len(parts) == 1:
            return parts[0]
        
        # 去重（相邻相同的合并）
        unique_parts = [parts[0]]
        for p in parts[1:]:
            if p != unique_parts[-1]:
                unique_parts.append(p)
        
        if len(unique_parts) == 1:
            return unique_parts[0]
        
        return 'X'.join(unique_parts)


# 单例
_processor_instance: Optional[ThicknessProcessor] = None


def get_thickness_processor() -> ThicknessProcessor:
    """获取壁厚处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ThicknessProcessor()
    return _processor_instance


# 测试代码
if __name__ == '__main__':
    processor = ThicknessProcessor()
    
    # 测试用例（输入, 期望输出）
    test_cases = [
        # 基础格式
        ("S-80", "S80"),
        ("S-10S", "S10S"),
        ("SCH80", "S80"),
        ("Sch40", "S40"),
        ("SCH40S", "S40S"),
        ("SCH 40S", "S40S"),
        ("SCH5S(L)", "S5S"),
        ("SCH10(L)", "S10"),
        ("SCHXXS", "XXS"),
        ("XXS", "XXS"),
        ("SCHXS", "XS"),
        ("XS", "XS"),
        ("SCHSTD", "STD"),
        ("STD", "STD"),
        
        # 异径格式
        ("Sch40XSch80", "S40XS80"),
        ("SCH80XSCH80", "S80"),
        ("SCH80XSCH80S", "S80XS80S"),
        ("STDXSTD", "STD"),
        ("STDXSCH20", "STDXS20"),
        ("S-10S x S-40S", "S10SXS40S"),
        ("S-60 x S-XS", "S60XXS"),
        ("SCH60(L)*SCH80(S)", "S60XS80"),
        ("SCH40 L x SCH80 S", "S40XS80"),
        ("SCH10S(L)×SCH40S(S)", "S10SXS40S"),
        ("STD(L)*SCH30(S)", "STDXS30"),
        ("SCH20(L)'×SCH40(S)", "S20XS40"),
        ("Sch 20 Sch 40", "S20XS40"),
        ("SCH20SCH40", "S20XS40"),
        ("SCH20SCH40S", "S20XS40S"),
        ("S40S20", "S40XS20"),
        ("40s20S", "S40SXS20S"),
        ("S40SS20S", "S40SXS20S"),
        ("S40S40S", "S40S"),
        ("S-40S, S-80S", "S40SXS80S"),
        ("Sch20_S", "S20S"),
        ("Sch40S_S", "S40SXS"),
        ("Sch40_S", "S40XS"),
        ("Sch40_STD", "S40XSTD"),
        ("STD_STD", "STD"),
        ("STD_Sch2", "STDXS2"),
        ("Sch20_STD", "STDXS20"),
        ("XS_XS", "XS"),
        
        # mm 格式
        ("8.1mm", "8.1MM"),
        ("4mm/16.6mm", "4MMX16.6MM"),
        ("3mm/2.4mm", "3MMX2.4MM"),
        ("8.1mmX5.6mm", "8.1MMX5.6MM"),
        ("22.0mmx20", "22MMX20MM"),
        ("4.5mmX4.5mm", "4.5MM"),
        
        # T=/THK= 格式
        ("T=3.0mm", "3MM"),
        ("T=3.5x3mm", "3.5MMX3MM"),
        ("THK=5.6mmX4.0mm", "5.6MMX4MM"),
        ("THK=6.3mmX6.3mm", "6.3MM"),
        ("THK=6.0X4.5mm", "6MMX4.5MM"),
        ("THK=3.0X3.0mm", "3MM"),
        ("II-3/2.5", "3MMX2.5MM"),
        ("Ⅱ-4×3", "4MMX3MM"),
        ("II-2.5/2.5", "2.5MM"),
        ("Ⅱ-3", "3MM"),
        
        # 混合格式
        ("S6.3XS6.3", "S6.3"),
        ("11.91 mm x S-40", "11.91MMXS40"),
    ]
    
    print("=== 壁厚处理器测试 ===")
    passed = 0
    failed = 0
    
    for input_val, expected in test_cases:
        result = processor.process(input_val)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{input_val}' → '{result}' (期望: '{expected}')")
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
