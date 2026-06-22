"""
规则提取器
通过正则表达式提取特定标签，不依赖NER标注
"""
import re
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """单个提取结果"""
    label: str          # 标签类型（ENDS, SEAL 等）
    value: str          # 原始提取值（用于显示，如"承插焊"）
    code: str           # 转换后的值（用于 seq2seq 编码，如"SW"）
    start: int          # 起始位置
    end: int            # 结束位置


class RegexExtractor:
    """
    规则提取器
    
    - encoder_config.yaml: 定义需要匹配的关键词列表
    - pipe_code_mapping.yaml: 定义编码映射
    
    提取规则：关键词两侧是空格、分隔符(;,-)或字符串边界，忽略大小写
    """
    
    def __init__(self, encoder_config_path: str = None):
        """
        初始化提取器
        
        Args:
            encoder_config_path: 编码配置文件路径（定义需要匹配的关键词）
        """
        base_dir = Path(__file__).parent.parent
        
        # 加载编码配置（获取需要匹配的关键词）
        if encoder_config_path is None:
            encoder_config_path = base_dir / "config" / "encoder_config.yaml"
        
        with open(encoder_config_path, 'r', encoding='utf-8') as f:
            encoder_config = yaml.safe_load(f)
        
        # regex_extraction: {ENDS: [SO, WN, ...], SEAL: [FF, RF, ...]}
        self.extraction_keywords = encoder_config.get('regex_extraction', {})
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        self.patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self.aliases: Dict[str, Dict[str, str]] = {}  # 别名转换映射
        self.resolution_rules: Dict[str, Dict[str, Any]] = {}

        def _has_cjk(text: str) -> bool:
            return bool(re.search(r'[\u4e00-\u9fff]', text))

        def _wrap_with_default_boundary(token_pattern: str, *, contains_cjk: bool) -> str:
            if contains_cjk:
                # 中文词默认只要求两侧不是中文，数字和符号允许贴边。
                return rf'(?<![\u4e00-\u9fff])({token_pattern})(?![\u4e00-\u9fff])'
            # 英文/缩写默认只要求两侧不是英文，数字和符号允许贴边。
            return rf'(?<![A-Za-z])({token_pattern})(?![A-Za-z])'

        def _wrap_with_strict_boundary(token_pattern: str) -> str:
            boundary = r'(?:^|[\s;,\-/\(\)\[\]:：|，、。（）+])'
            boundary_end = r'(?:$|[\s;,\-/\(\)\[\]:：|，、。（）+])'
            return rf'{boundary}({token_pattern}){boundary_end}'
        
        for label, config in self.extraction_keywords.items():
            if not config:
                continue
                
            self.patterns[label] = []
            
            # 支持两种配置格式：
            # 1. 旧格式：直接是关键词列表 [keyword1, keyword2, ...]
            # 2. 新格式：字典 {keywords: [...], patterns: [...], aliases: {...}}
            
            if isinstance(config, dict):
                def _append_keywords(keywords: List[str], *, strict: bool) -> None:
                    for keyword in keywords:
                        contains_cjk = _has_cjk(keyword)
                        wrapped = (
                            _wrap_with_strict_boundary(re.escape(keyword))
                            if strict
                            else _wrap_with_default_boundary(re.escape(keyword), contains_cjk=contains_cjk)
                        )
                        pattern = re.compile(wrapped, re.IGNORECASE)
                        self.patterns[label].append((keyword, pattern))

                # 新格式：处理强边界 / 松边界关键词。
                # 兼容旧配置：keywords 等价于 keywords_loose。
                _append_keywords(config.get('keywords_strict', []), strict=True)
                _append_keywords(config.get('keywords_loose', []), strict=False)
                _append_keywords(config.get('keywords', []), strict=False)
                
                # 新格式：处理 patterns（自定义正则模式，带边界）
                custom_patterns = config.get('patterns', [])
                for pattern_str in custom_patterns:
                    try:
                        contains_cjk = _has_cjk(pattern_str)
                        full_pattern = re.compile(
                            _wrap_with_default_boundary(pattern_str, contains_cjk=contains_cjk),
                            re.IGNORECASE
                        )
                        self.patterns[label].append((pattern_str, full_pattern))
                    except re.error as e:
                        print(f"警告: 无效的正则表达式 '{pattern_str}': {e}")
                
                # 新格式：处理 no_boundary（无边界关键词，用于中文如"承插焊"）
                no_boundary_keywords = config.get('no_boundary', [])
                for keyword in no_boundary_keywords:
                    pattern = re.compile(
                        f'({re.escape(keyword)})',
                        re.IGNORECASE
                    )
                    self.patterns[label].append((keyword, pattern))
                
                # 新格式：处理 aliases（别名转换，如 承插焊 -> SW）
                aliases = config.get('aliases', {})
                if aliases:
                    self.aliases[label] = {k.upper(): v.upper() for k, v in aliases.items()}
                resolution = config.get('resolution', {})
                if isinstance(resolution, dict) and resolution:
                    self.resolution_rules[label] = resolution
            else:
                # 旧格式：直接是关键词列表
                keywords = config
                for keyword in keywords:
                    contains_cjk = _has_cjk(keyword)
                    pattern = re.compile(
                        _wrap_with_default_boundary(re.escape(keyword), contains_cjk=contains_cjk),
                        re.IGNORECASE
                    )
                    self.patterns[label].append((keyword, pattern))

    @staticmethod
    def _result_identity(result: ExtractionResult) -> str:
        text = str(result.code or result.value or "").strip()
        return text.upper()

    def _apply_resolution_rules(self, results: List[ExtractionResult]) -> List[ExtractionResult]:
        if not results:
            return results

        grouped: Dict[str, List[ExtractionResult]] = defaultdict(list)
        label_order: List[str] = []
        for result in results:
            if result.label not in grouped:
                label_order.append(result.label)
            grouped[result.label].append(result)

        resolved: List[ExtractionResult] = []
        for label in label_order:
            label_results = grouped[label]
            resolution = self.resolution_rules.get(label) or {}
            if not resolution:
                resolved.extend(label_results)
                continue

            kept = list(label_results)
            suppress_rules = resolution.get("suppress_rules", [])
            for rule in suppress_rules:
                if not isinstance(rule, dict):
                    continue
                when_any = {
                    str(item).strip().upper()
                    for item in (rule.get("when_any") or [])
                    if str(item).strip()
                }
                when_all = {
                    str(item).strip().upper()
                    for item in (rule.get("when_all") or [])
                    if str(item).strip()
                }
                suppress = {
                    str(item).strip().upper()
                    for item in (rule.get("suppress") or [])
                    if str(item).strip()
                }
                if not suppress:
                    continue

                current_identities = {self._result_identity(item) for item in kept}
                any_hit = not when_any or bool(current_identities & when_any)
                all_hit = not when_all or when_all.issubset(current_identities)
                if not (any_hit and all_hit):
                    continue

                kept = [
                    item for item in kept
                    if self._result_identity(item) not in suppress
                ]

            resolved.extend(kept)
        return resolved

    def resolve_values(self, label: str, values: List[str]) -> List[str]:
        """对已有值列表应用同一套 resolution 规则，保持原顺序。"""
        normalized_values: List[str] = []
        for value in values:
            text = str(value or '').strip()
            if text and text not in normalized_values:
                normalized_values.append(text)
        if not normalized_values:
            return []

        pseudo_results = [
            ExtractionResult(
                label=label,
                value=text,
                code=text,
                start=index,
                end=index + len(text),
            )
            for index, text in enumerate(normalized_values)
        ]
        resolved = self._apply_resolution_rules(pseudo_results)
        return [str(item.code or item.value or '').strip() for item in resolved if str(item.code or item.value or '').strip()]
    
    def _normalize_standard_grade(self, value: str) -> str:
        """
        智能标准化 STANDARD_GRADE 值
        
        注意：这些值会拼接到 STANDARD 后面，然后由 StandardProcessor 进一步处理
        所以这里只做最基础的清理，保持原格式让 StandardProcessor 处理
        
        处理逻辑：
        1. 去除修饰词：I类填充金属 → I类
        2. 其他格式（TYPE I, Series I等）保持原样
        
        Args:
            value: 原始提取值
        
        Returns:
            标准化后的值
        """
        import re
        
        # 只处理中文"X类"格式：去除"填充金属/焊缝/填充材料"修饰词
        if '类' in value:
            # I类填充金属 → I类, 1类焊缝 → 1类
            value = re.sub(r'(类)(?:填充金属|焊缝|填充材料)$', r'\1', value)
        
        # 其他格式（TYPE I, Series I, type-1等）保持原样
        # StandardProcessor._convert_grade 会进一步处理它们
        return value

    @staticmethod
    def _normalize_radius_code(value: str) -> str:
        """
        将半径提取值统一规范到编码值。

        示例：
        - R=1.5D -> 1.5D
        - R 1.5D -> 1.5D
        - R1.5D -> 1.5D
        - 1.5D -> 1.5D
        - LR -> LR
        - SR -> SR
        """
        text = str(value or '').strip().upper()
        if not text:
            return ""

        compact = re.sub(r'\s+', '', text)
        if compact in {"LR", "SR"}:
            return compact

        m = re.match(r'^R=?(.+)$', compact)
        if m:
            compact = m.group(1).strip()

        return compact
    
    def extract(self, text: str, exclude_ranges: List[Tuple[int, int]] = None) -> List[ExtractionResult]:
        """
        从文本中提取所有匹配的标签
        
        Args:
            text: 输入文本
            exclude_ranges: 需要排除的范围列表，如 [(0, 20), (50, 60)]
                           用于排除 TYPE 实体所在范围，避免从 "WELDING NECK FLANGE" 中错误提取 WELDING
            
        Returns:
            提取结果列表
        """
        results = []
        exclude_ranges = exclude_ranges or []
        
        for label, pattern_list in self.patterns.items():
            for keyword, pattern in pattern_list:
                for match in pattern.finditer(text):
                    # match.group(1) 是捕获组中的关键词
                    matched_value = match.group(1)
                    
                    # 计算实际位置（考虑边界字符）
                    start = match.start(1)
                    end = match.end(1)
                    
                    # 检查是否在排除范围内
                    in_excluded = False
                    for ex_start, ex_end in exclude_ranges:
                        if start >= ex_start and end <= ex_end:
                            in_excluded = True
                            break
                    
                    if in_excluded:
                        continue
                    
                    # 特殊规则：排除"连接方式:"后面的"焊接"（代表对焊BW，不是工艺）
                    if label == 'MANU' and matched_value.lower() == '焊接':
                        # 检查前面是否有"连接方式"或"连接形式"
                        prefix = text[:start]
                        if re.search(r'连接方式[:：]?\s*$|连接形式[:：]?\s*$', prefix):
                            continue
                    
                    # 标准化处理
                    if label == 'STANDARD_GRADE':
                        # STANDARD_GRADE 保持原样，不做标准化
                        # 因为它会拼接到 STANDARD 后面，由 StandardProcessor 统一处理
                        normalized = matched_value
                    elif label == 'RADIUS':
                        normalized = self._normalize_radius_code(matched_value)
                    else:
                        # 其他标签：应用别名转换（如 承插焊 -> SW）
                        value_upper = matched_value.upper()
                        label_aliases = self.aliases.get(label, {})
                        normalized = label_aliases.get(value_upper, value_upper)
                    
                    results.append(ExtractionResult(
                        label=label,
                        value=matched_value,     # 原始值（保持原样，用于显示）
                        code=normalized,         # 标准化后的值（供编码使用）
                        start=start,
                        end=end
                    ))
        
        # 按位置排序
        results.sort(key=lambda x: x.start)
        
        # 去重：如果一个匹配完全包含在另一个匹配内，保留原始范围较长的
        filtered = []
        for r in results:
            is_substring = False
            for other in results:
                if r is not other:
                    # 如果 r 完全被 other 包含，且原始匹配范围更短
                    r_len = r.end - r.start
                    other_len = other.end - other.start
                    if other.start <= r.start and r.end <= other.end and r_len < other_len:
                        is_substring = True
                        break
            if not is_substring:
                filtered.append(r)

        return self._apply_resolution_rules(filtered)
    
    def extract_by_label(self, text: str, label: str, exclude_ranges: List[Tuple[int, int]] = None) -> List[ExtractionResult]:
        """
        提取指定标签的值
        
        Args:
            text: 输入文本
            label: 标签类型（ENDS, SEAL 等）
            exclude_ranges: 需要排除的范围列表
            
        Returns:
            该标签的提取结果列表
        """
        all_results = self.extract(text, exclude_ranges)
        return [r for r in all_results if r.label == label]
    
    def extract_as_dict(self, text: str, exclude_ranges: List[Tuple[int, int]] = None) -> Dict[str, List[Dict]]:
        """
        提取并按标签分组返回
        
        Args:
            text: 输入文本
            exclude_ranges: 需要排除的范围列表
            
        Returns:
            {
                'ENDS': [{'value': 'SO', 'code': 'SO', 'start': 8, 'end': 10}],
                'SEAL': [{'value': 'FF', 'code': 'FF', 'start': 15, 'end': 17}]
            }
        """
        results = self.extract(text, exclude_ranges)
        grouped: Dict[str, List[Dict]] = {}
        
        for r in results:
            if r.label not in grouped:
                grouped[r.label] = []
            grouped[r.label].append({
                'value': r.value,
                'code': r.code,
                'start': r.start,
                'end': r.end
            })
        
        return grouped
    
    def get_codes(self, text: str, exclude_ranges: List[Tuple[int, int]] = None) -> Dict[str, str]:
        """
        提取并返回每个标签的编码（多个值时取第一个）
        
        Args:
            text: 输入文本
            exclude_ranges: 需要排除的范围列表
            
        Returns:
            {'ENDS': 'SO', 'SEAL': 'FF'}
        """
        grouped = self.extract_as_dict(text, exclude_ranges)
        codes = {}
        
        for label, items in grouped.items():
            if items:
                codes[label] = items[0]['code']
        
        return codes
    
    def get_values_and_codes(self, text: str, exclude_ranges: List[Tuple[int, int]] = None) -> Dict[str, Dict[str, str]]:
        """
        提取并返回每个标签的原始值和编码（多个值时取第一个）
        
        Args:
            text: 输入文本
            exclude_ranges: 需要排除的范围列表（用于排除 TYPE 实体范围）
            
        Returns:
            {
                'CONN': {'value': 'NPT', 'code': 'N'},
                'MANU': {'value': 'EFW', 'code': 'W'}
            }
        """
        grouped = self.extract_as_dict(text, exclude_ranges)
        result = {}
        
        for label, items in grouped.items():
            if items:
                result[label] = {
                    'value': items[0]['value'],
                    'code': items[0]['code']
                }
        
        return result


# 单例
_extractor_instance: Optional[RegexExtractor] = None


def get_regex_extractor() -> RegexExtractor:
    """获取规则提取器单例"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = RegexExtractor()
    return _extractor_instance


# 测试代码
if __name__ == '__main__':
    extractor = RegexExtractor()
    
    test_cases = [
        "SO FLANGE / HG/T20592 / ASME B36.19M / PN10 / RF / A182 F304 / POLISHING DN50",
        "法兰;PN16;SO-RF;F304;NB/T47010;HG/T20592(B);DN80",
        "法兰;PN16;SO;F304;NB/T47010;HG/T20592(B);DN80",
        "8字盲板, S30408 GB/T4237, FF, CL 150, HG/T 21547 , DN50",
        "法兰;PN25;SO-RF;F304;fw;NB/T47010;HG/T20592(B),DN150",
    ]
    
    print("=" * 60)
    print("规则提取测试")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\n输入: {text}")
        results = extractor.extract(text)
        
        if results:
            for r in results:
                print(f"  [{r.label}] {r.value} → {r.code} (位置: {r.start}-{r.end})")
        else:
            print("  (无匹配)")
        
        codes = extractor.get_codes(text)
        print(f"  编码: {codes}")
