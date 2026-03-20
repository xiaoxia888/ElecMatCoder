"""
标准/规范处理器
负责规范的格式化、分类和排序
"""
import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class StandardProcessor:
    MODIFIER_ORDER = ["STANDARD_GRADE", "STANDARD_APPENDIX", "STANDARD_METHOD"]

    def __init__(self, config_path: str = None):
        # 配置文件路径（processors 文件夹的上级目录下的 config）
        base_dir = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = base_dir / "config" / "standard_classification.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载编码配置中的 remove_chars
        encoder_config_path = base_dir / "config" / "encoder_config.yaml"
        if encoder_config_path.exists():
            with open(encoder_config_path, 'r', encoding='utf-8') as f:
                encoder_config = yaml.safe_load(f)
                self.remove_chars = encoder_config.get('standard_processing', {}).get('remove_chars', [])
        else:
            self.remove_chars = ['-', '/', ' ']
        
        self.prefix_priority = self.config.get('prefix_priority', {})
        self.code_conversion = self.config.get('code_conversion', {})
        self.standards = self.config.get('standards', {})
    
    def process_standards(self, standards: List[str]) -> str:
        """
        处理标准数组，返回编码后的字符串
        
        Args:
            standards: 原始标准列表，如 ['GB/T12459-17 SERIES I', 'ASME B16.9']
        
        Returns:
            编码后的标准字符串，如 'GBT12459IASME B16.9'
        """
        if not standards:
            return ""
        
        # 0. 先展开斜杠分隔的复合规范（如 ASME B36.19/B36.10 → ASME B36.19, ASME B36.10）
        expanded_standards = self._expand_slash_standards(standards)
        
        # 1. 格式化每个标准
        formatted = [self._format_standard(s) for s in expanded_standards]
        
        # 2. 分类并编码
        production = []
        manufacturing = []
        unknown = []
        for std in formatted:
            encoded = self._encode_standard(std)
            category = self._classify_standard(std)
            if category == 'production':
                production.append(encoded)
            elif category == 'manufacturing':
                manufacturing.append(encoded)
            else:
                unknown.append(encoded)
        
        # 3. 分别排序（在编码后排序，格式统一）
        production = self._sort_encoded(production)
        manufacturing = self._sort_encoded(manufacturing)
        unknown = self._sort_encoded(unknown)
        
        # 4. 顺序：生产标准 > 制造标准 > 未知
        all_encoded = production + manufacturing + unknown
        
        # 5. 拼接返回
        return ''.join(all_encoded)
    
    def _expand_slash_standards(self, standards: List[str]) -> List[str]:
        """
        保留原始斜杠写法，不再自动拆分复合规范。

        之前这里会把类似 `SH/20592`、`ASME B36.19/B36.10` 的写法拆成多条规范，
        但这会导致部分业务场景下误拆，和平台期望不一致。
        现在统一按原文整体保留，仅做首尾空白清理。
        """
        return [std.strip() for std in standards if str(std or "").strip()]

    @staticmethod
    def _remove_first_occurrence(text: str, fragment: str) -> str:
        if not text or not fragment:
            return text
        idx = text.lower().find(fragment.lower())
        if idx < 0:
            return text
        merged = f"{text[:idx]} {text[idx + len(fragment):]}"
        merged = re.sub(r'\s+', ' ', merged).strip()
        return merged.strip(' ,;|')

    @staticmethod
    def _extract_by_patterns(text: str, patterns: List[str]) -> Tuple[str, str]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fragment = re.sub(r'\s+', ' ', match.group(0)).strip()
                remaining = StandardProcessor._remove_first_occurrence(text, fragment)
                return fragment, remaining
        return '', text

    def parse_standard_structure(self, standard: str) -> Dict[str, str]:
        text = re.sub(r'\s+', ' ', str(standard or '')).strip()
        if not text:
            return {
                'subject': '',
                'grade': '',
                'method': '',
                'appendix': ''
            }

        appendix, remaining = self._extract_by_patterns(text, [
            r'(?:Appendix|APPENDIX)\s*[-]?\s*[A-Za-z0-9]+',
            r'附录\s*[A-Za-z0-9]+',
        ])
        method, remaining = self._extract_by_patterns(remaining, [
            r'(?:Method|METHOD|Design|DESIGN)\s*[-]?\s*[A-Za-z0-9]+',
            r'(?:方法|设计)\s*[A-Za-z0-9]+',
        ])
        grade, remaining = self._extract_by_patterns(remaining, [
            r'(?:Series|SERIES|Serial|SERIAL|Type|TYPE|Class|CLASS)\s*[-]?\s*(?:[IVXivx]+[a-z]?|\d+[a-z]?|[A-Za-z0-9]+)',
            r'CL\s*[-]?\s*[A-Za-z0-9]+',
            r'[IVXivx\d]+类',
            r'[IVXivx\d]+系列',
        ])

        subject = re.sub(r'\s+', ' ', remaining).strip(' ,;|')
        return {
            'subject': subject,
            'grade': grade,
            'method': method,
            'appendix': appendix,
        }
    
    def _format_standard(self, standard: str) -> str:
        """
        格式化单个标准（保留原始大小写）
        1. 去除年份
        2. 去除 ASME 标准的 M 后缀（米制单位标识）
        3. 处理规范等级（使用正则提取，保留原始大小写）
        4. 规范化空格
        5. 补全无前缀标准的前缀（如 20592Ⅰ → HG20592Ⅰ）
        
        注意：不进行全局大写转换，保留 Ia, II 等等级的原始大小写
        """
        result = standard.strip()
        
        # 1. 去除年份 (支持多种格式)
        # 特征1：4位数年份 (19xx 或 20xx)
        # 前置字符可以是数字、右括号、右方括号等（如 HG/T20553(la)-2011）
        result = re.sub(r'([\d\)\]a-zA-Z])\s*-\s*(19|20)\d{2}(?=\s|$|\(|,|;|[A-Za-z])', r'\1', result)
        # 特征2：2位数年份（前面必须是数字/右括号/右方括号，不含字母，避免误删 SP-97 等）
        result = re.sub(r'([\d\)\]])\s*-\s*(0[0-9]|1[0-9]|2[0-9]|9[0-9])(?=\s|,|;|$|\()', r'\1', result)
        
        # 2. 去除 ASME 标准的 M 后缀（米制单位标识，如 B36.19M → B36.19）
        # 只对 ASME 标准生效：数字后面紧跟的 M 且是结尾
        # 注意：ASME 后面可能直接跟 B（如 ASMEB36.19M），不能用 \b 边界
        if re.match(r'^ASME', result, re.IGNORECASE):
            result = re.sub(r'(\d)M(?=\s|$)', r'\1', result)
        
        # 3. 处理规范等级转换（使用正则方法，保留原始大小写）
        result = self._convert_grade(result)
        
        # 4. 规范化多余空格
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 5. 补全无前缀标准的前缀（如 20592Ⅰ → HG20592Ⅰ）
        result = self._add_missing_prefix(result)
        
        return result
    
    def _add_missing_prefix(self, standard: str) -> str:
        """
        为缺少前缀的标准号补全前缀
        
        三层匹配策略（数字编号先匹配，再验证前缀）：
        1. 精确后缀匹配：输入是配置键的后缀（如 SP97 → MSSSP97）
        2. 部分前缀匹配：输入带了不完整的前缀（如 SH3419 → SHT3419）
        3. 纯数字匹配：输入只有数字（如 3419 → SHT3419）
        
        例如：
        - 20592Ⅰ → 匹配 HGT20592 → HG/T 20592Ⅰ
        - SP-97  → 匹配 MSSSP97  → MSS SP-97
        - B16.9  → 匹配 ASMEB169 → ASME B16.9
        - SH 3419 → 匹配 SHT3419 → SH/T 3419
        - HG 20553 → 匹配 HGT20553 → HG/T 20553
        """
        std = standard.strip()
        
        normalized = re.sub(r'[^A-Z0-9]', '', std.upper())
        
        if not normalized:
            return standard
        
        input_numbers = re.findall(r'\d+', normalized)
        if not input_numbers:
            return standard
        
        main_number = input_numbers[0]
        
        input_alpha = re.match(r'^([A-Z]*)', normalized).group(1)
        
        best_match = None
        
        for std_key in self.standards.keys():
            config_normalized = re.sub(r'[^A-Z0-9]', '', std_key.upper())
            config_numbers = re.findall(r'\d+', config_normalized)
            
            if not config_numbers or config_numbers[0] != main_number:
                continue
            
            config_alpha = re.match(r'^([A-Z]*)', config_normalized).group(1)
            
            # 策略1：精确后缀匹配（如 B169 是 ASMEB169 的后缀）
            if config_normalized.endswith(normalized):
                missing_prefix = config_normalized[:-len(normalized)]
                if missing_prefix:
                    readable_prefix = self._format_prefix(missing_prefix)
                    return readable_prefix + std
                return standard
            
            # 策略2：部分前缀匹配
            # 输入有字母前缀，且它是配置前缀的前缀子串
            # 如 SH 是 SHT 的前缀，HG 是 HGT 的前缀
            if input_alpha and config_alpha.startswith(input_alpha) and input_alpha != config_alpha:
                best_match = (std_key, config_alpha, config_normalized)
        
        if best_match:
            std_key, config_alpha, config_normalized = best_match
            full_prefix = self._format_prefix(config_alpha)
            num_part = re.sub(r'^[A-Za-z/\s]+', '', std).strip()
            return full_prefix + num_part
        
        return standard
    
    def _format_prefix(self, prefix: str) -> str:
        """
        将规范化的前缀转换为可读格式
        
        如：HGT → HG/T , MSS → MSS , ASME → ASME 
        """
        # 常见前缀的可读格式映射
        prefix_formats = {
            'GBT': 'GB/T ',
            'HGT': 'HG/T ',
            'SHT': 'SH/T ',
            'NBT': 'NB/T ',
            'SYT': 'SY/T ',
            'JBT': 'JB/T ',
            'MSS': 'MSS ',
            'ASME': 'ASME ',
            'API': 'API ',
            'EN': 'EN ',
            'ISO': 'ISO ',
            'BS': 'BS ',
        }
        
        upper_prefix = prefix.upper()
        if upper_prefix in prefix_formats:
            return prefix_formats[upper_prefix]
        
        # 默认：前缀 + 空格
        return prefix + ' '
    
    def _convert_grade(self, standard: str) -> str:
        """
        使用正则方法转换规范修饰信息（保留原始大小写）
        
        规则：
        - SERIES I / Series I / SERIESI → I (保留 I 的原始大小写)
        - TYPE V / Type V / TYPEV → V (保留 V 的原始大小写)
        - I系列 / II系列 / Ⅰ系列 / Ⅱ系列 → I / II / Ⅰ / Ⅱ (中文系列格式)
        - 附录B / Appendix B → B
        - (Ia) / (II) / (A) → Ia / II / A (去除括号，保留内容)
        """
        result = standard
        
        # 1. SERIES/Serial xxx → xxx (忽略大小写匹配，保留后面的等级)
        # 匹配 "SERIES I", "Series II", "Serial B", "SERIESI" 等
        result = re.sub(
            r'\b(?:SERIES|Serial)\s*([IVXABab]+)\b',
            r'\1',
            result,
            flags=re.IGNORECASE
        )
        
        # 2. TYPE xxx → xxx (忽略大小写匹配 TYPE，保留后面的等级)
        # 匹配 "TYPE I", "Type V", "TYPEI" 等
        result = re.sub(
            r'\bTYPE\s*([IVXABab]+)\b',
            r'\1',
            result,
            flags=re.IGNORECASE
        )
        
        # 3. xxx系列 → xxx (中文系列格式)
        # 匹配 "I系列", "II系列", "Ⅰ系列", "Ⅱ系列" 等
        result = re.sub(
            r'\b([IVXABab0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)系列\b',
            r'\1',
            result
        )
        
        # 4. 附录X / Appendix X → X
        result = re.sub(
            r'附录\s*([A-Za-z0-9]+)',
            r'\1',
            result
        )
        result = re.sub(
            r'\bAppendix\s*([A-Za-z0-9]+)\b',
            r'\1',
            result,
            flags=re.IGNORECASE
        )

        # 5. 方法X → X (中文方法格式)
        # 匹配 "方法A", "方法B", "方法C", "方法E" 等
        result = re.sub(
            r'方法\s*([A-Za-z0-9]+)',
            r'\1',
            result
        )
        
        # 6. Method X → X (英文方法格式)
        # 匹配 "Method A", "Method B", "MethodC" 等
        result = re.sub(
            r'\bMethod\s*([A-Za-z0-9]+)\b',
            r'\1',
            result,
            flags=re.IGNORECASE
        )
        
        # 6.5. Design X → X (设计类型格式)
        # 匹配 "DESIGN A", "Design B", "DesignA" 等
        result = re.sub(
            r'\bDesign\s*([A-Za-z0-9]+)\b',
            r'\1',
            result,
            flags=re.IGNORECASE
        )
        
        # 7. X类填充金属/焊缝/填充材料 → X（去除修饰词，保留等级）
        # 匹配 "I类填充金属", "II类焊缝", "1类填充材料" 等
        # 保留等级标记（I、II、1 等），去除"类"和修饰词
        result = re.sub(
            r'\s*([IVXivx0-9]+)类(?:填充金属|焊缝|填充材料)',
            r' \1',
            result
        )
        
        # 8. X类 → X（去除"类"字，保留等级标记）
        # 匹配 "I类", "II类", "1类" 等
        result = re.sub(
            r'\s*([IVXivx0-9]+)类',
            r' \1',
            result
        )
        
        # 9. (xxx) → xxx (去除括号，保留内容，用于等级如 (Ia), (II), (A), (B))
        # 只处理末尾的括号内容（通常是等级）
        result = re.sub(
            r'\(([IVXABab0-9]+)\)\s*$',
            r'\1',
            result
        )
        
        # 10. 删除单独的关键词（没有后缀）
        # 匹配单独的 SERIES、TYPE、Serial、Method（没有紧跟等级标记）
        # 这些是错误识别或无效的等级标记，应该删除
        result = re.sub(
            r'\b(SERIES|Serial|TYPE|Type|METHOD|Method|APPENDIX|Appendix)\b(?!\s*[IVXABab0-9])',
            '',
            result,
            flags=re.IGNORECASE
        )
        
        # 规范化空格（可能因删除关键词产生多余空格）
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _classify_standard(self, standard: str) -> str:
        """
        分类标准为生产标准或制造标准
        通过模糊匹配，忽略前缀，要求输入包含配置中的核心标准号
        
        例如：
        - DIN EN 10253-4 → 包含 EN10253，匹配
        - MSS SP-97 → 包含 MSSSP97，匹配
        """
        # 从 code_conversion 配置中提取可忽略前缀
        # 如 "DIN EN": "EN" → DIN 是可忽略前缀
        ignorable_prefixes = set()
        for old_prefix, new_prefix in self.code_conversion.items():
            # 如果旧前缀包含新前缀，则旧前缀中除了新前缀以外的部分是可忽略的
            # 如 "DIN EN" → "EN"，则 "DIN" 是可忽略的
            old_normalized = re.sub(r'[^A-Z0-9]', '', old_prefix.upper())
            new_normalized = re.sub(r'[^A-Z0-9]', '', new_prefix.upper())
            if old_normalized.endswith(new_normalized) and old_normalized != new_normalized:
                ignorable = old_normalized[:-len(new_normalized)]
                if ignorable:
                    ignorable_prefixes.add(ignorable)
        
        # 已知的标准体系前缀（用于从配置键中剥离）
        system_prefixes = ['ASME', 'MSS', 'GBT', 'HGT', 'SHT', 'NBT', 'SYT', 'JBT', 'API', 'EN', 'ISO', 'BS']
        
        # 提取标准的标准化key (去除空格、斜杠等，只保留字母和数字)
        normalized = re.sub(r'[^A-Z0-9]', '', standard.upper())
        
        # 预处理：去除可忽略前缀
        input_normalized = normalized
        for prefix in ignorable_prefixes:
            if input_normalized.startswith(prefix):
                input_normalized = input_normalized[len(prefix):]
                break
        
        # 尝试匹配配置中的标准
        for std_key, category in self.standards.items():
            # 标准化配置中的key
            config_normalized = re.sub(r'[^A-Z0-9]', '', std_key.upper())
            
            # 1. 精确匹配
            if input_normalized == config_normalized:
                return category
            
            # 2. 输入包含配置的标准号（如 EN102534 包含 EN10253，MSSSP97 包含 MSSSP97）
            if config_normalized in input_normalized:
                return category
            
            # 3. 忽略体系前缀匹配：SP97 匹配 MSSSP97（去掉 MSS 后变成 SP97）
            for prefix in system_prefixes:
                if config_normalized.startswith(prefix):
                    # 去掉前缀后的核心部分
                    core = config_normalized[len(prefix):]
                    if core and core in input_normalized:
                        return category
        
        # 默认为未知（排在生产和制造之后）
        return 'unknown'
    
    def _sort_standards(self, standards: List[str]) -> List[str]:
        """
        对同类标准进行排序（格式化后的标准）
        1. 按前缀优先级
        2. 同前缀按字典序（逐字符比较）
        """
        def sort_key(std: str) -> Tuple[int, str]:
            # 提取前缀
            prefix_match = re.match(r'^([A-Z/]+)', std.upper())
            prefix = prefix_match.group(1) if prefix_match else 'ZZZ'
            
            # 标准化前缀 (去除斜杠)
            normalized_prefix = prefix.replace('/', '')
            
            # 获取优先级
            priority = 999
            for p, pri in self.prefix_priority.items():
                if normalized_prefix.startswith(p):
                    priority = pri
                    break
            
            # 同前缀按字典序排序（逐字符比较：'1' < '4'）
            return (priority, std.upper())
        
        return sorted(standards, key=sort_key)
    
    def _sort_encoded(self, encoded_list: List[str]) -> List[str]:
        """
        对编码后的标准进行排序（编码格式统一，直接字典序）
        1. 按前缀优先级
        2. 同前缀按字典序（逐字符比较）
        """
        def sort_key(encoded: str) -> Tuple[int, str]:
            # 提取前缀（编码后的前缀，如 GBT, HGT, AB）
            prefix_match = re.match(r'^([A-Z]+)', encoded.upper())
            prefix = prefix_match.group(1) if prefix_match else 'ZZZ'
            
            # 获取优先级
            priority = 999
            for p, pri in self.prefix_priority.items():
                if prefix.startswith(p):
                    priority = pri
                    break
            
            # 同前缀按字典序排序
            return (priority, encoded.upper())
        
        return sorted(encoded_list, key=sort_key)
    
    def _encode_standard(self, standard: str) -> str:
        """
        将标准转换为编码格式
        1. 前缀部分转大写（GB/T, HG/T, ASME 等）
        2. 应用代码转换规则（如 ASMEB → AB）
        3. 去除配置的特殊字符
        
        注意：保留等级部分的原始大小写（如 Ia, II）
        """
        result = standard
        
        # 1. 分离前缀和其余部分，只对前缀转大写
        # 匹配前缀: 字母 + 可选的 /T 或 / + 字母
        prefix_match = re.match(r'^([A-Za-z]+(?:/[A-Za-z]*)?)', result)
        if prefix_match:
            prefix = prefix_match.group(1).upper()
            rest = result[len(prefix_match.group(1)):]
            result = prefix + rest
        
        # 2. 先去除空格（为了正确匹配转换规则）
        result = result.replace(' ', '')
        
        # 3. 罗马数字转换（全角→普通字母）
        roman_map = {
            'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
            'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
            'ⅰ': 'I', 'ⅱ': 'II', 'ⅲ': 'III', 'ⅳ': 'IV', 'ⅴ': 'V',
            'ⅵ': 'VI', 'ⅶ': 'VII', 'ⅷ': 'VIII', 'ⅸ': 'IX', 'ⅹ': 'X',
        }
        for roman, normal in roman_map.items():
            result = result.replace(roman, normal)
        
        # 4. 应用代码转换规则（忽略大小写匹配）
        for old, new in self.code_conversion.items():
            old_normalized = old.replace(' ', '')
            # 使用正则忽略大小写匹配
            pattern = re.compile(re.escape(old_normalized), re.IGNORECASE)
            # 只替换匹配到的部分，但保留后续内容
            match = pattern.search(result)
            if match:
                result = result[:match.start()] + new + result[match.end():]
        
        # 5. 去除配置的特殊字符
        for char in self.remove_chars:
            result = result.replace(char, '')
        
        return result
    
    def _split_code_and_grade(self, encoded: str) -> Dict[str, str]:
        """
        分离编码的基础部分和等级后缀
        
        等级后缀定义：
        - 数字后面紧跟的罗马数字（I, II, III, IV, V, VI 等）
        - 数字后面紧跟的小写字母（如 Ia, IIa）
        
        Args:
            encoded: 完整编码（如 GBT13401I, GBT12459IIa）
            
        Returns:
            {
                'base': '基础编码（如 GBT13401）',
                'grade': '等级后缀（如 I, IIa）',
                'full': '完整编码'
            }
        """
        if not encoded:
            return {'base': '', 'grade': '', 'full': ''}
        
        # 匹配模式：数字后面紧跟的罗马数字（可能带小写字母后缀）
        # 例如：GBT13401I, GBT12459IIa, HGT20553Ia
        match = re.match(r'^(.+\d)((?:I{1,3}|IV|V|VI{0,3})(?:[a-z])?)$', encoded)
        if match:
            return {
                'base': match.group(1),
                'grade': match.group(2),
                'full': encoded
            }
        
        # 没有匹配到等级后缀
        return {'base': encoded, 'grade': '', 'full': encoded}
    
    def get_standard_info(self, standard: str) -> Dict:
        """
        获取标准的详细信息
        
        如果输入包含斜杠分隔的多个规范（如 ASME B36.19/B36.10），
        会先展开，分别处理，然后合并编码结果。
        """
        # 先展开斜杠分隔的规范
        expanded = self._expand_slash_standards([standard])
        
        # 统一处理（无论单个还是多个）
        all_formatted = []
        all_encoded = []
        all_categories = []
        
        for std in expanded:
            formatted = self._format_standard(std)
            category = self._classify_standard(formatted)
            encoded = self._encode_standard(formatted)
            all_formatted.append(formatted)
            all_encoded.append(encoded)
            all_categories.append(category)
        
        # 按分类排序：production > manufacturing > unknown
        category_order = {'production': 0, 'manufacturing': 1, 'unknown': 2}
        sorted_items = sorted(
            zip(all_encoded, all_categories, all_formatted),
            key=lambda x: (category_order.get(x[1], 2), x[0])
        )
        sorted_encoded = [item[0] for item in sorted_items]
        sorted_formatted = [item[2] for item in sorted_items]
        
        combined_encoded = ''.join(sorted_encoded)
        combined_formatted = ' | '.join(sorted_formatted) if len(sorted_formatted) > 1 else sorted_formatted[0]
        main_category = sorted_items[0][1] if sorted_items else 'unknown'
        
        # 对于单个规范，提取等级信息
        if len(expanded) == 1:
            code_parts = self._split_code_and_grade(combined_encoded)
            base_code = code_parts['base']
            grade = code_parts['grade']
        else:
            base_code = combined_encoded
            grade = ''
        
        return {
            'original': standard,
            'formatted': combined_formatted,
            'category': main_category,
            'encoded': combined_encoded,
            'base_code': base_code,
            'grade': grade
        }
    
    def process(self, value: str) -> str:
        """
        处理单个规范值，返回编码字符串
        
        Args:
            value: 规范值
            
        Returns:
            编码字符串
        """
        if not value:
            return ""
        info = self.get_standard_info(value.strip())
        return info['encoded']
    
    def process_multi(self, values: List[str]) -> str:
        """
        处理多个规范值并排序
        
        Args:
            values: 规范值列表
            
        Returns:
            排序后拼接的编码
        """
        return self.process_standards(values)
    
    def process_multi_with_detail(self, values: List[str]) -> Dict:
        """
        处理多个规范值并返回带分类详情的结果
        
        Args:
            values: 规范值列表
            
        Returns:
            带分类详情的字典
        """
        return self.process_standards_with_detail(values)
    
    def process_standards_with_detail(self, standards: List[str]) -> Dict:
        """
        处理标准数组，返回带分类详情的结果
        
        Args:
            standards: 原始标准列表
            
        Returns:
            {
                'encoded': '编码字符串',
                'production': ['生产标准1', '生产标准2'],  
                'manufacturing': ['制造标准1', '制造标准2'],
                'unknown': ['未知标准1', '未知标准2'],
                'production_encoded': ['编码1', '编码2'],
                'manufacturing_encoded': ['编码1', '编码2'],
                'unknown_encoded': ['编码1', '编码2'],
                'display': '编码(类型) 编码(类型)'  # 用于显示
            }
        """
        if not standards:
            return {
                'encoded': '',
                'production': [],
                'manufacturing': [],
                'unknown': [],
                'production_encoded': [],
                'manufacturing_encoded': [],
                'unknown_encoded': [],
                'display': '无'
            }
        
        # 0. 先展开斜杠分隔的复合规范（如 ASME B36.19/B36.10 → ASME B36.19, ASME B36.10）
        expanded_standards = self._expand_slash_standards(standards)
        
        # 分类处理
        production_items = []  # [(原始, 格式化, 编码), ...]
        manufacturing_items = []
        unknown_items = []
        
        for std in expanded_standards:
            formatted = self._format_standard(std)
            category = self._classify_standard(formatted)
            encoded = self._encode_standard(formatted)
            code_parts = self._split_code_and_grade(encoded)
            
            # item: (原始, 格式化, 完整编码, 基础编码, 等级)
            item = (std, formatted, encoded, code_parts['base'], code_parts['grade'])
            if category == 'production':
                production_items.append(item)
            elif category == 'manufacturing':
                manufacturing_items.append(item)
            else:
                unknown_items.append(item)
        
        # 分别排序
        production_items = self._sort_items(production_items)
        manufacturing_items = self._sort_items(manufacturing_items)
        unknown_items = self._sort_items(unknown_items)
        
        # 组装结果
        production_encoded = [item[2] for item in production_items]
        manufacturing_encoded = [item[2] for item in manufacturing_items]
        unknown_encoded = [item[2] for item in unknown_items]
        
        # 生成显示字符串：编码(生产/制造/无)，按基础编码去重
        # 如果同一基础编码有多个版本（有等级/无等级），优先保留有等级的
        display_parts = []
        seen_base_codes = {}  # base_code -> (full_code, category, has_grade)
        
        all_items = (
            [(item, '生产') for item in production_items] +
            [(item, '制造') for item in manufacturing_items] +
            [(item, '') for item in unknown_items]
        )
        
        for item, category_label in all_items:
            full_code = item[2]
            base_code = item[3]
            grade = item[4]
            
            if not full_code:
                continue
            
            if base_code not in seen_base_codes:
                # 第一次见到这个基础编码
                seen_base_codes[base_code] = (full_code, category_label, bool(grade))
            else:
                # 已有这个基础编码，判断是否替换
                existing_code, existing_category, existing_has_grade = seen_base_codes[base_code]
                if grade and not existing_has_grade:
                    # 新的有等级，旧的没有，替换
                    seen_base_codes[base_code] = (full_code, category_label, True)
        
        # 按原始顺序生成显示部分
        added_bases = set()
        for item, category_label in all_items:
            base_code = item[3]
            if base_code and base_code not in added_bases and base_code in seen_base_codes:
                full_code, cat_label, _ = seen_base_codes[base_code]
                if cat_label:
                    display_parts.append(f"{full_code}({cat_label})")
                else:
                    display_parts.append(f"{full_code}")
                added_bases.add(base_code)
        
        # 最终编码：按基础编码去重，优先保留有等级的版本
        unique_encoded = [info[0] for info in seen_base_codes.values()]
        
        return {
            'encoded': ''.join(unique_encoded),
            'production': [item[0] for item in production_items],
            'manufacturing': [item[0] for item in manufacturing_items],
            'unknown': [item[0] for item in unknown_items],
            'production_encoded': production_encoded,
            'manufacturing_encoded': manufacturing_encoded,
            'unknown_encoded': unknown_encoded,
            'display': ' '.join(display_parts) if display_parts else '无'
        }
    
    def _sort_items(self, items: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        对标准项进行排序
        items: [(原始, 格式化, 编码), ...]
        使用编码后的值排序（格式统一）
        """
        def sort_key(item) -> Tuple[int, str]:
            encoded = item[2]  # 使用编码后的值排序（格式统一）
            prefix_match = re.match(r'^([A-Z]+)', encoded.upper())
            prefix = prefix_match.group(1) if prefix_match else 'ZZZ'
            
            priority = 999
            for p, pri in self.prefix_priority.items():
                if prefix.startswith(p):
                    priority = pri
                    break
            
            # 同前缀按字典序排序（逐字符比较）
            return (priority, encoded.upper())
        
        return sorted(items, key=sort_key)
    
    def encode_grade(self, grade_value: str) -> str:
        """
        编码规范等级
        
        Args:
            grade_value: 等级值（如 "Series I", "TYPE II", "I类", "Ia"）
            
        Returns:
            等级编码（如 "I", "II", "Ia"）
        """
        if not grade_value:
            return ''
        
        value = grade_value.strip()
        
        # 1. 处理 "Series I", "Serial II", "TYPE III", "Method C", "Design A" 等格式
        match = re.search(r'(?:series|serial|type|method|design)\s*[-]?\s*([IVXivx]+[a-z]?|\d+[a-z]?|[A-Za-z])', value, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 2. 处理中文"方法A", "方法C" 等格式
        match = re.search(r'方法\s*([A-Za-z0-9]+)', value)
        if match:
            return match.group(1).upper()
        
        # 3. 处理 "I类", "II类填充金属" 等中文格式
        match = re.search(r'([IVXivx\d]+)类', value)
        if match:
            return match.group(1).upper() if match.group(1).isalpha() else match.group(1)
        
        # 4. 处理 "I系列", "II系列" 等格式
        match = re.search(r'([IVXivx\d]+)系列', value)
        if match:
            return match.group(1).upper() if match.group(1).isalpha() else match.group(1)
        
        # 5. 如果是纯等级标记（如 "I", "II", "Ia"），直接返回
        if re.match(r'^[IVXivx]+[a-z]?$', value):
            return value.upper() if value.isalpha() else value
        
        # 6. 全角罗马数字转换
        roman_map = {
            'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V',
            'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX', 'Ⅹ': 'X',
        }
        for roman, normal in roman_map.items():
            if roman in value:
                return normal
        
        # 默认返回原值（去除空格）
        return value.replace(' ', '')

    def encode_appendix(self, appendix_value: str) -> str:
        """编码规范附录，如 附录B / Appendix B -> B。"""
        if not appendix_value:
            return ''
        value = appendix_value.strip()

        match = re.search(r'(?:附录|Appendix)\s*([A-Za-z0-9]+)', value, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        if re.match(r'^[A-Za-z0-9]+$', value):
            return value.upper()

        return value.replace(' ', '')

    def encode_method(self, method_value: str) -> str:
        """编码规范方法，如 方法B / Method B / Design A -> B / A。"""
        if not method_value:
            return ''
        value = method_value.strip()

        match = re.search(r'(?:方法|Method|Design)\s*([A-Za-z0-9]+)', value, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        if re.match(r'^[A-Za-z0-9]+$', value):
            return value.upper()

        return value.replace(' ', '')

    def encode_modifier(self, modifier_type: str, modifier_value: str) -> str:
        """按修饰符类型编码。"""
        if modifier_type == 'STANDARD_GRADE':
            return self.encode_grade(modifier_value)
        if modifier_type == 'STANDARD_APPENDIX':
            return self.encode_appendix(modifier_value)
        if modifier_type == 'STANDARD_METHOD':
            return self.encode_method(modifier_value)
        return ''

    def process_with_modifiers(
        self,
        standards: List[str],
        modifier_map: Dict[int, Dict[str, List[str]]] = None
    ) -> Dict:
        """
        处理规范列表，并按固定顺序拼接修饰项：
        STANDARD_GRADE + STANDARD_APPENDIX + STANDARD_METHOD
        """
        if not standards:
            return self.process_standards_with_detail([])

        merged_standards = list(standards)
        modifier_map = modifier_map or {}

        for idx, modifier_info in modifier_map.items():
            if not (0 <= idx < len(merged_standards)):
                continue

            std = merged_standards[idx]
            std_encoded = self._encode_standard(self._format_standard(std))
            suffix_parts: List[str] = []
            appended_by_field: Dict[str, List[str]] = {}

            for modifier_type in self.MODIFIER_ORDER:
                raw_values = modifier_info.get(modifier_type, []) or []
                if not isinstance(raw_values, list):
                    raw_values = [raw_values]

                for raw_value in raw_values:
                    if not raw_value:
                        continue
                    modifier_code = self.encode_modifier(modifier_type, str(raw_value))
                    if not modifier_code:
                        continue

                    if modifier_type == 'STANDARD_GRADE' and std_encoded.endswith(modifier_code):
                        continue

                    existing_codes = appended_by_field.setdefault(modifier_type, [])
                    if modifier_code in existing_codes:
                        continue

                    suffix_parts.append(str(raw_value).strip())
                    existing_codes.append(modifier_code)

            if suffix_parts:
                merged_standards[idx] = f"{std} {' '.join(suffix_parts)}"

        return self.process_standards_with_detail(merged_standards)
    
    def process_with_grades(self, standards: List[str], grade_map: Dict[int, str] = None) -> Dict:
        """
        处理规范列表，支持单独的等级映射
        
        Args:
            standards: 规范列表（如 ['NB/T47010', 'GB/T 14383']）
            grade_map: 等级映射 {规范索引: 等级值}（如 {1: 'Series I'}）
            
        Returns:
            与 process_standards_with_detail 相同的结构，但等级已正确拼接
        """
        modifier_map = {}
        for idx, grade_value in (grade_map or {}).items():
            modifier_map[idx] = {"STANDARD_GRADE": [grade_value]}
        return self.process_with_modifiers(standards, modifier_map)


# 单例
_processor_instance: Optional[StandardProcessor] = None


def get_standard_processor() -> StandardProcessor:
    """获取标准处理器单例"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = StandardProcessor()
    return _processor_instance


# 测试代码
if __name__ == '__main__':
    processor = StandardProcessor()
    
    # 测试用例
    test_standards = [
        # 年份处理
        'GB/T12459-17 SERIES I',  # 2位年份 + 等级
        'GB/T12459-2017 Series I',# 4位年份 + 等级（保留 Series I 的大小写）
        'GB/T12459-17',           # 无等级，直接结尾
        'GB/T14383-08',           # 08年
        
        # 等级处理（各种格式）
        'HG/T 20553(Ia)',         # 括号等级 Ia
        'HG/T 20553(II)',         # 括号等级 II  
        'GB/T 12771 TYPE V',      # TYPE 等级
        'GB/T 19326 Series I',    # Series 等级（小写 eries）
        
        # 前缀转换
        'ASME B16.9',             # ASMEB -> AB
        'MSS SP-97',              # MSSSP -> MS, -97 保留
        
        # 其他
        'NB/T 47010',
        'GB/T 14976',
        'GB/T 4334-C',            # -C 是后缀，不是年份
    ]
    
    print("=== 单个标准测试 ===")
    for std in test_standards:
        info = processor.get_standard_info(std)
        print(f"原始: {info['original']}")
        print(f"格式化: {info['formatted']}")
        print(f"分类: {info['category']}")
        print(f"编码: {info['encoded']}")
        print("-" * 40)
    
    print("\n=== 多标准排序测试 ===")
    result = processor.process_standards(test_standards)
    print(f"输入: {test_standards}")
    print(f"输出: {result}")
    
    print("\n=== 斜杠分隔规范测试 ===")
    slash_tests = [
        # 应该拆分的（斜杠不是前缀的一部分）
        ('ASME B36.19/B36.10', True, '两个ASME标准'),
        ('ASME B36.10/B36.19', True, '两个ASME标准（反序）'),
        ('GB2019-b/2018', True, '斜杠分隔两个标准号'),
        ('ASME B16.5/B16.47', True, '两个ASME标准'),
        
        # 不应该拆分的（前缀格式：字母/单字母）
        ('GB/T 12459', False, '前缀 GB/T'),
        ('GB/T12459', False, '前缀 GB/T（无空格）'),
        ('HG/T 20553', False, '前缀 HG/T'),
        ('HG/T20553', False, '前缀 HG/T（无空格）'),
        ('SH/T 3406', False, '前缀 SH/T'),
        ('NB/T 47010', False, '前缀 NB/T'),
        ('SH/B 12345', False, '假设的新前缀 SH/B（字母/单字母）'),
        ('XX/Y 99999', False, '任意新前缀 XX/Y（字母/单字母）'),
    ]
    for test_input, should_split, description in slash_tests:
        expanded = processor._expand_slash_standards([test_input])
        was_split = len(expanded) > 1
        status = '✓' if was_split == should_split else '✗'
        print(f"{status} 输入: {test_input}")
        print(f"  说明: {description}")
        print(f"  展开: {expanded}")
        if was_split != should_split:
            print(f"  警告: 预期{'拆分' if should_split else '不拆分'}，实际{'拆分' if was_split else '不拆分'}")
        print("-" * 40)
