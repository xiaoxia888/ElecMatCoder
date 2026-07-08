"""
文本预处理模块
"""

import re
from pathlib import Path

import yaml


class TextPreprocessor:
    """文本预处理器"""
    
    # 全角罗马数字转半角映射（统一格式）
    ROMAN_FULL_TO_HALF = {
        'Ⅰ': 'I',     # 1
        'Ⅱ': 'II',    # 2
        'Ⅲ': 'III',   # 3
        'Ⅳ': 'IV',    # 4
        'Ⅴ': 'V',     # 5
        'Ⅵ': 'VI',    # 6
        'Ⅶ': 'VII',   # 7
        'Ⅷ': 'VIII',  # 8
        'Ⅸ': 'IX',    # 9
        'Ⅹ': 'X',     # 10
        'Ⅺ': 'XI',    # 11
        'Ⅻ': 'XII',   # 12
    }
    
    # 全角转半角映射
    FULL_TO_HALF = {
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
        'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
        'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
        'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
        'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
        'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
        'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
        'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
        'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
        'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y',
        'ｚ': 'z',
        '．': '.', '／': '/', '－': '-', '＋': '+',
        '（': '(', '）': ')', '　': ' ',
    }

    # 安全可替换的列表分隔符（不包含 "/"，避免破坏 304/316、GB/T 等语义）
    SAFE_LIST_SEPARATORS = "，；、|｜"
    _COMMON_DN_VALUES: set[int] | None = None

    def __init__(self):
        pass
    
    def process(self, text: str) -> str:
        """
        完整预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 1. 去除首尾空格
        text = text.strip()

        # 2. 全角转半角
        text = self.full_to_half(text)

        # 3. 统一罗马数字（半角转全角，匹配训练数据格式）
        text = self.normalize_roman_numerals(text)

        # 4. 标准化常见乘号（不改变 slash 语义）
        text = self.normalize_multiplication(text)

        # 5. 收紧小数点两侧数字间的误空格：12. 70 -> 12.70, 12 .70 -> 12.70
        text = self.normalize_decimal_spacing(text)

        # 6. 统一外径前缀写法
        text = self.normalize_diameter_prefix(text)

        # 7. 保守分隔符标准化（只处理安全分隔符）
        text = self.normalize_safe_separators(text)

        # 8. 窄范围规格归一化（只修规则层高频脏写法）
        text = self.normalize_pipe_spec_tokens(text)

        # 9. 切开历史字段标签/规格粘连，后续所有模块统一吃同一份 processed_text
        text = self.normalize_section_labels(text)

        # 10. 切开 DN 与壁厚/壁厚号粘连，避免规则层各自私改文本
        text = self.normalize_glued_dn_wall_thickness(text)

        # 11. 结构字段局部 OCR/录入纠错（只在强锚点局部片段中生效）
        # text = self.normalize_structural_ocr_tokens(text)

        # 12. 材质 token 局部 OCR/录入纠错（只修高置信脏写法）
        text = self.normalize_material_ocr_tokens(text)

        # 13. 种类别名/短写归一化（只修高置信历史简称）
        text = self.normalize_type_alias_tokens(text)

        # 14. 删除连接方式噪声词，避免干扰尺寸/壁厚等结构字段
        # text = self.remove_connection_noise_tokens(text)

        # 15. OCR 纠偏后再次收紧小数点两侧空格：
        # 例如 S-3. Omm -> S-3. 0mm -> S-3.0mm
        text = self.normalize_decimal_spacing(text)

        # 16. 对强结构 token 做安全切分，避免与前后脏串粘连
        text = self.normalize_strong_structural_tokens(text)

        # 17. 删除容易被误判为壁厚的工程标准短语
        text = self.remove_non_thickness_standard_phrases(text)

        # 18. 空白压缩
        text = self.normalize_whitespace(text)

        return text
    
    def full_to_half(self, text: str) -> str:
        """全角转半角"""
        result = []
        for char in text:
            if char in self.FULL_TO_HALF:
                result.append(self.FULL_TO_HALF[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def normalize_roman_numerals(self, text: str) -> str:
        """
        统一罗马数字格式
        将全角罗马数字转为半角（统一格式，训练和预测保持一致）
        例如：NB/T 47010Ⅱ → NB/T 47010II
              HG/T20592Ⅱ → HG/T20592II
        """
        result = []
        for char in text:
            if char in self.ROMAN_FULL_TO_HALF:
                result.append(self.ROMAN_FULL_TO_HALF[char])
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def normalize_multiplication(text: str) -> str:
        """统一乘号写法，便于后续尺寸/壁厚识别。"""
        # 不把 '×' 强转成 'X'。
        # 在管道场景里，`DN15×XS` 若被改成 `DN15XXS`，会直接把原文语义改坏。
        # 结构提示词和规则层本身都支持 `×`，所以这里只把 `*` 归一到 `×`。
        return text.replace('*', '×')

    @staticmethod
    def normalize_decimal_spacing(text: str) -> str:
        """
        收紧小数点左右被误插入的空格。

        仅处理明确满足“点号左右都是数字”的情况：
        - 12. 70 -> 12.70
        - 12 .70 -> 12.70
        - 12 . 70 -> 12.70
        """
        if not text:
            return ""
        return re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)

    @staticmethod
    def normalize_diameter_prefix(text: str) -> str:
        """
        统一外径前缀写法到 `Φ`。

        目标：
        - φ / Φ / Ф / ф / Ø / ø 统一成 `Φ`

        说明：
        - 这里只处理明显是“直径前缀字符”的变体，不动 `D/OD`
        - 放在公共预处理里，供尺寸/壁厚/测试入口统一复用
        """
        if not text:
            return ""
        return re.sub(r"[ΦφФфØø]", "Φ", text)

    def normalize_safe_separators(self, text: str) -> str:
        """
        保守分隔符标准化。
        只替换确定是“列表分隔”的符号为 ';'，保留 '/' 原样。
        """
        trans = str.maketrans({ch: ';' for ch in self.SAFE_LIST_SEPARATORS})
        text = text.translate(trans)

        # 合并重复分号，并去掉两侧多余空白
        text = re.sub(r'\s*;\s*', ';', text)
        text = re.sub(r';{2,}', ';', text)
        return text.strip(';')

    @classmethod
    def _get_common_dn_values(cls) -> set[int]:
        if cls._COMMON_DN_VALUES is not None:
            return cls._COMMON_DN_VALUES
        config_path = Path(__file__).resolve().parent.parent / "encoder" / "config" / "encoder_config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            size_config = config.get("size_processing", {}) or {}
            cls._COMMON_DN_VALUES = {
                int(v) for v in size_config.get("common_dn_values", []) if str(v).strip()
            }
        except Exception:
            cls._COMMON_DN_VALUES = set()
        return cls._COMMON_DN_VALUES

    @staticmethod
    def normalize_section_labels(text: str) -> str:
        """
        切开历史表中常见的编号字段标签粘连：
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

    @classmethod
    def normalize_glued_dn_wall_thickness(cls, text: str) -> str:
        """
        切开 `DN1506.3mm` / `DN150S-40` 这类 `DN + 壁厚` 粘连：
        - DN200XDN1506.3mmX7.1mm -> DN200XDN150 6.3mmX7.1mm
        - DN150×DN40S-10S×SCH40S -> DN150×DN40 S-10S×SCH40S

        这是统一格式化的一部分，不属于尺寸/壁厚规则器自己的私有处理。
        """
        common_dn_values = cls._get_common_dn_values()
        if not common_dn_values:
            return text
        dn_tokens = sorted((str(v) for v in common_dn_values), key=len, reverse=True)
        atomic_dn_group = f"(?>{'|'.join(map(re.escape, dn_tokens))})"
        decimal_mm_pattern = re.compile(
            rf'(?i)(DN\s*)({atomic_dn_group})(?=(\d+\.\d+\s*(?:MM|毫米)(?:\b|\s*[xX×/,;)])))'
        )
        text = decimal_mm_pattern.sub(r'\1\2 ', text)
        glued_schedule_pattern = re.compile(
            rf'(?i)(DN\s*)({atomic_dn_group})(?=((?:S-\d+S?|SCH\d+S?|\d+S)\b))'
        )
        return glued_schedule_pattern.sub(r'\1\2 ', text)

    @staticmethod
    def normalize_pipe_spec_tokens(text: str) -> str:
        """
        针对管道编码高频脏写法做最小范围归一化。

        只修 token 内部空格，不改普通分隔空格：
        - SCH10 S -> SCH10S
        - SCH 10 S -> SCH10S
        - S-10 S -> S-10S
        - SCH10 SXSCH10S -> SCH10SXSCH10S
        """
        if not text:
            return ""

        sch_end_guard = r'(?=$|[;,/()xX×*]|\s+(?![0-9.]))'

        # 先只修 SCH 字母自身被空格打断的情况：S C H40 / SC H 40 -> SCH40
        text = re.sub(
            rf'(?i)(?:(?<=^)|(?<=[;,\s/xX×]))S\s*C\s*H',
            'SCH',
            text,
        )

        # 只有在存在尾部 S 时，才允许把 SCH 与尾部 S 之间“只由空格和数字组成”的片段压紧：
        # SCH 4 0 S -> SCH40S
        text = re.sub(
            rf'(?i)(?:(?<=^)|(?<=[;,\s/xX×]))SCH\s*(([0-9]\s*)+)S{sch_end_guard}',
            lambda m: f"SCH{re.sub(r'\s+', '', m.group(1) or '')}S",
            text,
        )

        # SCH 字母归一后，允许收紧 SCH 与纯数字之间的空格：SCH 40 -> SCH40
        text = re.sub(
            rf'(?i)(?:(?<=^)|(?<=[;,\s/xX×]))SCH\s*([0-9]+)(S?){sch_end_guard}',
            lambda m: f"SCH{m.group(1)}{(m.group(2) or '').upper()}",
            text,
        )

        # SCH 体系：SCH10 S / SCH 10 S / SCH40 S -> SCH10S / SCH40S
        text = re.sub(
            rf'(?i)(?:(?<=^)|(?<=[;,\s/xX×]))SCH\s*([0-9]+)\s*S{sch_end_guard}',
            lambda m: f"SCH{m.group(1)}S",
            text,
        )
        text = re.sub(
            rf'(?i)\bSCH\s*([0-9]+){sch_end_guard}',
            lambda m: f"SCH{m.group(1)}",
            text,
        )

        # S- 体系：S-10 S -> S-10S
        text = re.sub(
            rf'(?i)(?:(?<=^)|(?<=[;,\s/xX×]))S-\s*([0-9]+)\s*S{sch_end_guard}',
            lambda m: f"S-{m.group(1)}S",
            text,
        )

        # 紧凑 x 组合里残留空格：SCH10S X SCH10 S -> SCH10SX SCH10S
        text = re.sub(r'(?i)\b(SCH[0-9]+S?)\s+([xX×])\s+(SCH[0-9]+S?)\b', r'\1\2\3', text)
        text = re.sub(r'(?i)\b(S-[0-9]+S?)\s+([xX×])\s+(SCH[0-9]+S?)\b', r'\1\2\3', text)
        text = re.sub(r'(?i)\b(SCH[0-9]+S?)\s+([xX×])\s+(S-[0-9]+S?)\b', r'\1\2\3', text)

        return text

    @staticmethod
    def normalize_strong_structural_tokens(text: str) -> str:
        """
        对极强结构 token 做安全切分：如果前后与其他内容粘连，则补空格。

        目标：
        - H2SSCH30×SCH160 -> H2S SCH30×SCH160
        - XXXDN300X40YYY -> XXX DN300X40 YYY
        - A105OD88.9X6.3 -> A105 OD88.9X6.3

        这里只切这类结构非常强、歧义很低的 token，不改 token 内部内容。
        """
        if not text:
            return ""

        def _is_wordlike_neighbor(ch: str) -> bool:
            return ch.isalnum() or ("\u4e00" <= ch <= "\u9fff")

        schedule_token = r'(?:SCH[.\s-]*\d+S?|S-\d+S?|S\d+S?|\d+S|XXS|XS|STD)'
        strong_patterns = (
            # 尺寸：DN数字x数字 / DN数字
            re.compile(r'(?i)DN\s*\d+(?:\.\d+)?\s*[xX×*]\s*(?:DN\s*)?\d+(?:\.\d+)?'),
            re.compile(r'(?i)DN\s*\d+(?:\.\d+)?(?!\s*[xX×*]\s*(?:DN\s*)?\d)'),
            # 尺寸：OD/外径/Φ 数字 x 数字 / 单值
            re.compile(r'(?i)(?:OD|外径|Φ)\s*\d+(?:\.\d+)?\s*[xX×*]\s*(?:(?:OD|外径|Φ)\s*)?\d+(?:\.\d+)?(?:\s*(?:MM|毫米))?'),
            re.compile(r'(?i)(?:OD|外径|Φ)\s*\d+(?:\.\d+)?(?!\s*[xX×*]\s*(?:(?:OD|外径|Φ)\s*)?\d)'),
            # 壁厚：SCH数字 / SCH数字S / SCH...xSCH...
            re.compile(rf'(?i){schedule_token}\s*[xX×*]\s*{schedule_token}'),
            re.compile(rf'(?i)SCH[.\s-]*\d+S?(?!\s*[xX×*]\s*{schedule_token})'),
            # 壁厚：S-数字 / S-数字S / S-...xS-...
            re.compile(rf'(?i)S-\d+S?(?!\s*[xX×*]\s*{schedule_token})'),
        )

        matches: list[tuple[int, int]] = []
        for pattern in strong_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                matches.append((start, end))

        if not matches:
            return text

        # 同一起点优先保留更长命中的 token，并丢弃被更长 token 覆盖的短命中。
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        selected: list[tuple[int, int]] = []
        for start, end in matches:
            if selected and start < selected[-1][1]:
                continue
            selected.append((start, end))

        result = text
        for start, end in reversed(selected):
            prefix = ""
            suffix = ""
            if start > 0 and _is_wordlike_neighbor(result[start - 1]) and not result[start - 1].isspace():
                prefix = " "
            if end < len(result) and _is_wordlike_neighbor(result[end]) and not result[end].isspace():
                suffix = " "
            result = result[:start] + prefix + result[start:end] + suffix + result[end:]
        return result

    @staticmethod
    def normalize_structural_ocr_tokens(text: str) -> str:
        """
        只在强结构片段里修正常见 OCR/录入错误：
        - l / I -> 1
        - O / o -> 0

        不做全文替换，避免打坏材质、标准号、普通单词。
        当前仅覆盖：
        - DN 规格片段
        - OD/φ/Φ/Ф/D 规格片段
        - THK/T= 壁厚片段
        - L= 长度片段
        - SCH / S- schedule 片段
        """
        if not text:
            return ""

        light_delimiters = set(" .,/xX×*-")
        confusion_map = {
            "O": "0",
            "o": "0",
            "I": "1",
            "i": "1",
            "l": "1",
        }

        def nearest_effective_is_digit(segment: str, idx: int) -> bool:
            left = idx - 1
            while left >= 0 and segment[left] in light_delimiters:
                left -= 1
            if left >= 0 and segment[left].isdigit():
                return True

            right = idx + 1
            while right < len(segment) and segment[right] in light_delimiters:
                right += 1
            if right < len(segment) and segment[right].isdigit():
                return True

            return False

        def normalize_numeric_confusions(segment: str) -> str:
            chars = list(segment)
            for idx, ch in enumerate(chars):
                repl = confusion_map.get(ch)
                if repl is None:
                    continue
                if nearest_effective_is_digit(segment, idx):
                    chars[idx] = repl
            return "".join(chars)

        def apply(pattern: str, src: str) -> str:
            return re.sub(pattern, lambda m: normalize_numeric_confusions(m.group(0)), src)

        def apply_od_like(src: str) -> str:
            pattern = r"(?i)(?:\bOD|[ΦφФD])\s*[0-9OIol]+(?:\.[0-9OIol]+)?(?:\s*[xX×*]\s*(?:[ΦφФD]\s*)?[0-9OIol]+(?:\.[0-9OIol]+)?){0,2}"

            def repl(match: re.Match[str]) -> str:
                segment = match.group(0)
                prefix_match = re.match(r"(?i)(OD|[ΦφФD])", segment)
                if not prefix_match:
                    return normalize_numeric_confusions(segment)
                prefix = prefix_match.group(0)
                rest = segment[len(prefix) :]
                return prefix + normalize_numeric_confusions(rest)

            return re.sub(pattern, repl, src)

        def apply_pressure_like(src: str) -> str:
            patterns = (
                r"(?i)\bPN\s*[0-9OIoli]+(?:\.[0-9OIoli]+)?\b",
                r"(?i)\bCL\s*\.?\s*[0-9OIoli]+\b",
                r"(?i)\bCLASS\s*\.?\s*[0-9OIoli]+\b",
                r"(?i)\b[0-9OIoli]+\s*(?:LB|LBS)\b",
                r"(?i)\b[0-9OIoli]+#(?![A-Za-z0-9])",
            )

            def repl(match: re.Match[str]) -> str:
                segment = match.group(0)
                prefix_match = re.match(r"(?i)(PN|CL|CLASS)", segment)
                if prefix_match:
                    prefix = prefix_match.group(0)
                    rest = segment[len(prefix) :]
                    return prefix + normalize_numeric_confusions(rest)

                suffix_match = re.search(r"(?i)(LB|LBS|#)$", segment)
                if suffix_match:
                    suffix = suffix_match.group(0)
                    rest = segment[: -len(suffix)]
                    return normalize_numeric_confusions(rest) + suffix

                return normalize_numeric_confusions(segment)

            for pattern in patterns:
                src = re.sub(pattern, repl, src)
            return src

        # DN 尺寸：DN150xl5、DN25O、DN1OO、DN150xDN5O
        text = apply(
            r"(?i)\bDN\s*[0-9OIol]+(?:\s*[xX×*/-]\s*(?:DN\s*)?[0-9OIol]+){0,2}",
            text,
        )

        # 显式 OD/φ/D 尺寸：φ1O8x4.O、OD1O8x4.O、D6O.3x3.91
        text = apply_od_like(text)

        # 显式壁厚：THK=4. Omm、T=1O.5mm
        text = apply(
            r"(?i)\b(?:THK|T)\s*=\s*[0-9OIol.\s]+(?:MM|毫米)\b",
            text,
        )

        # 长度：L=1OOOmm
        text = apply(
            r"(?i)\bL\s*=\s*[0-9OIol.\s]+(?:MM|毫米)\b",
            text,
        )

        # S 数值型壁厚：S-3. Omm、S=3. O、S3. Omm、S3mm
        # 这里不做全文放宽，只覆盖：
        # 1) 显式带符号：S- / S= / S:
        # 2) 无符号但带小数点
        # 3) 无符号但显式带 mm/毫米
        text = apply(
            r"(?i)\bS\s*[-:=：]\s*[0-9OIol]+(?:\s*\.\s*[0-9OIol]+)?\s*(?:MM|毫米)?\b",
            text,
        )
        text = apply(
            r"(?i)\bS\s*[0-9OIol]+\s*\.\s*[0-9OIol]+\s*(?:MM|毫米)?\b",
            text,
        )
        text = apply(
            r"(?i)\bS\s*[0-9OIol]+\s*(?:MM|毫米)\b",
            text,
        )

        # Schedule：SCH3O、SCH1OS、S-1OS
        text = apply(
            r"(?i)\bSCH\s*[0-9OIol]+\s*S?\b",
            text,
        )
        text = apply(
            r"(?i)\bS-\s*[0-9OIol]+\s*S?\b",
            text,
        )

        # 磅级：PNi6、CL3OO、Class3OO0、15OLB、15O#
        text = apply_pressure_like(text)

        return text

    @staticmethod
    def normalize_material_ocr_tokens(text: str) -> str:
        """
        只修高置信材质/特殊要求 token 的 OCR 误写。

        当前仅覆盖：
        - Ga1V / GA1V -> GaLV / GALV
        - Ga1V. / GA1V. -> GaLV. / GALV.

        说明：
        - 这是 `GALV` 中字母 `L` 被 OCR 误成数字 `1` 的高频脏写法。
        - 只修完整 token，避免误伤普通单词和型号串。
        """
        if not text:
            return ""

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            suffix = ""
            if token.endswith("."):
                suffix = "."
                token = token[:-1]

            if token.isupper():
                normalized = "GALV"
            elif token.islower():
                normalized = "galv"
            elif token.startswith("Ga"):
                normalized = "GaLV"
            else:
                normalized = "GALV"
            return normalized + suffix

        return re.sub(r"(?i)\bga1v\.?\b", repl, text)

    @staticmethod
    def normalize_type_alias_tokens(text: str) -> str:
        """
        归一化高频历史简称/短写种类词。

        当前仅覆盖：
        - 无偏头 -> 无缝偏心大小头
        - 无缝偏大 -> 无缝偏心大小头
        - 偏心头 -> 偏心大小头
        - SW弯头 -> 承插焊弯头

        说明：
        - 这些写法是历史描述里的高频简称，语义比较稳定。
        - 左右边界按 alias 首尾字符类型自动推断：
          - 首字符是汉字 -> 左边不能是汉字
          - 首字符是英文字母 -> 左边不能是英文字母
          - 首字符是数字 -> 左边不能是数字
          - 尾字符同理
        - 若 alias 含英文字母，则自动忽略大小写。
        - 多个 alias 有包含关系时，按“最长优先”匹配，避免短词抢占长词。
        """
        if not text:
            return ""

        replacements = (
            ("无缝偏大", "无缝偏心大小头"),
            ("无偏头", "无缝偏心大小头"),
            ("偏心头", "偏心大小头"),
            ("SW弯头", "承插焊弯头"),
            ("偏头", "偏心大小头"),
            ("偏大", "偏心大小头"),
            ("CAP", "管帽"),
            ("TEE", "三通"),
            ("BW Olet", "对焊支管台"),
            ("RTS", "异径三通"),
            ("RK", "同心异径管"),
            ("WOL-90", "90度对焊支管台"),
            ("WOL-45", "45度对焊支管台"),
            
        )

        def _char_boundary(kind_char: str, *, is_left: bool) -> str:
            if "\u4e00" <= kind_char <= "\u9fff":
                return r"(?<![\u4e00-\u9fff])" if is_left else r"(?![\u4e00-\u9fff])"
            if re.match(r"[A-Za-z]", kind_char):
                return r"(?<![A-Za-z])" if is_left else r"(?![A-Za-z])"
            if kind_char.isdigit():
                return r"(?<!\d)" if is_left else r"(?!\d)"
            return ""

        normalized_rules = sorted(replacements, key=lambda item: len(str(item[0] or "")), reverse=True)

        for source, target in normalized_rules:
            if not source or not target:
                continue
            left_boundary = _char_boundary(source[0], is_left=True)
            right_boundary = _char_boundary(source[-1], is_left=False)
            flags = re.IGNORECASE if re.search(r"[A-Za-z]", source) else 0
            pattern = re.compile(
                left_boundary + "(" + re.escape(source) + ")" + right_boundary,
                flags,
            )
            text = pattern.sub(target, text)

        return text

    @staticmethod
    def remove_connection_noise_tokens(text: str) -> str:
        """
        删除只表示连接方式、但容易干扰结构字段的噪声词。

        - 中文 `对焊` 直接删除。
        - 英文 `BW` 只在左右都不是英文字母时删除，避免误伤 BWG / ABW / BWN。
        """
        if not text:
            return ""

        text = text.replace("对焊", " ")
        text = re.sub(r"(?i)(?<![A-Za-z])BW(?![A-Za-z])(?:\s*[-_/]\s*)?", " ", text)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """压缩空白字符，保留单个空格。"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_non_thickness_standard_phrases(text: str) -> str:
        """
        删除容易被误判为壁厚的工程标准短语。

        这些短语在实际描述里通常表示工程标准/企业标准，
        不是壁厚信息，例如：ENR STD、MFR STD。
        """
        if not text:
            return ""

        patterns = (
            r"(?i)\bMNF\s+STD\b",
            r"(?i)\bMFR\s+STD\b",
            r"(?i)\bMFRS\s+STD\b",
            r"(?i)\bENR\s+STD\b",
        )

        for pattern in patterns:
            text = re.sub(pattern, " ", text)

        return text
