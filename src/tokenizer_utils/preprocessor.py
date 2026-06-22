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
        text = self.normalize_structural_ocr_tokens(text)

        # 12. OCR 纠偏后再次收紧小数点两侧空格：
        # 例如 S-3. Omm -> S-3. 0mm -> S-3.0mm
        text = self.normalize_decimal_spacing(text)

        # 13. 空白压缩
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
    def normalize_whitespace(text: str) -> str:
        """压缩空白字符，保留单个空格。"""
        return re.sub(r'\s+', ' ', text).strip()
