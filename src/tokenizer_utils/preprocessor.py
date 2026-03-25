"""
文本预处理模块
"""

import re


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

        # 5. 保守分隔符标准化（只处理安全分隔符）
        text = self.normalize_safe_separators(text)

        # 6. 空白压缩
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
        return text.replace('×', 'X').replace('*', 'X')

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

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """压缩空白字符，保留单个空格。"""
        return re.sub(r'\s+', ' ', text).strip()
