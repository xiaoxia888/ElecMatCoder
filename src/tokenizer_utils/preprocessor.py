"""
文本预处理模块
"""


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

