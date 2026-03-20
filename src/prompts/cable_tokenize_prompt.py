"""
分词Prompt模板

专门用于电力材料描述的分词任务
"""

from typing import List, Dict
import json

# 分词系统提示词
TOKENIZE_SYSTEM_PROMPT = """你是一个专业的中文分词专家，专门处理电力材料描述文本。

## 任务说明
对电力材料描述进行分词，将文本切分成有意义的词语。

## 分词规则

### 1. 保持完整性
- 材料名称保持完整：电力电缆、控制电缆、软电缆、计算机用软电缆
- 材质描述保持完整：铜芯、铝芯
- 特征描述保持完整：聚乙烯绝缘、聚氯乙烯护套、交联聚乙烯绝缘
- 功能描述保持完整：阻燃、耐火、编织分屏蔽、总屏蔽
- 型号保持完整：ZR-DJYPVPR、YJV22、NH-YJV
- 规格保持完整：2x2x1.5、3×95+2×50

### 2. 分词边界
- 中文逗号、顿号作为分隔符，单独成词
- 连接符"-"如果在型号中间则保持连接
- 乘号"×"或"x"如果在规格中则保持连接

### 3. 常见电力材料词汇
- 材质：铜芯、铝芯
- 绝缘：聚乙烯、交联聚乙烯、聚氯乙烯、橡胶
- 护套：聚氯乙烯护套、聚乙烯护套
- 特性：阻燃、耐火、低烟无卤、屏蔽、铠装
- 屏蔽：编织屏蔽、总屏蔽、分屏蔽、编织分屏蔽
- 名称：电力电缆、控制电缆、软电缆、计算机电缆、通讯电缆

## 输出格式

直接输出JSON数组，每个元素是一个分词结果：
```json
{"tokens": ["词1", "词2", "词3", ...]}
```

注意：
- 只输出JSON，不要输出其他内容
- tokens数组中的词语按原文顺序排列
- 所有词语连接起来应该等于原文
"""

# 分词示例
TOKENIZE_EXAMPLES = [
    {
        "input": "电力电缆ZA-YJV-0.6/1KV-3×95",
        "output": {"tokens": ["电力电缆", "ZA-YJV", "-", "0.6/1KV", "-", "3×95"]}
    },
    {
        "input": "铜芯聚乙烯绝缘阻燃聚氯乙烯护套编织分、总屏蔽计算机用软电缆ZR-DJYPVPR 2x2x1.5",
        "output": {"tokens": ["铜芯", "聚乙烯绝缘", "阻燃", "聚氯乙烯护套", "编织分", "、", "总屏蔽", "计算机用软电缆", "ZR-DJYPVPR", " ", "2x2x1.5"]}
    },
    {
        "input": "铝芯交联聚乙烯绝缘聚氯乙烯护套电力电缆-0.6/1KV-4×25+1×16",
        "output": {"tokens": ["铝芯", "交联聚乙烯绝缘", "聚氯乙烯护套", "电力电缆", "-", "0.6/1KV", "-", "4×25+1×16"]}
    },
    {
        "input": "阻燃耐火控制电缆NH-KVV-450V/750V-10×2.5",
        "output": {"tokens": ["阻燃耐火", "控制电缆", "NH-KVV", "-", "450V/750V", "-", "10×2.5"]}
    },
    {
        "input": "12芯单模光纤",
        "output": {"tokens": ["12芯单模光纤"]}
    }
]


class TokenizePromptTemplate:
    """分词Prompt模板类"""
    
    def __init__(self, custom_examples: List[Dict] = None):
        self.system_prompt = TOKENIZE_SYSTEM_PROMPT
        self.examples = TOKENIZE_EXAMPLES.copy()
        if custom_examples:
            self.examples.extend(custom_examples)
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.system_prompt
    
    def get_few_shot_prompt(self) -> str:
        """获取Few-shot示例部分"""
        examples_text = "\n## 分词示例\n\n"
        for i, example in enumerate(self.examples, 1):
            examples_text += f"### 示例{i}\n"
            examples_text += f"输入: \"{example['input']}\"\n"
            examples_text += f"输出: {json.dumps(example['output'], ensure_ascii=False)}\n\n"
        return examples_text
    
    def get_full_system_prompt(self) -> str:
        """获取完整的系统提示词（包含示例）"""
        return self.system_prompt + "\n" + self.get_few_shot_prompt()
    
    def get_user_prompt(self, text: str) -> str:
        """获取用户输入的prompt"""
        return f"""请对以下电力材料描述进行分词：

"{text}"

直接输出JSON格式的分词结果，不要输出任何其他内容。
/no_think"""

