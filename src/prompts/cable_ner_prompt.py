"""
NER标注Prompt模板

包含系统提示词、实体定义、标注示例等
"""

from typing import List, Dict

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的电力材料命名实体识别专家。你的任务是对电力材料描述文本进行实体标注。

## 实体类别定义

### 1. NAME（名称）
材料的基本名称，表明这是什么类型的电缆或线材。
- 包含：电力电缆、控制电缆、补偿电缆、CAN总线、通讯电缆、光纤、网线、照明电缆等
- 示例：
  - "电力电缆" → NAME
  - "12芯单模光纤" → NAME
  - "非屏蔽六类线" → NAME
  - "六类非屏蔽双绞线" → NAME
  - "防水网线" → NAME

### 2. MATERIAL（材质）
电缆的导体材质，通常是铜芯或铝芯，以及是否为本安型。
- 包含：铜芯、铝芯、铜芯本安型、铜芯本安电缆、铝芯本安型等
- 示例：
  - "铜芯" → MATERIAL
  - "铜芯本安型电缆" → MATERIAL
  - "铝芯交联聚乙烯电缆" → MATERIAL

### 3. TYPE（类型）
电缆的防火等级分类。
- 包含：阻燃、阻燃耐火、阻燃A类、阻燃B类、阻燃C类、CAN总线等
- 型号标识：ZA（阻燃A类）、ZB（阻燃B类）、ZC（阻燃C类）、NH（耐火）、WDZ（低烟无卤阻燃）
- 示例：
  - "阻燃" → TYPE
  - "阻燃A类" → TYPE
  - "ZA" → TYPE（阻燃A类）
  - "NH" → TYPE（耐火）

### 4. ARMOR（铠装）
电缆是否有铠装层及铠装类型。
- 铠装代码：22（钢带铠装）、32（细钢丝铠装）、42（粗钢丝铠装）、53（皱纹钢带）
- 示例：
  - "22" 在型号中出现 → ARMOR（钢带铠装）
  - "YJV22" 中的 "22" → ARMOR
  - "非铠装" → ARMOR

### 5. FEATURE（特征）
电缆的绝缘和护套材料等结构特征。
- 包含：交联聚乙烯绝缘聚氯乙烯护套、聚氯乙烯绝缘、铜线编织屏蔽等
- 型号标识：YJ（交联聚乙烯）、V（聚氯乙烯）、Y（聚乙烯）、P（屏蔽）
- 示例：
  - "YJV" → 包含FEATURE（交联聚乙烯绝缘聚氯乙烯护套）
  - "交联聚乙烯绝缘聚氯乙烯护套" → FEATURE
  - "铜线编织屏蔽" → FEATURE

### 6. VOLTAGE（额定电压）
电缆的额定工作电压。
- 格式：通常为 "X/YKV" 或 "XV/YV" 格式
- 示例：
  - "0.6/1KV" → VOLTAGE
  - "8.7/15KV" → VOLTAGE
  - "450V/750V" → VOLTAGE
  - "0.45/0.75KV" → VOLTAGE
  - "10KV" → VOLTAGE

### 7. SPEC（规格）
电缆的芯数和截面积规格。
- 格式：芯数×截面积，可能有多组，用+连接
- 注意：×、*、x、X 都表示乘号
- 示例：
  - "3×95" → SPEC
  - "3*6" → SPEC
  - "3x6" → SPEC
  - "3×95+2×50" → SPEC
  - "4芯单模" → SPEC
  - "12B1" → SPEC（12芯单模）
  - "10x2x1.5" → SPEC

## 标注规则

1. **完整提取**：提取完整的实体文本，不要截断
2. **位置准确**：start是实体第一个字符的索引（从0开始），end是最后一个字符索引+1
3. **不重叠**：同一文本片段只能属于一个实体类别
4. **型号解析**：对于型号（如ZA-YJV22-0.6/1KV），需要拆分识别各部分
5. **优先级**：当一个词可能属于多个类别时，选择最具体的类别

## 输出格式

请以JSON格式输出，严格按照以下结构：
```json
{
  "entities": [
    {"text": "实体文本", "label": "实体类型", "start": 起始位置, "end": 结束位置}
  ]
}
```

注意：
- label必须是以下之一：NAME, MATERIAL, TYPE, ARMOR, FEATURE, VOLTAGE, SPEC
- start和end必须是整数
- 如果没有识别到任何实体，返回空数组：{"entities": []}
"""

# Few-shot示例
FEW_SHOT_EXAMPLES = [
    {
        "input": "电力电缆ZA-YJV-0.6/1KV-3×95",
        "output": {
            "entities": [
                {"text": "电力电缆", "label": "NAME", "start": 0, "end": 4},
                {"text": "ZA", "label": "TYPE", "start": 4, "end": 6},
                {"text": "YJV", "label": "FEATURE", "start": 7, "end": 10},
                {"text": "0.6/1KV", "label": "VOLTAGE", "start": 11, "end": 18},
                {"text": "3×95", "label": "SPEC", "start": 19, "end": 23}
            ]
        }
    },
    {
        "input": "铜芯阻燃耐火电力电缆NH-YJV22-8.7/15KV-3×240",
        "output": {
            "entities": [
                {"text": "铜芯", "label": "MATERIAL", "start": 0, "end": 2},
                {"text": "阻燃耐火", "label": "TYPE", "start": 2, "end": 6},
                {"text": "电力电缆", "label": "NAME", "start": 6, "end": 10},
                {"text": "NH", "label": "TYPE", "start": 10, "end": 12},
                {"text": "YJV", "label": "FEATURE", "start": 13, "end": 16},
                {"text": "22", "label": "ARMOR", "start": 16, "end": 18},
                {"text": "8.7/15KV", "label": "VOLTAGE", "start": 19, "end": 27},
                {"text": "3×240", "label": "SPEC", "start": 28, "end": 33}
            ]
        }
    },
    {
        "input": "控制电缆KVV-450V/750V-10×2.5",
        "output": {
            "entities": [
                {"text": "控制电缆", "label": "NAME", "start": 0, "end": 4},
                {"text": "KVV", "label": "FEATURE", "start": 4, "end": 7},
                {"text": "450V/750V", "label": "VOLTAGE", "start": 8, "end": 17},
                {"text": "10×2.5", "label": "SPEC", "start": 18, "end": 24}
            ]
        }
    },
    {
        "input": "12芯单模光纤",
        "output": {
            "entities": [
                {"text": "12芯单模光纤", "label": "NAME", "start": 0, "end": 7}
            ]
        }
    },
    {
        "input": "铝芯交联聚乙烯绝缘聚氯乙烯护套电力电缆-0.6/1KV-4×25+1×16",
        "output": {
            "entities": [
                {"text": "铝芯", "label": "MATERIAL", "start": 0, "end": 2},
                {"text": "交联聚乙烯绝缘聚氯乙烯护套", "label": "FEATURE", "start": 2, "end": 15},
                {"text": "电力电缆", "label": "NAME", "start": 15, "end": 19},
                {"text": "0.6/1KV", "label": "VOLTAGE", "start": 20, "end": 27},
                {"text": "4×25+1×16", "label": "SPEC", "start": 28, "end": 37}
            ]
        }
    },
    {
        "input": "非屏蔽六类线",
        "output": {
            "entities": [
                {"text": "非屏蔽六类线", "label": "NAME", "start": 0, "end": 6}
            ]
        }
    },
    {
        "input": "阻燃C类铜芯本安电缆ZC-ia-YJV-0.5KV-2×2.5",
        "output": {
            "entities": [
                {"text": "阻燃C类", "label": "TYPE", "start": 0, "end": 4},
                {"text": "铜芯本安电缆", "label": "MATERIAL", "start": 4, "end": 10},
                {"text": "ZC", "label": "TYPE", "start": 10, "end": 12},
                {"text": "ia", "label": "FEATURE", "start": 13, "end": 15},
                {"text": "YJV", "label": "FEATURE", "start": 16, "end": 19},
                {"text": "0.5KV", "label": "VOLTAGE", "start": 20, "end": 25},
                {"text": "2×2.5", "label": "SPEC", "start": 26, "end": 31}
            ]
        }
    }
]


class NERPromptTemplate:
    """NER标注Prompt模板类"""
    
    def __init__(self, custom_examples: List[Dict] = None):
        """
        初始化Prompt模板
        
        Args:
            custom_examples: 自定义示例，可以添加更多领域特定的示例
        """
        self.system_prompt = SYSTEM_PROMPT
        self.examples = FEW_SHOT_EXAMPLES.copy()
        if custom_examples:
            self.examples.extend(custom_examples)
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.system_prompt
    
    def get_few_shot_prompt(self) -> str:
        """获取Few-shot示例部分"""
        examples_text = "\n## 标注示例\n\n"
        for i, example in enumerate(self.examples, 1):
            examples_text += f"### 示例{i}\n"
            examples_text += f"输入: \"{example['input']}\"\n"
            examples_text += f"输出:\n```json\n{self._format_json(example['output'])}\n```\n\n"
        return examples_text
    
    def get_full_system_prompt(self) -> str:
        """获取完整的系统提示词（包含示例）"""
        return self.system_prompt + "\n" + self.get_few_shot_prompt()
    
    def get_user_prompt(self, text: str) -> str:
        """
        获取用户输入的prompt
        
        Args:
            text: 待标注的文本
            
        Returns:
            格式化的用户prompt
        """
        return f"""请对以下电力材料描述进行NER标注：

"{text}"

重要：请直接输出JSON，不要输出任何思考过程或解释。只输出一个JSON对象，格式如下：
{{"entities": [{{"text": "...", "label": "...", "start": 0, "end": 0}}]}}

/no_think"""
    
    def _format_json(self, obj: dict) -> str:
        """格式化JSON输出"""
        import json
        return json.dumps(obj, ensure_ascii=False, indent=2)
    
    def add_example(self, input_text: str, output: Dict):
        """
        添加自定义示例
        
        Args:
            input_text: 输入文本
            output: 标注结果
        """
        self.examples.append({
            "input": input_text,
            "output": output
        })

