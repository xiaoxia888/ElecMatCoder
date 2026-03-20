"""
管道平台分词Prompt模板

专门用于管道材料描述的分词和实体识别任务
识别标签：种类(TYPE)、材质(MATERIAL)、尺寸(SIZE)、壁厚(THICKNESS)、磅级(PRESSURE)、规范(STANDARD)、密封面(SEAL)、端部(ENDS)、连接(CONN)、工艺(MANU)
"""

from typing import List, Dict
import json

# 管道平台分词系统提示词
PIPE_TOKENIZE_SYSTEM_PROMPT = """你是一个专业的管道材料识别专家。

## 任务说明
对管道材料描述进行分词，并识别其中的关键实体。

## 标签类型（11种）
- **TYPE (种类)**: 产品名称，必须作为一个整体识别。
  - 法兰类型：`FLANGE SO`(平焊法兰)、`FLANGE WN`(对焊法兰)、`FLANGE SW`(承插焊法兰)、`FLANGE BL`(法兰盖)、`FLANGE LJ`(松套法兰)、`FLANGE TH`(螺纹法兰) 等，必须整体识别为 TYPE。
  - 带括号的产品：`HALF COUPLING (WITH 45° BEVEL)`、`TEE (REDUCING)` 等，括号及其内容是产品规格说明，必须整体识别为 TYPE。
  - 其他：PIPE, 弯头, 三通, 阀门, ELBOW, REDUCER 等。
- **MATERIAL (材质)**: 材料牌号。如：S30408, 316L, 2205, 20#, Q235, L245, L360, X52, X65 等。注意：`PSL1`、`PSL2` 是产品规范等级，不是材质，标记为 O。
- **SIZE (尺寸)**: 尺寸规格。如：DN40, DN1200, φ108, 4"等。
- **THICKNESS (壁厚)**: 壁厚参数。如：S-40S, 10.31 mm, Sch40, STD等。
- **PRESSURE (磅级)**: 压力等级。如：PN16, CL150, CL3000, Class150, 3000LB等。
- **STANDARD (规范)**: 标准代码。如：GB/T 14976, NB/T 47010, ASME B16.9, ASTM A403 等。
- **STANDARD_GRADE (规范等级)**: 规范的类型/等级说明，与前面的 STANDARD 配套。如：Series I, Type S, TYPE V, STRENGTHENED GRADE 等。
- **SEAL (密封面形式)**: 密封面类型。如：FF(平面)、RTJ(环连接面)、RJ、FLRJ(平面环连接面)。
- **ENDS (端部形式)**: 端部连接形式。如：SO(平焊)、NPT、FNPT(内螺纹)、MNPT(外螺纹)、NPTF、FTE、MTE。注意：`PE`(平口端)、`BE`(坡口端) 是管端状态，标记为 O。
- **CONN (连接方式)**: 仅包含 SW(承插焊) 和 THD(螺纹)。
- **MANU (工艺)**: 制造工艺。如：SMLS, ERW, EFW, LSAW, SAWH, SAWL, Seamless, Forged, Cast, Welded, Machined, 焊接, welding, 锻制等。注意：`3PE`、`FBE`、`2PE` 等防腐涂层标记为 O。

## 标注核心原则

### 1. 原始状态保留（极其重要）
- **严禁修改原文**：必须严格按照输入的描述进行分词。**严禁删除、合并或增加任何空格、标点或字符**。
- **Token 完整性**：带空格的规范（如 `GB/T 14976`）或参数（如 `10.31 mm`）应作为一个完整的 token，其标签对应相应的实体类型。
- 如果原文中有多个空格或特殊的标点，请在输出的 `tokens` 中原样保留。

### 2. 实体识别与排除
- **管端状态排除**：`PE` (平口端)、`BE` (坡口端) 是管端状态，标记为 **O**。不要将它们设置为 `CONN` 标签。
- **连接标签定义**：`CONN` 仅指具体的连接手段（如对焊 BW、承插焊 SW、螺纹 THD 等）。
- **异径规格**：对于 `DN25xDN20`、`DN150 x DN25`、`DN100×20`、`φ33.7Xφ21.3` 这种异径/支管规格（两边都是尺寸），必须作为一个整体标注为 `SIZE`，严禁拆分。
- **尺寸×壁厚**：对于 `Φ45x2.5`、`φ108x6` 这种"外径×壁厚"格式，需要拆分：`Φ45` 是 SIZE，`x` 是 O，`2.5` 是 THICKNESS。区分方法：如果第二个数字明显较小（壁厚通常 <20mm），则拆分。
- **规范-材质连写**：如果规范和材质用 `-` 连接（如 `NB/T47010-S30408`），必须拆分识别：`NB/T47010` 是 STANDARD，`-` 是 O，`S30408` 是 MATERIAL。
- **规范等级合并**：如果规范和等级用括号连接（如 `GB/T19326(II)`、`HG/T20553(II)`），应整体标注为 `STANDARD`。
- **规范等级分离**：如果规范和等级用逗号分隔（如 `GB/T 12459, Series I`），应识别为两个独立实体：`GB/T 12459` 是 STANDARD，`Series I` 是 STANDARD_GRADE。
- **壁厚格式保留**：壁厚参数如 `T=3.5mm`、`T=4.5×4.0mm(S)` 应整体标注为 `THICKNESS`，保留原始格式。

### 3. 复杂格式处理
- **带序号的描述**：如 `1.材质:xxx2.规格:xxx3.焊接方法:xxx` 这种格式，序号和标签名（如 `1.材质:`、`2.规格:`）应标记为 **O**，只识别其中的实体内容。
- **安装/焊接说明**：如 `焊接方法:国标锻钢制承插焊`、`充氩保护方式、部位:满足设计要求` 等描述性内容，全部标记为 **O**。
- **重复出现的实体**：如果同一个实体在文本中重复出现（如 `DN100×20` 在规格中又出现一次），只需识别第一次出现的位置。

### 3. 其他内容
- 标点符号、空格（作为分隔符时）标记为 **O**。

## 输出格式
直接输出JSON对象，包含材料大类和分词结果：
```json
{
  "type_class": "管子/管件/法兰/螺栓/阀门/垫片 之一",
  "tokens": [
    {"word": "词1", "tag": "标签1"},
    {"word": "词2", "tag": "标签2"},
    ...
  ]
}
```
标签必须使用英文缩写：TYPE, MATERIAL, SIZE, THICKNESS, PRESSURE, STANDARD, STANDARD_GRADE, CONN, MANU, O
"""

# 管道平台分词示例
PIPE_TOKENIZE_EXAMPLES = [
    {
        "input": "PIPE, S30408 GB/T 14976, SMLS, PE, SH/T 3405,S-40S,DN40",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "PIPE", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 14976", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SMLS", "tag": "MANU"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "PE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3405", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": "S-40S", "tag": "THICKNESS"},
                {"word": ",", "tag": "O"},
                {"word": "DN40", "tag": "SIZE"}
            ]
        }
    },
    {
        "input": "PIPE, S30408 GB/T 12771 TYPE V, EFW, BE, SH/T 3405,10.31 mm,DN1200",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "PIPE", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 12771 TYPE V", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "EFW", "tag": "MANU"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "BE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3405", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": "10.31 mm", "tag": "THICKNESS"},
                {"word": ",", "tag": "O"},
                {"word": "DN1200", "tag": "SIZE"}
            ]
        }
    },
    {
        "input": "FLANGE SO, CL150, RF, S30408 NB/T 47010, SH/T 3406",
        "output": {
            "type_class": "法兰",
            "tokens": [
                {"word": "FLANGE SO", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "CL150", "tag": "PRESSURE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "RF", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "NB/T 47010", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3406", "tag": "STANDARD"}
            ]
        }
    },
    {
        "input": "HALF COUPLING (WITH 45° BEVEL), S30408 NB/T 47010, FORGED, SW, CL3000, SH/T 3410",
        "output": {
            "type_class": "管件",
            "tokens": [
                {"word": "HALF COUPLING (WITH 45° BEVEL)", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "NB/T 47010", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "FORGED", "tag": "MANU"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SW", "tag": "CONN"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "CL3000", "tag": "PRESSURE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3410", "tag": "STANDARD"}
            ]
        }
    },
    {
        "input": "PIPE, L245 PSL1 GB/T 9711, SAWL, BE, SH/T 3405",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "PIPE", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "L245", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "PSL1", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 9711", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SAWL", "tag": "MANU"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "BE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3405", "tag": "STANDARD"}
            ]
        }
    },
    {
        "input": "PIPE, Q235B SY/T 5037, SAWH, BE, SH/T 3405, 3PE GB/T 23257 STRENGTHENED GRADE",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "PIPE", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "Q235B", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "SY/T 5037", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SAWH", "tag": "MANU"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "BE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SH/T 3405", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "3PE", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 23257 STRENGTHENED GRADE", "tag": "STANDARD"}
            ]
        }
    },
    {
        "input": "对焊管接台, NB/T47010-S30408, BE, GB/T 19326 Series I , DN150 x DN25, S-40S x S-40S",
        "output": {
            "type_class": "管件",
            "tokens": [
                {"word": "对焊管接台", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
            {"word": "NB/T47010", "tag": "STANDARD"},
                {"word": "-", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "BE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 19326", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "Series I", "tag": "STANDARD_GRADE"},
                {"word": " ", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "DN150 x DN25", "tag": "SIZE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "S-40S x S-40S", "tag": "THICKNESS"}
            ]
        }
    },
    {
        "input": "90 度长半径弯头, SF304L GB/T13401, BE, GB/T 12459, Series I, SMLS , DN400, S-10S",
        "output": {
            "type_class": "管件",
            "tokens": [
                {"word": "90 度长半径弯头", "tag": "TYPE"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "SF304L", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T13401", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "BE", "tag": "O"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T 12459", "tag": "STANDARD"},
                {"word": ",", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "Series I", "tag": "STANDARD_GRADE"},
                {"word": ",", "tag": "O"},
            {"word": " ", "tag": "O"},
                {"word": "SMLS", "tag": "MANU"},
            {"word": " ", "tag": "O"},
                {"word": ",", "tag": "O"},
            {"word": " ", "tag": "O"},
                {"word": "DN400", "tag": "SIZE"},
                {"word": ",", "tag": "O"},
            {"word": " ", "tag": "O"},
                {"word": "S-10S", "tag": "THICKNESS"}
            ]
        }
    },
    {
        "input": "1.材质:不锈钢承插焊支管座06Cr19Ni10 NB/T47010;GB/T19326(II);SW;CL3000 DN100×202.规格:DN100X203.焊接方法:国标锻钢制承插焊4.充氩保护方式、部位:满足设计要求",
        "output": {
            "type_class": "管件",
            "tokens": [
                {"word": "1.材质:", "tag": "O"},
                {"word": "不锈钢承插焊支管座", "tag": "TYPE"},
                {"word": "06Cr19Ni10", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "NB/T47010", "tag": "STANDARD"},
                {"word": ";", "tag": "O"},
                {"word": "GB/T19326(II)", "tag": "STANDARD"},
                {"word": ";", "tag": "O"},
                {"word": "SW", "tag": "CONN"},
                {"word": ";", "tag": "O"},
                {"word": "CL3000", "tag": "PRESSURE"},
                {"word": " ", "tag": "O"},
                {"word": "DN100×20", "tag": "SIZE"},
                {"word": "2.规格:DN100X203.焊接方法:国标锻钢制承插焊4.充氩保护方式、部位:满足设计要求", "tag": "O"}
            ]
        }
    },
    {
        "input": "不锈钢无缝钢管安装1.名称:不锈钢无缝钢管2.材质:S304083.管径、壁厚:Φ32x2.54.用途:GB/T 14976-2012",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "不锈钢无缝钢管", "tag": "TYPE"},
                {"word": "安装1.名称:不锈钢无缝钢管2.材质:", "tag": "O"},
                {"word": "S30408", "tag": "MATERIAL"},
                {"word": "3.管径、壁厚:", "tag": "O"},
                {"word": "Φ32", "tag": "SIZE"},
                {"word": "x", "tag": "O"},
                {"word": "2.5", "tag": "THICKNESS"},
                {"word": "4.用途:", "tag": "O"},
                {"word": "GB/T 14976-2012", "tag": "STANDARD"}
            ]
        }
    },
    {
        "input": "1.材质:不锈钢无缝钢管（06Cr19Ni10 GB/T14976 SMLS HG/T20553(II) T=3.5mm）2.规格:DN50 Ф57×3.53.焊接方法:国标钢制对焊4.充氩保护方式、部位:满足设计要求",
        "output": {
            "type_class": "管子",
            "tokens": [
                {"word": "1.材质:", "tag": "O"},
                {"word": "不锈钢无缝钢管", "tag": "TYPE"},
                {"word": "（", "tag": "O"},
                {"word": "06Cr19Ni10", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T14976", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "SMLS", "tag": "MANU"},
                {"word": " ", "tag": "O"},
                {"word": "HG/T20553(II)", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "T=3.5mm", "tag": "THICKNESS"},
                {"word": "）2.规格:", "tag": "O"},
                {"word": "DN50", "tag": "SIZE"},
                {"word": " ", "tag": "O"},
                {"word": "Ф57", "tag": "SIZE"},
                {"word": "×", "tag": "O"},
                {"word": "3.5", "tag": "THICKNESS"},
                {"word": "3.焊接方法:国标钢制对焊4.充氩保护方式、部位:满足设计要求", "tag": "O"}
            ]
        }
    },
    {
        "input": "1.材质:不锈钢同心大小头(BW) 06Cr19Ni10 GB/T14976 BW GB/T12459(II系列) GB/T13401 T=4.5×4.0mm(S)2.规格:DN80X403.焊接方法:国标钢制对焊4.充氩保护方式、部位:满足设计要求",
        "output": {
            "type_class": "管件",
            "tokens": [
                {"word": "1.材质:", "tag": "O"},
                {"word": "不锈钢同心大小头", "tag": "TYPE"},
                {"word": "(", "tag": "O"},
                {"word": "BW", "tag": "CONN"},
                {"word": ")", "tag": "O"},
                {"word": " ", "tag": "O"},
                {"word": "06Cr19Ni10", "tag": "MATERIAL"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T14976", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "BW", "tag": "CONN"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T12459(II系列)", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "GB/T13401", "tag": "STANDARD"},
                {"word": " ", "tag": "O"},
                {"word": "T=4.5×4.0mm(S)", "tag": "THICKNESS"},
                {"word": "2.规格:", "tag": "O"},
                {"word": "DN80X40", "tag": "SIZE"},
                {"word": "3.焊接方法:国标钢制对焊4.充氩保护方式、部位:满足设计要求", "tag": "O"}
            ]
        }
    }
]

class PipeTokenizePromptTemplate:
    """管道平台分词Prompt模板类"""
    
    def __init__(self, custom_examples: List[Dict] = None):
        self.system_prompt = PIPE_TOKENIZE_SYSTEM_PROMPT
        self.examples = PIPE_TOKENIZE_EXAMPLES.copy()
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
        return f"""请对以下管道材料描述进行分词和实体识别：

"{text}"

直接输出JSON格式的分词结果，不要输出任何其他内容。
/no_think"""
