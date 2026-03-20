# 一阶段 NER 数据集标注规范

本文档用于统一当前一阶段抽取数据集的标注口径，适用于 `data/pipe/llm_lora/ner_data_new_schema.json`。

目标：

- 明确当前一阶段训练数据的字段 schema
- 统一各字段的取值格式和边界
- 明确 `STANDARD` 及其修饰项的绑定规则
- 说明当前新 schema 与旧版 BIO / 旧字段体系的关系

本文档描述的是**当前生效的一阶段 LLM 抽取标注规范**。若与旧文档存在冲突，以本文档为准。

## 1. 适用范围

适用对象：

- 管道材料一阶段抽取数据
- Qwen3 一阶段结构化抽取训练数据
- `input -> output JSON` 形式的 NER 样本

不直接适用对象：

- 旧版逐字 BIO 标注文件
- 旧版单独输出 `CONN` / `MANU` / `ENDS` / `SEAL` 的标注体系

## 2. 样本格式

每条样本由两部分组成：

- `input`：原始物料描述文本
- `output`：结构化抽取结果

示例：

```json
{
  "input": "异径三通, 15CrMoG, SW, 300LB, GB/T19326 SERIAL I , DN25 x DN15",
  "output": {
    "TYPE": "异径三通;SW",
    "SIZE": {
      "DN": ["DN25", "DN15"]
    },
    "PRESSURE": "300LB",
    "MATERIAL": {
      "RELATION": "single",
      "ITEMS": [
        {
          "MATERIAL_GRADE_CODE": "15CrMoG",
          "SPECIAL_REQ": []
        }
      ]
    },
    "STANDARD": "GB/T19326",
    "STANDARD_GRADE": [
      {"value": "SERIAL I", "bind_to_index": 0}
    ]
  }
}
```

## 3. 总体原则

### 3.1 只输出识别到的字段

未识别到的字段不输出，不写空字符串，不写空数组，不写 `null`。

### 3.2 忠实于原文，允许轻量规范化

标注层以忠实表达原文为主，但允许做少量、稳定、可复现的轻量规范化，目的是让训练数据结构更稳定、后续编码更容易。

允许：

- 去除无意义空格
- 拆开明显粘连的边界
- 尺寸和壁厚按固定对象 schema 归类
- 同一字段内按统一格式表达

不允许：

- 跨语言翻译
- 凭经验补充原文中不存在的信息
- 为了“好看”而过度改写

### 3.3 保留原文顺序

多值字段默认按原文出现顺序保留。

适用字段：

- `SIZE`
- `THICKNESS`
- `STANDARD`
- `STANDARD_*`

### 3.4 当前新 schema 不单独输出旧字段

当前一阶段新 schema 中，不再单独输出以下旧字段：

- `CONN`
- `MANU`
- `ENDS`
- `SEAL`

如果这些信息在当前样本中对“产品是什么”有帮助，允许并入 `TYPE`。当前数据集中常见写法是用分号连接，例如：

- `TYPE = "PIPE;SAWL;SW"`
- `TYPE = "螺纹法兰;RTJ"`
- `TYPE = "短管;THD"`

## 4. 当前字段清单

当前一阶段允许出现的字段如下：

- `TYPE`
- `SIZE`
- `THICKNESS`
- `PRESSURE`
- `MATERIAL`
- `STANDARD`
- `STANDARD_GRADE`
- `STANDARD_APPENDIX`
- `STANDARD_METHOD`

字段含义简表：

| 字段 | 含义 | 典型数据形态 |
|------|------|--------------|
| `TYPE` | 产品种类及必要的类型修饰 | 字符串 |
| `SIZE` | 尺寸信息 | 对象 |
| `THICKNESS` | 壁厚信息 | 对象 |
| `PRESSURE` | 压力等级 | 字符串 |
| `MATERIAL` | 材质 | 对象 |
| `STANDARD` | 规范主体 | 字符串或字符串数组 |
| `STANDARD_GRADE` | 规范等级 | 对象数组 |
| `STANDARD_APPENDIX` | 规范附录 | 对象数组 |
| `STANDARD_METHOD` | 规范方法 | 对象数组 |

## 5. 各字段标注要求

### 5.1 `TYPE`

定义：

- 表示“这是什么物料”的核心产品名
- 可包含必要的产品子类型、结构修饰、连接/工艺等补充信息
- 当前新 schema 下，旧字段中的部分信息允许并入 `TYPE`

建议保留内容：

- 主类名：`PIPE`、`REDUCER`、`ELBOW`、`TEE`、`法兰`、`弯头`、`管帽`
- 产品级修饰：`异径`、`偏心`、`90度`
- 对产品识别有帮助的旧字段信息：`SW`、`THD`、`SAWL`、`RTJ`

当前数据集常见写法：

- 用分号连接多个片段
- 片段顺序基本按原文或产品理解顺序保留

示例：

- `PIPE;NPTF`
- `PIPE;ERW;MNPT`
- `螺纹法兰;RTJ`
- `WELDOLET;FNPT`

注意：

- `TYPE` 不是编码结果，不要求归一成最终标准名
- 但也不是自由文本，应该尽量保持稳定、简洁、可复用

### 5.2 `SIZE`

定义：

从原文中提取所有与尺寸有关、能用于后续编码的信息，按类别写入对象中。

固定结构：

```json
"SIZE": {
  "DN": [],
  "OD": [],
  "INCH": []
}
```

只输出非空 key。

各子类含义：

- `DN`：公称直径 / 工程直径
- `OD`：外径
- `INCH`：英寸尺寸

统一格式：

- `DN` 统一写成 `DN+数字`
- `OD` 统一只保留数字本体
- `INCH` 统一写成 `数字+"`

示例：

- `DN25 x DN15` -> `"DN": ["DN25", "DN15"]`
- `φ159X5mm;NPS6` -> `"OD": ["159"]`, `"INCH": ["6\""]`
- `DN80xDN50` -> `"DN": ["DN80", "DN50"]`

### 5.3 `THICKNESS`

定义：

从原文中提取所有与壁厚有关的信息，按类别写入对象中。

固定结构：

```json
"THICKNESS": {
  "MM": [],
  "INCH": [],
  "SCHEDULE": [],
  "SERIES": [],
  "BWG": []
}
```

只输出非空 key。

各子类含义：

- `MM`：毫米壁厚
- `INCH`：英寸壁厚
- `SCHEDULE`：表号壁厚
- `SERIES`：系列壁厚，如 `STD`、`XS`
- `BWG`：线规

统一格式：

- `MM` 只保留数字本体
- `INCH` 只保留数字本体
- `SCHEDULE` 统一保留 `SCH` 前缀
- `SERIES` 保留系列本体
- `BWG` 只保留规号数字

示例：

- `SCH 10S` -> `"SCHEDULE": ["SCH10S"]`
- `T=4x3.5mm` -> `"MM": ["4", "3.5"]`
- `STD` -> `"SERIES": ["STD"]`

### 5.4 `PRESSURE`

定义：

压力等级字段，保留为单字符串，不做对象化。

典型值：

- `CL150`
- `CLASS900`
- `300LB`
- `6000LB`

要求：

- 尽量保留稳定、可编码的压力表达
- 当前一阶段以单值为主
- 不在标注层提前编码成目标码

### 5.5 `MATERIAL`

定义：

材质采用结构化对象标注，用于表达材质关系、执行标准、材质牌号表达和特殊要求。

典型值：

- `ASTM A403 WP316L`
- `S31603`
- `15CrMoG`
- `A105-ANTI-H2S`
- `A312 GR.TP316L`

统一结构：

```json
"MATERIAL": {
  "RELATION": "single",
  "ITEMS": [
    {
      "EXEC_STANDARD": "ASTM A403",
      "MATERIAL_GRADE_CODE": "WP316L",
      "SPECIAL_REQ": []
    }
  ]
}
```

字段说明：

- `RELATION`：当前只用 `single` 或 `alternative`
- `ITEMS`：材质项列表
- `EXEC_STANDARD`：材质执行标准主体，如 `ASTM A403`、`A182`、`API5L`
- `MATERIAL_GRADE_CODE`：完整材质牌号表达，按原文表面形式保留
- `SPECIAL_REQ`：特殊要求列表，按原文表面形式保留

标注要求：

- 优先保留原文表面形式，不在一阶段提前做编码归一
- `MATERIAL_GRADE_CODE` 不再拆分独立的 `LEVEL` 字段
- `Gr.6`、`Grade 70`、`Gr.F304`、`316L_Gr.II` 这类依附型等级表达，整体保留在 `MATERIAL_GRADE_CODE`
- `EXEC_STANDARD` 只保留真正的材质标准号，不写空字符串；为空时该键可省略
- `SPECIAL_REQ` 保留原文写法，如 `NACE`、`CE`、`GALVANIZED`、`GALV.`、`Zn#`、`Glav.`
- `CL32`、`Class 1` 这类当前不影响材质编码的 class 信息，不单独提取
- 括号中的别名或补充牌号，如果与主材质构成一个整体表达，可整体并入 `MATERIAL_GRADE_CODE`
- 复选材质用 `RELATION = "alternative"`；明显不完整或无法稳定拆分的样本进入人工复核

示例：

```json
"MATERIAL": {
  "RELATION": "single",
  "ITEMS": [
    {
      "EXEC_STANDARD": "ASTM A516",
      "MATERIAL_GRADE_CODE": "Grade 70",
      "SPECIAL_REQ": []
    }
  ]
}
```

```json
"MATERIAL": {
  "RELATION": "single",
  "ITEMS": [
    {
      "MATERIAL_GRADE_CODE": "316L_Gr.II",
      "SPECIAL_REQ": []
    }
  ]
}
```

```json
"MATERIAL": {
  "RELATION": "single",
  "ITEMS": [
    {
      "EXEC_STANDARD": "A53",
      "MATERIAL_GRADE_CODE": "GR.B",
      "SPECIAL_REQ": ["GALVANIZED"]
    }
  ]
}
```

要求：

- 以原文中的材质主表达为准
- 不在标注层做编码归一
- 不把 `STANDARD` 中的等级、方法错误并入材质

### 5.6 `STANDARD`

定义：

规范主体，不包含其等级、附录、方法等修饰项。

单规范时可写字符串，多规范时写数组。

示例：

```json
"STANDARD": "ASME B16.9"
```

或

```json
"STANDARD": ["GB/T5310", "HG20553"]
```

要求：

- `STANDARD` 只保留规范主体
- 不把 `Series I`、`附录B`、`方法E` 混在 `STANDARD` 中
- 多规范时按原文顺序排列

### 5.7 `STANDARD_GRADE`

定义：

规范等级或等级性修饰信息。

统一格式：

```json
"STANDARD_GRADE": [
  {"value": "Series I", "bind_to_index": 0}
]
```

说明：

- `value` 为等级内容
- `bind_to_index` 表示绑定到 `STANDARD` 中的哪一项

常见值：

- `Series I`
- `SERIAL I`
- `Sr B`
- `Ia`
- `I类填充金属`

注意：

- 当前数据集中保留了一部分接近原文的大小写和词面形式
- 若需统一，应优先遵循“轻量规范化、不过度翻译”的原则

### 5.8 `STANDARD_APPENDIX`

定义：

规范附录信息。

统一格式：

```json
"STANDARD_APPENDIX": [
  {"value": "附录B", "bind_to_index": 0}
]
```

常见值：

- `附录A`
- `附录B`
- `附录C`

### 5.9 `STANDARD_METHOD`

定义：

规范方法信息。

统一格式：

```json
"STANDARD_METHOD": [
  {"value": "方法E", "bind_to_index": 0}
]
```

或

```json
"STANDARD_METHOD": [
  {"value": "Method E", "bind_to_index": 0}
]
```

要求：

- 原文是中文就保留中文
- 原文是英文就保留英文
- 不做中英互转

## 6. `STANDARD_*` 绑定规则

### 6.1 为什么需要绑定

当一条样本中出现多个规范时，等级、方法、附录必须说明属于哪一个规范主体。

### 6.2 绑定方式

统一使用对象数组，每个对象至少包含：

- `value`
- `bind_to_index`

其中：

- `bind_to_index = 0` 表示绑定到 `STANDARD[0]`
- `bind_to_index = 1` 表示绑定到 `STANDARD[1]`
- 以此类推

示例：

```json
{
  "STANDARD": ["SH/T3410", "GB/T19326", "NB/T47010"],
  "STANDARD_GRADE": [
    {"value": "Series I", "bind_to_index": 1}
  ],
  "STANDARD_METHOD": [
    {"value": "方法E", "bind_to_index": 2}
  ]
}
```

### 6.3 无法明确绑定时

如果人工也无法判断某个修饰项属于哪个规范：

- 不要强行绑定
- 该样本应进入人工复核

## 7. 推荐输出 schema

推荐的完整 schema 如下：

```json
{
  "TYPE": "异径三通;SW",
  "SIZE": {
    "DN": ["DN25", "DN15"]
  },
  "THICKNESS": {
    "SCHEDULE": ["SCH40S"]
  },
  "PRESSURE": "300LB",
  "MATERIAL": {
    "RELATION": "single",
    "ITEMS": [
      {
        "MATERIAL_GRADE_CODE": "15CrMoG",
        "SPECIAL_REQ": []
      }
    ]
  },
  "STANDARD": ["GB/T19326", "NB/T47010"],
  "STANDARD_GRADE": [
    {"value": "Series I", "bind_to_index": 0}
  ],
  "STANDARD_METHOD": [
    {"value": "方法E", "bind_to_index": 1}
  ]
}
```

说明：

- 单值字段可直接写字符串
- 多值字段使用数组
- `SIZE` / `THICKNESS` 使用对象结构
- `STANDARD_*` 使用对象数组，并显式绑定

## 8. 当前数据集快照

基于当前 `data/pipe/llm_lora/ner_data_new_schema.json` 的统计结果：

- 总样本数：`5851`
- `TYPE`：`5851`
- `MATERIAL`：`5850`
- `SIZE`：`5800`
- `STANDARD`：`5336`
- `THICKNESS`：`3764`
- `PRESSURE`：`2038`
- `STANDARD_GRADE`：`1189`
- `STANDARD_METHOD`：`264`
- `STANDARD_APPENDIX`：`22`

补充说明：

- `STANDARD` 单值样本较多，但也明确存在多标准样本
- `SIZE` 以 `DN` 为主，也有 `OD` 和 `INCH`
- `THICKNESS` 以 `SCHEDULE` 和 `MM` 为主

## 9. 与旧体系的关系

### 9.1 与旧版 BIO 标注的关系

旧版 BERT / GlobalPointer 训练链路主要面向：

- `TYPE`
- `MATERIAL`
- `SIZE`
- `THICKNESS`
- `PRESSURE`
- `STANDARD`
- 以及旧字段 `CONN` / `MANU`

而当前一阶段新 schema 已升级为：

- 结构化 JSON 输出
- `SIZE` / `THICKNESS` 对象化
- `STANDARD` 修饰项拆分
- 旧字段不再单独输出

因此当前主线标注规范更适合 LLM 抽取模型，而不是直接等价于旧版 BIO 标签体系。

### 9.2 与旧文档的关系

当前 `docs/` 下已有若干分字段文档，例如：

- `docs/NER数据集TYPE标注规范.md`
- `docs/NER数据集SIZE标注规范.md`
- `docs/NER数据集THICKNESS标注规范.md`
- `docs/规范字段标注要求.md`

这些文档仍可作为字段细节参考，但若与本文有冲突，以本文档为准，因为本文档描述的是当前 `ner_data_new_schema.json` 实际采用的一阶段 schema。

## 10. 标注执行建议

推荐执行顺序：

1. 先识别产品主类，确定 `TYPE`
2. 再抽取 `SIZE`、`THICKNESS`、`PRESSURE`、`MATERIAL`
3. 识别所有 `STANDARD`
4. 再抽取 `STANDARD_GRADE`、`STANDARD_APPENDIX`、`STANDARD_METHOD`
5. 对每个 `STANDARD_*` 写明 `bind_to_index`
6. 无法确定归属时进入人工复核

## 11. 常见错误

以下情况应避免：

- 把 `Series I`、`附录B`、`方法E` 仍留在 `STANDARD` 主体里
- 多个 `STANDARD` 存在时，不写 `bind_to_index`
- 把中文 `方法E` 改写成英文 `Method E`
- 把 `SIZE`、`THICKNESS` 重新写回单字符串
- 沿用旧字段体系继续单独输出 `CONN` / `MANU` / `ENDS` / `SEAL`

## 12. 结论

当前一阶段标注规范的核心特点是：

- 以结构化 JSON 为目标输出
- 以 `TYPE`、`SIZE`、`THICKNESS`、`PRESSURE`、`MATERIAL`、`STANDARD` 为主字段
- `STANDARD` 做主体与修饰项拆分
- 使用 `bind_to_index` 建立多标准场景下的归属关系
- 不再把旧字段作为独立输出字段

后续如果继续扩充或回修一阶段数据，建议统一按本文档执行。
