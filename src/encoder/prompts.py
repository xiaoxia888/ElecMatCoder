# -*- coding: utf-8 -*-
"""
材料编码LLM提示词模板
"""

# ============================================================
# 分类归一化提示词
# ============================================================

CLASSIFICATION_SYSTEM_PROMPT = """你是一个材料分类专家，负责判断材料描述中的具体名称与标准大类的关系。

【核心原则】
1. **配件独立性**：如果原始名称是具体的配件（如：三通、弯通、四通、封头、变径等），绝对不能将其归类为通用的“桥架”或“线槽”。它们是独立的零件。
2. **严禁强行归类**：如果对待分类的名称不确定，或者它明显不属于提供的标准列表，必须将其标记为新类别（is_new: true）。
3. **行业常识**：利用你的专业知识判断别名。例如“等径三通”和“水平三通”如果列表里只有“等径三通”，且你认为它们不同，请标记为新类别。

回答要求：只返回JSON格式"""

CLASSIFICATION_PROMPT_TEMPLATE = """请将以下{entity_type}描述归入标准大类。

【完整材料描述】
{original_text}

【标准大类列表】
{categories}

【待分类的{entity_type}】
{value}

【任务】
判断 "{value}" 应该归入哪个标准大类。

【输出要求】
1. category: 必须是上面列表中的某一个，或者 null
2. confidence: 必须是 0-1 之间的数字
3. 如果能归类，is_new 为 false
4. 如果无法归类，is_new 为 true，并用 suggested_name 给出规范名称。**注意：不准改变词义，比如不准把“三通”改为“弯通”！**

【示例1】如果 "热镀锌" 可以归入 "镀锌"：
{{"category": "镀锌", "is_new": false, "reason": "属于同种材质", "confidence": 0.9}}

【示例2】如果 "碳化硅" 不在列表中：
{{"category": null, "is_new": true, "suggested_name": "碳化硅", "reason": "全新材质", "confidence": 0.8}}

请分析 "{value}" 并返回 JSON："""


# ============================================================
# 编码生成提示词
# ============================================================

CODE_GENERATION_SYSTEM_PROMPT = """你是一个材料编码专家，负责为新的材料类别生成唯一的字母编码。

【提取规则 - 必须严格遵守】
1. **一个汉字对应一个字母**：编码必须且只能由每个汉字的拼音首字母组成。
   - 严禁为了凑长度而添加无关字母。
   - 严禁自行脑补不存在的拼音。
   - 例如：铝合金 -> L(v) H(e) J(in) -> LHJ
   - 例如：水平三通 -> S(hui) P(ing) S(an) T(ong) -> SPST
2. **严禁混淆**：不要受到相似词的影响。例如“三通”首字母是“ST”，绝对不能提取为“WT”（弯通）。
3. **长度对齐**：汉字有多少个，编码就有多少位。如果是 2 个汉字，编码必须是 2 位。如果是 3 个汉字，必须是 3 位。

回答要求：只返回JSON格式"""

CODE_GENERATION_PROMPT_TEMPLATE = """请为新的{entity_type}大类生成编码。

【已有编码规则示例】
{existing_codes}

【已使用的编码】
{used_codes}

【新类别】
{category}

【输出格式】
{{
  "code": "生成的编码",
  "derivation": "推导过程，如：铝(L)+合(H)+金(J)=LHJ",
  "has_conflict": false
}}

如果生成的编码与已有编码冲突，请生成替代方案：
{{
  "code": "替代编码",
  "derivation": "推导过程",
  "has_conflict": true,
  "conflict_with": "冲突的原编码",
  "alternatives": ["备选1", "备选2"]
}}"""


# ============================================================
# 信息补全提示词
# ============================================================

INFO_COMPLETION_SYSTEM_PROMPT = """你是一个电气材料专家。你的任务是根据给出的"材料描述"全文，结合行业常识，推断出 NER 模型漏识别的属性。

【字段定义 - 务必区分清楚！！！】
- name（名称）：产品的名字，如 桥架、弯通、三通、四通、直通、线槽、槽盒 等
- material（材质）：制作材料，如 铝合金、不锈钢、镀锌、热镀锌、玻璃钢、PVC 等
- type（类型）：结构形式，**只有这几种**：梯式、梯级式、槽式、托盘式、槽盒式

【严重警告 - 不要混淆！】
1. "弯通"、"三通"、"四通"、"直通" 是 **name（名称）**，绝对不是 type！
2. "梯式"、"槽式"、"托盘式" 是 **type（类型）**，绝对不是 name！
3. type 只能是：梯式、梯级式、槽式、托盘式、槽盒式，没有其他的！

【推理原则】
1. 仔细阅读材料描述，结合行业常识进行推断。
2. 如果描述中没有明确的类型信息，type 应该放入 cannot_infer。
3. 不要把产品名称当成类型！"""

INFO_COMPLETION_PROMPT_TEMPLATE = """分析以下材料描述，推断缺失字段。

【材料描述（全文线索）】
{original_text}

【已识别实体】
{known_info}

【缺失字段】
{missing_fields}

【常见值参考】
{options}

【特别警告 - type 字段】
type（类型）只能是以下几种：梯式、梯级式、槽式、托盘式、槽盒式
"弯通"、"三通"、"四通"、"直通"、"水平弯通" 是 name，不是 type！
如果描述中没有提到"梯式/槽式/托盘式"等，type 必须放入 cannot_infer！

【推理准则】
1. 根据材料描述和专业知识进行推断。
2. 不要把产品名称当成结构类型！
3. reason 中说明推断依据。
4. confidence 给出 0-1 的置信度。

【输出格式】
- inferred 的 key 只能是: material, type, name 中的一个
- 如果无法确定某个字段，放入 cannot_infer

【输出JSON结构】
{{
  "inferred": {{
    "<字段名>": {{
      "value": "<推断的值>",
      "reason": "<推断依据>",
      "confidence": <0-1数值>,
      "is_new": <true/false，是否为新值>
    }}
  }},
  "cannot_infer": ["<无法推断的字段名>"]
}}"""


# ============================================================
# 规格解析提示词
# ============================================================

SPEC_PARSING_SYSTEM_PROMPT = """你是一个规格解析专家，专门负责从各种格式的规格描述中提取关键尺寸。

【核心原则 - 必须遵守】
1. 只能从原文中提取数值，绝对禁止编造或猜测任何数字
2. 如果原文中没有某个尺寸，该字段必须为null
3. 仔细识别每个数字对应的含义（宽W/高H/长L）

【常见格式示例】
- W×H×L：800×200×6000 → 宽800，高200，长6000
- WXHXL:300X200X6000 → 宽300，高200，长6000
- W600xH150L=6000 → 宽600，高150，长6000
- W600*H200 → 宽600，高200
- 50W×50H → 宽50，高50
- 400mm×150mm×2000mm → 宽400，高150，长2000
- W=600,H=200,L=6000 → 宽600，高200，长6000
- 600宽×200高 → 宽600，高200
- 600x200x6000 → 宽600，高200，长6000（按顺序：宽×高×长）
- φ89×4 → 外径89，壁厚4（管道）
- DN100 → 公称直径100（管道）

【识别技巧】
- W/w/宽 后面的数字是宽度
- H/h/高 后面的数字是高度
- L/l/长 后面的数字是长度
- 如果只有三个数字用×连接，按顺序是 宽×高×长
- 如果只有两个数字，通常是 宽×高"""

SPEC_PARSING_PROMPT_TEMPLATE = """请从以下规格描述中提取关键尺寸。

【材料类型】
{material_type}

【规格描述】
{spec_text}

【提取规则】
- 桥架类：提取 宽度(W) × 高度(H)，忽略长度(L)
- 管道类：提取 外径 × 壁厚，或 公称直径DN
- 电缆类：提取 截面积

【重要提醒】
1. 必须从原文中提取具体的数字，禁止编造
2. code_format 必须是具体的数字，如 "600X150"，绝对不能是 "W×H" 这种模板

【示例】
输入: "W600xH150L=6000"
输出: {{"width": 600, "height": 150, "code_format": "600X150"}}

输入: "WXHXL:300X200X6000"
输出: {{"width": 300, "height": 200, "code_format": "300X200"}}

输入: ":W600xH150L=6000"
输出: {{"width": 600, "height": 150, "code_format": "600X150"}}

【你的任务】
从 "{spec_text}" 中提取宽度和高度的具体数值。

【输出格式】
{{
  "width": 宽度的具体数字,
  "height": 高度的具体数字,
  "code_format": "宽度数字X高度数字"
}}"""


# ============================================================
# 辅助函数
# ============================================================

def build_classification_prompt(
    entity_type: str,
    value: str,
    categories: list,
    original_text: str = ""
) -> list:
    """
    构建分类归一化的消息列表
    
    Args:
        entity_type: 实体类型 (名称/材质/类型)
        value: 待分类的值
        categories: 标准大类列表
        original_text: 完整的原始材料描述（上下文）
        
    Returns:
        消息列表
    """
    type_name_map = {
        'name': '名称',
        'material': '材质',
        'type': '类型'
    }
    
    categories_text = "\n".join(f"- {c}" for c in categories)
    
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        entity_type=type_name_map.get(entity_type, entity_type),
        value=value,
        categories=categories_text,
        original_text=original_text or "(未提供)"
    )
    
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def build_code_generation_prompt(
    entity_type: str,
    category: str,
    existing_codes: dict,
    used_codes: list
) -> list:
    """
    构建编码生成的消息列表
    
    Args:
        entity_type: 实体类型
        category: 新类别名称
        existing_codes: 已有的编码映射
        used_codes: 已使用的编码列表
        
    Returns:
        消息列表
    """
    type_name_map = {
        'name': '名称',
        'material': '材质',
        'type': '类型'
    }
    
    existing_text = "\n".join(f"- {k} → {v}" for k, v in existing_codes.items())
    used_text = ", ".join(used_codes)
    
    prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        entity_type=type_name_map.get(entity_type, entity_type),
        category=category,
        existing_codes=existing_text,
        used_codes=used_text
    )
    
    return [
        {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def build_info_completion_prompt(
    known_info: dict,
    missing_fields: list,
    options: dict,
    original_text: str = ""
) -> list:
    """
    构建信息补全的消息列表
    
    Args:
        known_info: 已知信息
        missing_fields: 缺失字段列表
        options: 各字段的可选值
        original_text: 完整的原始材料描述（上下文）
        
    Returns:
        消息列表
    """
    known_text = "\n".join(f"- {k}: {v}" for k, v in known_info.items() if v)
    missing_text = ", ".join(missing_fields)
    options_text = "\n".join(
        f"- {k}: {', '.join(v)}" 
        for k, v in options.items()
    )
    
    prompt = INFO_COMPLETION_PROMPT_TEMPLATE.format(
        original_text=original_text or "(未提供)",
        known_info=known_text,
        missing_fields=missing_text,
        options=options_text
    )
    
    return [
        {"role": "system", "content": INFO_COMPLETION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def build_spec_parsing_prompt(
    material_type: str,
    spec_text: str
) -> list:
    """
    构建规格解析的消息列表
    
    Args:
        material_type: 材料类型 (桥架/管道/电缆)
        spec_text: 规格描述
        
    Returns:
        消息列表
    """
    prompt = SPEC_PARSING_PROMPT_TEMPLATE.format(
        material_type=material_type,
        spec_text=spec_text
    )
    
    return [
        {"role": "system", "content": SPEC_PARSING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

