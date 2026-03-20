# 电力材料NER标注系统技术方案

> 基于 BERT + CRF 的电力材料命名实体识别系统

## 一、项目概述

### 1.1 目标

对电力材料描述文本（如 `电力电缆ZA-YJV-0.6/1kV-3x95`）进行命名实体识别（NER），自动识别出：

| 实体类别 | 代号 | 示例 |
|---------|------|------|
| 材料名称 | MAT | 电力电缆、控制电缆 |
| 阻燃等级 | FLAME | ZA、ZB、ZC |
| 绝缘/护套类型 | TYPE | YJV、VV、XLPE |
| 电压等级 | VOLT | 0.6/1kV、10kV |
| 芯数规格 | SPEC | 3x95、4x25+1x16 |
| 品牌/厂家 | BRAND | 远东、宝胜 |

### 1.2 技术路线

```
数据采集 → LLM预标注 → 人工校验 → 数据集生成 → BERT+CRF训练 → 模型部署
```

### 1.3 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           系统整体架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  数据采集    │───▶│  标注平台    │───▶│  数据集管理  │                  │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘                  │
│                            │                  │                         │
│                     ┌──────▼──────┐           │                         │
│                     │  LLM预标注   │           │                         │
│                     └─────────────┘           │                         │
│                                               │                         │
│  ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐                  │
│  │  API服务    │◀───│  模型部署    │◀───│  模型训练    │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、阶段一：标注平台开发

### 2.1 功能需求

- 数据导入与管理
- LLM自动预标注
- 人工标注与校验界面
- 标注质量控制
- 数据集导出

### 2.2 技术栈

| 层级 | 技术选型 | 版本 | 说明 |
|-----|---------|------|------|
| **前端框架** | Vue 3 | 3.4+ | 响应式UI框架 |
| **构建工具** | Vite | 5.0+ | 快速开发构建 |
| **UI组件库** | Element Plus | 2.5+ | 企业级组件库 |
| **CSS框架** | Tailwind CSS | 3.4+ | 原子化CSS |
| **后端框架** | FastAPI | 0.109+ | 高性能Python Web框架 |
| **ORM** | SQLAlchemy | 2.0+ | 数据库ORM |
| **数据库** | PostgreSQL | 15+ | 关系型数据库 |
| **缓存** | Redis | 7.0+ | 缓存LLM标注结果 |

### 2.3 核心模块

```
标注平台/
├── frontend/                 # 前端项目
│   ├── src/
│   │   ├── views/
│   │   │   ├── DataImport.vue       # 数据导入页
│   │   │   ├── Annotation.vue       # 标注界面
│   │   │   ├── Review.vue           # 校验页面
│   │   │   └── Export.vue           # 数据导出
│   │   ├── components/
│   │   │   ├── TextAnnotator.vue    # 文本标注组件
│   │   │   ├── EntityTag.vue        # 实体标签组件
│   │   │   └── AnnotationStats.vue  # 统计组件
│   │   └── stores/
│   │       └── annotation.ts        # 状态管理
│   └── package.json
│
└── backend/                  # 后端项目
    ├── app/
    │   ├── api/
    │   │   ├── data.py              # 数据管理API
    │   │   ├── annotation.py        # 标注API
    │   │   └── export.py            # 导出API
    │   ├── services/
    │   │   ├── llm_service.py       # LLM预标注服务
    │   │   └── annotation_service.py
    │   ├── models/
    │   │   └── schemas.py           # 数据模型
    │   └── core/
    │       └── config.py            # 配置管理
    └── requirements.txt
```

### 2.4 标注界面交互设计

```
┌────────────────────────────────────────────────────────────────────┐
│  📝 电力材料NER标注系统                              进度: 234/1000  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  原文:                                                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ 电力电缆ZA-YJV-0.6/1kV-3x95                                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  标注结果:                                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ [电力电缆]  [ZA]     [YJV]    [0.6/1kV]   [3x95]              │ │
│  │   MAT      FLAME    TYPE      VOLT        SPEC               │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  实体类别:                                                          │
│  [MAT] [FLAME] [TYPE] [VOLT] [SPEC] [BRAND] [O-删除标注]           │
│                                                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │ ◀ 上一条 │ │  跳过   │ │ ✓ 确认  │ │ 🔄 重置  │ │ 下一条 ▶ │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│                                                                    │
│  快捷键: 1-MAT 2-FLAME 3-TYPE 4-VOLT 5-SPEC 6-BRAND 0-删除        │
└────────────────────────────────────────────────────────────────────┘
```

---

## 三、阶段二：LLM预标注

### 3.1 技术选型

| LLM服务 | 优势 | API费用 | 推荐度 |
|--------|------|---------|-------|
| **通义千问** | 中文理解好、国内访问快 | 较低 | ⭐⭐⭐⭐⭐ |
| **GPT-4** | 效果最好 | 较高 | ⭐⭐⭐⭐ |
| **Claude** | 推理能力强 | 中等 | ⭐⭐⭐⭐ |
| **文心一言** | 国内服务稳定 | 较低 | ⭐⭐⭐ |

### 3.2 Prompt设计

```
## System Prompt

你是一个专业的电力材料命名实体识别专家。你的任务是对电力材料描述文本进行BIO标注。

### 实体类别定义

1. **MAT (材料名称)**: 材料的基本名称，如"电力电缆"、"控制电缆"、"绝缘导线"
2. **FLAME (阻燃等级)**: 阻燃/耐火等级标识，如"ZA"、"ZB"、"ZC"、"NH"
3. **TYPE (绝缘/护套类型)**: 绝缘和护套材料类型，如"YJV"、"VV"、"XLPE"、"YJY"
4. **VOLT (电压等级)**: 额定电压，如"0.6/1kV"、"10kV"、"35kV"
5. **SPEC (芯数规格)**: 芯数和截面积，如"3×95"、"4×25+1×16"、"3*185+1*95"
6. **BRAND (品牌/厂家)**: 品牌或生产厂家名称

### 标注规则

1. 使用BIO标注体系：B-开头，I-中间，O-非实体
2. 连字符"-"、"/"等分隔符标注为O
3. 如遇无法识别的部分，标注为O

### 输出格式

以JSON格式输出，包含entities数组：
{
  "text": "原始文本",
  "entities": [
    {"start": 起始位置, "end": 结束位置, "label": "实体类别", "text": "实体文本"}
  ]
}

## Few-shot Examples

输入: "电力电缆ZA-YJV-0.6/1kV-3x95"
输出:
{
  "text": "电力电缆ZA-YJV-0.6/1kV-3x95",
  "entities": [
    {"start": 0, "end": 4, "label": "MAT", "text": "电力电缆"},
    {"start": 4, "end": 6, "label": "FLAME", "text": "ZA"},
    {"start": 7, "end": 10, "label": "TYPE", "text": "YJV"},
    {"start": 11, "end": 18, "label": "VOLT", "text": "0.6/1kV"},
    {"start": 19, "end": 23, "label": "SPEC", "text": "3x95"}
  ]
}

输入: "远东牌阻燃耐火电缆NH-YJV22-10kV-3×240"
输出:
{
  "text": "远东牌阻燃耐火电缆NH-YJV22-10kV-3×240",
  "entities": [
    {"start": 0, "end": 2, "label": "BRAND", "text": "远东"},
    {"start": 3, "end": 9, "label": "MAT", "text": "阻燃耐火电缆"},
    {"start": 9, "end": 11, "label": "FLAME", "text": "NH"},
    {"start": 12, "end": 17, "label": "TYPE", "text": "YJV22"},
    {"start": 18, "end": 22, "label": "VOLT", "text": "10kV"},
    {"start": 23, "end": 28, "label": "SPEC", "text": "3×240"}
  ]
}
```

### 3.3 预标注服务实现

```python
# llm_service.py 核心逻辑

import openai
from typing import List, Dict

class LLMAnnotationService:
    """LLM预标注服务"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.system_prompt = """..."""  # 上述Prompt
    
    async def annotate(self, text: str) -> Dict:
        """对单条文本进行标注"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"请对以下文本进行NER标注：\n{text}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    async def batch_annotate(self, texts: List[str]) -> List[Dict]:
        """批量标注"""
        tasks = [self.annotate(text) for text in texts]
        return await asyncio.gather(*tasks)
```

---

## 四、阶段三：数据集生成

### 4.1 数据格式

#### BIO格式（训练用）

```
电 B-MAT
力 I-MAT
电 I-MAT
缆 I-MAT
Z B-FLAME
A I-FLAME
- O
Y B-TYPE
J I-TYPE
V I-TYPE
- O
0 B-VOLT
. I-VOLT
6 I-VOLT
/ I-VOLT
1 I-VOLT
k I-VOLT
V I-VOLT
- O
3 B-SPEC
x I-SPEC
9 I-SPEC
5 I-SPEC

```

#### JSON格式（备份/分析用）

```json
{
  "id": "sample_001",
  "text": "电力电缆ZA-YJV-0.6/1kV-3x95",
  "entities": [
    {"start": 0, "end": 4, "label": "MAT", "text": "电力电缆"},
    {"start": 4, "end": 6, "label": "FLAME", "text": "ZA"},
    {"start": 7, "end": 10, "label": "TYPE", "text": "YJV"},
    {"start": 11, "end": 18, "label": "VOLT", "text": "0.6/1kV"},
    {"start": 19, "end": 23, "label": "SPEC", "text": "3x95"}
  ],
  "annotator": "user_001",
  "create_time": "2024-01-15T10:30:00Z",
  "status": "verified"
}
```

### 4.2 数据集划分

| 数据集 | 比例 | 用途 |
|-------|------|------|
| 训练集 (train) | 80% | 模型训练 |
| 验证集 (dev) | 10% | 训练时验证、调参 |
| 测试集 (test) | 10% | 最终评估 |

### 4.3 数据质量控制

- **一致性检查**：检查同一文本不同标注者的一致性
- **完整性检查**：确保每个字符都有标签
- **合法性检查**：B标签后只能跟I或O，不能直接跟B

---

## 五、阶段四：模型训练

### 5.1 技术栈

| 组件 | 技术选型 | 版本 | 说明 |
|-----|---------|------|------|
| **深度学习框架** | PyTorch | 2.1+ | GPU加速训练 |
| **NLP框架** | Transformers | 4.36+ | HuggingFace生态 |
| **预训练模型** | bert-base-chinese | - | 中文BERT基座 |
| **序列标注层** | CRF | - | 条件随机场 |
| **实验管理** | Weights & Biases | - | 训练监控 |
| **GPU** | NVIDIA | 8GB+ | RTX 3060及以上 |

### 5.2 模型架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT + CRF 模型架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入层                                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ [CLS] 电 力 电 缆 Z A - Y J V - 0 . 6 / 1 k V [SEP]       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  Embedding层                                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Token Embedding + Position Embedding + Segment Embedding │ │
│  │  维度: 768                                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  BERT Encoder (12层 Transformer)                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Multi-Head Self-Attention (12 heads)               │ │ │
│  │  │  维度: 768, heads: 12, head_dim: 64                 │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                         │                                 │ │
│  │                         ▼                                 │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │  Feed Forward Network                               │ │ │
│  │  │  768 → 3072 → 768                                   │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │                    × 12 layers                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  输出层                                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Linear: 768 → num_labels (13个标签)                       │ │
│  │  标签: O, B-MAT, I-MAT, B-FLAME, I-FLAME, B-TYPE, ...     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  CRF层                                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Conditional Random Field                                 │ │
│  │  学习标签转移概率，确保标签序列合法性                         │ │
│  │  如：B-MAT 后只能接 I-MAT 或 O，不能接 B-MAT               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  输出: B-MAT I-MAT I-MAT I-MAT B-FLAME I-FLAME O B-TYPE ...    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 标签体系

```python
# 13个标签
LABELS = [
    "O",           # 非实体
    "B-MAT",       # 材料名称-开始
    "I-MAT",       # 材料名称-中间
    "B-FLAME",     # 阻燃等级-开始
    "I-FLAME",     # 阻燃等级-中间
    "B-TYPE",      # 绝缘类型-开始
    "I-TYPE",      # 绝缘类型-中间
    "B-VOLT",      # 电压等级-开始
    "I-VOLT",      # 电压等级-中间
    "B-SPEC",      # 规格参数-开始
    "I-SPEC",      # 规格参数-中间
    "B-BRAND",     # 品牌-开始
    "I-BRAND",     # 品牌-中间
]
```

### 5.4 训练配置

```python
# 训练超参数
training_args = {
    "model_name": "bert-base-chinese",
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "crf_learning_rate": 1e-3,  # CRF层使用更大学习率
    "gradient_accumulation_steps": 2,
    "fp16": True,  # 混合精度训练
}
```

### 5.5 项目结构

```
training/
├── configs/
│   └── config.yaml              # 训练配置
├── data/
│   ├── train.txt                # 训练集 (BIO格式)
│   ├── dev.txt                  # 验证集
│   └── test.txt                 # 测试集
├── src/
│   ├── models/
│   │   ├── bert_crf.py          # BERT+CRF模型定义
│   │   └── crf.py               # CRF层实现
│   ├── data/
│   │   ├── dataset.py           # 数据集类
│   │   └── preprocessor.py      # 数据预处理
│   ├── trainer.py               # 训练器
│   ├── evaluator.py             # 评估器
│   └── utils.py                 # 工具函数
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── predict.py               # 预测脚本
├── outputs/
│   ├── checkpoints/             # 模型检查点
│   └── logs/                    # 训练日志
└── requirements.txt
```

### 5.6 评估指标

| 指标 | 说明 | 目标值 |
|-----|------|-------|
| **Precision** | 预测为实体中正确的比例 | > 90% |
| **Recall** | 真实实体被正确预测的比例 | > 90% |
| **F1-Score** | P和R的调和平均 | > 90% |
| **Entity-level F1** | 实体级别的F1（完全匹配） | > 85% |

---

## 六、阶段五：模型部署

### 6.1 技术栈

| 组件 | 技术选型 | 说明 |
|-----|---------|------|
| **模型格式** | ONNX | 跨平台、推理优化 |
| **推理框架** | ONNX Runtime | 高性能推理 |
| **API框架** | FastAPI | RESTful API |
| **容器化** | Docker | 环境一致性 |
| **负载均衡** | Nginx | 反向代理 |

### 6.2 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      部署架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   客户端请求                                                  │
│       │                                                     │
│       ▼                                                     │
│   ┌───────────┐                                             │
│   │   Nginx   │  ← 负载均衡、SSL终止                         │
│   └─────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│   ┌───────────────────────────────────────┐                │
│   │           FastAPI 服务集群             │                │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ │                │
│   │  │ Worker1 │ │ Worker2 │ │ Worker3 │ │                │
│   │  └────┬────┘ └────┬────┘ └────┬────┘ │                │
│   │       │           │           │      │                │
│   │       └───────────┼───────────┘      │                │
│   │                   ▼                  │                │
│   │           ┌─────────────┐            │                │
│   │           │ ONNX Runtime │            │                │
│   │           │   (GPU/CPU)  │            │                │
│   │           └─────────────┘            │                │
│   └───────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 API接口设计

```python
# POST /api/v1/ner/predict
# 请求
{
    "text": "电力电缆ZA-YJV-0.6/1kV-3x95"
}

# 响应
{
    "code": 0,
    "message": "success",
    "data": {
        "text": "电力电缆ZA-YJV-0.6/1kV-3x95",
        "entities": [
            {"start": 0, "end": 4, "label": "MAT", "text": "电力电缆"},
            {"start": 4, "end": 6, "label": "FLAME", "text": "ZA"},
            {"start": 7, "end": 10, "label": "TYPE", "text": "YJV"},
            {"start": 11, "end": 18, "label": "VOLT", "text": "0.6/1kV"},
            {"start": 19, "end": 23, "label": "SPEC", "text": "3x95"}
        ]
    },
    "latency_ms": 15
}
```

### 6.4 性能指标

| 指标 | 目标值 |
|-----|-------|
| 单条推理延迟 | < 50ms |
| QPS (单实例) | > 100 |
| GPU显存占用 | < 2GB |
| 模型大小 (ONNX) | < 400MB |

---

## 七、技术栈汇总

### 7.1 全栈技术一览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          技术栈全景图                                     │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│    前端       │    后端       │   AI/ML      │    数据      │    部署     │
├──────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ Vue 3        │ FastAPI      │ PyTorch      │ PostgreSQL   │ Docker      │
│ Vite         │ Pydantic     │ Transformers │ Redis        │ Nginx       │
│ Element Plus │ SQLAlchemy   │ BERT         │ JSON/BIO     │ ONNX Runtime│
│ Tailwind CSS │ Uvicorn      │ CRF          │              │ Gunicorn    │
│ TypeScript   │ Python 3.10+ │ ONNX         │              │             │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
```

### 7.2 各阶段技术依赖

| 阶段 | 核心技术 | Python依赖 | 其他依赖 |
|-----|---------|-----------|---------|
| **标注平台** | Vue3 + FastAPI | fastapi, sqlalchemy, pydantic | Node.js, PostgreSQL |
| **LLM预标注** | LLM API | openai, httpx | LLM API密钥 |
| **数据处理** | Python | pandas, numpy | - |
| **模型训练** | BERT + CRF | torch, transformers, pytorch-crf | GPU (CUDA) |
| **模型部署** | ONNX + FastAPI | onnxruntime, fastapi | Docker |

---

## 八、时间规划

| 阶段 | 任务 | 预计时间 | 产出 |
|-----|------|---------|------|
| 1 | 标注平台开发 | 2-3周 | 可用的标注系统 |
| 2 | LLM预标注集成 | 1周 | 自动预标注功能 |
| 3 | 数据标注 | 2-4周 | 3000-5000条标注数据 |
| 4 | 模型训练 | 1周 | 训练好的BERT+CRF模型 |
| 5 | 模型部署 | 1周 | 可用的API服务 |
| **总计** | | **7-10周** | |

---

## 九、参考资源

### 9.1 论文

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) (BiLSTM-CRF)

### 9.2 开源项目

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- [bert-base-chinese](https://huggingface.co/bert-base-chinese)

### 9.3 数据标注工具

- [doccano](https://github.com/doccano/doccano)
- [Label Studio](https://labelstud.io/)

---

## 附录：标签转移矩阵示例

CRF层学习的标签转移概率矩阵（示意）：

|  | O | B-MAT | I-MAT | B-FLAME | I-FLAME | ... |
|--|---|-------|-------|---------|---------|-----|
| **O** | ✓ | ✓ | ✗ | ✓ | ✗ | ... |
| **B-MAT** | ✓ | ✗ | ✓ | ✓ | ✗ | ... |
| **I-MAT** | ✓ | ✗ | ✓ | ✓ | ✗ | ... |
| **B-FLAME** | ✓ | ✓ | ✗ | ✗ | ✓ | ... |
| **I-FLAME** | ✓ | ✓ | ✗ | ✗ | ✓ | ... |

- ✓ 表示允许转移
- ✗ 表示不允许转移（如 O 后不能直接跟 I-XXX）

