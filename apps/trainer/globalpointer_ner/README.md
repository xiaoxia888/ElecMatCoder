# GlobalPointer NER

基于 GlobalPointer 架构的命名实体识别模型。

## 特点

相比 BIO+CRF 架构：

1. **更鲁棒的边界识别**：直接预测实体的起止位置，不会出现 `Pipe` → `Pi` 的截断问题
2. **支持嵌套实体**：天然支持嵌套实体识别
3. **推理速度相当**：与 BERT+CRF 推理速度相当

## 使用方法

### 1. 一键运行

```bash
bash apps/trainer/globalpointer_ner/run.sh
```

### 2. 分步运行

#### 数据转换

将 BIO 格式转换为 Span 格式：

```bash
python apps/trainer/globalpointer_ner/convert_data.py \
    --input data/pipe/raw/总数据_enhanced.jsonl \
    --output data/globalpointer/train.jsonl
```

#### 训练模型

```bash
python apps/trainer/globalpointer_ner/train.py \
    --encoder hfl/chinese-roberta-wwm-ext \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 2e-5
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder` | `hfl/chinese-roberta-wwm-ext` | 预训练编码器 |
| `--epochs` | 15 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--max_len` | 256 | 最大序列长度 |
| `--output_dir` | `outputs/globalpointer_ner` | 输出目录 |

#### 预测测试

```bash
# 命令行模式
python apps/trainer/globalpointer_ner/predict.py "90度弯头 DN50 S30408"

# 交互模式
python apps/trainer/globalpointer_ner/predict.py
```

## 数据格式

### 输入格式 (BIO)

```json
{
  "text": "90度弯头 DN50 S30408",
  "ner_labels": ["B-TYPE", "I-TYPE", "I-TYPE", "I-TYPE", "O", ...]
}
```

### 输出格式 (Span)

```json
{
  "text": "90度弯头 DN50 S30408",
  "entities": [
    {"start": 0, "end": 4, "type": "TYPE", "text": "90度弯头"},
    {"start": 5, "end": 9, "type": "SIZE", "text": "DN50"},
    {"start": 10, "end": 16, "type": "MATERIAL", "text": "S30408"}
  ]
}
```

## 模型架构

```
Input
  ↓
BERT/RoBERTa Encoder
  ↓
Linear: hidden_size → num_labels * head_size * 2
  ↓
Reshape → Query (start) + Key (end)
  ↓
RoPE (Rotary Position Embedding)
  ↓
Span Score Matrix: [batch, num_labels, seq_len, seq_len]
  ↓
Multi-label Classification Loss
```

## 集成到平台

训练完成后，可以在 `platform_config.yaml` 中配置使用 GlobalPointer 模型：

```yaml
ner:
  model_type: "globalpointer"  # bert / qwen / globalpointer
  globalpointer:
    model_path: "outputs/globalpointer_ner/best_model"
```
