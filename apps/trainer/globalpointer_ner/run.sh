#!/bin/bash
# GlobalPointer NER 训练运行脚本

set -e

cd "$(dirname "$0")/../../.."
echo "工作目录: $(pwd)"

# 1. 数据转换
echo ""
echo "=========================================="
echo "步骤 1: 数据转换 (BIO -> Span)"
echo "=========================================="
python apps/trainer/globalpointer_ner/convert_data.py \
    --input data/pipe/raw/总数据_enhanced.jsonl \
    --output data/globalpointer/train.jsonl \
    --val_split 0.1

# 2. 训练模型
echo ""
echo "=========================================="
echo "步骤 2: 训练 GlobalPointer 模型"
echo "=========================================="
python apps/trainer/globalpointer_ner/train.py \
    --encoder hfl/chinese-roberta-wwm-ext \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_len 256 \
    --output_dir outputs/globalpointer_ner

# 3. 测试预测
echo ""
echo "=========================================="
echo "步骤 3: 测试预测"
echo "=========================================="
python apps/trainer/globalpointer_ner/predict.py "90度弯头 DN50 S30408 GB/T12459"

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
