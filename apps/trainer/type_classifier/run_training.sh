#!/bin/bash
# TYPE 分类模型训练脚本

# 设置路径
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/type_classifier"
OUTPUT_DIR="${PROJECT_ROOT}/models/type_classifier"

echo "=============================================="
echo "TYPE 分类模型训练"
echo "=============================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# Step 1: 准备数据
echo "Step 1: 准备训练数据..."
python "${PROJECT_ROOT}/apps/trainer/type_classifier/prepare_data.py"

if [ $? -ne 0 ]; then
    echo "数据准备失败！"
    exit 1
fi

echo ""

# Step 2: 训练模型
echo "Step 2: 训练模型..."
python "${PROJECT_ROOT}/apps/trainer/type_classifier/train.py" \
    --model_name "xlm-roberta-base" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5

if [ $? -ne 0 ]; then
    echo "模型训练失败！"
    exit 1
fi

echo ""

# Step 3: 评估模型
echo "Step 3: 评估模型..."
python "${PROJECT_ROOT}/apps/trainer/type_classifier/evaluate.py" \
    --model_dir "${OUTPUT_DIR}/final_model" \
    --data_file "${DATA_DIR}/val.jsonl" \
    --test_generalization

echo ""
echo "=============================================="
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}/final_model"
echo "=============================================="
