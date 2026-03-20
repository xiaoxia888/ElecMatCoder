#!/bin/bash
# ============================================
# TYPE Seq2Seq 模型训练脚本
# ============================================

# 进入项目根目录
cd "$(dirname "$0")/../../.."

echo "============================================"
echo "TYPE Seq2Seq 编码生成模型训练"
echo "============================================"

# 激活环境（如果需要）
# source /opt/miniconda3/etc/profile.d/conda.sh
# conda activate elecMatCoder

# 训练模型
python apps/trainer/type_seq2seq/train.py \
    --config src/seq2seq/config/training.yml \
    "$@"

echo ""
echo "训练完成！"
echo "模型保存至: outputs/type_seq2seq/final_model"
