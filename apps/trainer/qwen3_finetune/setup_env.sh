#!/bin/bash
# Qwen3.5-4B/9B LoRA 微调环境安装脚本
# Qwen3.5 使用 Gated DeltaNet + Gated Attention 混合架构（非标准 Transformer）
# 需要 transformers v5+、flash-linear-attention、causal-conv1d 等专属依赖
#
# 使用方法:
#   bash apps/trainer/qwen3_finetune/setup_env.sh
#   # 可选：优先使用本地 wheel（离线/弱网推荐）
#   # LOCAL_WHEEL_DIR=/workspace/wheels bash apps/trainer/qwen3_finetune/setup_env.sh
#
# 注意事项:
#   - Qwen3.5 官方不建议使用 QLoRA (4-bit)，应使用 bf16 LoRA
#   - 4B bf16 LoRA 约需 10GB 显存，9B 约需 22GB 显存
#   - 训练首次启动会编译 Mamba Triton 内核，耗时较长属正常现象

set -e

ENV_NAME="qwen3.5ft"
PYTHON_VERSION="3.11"
ENABLE_QWEN35_FAST_KERNELS="${ENABLE_QWEN35_FAST_KERNELS:-0}"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
LOCAL_WHEEL_DIR="${LOCAL_WHEEL_DIR:-}"
if [ -z "$LOCAL_WHEEL_DIR" ]; then
    if [ -d "/workspace/wheels" ]; then
        LOCAL_WHEEL_DIR="/workspace/wheels"
    elif [ -d "/workspace/wheel" ]; then
        LOCAL_WHEEL_DIR="/workspace/wheel"
    else
        LOCAL_WHEEL_DIR="/workspace/wheels"
    fi
fi

echo "本地 wheel 目录: $LOCAL_WHEEL_DIR"

echo "=========================================="
echo "  Qwen3.5 微调环境安装"
echo "=========================================="

# 检查是否已在 Conda 环境中
if command -v conda &> /dev/null; then
    echo "发现 conda，准备初始化环境..."
    conda env remove -n $ENV_NAME -y 2>/dev/null || true
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    echo "未检测到 conda，将直接使用当前 Python 环境安装。"
fi

echo ""
echo "=========================================="
echo "  [1/5] 安装 PyTorch"
echo "=========================================="
if [ "$OS_NAME" = "Darwin" ]; then
    echo "检测到 macOS ($ARCH_NAME)，安装本地通用 PyTorch 包（CPU/MPS）..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
else
    echo "检测到 Linux，安装 CUDA 12.4 PyTorch 包..."
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi

echo ""
echo "=========================================="
echo "  [2/5] 安装 transformers v5+ (Qwen3.5 必需)"
echo "=========================================="
# Qwen3.5 的 model_type='qwen3_5' 需要 transformers >= 5.0.0
# 锁定 v5.3.0（2026-03-04 发布，已确认支持 Qwen3.5）
pip install transformers==5.3.0

echo ""
echo "=========================================="
echo "  [3/5] 安装 Flash Attention 2 (预编译包)"
echo "=========================================="
if [ "$OS_NAME" = "Darwin" ]; then
    echo "macOS 环境跳过 flash-attn（Linux CUDA 预编译包专用）"
else
    # Qwen3.5 混合架构中的 Gated Attention 层仍需要 flash-attn
    # 先下载到本地再安装，避免 pip 直连 GitHub 时出现 0-byte/invalid wheel
    FLASH_ATTN_BASENAME="flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    FLASH_ATTN_WHL="/tmp/${FLASH_ATTN_BASENAME}"
    FLASH_ATTN_LOCAL_WHL="${LOCAL_WHEEL_DIR}/${FLASH_ATTN_BASENAME}"
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_BASENAME}"

    if [ -s "$FLASH_ATTN_LOCAL_WHL" ]; then
        echo "使用本地 flash-attn wheel: $FLASH_ATTN_LOCAL_WHL"
        cp -f "$FLASH_ATTN_LOCAL_WHL" "$FLASH_ATTN_WHL"
    else
        if command -v curl &> /dev/null; then
            curl -L --retry 2 --retry-delay 2 --connect-timeout 15 --max-time 120 -o "$FLASH_ATTN_WHL" "$FLASH_ATTN_URL" || true
        elif command -v wget &> /dev/null; then
            wget --timeout=15 --tries=2 -O "$FLASH_ATTN_WHL" "$FLASH_ATTN_URL" || true
        fi
    fi

    if [ -s "$FLASH_ATTN_WHL" ]; then
        pip install --no-cache-dir --no-deps "$FLASH_ATTN_WHL" || {
            echo "⚠ Flash Attention 2 预编译包安装失败，回退到 sdpa (不影响训练正确性，仅影响速度)"
        }
    else
        echo "⚠ 未成功下载 Flash Attention 2 预编译包，回退到 sdpa (不影响训练正确性，仅影响速度)"
    fi
fi

echo ""
echo "=========================================="
echo "  [4/5] 安装 Qwen3.5 专属依赖"
echo "=========================================="
if [ "$ENABLE_QWEN35_FAST_KERNELS" = "1" ]; then
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "macOS 环境跳过 flash-linear-attention/causal-conv1d 快路径（Linux CUDA 内核专用）"
    else
    # flash-linear-attention: Qwen3.5 DeltaNet 层的快速计算内核
    # causal-conv1d: DeltaNet 层的因果卷积加速
    pip install flash-linear-attention

    # 先尝试安装与 torch2.5 + cu12 + cp311 匹配的预编译 causal-conv1d
    # 若网络或 wheel 不可用，再尝试 no-build-isolation；失败则继续（仅影响速度）
    CAUSAL_CONV_BASENAME="causal_conv1d-1.6.0+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    CAUSAL_CONV_WHL="/tmp/${CAUSAL_CONV_BASENAME}"
    CAUSAL_CONV_LOCAL_WHL="${LOCAL_WHEEL_DIR}/${CAUSAL_CONV_BASENAME}"
    CAUSAL_CONV_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/${CAUSAL_CONV_BASENAME}"

    if [ -s "$CAUSAL_CONV_LOCAL_WHL" ]; then
        echo "使用本地 causal-conv1d wheel: $CAUSAL_CONV_LOCAL_WHL"
        cp -f "$CAUSAL_CONV_LOCAL_WHL" "$CAUSAL_CONV_WHL"
    else
        if command -v curl &> /dev/null; then
            curl -L --retry 2 --retry-delay 2 --connect-timeout 15 --max-time 120 -o "$CAUSAL_CONV_WHL" "$CAUSAL_CONV_URL" || true
        elif command -v wget &> /dev/null; then
            wget --timeout=15 --tries=2 -O "$CAUSAL_CONV_WHL" "$CAUSAL_CONV_URL" || true
        fi
    fi

    if [ -s "$CAUSAL_CONV_WHL" ]; then
        pip install --no-cache-dir --no-deps "$CAUSAL_CONV_WHL" || true
    fi

    python -c "import causal_conv1d" 2>/dev/null || {
        echo "⚠ causal-conv1d 预编译包不可用，尝试 no-build-isolation 安装..."
        pip install --no-build-isolation causal-conv1d || \
        echo "⚠ causal-conv1d 安装失败，继续训练（不影响正确性，仅影响速度）"
    }
    fi
else
    echo "默认跳过 flash-linear-attention/causal-conv1d（更稳定）"
    echo "如需启用快路径，请设置: ENABLE_QWEN35_FAST_KERNELS=1"
fi

echo ""
echo "=========================================="
echo "  [5/5] 安装大模型微调依赖生态"
echo "=========================================="
pip install -r apps/trainer/qwen3_finetune/requirements.txt

echo ""
echo "=========================================="
echo "  安装完成! 版本验证:"
echo "=========================================="
python -c "
import transformers, trl, peft, accelerate, torch
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  accelerate:     {accelerate.__version__}')
print(f'  CUDA:           {torch.cuda.is_available()}')

try:
    import flash_attn
    print(f'  flash-attn:     {flash_attn.__version__}')
except ImportError:
    print('  flash-attn:     未安装 (将使用 sdpa 替代)')

try:
    import fla
    print(f'  flash-linear-attention: 已安装')
except ImportError:
    print('  flash-linear-attention: 未安装 (DeltaNet 层将使用 PyTorch 实现)')

try:
    import causal_conv1d
    print(f'  causal-conv1d:  已安装')
except ImportError:
    print('  causal-conv1d:  未安装')

if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    print(f'  显存:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "========================================================"
echo "  环境就绪! 重要提醒:"
echo "  - Qwen3.5 不建议使用 QLoRA(4-bit)，请使用 bf16 LoRA"
echo "  - 训练时请加 --no_quantize 参数禁用 4-bit 量化"
echo "  - 首次训练会编译 Triton 内核，请耐心等待"
echo ""
echo "  激活环境并开始训练:"
echo "    conda activate $ENV_NAME"
echo "    python -m apps.trainer.qwen3_finetune.train --no_quantize"
echo "========================================================"
