#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
用法：
  bash install_openbayes.sh /持久化目录/unsloth

示例：
  bash install_openbayes.sh /openbayes/home/unsloth
  bash install_openbayes.sh /home/unsloth

可选环境变量：
  UNSLOTH_PYTHON_VERSION=3.12
  UNSLOTH_TORCH_FAMILY=cu128  # 可选，但必须与本机 nvcc 主次版本一致
  UNSLOTH_INSTALL_URL=https://unsloth.ai/install.sh
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

ROOT=$1
if [[ $ROOT != /* ]]; then
    echo "错误：安装路径必须是绝对路径，收到：$ROOT" >&2
    exit 2
fi

mkdir -p "$ROOT"
ROOT=$(cd -L "$ROOT" && pwd -L)
if [[ ! -w $ROOT ]]; then
    echo "错误：安装路径不可写，收到：$ROOT" >&2
    exit 2
fi

PYTHON_VERSION=${UNSLOTH_PYTHON_VERSION:-3.12}
TORCH_FAMILY=${UNSLOTH_TORCH_FAMILY:-}
INSTALL_URL=${UNSLOTH_INSTALL_URL:-}

find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
    elif [[ -n ${CUDA_HOME:-} && -x ${CUDA_HOME}/bin/nvcc ]]; then
        printf '%s\n' "${CUDA_HOME}/bin/nvcc"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        printf '%s\n' /usr/local/cuda/bin/nvcc
    else
        return 1
    fi
}

NVCC_BIN=$(find_nvcc || true)
if [[ -z $NVCC_BIN ]]; then
    if [[ -z $TORCH_FAMILY ]]; then
        echo "错误：没有找到 nvcc，无法确定本机 CUDA Toolkit 版本。" >&2
        echo "请安装 nvcc，或显式指定，例如：UNSLOTH_TORCH_FAMILY=cu128 bash $0 $ROOT" >&2
        exit 2
    fi
    case "$TORCH_FAMILY" in
        cu118|cu124|cu126|cu128|cu130) ;;
        *)
            echo "错误：没有 nvcc 时，UNSLOTH_TORCH_FAMILY 必须是 cu118、cu124、cu126、cu128 或 cu130。" >&2
            exit 2
            ;;
    esac
    NVCC_VERSION=未检测到
    NVCC_BIN=未找到
    TORCH_SELECTION_SOURCE="用户显式指定，未与本机 CUDA Toolkit 验证"
else
    NVCC_VERSION=$(
        "$NVCC_BIN" --version \
            | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
            | head -n 1
    )
    if [[ -z $NVCC_VERSION ]]; then
        echo "错误：无法从 $NVCC_BIN --version 解析 CUDA Toolkit 版本。" >&2
        exit 2
    fi

    case "$NVCC_VERSION" in
        11.8) DETECTED_TORCH_FAMILY=cu118 ;;
        12.4) DETECTED_TORCH_FAMILY=cu124 ;;
        12.6) DETECTED_TORCH_FAMILY=cu126 ;;
        12.8) DETECTED_TORCH_FAMILY=cu128 ;;
        13.0) DETECTED_TORCH_FAMILY=cu130 ;;
        *)
            echo "错误：本机 nvcc=${NVCC_VERSION}，但当前 Unsloth/PyTorch 安装器没有对应的精确 wheel 系列。" >&2
            echo "支持的精确对应关系：11.8→cu118、12.4→cu124、12.6→cu126、12.8→cu128、13.0→cu130。" >&2
            echo "脚本不会替你选择相近版本。请安装受支持的 CUDA Toolkit，或自行修改安装策略。" >&2
            exit 2
            ;;
    esac

    if [[ -n $TORCH_FAMILY && $TORCH_FAMILY != "$DETECTED_TORCH_FAMILY" ]]; then
        echo "错误：UNSLOTH_TORCH_FAMILY=${TORCH_FAMILY} 与本机 nvcc=${NVCC_VERSION}（${DETECTED_TORCH_FAMILY}）不一致。" >&2
        exit 2
    fi
    TORCH_FAMILY=$DETECTED_TORCH_FAMILY
    TORCH_SELECTION_SOURCE="与 nvcc 精确匹配"
fi

STUDIO_HOME=$ROOT/studio
CACHE_HOME=$ROOT/cache
ENV_FILE=$ROOT/env.sh
START_FILE=$ROOT/start-studio.sh

mkdir -p \
    "$STUDIO_HOME" \
    "$ROOT/bin" \
    "$ROOT/python" \
    "$ROOT/tools" \
    "$ROOT/share" \
    "$ROOT/datasets" \
    "$ROOT/outputs" \
    "$ROOT/tmp" \
    "$CACHE_HOME/huggingface" \
    "$CACHE_HOME/torch" \
    "$CACHE_HOME/uv" \
    "$CACHE_HOME/pip" \
    "$CACHE_HOME/triton" \
    "$CACHE_HOME/cuda"

write_export() {
    local name=$1
    local value=$2
    printf 'export %s=%q\n' "$name" "$value"
}

{
    echo '#!/usr/bin/env bash'
    write_export UNSLOTH_ROOT "$ROOT"
    write_export UNSLOTH_STUDIO_HOME "$STUDIO_HOME"
    write_export HF_HOME "$CACHE_HOME/huggingface"
    write_export TORCH_HOME "$CACHE_HOME/torch"
    write_export XDG_CACHE_HOME "$CACHE_HOME"
    write_export XDG_DATA_HOME "$ROOT/share"
    write_export UV_CACHE_DIR "$CACHE_HOME/uv"
    write_export UV_PYTHON_INSTALL_DIR "$ROOT/python"
    write_export UV_PYTHON_BIN_DIR "$ROOT/bin"
    write_export UV_TOOL_DIR "$ROOT/tools"
    write_export UV_TOOL_BIN_DIR "$ROOT/bin"
    write_export PIP_CACHE_DIR "$CACHE_HOME/pip"
    write_export TRITON_CACHE_DIR "$CACHE_HOME/triton"
    write_export CUDA_CACHE_PATH "$CACHE_HOME/cuda"
    write_export TMPDIR "$ROOT/tmp"
    printf 'export PATH=%q:%q:$PATH\n' "$STUDIO_HOME/bin" "$ROOT/bin"
} > "$ENV_FILE"

cat > "$START_FILE" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "$ROOT/env.sh"
HOST=${UNSLOTH_HOST:-0.0.0.0}
PORT=${UNSLOTH_PORT:-8888}
exec "$UNSLOTH_STUDIO_HOME/bin/unsloth" studio -H "$HOST" -p "$PORT" "$@"
EOF
chmod +x "$ENV_FILE" "$START_FILE"

source "$ENV_FILE"

echo "安装根目录：${ROOT}"
echo "Studio目录：${STUDIO_HOME}"
echo "训练集目录：${ROOT}/datasets"
echo "训练输出目录：${ROOT}/outputs"
echo "Python版本：${PYTHON_VERSION}"
echo "本机CUDA Toolkit：${NVCC_VERSION}（${NVCC_BIN}）"
echo "PyTorch计算平台：${TORCH_FAMILY}（${TORCH_SELECTION_SOURCE}）"

INSTALL_SCRIPT=$ROOT/tmp/unsloth-install.$$.sh
cleanup() {
    rm -f "$INSTALL_SCRIPT"
}
trap cleanup EXIT

download_installer() {
    local url=$1
    echo "下载安装脚本：$url"
    if command -v curl >/dev/null 2>&1; then
        curl \
            --fail \
            --location \
            --silent \
            --show-error \
            --http1.1 \
            --retry 8 \
            --retry-all-errors \
            --retry-delay 3 \
            --connect-timeout 20 \
            --max-time 600 \
            --output "$INSTALL_SCRIPT" \
            "$url" && [[ -s $INSTALL_SCRIPT ]] && return 0
    fi
    rm -f "$INSTALL_SCRIPT"
    if command -v wget >/dev/null 2>&1; then
        wget \
            --quiet \
            --tries=8 \
            --timeout=30 \
            --output-document="$INSTALL_SCRIPT" \
            "$url" && [[ -s $INSTALL_SCRIPT ]] && return 0
    fi
    rm -f "$INSTALL_SCRIPT"
    return 1
}

if [[ -n $INSTALL_URL ]]; then
    URLS=("$INSTALL_URL")
else
    URLS=(
        "https://unsloth.ai/install.sh"
        "https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh"
    )
fi

DOWNLOADED_URL=
for url in "${URLS[@]}"; do
    if download_installer "$url"; then
        DOWNLOADED_URL=$url
        break
    fi
    echo "下载失败，尝试下一个官方地址。" >&2
done

if [[ -z $DOWNLOADED_URL ]]; then
    echo "错误：Unsloth 官方安装脚本下载失败。请检查代理或设置 UNSLOTH_INSTALL_URL。" >&2
    exit 1
fi

if ! head -n 5 "$INSTALL_SCRIPT" | grep -q 'Unsloth Studio Installer'; then
    echo "错误：下载内容不像 Unsloth 官方安装脚本，拒绝执行：$DOWNLOADED_URL" >&2
    exit 1
fi

echo "安装脚本下载完成：$DOWNLOADED_URL"

env \
    UNSLOTH_PYTHON="$PYTHON_VERSION" \
    UNSLOTH_TORCH_INDEX_FAMILY="$TORCH_FAMILY" \
    UNSLOTH_STUDIO_HOME="$STUDIO_HOME" \
    UNSLOTH_SKIP_AUTOSTART=1 \
    HF_HOME="$HF_HOME" \
    TORCH_HOME="$TORCH_HOME" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    XDG_DATA_HOME="$XDG_DATA_HOME" \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
    UV_INSTALL_DIR="$ROOT/bin" \
    UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
    UV_PYTHON_BIN_DIR="$UV_PYTHON_BIN_DIR" \
    UV_TOOL_DIR="$UV_TOOL_DIR" \
    UV_TOOL_BIN_DIR="$UV_TOOL_BIN_DIR" \
    PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    CUDA_CACHE_PATH="$CUDA_CACHE_PATH" \
    TMPDIR="$TMPDIR" \
    sh "$INSTALL_SCRIPT"

if [[ ! -x $STUDIO_HOME/bin/unsloth ]]; then
    echo "错误：安装结束，但没有找到 $STUDIO_HOME/bin/unsloth" >&2
    exit 1
fi

echo
echo "安装完成。使用官方命令启动："
echo "  source $ENV_FILE"
echo "  unsloth studio -H 0.0.0.0 -p 8888"
echo
echo "也可以使用等价的一键启动脚本："
echo "  $START_FILE"
