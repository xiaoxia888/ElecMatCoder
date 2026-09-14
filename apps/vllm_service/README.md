# vLLM Multi-LoRA Deployment Service

用于在Linux NVIDIA服务器上部署平台的四个任务模型，并保持现有MLX服务的接口：

- `POST /predict`
- `GET /health`
- `GET /models`

平台仍然访问`http://服务器IP:8200/predict`，请求和响应字段不需要修改。

## 部署结构

```text
统一网关 :8200
  ├─ 8B vLLM engine :8301
  │    └─ type LoRA
  └─ 4B vLLM engine :8302
       ├─ size-thick-pressure LoRA
       ├─ material-standard LoRA
       └─ coder LoRA
```

三个4B任务必须使用相同的基座模型及tokenizer。配置中的LoRA路径应指向未合并的LlamaFactory checkpoint或导出的adapter目录，不能指向三个合并后的完整模型。

## 安装

推荐使用仓库内的统一 Conda 环境。它面向 3090、4090、5090 和 A100，Python、vLLM 及服务依赖均已固定：

初始化文件使用 `nodefaults`，不会读取用户全局 `.condarc` 中额外配置的 `defaults`、`pkgs/free` 或 `pytorch` 频道，避免重复下载仓库索引。

```bash
conda env create -f apps/vllm_service/environment-linux.yml
conda activate elecmat-vllm
python -m pip check
```

已有同名环境时，使用下面的命令按文件更新：

```bash
conda env update -n elecmat-vllm \
  -f apps/vllm_service/environment-linux.yml \
  --prune
```

环境文件不单独安装 `torch`。预编译 vLLM wheel 会带入与自身匹配的 PyTorch 和 CUDA 运行时；服务器只需提供兼容的 NVIDIA 驱动，无需让本机 `nvcc` 与 wheel 完全一致。

部分云平台会优先加载系统旧版 `libstdc++.so.6`。启动器会自动将当前 Python/Conda 环境的 `lib` 目录放到子进程的 `LD_LIBRARY_PATH` 首位，避免导入 sqlite/ICU 时出现 `CXXABI_* not found`。若直接绕过本项目启动器运行 vLLM，则仍需手动设置：`export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`。

初始化后可执行以下检查：

```bash
python -c 'import torch, vllm; print("vLLM:", vllm.__version__); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0))'
python -m pytest apps/vllm_service/tests -q
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.eval.qwen35-9b.yaml \
  --dry-run
```

A100、4090、5090可根据模型支持选择`float16`或`bfloat16`。注意：当前 vLLM 官方 GPU wheel 要求 NVIDIA GPU 计算能力不低于 7.5，因此仓库中的 V100（计算能力 7.0）配置不能直接使用此统一环境，需要另行维护旧版环境或源码构建方案。

## 配置

公共配置与硬件配置分离：

```text
apps/vllm_service/service.yaml
apps/vllm_service/profiles/*.yaml
```

模型 Excel 评测器使用独立的 `service.eval.*.yaml` 和 `profiles/single-3090.yaml`。这些配置由 `apps/model_excel_evaluator` 自动启动与停止，不要同时手动占用统一网关的 `8200` 端口。

`service.yaml`中的`profile`为必填项，启动时会自动加载同目录`profiles`下的配置：

```yaml
profile: dual-5090
```

已有Profile：

- `dual-3090`：双RTX 3090 24GB。
- `dual-4090`：双RTX 4090 24GB，关闭需要本地 C++ JIT 工具链的 FlashInfer sampler。
- `dual-5090`：双RTX 5090 32GB，关闭当前不兼容的FlashInfer sampler。
- `dual-v100-32gb`：双V100 32GB，强制使用`float16`。
- `single-a100-80gb`：单A100 80GB，两个engine共享GPU 0。
- `single-3090`：单RTX 3090 24GB，供自动 Excel 评测配置使用。

Profile只能覆盖GPU、精度、显存、并发和兼容环境变量，不能覆盖模型路径、LoRA、提示词、端口或模型路由。双V100 16GB不能直接套用32GB配置，需要量化模型和独立Profile。

同一硬件 Profile 可以预置多个 service 配置可能使用的 engine；当前 `models` 路由没有启用的 engine 预置项会被忽略。当前路由实际使用的每个 engine 则必须在 Profile 中配置完整硬件参数。

至少替换以下路径：

```yaml
engines:
  qwen3_8b:
    model_path: /home/waas/base-models/Qwen3-8B
    lora_modules:
      type: /home/waas/lora/type

  qwen3_4b:
    model_path: /home/waas/base-models/Qwen3-4B-Instruct-2507
    lora_modules:
      size-thick-pressure: /home/waas/lora/size-thick-pressure
      material-standard: /home/waas/lora/material-standard
      coder: /home/waas/lora/coder
```

每个对外模型还必须配置独立提示词文件：

```yaml
models:
  type:
    prompt_file: /data/MaterialsCode/prompts/种类微调提示词.txt
```

`prompt_file`支持绝对路径；相对路径按`service.yaml`所在目录解析。服务启动时读取
UTF-8文本，路径为空、文件不存在或内容为空都会直接报错。

先校验配置并查看实际启动命令：

```bash
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.yaml \
  --dry-run
```

临时测试其他硬件Profile时可以覆盖`service.yaml`中的选择：

```bash
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.yaml \
  --profile dual-4090 \
  --dry-run
```

## 一键启动

```bash
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.yaml
```

启动器会：

1. 依次启动8B和4B两个vLLM engine，避免启动阶段争抢显存。
2. 等待两个engine通过健康检查。
3. 启动兼容MLX协议的统一网关。
4. 任意子进程退出时停止整组服务，避免残留进程占用显存。

## 单卡与双卡

双卡参数配置在对应Profile中：

```yaml
qwen3_8b:
  cuda_visible_devices: "0"
  gpu_memory_utilization: 0.90

qwen3_4b:
  cuda_visible_devices: "1"
  gpu_memory_utilization: 0.90
```

单张40GB及以上显卡可以让两个engine共用GPU 0，但两个进程的显存比例之和应小于1。建议复制`single-a100-80gb.yaml`创建新Profile，不要把硬件参数写回`service.yaml`：

```yaml
qwen3_8b:
  cuda_visible_devices: "0"
  gpu_memory_utilization: 0.56

qwen3_4b:
  cuda_visible_devices: "0"
  gpu_memory_utilization: 0.34
```

具体比例取决于量化方式、上下文和并发数。24GB显卡通常需要量化。不要通过模型反复卸载实现一阶段推理，因为每条数据都需要同时使用8B和4B基座。

## 量化

预量化模型：将`model_path`指向量化后的Hugging Face模型目录，通常可以让vLLM自动读取`config.json`中的量化配置。

在线bitsandbytes量化示例：

```yaml
quantization: bitsandbytes
```

先使用非量化FP16建立准确率基线，再测试INT8；INT4必须使用独立测试集检查字段完全匹配率。

## 接口测试

```bash
curl http://127.0.0.1:8200/health
```

```bash
curl http://127.0.0.1:8200/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "type",
    "text": "TEE,RED SMLS BW A234 WPB ASME B16.9 DN100x80"
  }'
```

## 并发与显存基准

直接测试默认三条描述：

```bash
python -m apps.vllm_service.benchmark \
  --service-url http://127.0.0.1:8200 \
  --models type size-thick-pressure material-standard \
  --warmup
```

批量测试Excel：

```bash
python -m apps.vllm_service.benchmark \
  --service-url http://127.0.0.1:8200 \
  --models type size-thick-pressure material-standard \
  --input /data/test.xlsx \
  --text-column 材料描述 \
  --limit 100 \
  --rounds 1 \
  --group-concurrency 4 \
  --warmup \
  --output outputs/vllm_100条测试.json
```

脚本先按模型串行运行全部测试，再按模型并发运行相同测试，输出：

- 串行总耗时
- 并发总耗时
- 加速比和节省时间
- 每个模型的平均、P50、P95延迟
- 整组请求的平均、P50、P95延迟
- 每张GPU的峰值显存与利用率

判断方式：如果尺寸单独3秒、材质单独1秒，并发组耗时接近3秒，说明并发有效；接近4秒说明实际接近串行。

## 对比vLLM与MLX输出

向云端vLLM和Mac MLX发送完全相同的请求，并分别测试“使用服务配置提示词”和“强制相同提示词”：

```bash
python -m apps.vllm_service.compare_vllm_mlx \
  --text 'Olet 10"*1" SCH20*SCH80 Olet,10"*1",SCH20*SCH80 BW,A105,MSS SP-97' \
  --repeats 3 \
  --output vllm_mlx_comparison.json
```

脚本默认使用项目配置的云端vLLM地址、Mac MLX地址和`type`模型。报告会保存两端的
instruction、prompt、原始输出、解析JSON、重复调用稳定性和初步诊断。

## 与现有平台连接

现有平台配置中的后端名称可以暂时保留`mlx_service`，只需把`service_url`改为Linux网关地址。这里复用的是协议，并不代表Linux服务器仍在运行MLX。
