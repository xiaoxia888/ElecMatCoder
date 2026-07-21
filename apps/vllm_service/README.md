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

新建独立Linux环境，依据CUDA驱动安装PyTorch后执行：

```bash
pip install -r apps/vllm_service/requirements-linux.txt
```

V100使用`float16`。A100、4090、5090可根据模型支持选择`float16`或`bfloat16`。

## 配置

编辑：

```text
apps/vllm_service/service.yaml
```

至少替换以下路径：

```yaml
engines:
  qwen3_8b:
    model_path: /models/Qwen3-8B
    lora_modules:
      type: /models/lora/type

  qwen3_4b:
    model_path: /models/Qwen3-4B
    lora_modules:
      size-thick-pressure: /models/lora/size-thick-pressure
      material-standard: /models/lora/material-standard
      coder: /models/lora/coder
```

先校验配置并查看实际启动命令：

```bash
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.yaml \
  --dry-run
```

## 一键启动

```bash
python -m apps.vllm_service.launch \
  --config apps/vllm_service/service.yaml
```

启动器会：

1. 同时启动8B和4B两个vLLM engine。
2. 等待两个engine通过健康检查。
3. 启动兼容MLX协议的统一网关。
4. 任意子进程退出时停止整组服务，避免残留进程占用显存。

## 单卡与双卡

双卡推荐配置：

```yaml
qwen3_8b:
  cuda_visible_devices: "0"
  gpu_memory_utilization: 0.90

qwen3_4b:
  cuda_visible_devices: "1"
  gpu_memory_utilization: 0.90
```

单张40GB及以上显卡可以让两个engine共用GPU 0，但两个进程的显存比例之和应小于1：

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

## 与现有平台连接

现有平台配置中的后端名称可以暂时保留`mlx_service`，只需把`service_url`改为Linux网关地址。这里复用的是协议，并不代表Linux服务器仍在运行MLX。

