# HF Lazy Model Service

统一的 HuggingFace 模型服务，按模型名懒加载，保留最近使用模型，空闲模型自动释放。

## 功能

- 多模型注册
- 按模型名调用
- 懒加载
- LRU 淘汰
- 空闲释放
- 返回原始输出和解析后的 JSON

## 注册表示例

```yaml
models:
  fittings:
    model_path: /Users/guoxi/Desktop/workspace/NJNCC/python_code/LlamaFactory/saves/qwen3-4b-base/lora/fittings
    device: mps
    dtype: auto
    instruction: 你是一个工业管道材料结构化信息提取助手。请从材料描述中提取结构化信息，并以 JSON 格式返回。
    max_new_tokens: 256
    temperature: 0.0
    top_p: 1.0

  material:
    model_path: /Users/guoxi/Desktop/workspace/NJNCC/python_code/LlamaFactory/saves/qwen3-4b-base/lora/material
    device: mps
    dtype: auto
    max_new_tokens: 256
    temperature: 0.0
    top_p: 1.0
```

## 启动

```bash
python -m apps.hf_lazy_service.server \
  --registry apps/hf_lazy_service/models.example.yaml \
  --host 0.0.0.0 \
  --port 8100 \
  --device auto \
  --max-loaded-models 2 \
  --idle-timeout-seconds 1800
```

## 接口

### 健康检查

```bash
curl http://127.0.0.1:8100/health
```

### 查看模型

```bash
curl http://127.0.0.1:8100/models
```

### 预测

```bash
curl http://127.0.0.1:8100/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fittings",
    "text": "NB/T 47008(II);GB/T 19326 SOL-90 SW Olet CL3000 DN50xDN15 20 DN50x15"
  }'
```

### 卸载单个模型

```bash
curl -X POST http://127.0.0.1:8100/models/fittings/unload
```

### 卸载全部模型

```bash
curl -X POST http://127.0.0.1:8100/models/unload_all
```
