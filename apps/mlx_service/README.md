# MLX Lazy Model Service

一个给平台一阶段结构化模型用的本地 MLX-LM 服务，接口兼容 `apps/hf_lazy_service` 的最小子集。

## 单 worker 启动

```bash
python -m apps.mlx_service.server \
  --registry apps/mlx_service/models.example.yaml \
  --host 0.0.0.0 \
  --port 8200
```

## 多 worker + gateway 启动

如果你不想手工起 3 个进程，直接用启动器：

```bash
python -m apps.mlx_service.launch \
  --registry apps/mlx_service/cluster.example.yaml
```

这条命令会统一拉起：
- `type` worker
- `material-standard` worker
- gateway

你只需要准备：
- `apps/mlx_service/type.worker.yaml`
- `apps/mlx_service/material.worker.yaml`
- `apps/mlx_service/gateway.example.yaml`

### 1. 启动 `type` worker

准备一个只包含 `type` 的 registry，例如 `apps/mlx_service/type.worker.yaml`：

```yaml
models:
  type:
    model_path: /abs/path/to/qwen3-8b-type-mlx
    instruction: "你是一个工业管道材料结构化信息提取助手。请从材料描述中提取结构化信息，并以 JSON 格式返回。"
    max_tokens: 512
    temperature: 0.0
    top_p: 1.0
    trust_remote_code: true
```

启动：

```bash
python -m apps.mlx_service.server \
  --registry apps/mlx_service/type.worker.yaml \
  --host 0.0.0.0 \
  --port 8201
```

### 2. 启动 `material-standard` worker

准备一个只包含 `material-standard` 的 registry，例如 `apps/mlx_service/material.worker.yaml`：

```yaml
models:
  material-standard:
    model_path: /abs/path/to/qwen3-8b-material-standard-mlx
    instruction: "你是一个工业管道材料结构化信息提取助手。请从材料描述中提取结构化信息，并以 JSON 格式返回。"
    max_tokens: 512
    temperature: 0.0
    top_p: 1.0
    trust_remote_code: true
```

启动：

```bash
python -m apps.mlx_service.server \
  --registry apps/mlx_service/material.worker.yaml \
  --host 0.0.0.0 \
  --port 8202
```

### 3. 启动 gateway

编辑：

- [gateway.example.yaml](/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder/apps/mlx_service/gateway.example.yaml)

其中：

- `gateway.concurrency_mode: "serial"` 表示 gateway 全局串行放行请求，适合本地 Apple Silicon
- `gateway.concurrency_mode: "parallel"` 表示允许不同 worker 并发

启动：

```bash
python -m apps.mlx_service.gateway \
  --registry apps/mlx_service/gateway.example.yaml \
  --host 0.0.0.0 \
  --port 8200
```

平台统一指向：

- `http://127.0.0.1:8200`

## 接口

- `GET /health`
- `GET /models`
- `POST /predict`
- `POST /models/{name}/unload`
- `POST /models/unload_all`

`POST /predict` 请求体示例：

```json
{
  "model": "type",
  "text": "90度弯头, 20 NB/T47008, FTE,CL 2000, SH/T3410, Galvanized , DN25",
  "instruction": "你是一个工业管道材料结构化信息提取助手。请从材料描述中提取结构化信息，并以 JSON 格式返回。",
  "max_new_tokens": 512,
  "temperature": 0.0,
  "top_p": 1.0
}
```

返回体关键字段：

- `raw_response`
- `parsed_json`
- `json_parse_ok`

## 说明

- HTTP 层是 FastAPI。
- 单个 worker 内部仍然串行推理，避免本地 Apple Silicon 上多请求争抢 Metal 资源。
- `gateway` 可配置：
  - `serial`：全局串行
  - `parallel`：按 worker 并发
- 真正并发依赖多个 worker 进程，由 gateway 负责按模型路由。
- 这是本地开发/内网服务方案，不是公网生产网关。
