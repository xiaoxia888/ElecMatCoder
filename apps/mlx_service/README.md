# MLX Lazy Model Service

给平台一阶段结构化模型和二阶段编码模型使用的本地 MLX-LM 服务，接口兼容 `apps/hf_lazy_service` 的最小子集。

## 配置文件

最终只维护一个配置文件：

- `apps/mlx_service/service.yaml`

里面同时包含：

- 启动模式
- 模型自动休眠配置
- 单进程端口
- 多 worker 端口
- gateway 配置
- 模型路径和推理参数

## 启动

统一使用这一条命令：

```bash
python -m apps.mlx_service.launch \
  --config apps/mlx_service/service.yaml
```

## 切换启动方式

编辑 `apps/mlx_service/service.yaml`：

```yaml
deployment:
  mode: single
```

可选值：

- `single`：单进程，一个服务同时管理 `type`、`material-standard` 和 `coder`
- `gateway_serial`：多 worker + gateway，gateway 全局串行
- `gateway_parallel`：多 worker + gateway，按 worker 并行

## 自动休眠

配置在 `service.yaml`：

```yaml
service:
  max_loaded_models: 3
  idle_timeout_seconds: 1800
  idle_check_interval_seconds: 60
```

含义：

- `max_loaded_models`：最多同时保留几个已加载模型
- `idle_timeout_seconds`：模型空闲多久后自动卸载，`1800` 秒等于 30 分钟
- `idle_check_interval_seconds`：后台检查间隔，`60` 秒表示最多延迟约一分钟触发卸载

设置 `idle_timeout_seconds: 0` 可以关闭自动休眠。

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

平台统一访问：

- `http://127.0.0.1:8200`

模型名约定：

- `type`：一阶段种类结构化抽取
- `material-standard`：一阶段材质/规范结构化抽取
- `coder`：二阶段字段编码

## 说明

- `single` 模式最简单，内存通常最低。
- `gateway_serial` 模式保留给对并发非常保守的场景。
- `gateway_parallel` 模式适合批量请求，允许不同 worker 并行。
- 单个 worker 内部仍然串行推理，避免本地 Apple Silicon 上多请求争抢 Metal 资源。
