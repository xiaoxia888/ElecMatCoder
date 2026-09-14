# 多模型 Excel 编码评测

该工具完全独立于训练脚本。通过 `--task` 既可单独测试种类、尺寸等字段，也可运行生产级完整编码链路。默认的 `managed_vllm` 后端会自动启动服务、完成并发推理、停止服务并切换下一个底座，用户只需执行一次命令。

## 自动 vLLM 调度

`config.yaml` 中的单字段模型使用：

```yaml
models:
  qwen3-8B:
    routes:
      type:
        backend: managed_vllm
        service_config: ../vllm_service/service.eval.qwen3-8b.yaml
        served_model: type
      size:
        backend: managed_vllm
        service_config: ../vllm_service/service.eval.qwen3-8b.yaml
        served_model: size-thick-pressure
```

`config.full.yaml` 中的每个 group 按阶段指定服务：

```yaml
groups:
  group1:
    stages:
      type:
        backend: managed_vllm
        service_config: ../vllm_service/service.eval.qwen3-8b.yaml
        served_model: type
      material:
        backend: managed_vllm
        service_config: ../vllm_service/service.eval.qwen3-4b.yaml
        served_model: material-standard
      size:
        backend: managed_vllm
        service_config: ../vllm_service/service.eval.qwen3-8b.yaml
        served_model: size-thick-pressure
```

当前模型矩阵为：种类测试 Qwen3-8B、Qwen3.5-4B、Qwen3.5-9B；尺寸测试 Qwen3-8B、Qwen3.5-9B；材质规范暂时只测试 Qwen3-4B。Qwen3.5-9B 材质 LoRA 尚未训练，因此当前两个完整编码组合都复用 Qwen3-4B 材质模型，训练完成后再单独注册。

调度器会将相同 `service_config` 的 route 放在同一次服务启动中执行，并对多个 group 共用的完全相同阶段复用推理结果。切换底座时会先向启动器发送 `SIGTERM`，等待 engine 和网关退出后再启动下一个。`Ctrl+C` 也会进入同一清理流程。

启动前请确保 `8200` 端口没有已运行的服务；评测器不会终止或接管未知进程。单卡参数在 `apps/vllm_service/profiles/single-3090.yaml`，请按服务器实际路径修改各个 `service.eval.*.yaml`。

## task 控制

- `--task type` 或 `--task 种类`：只运行种类 LoRA，只比较 TYPE 的最终编码。
- `--task size` 或 `--task 尺寸`：只运行尺寸/壁厚/磅级 LoRA，只比较 SIZE 编码。
- `--task thickness` / `pressure` / `material` / `standard`：只比较对应字段。
- `--task full` 或 `--task 完整编码`：对每套候选“组合方案”依次运行 `type → material → size` 三个模型，合并 JSON 后调用项目正式 `LlmPipeEncoder`，比较完整 C1。三个阶段可使用不同底座。
- `--task code`：保留旧功能，单个“直接编码”LoRA 的原始文本输出就是结果，不等同于 `full`。

## 输入

Excel 至少包含两列：`材料描述`、`正确代码`。列名可通过命令行覆盖。
为了保留编码中的前导零，`正确代码` 列应在 Excel 中设置为文本格式。

配置文件已拆分：

- `config.yaml`：单字段评测，`models` 下是直接对比的单个模型。
- `config.full.yaml`：完整编码评测，`groups.group1`、`groups.group2` 分别代表一套流水线方案；每组分别配置 `stages.type`、`stages.material`、`stages.size` 的底座和 LoRA。
- `coder`：二阶段 coder 的底座和 LoRA；也可切换到 `mlx_service`。
- `tasks`：各任务提示词路径。

不传 `--config` 时，程序会根据 `--task` 自动选择：单字段用 `config.yaml`，`full` 用 `config.full.yaml`。

`thickness/pressure` 默认复用 `size` 权重，`standard` 默认复用 `material` 权重。

每个任务可在 `tasks.<task>.truth_column` 指定自己的真值列。例如 `type` 应对比 TYPE 标签列，`size` 应对比 SIZE 标签列，`full` 才对比完整 `C1-名称简写`。如果 Excel 只有一个真值列，也可用 `--truth-column` 覆盖。

## 运行

从项目根目录执行：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task type \
  --input /path/to/test.xlsx \
  --output-dir /path/to/type-evaluation \
  --models qwen3-8B qwen3.5-4B qwen3.5-9B
```

材质规范任务：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task material \
  --input /path/to/test.xlsx \
  --output-dir /path/to/material-result \
  --models qwen3-4B
```

尺寸任务：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task 尺寸 \
  --input /path/to/test.xlsx \
  --output-dir /path/to/尺寸模型对比
```

完整编码：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task full \
  --input /path/to/test.xlsx \
  --output-dir /path/to/完整编码对比 \
  --groups group1 group2
```

省略 `--groups` 会测试配置中的全部组。

`managed_vllm` 的底座、LoRA 和提示词都由对应 `service.eval.*.yaml` 管理，不能用 `--base-model`、`--adapter` 或 `--prompt` 临时覆盖。修改服务配置后重新执行同一条评测命令即可。

仅当模型改回 `backend: local_transformers` 时，才可以临时覆盖提示词或权重：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task 种类 \
  --input /path/to/test.xlsx \
  --prompt /path/to/type_prompt.txt \
  --adapter qwen3-8B=/path/to/8b-lora \
  --adapter qwen3.5-4B=/path/to/4b-lora \
  --adapter qwen3.5-9B=/path/to/9b-lora
```

先进行路径和表头校验，不加载模型：

```bash
python -m apps.model_excel_evaluator.evaluate \
  --task type \
  --input /path/to/test.xlsx \
  --validate-only
```

`vllm.concurrency` 是评测器同时发送的 HTTP 请求数，单张 3090 默认为 `16`。`inference.batch_size` 只在将某个模型显式改回 `backend: local_transformers` 时生效。

未指定 `--output-dir` 时，程序在 `apps/model_excel_evaluator/outputs/` 下创建带时间戳的目录。指定目录后，其中会生成：

- `评测结果.xlsx`
- `准确率对比.png`
- `耗时对比.png`
- `模型差异.png`

Excel 包含 `评测明细`、`模型汇总`、`模型差异` 和 `运行配置` 四个工作表。旧的 `--output /path/result.xlsx` 仍然可用，分析图会写到该 Excel 所在目录。
