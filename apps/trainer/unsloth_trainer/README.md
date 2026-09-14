# Unsloth YAML 训练入口

## Unsloth Studio 一键安装

脚本接受任意绝对且可写的安装根目录，并将 Unsloth Studio、uv 管理的 Python、模型缓存和训练输出统一放在其中。OpenBayes 示例：

```bash
bash apps/trainer/unsloth_trainer/install_openbayes.sh /openbayes/home/unsloth
```

普通 Linux 服务器示例：

```bash
bash apps/trainer/unsloth_trainer/install_openbayes.sh /home/unsloth
```

安装完成后加载持久化环境变量，并使用 Unsloth 官方命令启动：

```bash
source /openbayes/home/unsloth/env.sh
unsloth studio -H 0.0.0.0 -p 8888
```

`/openbayes/home/unsloth/start-studio.sh` 只是上述两步的快捷封装。

训练集建议放入 `/openbayes/home/unsloth/datasets`，LoRA、checkpoint 和日志的输出目录设置为 `/openbayes/home/unsloth/outputs`。不要在安装完成后移动 `/openbayes/home/unsloth/studio`，Python 虚拟环境记录了绝对路径。

OpenBayes 某些容器会将 `/openbayes/home` 映射到物理路径 `/output`；脚本会保留平台公开的 `/openbayes/home/...` 逻辑路径，该映射属于正常现象。

脚本还会把 uv 管理的 Python 固定到 `/openbayes/home/unsloth/python`。只持久化 uv 缓存并不够：如果 Python 仍位于默认的 `/root/.local/share/uv/python`，容器重启后虚拟环境会因解释器目标消失而报 `bad interpreter`。

脚本优先按本机 `nvcc --version` 的 CUDA Toolkit 主次版本严格选择相同的 PyTorch wheel，不采用 `nvidia-smi` 显示的驱动上限。当前允许的精确对应关系为 11.8→cu118、12.4→cu124、12.6→cu126、12.8→cu128、13.0→cu130；没有对应 wheel 时直接停止，不会替用户选择相近版本。有 `nvcc` 时，`UNSLOTH_TORCH_FAMILY` 与其不一致会停止；没有安装 `nvcc` 时，必须由用户显式指定上述受支持系列，脚本会标记为“未与本机 Toolkit 验证”。

这是独立于 LLaMA Factory 和 Unsloth Studio 的 Unsloth Core SFT 训练模块，底层直接调用官方 `FastLanguageModel`、`SFTTrainer` 和 `SFTConfig`。

配置分为两层：

- `train_config/<任务>/train.yaml`：每个任务各自独立的任务配置，保存模型、数据、输出位置以及确实因任务而不同的训练参数。
- `config.yaml`：所有任务完全相同的公共设置，例如 Alpaca 格式、LoRA 底层参数、自动换算为整数步数的 3% warmup、日志、验证/保存策略和断点续训。

`train_config/种类`、`train_config/尺寸`、`train_config/材质规范`、`train_config/编码` 中的 `train.yaml` 是正式入口。同目录中带 4090/5090 名称的 YAML 仅保留为旧实验记录，不再作为新训练入口。

每份配置顶部有 5 项日常参数：

```yaml
finetuning_method: "lora"
base_model: "/openbayes/input/input0/Qwen3.5-9B"
train_dataset: "训练集.json"
validation_dataset: "验证集.json"
output_dir: "/openbayes/input/input2/unsloth_models/material"
```

它们分别表示：

- `finetuning_method`：微调方式。`lora` 使用 16-bit 基座权重；`qlora` 使用 4-bit 基座权重以节省显存；`full` 是显存需求很高的全参数微调。三种方式都受配置入口支持，程序按用户选择直接执行。
- `base_model`：基座模型所在的本地目录。
- `train_dataset`：训练数据文件。
- `validation_dataset`：验证数据文件，用于观察模型是否真的变好。
- `output_dir`：checkpoint、日志和最终 LoRA adapter 的保存目录。

学习率、轮次、batch、评估和保存间隔等任务专属参数保存在同一个 `train.yaml` 的下半部分，并已按对应数据集设置好。warmup 使用公共的 3% 比例随总训练步数自动换算为 `warmup_steps`，不会把已弃用的 `warmup_ratio` 传给 Trainer。`save_total_limit` 和 `load_best_model_at_end` 特意保留在每个任务文件中，允许各模型独立控制候选数量和自动选优。4-bit/16-bit 加载和 LoRA 开关由 `finetuning_method` 自动协调。

## 支持的数据格式

- `alpaca`：JSON/JSONL，字段默认为 `instruction`、`input`、`output`。
- `chatml`：JSON/JSONL，每条样本包含 `messages` 数组。
- `text`：JSON/JSONL，每条样本已经包含套用聊天模板后的 `text`。

当前仓库中的 `种类_train.json` 和 `种类_val.json` 可以直接按 `alpaca` 格式使用，不需要 LLaMA Factory 的 `dataset_info.json`。

## 四个任务配置

根据要训练的模型选择对应文件：

```text
train_config/种类/train.yaml
train_config/尺寸/train.yaml
train_config/材质规范/train.yaml
train_config/编码/train.yaml
```

四份配置的提示词由各自 Alpaca 数据集的 `instruction` 字段提供，不额外叠加统一 system prompt。种类、尺寸和材质规范使用 1024 上下文；短输入的编码模型使用 256。所有文件均不绑定 GPU 型号。

程序在加载权重前检查本地 `config.json` 和权重文件，并打印模型架构、层数、hidden size 与有效 batch，避免误用其他模型。

训练结束后，运行目录会自动生成以下图表：

- `training_loss.png`：原始训练损失和移动平均曲线；跨度过大时自动使用对数坐标。
- `validation_loss.png`：各次验证损失、数值标签和最佳验证检查点。
- `training_curves.png`：全程曲线与后期局部放大的两面板汇总图，兼容原有文件名。
- `learning_rate.png`：学习率调度曲线（训练日志记录了学习率时生成）。

`run_summary.json` 的 `training_curve_artifacts` 字段记录所有图片路径；原有
`training_curves` 字段继续指向汇总图。没有配置验证集时不会生成
`validation_loss.png`，这时也不能依据验证损失自动选择最佳 checkpoint。

## 运行

服务器部署结构为：

```text
/home/waas/
└── unsloth_trainer/
```

在 `/home/waas` 下运行时，先创建隔离环境并安装固定版本：

```bash
conda env create \
  -f unsloth_trainer/environment-qwen35-rtx5090.yml
conda activate qwen35-unsloth
python -m pip install \
  -r unsloth_trainer/requirements-qwen35-rtx5090.txt
python -m pip check
```

该环境固定为 Python 3.12、PyTorch 2.8.0/cu128、Triton 3.4.0、Transformers 5.3.0 和对应 Unsloth 版本。它是 Unsloth Core 环境，不包含 Studio、`causal-conv1d` 或 FLA。

先只检查 YAML 和数据字段：

```bash
cd /home/waas
python -m unsloth_trainer.train \
  --config unsloth_trainer/train_config/材质规范/train.yaml \
  --validate-only
```

使用当前 Python 环境正式训练：

```bash
cd /home/waas
python -m unsloth_trainer.train \
  --config unsloth_trainer/train_config/材质规范/train.yaml
```

建议使用独立的 Unsloth Core Conda 环境，不要调用 Studio 的 Python 环境。正式运行前先使用 `--validate-only` 检查模型路径、数据路径和字段格式。

完整仓库布局仍然支持原来的模块路径：

```bash
python -m apps.trainer.unsloth_trainer.train \
  --config "apps/trainer/unsloth_trainer/train_config/材质规范/train.yaml"
```

首次安装并验证训练可启动后，可生成包含全部间接依赖的机器级锁文件：

```bash
python -m pip freeze --local > unsloth_trainer/requirements-lock-rtx5090.txt
```

每次运行会生成 `run_YYYYMMDD_HHMMSS` 目录，并保存合并后的完整 `config.yaml`、任务 `source_config.yaml`、checkpoint、训练状态、指标和最终 LoRA adapter。断点恢复路径通过任务配置中的 `training.resume_from_checkpoint` 设置。

默认使用 `resume_from_checkpoint: auto`：首次执行创建新的时间戳目录；训练意外中断后，重新执行完全相同的命令会自动恢复该任务最新且包含 `trainer_state.json` 的 checkpoint，并继续写入原运行目录。若检测到 `run_summary.json` 和完整的 `final_adapter`，程序会提示训练已经完成并退出，避免重复训练。

需要主动开始新实验时，可以更换 `training.output_dir`，或者临时设置：

```yaml
training:
  resume_from_checkpoint: null
```

也可以填写具体的 `checkpoint-N` 绝对路径进行手工恢复。

训练结束后还会生成：

- `training_curves.png`：训练损失与验证损失曲线。
- `best_checkpoint.json`：最佳 checkpoint、最佳 eval loss 和推荐 LoRA 路径。
- `run_summary.json`：本次运行目录、最终 adapter、最佳指标和训练指标汇总。
- `tensorboard/`：Trainer 原生实时事件日志。

启用 `load_best_model_at_end: true` 时，`final_adapter` 是从最佳 checkpoint 加载后保存的推荐 LoRA 权重；不需要再手工复制 checkpoint 中的 adapter。

材质规范任务的上述策略直接保存在 `train_config/材质规范/train.yaml`：每 1000 步同时执行验证和保存，最多保留 3 个 checkpoint。程序根据 `eval_loss` 自动将最佳结果导出为 `final_adapter`，保留的 `checkpoint-*` 仍可逐个推理测试；`training_curves.png` 用于辅助观察训练损失和验证损失，但不能替代业务测试集的人工效果比较。

训练启动后，可以在另一个终端启动实时面板：

```bash
conda activate qwen35-unsloth
tensorboard \
  --logdir /home/waas/outputs/unsloth_qwen35_9b/category_rtx5090 \
  --host 0.0.0.0 \
  --port 6006
```

浏览器访问 `http://服务器地址:6006`。训练 loss 每 `logging_steps`（默认 20）更新，验证 loss 每 `eval_steps`（默认 200）更新；公网服务器需要在算力平台放行或映射 6006 端口。

Qwen3.5 的默认参数遵循 [Unsloth 官方 Qwen3.5 指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune)：BF16 LoRA、标准语言层 LoRA target 和 Unsloth gradient checkpointing。
