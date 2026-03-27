# Qwen3-4B 管道材料 NER + 编码 — LoRA 微调

将 Qwen3-4B 微调为双任务模型：
1. **NER 分词**: 管道材料描述 → 结构化实体 JSON
2. **编码**: 提取的实体 → 标准编码

通过不同的 system prompt 区分两个任务，共享同一个模型。

## 目录结构

```
qwen3_finetune/
├── config.yaml                 # 一阶段 SFT 训练超参 & 数据路径配置
├── config_dpo.yaml             # 二阶段 DPO 训练配置
├── prepare_training_data.py    # 源数据 → ChatML 训练格式（统一入口）
├── train.py                    # LoRA + QLoRA 训练
├── train_dpo.py                # 二阶段 DPO 偏好优化训练
├── predict.py                  # 双任务预测测试
├── inference.py                # 推理 & 批量评估
├── merge_and_export.py         # LoRA 合并 & GGUF 导出
├── setup_env.sh                # 环境安装脚本
├── requirements.txt            # Python 依赖
└── README.md
```

## 数据组织

```
data/pipe/
├── llm_lora/                    # 源数据（人工维护，JSON 格式）
│   ├── ner_data.json            #   NER 源数据 (5,480 条)
│   └── encoding_data.json       #   编码源数据 (6,436 条)
└── qwen3_mixed/                 # 训练数据（自动生成，ChatML JSONL 格式）
    ├── train.jsonl              #   训练集 (~10,725 条)
    └── val.jsonl                #   验证集 (~1,191 条)
```

**数据流向:**

```
ner_data.json ──────┐
                    ├→ prepare_training_data.py ──→ train.jsonl / val.jsonl
encoding_data.json ─┘
```

## 快速开始

# 设置允许4个并发请求（根据你的并发数设置）
OLLAMA_NUM_PARALLEL=4 ollama serve


### 1. 环境安装

```bash
/workspace/wheels/ 文件要存在
conda create -n qwen3ft python=3.11
conda activate qwen3ft
bash apps/trainer/qwen3_finetune/setup_env.sh
```

### 2. 准备数据


```bash
# 通过映射表准备编码训练集
python -m apps.trainer.qwen3_finetune.gen_encoding_data --single-repeat 1 --multi-count 8000  --type-permute \
  --type-permute-max-tokens 4 \
  --type-permute-max-variants 4
  

# 方式 A：直接从现有源数据生成训练集（最常用）粘连数据增强概率glue_prob --ner_data
python -m apps.trainer.qwen3_finetune.prepare_training_data --augment --glue_prob 0.2

# 方式 B：只生成 NER 训练数据
python -m apps.trainer.qwen3_finetune.prepare_training_data --ner_only

# 方式 C：指定自定义路径
python -m apps.trainer.qwen3_finetune.prepare_training_data \
    --ner_data path/to/ner.json \
    --encoding_data path/to/enc.json \
    --output_dir path/to/output
```

### 3. 训练

```bash
# qwen3训练
python -m apps.trainer.qwen3_finetune.train_qwen3 --config apps/trainer/qwen3_finetune/config_qwen3.yaml
# qwen3.5训练不使用量化 BF16 LoRA方案
python -m apps.trainer.qwen3_finetune.train --no_quantize

# 覆盖参数
python -m apps.trainer.qwen3_finetune.train --epochs 5 --lr 1e-4 --lora_r 32

# 从 checkpoint 恢复
python -m apps.trainer.qwen3_finetune.train --resume_from outputs/qwen3_finetune/run_xxx/checkpoint-400
```

### 3.1 一阶段 SFT 完成后先合并 LoRA

二阶段 DPO 默认是基于**一阶段 SFT merged 模型**继续训练，因此在开始 DPO 前，先完成一次 merge：

```bash
python -m apps.trainer.qwen3_finetune.merge_and_export \
    --adapter_path outputs/qwen3_finetune_qwen3/run_xxx/final \
    --output_dir outputs/qwen3_finetune_qwen3/merged
```

例如：

```bash
python -m apps.trainer.qwen3_finetune.merge_and_export \
    --adapter_path outputs/qwen3_finetune_qwen3/run_20260318_085014/final \
    --output_dir outputs/qwen3_finetune_qwen3/merged
```

这一步的产物 `merged/` 是后续 DPO 的基座模型输入。

### 3.2 二阶段 DPO 训练

先完成一阶段 SFT，并完成上面的 merge，再进行 DPO。最小示例数据已放在 `data/pipe/dpo_demo/`。

```bash
# 先按你的实际情况修改 config_dpo.yaml 中的模型路径
# 推荐改成一阶段 SFT merged 模型路径
python -m apps.trainer.qwen3_finetune.train_dpo \
  --config apps/trainer/qwen3_finetune/config_dpo.yaml

# 覆盖关键超参
python -m apps.trainer.qwen3_finetune.train_dpo \
    --config apps/trainer/qwen3_finetune/config_dpo.yaml \
    --beta 0.2 \
    --lr 5e-6
```

示例 DPO 数据格式：

```json
{
  "input": "PIPE S31603 DN80 SCH40S",
  "chosen": {
    "TYPE": "PIPE",
    "SIZE": {"DN": ["DN80"]},
    "THICKNESS": {"SCHEDULE": ["SCH40S"]},
    "MATERIAL": "S31603"
  },
  "rejected": {
    "TYPE": "PIPE",
    "SIZE": {"DN": ["DN80"]},
    "THICKNESS": {"SCHEDULE": ["SCH40S"]},
    "MATERIAL": "S31603",
    "PRESSURE": "CL150"
  },
  "meta": {
    "task": "一阶段抽取",
    "error_type": "幻觉",
    "field": "PRESSURE",
    "source": "示例数据"
  }
}
```

### 4. 推理测试

```bash
python -m apps.trainer.qwen3_finetune.inference \
    --model_path outputs/qwen3_finetune/final \
    --text "90度弯头 DN50 SCH40 A234 WPB ASME B16.9"

python -m apps.trainer.qwen3_finetune.inference \
    --model_path outputs/qwen3_finetune/final \
    --device cpu \
    --text "90度弯头 DN50 SCH40 A234 WPB ASME B16.9"
```

### 5. 合并 LoRA 权重

这里分两种情况：

- 一阶段 SFT 训练后：把 SFT adapter 合并成 `SFT merged`
- 二阶段 DPO 训练后：把 DPO adapter 合并成 `DPO merged`

#### 5.1 合并一阶段 SFT adapter

```bash
python -m apps.trainer.qwen3_finetune.merge_and_export \
    --adapter_path outputs/qwen3_finetune/final \
    --output_dir outputs/qwen3_finetune/merged
```

python -m apps.trainer.qwen3_finetune.merge_and_export --adapter_path outputs/qwen3_finetune_qwen3/run_20260318_085014/final --output_dir outputs/qwen3_finetune_qwen3/run_20260318_085014/merged

该命令会：
- 合并 LoRA 权重到基座模型，保存到 `outputs/qwen3_finetune/merged/`
- 自动创建 `outputs/qwen3_finetune/merged/Modelfile`（用于 Ollama 部署）

#### 5.2 合并二阶段 DPO adapter

DPO 训练完成后，目录通常类似：

- `outputs/qwen3_dpo/run_xxx/final`

这时需要再次 merge，得到新的可部署模型：

```bash
python -m apps.trainer.qwen3_finetune.merge_and_export \
    --adapter_path outputs/qwen3_finetune0323/final \
    --output_dir outputs/qwen3_finetune0323/merged \
    --base_model_path outputs/qwen3_finetune0319/merged
```

后续量化、GGUF 导出、Ollama 部署，都应使用这个 **DPO merged** 目录，而不是 DPO 的 `final/` adapter 目录。
如果训练发生在云端容器、合并发生在本地，推荐始终显式传 `--base_model_path`，不要依赖 adapter 中保存的历史绝对路径。

### 6. GGUF 量化 & Ollama 部署

#### 6.1 准备 llama.cpp（仅首次）

```bash
git clone --depth 1 https://ghfast.top/https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
conda install -n qwen3ft -y cmake
cd ~/llama.cpp
conda activate qwen3ft
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-quantize -j4
pip install gguf numpy sentencepiece
```

#### 6.2 转换 & 量化

```bash
mkdir -p outputs/qwen3_finetune/merged/gguf

# HF → GGUF fp16
python ~/llama.cpp/convert_hf_to_gguf.py \
    outputs/qwen3_finetune/merged \
    --outfile outputs/qwen3_finetune/merged/gguf/model-fp16.gguf \
    --outtype f16

# fp16 → Q4_K_M/Q8_0 (7.5GB → 2.3GB)
~/llama.cpp/build/bin/llama-quantize \
    outputs/qwen3_finetune/merged/gguf/model-fp16.gguf \
    outputs/qwen3_finetune/merged/gguf/model-q8.gguf \
    Q8_0

~/llama.cpp/build/bin/llama-quantize \
    outputs/qwen3_finetune/merged/gguf/model-fp16.gguf \
    outputs/qwen3_finetune/merged/gguf/model-q4km.gguf \
    Q4_K_M
```

#### 6.3 注册到 Ollama

```bash
ollama create qwen3-4b-q8 -f outputs/qwen3_finetune/merged/Modelfile

ollama create qwen3-4b-q8 -f outputs/qwen3_finetune/merged/Modelfile
ollama create qwen3-4b-q4km -f outputs/qwen3_finetune/merged/Modelfile
ollama list  # 应看到 qwen3-pipe:latest 约 2.5GB
```

> **注意**: 每次重新训练并量化后，都需要重新执行 6.2 和 6.3 来更新 Ollama 中的模型。

### 7. 预测测试

```bash
# Ollama 后端（默认）—— 测分词
python -m apps.trainer.qwen3_finetune.predict ner \
    --text "90°长半径弯头, DN100-8.0 ,HG/T3651-2008(A系列),TA10 DN100"

# Ollama 后端 —— 测分词 + 编码
python -m apps.trainer.qwen3_finetune.predict all\
    --text "法兰盖 BL-RF 150lbs ASME-B16.5 A.240 gr 304L/A.182.F.304 DN100"

# Transformers 后端（用于对比验证量化前后效果）
python -m apps.trainer.qwen3_finetune.predict all \
    --backend transformers \
    --model_path outputs/qwen3_finetune/merged \
    --text "无缝钢管,DN80,BE,,A312-TP304,ASME B36.19M-2004(R2015),SCH40S"
```

