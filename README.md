ElecMatCoder/
├── apps/                          # 应用程序入口
│   ├── annotation/                # 📌 标注平台
│   │   ├── server.py              # FastAPI 后端
│   │   └── frontend/              # Vue 前端
│   │       ├── src/
│   │       ├── index.html
│   │       └── package.json
│   │
│   └── training/                  # 📌 训练平台
│       ├── train.py               # 训练脚本
│       └── analyze.py             # 数据分析脚本
│
├── src/                           # 🔧 核心共享代码库
│   ├── config/                    # 配置
│   │   └── label_config.py        # 标签配置 (GB, HG, ANSI...)
│   │
│   ├── models/                    # 模型相关
│   │   ├── bert_crf.py            # BERT+CRF 模型定义
│   │   ├── trainer.py             # 训练器
│   │   ├── predictor.py           # 预测器
│   │   └── dataset.py             # 数据集处理
│   │
│   ├── tokenizers/                # 分词器
│   │   ├── jieba_tokenizer.py     # Jieba 分词
│   │   ├── llm_tokenizer.py       # LLM 分词
│   │   └── preprocessor.py        # 文本预处理
│   │
│   └── prompts/                   # LLM Prompt 模板
│       ├── cable_prompt.py
│       └── pipe_prompt.py
│
├── data/                          # 📊 数据目录
│   ├── cable/                     # 电缆平台数据
│   │   └── *.bio
│   ├── pipe/                      # 管道平台数据
│   │   └── *.bio
│   └── dicts/                     # 词典
│       ├── cable_dict.txt
│       └── pipe_dict.txt
│
├── models/                        # 🚀 部署模型 (推理用)
│   ├── cable_model/
│   └── pipe_model/
│
├── outputs/                       # 📦 训练输出 (临时)
│
├── requirements.txt
└── README.md