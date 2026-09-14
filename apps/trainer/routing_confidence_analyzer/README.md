# 分流与模型置信分分析

该脚本分析模型编码正确性、简单/困难分流、模型置信分及三者交集，不调用模型，也不修改输入 Excel。

从项目根目录执行：

```bash
python -m apps.trainer.routing_confidence_analyzer.analyze \
  --input /path/to/法兰分流运行结果.xlsx \
  --output-dir /path/to/法兰分流分析
```

覆盖已有结果：

```bash
python -m apps.trainer.routing_confidence_analyzer.analyze \
  --input /path/to/法兰分流运行结果.xlsx \
  --output-dir /path/to/法兰分流分析 \
  --overwrite
```

常用可选参数：

- `--sheet Sheet1`：指定工作表。
- `--project-column 项目简称`：覆盖项目维度列。
- `--reference-threshold 0.99`：三种策略直接对比使用的模型置信分阈值。
- `--target-accuracy 0.99 --target-accuracy 0.995`：指定自动放行准确率目标。
- `--min-project-samples 30`：项目异常判定的最少样本数。

输出目录包含：

- `分流置信分分析.xlsx`：结论、三种策略对比、项目、分流、置信分、阈值策略和明细。
- `分析结论.txt`：无需看图即可阅读的关键结论。
- 4 张核心 PNG：三种策略覆盖率/准确率、三种策略错误控制、置信分区间准确率、项目分流异常。
