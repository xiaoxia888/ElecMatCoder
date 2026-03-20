# NER 数据集 TYPE 标注规范

## 1. 提取什么

从原文中提取所有描述**"这是什么东西"**的词：

- **主类名**：弯头、ELBOW、三通、TEE、法兰、FLANGE、异径管、REDUCER、管帽、CAP、PIPE、管子、OLET、管接台 等
- **角度**：90°、45度、A-90、ANGLE=45 DEG、45E(L) 等
- **半径类型**：LR、SR、R 1.5D、R=3D、Bending radius 3D、长半径、短半径 等
- **形状/子类型修饰**：同心、偏心、等径、异径、Reducing、Lateral、HEX HEAD 等

## 2. 不提取什么

以下信息属于其他标签，不放入 TYPE：

| 标签 | 不放入 TYPE 的词 |
|------|----------------|
| MANU | SMLS、WELDED、EFW、ERW、FORGED、锻制、无缝、焊接 |
| CONN | SW、THD、承插焊 |
| ENDS | NPT、FNPT、MNPT、SO、BE、PE |
| SEAL | RF、RTJ、RJ、FF |
| 其他 | SIZE、MATERIAL、STANDARD、PRESSURE、THICKNESS 相关内容 |

**例外**：当上述词是产品名的一部分时保留（如 `WELDOLET`、`BW OLET`、`WELDED PIPE`、`碳钢法兰`、`FLANGE SO`）。判断标准：去掉该词后产品名不完整或含义改变。

## 3. 格式规则

1. **原样提取**：不加等号、不改大小写、不翻译、不归一化（原文 `R 1.5D` 就写 `R 1.5D`，不写 `R=1.5D`；原文 `45E(L)` 就写 `45E(L)`，不写 `45度弯头`）
2. **空格拼接**：非连续的 TYPE 词之间用空格连接
3. **原文顺序**：按在原文中出现的先后顺序排列
4. **中英文只留一个**：同一产品的中英文名只保留先出现的（`偏心异径管|Ecc.Reducer` → `偏心异径管`）
5. **不用特殊分隔符**：不使用 `|` 或 `;`

## 4. 与关联标签的边界

- **MANU**：独立出现的制造方式归 MANU；作为产品名一部分时归 TYPE（如 `WELDOLET`、`无缝碳钢管`）
- **CONN**：独立出现的连接方式归 CONN；`BW OLET` 中的 BW 归 TYPE（产品名组成部分）
- **ENDS**：独立出现的端部形式归 ENDS；`FLANGE SO`、`FLANGE MNPT` 中作为法兰子类型标识时归 TYPE
- **SEAL**：密封面信息始终归 SEAL，不放入 TYPE
