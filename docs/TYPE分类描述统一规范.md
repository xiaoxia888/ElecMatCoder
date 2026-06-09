# TYPE分类描述统一规范

## 1. 目的

本规范只定义一件事：

> `TYPE.BODY` 应该如何统一写法。

本规范不讨论：

- 材质
- 尺寸
- 壁厚
- 标准
- 压力等级
- 最终编码拼接

---

## 2. 当前采用的方案

当前统一采用**方案 2：结构化标注**。

`TYPE` 结构如下：

```json
"TYPE": {
  "BODY": "",
  "MANU": "",
  "CONN": "",
  "SEAL": "",
  "ENDS": ""
}
```

含义如下：

- `BODY`：种类主体名或种类主体短语
- `MANU`：制造方式
- `CONN`：连接方式
- `SEAL`：密封面
- `ENDS`：端部形式

### 关键约束

- `BODY` 不吞 `MANU`
- `BODY` 不吞 `CONN`
- `BODY` 不吞 `SEAL`
- `BODY` 不吞 `ENDS`

也就是说：

- `法兰 + SW` 不标成 `承插焊法兰`
- `法兰 + THD` 不标成 `螺纹法兰`
- `钢管 + WELDED` 不标成 `焊接钢管`
- `支管座 + SW` 不标成 `承插焊支管座`

正确做法是：

```json
"TYPE": {
  "BODY": "法兰",
  "CONN": "SW",
  "SEAL": "",
  "ENDS": "",
  "MANU": ""
}
```

---

## 3. BODY 的标注原则

### 3.1 统一为固定中文表达

`TYPE.BODY` 统一使用固定中文表达，不中英混用，不直接使用最终编码。

例如：

- `90deg elbow`
- `90° Elbow`
- `90弯头`
- `90度弯头`

统一写成：

- `90度弯头`

### 3.2 只保留主体名或主体短语

`BODY` 只保留种类主体本身。

允许保留在 `BODY` 中的，是**当前没有独立字段承接**、但又属于种类核心描述的内容，例如：

- 弯头角度
- 弯头半径
- 三通的等径/异径/斜三通
- 异径管的同心/偏心

不允许保留在 `BODY` 中的，是已有独立字段承接的信息：

- `SMLS`
- `SEAMLESS`
- `WELDED`
- `ERW`
- `EFW`
- `SAWL`
- `SAWH`
- `SW`
- `THD`
- `RF`
- `RJ`
- `RTJ`
- `MFM`
- `NPT`
- `FNPT`
- `MNPT`
- `FTE`
- `MTE`

### 3.3 允许轻度格式化

允许把中英文、数字写法、符号差异统一成固定中文表达。

例如：

- `90deg elbow` -> `90度弯头`
- `Long Radius Elbow` -> `长半径弯头`
- `Concentric Reducer` -> `同心异径管`

但不允许直接格式化成最终编码：

- `90EL`
- `FS`
- `RT`

---

## 4. 标准写法

## 4.1 弯头

### BODY 统一写法

- `30度弯头`
- `45度弯头`
- `60度弯头`
- `90度弯头`
- `180度弯头`
- `45度长半径弯头`
- `45度短半径弯头`
- `90度长半径弯头`
- `90度短半径弯头`
- `45度弯头 R=1.5D`
- `90度弯头 R=1.5D`
- `90度弯头 3D`
- `90度弯头 5D`
- `90度异径弯头`

### 示例

- `90deg elbow` -> `90度弯头`
- `90弯头` -> `90度弯头`
- `45°长半径弯头` -> `45度长半径弯头`
- `90°弯头|Elbow,LR` -> `90度长半径弯头`
- `90°弯头|Elbow,ES` -> `90度短半径弯头`

### 说明

- `SMLS / SW / THD / NPT / WELDED` 不进入 `BODY`

---

## 4.2 三通

### BODY 统一写法

- `等径三通`
- `异径三通`
- `斜三通`
- `等径斜三通`
- `异径斜三通`
- `45度等径斜三通`
- `45度异径斜三通`
- `Y型三通`
- `90度Y型三通`
- `120度Y型三通`

### 示例

- `Equal Tee` -> `等径三通`
- `Reducing Tee` -> `异径三通`
- `Lateral Tee` -> `斜三通`
- `45deg lateral tee` -> `45度斜三通`
- `90deg wye tee` -> `90度Y型三通`

---

## 4.3 异径管

### BODY 统一写法

- `同心异径管`
- `偏心异径管`

### 示例

- `Concentric Reducer` -> `同心异径管`
- `Eccentric Reducer` -> `偏心异径管`
- `同心大小头` -> `同心异径管`
- `偏心大小头` -> `偏心异径管`

---

## 4.4 法兰

### BODY 统一写法

- `法兰`
- `盲法兰`
- `八字盲板`

### 示例

- `Flange` -> `法兰`
- `Blind Flange` -> `盲法兰`
- `Spectacle Blind` -> `八字盲板`

### 特别说明

以下信息不进入 `BODY`，应进入其他字段：

- `Welding Neck` / `WN`
- `Socket Weld` / `SW`
- `Threaded` / `THD`
- `Slip-on` / `SO`
- `RF / RJ / RTJ / MFM`
- `NPT / FNPT / MNPT`

例如：

```json
"TYPE": {
  "BODY": "法兰",
  "MANU": "",
  "CONN": "SW",
  "SEAL": "RF",
  "ENDS": ""
}
```

---

## 4.5 支管座

### BODY 统一写法

- `支管座`
- `斜支管座`
- `轻型支管座`
- `Boss`

### 示例

- `Weldolet` -> `支管座`
- `Sockolet` -> `支管座`
- `Threadolet` -> `支管座`
- `Latrolet` -> `斜支管座`
- `Light Weight Olet` -> `轻型支管座`
- `Boss` -> `Boss`

### 特别说明

以下信息不进入 `BODY`：

- `BW`
- `SW`
- `THD`
- `NPT`
- `FNPT`

例如：

```json
"TYPE": {
  "BODY": "支管座",
  "MANU": "",
  "CONN": "SW",
  "SEAL": "",
  "ENDS": ""
}
```

---

## 4.6 管帽

### BODY 统一写法

- `管帽`

### 特别说明

以下信息不进入 `BODY`：

- `SW`
- `NPT`
- `FNPT`
- `FTE`
- `WELDED`

例如：

```json
"TYPE": {
  "BODY": "管帽",
  "MANU": "",
  "CONN": "SW",
  "SEAL": "",
  "ENDS": ""
}
```

---

## 4.7 管箍 / 接头 / 短节

### BODY 统一写法

- `管箍`
- `半管接头`
- `双头管箍`
- `异径管箍`
- `活接头`
- `快速接头`
- `短节`
- `螺纹短节`
- `双头短节`

### 示例

- `Coupling` -> `管箍`
- `Half Coupling` -> `半管接头`
- `Double Coupling` -> `双头管箍`
- `Reducing Coupling` -> `异径管箍`
- `Pipe Union` -> `活接头`
- `Quick Release Coupling` -> `快速接头`
- `Pipe Nipple` -> `短节`
- `Threaded One End Nipple` -> `螺纹短节`

### 特别说明

`NPT / FNPT / FTE / MTE / SW` 这类信息不进入 `BODY`。

---

## 4.8 管子

### BODY 统一写法

- `钢管`
- `法兰管`
- `夹套钢管`

### 示例

- `Pipe` -> `钢管`
- `Steel Pipe` -> `钢管`
- `Flanged Pipe` -> `法兰管`
- `Jacketed Pipe` -> `夹套钢管`

### 特别说明

以下信息不进入 `BODY`：

- `WELDED`
- `SMLS`
- `SEAMLESS`
- `ERW`
- `EFW`
- `SAWL`
- `SAWH`

例如：

```json
"TYPE": {
  "BODY": "钢管",
  "MANU": "WELDED",
  "CONN": "",
  "SEAL": "",
  "ENDS": ""
}
```

---

## 5. 禁止事项

### 5.1 不要直接用英文原文作为最终 BODY

不推荐：

- `90deg elbow`
- `Socket Olet`
- `Threaded Flange`

### 5.2 不要直接用编码作为 BODY

不推荐：

- `90EL`
- `OS`
- `FS`

### 5.3 不要把已有独立字段承接的信息再放回 BODY

不推荐：

- `承插焊法兰`
- `螺纹法兰`
- `承插焊支管座`
- `焊接钢管`

在方案 2 中，这些应拆成：

- `BODY`
- `MANU`
- `CONN`
- `SEAL`
- `ENDS`

---

## 6. 典型示例

### 示例 1

原文：

```text
90deg elbow
```

标注：

```json
"TYPE": {
  "BODY": "90度弯头",
  "MANU": "",
  "CONN": "",
  "SEAL": "",
  "ENDS": ""
}
```

### 示例 2

原文：

```text
法兰;DN10;SW
```

标注：

```json
"TYPE": {
  "BODY": "法兰",
  "MANU": "",
  "CONN": "SW",
  "SEAL": "",
  "ENDS": ""
}
```

### 示例 3

原文：

```text
螺纹法兰 NPT RF
```

标注：

```json
"TYPE": {
  "BODY": "法兰",
  "MANU": "",
  "CONN": "THD",
  "SEAL": "RF",
  "ENDS": "NPT"
}
```

### 示例 4

原文：

```text
承插焊支管座
```

标注：

```json
"TYPE": {
  "BODY": "支管座",
  "MANU": "",
  "CONN": "SW",
  "SEAL": "",
  "ENDS": ""
}
```

### 示例 5

原文：

```text
Welded Pipe
```

标注：

```json
"TYPE": {
  "BODY": "钢管",
  "MANU": "WELDED",
  "CONN": "",
  "SEAL": "",
  "ENDS": ""
}
```
