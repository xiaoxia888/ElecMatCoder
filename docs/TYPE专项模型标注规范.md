# TYPE专项模型标注规范

## 1. 目标

本规范用于构建 `TYPE` 相关标注数据。

当前统一采用**方案 2：结构化标注**。

核心思路：

1. `mentions`：保留原文证据
2. `semantics`：标注语义角色
3. `decisions`：输出结构化 `TYPE` 结果

核心原则：

- `mentions` 保留原文，不做归一化
- `semantics` 只标角色，不做编码决策
- `decisions` 做字段级结构化输出
- `BODY` 不吞 `MANU / CONN / SEAL / ENDS`
- 只认显式证据，不做推断，不做脑补

---

## 2. 适用范围

当前 `TYPE` 标注只关注与类型相关的信息。

纳入范围：

- 管子主体
- 管件主体
- 法兰主体
- 显式制造方式
- 显式连接方式
- 显式密封面
- 显式端部形式

不纳入当前 `TYPE` 标注范围：

- `SIZE`
- `THICKNESS`
- `MATERIAL`
- `STANDARD`
- `PRESSURE`

如果某条样本同时包含这些信息，`TYPE` 标注时忽略它们。

---

## 3. 输出结构

统一输出结构如下：

```json
{
  "mentions": [
    {
      "id": "m1",
      "text": "",
      "type": "TYPE_TERM"
    }
  ],
  "semantics": [
    {
      "mention_id": "m1",
      "semantic_tag": ""
    }
  ],
  "decisions": {
    "TYPE": {
      "BODY": "",
      "MANU": "",
      "CONN": "",
      "SEAL": "",
      "ENDS": ""
    }
  }
}
```

说明：

- `mentions` 可以有多个
- `TYPE` 相关 `mentions.type` 当前统一为 `TYPE_TERM`
- `decisions.TYPE` 是唯一的结构化输出

---

## 4. 三层含义

### 4.1 `mentions`

`mentions` 是原文中的直接证据片段。

要求：

- 尽量贴近原文连续片段
- 不要提前做归一化
- 一个 `mention` 尽量只承接一个主要语义角色
- 能稳定拆开的复合词，优先拆开

示例：

原文：

```text
焊接钢管BE,T=10mm DN1400,GB/T3091(Ⅱ)，Q235B
```

推荐：

```json
[
  {"id": "m1", "text": "焊接", "type": "TYPE_TERM"},
  {"id": "m2", "text": "钢管", "type": "TYPE_TERM"}
]
```

---

### 4.2 `semantics`

`semantics` 用来说明每个 `mention` 的语义角色。

当前允许值：

- `TYPE_BODY`
- `TYPE_MANU`
- `TYPE_CONN`
- `TYPE_SEAL`
- `TYPE_ENDS`

说明：

- `TYPE_BODY`：主体种类
- `TYPE_MANU`：显式制造方式
- `TYPE_CONN`：显式连接方式
- `TYPE_SEAL`：显式密封面
- `TYPE_ENDS`：显式端部形式

---

### 4.3 `decisions`

`decisions` 是最终给下游编码使用的结构化结果。

规则：

- `BODY` 统一成固定中文主体表达
- `MANU / CONN / SEAL / ENDS` 使用统一规范值
- `BODY` 不吞 `MANU / CONN / SEAL / ENDS`
- 无可靠证据时留空

---

## 5. `decisions.TYPE` 字段定义

### 5.1 `BODY`

`BODY` 只放统一后的主体中文表达。

规则：

- `BODY` 保留主体名或主体短语
- `BODY` 允许保留当前没有单独字段承接的种类核心描述，例如：
  - 弯头角度
  - 弯头半径
  - 三通的等径/异径/斜三通
  - 异径管的同心/偏心
- `BODY` 不保留已有独立字段承接的信息

推荐示例：

- `90deg elbow` -> `BODY = 90度弯头`
- `90° Elbow LR` -> `BODY = 90度长半径弯头`
- `Reducing Tee` -> `BODY = 异径三通`
- `Concentric Reducer` -> `BODY = 同心异径管`
- `Flange` -> `BODY = 法兰`
- `Socket Olet` -> `BODY = 支管座`
- `Pipe` -> `BODY = 钢管`

不推荐：

- `BODY = 承插焊法兰`
- `BODY = 螺纹法兰`
- `BODY = 承插焊支管座`
- `BODY = 焊接钢管`

这些应拆到：

- `BODY + CONN`
- `BODY + MANU`

### 5.2 `MANU`

`MANU` 用于承接显式制造方式。

推荐值：

- `SMLS`
- `SEAMLESS`
- `WELDED`
- `ERW`
- `EFW`
- `SAWL`
- `SAWH`
- `FORGED`

规则：

- 只认显式证据
- `焊接钢管` 类场景，标为：
  - `BODY = 钢管`
  - `MANU = WELDED`
- `无缝钢管` 类场景，标为：
  - `BODY = 钢管`
  - `MANU = SMLS` 或 `SEAMLESS`

### 5.3 `CONN`

`CONN` 用于承接显式连接方式。

推荐值：

- `SW`
- `THD`

规则：

- `BW` 当前不参与编码时，不标
- `承插焊法兰` 类场景，标为：
  - `BODY = 法兰`
  - `CONN = SW`
- `螺纹法兰` 类场景，标为：
  - `BODY = 法兰`
  - `CONN = THD`
- `承插焊支管座` 类场景，标为：
  - `BODY = 支管座`
  - `CONN = SW`
- `螺纹支管座` 类场景，标为：
  - `BODY = 支管座`
  - `CONN = THD`

### 5.4 `SEAL`

`SEAL` 用于承接密封面。

推荐值：

- `RF`
- `FF`
- `RJ`
- `RTJ`
- `MFM`

### 5.5 `ENDS`

`ENDS` 用于承接端部形式。

推荐值：

- `NPT`
- `FNPT`
- `MNPT`
- `FTE`
- `MTE`
- `BLEXTSE`

规则：

- `NPT / FNPT / MNPT` 不与 `THD` 合并
- `THD` 是连接方式
- `NPT / FNPT / MNPT` 是端部形式

---

## 6. 示例

### 6.1 `法兰;DN10;SW`

```json
{
  "mentions": [
    {"id": "m1", "text": "法兰", "type": "TYPE_TERM"},
    {"id": "m2", "text": "SW", "type": "TYPE_TERM"}
  ],
  "semantics": [
    {"mention_id": "m1", "semantic_tag": "TYPE_BODY"},
    {"mention_id": "m2", "semantic_tag": "TYPE_CONN"}
  ],
  "decisions": {
    "TYPE": {
      "BODY": "法兰",
      "MANU": "",
      "CONN": "SW",
      "SEAL": "",
      "ENDS": ""
    }
  }
}
```

### 6.2 `螺纹法兰 NPT RF`

```json
{
  "mentions": [
    {"id": "m1", "text": "螺纹法兰", "type": "TYPE_TERM"},
    {"id": "m2", "text": "NPT", "type": "TYPE_TERM"},
    {"id": "m3", "text": "RF", "type": "TYPE_TERM"}
  ],
  "semantics": [
    {"mention_id": "m1", "semantic_tag": "TYPE_BODY"},
    {"mention_id": "m1", "semantic_tag": "TYPE_CONN"},
    {"mention_id": "m2", "semantic_tag": "TYPE_ENDS"},
    {"mention_id": "m3", "semantic_tag": "TYPE_SEAL"}
  ],
  "decisions": {
    "TYPE": {
      "BODY": "法兰",
      "MANU": "",
      "CONN": "THD",
      "SEAL": "RF",
      "ENDS": "NPT"
    }
  }
}
```

### 6.3 `承插焊支管座`

```json
{
  "mentions": [
    {"id": "m1", "text": "承插焊", "type": "TYPE_TERM"},
    {"id": "m2", "text": "支管座", "type": "TYPE_TERM"}
  ],
  "semantics": [
    {"mention_id": "m1", "semantic_tag": "TYPE_CONN"},
    {"mention_id": "m2", "semantic_tag": "TYPE_BODY"}
  ],
  "decisions": {
    "TYPE": {
      "BODY": "支管座",
      "MANU": "",
      "CONN": "SW",
      "SEAL": "",
      "ENDS": ""
    }
  }
}
```

### 6.4 `Welded Pipe`

```json
{
  "mentions": [
    {"id": "m1", "text": "Welded", "type": "TYPE_TERM"},
    {"id": "m2", "text": "Pipe", "type": "TYPE_TERM"}
  ],
  "semantics": [
    {"mention_id": "m1", "semantic_tag": "TYPE_MANU"},
    {"mention_id": "m2", "semantic_tag": "TYPE_BODY"}
  ],
  "decisions": {
    "TYPE": {
      "BODY": "钢管",
      "MANU": "WELDED",
      "CONN": "",
      "SEAL": "",
      "ENDS": ""
    }
  }
}
```

### 6.5 `90deg elbow`

```json
{
  "mentions": [
    {"id": "m1", "text": "90deg elbow", "type": "TYPE_TERM"}
  ],
  "semantics": [
    {"mention_id": "m1", "semantic_tag": "TYPE_BODY"}
  ],
  "decisions": {
    "TYPE": {
      "BODY": "90度弯头",
      "MANU": "",
      "CONN": "",
      "SEAL": "",
      "ENDS": ""
    }
  }
}
```

---

## 7. 总规则

### 7.1 只认显式证据

没有原文证据，不补。

### 7.2 `BODY` 与其他字段分层

- `BODY` 负责主体名/主体短语
- `MANU` 负责制造方式
- `CONN` 负责连接方式
- `SEAL` 负责密封面
- `ENDS` 负责端部形式

### 7.3 编码层再组合

不要在标注层提前把：

- `法兰 + SW`

直接写成：

- `承插焊法兰`

不要在标注层提前把：

- `钢管 + WELDED`

直接写成：

- `焊接钢管`

这些组合关系留给后续编码层处理。
