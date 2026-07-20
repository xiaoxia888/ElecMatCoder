# 管件 BODY 标注词表 V1

## 1. 使用原则

`BODY`只表示管件的标准产品类型。标注时先识别基础产品，再叠加会改变产品类型的结构特征，最后从本词表选择标准词，禁止自行创造新名称。

判定优先级：完整产品名称 > 公认英文名称或缩写 > 结构特征与基础产品组合 > 通用产品名称。

以下内容不进入`BODY`：材质、尺寸、壁厚、压力、标准、制造工艺，以及没有形成独立产品类型的普通连接方式。

连接方式只有在本词表已定义独立产品类型时才进入`BODY`。通常参与编码的非默认连接方式写入`CONN`；管箍名称明确包含“承口”或“螺口”时按第6节专用规则处理。`BW/对焊`是默认连接，不写入`CONN`。

## 2. 弯头与弯管

| 标准 BODY | 典型表达 | 判定条件 |
|---|---|---|
| 弯头 | ELBOW、ELB、弯头 | 未明确异径或夹套 |
| 异径弯头 | REDUCING ELBOW、异径弯头 | 明确异径 |
| BEND | BEND、Bend、弯管 | 产品明确为弯管而非弯头 |
| 夹套弯头 | JACKET ELBOW、夹套弯头 | 明确夹套结构 |

角度和半径不拼入`BODY`，分别写入`GEOMETRY.ANGLE`和`GEOMETRY.RADIUS`。

## 3. 三通与四通

| 标准 BODY | 典型表达 |
|---|---|
| 三通 | TEE、三通，未说明等径或异径 |
| 等径三通 | EQUAL TEE、STRAIGHT TEE、同径三通、等径三通 |
| 异径三通 | REDUCING TEE、RED.TEE、异径三通 |
| 斜三通 | LATERAL TEE、斜三通 |
| 异径斜三通 | REDUCING LATERAL TEE、异径斜三通 |
| Y型三通 | Y-TEE、Y型三通，未说明等径或异径 |
| Y型等径三通 | Y EQUAL TEE、Y型等径三通 |
| Y型异径三通 | Y REDUCING TEE、Y型异径三通 |
| Y型斜三通 | Y LATERAL TEE、斜Y型三通 |
| 仪表三通 | INSTRUMENT TEE、仪表三通 |
| 清管三通 | PIGGING TEE、清管三通 |
| 剖分等径三通 | SPLIT EQUAL TEE、剖切/剖分同径三通 |
| 剖分异径三通 | SPLIT REDUCING TEE、剖切/剖分异径三通 |
| 等径四通 | EQUAL CROSS、同径/等径四通 |
| 异径四通 | REDUCING CROSS、异径四通 |
| 剖分等径四通 | SPLIT EQUAL CROSS、剖切/剖分同径四通 |
| 夹套等径三通 | JACKET EQUAL TEE、夹套等径三通 |
| 夹套异径三通 | JACKET REDUCING TEE、夹套异径三通 |
| 夹套剖分等径三通 | 夹套管剖分三通、JACKET SPLIT EQUAL TEE |
| 夹套剖分异径三通 | 夹套剖分异径三通、JACKET SPLIT REDUCING TEE |
| 夹套剖分三通 | 夹套剖分三通、JACKET SPLIT TEE，未说明等径或异径 |

“剖分”必须有明确结构证据，例如`剖分、剖切、纵向剖分、SPLIT、SPLIT IN 2 HALVES、LONGITUDINALLY SPLIT`。仅有“成对包装、配对供货”不能推导为剖分。夹套、等径/异径和剖分同时出现时必须组合为对应的夹套剖分标准词，不能丢失任一结构特征。

`SW/THD/NPT`等端部方式不改变普通三通、四通的`BODY`，只写入`CONN`。

## 4. 异径管、接头与短节

| 产品族 | 未说明同偏心 | 同心 | 偏心 |
|---|---|---|---|
| 异径管 | 异径管 | 同心异径管 | 偏心异径管 |
| 异径管接头 | 异径管接头 | 同心异径管接头 | 偏心异径管接头 |
| 异径管箍 | 异径管箍 | 同心异径管箍 | 偏心异径管箍 |
| 异径短节 | 异径短节 | 同心异径短节 | 偏心异径短节 |

同心证据：`CON`作为独立缩写、`CONC`、`CONCENTRIC`、`R(C)`、`RC`、同心。

偏心证据：`ECC`、`ECCENTRIC`、`R(E)`、`RE`、偏心。

`SWAGE NIPPLE, CON`标为`同心异径短节`；`SWAGE NIPPLE, ECC`标为`偏心异径短节`；没有同偏心证据时标为`异径短节`。`SWAGE`中的`SW`不得识别为承插焊。

`异径接头`与`异径管接头`是同一种产品，统一使用`异径管接头`；`同心异径接头`统一为`同心异径管接头`；`偏心异径接头`统一为`偏心异径管接头`。

夹套同心异径管使用独立标准词`夹套同心异径管`。

## 5. 支管台

| 标准 BODY | 典型表达 | CONN约束 |
|---|---|---|
| 支管台 | OLET、支管台，未明确支管端方式 | 按原文明确信息标注 |
| 对焊支管台 | WELDOLET、WELD OLET、对焊支管台/支管座 | `[]`，BW不标注 |
| 承插焊支管台 | SOCKOLET、SOCKET OLET、SOL、承插焊支管台/支管座 | `["SW"]` |
| 螺纹支管台 | THREDOLET、THREAD OLET、TOL、螺纹支管台/支管座 | 使用THD/NPT/FNPT/MNPT等最具体值 |
| 斜支管台 | LATROLET、斜支管台 | 按原文明确信息标注 |
| 加强管嘴 | 加强管嘴、REINFORCED NOZZLE | 按原文明确信息标注 |
| 加强管接头 | 加强管接头 | 按原文明确信息标注 |
| 补强板开口焊 | 补强板开口焊 | `[]` |

`SOCKET OLET`整体表示承插焊支管台，即使没有独立`SW`也必须标为`BODY=承插焊支管台，CONN=["SW"]`。

## 6. 管箍与管接头

标准词：`管箍、FULL COUPLING、COUPLING COLLAR、单口管箍、双口管箍、单承口管箍、双承口管箍、单螺口管箍、双螺口管箍、双承口异径管箍、双螺口同径管箍、双螺口异径管箍、单头管箍、双头管箍、等径双口管箍、异径双口管箍、异径双头管箍、同径双口管箍、同心管箍、同心等径管箍、同心异径管箍、同心双口管箍、同心异径双口管箍、偏心管箍、偏心等径管箍、偏心异径管箍、偏心双口管箍、偏心异径双口管箍、管接头、半管接头、异径管接头、同心异径管接头、偏心异径管接头、快速接头、活接头、卡箍接头`。

| 别名 | 标准 BODY |
|---|---|
| COUPLING | 管箍 |
| FULL COUPLING | FULL COUPLING |
| HALF COUPLING | 半管接头 |
| QUICK RELEASE COUPLING | 快速接头 |
| UNION | 活接头 |
| COUPLING COLLAR | COUPLING COLLAR |

等径、异径、同心、偏心、单口、双口只有在原文明示时才加入`BODY`，不能仅根据尺寸数量猜测。

管箍口型遵循“原文名称优先”规则，不将承口或螺口折叠成普通单口、双口：

| 原文产品名称 | 标准 BODY | CONN |
|---|---|---|
| 单口管箍 | 单口管箍 | 按独立连接信息标注 |
| 双口管箍 | 双口管箍 | 按独立连接信息标注 |
| 单承口管箍 | 单承口管箍 | 原文明示`SW`时标注`["SW"]`，否则`[]` |
| Single Socket Pipe Coupling | 单承口管箍 | 原文明示`SW`时标注`["SW"]`，否则`[]` |
| 双承口管箍 | 双承口管箍 | `[]` |
| 单螺口管箍 | 单螺口管箍 | `[]` |
| 双螺口管箍 | 双螺口管箍 | `[]` |
| 双承口异径管箍 | 双承口异径管箍 | `[]` |
| 双螺口同径管箍 | 双螺口同径管箍 | `[]` |
| 双螺口异径管箍 | 双螺口异径管箍 | `[]` |

该规则只在产品名称本身明确出现“承口”或“螺口”时生效。承口属性进入`BODY`不代表忽略原文独立的连接信息：单承口管箍或`Single Socket Pipe Coupling`如果同时明示`SW`，`CONN`仍标注`SW`；未明示则为空。产品名称仅为单口/双口管箍时，不根据后续`SW/NPT/FNPT`反向改写`BODY`。

## 7. 短节、管帽与管塞

| 标准 BODY | 典型表达 |
|---|---|
| 短节 | NIPPLE、PIPE NIPPLE、短节 |
| 异径短节 | SWAGE NIPPLE，未说明同偏心 |
| 同心异径短节 | CONCENTRIC SWAGE NIPPLE |
| 偏心异径短节 | ECCENTRIC SWAGE NIPPLE |
| 单头螺纹短节 | THREADED ONE END NIPPLE、单头螺纹短节 |
| 双头螺纹短节 | THREADED BOTH ENDS NIPPLE、双头螺纹短节 |
| 单丝头 | 单丝头 |
| 双丝头 | 双丝头 |
| 翻边短节 | STUB END、翻边短节 |
| 管帽 | CAP、管帽 |
| 螺纹管帽 | THREADED CAP、螺纹管帽 |
| 单承口管盖 | 单承口管盖 |
| 管塞 | PLUG、管塞，未明确头型 |
| 六角头管塞 | HEX HEAD PLUG、六角头管塞 |
| 方头管塞 | SQUARE HEAD PLUG、方头管塞 |
| 圆形丝堵 | ROUND HEAD PLUG、圆形丝堵 |
| 方形丝堵 | 方形丝堵 |

产品`BODY`已归一化为“螺纹短节”、“螺纹管帽”等带泛化螺纹含义的类型时，不再重复标注`THD`；只有原文明确NPT、NPTF、FNPT、MNPT、SCRD等具体形式时才写入`CONN`。如果`BODY`本身不含螺纹语义，原文另外明示`THD/THREADED/螺纹连接`时，`CONN`标注为`THD`。

## 8. 其他管件

标准词：`插板、仪表三通、金属软管、波纹金属软管、波纹非金属软管`。

`PADDLE SPACER/BLANK AND SPACER`按当前业务词表标为`插板`。法兰本体不得进入管件训练集。

## 9. 其他字段约束

### CONN

允许值：`SW、THD、SCRD、NPT、NPTF、FNPT、MNPT、FTE、MTE、TBE、TSE`。数组按原文顺序去重；同一端同时出现泛化和具体螺纹时只保留最具体值。`BW/BE/PE/PBE`不进入当前`CONN`。

- `NPTF`保留为`NPTF`，不得转换为`FNPT`。
- 只有显式`FNPT`或`Female NPT`才标注为`FNPT`。
- 只有显式`MNPT`或`Male NPT`才标注为`MNPT`。
- 原文明示`SCRD`时保留为`SCRD`，不得转换为`THD`或`NPT`。
- 裸`NPT`标注为`NPT`；`THREADED/螺纹`等未提供具体制式的泛化表达，仅在`BODY`未表达螺纹语义时标注为`THD`。
- 原文明示`SW`、`SOCKET WELD`或“承插焊”时，无论`BODY`是否已包含承插焊含义，`CONN`都标注为`SW`。`SOCKOLET/SOCKET OLET/承插焊支管台`整体表达同样视为明确`SW`证据。
- 单独“承口”、`Single Socket Pipe Coupling`不自动推断`SW`；`SWAGE`中的`SW`也不得识别为承插焊。
- 辅助部件的连接不进入主体`CONN`，例如孔板法兰中`Pressure Taps 1/2'' SW`表示取压口连接，不代表法兰主体为`SW`。

### MANU

允许值：`SMLS、WELDED、EFW、ERW、HFW、SAW、SAWL、SAWH、DASW、DSAW、DSAWL、DSAWH`。具体焊接工艺优先于`WELDED`，不得同时输出；`SMLS`与焊接类工艺互斥。`FORGED/锻制/锻造`不属于当前`MANU`。

### GEOMETRY

`ANGLE`保留原文数值，仅去掉角度单位，不得取整；原文未出现角度时必须为空，不得根据弯头类型默认补`90`。`RADIUS`只使用原文明示的`LR、SR、1D、1.5D、3D`等值，精确倍数优先于LR/SR；原文未明示时必须为空，不得默认补`LR/5D`。

### FLANGE_STYLE

管件自身带法兰端标为`FLANGED`，管件自身为松套法兰结构标为`LAP_JOINT_FLANGED`；仅提到法兰标准或配套法兰时为空。

## 10. 固定输出骨架

```json
{
  "CATEGORY": "管件",
  "TYPE": {
    "BODY": "",
    "GEOMETRY": {
      "ANGLE": "",
      "RADIUS": ""
    },
    "FLANGE_STYLE": "",
    "MANU": [],
    "CONN": []
  }
}
```
