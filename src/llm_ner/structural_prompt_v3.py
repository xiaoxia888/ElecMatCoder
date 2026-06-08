# -*- coding: utf-8 -*-
"""Third-version prompts for prompt-based structural extraction.

说明：
- 本文件不修改现有 `structural_prompt.py`。
- 提示词内容仅以 `/Users/guoxi/Downloads/first_stage_prompt_templates.py`
  中 `specification / wall_thickness / pressure_rating` 三段规则为基础。
- 唯一额外处理：把模型输出格式适配为当前平台需要的结构：
  - SIZE_ITEMS / LENGTH
  - THICKNESS_ITEMS
  - PRESSURE
"""

COMMON_RULES_TEXT_V3 = """
**解析规则**:
0. **批量隔离硬约束（最高优先级）**：
   - `results` 中每个对象必须仅依据其对应 `material_id` 的“材料描述”独立生成，所有字段（`name/specification/wall_thickness/pressure_rating/material/standard`）严禁跨对象复制、借用、补全或对齐。
   - 即使同批材料描述高度相似，也不得复用其他 `material_id` 的任一字段值；无论单条或批量解析，若当前材料描述对某字段证据不足，相关字段必须留空，不得猜测、借值或跨字段补全。
通用边界硬约束：只提取本次要求的字段，其他标签信息只能作为排除依据，不得写入本字段；证据不足必须留空，不得猜测或跨材料借值。
""".strip()


SIZE_FIELD_RULES_TEXT_V3 = """
2. **规格**: 提取尺寸规格信息
   - 公称直径：DN40、DN100、DN150等
   - **异径规格**：DN300xDN200、250x40等（必须保留两个尺寸）
   - **数字-SCH 规格硬约束**：若出现 `数字-SCH...`、`数字-STD`、`数字-XS`、`数字-XXS`、`数字-S40S` 等“数字-壁厚等级”结构，其中前面的数字视为公称直径，必须提取到 `specification` 并标准化为 `DN数字`；后面的 `SCH/STD/XS/XXS/S40S` 等写入 `wall_thickness`；该前置数字绝不能写入 `material`。例如：`20-SCH40S` → `specification=DN20`，`wall_thickness=SCH40S`
   - 螺栓规格：M20x100、M16x90等
   - **重要**：对于异径部件，必须提取两个尺寸，格式为"左x右"，并且**严格按原描述中出现的顺序输出（左到右）**，不要根据大小做排序/纠正
   - **强制约束**：φ60.3Xφ48.3必须提取为"60.3x48.3"，不能只提取一个尺寸
   - **🚨 关键识别规则**：
     * **单一尺寸格式**：DN250X7mmDN250 → 规格=DN250（只提取第一个DN值）
     * **双尺寸格式**：DN300xDN200 → 规格=DN300xDN200（保留两个尺寸）
     * **🔥 简化异径格式**：DN50X20 → 规格=DN50x20（X后纯数字且小于DN值时为异径）
     * **🔥 完整异径格式**：DN50XDN20 → 规格=DN50x20（标准化为小写x连接）
     * **壁厚条件拆分**：当命中 `DNxxxXyyyMM`（左侧为单一主尺寸，右侧为单一毫米壁厚）时，必须拆分为 `specification=DNxxx`、`wall_thickness=yyyMM`；但若命中独立壁厚对 `aXbmm/a×bmm/a*bmm` 且存在异径证据（如 `DN大x小`、`REDUCING`、`异径`），不得按单值拆分，必须输出双值壁厚 `aMMXbMM`。
     * **连接符判断**：X（大写）仅当右侧带壁厚标记（mm/THK/壁厚/SCH/STD/SxxS）时，才视为尺寸×壁厚；否则优先当异径规格；x（小写）通常连接两个尺寸
     * **🚨 异径判断逻辑**：DN后面紧跟X+数字时，如果该数字<DN值且无mm单位，则为异径规格
     * **重复DN处理**：DN250X7mmDN250中，第二个DN250是冗余信息，规格只取DN250
     * **🔥 规格优先级（硬约束）**：规格输出必须按证据类型分流处理。若同一主规格为**纯DN证据**（如 `DN100`、`DN300xDN200`、`DN300×DN200`、`DN300XDN200`），`specification` 必须输出对应 `DN...`（含异径必须保留两端）；若同一主规格为**纯英寸/NPS证据**（如 `6"`、`3/4"`、`NPS6xNPS3/4`），`specification` 必须包含 `NPS` 前缀并输出为 `NPS...`，禁止输出不含英寸证据的纯数字/纯小数（如 `6x0.75`）；若同一主规格为**混合单位证据**（同时出现 `DN` 与英寸/NPS），则按“混合单位顺序与冲突仲裁硬约束”输出为 `NPS...xDN...` 或 `DN...xNPS...`。第一阶段禁止直接把英寸换算为 `DN`；仅当 `DN` 与英寸/NPS 证据都不存在时，才允许使用 `φ/Φ/OD/外径` 等尺寸兜底。
     * **英寸顺序（硬约束）**：当 `specification` 输出为 `NPS...xNPS...` 时，必须**严格按原描述出现顺序输出**，不要把 `2"x3/4"` 改成 `3/4"x2"`，也不要把 `3/4"x2"` 改成 `2"x3/4"`；例如 `3 in x 2 in` 必须输出为 `NPS3xNPS2`。
     * **混合单位顺序与冲突仲裁硬约束**：当同一主规格同时出现 `DN` 与英寸/NPS 证据时，`specification` 在第一阶段仅允许做连接符与英寸写法规一，连接符统一为小写 `x`，英寸侧统一规范为 `NPS` 前缀、`DN` 侧保留 `DN` 前缀，且必须严格保持原文左右顺序，禁止重排、交换或在第一阶段直接做英寸→DN换算；若同句同时存在多组尺寸证据（如 `3"*2.5"` 与 `3"*DN65`/`DN65x3"`）且共享同一端尺寸，则必须以“包含DN的那一对”为唯一主规格，仅基于该对输出 `specification`（如 `3"*DN65`→`NPS3xDN65`，`DN65x3"`→`DN65xNPS3`），其余尺寸片段仅作辅证，不得参与主规格改写或跨对拼接生成新规格。
     * **🚫 DN覆盖**：即便描述里同步出现了 `114.3x60.3` 等外径或“OD=114.3mm”这类片段，只要原文包含 `DN100x50`、`DN300` 等字样，`specification` 必须直接写对应的 `DN`，外径只能在完全找不到 `DN` 时兜底
     * **显式长度强制追加**：只要原文出现明确长度字段 `L=...`、`L:...`、`LG=...`、`LENGTH=...`、`LEN=...`、`长度...`、`总长...`，且同条存在明确主规格 `DN...` 或 `NPS...`，必须将长度追加到 `specification`，格式为 `DN200L3000` 或 `NPS2L3000`；长度单位为 `m/米` 时换算为毫米（如 `L=3m` → `L3000`），单位为 `mm/毫米` 时直接取数值，未写单位时保留数值本身；不得把长度写入 `wall_thickness` 或 `material`。示例：`灰口铸铁, 壁厚5.8mm, L=3m, DN200` → `specification=DN200L3000`。
     * **管子CUT TO长度特例（窄规则）**：当原文出现 `CUT TO`/`CUT-TO` + 数值，且同句存在明确主管规格（`DN...` 或 `NPS...`）时，允许将该数值作为长度追加到 `specification`（`L{数值}`）；不满足上述条件时不得追加，避免泛化到普通管子场景。示例：`PIPE ... CUT TO 1505 DN100` → `specification=DN100L1505`。
     * **PN/尺寸紧邻拆分**：当描述中出现 `PNxx` 后紧邻公称直径数字的结构时，包括 `PNxx-数字`、`PNxx 数字`、`PNxx/DN数字`、`PNxx-DN数字`、`PNxx-数字I/II/III`、`PNxx 数字Ⅰ/Ⅱ/Ⅲ`、`PN16-125/END` 等，若 `PNxx` 为标准公称压力等级写法（如 `PN10/PN16/PN25/PN40` 等），则 `PNxx` 必须写入 `pressure_rating`；其后的数字若为常见公称直径序列值（如 `15/20/25/32/40/50/65/80/100/125/150/...`），必须加 `DN` 补全并写入 `specification`（如 `DN25`、`DN125`）；数字后的 `/END` 不得并入 `pressure_rating`。数字后的 `I/II/III/IV`、`Ⅰ/Ⅱ/Ⅲ/Ⅳ` 若不是材质的一部分，且描述中存在明确规范，则应作为离它最近的规范后缀写入 `standard`，不得并入 `pressure_rating` 或导致 `specification` 为空；例如 `PN16 25Ⅱ;20;GB/T12459-2017` 应输出 `pressure_rating=PN16`、`specification=DN25`、`material=20`、`standard=GB/T12459-2017Ⅱ` 或等价后缀形态。普通独立压力组合如 `PN16/PN25`、`PN16 PN25` 仍整体视为压力信息，不得拆成尺寸。
     * **夹套双规格硬约束**：当夹套件描述中出现以 `/` 连接的两组尺寸（如 `DN150x125/DN200x150`）时，必须将两组尺寸按原顺序完整保留到 `specification`，因为 `/` 两侧通常代表外管/内管或主管/夹套两组有效规格，禁止截断为前半段单组尺寸。
""".strip()


THICKNESS_FIELD_RULES_TEXT_V3 = """
3. **壁厚**: 提取壁厚规格
   - **壁厚判定顺序（硬约束）**：
     1) 先判断当前描述是单主件还是多主件；若有多个主件，必须先按主件切分各自的尺寸与壁厚证据，禁止把前一主件的壁厚误并到后一主件。
     2) 再判断当前主件是否命中“双壁厚/组合壁厚触发条件”。仅以下情况允许输出双壁厚或组合壁厚：
        * 异径件明确出现两端壁厚（如 `SCH40XSCH80`、`5MMX4.5MM`）
        * 夹套件明确出现两组壁厚（如 `6.3x6.3/7.1x6.3`）
        * 原文明确出现混合壁厚表达（如 `12mm*SCH40S`、`SCH40X5mm`、`32.0mmxS-60`、`18.0mmxS-80`）
        * 多主件描述中，不同主件各自拥有独立壁厚证据
     3) 若未命中上述任一双壁厚/组合壁厚/混合壁厚条件，则按“单壁厚”处理；此时当 `mm` 壁厚与 `SCH/STD/XS/XXS/SxxS` 等级壁厚同时出现时，`wall_thickness` 必须优先输出 `mm` 壁厚。
     4) 只有在原文明确支持双壁厚/组合壁厚时，才允许输出双值结果；不得因为同句同时出现多个壁厚词，就机械拼接为双壁厚。
   - **壁厚类型与优先级**：若原文出现 `THK=`、`壁厚=`、`厚度=`，或出现 `φ/Φ/OD/O.D./外径/Ф/Ø + 外径×厚度` 结构（如 `φ139.7×3.6`、`OD114.3x6.02`、`外径60.3×3.8`），则 `wall_thickness` 优先输出毫米壁厚（单值如 `3.5MM`，双值如 `5MMX4.5MM`）；若原文明确为“毫米壁厚 + 等级壁厚”混合表达（如 `12mm*SCH40S`、`SCH40X5mm`、`32.0mmxS-60`、`18.0mmxS-80`），则保留混合壁厚；仅在原文不存在可提取毫米壁厚时，才使用等级壁厚（如 `SCH160`、`S40S`、`STD`、`XS`、`XXS`）；若原文同时出现 `外径×毫米厚度` 与 `SCH/STD/...` 且并非明确混合壁厚表达，则 `wall_thickness` 必须输出毫米壁厚，不得仅输出 `SCH10S/S40S/STD` 等等级壁厚。
   - **单壁厚优先级（硬约束）**：对于单主件、非异径、非夹套、非明确混合壁厚表达的描述，若同时出现毫米壁厚与等级壁厚，`wall_thickness` 必须优先输出毫米壁厚（如 `THK=6.3mm`、`12mm`、`3.5mm`），不得仅输出 `SCH40`、`STD`、`XS`、`S40S` 等等级壁厚；仅当原文不存在可提取的毫米壁厚时，才允许输出等级壁厚。
   - **异径双毫米硬约束**：当出现 `aXbmm`、`a×bmm`、`a*bmm` 且不属于外径语境（非 `φ/OD/外径/尺寸` 前缀）时，`wall_thickness` 必须输出为 `aMMXbMM`；若同句存在异径证据（`DN大x小`、`REDUCING`、`异径`），严禁退化为单值 `bMM` 或 `aMM`。
   - **GB/T12459（DN+罗马后缀）**：出现 `DNxxⅡ-n/II-n` 时，输出 `specification=DNxx`、`wall_thickness=nMM`，并将 `Ⅱ/II` 并入最近的 `GB/T12459II`；出现 `DNxxXyyⅡ-nXm` 时，输出 `specification=DNxxXyy`、`wall_thickness=nMMXmMM`，不得只保留后半壁厚。
     示例：`异径管 R(E)，DN80x50Ⅱ-4x3.5，20，GB/T12459-2017` → `specification=DN80x50`, `wall_thickness=4MMX3.5MM`, `standard=GB/T12459II`。
   - **🔥 异径复合优先**：同句出现等级壁厚+毫米壁厚时，若规格为异径（DN大x小）则必须保留组合壁厚（按原出现顺序，如 `12MMXSCH40S`）；非异径则优先保留毫米壁厚（如 `12MM`），等级壁厚（如 `SCH80`）仅作次选，不得默认优先。
   - **异径详细尺寸串（硬约束）**：当描述出现 `外径x壁厚/壁厚 - 外径x壁厚` 或预清洗后的 `DNx壁厚/壁厚 - DNx壁厚` 结构（如 `273x5.0/3.2 - 219.1x2.9`、`DN80x5.6/3.2 - DN50x4.5`）时，必须按原文顺序保留三个壁厚值，`wall_thickness` 输出为 `大端第一壁厚X大端第二壁厚X小端壁厚MM`（示例必须输出 `5MMX3.2MMX2.9MM`、`5.6MMX3.2MMX4.5MM`），禁止截断为 `5MMX2.9MM`、`5.6MMX3.2MM` 或仅保留前两段。
   - **mm 与 SCH 消歧**：`mm` 与 `SCH/S-数字/S数字/STD/XS/XXS/SxxS` 由 `* / x / X / ×` 连接时默认视为复合壁厚；仅当 `mm` 前有明确尺寸标记（`φ`/`OD=`/“外径”/“尺寸”）才当尺寸。
   - **组合壁厚保留**：`SCH40XSCH80`、`STDxXS`、`S40xS80`、`12mm*SCH40S`、`S-40S X S-160` 等组合壁厚必须完整保留整段，连接符 `x/X/×/*` 等价；当异径件或 OLET/支管台类描述中连续出现两个等级壁厚词（如 `SCH 20 SCH 40`、`SCH40 SCH80`、`STD SCH40`）时，也视为两端组合壁厚，必须按原文顺序输出为 `SCH20XSCH40`、`SCH40XSCH80`、`STDXSCH40`，不得只保留后一个或较大的一个。解析时必须先识别完整壁厚词，再处理连接符，禁止按单个字符机械切分。
   - **S-STD 消歧（硬约束）**：`S-STD`、`S-XS`、`S-XXS` 均为完整壁厚词；若原文出现 `S-STDXS-STD`，必须按 `S-STD X S-STD` 解析，其中中间 `X` 仅表示两段连接，禁止将其误拆为 `STDxXS` 或 `STD + XS`。示例：`S-STDXS-STD` → `STDXSTD`。
   - **尾部S保留与S等级壁厚**：`10S/20S/40S/80S`、`S10S/S20S/S40S/S80S`、`SCH10S/SCH20S/SCH40S/SCH80S` 中的尾部 `S` 都是壁厚等级的一部分，必须保留；组合壁厚中每一段都必须逐段保留尾部 `S`，连接符 `x/X/×/*` 仅表示两端连接，不得吞掉连接符前后任一段的尾部 `S`。例如 `S10S x S40S`、`S10SXS40S` 必须输出 `S10SXS40S`，`SCH40SXSCH40S` 必须输出 `SCH40SXSCH40S`；禁止简化为 `S10XS40S`、`S10SXS40`、`SCH40XSCH40S` 或 `SCH40SXSCH40`。原文出现独立 `S-5/S-10/S-20/S-30/S-40/S-80/S-160` 或 `S5/S10/S20/S30/S40/S80/S160` 这类不带尾部 `S` 的等级壁厚时，标准化写入 `wall_thickness` 为 `S5/S10/S20/S30/S40/S80/S160`，不得写入 `material`。
   - **禁止提取衬层厚度当作壁厚。
   - **夹套双壁厚硬约束**：当夹套件描述中出现以 `/` 连接的两组壁厚（如 `6.3x6.3/7.1x6.3`）时，必须将两组壁厚按原顺序完整保留到 `wall_thickness`，禁止截断为前半段单组壁厚。
""".strip()


PRESSURE_FIELD_RULES_TEXT_V3 = """
5. **磅级**: 提取压力等级信息
   - 仅当描述出现标准等级写法时才填写，如 `PNxx`（PN10/PN16/PN40/PN63等）、`CL150/CL300/CL600/CL900/CL1500/CL2500/CL3000`/`Class 150/300/600/900/1500/2500/3000`/`C150/C300/C600/C900/C1500/C2500/C3000`（含 `CL 300`、`CL.3000`、`PN4O` 等变体）、`xxxLB`/`xxx#`（含 `300Lb`、`600 Lbs`、`3000#`）等常见格式；若 `CL/CLASS + 数字` 紧跟在明确材质牌号后，视为材质等级，禁止写入 `pressure_rating`，不受数字大小限制
   - **多磅级保留**：当原文明确出现多个压力等级由 `/`、`;`、`,`、`或` 连接时，必须按原文顺序完整保留到 `pressure_rating`，不得只取第一个。例如 `PN16/PN10` 必须输出 `PN16/PN10`。
   - **禁止裸数字脑补磅级**：不得仅凭尺寸/型号/标准号/壁厚/材质中的数字推断 `CL/C/Class`，如 `DN150`、`BL150-16`、`Buff#300`、`Φ168.3` 均不能作为 `CL150/C150/CL300/C300` 证据；只有原文明确出现 `CL/CLASS/C/LB/LBS/数字#` 时，才可输出 `CL/C` 类磅级。
   - **#号方向限制**：只有 `数字#`（如 `300#`、`3000#`）可视为磅级；`#数字` 默认不是磅级，`Buff#300/BUFF #300/抛光#300` 等表面处理等级必须忽略。
   - **Class 数值边界（硬约束）**：只有 `Class`/`CL` 后面紧跟阿拉伯数字压力等级时，才允许写入 `pressure_rating`（如 `Class 150`、`Class3000`、`CL600`）；凡 `Class`/`CL` 后面是罗马数字、字母或其组合（如 `Class I`、`Class Ia`、`CL II`、`Class Ib`），一律不得写入 `pressure_rating`。
   - **Class 等级排除**：当描述中出现 `Class I/II/III/IV/V`、`Class Ia/Ib` 或 `CL I/II/III/IV/V` 这类非数值等级，且未与明确压力数值（如 `Class 150/300/600/1500/2500/3000`、`CL150/CL300/CL600`）组合出现时，不得按常规 `pressure_rating` 提取。
   - 法兰型号后缀识别：当描述中出现 `WN/SO/BL/PL + 规格 + (类型) - 数字 + RF/FF/RTJ` 这类格式时，`-数字` 视为公称压力等级 `PN数字`（如 `WN350(B)-25 RF` → pressure_rating=PN25），不得漏掉；前一段数字是规格/尺寸，不得当作 `CL/C/Class`，如 `BL150-16 RF` → `specification=DN150, pressure_rating=PN16`。
   - 与明确工况语句（如“设计压力/工作压力/试验压力”）绑定的数值一律视为工况，不写入 `pressure_rating`（例如 `设计压力：FV/6Bar`）；若描述中出现独立压力等级写法 `x.xMPa`（如 `1.0MPa`、`1.6MPa`、`2.5MPa`），可先提取到 `pressure_rating`。
""".strip()


SIZE_PLATFORM_ADAPTER_TEXT_V3 = """
平台输出适配要求（只改输出格式，不改上述判定规则）：
- 先按上面的 `specification` 规则完成判断，再把结果转换为当前平台结构。
- 当前平台不接收 `specification` 字段，只接收：
  {
    "SIZE_ITEMS": [
      {"type": "DN|OD|INCH", "value": "..."}
    ],
    "LENGTH": ""
  }
- 规格若为单值 `DN100`，输出一个 item：`{"type":"DN","value":"DN100"}`
- 规格若为 `DN300xDN200`，拆成两个 item，顺序必须与原文一致：
  `{"type":"DN","value":"DN300"}`、`{"type":"DN","value":"DN200"}`
- 规格若为 `DN50x20`，拆成两个 `DN` item，第二段补成 `DN20`
- 规格若为 `NPS3xDN65`，拆成 `INCH` 与 `DN` 两个 item，顺序与原文一致
- 规格若为 `DN65xNPS3`，拆成 `DN` 与 `INCH` 两个 item，顺序与原文一致
- 规格若来自 `φ/Φ/OD/外径` 兜底，则对应输出 `OD` item，值写成 `OD60.3` 这种形式
- 若存在长度（例如模板中的 `DN200L3000` / `NPS2L3000`），不要把 `L3000` 拼进尺寸 item；尺寸仍按上面规则拆分，长度单独写到顶层 `LENGTH="3000"`
- 若没有长度，`LENGTH=""`
- 最终只输出 JSON，不要解释。
""".strip()


THICKNESS_PLATFORM_ADAPTER_TEXT_V3 = """
平台输出适配要求（只改输出格式，不改上述判定规则）：
- 先按上面的 `wall_thickness` 规则完成判断，再把结果转换为当前平台结构。
- 当前平台不接收 `wall_thickness` 字段，只接收：
  {
    "THICKNESS_ITEMS": [
      {"type": "MM|SCHEDULE|BWG|INCH", "value": "..."}
    ]
  }
- 如果最终壁厚结果是毫米壁厚或包含毫米主值的复合壁厚（如 `3.5MM`、`4MMX3.5MM`、`12MMXSCH40S`、`5MMX3.2MMX2.9MM`），输出为 `type="MM"`，`value` 保留完整最终字符串
- 如果最终壁厚结果是纯等级壁厚（如 `SCH40S`、`SCH40XSCH80`、`STD`、`STDXSTD`、`S10SXS40S`），输出为 `type="SCHEDULE"`，`value` 保留完整最终字符串
- 如果最终壁厚结果是 `BWG` 体系，输出为 `type="BWG"`
- 如果最终壁厚结果是英寸壁厚，输出为 `type="INCH"`
- 若同一材料描述中存在多个独立壁厚结果，按原文顺序输出多个 item
- 最终只输出 JSON，不要解释。
""".strip()


PRESSURE_PLATFORM_ADAPTER_TEXT_V3 = """
平台输出适配要求（只改输出格式，不改上述判定规则）：
- 先按上面的 `pressure_rating` 规则完成判断，再把结果转换为当前平台结构。
- 当前平台不接收 `pressure_rating` 字段，只接收：
  {
    "PRESSURE": ""
  }
- 也就是把最终的 `pressure_rating` 字符串直接写入 `PRESSURE`
- 若无压力等级，输出 `PRESSURE=""`
- 最终只输出 JSON，不要解释。
""".strip()


SIZE_EXAMPLES_TEXT_V3 = """
**复杂格式处理示例**:
- `DN300xDN200` → `SIZE_ITEMS=[{"type":"DN","value":"DN300"},{"type":"DN","value":"DN200"}]`, `LENGTH=""`
- `DN50X20` → `SIZE_ITEMS=[{"type":"DN","value":"DN50"},{"type":"DN","value":"DN20"}]`, `LENGTH=""`
- `φ60.3Xφ48.3` → `SIZE_ITEMS=[{"type":"OD","value":"OD60.3"},{"type":"OD","value":"OD48.3"}]`, `LENGTH=""`
- `3"*DN65` → `SIZE_ITEMS=[{"type":"INCH","value":"NPS3"},{"type":"DN","value":"DN65"}]`, `LENGTH=""`
- `灰口铸铁, 壁厚5.8mm, L=3m, DN200` → `SIZE_ITEMS=[{"type":"DN","value":"DN200"}]`, `LENGTH="3000"`
- `PIPE ... CUT TO 1505 DN100` → `SIZE_ITEMS=[{"type":"DN","value":"DN100"}]`, `LENGTH="1505"`
""".strip()


THICKNESS_EXAMPLES_TEXT_V3 = """
**复杂格式处理示例**:
- `THK=6.3mm` → `THICKNESS_ITEMS=[{"type":"MM","value":"6.3MM"}]`
- `异径管 R(E)，DN80x50Ⅱ-4x3.5，20，GB/T12459-2017` → `THICKNESS_ITEMS=[{"type":"MM","value":"4MMX3.5MM"}]`
- `273x5.0/3.2 - 219.1x2.9` → `THICKNESS_ITEMS=[{"type":"MM","value":"5MMX3.2MMX2.9MM"}]`
- `12mm*SCH40S` → `THICKNESS_ITEMS=[{"type":"MM","value":"12MMXSCH40S"}]`
- `SCH40XSCH80` → `THICKNESS_ITEMS=[{"type":"SCHEDULE","value":"SCH40XSCH80"}]`
- `S-STDXS-STD` → `THICKNESS_ITEMS=[{"type":"SCHEDULE","value":"STDXSTD"}]`
""".strip()


PRESSURE_EXAMPLES_TEXT_V3 = """
**复杂格式处理示例**:
- `PN16/PN10` → `PRESSURE="PN16/PN10"`
- `CLASS 3000` → `PRESSURE="CLASS3000"`
- `CL600` → `PRESSURE="CL600"`
- `300LB` → `PRESSURE="300LB"`
- `300#` → `PRESSURE="300#"`
- `WN350(B)-25 RF` → `PRESSURE="PN25"`
- `BL150-16 RF` → `PRESSURE="PN16"`
- `Class I` → `PRESSURE=""`
- `A672 C60 CL12` → `PRESSURE=""`
""".strip()


SIZE_LENGTH_SYSTEM_PROMPT_V3 = "\n\n".join([
    "你是工业管道材料描述的结构化抽取器。",
    "任务：只从输入原文中抽取尺寸规格和长度信息。",
    "输出必须是严格 JSON，不要解释，不要 Markdown。",
    COMMON_RULES_TEXT_V3,
    SIZE_FIELD_RULES_TEXT_V3,
    SIZE_PLATFORM_ADAPTER_TEXT_V3,
    SIZE_EXAMPLES_TEXT_V3,
    """请严格按照以下JSON格式返回，不要添加任何解释：
{
  "SIZE_ITEMS": [
    {"type": "DN", "value": "DN100"}
  ],
  "LENGTH": ""
}""",
])


THICKNESS_SYSTEM_PROMPT_V3 = "\n\n".join([
    "你是工业管道材料描述的结构化抽取器。",
    "任务：只从输入原文中抽取壁厚信息。",
    "输出必须是严格 JSON，不要解释，不要 Markdown。",
    COMMON_RULES_TEXT_V3,
    THICKNESS_FIELD_RULES_TEXT_V3,
    THICKNESS_PLATFORM_ADAPTER_TEXT_V3,
    THICKNESS_EXAMPLES_TEXT_V3,
    """请严格按照以下JSON格式返回，不要添加任何解释：
{
  "THICKNESS_ITEMS": [
    {"type": "MM", "value": "6.0MM"}
  ]
}""",
])


PRESSURE_SYSTEM_PROMPT_V3 = "\n\n".join([
    "你是工业管道材料描述的结构化抽取器。",
    "任务：只从输入原文中抽取压力等级/磅级信息。",
    "输出必须是严格 JSON，不要解释，不要 Markdown。",
    COMMON_RULES_TEXT_V3,
    PRESSURE_FIELD_RULES_TEXT_V3,
    PRESSURE_PLATFORM_ADAPTER_TEXT_V3,
    PRESSURE_EXAMPLES_TEXT_V3,
    """请严格按照以下JSON格式返回，不要添加任何解释：
{
  "PRESSURE": ""
}""",
])


STRUCTURAL_DEBUG_SUFFIX_V3 = """

【DEBUG 输出模式】
输出严格 JSON，结构如下：
{
  "task": "size_length|thickness|pressure",
  "trace": [],
  "final": {}
}
"""


def get_size_length_system_prompt_v3(debug: bool = False) -> str:
    return SIZE_LENGTH_SYSTEM_PROMPT_V3 + STRUCTURAL_DEBUG_SUFFIX_V3 if debug else SIZE_LENGTH_SYSTEM_PROMPT_V3


def get_thickness_system_prompt_v3(debug: bool = False, version: str = "v3") -> str:
    _ = version
    return THICKNESS_SYSTEM_PROMPT_V3 + STRUCTURAL_DEBUG_SUFFIX_V3 if debug else THICKNESS_SYSTEM_PROMPT_V3


def get_pressure_system_prompt_v3(debug: bool = False) -> str:
    return PRESSURE_SYSTEM_PROMPT_V3 + STRUCTURAL_DEBUG_SUFFIX_V3 if debug else PRESSURE_SYSTEM_PROMPT_V3
