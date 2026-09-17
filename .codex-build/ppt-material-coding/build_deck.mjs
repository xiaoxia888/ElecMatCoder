import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder";
const SKILL_DIR = "/Users/guoxi/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build/ppt-material-coding");
const FINAL_PPTX = path.join(
  workspaceDir,
  "artifacts/presentations/材料编码平台与模型微调汇报_20260916.pptx",
);

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const FONT = "PingFang SC";
const COLORS = {
  navy: "#152A4A",
  blue: "#2F6BFF",
  blue2: "#5D8BFF",
  cyan: "#23A6A8",
  orange: "#F2A13A",
  ink: "#172033",
  body: "#4B5870",
  muted: "#7D899D",
  line: "#DCE3EF",
  pale: "#EEF3FF",
  paleCyan: "#EAF8F7",
  paleOrange: "#FFF4E6",
  white: "#FFFFFF",
  bg: "#F7F9FC",
};

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function addText(slide, text, pos, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: pos,
    fill: opts.fill ?? "none",
    line: opts.line ?? { fill: "none", width: 0 },
    borderRadius: opts.borderRadius,
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 20,
    bold: opts.bold ?? false,
    color: opts.color ?? COLORS.ink,
    alignment: opts.alignment ?? "left",
    autoFit: opts.autoFit ?? "shrinkText",
  };
  return shape;
}

function addTitle(slide, title, subtitle, page) {
  addText(slide, String(page).padStart(2, "0"),
    { left: 72, top: 42, width: 40, height: 28 },
    { fontSize: 14, bold: true, color: COLORS.blue });
  slide.shapes.add({
    geometry: "line",
    position: { left: 115, top: 56, width: 42, height: 0 },
    line: { style: "solid", fill: COLORS.blue, width: 3 },
  });
  addText(slide, title,
    { left: 72, top: 78, width: 870, height: 54 },
    { fontSize: 34, bold: true, color: COLORS.navy });
  addText(slide, subtitle,
    { left: 72, top: 134, width: 900, height: 34 },
    { fontSize: 17, color: COLORS.body });
  addText(slide, "材料编码项目汇报",
    { left: 1010, top: 52, width: 198, height: 25 },
    { fontSize: 13, color: COLORS.muted, alignment: "right" });
}

function addFooter(slide, text) {
  slide.shapes.add({
    geometry: "line",
    position: { left: 72, top: 664, width: 1136, height: 0 },
    line: { style: "solid", fill: COLORS.line, width: 1 },
  });
  addText(slide, text,
    { left: 72, top: 674, width: 1136, height: 24 },
    { fontSize: 14, color: COLORS.body });
}

function addStep(slide, x, y, number, title, caption, accent) {
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: x, top: y, width: 38, height: 38 },
    fill: accent,
    line: { fill: "none", width: 0 },
  });
  addText(slide, number,
    { left: x, top: y + 5, width: 38, height: 25 },
    { fontSize: 15, bold: true, color: COLORS.white, alignment: "center" });
  addText(slide, title,
    { left: x + 52, top: y - 2, width: 238, height: 28 },
    { fontSize: 20, bold: true, color: COLORS.ink });
  addText(slide, caption,
    { left: x + 52, top: y + 29, width: 250, height: 46 },
    { fontSize: 14, color: COLORS.body });
}

// Slide 1: Platform implementation
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(slide, "材料编码智能平台实现", "从原始材料描述到结构化字段与标准编码", 1);

  addText(slide, "模型理解描述，规则生成编码，人工处理异常",
    { left: 72, top: 190, width: 610, height: 44 },
    { fontSize: 25, bold: true, color: COLORS.ink });

  addStep(slide, 76, 258, "1", "批量接入", "支持 Excel、CSV 等业务数据批量导入", COLORS.blue);
  addStep(slide, 76, 350, "2", "结构化识别", "提取种类、材质、规范、尺寸和压力等级", COLORS.cyan);
  addStep(slide, 76, 442, "3", "规则编码与复核", "编码规则完成拼接，异常结果进入人工复核", COLORS.orange);

  // Screenshot placeholder: one reserved image area, deliberately smaller than the text area.
  slide.shapes.add({
    geometry: "roundRect",
    name: "platform-screenshot-placeholder",
    position: { left: 744, top: 205, width: 464, height: 300 },
    fill: COLORS.white,
    line: { style: "solid", fill: "#BFC9DA", width: 1.5 },
    borderRadius: "rounded-2xl",
    shadow: "shadow-sm",
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 744, top: 205, width: 464, height: 34 },
    fill: COLORS.navy,
    line: { fill: "none", width: 0 },
  });
  slide.shapes.add({ geometry: "ellipse", position: { left: 760, top: 217, width: 9, height: 9 }, fill: "#FF6B6B", line: { fill: "none", width: 0 } });
  slide.shapes.add({ geometry: "ellipse", position: { left: 777, top: 217, width: 9, height: 9 }, fill: "#FFD166", line: { fill: "none", width: 0 } });
  slide.shapes.add({ geometry: "ellipse", position: { left: 794, top: 217, width: 9, height: 9 }, fill: "#5DD39E", line: { fill: "none", width: 0 } });
  addText(slide, "平台界面截图预留区",
    { left: 790, top: 300, width: 370, height: 38 },
    { fontSize: 23, bold: true, color: COLORS.navy, alignment: "center" });
  addText(slide, "建议替换为包含原始描述、识别字段、最终编码\n和人工复核入口的实际操作截图",
    { left: 790, top: 354, width: 370, height: 70 },
    { fontSize: 15, color: COLORS.body, alignment: "center" });

  const stages = ["数据导入", "描述规范化", "模型识别", "编码生成", "人工复核"];
  const stageColors = [COLORS.blue, COLORS.blue2, COLORS.cyan, COLORS.orange, COLORS.navy];
  stages.forEach((label, i) => {
    const x = 100 + i * 220;
    slide.shapes.add({
      geometry: "roundRect",
      position: { left: x, top: 568, width: 150, height: 42 },
      fill: i % 2 === 0 ? COLORS.pale : COLORS.white,
      line: { style: "solid", fill: stageColors[i], width: 1.3 },
      borderRadius: "rounded-xl",
    });
    addText(slide, label,
      { left: x, top: 578, width: 150, height: 22 },
      { fontSize: 15, bold: true, color: stageColors[i], alignment: "center" });
    if (i < stages.length - 1) {
      slide.shapes.add({
        geometry: "rightArrow",
        position: { left: x + 162, top: 581, width: 34, height: 16 },
        fill: COLORS.line,
        line: { fill: "none", width: 0 },
      });
    }
  });

  addFooter(slide, "模型负责语义识别，编码规则保证结果稳定，人工复核处理异常数据");
  slide.speakerNotes.textFrame.setText(
    "平台接收业务中的材料描述，先完成格式规范化，再由领域模型提取材料种类、材质标准、尺寸等关键字段。模型输出不会直接作为最终编码，而是交给确定性的编码规则进行规范化和拼接。对于置信度较低或存在异常的数据，平台提供人工复核入口，从而兼顾处理效率和编码准确性。右侧区域可替换为实际平台截图。",
  );
}

// Slide 2: Dataset construction
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addTitle(slide, "教师模型与人工复核协同构建训练数据", "自动预标注提升效率，人工审核保证关键标签可靠", 2);

  addText(slide, "训练数据制作流程",
    { left: 72, top: 194, width: 260, height: 32 },
    { fontSize: 22, bold: true, color: COLORS.navy });

  const nodes = [
    { title: "业务数据汇集", note: "历史材料库与业务表单", color: COLORS.blue, fill: COLORS.pale },
    { title: "教师模型预标注", note: "生成结构化字段与标签", color: COLORS.cyan, fill: COLORS.paleCyan },
    { title: "自动校验", note: "格式、重复与冲突检查", color: COLORS.orange, fill: COLORS.paleOrange },
    { title: "人工复核定稿", note: "修正不确定与复杂样本", color: COLORS.navy, fill: "#EEF1F6" },
  ];
  const nodeX = [76, 374, 672, 970];
  nodes.forEach((node, i) => {
    const x = nodeX[i];
    slide.shapes.add({
      geometry: "roundRect",
      name: `dataset-stage-${i + 1}`,
      position: { left: x, top: 260, width: 224, height: 138 },
      fill: node.fill,
      line: { style: "solid", fill: node.color, width: 1.4 },
      borderRadius: "rounded-2xl",
    });
    addText(slide, `0${i + 1}`,
      { left: x + 18, top: 278, width: 40, height: 26 },
      { fontSize: 15, bold: true, color: node.color });
    addText(slide, node.title,
      { left: x + 18, top: 317, width: 188, height: 30 },
      { fontSize: 20, bold: true, color: COLORS.ink });
    addText(slide, node.note,
      { left: x + 18, top: 357, width: 188, height: 28 },
      { fontSize: 14, color: COLORS.body });
    if (i < nodes.length - 1) {
      slide.shapes.add({
        geometry: "rightArrow",
        position: { left: x + 239, top: 318, width: 42, height: 22 },
        fill: COLORS.line,
        line: { fill: "none", width: 0 },
      });
    }
  });

  addText(slide, "质量控制",
    { left: 76, top: 455, width: 130, height: 30 },
    { fontSize: 19, bold: true, color: COLORS.navy });
  const quality = [
    ["格式统一", "统一字段结构、单位和标签表达"],
    ["重复清理", "同一描述与同一标签仅保留一条"],
    ["冲突复核", "标签冲突样本交由人工确认"],
  ];
  quality.forEach(([title, note], i) => {
    const x = 250 + i * 320;
    slide.shapes.add({
      geometry: "line",
      position: { left: x, top: 474, width: 36, height: 0 },
      line: { style: "solid", fill: [COLORS.blue, COLORS.cyan, COLORS.orange][i], width: 4 },
    });
    addText(slide, title,
      { left: x, top: 490, width: 210, height: 28 },
      { fontSize: 18, bold: true, color: COLORS.ink });
    addText(slide, note,
      { left: x, top: 522, width: 260, height: 48 },
      { fontSize: 14, color: COLORS.body });
  });

  slide.shapes.add({
    geometry: "line",
    position: { left: 230, top: 420, width: 850, height: 150 },
    line: { style: "dash", fill: COLORS.cyan, width: 2 },
  });
  addText(slide, "审核结果回流样本库",
    { left: 860, top: 575, width: 260, height: 28 },
    { fontSize: 15, bold: true, color: COLORS.cyan, alignment: "right" });

  addFooter(slide, "教师模型承担批量预标注，人工审核聚焦不确定样本，形成可持续迭代的数据闭环");
  slide.speakerNotes.textFrame.setText(
    "训练数据采用教师模型预标注和人工审核相结合的方式制作。教师模型先完成大批量样本的初步识别，程序再执行格式校验、重复清理和冲突检测。人工审核集中处理不确定样本和复杂描述。审核后的结果进入标准训练集，同时持续回流样本库，为后续训练提供更可靠的数据基础。",
  );
}

// Slide 3: Fine-tuning and results
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;
  addTitle(slide, "基于 Unsloth 的 LoRA SFT 微调", "以 Qwen3.5-9B 材质规范模型为例", 3);

  addText(slide, "微调方案",
    { left: 72, top: 196, width: 180, height: 34 },
    { fontSize: 22, bold: true, color: COLORS.navy });
  const configItems = [
    ["基础模型", "Qwen3.5-9B"],
    ["训练框架", "Unsloth"],
    ["微调方式", "LoRA SFT"],
    ["训练数据", "77,468 条"],
    ["训练过程", "1 Epoch / 4,358 Steps"],
  ];
  configItems.forEach(([label, value], i) => {
    const y = 250 + i * 54;
    addText(slide, label,
      { left: 72, top: y, width: 100, height: 26 },
      { fontSize: 14, color: COLORS.muted });
    addText(slide, value,
      { left: 180, top: y - 2, width: 280, height: 30 },
      { fontSize: 19, bold: true, color: COLORS.ink });
    slide.shapes.add({
      geometry: "line",
      position: { left: 72, top: y + 34, width: 388, height: 0 },
      line: { style: "solid", fill: COLORS.line, width: 1 },
    });
  });
  addText(slide, "训练集 69,719 条，验证集 7,749 条",
    { left: 72, top: 536, width: 390, height: 28 },
    { fontSize: 14, color: COLORS.body });

  addText(slide, "验证损失",
    { left: 520, top: 196, width: 180, height: 34 },
    { fontSize: 22, bold: true, color: COLORS.navy });
  const chart = slide.charts.add("line", {
    position: { left: 510, top: 230, width: 698, height: 320 },
    title: "验证损失随训练步数下降",
    titlePlacement: "aboveChart",
    titleTextStyle: { typeface: FONT, fontSize: 16, fill: COLORS.body, bold: false },
    categories: ["1000", "2000", "3000", "4000", "4358"],
    series: [{
      name: "Eval loss",
      values: [0.00197221455, 0.00111917942, 0.00094127137, 0.00077259756, 0.00076551776],
      line: { style: "solid", fill: COLORS.blue, width: 3 },
      marker: { symbol: "circle", size: 7 },
    }],
    hasLegend: false,
    lineOptions: { grouping: "standard", smooth: false },
    xAxis: {
      title: "训练步数",
      textStyle: { typeface: FONT, fontSize: 12, fill: COLORS.body },
      line: { style: "solid", fill: COLORS.line, width: 1 },
      majorGridlines: null,
    },
    yAxis: {
      title: "验证损失",
      min: 0.0006,
      max: 0.0021,
      majorUnit: 0.0003,
      numberFormatCode: "0.0000",
      textStyle: { typeface: FONT, fontSize: 12, fill: COLORS.body },
      line: { style: "solid", fill: COLORS.line, width: 1 },
      majorGridlines: { style: "solid", fill: "#E6EBF3", width: 1 },
    },
    dataLabels: {
      showValue: true,
      position: "outEnd",
      textStyle: { typeface: FONT, fontSize: 10, fill: COLORS.body },
    },
    chartFill: COLORS.white,
    chartLine: { style: "solid", fill: COLORS.line, width: 1 },
    plotAreaFill: COLORS.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });

  const metrics = [
    ["0.000766", "最佳验证损失", COLORS.blue],
    ["61.2%", "验证损失下降", COLORS.cyan],
    ["0.31%", "可训练参数占比", COLORS.orange],
  ];
  metrics.forEach(([value, label, color], i) => {
    const x = 510 + i * 232;
    addText(slide, value,
      { left: x, top: 574, width: 190, height: 42 },
      { fontSize: 27, bold: true, color });
    addText(slide, label,
      { left: x, top: 616, width: 190, height: 24 },
      { fontSize: 13, color: COLORS.body });
    if (i < 2) {
      slide.shapes.add({
        geometry: "line",
        position: { left: x + 204, top: 582, width: 0, height: 52 },
        line: { style: "solid", fill: COLORS.line, width: 1 },
      });
    }
  });

  addFooter(slide, "LoRA 仅训练约 0.31% 的模型参数，最佳验证结果出现在第 4,358 步");
  slide.speakerNotes.textFrame.setText(
    "本次以 Qwen3.5-9B 为基座，使用 Unsloth 框架进行 LoRA SFT 微调。训练集为 69,719 条，验证集为 7,749 条，共训练一个轮次、4,358 步。LoRA 实际训练的参数约为 2,910 万，仅占模型总参数的 0.31%。最佳验证损失达到 0.000766，相比第一次验证下降约 61.2%。当前没有独立测试集上的 F1、准确率和召回率，因此本页不展示这些指标。",
  );
}

// Private previews for visual review.
for (let i = 0; i < presentation.slides.length; i += 1) {
  const slide = presentation.slides.getItemAt(i);
  const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(
    path.join(TMP_DIR, `slide-${i + 1}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(TMP_DIR, `slide-${i + 1}.layout.json`), await layout.text());
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer/material-coding-ppt");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 3,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [3],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [FONT], scriptFonts: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "材料编码平台与模型微调汇报_20260916.validation.json"),
});

console.log(FINAL_PPTX);
