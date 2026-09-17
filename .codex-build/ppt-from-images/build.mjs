import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/guoxi/Desktop/workspace/NJNCC/python_code/ElecMatCoder";
const SKILL_DIR = "/Users/guoxi/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const buildDir = path.join(workspaceDir, ".codex-build/ppt-from-images");
const finalPath = path.join(
  workspaceDir,
  "artifacts/presentations/材料代码数据集建设汇报_图片版_20260916.pptx",
);
const imagePaths = [
  path.join(workspaceDir, "artifacts/presentation_images/01_材料代码高质量数据集建设.png"),
  path.join(workspaceDir, "artifacts/presentation_images/02_教师模型与人工审核协同构建训练数据.png"),
  path.join(workspaceDir, "artifacts/presentation_images/03_通过领域模型微调验证数据集有效性.png"),
];
const notes = [
  "本页介绍材料代码高质量数据集的来源、原始数据问题和建设目标。",
  "本页介绍教师模型预标注、自动质量检查和人工审核相结合的数据制作流程。",
  "本页以 Qwen3.5-9B 材质规范模型为例，展示 Unsloth 与 LoRA SFT 微调对数据集有效性的验证结果。",
];

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

for (let index = 0; index < imagePaths.length; index += 1) {
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  const bytes = await fs.readFile(imagePaths[index]);
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: path.basename(imagePaths[index], ".png"),
    fit: "cover",
    position: { left: 0, top: 0, width: 1280, height: 720 },
  });
  slide.speakerNotes.textFrame.setText(notes[index]);

  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    path.join(buildDir, `slide-${index + 1}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const stagingDir = path.join(workspaceDir, ".codex-finalizer/ppt-from-images");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await finalizePresentation({
  explicitTotalSlideCount: 3,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000"],
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "validation.json"),
});

console.log(finalPath);
