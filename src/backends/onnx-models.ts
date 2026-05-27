import type { OnnxModelDefinition, OnnxModelCategory } from "../upscaler/types";

export const ONNX_MODEL_DEFINITIONS: readonly OnnxModelDefinition[] = [
  {
    id: "ecbsr_x2_m4c8_y",
    label: "ECBSR Y-only x2",
    modelPath: "models/ecbsr_x2_m4c8_y.onnx",
    scale: 2,
    inputLayout: "NCHW",
    inputChannels: 1,
    outputChannels: 1,
    inputColorSpace: "y-only",
    inputPacking: "luma_f32_planar",
    outputPacking: "luma_f32_planar",
    compositeMode: "luma_replace",
    sizePolicy: {
      widthAlign: 32,
      heightAlign: 1,
      minInputWidth: 32,
      minInputHeight: 1,
      fixedScale: 2,
    },
    executionProvider: "webgpu",
    compositor: "luma_replace",
    description: "当前默认的 ECBSR 移动版亮度超分模型。",
    category: "lightweight",
    quantization: "fp32",
  },
  {
    id: "rgb_bicubic_x2",
    label: "RGB Bicubic x2",
    modelPath: "models/rgb_bicubic_x2.onnx",
    scale: 2,
    inputLayout: "NCHW",
    inputChannels: 3,
    outputChannels: 3,
    inputColorSpace: "rgb",
    inputPacking: "rgb_f32_planar",
    outputPacking: "rgb_f32_planar",
    compositeMode: "rgb_replace",
    sizePolicy: {
      widthAlign: 1,
      heightAlign: 1,
      minInputWidth: 1,
      minInputHeight: 1,
      fixedScale: 2,
    },
    executionProvider: "webgpu",
    compositor: "rgb_replace",
    description: "用于验证 RGB 三通道 ONNX 通路的 2x 双三次基线模型。",
    category: "lightweight",
    quantization: "fp32",
  },
  {
    id: "realesrgan_x2plus",
    label: "Real-ESRGAN x2+",
    modelPath: "models/realesrgan_x2plus.onnx",
    scale: 2,
    inputLayout: "NCHW",
    inputChannels: 3,
    outputChannels: 3,
    inputColorSpace: "rgb",
    inputPacking: "rgb_f32_planar",
    outputPacking: "rgb_f32_planar",
    compositeMode: "rgb_replace",
    sizePolicy: {
      widthAlign: 1,
      heightAlign: 1,
      minInputWidth: 1,
      minInputHeight: 1,
      fixedScale: 2,
    },
    executionProvider: "webgpu",
    compositor: "rgb_replace",
    description: "Real-ESRGAN x2plus 高质量超分，适合动漫和真人视频。需要分块推理。",
    category: "quality",
    quantization: "fp32",
    tileSize: 256,
    source: {
      type: "download",
      downloadUrl: "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/Real-ESRGAN_x2plus.onnx",
      fileSize: 67_100_000,
    },
  },
  {
    id: "realesrgan_animevideo_v3_x4",
    label: "RealESR-AnimeVideo v3 x4",
    modelPath: "models/realesrgan_animevideo_v3_x4.onnx",
    scale: 4,
    inputLayout: "NCHW",
    inputChannels: 3,
    outputChannels: 3,
    inputColorSpace: "rgb",
    inputPacking: "rgb_f32_planar",
    outputPacking: "rgb_f32_planar",
    compositeMode: "rgb_replace",
    sizePolicy: {
      widthAlign: 1,
      heightAlign: 1,
      minInputWidth: 1,
      minInputHeight: 1,
      fixedScale: 4,
    },
    executionProvider: "webgpu",
    compositor: "rgb_replace",
    description: "RealESR-AnimeVideo v3 专为动漫视频优化的 4x 超分模型，体积小巧。",
    category: "balanced",
    quantization: "fp32",
    tileSize: 256,
    source: {
      type: "download",
      downloadUrl: "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/RealESR-AnimeVideo-v3_x4.onnx",
      fileSize: 2_500_000,
    },
  },
] as const;

export const DEFAULT_ONNX_MODEL_ID = ONNX_MODEL_DEFINITIONS[0].id;

const ONNX_MODEL_MAP = new Map(
  ONNX_MODEL_DEFINITIONS.map((definition) => [definition.id, definition]),
);

export function getOnnxModelDefinition(
  modelId?: string | null,
): OnnxModelDefinition {
  if (modelId && ONNX_MODEL_MAP.has(modelId)) {
    return ONNX_MODEL_MAP.get(modelId)!;
  }
  return ONNX_MODEL_MAP.get(DEFAULT_ONNX_MODEL_ID)!;
}

export function getOnnxModelOptions(): Array<
  Pick<OnnxModelDefinition, "id" | "label" | "category" | "source">
> {
  return ONNX_MODEL_DEFINITIONS.map(({ id, label, category, source }) => ({
    id,
    label,
    category,
    source,
  }));
}

export function getModelsByCategory(
  category: OnnxModelCategory,
): readonly OnnxModelDefinition[] {
  return ONNX_MODEL_DEFINITIONS.filter((m) => (m.category ?? "lightweight") === category);
}

export function getValidModelIds(): ReadonlySet<string> {
  return new Set(ONNX_MODEL_DEFINITIONS.map((m) => m.id));
}

export function resolvePrecisionModel(
  modelId: string,
  fp16Supported: boolean,
): OnnxModelDefinition {
  const model = getOnnxModelDefinition(modelId);
  if (!model.precisionAlternatives) return model;

  const target = fp16Supported
    ? model.precisionAlternatives.fp16
    : model.precisionAlternatives.fp32;

  if (target && ONNX_MODEL_MAP.has(target)) {
    return ONNX_MODEL_MAP.get(target)!;
  }
  return model;
}
