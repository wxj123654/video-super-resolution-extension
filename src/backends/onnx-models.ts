import type { OnnxModelDefinition } from "../upscaler/types";

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

export function getOnnxModelOptions(): Array<Pick<OnnxModelDefinition, "id" | "label">> {
  return ONNX_MODEL_DEFINITIONS.map(({ id, label }) => ({ id, label }));
}
