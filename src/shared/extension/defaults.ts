import type { EngineType, Settings } from "../../upscaler/types";

export const ONNX_MODEL_OPTIONS = [
  { id: "ecbsr_x2_m4c8_y", label: "ECBSR Y-only x2" },
  { id: "rgb_bicubic_x2", label: "RGB Bicubic x2" },
] as const;

export const DEFAULT_ONNX_MODEL_ID = ONNX_MODEL_OPTIONS[0].id;

export const VALID_ENGINES: ReadonlySet<EngineType> = new Set([
  "webgpu",
  "tiny-cnn",
  "ecbsr",
]);

export const DEFAULT_SETTINGS: Settings = {
  enabled: false,
  scale: 1.5,
  sharpness: 0.65,
  mode: "balanced",
  overlayOpacity: 0.8,
  displayMode: "overlay",
  engine: "tiny-cnn",
  modelId: DEFAULT_ONNX_MODEL_ID,
  targetFps: "auto",
};
