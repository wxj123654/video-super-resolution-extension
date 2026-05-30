import type { EngineType, Settings } from "../../upscaler/types";

export const ONNX_MODEL_OPTIONS = [
  { id: "ecbsr_x2_m4c8_y", label: "ECBSR Y-only x2", category: "lightweight" as const, source: undefined },
  { id: "rgb_bicubic_x2", label: "RGB Bicubic x2", category: "lightweight" as const, source: undefined },
  { id: "realesrgan_animevideo_v3_x4", label: "RealESR-AnimeVideo v3 x4", category: "balanced" as const, source: { type: "download" as const, fileSize: 2_500_000 } },
  { id: "realesrgan_x2plus", label: "Real-ESRGAN x2+", category: "quality" as const, source: { type: "download" as const, fileSize: 67_100_000 } },
] as const;

export const DEFAULT_ONNX_MODEL_ID = ONNX_MODEL_OPTIONS[0].id;

export const VALID_ENGINES: ReadonlySet<EngineType> = new Set([
  "webgpu",
  "tiny-cnn",
  "onnx",
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
