import type { EngineType, Settings } from "../../upscaler/types";
import {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  ONNX_MODEL_OPTIONS,
  VALID_ENGINES,
} from "./defaults";

const VALID_ONNX_MODEL_IDS: ReadonlySet<string> = new Set(
  ONNX_MODEL_OPTIONS.map((option) => option.id),
);

export {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  ONNX_MODEL_OPTIONS,
  VALID_ENGINES,
};

export function normalizeOnnxModelId(modelId: unknown): string {
  if (typeof modelId === "string" && VALID_ONNX_MODEL_IDS.has(modelId)) {
    return modelId;
  }

  return DEFAULT_ONNX_MODEL_ID;
}

export function normalizeSettings(
  value: Partial<Omit<Settings, "engine" | "modelId">> & {
    engine?: string;
    modelId?: unknown;
  } = {},
): Settings {
  return {
    ...DEFAULT_SETTINGS,
    ...value,
    engine: VALID_ENGINES.has(value.engine as EngineType)
      ? (value.engine as EngineType)
      : DEFAULT_SETTINGS.engine,
    modelId: normalizeOnnxModelId(value.modelId),
  };
}
