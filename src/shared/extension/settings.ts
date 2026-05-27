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
const VALID_MODES: ReadonlySet<Settings["mode"]> = new Set([
  "balanced",
  "quality",
  "performance",
]);
const VALID_DISPLAY_MODES: ReadonlySet<Settings["displayMode"]> = new Set([
  "overlay",
  "replace",
]);
const VALID_TARGET_FPS: ReadonlySet<Settings["targetFps"]> = new Set([
  "auto",
  "60",
  "30",
  "24",
  "15",
]);

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
  const settings = value as Record<string, unknown>;

  return {
    enabled:
      typeof settings.enabled === "boolean"
        ? settings.enabled
        : DEFAULT_SETTINGS.enabled,
    scale: normalizeFiniteNumber(settings.scale, DEFAULT_SETTINGS.scale),
    sharpness: normalizeFiniteNumber(
      settings.sharpness,
      DEFAULT_SETTINGS.sharpness,
    ),
    mode: VALID_MODES.has(settings.mode as Settings["mode"])
      ? (settings.mode as Settings["mode"])
      : DEFAULT_SETTINGS.mode,
    overlayOpacity: normalizeFiniteNumber(
      settings.overlayOpacity,
      DEFAULT_SETTINGS.overlayOpacity,
    ),
    displayMode: VALID_DISPLAY_MODES.has(
      settings.displayMode as Settings["displayMode"],
    )
      ? (settings.displayMode as Settings["displayMode"])
      : DEFAULT_SETTINGS.displayMode,
    engine: VALID_ENGINES.has(settings.engine as EngineType)
      ? (settings.engine as EngineType)
      : DEFAULT_SETTINGS.engine,
    modelId: normalizeOnnxModelId(settings.modelId),
    targetFps: VALID_TARGET_FPS.has(settings.targetFps as Settings["targetFps"])
      ? (settings.targetFps as Settings["targetFps"])
      : DEFAULT_SETTINGS.targetFps,
  };
}

function normalizeFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
