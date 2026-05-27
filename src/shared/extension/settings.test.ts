import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  normalizeSettings,
} from "./settings";
import { loadSettings, saveSettings } from "./storage";

const storageState = new Map<string, unknown>();
const getMock = vi.fn(async (defaults?: Record<string, unknown>) => {
  const result = { ...(defaults ?? {}) };
  for (const [key, value] of storageState.entries()) {
    result[key] = value;
  }
  return result;
});
const setMock = vi.fn(async (values: Record<string, unknown>) => {
  for (const [key, value] of Object.entries(values)) {
    storageState.set(key, value);
  }
});

vi.stubGlobal("chrome", {
  storage: {
    sync: {
      get: getMock,
      set: setMock,
    },
  },
});

beforeEach(() => {
  storageState.clear();
  getMock.mockClear();
  setMock.mockClear();
});

describe("normalizeSettings", () => {
  it("falls back to defaults for unsupported engine and model", () => {
    expect(
      normalizeSettings({
        enabled: true,
        engine: "unknown",
        modelId: "missing-model",
      }),
    ).toMatchObject({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: DEFAULT_SETTINGS.engine,
      modelId: DEFAULT_ONNX_MODEL_ID,
    });
  });

  it("preserves valid engine and model values", () => {
    expect(
      normalizeSettings({
        enabled: true,
        engine: "ecbsr",
        modelId: "rgb_bicubic_x2",
      }),
    ).toMatchObject({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
    });
  });

  it("falls back to defaults for invalid runtime values", () => {
    expect(
      normalizeSettings({
        enabled: "yes" as never,
        scale: Number.NaN,
        sharpness: Number.POSITIVE_INFINITY,
        overlayOpacity: Number.NaN,
        mode: "turbo" as never,
        displayMode: "blend" as never,
        targetFps: "120" as never,
      }),
    ).toEqual(DEFAULT_SETTINGS);
  });
});

describe("storage helpers", () => {
  it("loadSettings normalizes invalid persisted values", async () => {
    storageState.set("enabled", "true");
    storageState.set("scale", Number.NaN);
    storageState.set("mode", "turbo");

    await expect(loadSettings()).resolves.toEqual(DEFAULT_SETTINGS);
  });

  it("saveSettings merges patch values with existing settings", async () => {
    storageState.set("enabled", true);
    storageState.set("engine", "ecbsr");
    storageState.set("modelId", "rgb_bicubic_x2");

    await expect(saveSettings({ targetFps: "30" })).resolves.toEqual({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
      targetFps: "30",
    });

    expect(setMock).toHaveBeenCalledWith({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
      targetFps: "30",
    });
  });
});
