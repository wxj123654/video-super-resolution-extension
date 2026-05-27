import { describe, expect, it } from "vitest";

import {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  normalizeSettings,
} from "./settings";

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
});
