import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  queryActiveTab: vi.fn(),
  ensureTabReady: vi.fn(),
  sendMessageToTab: vi.fn(),
  openExtensionOptions: vi.fn(),
  loadSettings: vi.fn(),
  saveSettings: vi.fn(),
}));

vi.mock("./chrome", () => ({
  queryActiveTab: mocks.queryActiveTab,
  ensureTabReady: mocks.ensureTabReady,
  sendMessageToTab: mocks.sendMessageToTab,
  openExtensionOptions: mocks.openExtensionOptions,
}));

vi.mock("@src/shared/extension/storage", () => ({
  loadSettings: mocks.loadSettings,
  saveSettings: mocks.saveSettings,
}));

import {
  bootstrapPopup,
  launchOptionsPage,
  rescanPopup,
  updatePopupSettings,
} from "./popup-controller";

describe("popup-controller", () => {
  beforeEach(() => {
    mocks.queryActiveTab.mockReset();
    mocks.ensureTabReady.mockReset();
    mocks.sendMessageToTab.mockReset();
    mocks.openExtensionOptions.mockReset();
    mocks.loadSettings.mockReset();
    mocks.saveSettings.mockReset();

    mocks.loadSettings.mockResolvedValue({
      enabled: true,
      engine: "tiny-cnn",
      displayMode: "overlay",
      scale: 1.5,
      sharpness: 0.65,
      overlayOpacity: 0.8,
      mode: "balanced",
      targetFps: "auto",
      modelId: "ecbsr_x2_m4c8_y",
    });
  });

  it("returns an unavailable state when there is no active tab", async () => {
    mocks.queryActiveTab.mockResolvedValue(undefined);

    await expect(bootstrapPopup()).resolves.toMatchObject({
      tabId: null,
      model: {
        connectionLabel: "Unavailable",
      },
    });
    expect(mocks.ensureTabReady).not.toHaveBeenCalled();
    expect(mocks.sendMessageToTab).not.toHaveBeenCalled();
  });

  it("saves settings, sends update, and reflects returned state", async () => {
    mocks.saveSettings
      .mockResolvedValueOnce({
        enabled: true,
        engine: "tiny-cnn",
        displayMode: "overlay",
        scale: 1.5,
        sharpness: 0.65,
        overlayOpacity: 0.8,
        mode: "balanced",
        targetFps: "auto",
        modelId: "ecbsr_x2_m4c8_y",
      })
      .mockResolvedValueOnce({
        enabled: true,
        engine: "ecbsr",
        displayMode: "replace",
        scale: 1.5,
        sharpness: 0.65,
        overlayOpacity: 0.8,
        mode: "balanced",
        targetFps: "auto",
        modelId: "rgb_bicubic_x2",
      });
    mocks.sendMessageToTab.mockResolvedValue({
      message: "Using page-selected engine",
      hasVideo: true,
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
      displayMode: "replace",
    });

    const session = await updatePopupSettings(12, { enabled: true });

    expect(mocks.saveSettings).toHaveBeenCalledWith({ enabled: true });
    expect(mocks.sendMessageToTab).toHaveBeenCalledWith(12, {
      type: "VSR_UPDATE",
      settings: {
        enabled: true,
        engine: "tiny-cnn",
        displayMode: "overlay",
        scale: 1.5,
        sharpness: 0.65,
        overlayOpacity: 0.8,
        mode: "balanced",
        targetFps: "auto",
        modelId: "ecbsr_x2_m4c8_y",
      },
    });
    expect(mocks.saveSettings).toHaveBeenCalledWith({
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
      displayMode: "replace",
    });
    expect(session.model).toMatchObject({
      statusText: "Using page-selected engine",
      hasVideo: true,
      activeEngineLabel: "ECBSR",
      settings: {
        engine: "ecbsr",
        modelId: "rgb_bicubic_x2",
        displayMode: "replace",
      },
    });
  });

  it("sends a rescan message", async () => {
    mocks.sendMessageToTab.mockResolvedValue({
      message: "Rescanned",
      hasVideo: true,
      engine: "tiny-cnn",
      displayMode: "overlay",
    });

    await rescanPopup(7);

    expect(mocks.sendMessageToTab).toHaveBeenCalledWith(7, { type: "VSR_RESCAN" });
  });

  it("opens the extension options page", async () => {
    await launchOptionsPage();

    expect(mocks.openExtensionOptions).toHaveBeenCalledTimes(1);
  });
});
