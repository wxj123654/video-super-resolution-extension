import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PopupApp } from "./app";

describe("PopupApp", () => {
  it("renders the popup title and core controls without diagnostics", () => {
    render(
      <PopupApp
        model={{
          statusText: "Connected to current tab",
          connectionLabel: "Connected",
          hasVideo: true,
          activeEngineLabel: "ONNX",
          modelDownload: { state: "idle" as const },
          settings: {
            enabled: true,
            engine: "onnx",
            displayMode: "overlay",
            scale: 2,
            sharpness: 0.65,
            overlayOpacity: 0.8,
            mode: "balanced",
            targetFps: "auto",
            modelId: "ecbsr_x2_m4c8_y",
          },
        }}
        onSettingsChange={vi.fn()}
        onRescan={vi.fn()}
        onOpenOptions={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /视频 gpu 超分辨率/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /完整设置/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /诊断/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /重新扫描视频/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/引擎/i)).toBeInTheDocument();
    expect(screen.getByText("Connected")).toBeInTheDocument();
  });

  it("triggers settings and action handlers from popup controls", () => {
    const onSettingsChange = vi.fn();
    const onRescan = vi.fn();
    const onOpenOptions = vi.fn();

    render(
      <PopupApp
        model={{
          statusText: "Connected to current tab",
          connectionLabel: "Connected",
          hasVideo: true,
          activeEngineLabel: "ONNX",
          modelDownload: { state: "idle" as const },
          settings: {
            enabled: true,
            engine: "onnx",
            displayMode: "overlay",
            scale: 2,
            sharpness: 0.65,
            overlayOpacity: 0.8,
            mode: "balanced",
            targetFps: "auto",
            modelId: "ecbsr_x2_m4c8_y",
          },
        }}
        onSettingsChange={onSettingsChange}
        onRescan={onRescan}
        onOpenOptions={onOpenOptions}
      />,
    );

    fireEvent.click(screen.getByLabelText(/enable enhancement/i)); // aria-label kept in English
    expect(onSettingsChange).toHaveBeenCalledWith({ enabled: false });

    fireEvent.change(screen.getByLabelText(/^引擎$/i), {
      target: { value: "webgpu" },
    });
    expect(onSettingsChange).toHaveBeenCalledWith({ engine: "webgpu" });

    fireEvent.change(screen.getByLabelText(/^模型$/i), {
      target: { value: "rgb_bicubic_x2" },
    });
    expect(onSettingsChange).toHaveBeenCalledWith({
      modelId: "rgb_bicubic_x2",
    });

    fireEvent.click(screen.getByRole("button", { name: /重新扫描视频/i }));
    expect(onRescan).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /完整设置/i }));
    expect(onOpenOptions).toHaveBeenCalledTimes(1);
  });
});
