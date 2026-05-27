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
          activeEngineLabel: "ECBSR",
          settings: {
            enabled: true,
            engine: "ecbsr",
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
      screen.getByRole("heading", { name: /video gpu super resolution/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /full settings/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: /diagnostics/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /rescan videos/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/engine/i)).toBeInTheDocument();
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
          activeEngineLabel: "ECBSR",
          settings: {
            enabled: true,
            engine: "ecbsr",
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

    fireEvent.click(screen.getByLabelText(/enable enhancement/i));
    expect(onSettingsChange).toHaveBeenCalledWith({ enabled: false });

    fireEvent.change(screen.getByLabelText(/^engine$/i), {
      target: { value: "webgpu" },
    });
    expect(onSettingsChange).toHaveBeenCalledWith({ engine: "webgpu" });

    fireEvent.change(screen.getByLabelText(/^model$/i), {
      target: { value: "rgb_bicubic_x2" },
    });
    expect(onSettingsChange).toHaveBeenCalledWith({
      modelId: "rgb_bicubic_x2",
    });

    fireEvent.click(screen.getByRole("button", { name: /rescan videos/i }));
    expect(onRescan).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: /full settings/i }));
    expect(onOpenOptions).toHaveBeenCalledTimes(1);
  });
});
