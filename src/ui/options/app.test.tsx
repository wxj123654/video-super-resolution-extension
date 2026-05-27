import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OptionsApp } from "./app";

describe("OptionsApp", () => {
  const defaultModel = {
    settings: {
      enabled: false,
      scale: 1.5,
      sharpness: 0.65,
      mode: "balanced" as const,
      overlayOpacity: 0.8,
      displayMode: "overlay" as const,
      engine: "tiny-cnn" as const,
      modelId: "ecbsr_x2_m4c8_y",
      targetFps: "auto" as const,
    },
    diagnosticsText: "诊断尚未运行。",
  };

  it("renders diagnostics section", () => {
    render(
      <OptionsApp
        model={defaultModel}
        extensionVersion="0.3.0"
        onSettingsChange={vi.fn()}
        onRunDiagnostics={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /诊断/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/诊断尚未运行/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /运行诊断/i }),
    ).toBeInTheDocument();
  });

  it("renders general settings section", () => {
    render(
      <OptionsApp
        model={defaultModel}
        extensionVersion="0.3.0"
        onSettingsChange={vi.fn()}
        onRunDiagnostics={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /常规设置/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /精细调整/i }),
    ).toBeInTheDocument();
  });

  it("renders about section", () => {
    render(
      <OptionsApp
        model={defaultModel}
        extensionVersion="0.3.0"
        onSettingsChange={vi.fn()}
        onRunDiagnostics={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("heading", { name: /关于/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/0\.3\.0/)).toBeInTheDocument();
  });

  it("displays diagnostics text", () => {
    const modelWithDiagnostics = {
      ...defaultModel,
      diagnosticsText: "adapter: Intel Iris Xe",
    };

    render(
      <OptionsApp
        model={modelWithDiagnostics}
        extensionVersion="0.3.0"
        onSettingsChange={vi.fn()}
        onRunDiagnostics={vi.fn()}
      />,
    );

    expect(screen.getByText(/adapter: Intel Iris Xe/i)).toBeInTheDocument();
  });
});
