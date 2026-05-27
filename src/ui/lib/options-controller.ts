import type { Settings } from "@src/upscaler/types";

import {
  collectEnvironmentInfo,
  collectWebGpuInfo,
  formatDiagnosticsSection,
} from "@src/shared/extension/diagnostics";
import { loadSettings, saveSettings } from "@src/shared/extension/storage";

export type OptionsModel = {
  settings: Settings;
  diagnosticsText: string;
};

export async function loadOptions(): Promise<OptionsModel> {
  const settings = await loadSettings();

  return {
    settings,
    diagnosticsText: "诊断尚未运行。点击下方按钮开始诊断。",
  };
}

export async function persistOptionsSettings(
  patch: Partial<Settings>,
): Promise<Settings> {
  return saveSettings(patch);
}

export async function runDiagnostics(): Promise<string> {
  const sections: string[] = [];

  sections.push(
    formatDiagnosticsSection("环境信息", collectEnvironmentInfo()),
  );

  const webGpuInfo = await collectWebGpuInfo();
  sections.push(formatDiagnosticsSection("WebGPU 信息", webGpuInfo));

  sections.push(
    formatDiagnosticsSection("扩展信息", {
      version: chrome.runtime.getManifest().version,
      id: chrome.runtime.id,
    }),
  );

  return sections.join("\n\n");
}
