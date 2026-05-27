import { getAdapterInfo, requestWebGpuAdapter } from "../../upscaler/webgpu-utilities";

export type DiagnosticsData = Record<string, unknown>;

export function formatDiagnosticsSection(
  title: string,
  data: DiagnosticsData,
): string {
  const lines = [title];

  for (const [key, value] of Object.entries(data)) {
    const formatted = typeof value === "string" ? value : JSON.stringify(value);
    lines.push(`${key}: ${formatted}`);
  }

  return lines.join("\n");
}

export function collectEnvironmentInfo(): DiagnosticsData {
  return {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    cookiesEnabled: navigator.cookieEnabled,
    screenResolution: `${screen.width}x${screen.height}`,
    windowSize: `${window.innerWidth}x${window.innerHeight}`,
    devicePixelRatio: window.devicePixelRatio,
    colorDepth: screen.colorDepth,
  };
}

export async function collectWebGpuInfo(): Promise<DiagnosticsData> {
  if (!navigator.gpu) {
    return { status: "WebGPU not available" };
  }

  try {
    const adapter = await requestWebGpuAdapter({ allowSoftware: true });
    const info = getAdapterInfo(adapter);

    return {
      vendor: info.vendor || "unknown",
      architecture: info.architecture || "unknown",
      device: info.device || "unknown",
      description: info.description || "unknown",
    };
  } catch (error) {
    return {
      status: "WebGPU request failed",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
