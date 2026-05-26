import type { AdapterInfo, WebGpuAdapterOptions } from "../upscaler/types";

export async function requestOnnxWebGpuAdapter(
  options: WebGpuAdapterOptions = {},
): Promise<GPUAdapter> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available");
  }

  const { allowSoftware = false, preferCompatibility = false } = options;
  const attempts: Array<[string, GPURequestAdapterOptions]> =
    preferCompatibility
      ? [
          [
            "compatibility",
            {
              powerPreference: "low-power",
              featureLevel: "compatibility",
              forceFallbackAdapter: false,
            } as GPURequestAdapterOptions,
          ],
          ["default", { forceFallbackAdapter: false }],
          [
            "high-performance",
            {
              powerPreference: "high-performance",
              forceFallbackAdapter: false,
            },
          ],
        ]
      : [
          [
            "high-performance",
            {
              powerPreference: "high-performance",
              forceFallbackAdapter: false,
            },
          ],
          ["default", { forceFallbackAdapter: false }],
          [
            "compatibility",
            {
              powerPreference: "high-performance",
              featureLevel: "compatibility",
              forceFallbackAdapter: false,
            } as GPURequestAdapterOptions,
          ],
        ];
  const failures: string[] = [];

  for (const [name, adapterOptions] of attempts) {
    try {
      const adapter = await navigator.gpu.requestAdapter(adapterOptions);
      if (adapter) {
        const info = getAdapterInfo(adapter);
        if (!allowSoftware && isSoftwareAdapter(adapter, info)) {
          failures.push(
            `${name}: software adapter (${info.description || info.architecture || "unknown"})`,
          );
          continue;
        }
        return adapter;
      }
      failures.push(`${name}: null`);
    } catch (error) {
      failures.push(
        `${name}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const label = allowSoftware
    ? "WebGPU adapter"
    : "hardware WebGPU adapter";
  throw new Error(
    `Unable to request ${label}. ${failures.join("; ")}`,
  );
}

function getAdapterInfo(adapter: GPUAdapter): AdapterInfo {
  const info = adapter.info || ({} as GPUAdapterInfo);
  return {
    vendor: info.vendor || "",
    architecture: info.architecture || "",
    device: info.device || "",
    description: info.description || "",
  };
}

function isSoftwareAdapter(
  adapter: GPUAdapter,
  info: AdapterInfo = getAdapterInfo(adapter),
): boolean {
  if ((adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter) {
    return true;
  }

  const text =
    `${info.vendor} ${info.architecture} ${info.device} ${info.description}`.toLowerCase();
  return text.includes("swiftshader") || text.includes("software");
}
