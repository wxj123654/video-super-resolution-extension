import type * as OnnxRuntimeWeb from "onnxruntime-web";

declare global {
  const __VSR_DEBUG__: boolean;

  interface Window {
    __videoGpuSuperResolutionController?: unknown;
    ort?: typeof OnnxRuntimeWeb;
  }
}

export {};
