import type { Settings, UpscalerImpl, EngineType } from "./types";
import { TinyCnnUpscaler } from "../backends/webgl-upscaler";
import { WebGpuUpscaler } from "../backends/webgpu-upscaler";
import { EcbsrOnnxUpscaler } from "../backends/onnx-upscaler";

export type { Settings, UpscalerImpl, EngineType };

export class Upscaler {
  private backend: EngineType;
  private impl: UpscalerImpl;

  constructor(
    canvas: HTMLCanvasElement,
    options: { engine: string; modelId?: string },
  ) {
    this.backend = getBackend(options.engine);
    this.impl = createBackend(canvas, this.backend, options);
  }

  render(video: HTMLVideoElement, settings: Settings): boolean {
    return this.impl.render(video, settings);
  }

  destroy(): void {
    this.impl.destroy();
  }
}

function getBackend(engine: string): EngineType {
  if (engine === "webgpu") return "webgpu";
  if (engine === "onnx") return "onnx";
  if (engine === "tiny-cnn") return "tiny-cnn";
  return "tiny-cnn";
}

function createBackend(
  canvas: HTMLCanvasElement,
  backend: EngineType,
  options: { engine: string; modelId?: string },
): UpscalerImpl {
  if (backend === "webgpu") return new WebGpuUpscaler(canvas);
  if (backend === "onnx") return new EcbsrOnnxUpscaler(canvas, options);
  return new TinyCnnUpscaler(canvas);
}
