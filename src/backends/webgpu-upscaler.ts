import type { Settings, UpscalerImpl } from "../upscaler/types";
import { requestWebGpuAdapter } from "../upscaler/webgpu-utilities";
import { modeToInt } from "../upscaler/webgl-utilities";
import webGpuShader from "../shaders/webgpu.wgsl?raw";

export class WebGpuUpscaler implements UpscalerImpl {
  private canvas: HTMLCanvasElement;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private sampler: GPUSampler | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private format: GPUTextureFormat | null = null;
  private configured = false;
  private failed = false;
  private initError: Error | null = null;
  private lastCanvasWidth = 0;
  private lastCanvasHeight = 0;
  private initPromise: Promise<void>;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.initPromise = this.init().catch((error: Error) => {
      this.failed = true;
      this.initError = error;
    });
  }

  private async init(): Promise<void> {
    this.adapter = await requestWebGpuAdapter();
    this.device = await this.adapter.requestDevice();
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw new Error("Unable to create WebGPU canvas context");
    }

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = this.device.createShaderModule({ code: webGpuShader });
    this.pipeline = await this.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vertexMain",
      },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
    this.configureContext();
  }

  render(video: HTMLVideoElement, settings: Settings): boolean {
    if (this.failed) {
      throw this.initError ?? new Error("WebGPU initialization failed");
    }
    if (!this.pipeline || !this.device || !this.context) {
      return false;
    }
    if (!("importExternalTexture" in this.device)) {
      throw new Error("WebGPU external texture is not supported");
    }

    this.configureContext();

    const width = Math.max(
      1,
      video.videoWidth || this.canvas.width || 1,
    );
    const height = Math.max(
      1,
      video.videoHeight || this.canvas.height || 1,
    );
    const params = new Float32Array([
      1 / width,
      1 / height,
      Number(settings.sharpness) || 0,
      modeToInt(settings.mode),
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer!, 0, params);

    const externalTexture = this.device.importExternalTexture({
      source: video,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.sampler! },
        { binding: 2, resource: { buffer: this.uniformBuffer! } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  private configureContext(): void {
    if (!this.context || !this.device) return;
    const width = Math.max(1, this.canvas.width || 1);
    const height = Math.max(1, this.canvas.height || 1);
    if (
      this.configured &&
      width === this.lastCanvasWidth &&
      height === this.lastCanvasHeight
    ) {
      return;
    }

    this.context.configure({
      device: this.device,
      format: this.format!,
      alphaMode: "opaque",
    });
    this.lastCanvasWidth = width;
    this.lastCanvasHeight = height;
    this.configured = true;
  }

  destroy(): void {
    if (this.uniformBuffer) {
      this.uniformBuffer.destroy();
    }
    this.pipeline = null;
    this.sampler = null;
    this.bindGroupLayout = null;
    this.context = null;
    this.device = null;
    this.adapter = null;
    this.configured = false;
  }
}
