import type * as OnnxRuntimeWeb from "onnxruntime-web";
import type {
  Settings,
  UpscalerImpl,
  OnnxModelDefinition,
  OnnxInputPacking,
} from "../upscaler/types";
import {
  requestWebGpuAdapter,
  configureWebGpuContext,
  type WebGpuContextState,
} from "../upscaler/webgpu-utilities";
import { getOnnxModelDefinition } from "./onnx-models";
import lumaShaderCode from "../shaders/onnx-luma.wgsl?raw";
import compositeShaderCode from "../shaders/onnx-composite.wgsl?raw";
import videoCopyShaderCode from "../shaders/onnx-video-copy.wgsl?raw";
import packLumaShaderCode from "../shaders/onnx-pack-luma.wgsl?raw";
import packRgbShaderCode from "../shaders/onnx-pack-rgb.wgsl?raw";
import rgbUnpackShaderCode from "../shaders/onnx-rgb-unpack.wgsl?raw";
import presentShaderCode from "../shaders/onnx-present.wgsl?raw";

type OrtRuntime = typeof OnnxRuntimeWeb;
type SupportedTensorData = Float32Array | Float64Array | Uint8Array;
type OnnxPathMode = "gpu" | "cpu";
type OnnxCompositePath = "webgpu" | "2d";

interface SessionBundle {
  session: OnnxRuntimeWeb.InferenceSession;
  inputName: string;
  outputName: string;
}

interface EcbsrStats {
  runs: number;
  preMs: number;
  inferMs: number;
  postMs: number;
  totalMs: number;
  lastLogAt: number;
}

interface EcbsrStatsSample {
  preMs: number;
  inferMs: number;
  postMs: number;
  totalMs: number;
  width: number;
  height: number;
  outputWidth: number;
  outputHeight: number;
  inputPath: OnnxPathMode;
  outputPath: OnnxPathMode;
  compositePath: OnnxCompositePath;
}

const ORT_STATE = {
  configured: false,
  sessionPromises: new Map<string, Promise<SessionBundle>>(),
  adapter: null as GPUAdapter | null,
  sessionCount: 0,
};

export async function cleanupOrtState(): Promise<void> {
  const sessions = await Promise.allSettled(
    [...ORT_STATE.sessionPromises.values()],
  );
  for (const result of sessions) {
    if (result.status === "fulfilled") {
      await result.value.session.release();
    }
  }
  ORT_STATE.sessionPromises.clear();
  ORT_STATE.adapter = null;
  ORT_STATE.configured = false;
  ORT_STATE.sessionCount = 0;
}

const ECBSR_DEBUG = false;
const ECBSR_MAX_INPUT_PIXELS = 1920 * 1080;
const GPU_PACK_WORKGROUP_SIZE = 8;

let _ortAssetUrls: { mjs: string; wasm: string } | null = null;

function getOrtAssetUrls() {
  if (!_ortAssetUrls) {
    _ortAssetUrls = {
      mjs: chrome.runtime.getURL("vendor/onnxruntime/ort-wasm-simd-threaded.mjs"),
      wasm: chrome.runtime.getURL("vendor/onnxruntime/ort-wasm-simd-threaded.wasm"),
    };
  }
  return _ortAssetUrls;
}

function extractLumaCpu(rgba: Uint8ClampedArray): Float32Array {
  const pixels = rgba.length >> 2;
  const luma = new Float32Array(pixels);
  const src = new Uint32Array(rgba.buffer, rgba.byteOffset, pixels);
  for (let i = 0; i < pixels; i++) {
    const p = src[i];
    luma[i] =
      (0.299 * (p & 0xff) + 0.587 * ((p >> 8) & 0xff) + 0.114 * ((p >> 16) & 0xff)) /
      255;
  }
  return luma;
}

function extractRgbCpu(rgba: Uint8ClampedArray): Float32Array {
  const pixels = rgba.length >> 2;
  const rgb = new Float32Array(pixels * 3);
  let rOffset = 0;
  let gOffset = pixels;
  let bOffset = pixels * 2;

  for (let i = 0; i < rgba.length; i += 4) {
    rgb[rOffset++] = rgba[i] / 255;
    rgb[gOffset++] = rgba[i + 1] / 255;
    rgb[bOffset++] = rgba[i + 2] / 255;
  }

  return rgb;
}

class WebGpuLumaCompositor {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private lumaTexture: GPUTexture | null = null;
  private lumaWidth = 0;
  private lumaHeight = 0;
  private sampler: GPUSampler | null = null;
  private contextState: WebGpuContextState = {
    context: null as unknown as GPUCanvasContext,
    device: null as unknown as GPUDevice,
    format: null as unknown as GPUTextureFormat,
    canvas: null as unknown as HTMLCanvasElement,
    configured: false,
    lastCanvasWidth: 0,
    lastCanvasHeight: 0,
  };
  ready = false;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.canvas = canvas;
    this.device = device;
    this.contextState.canvas = canvas;
    this.contextState.device = device;
  }

  async init(): Promise<void> {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) throw new Error("WebGPU canvas context unavailable");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.contextState.context = this.context;
    this.contextState.format = this.format;
    this.configureContext();

    const module = this.device.createShaderModule({
      code: compositeShaderCode,
    });
    this.pipeline = await this.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vertexMain" },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.ready = true;
  }

  uploadLumaFromBuffer(
    gpuBuffer: GPUBuffer,
    width: number,
    height: number,
  ): void {
    this.ensureLumaTexture(width, height);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToTexture(
      { buffer: gpuBuffer, bytesPerRow: width * 4 },
      { texture: this.lumaTexture! },
      { width, height },
    );
    this.device.queue.submit([encoder.finish()]);
  }

  uploadLumaFromData(
    data: Float32Array,
    width: number,
    height: number,
  ): void {
    this.ensureLumaTexture(width, height);
    const bytes = new Uint8Array(data.byteLength);
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    this.device.queue.writeTexture(
      { texture: this.lumaTexture! },
      bytes,
      { bytesPerRow: width * 4 },
      { width, height },
    );
  }

  render(video: HTMLVideoElement): boolean {
    if (!this.ready || !this.lumaTexture) return false;
    this.configureContext();

    const externalTexture = this.device.importExternalTexture({
      source: video,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.sampler! },
        { binding: 2, resource: this.lumaTexture.createView() },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context!.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  clear(): void {
    if (!this.context) return;
    this.configureContext();
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
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    if (this.lumaTexture) {
      this.lumaTexture.destroy();
      this.lumaTexture = null;
    }
    this.ready = false;
    this.contextState.configured = false;
  }

  private configureContext(): void {
    if (!this.context || !this.format) return;
    configureWebGpuContext(this.contextState);
  }

  private ensureLumaTexture(width: number, height: number): void {
    if (
      !this.lumaTexture ||
      this.lumaWidth !== width ||
      this.lumaHeight !== height
    ) {
      if (this.lumaTexture) this.lumaTexture.destroy();
      this.lumaTexture = this.device.createTexture({
        size: [width, height],
        format: "r32float",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lumaWidth = width;
      this.lumaHeight = height;
    }
  }
}

class RgbCanvasCompositor {
  private canvas: HTMLCanvasElement;
  private context: CanvasRenderingContext2D | null = null;
  private frameCanvas: HTMLCanvasElement;
  private frameContext: CanvasRenderingContext2D | null = null;
  private ready = false;
  private hasFrame = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.frameCanvas = document.createElement("canvas");
  }

  init(): void {
    this.context = this.canvas.getContext("2d");
    this.frameContext = this.frameCanvas.getContext("2d");
    if (!this.context || !this.frameContext) {
      throw new Error("2D canvas context unavailable");
    }
    this.ready = true;
  }

  uploadPlanarRgbData(
    data: SupportedTensorData,
    width: number,
    height: number,
  ): void {
    if (!this.ready || !this.frameContext) {
      throw new Error("RGB compositor is not ready");
    }

    if (
      this.frameCanvas.width !== width ||
      this.frameCanvas.height !== height
    ) {
      this.frameCanvas.width = width;
      this.frameCanvas.height = height;
    }

    const imageData = this.frameContext.createImageData(width, height);
    const planeSize = width * height;

    for (let index = 0; index < planeSize; index++) {
      const pixelOffset = index * 4;
      imageData.data[pixelOffset] = toByte(data[index]);
      imageData.data[pixelOffset + 1] = toByte(data[planeSize + index]);
      imageData.data[pixelOffset + 2] = toByte(data[planeSize * 2 + index]);
      imageData.data[pixelOffset + 3] = 255;
    }

    this.frameContext.putImageData(imageData, 0, 0);
    this.hasFrame = true;
  }

  render(): boolean {
    if (!this.ready || !this.hasFrame || !this.context) return false;
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.drawImage(
      this.frameCanvas,
      0,
      0,
      this.canvas.width,
      this.canvas.height,
    );
    return true;
  }

  clear(): void {
    if (!this.context) return;
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.hasFrame = false;
  }

  destroy(): void {
    this.clear();
    this.context = null;
    this.frameContext = null;
    this.ready = false;
  }
}

class VideoFrameTextureRenderer {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private sampler: GPUSampler | null = null;
  private frameTexture: GPUTexture | null = null;
  private frameWidth = 0;
  private frameHeight = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<void> {
    const module = this.device.createShaderModule({
      code: videoCopyShaderCode,
    });
    this.pipeline = await this.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vertexMain" },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
  }

  encodeCapturePass(
    encoder: GPUCommandEncoder,
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): void {
    if (!this.pipeline || !this.sampler) {
      throw new Error("Video frame renderer is not initialized");
    }
    this.ensureFrameTexture(width, height);

    const externalTexture = this.device.importExternalTexture({
      source: video,
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.sampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.frameTexture!.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  getTextureView(): GPUTextureView {
    if (!this.frameTexture) {
      throw new Error("Video frame texture is unavailable");
    }
    return this.frameTexture.createView();
  }

  destroy(): void {
    if (this.frameTexture) {
      this.frameTexture.destroy();
      this.frameTexture = null;
    }
  }

  private ensureFrameTexture(width: number, height: number): void {
    if (
      !this.frameTexture ||
      this.frameWidth !== width ||
      this.frameHeight !== height
    ) {
      if (this.frameTexture) {
        this.frameTexture.destroy();
      }
      this.frameTexture = this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.frameWidth = width;
      this.frameHeight = height;
    }
  }
}

class WebGpuInputPacker {
  private device: GPUDevice;
  private frameRenderer: VideoFrameTextureRenderer;
  private paramsBuffer: GPUBuffer | null = null;
  private lumaPipeline: GPUComputePipeline | null = null;
  private rgbPipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.frameRenderer = new VideoFrameTextureRenderer(device);
  }

  async init(): Promise<void> {
    await this.frameRenderer.init();
    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const lumaModule = this.device.createShaderModule({
      code: packLumaShaderCode,
    });
    this.lumaPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: lumaModule, entryPoint: "computeMain" },
    });

    const rgbModule = this.device.createShaderModule({
      code: packRgbShaderCode,
    });
    this.rgbPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: rgbModule, entryPoint: "computeMain" },
    });
  }

  async packVideoToBuffer(
    video: HTMLVideoElement,
    width: number,
    height: number,
    packing: OnnxInputPacking,
    dstBuffer: GPUBuffer,
  ): Promise<void> {
    if (!this.paramsBuffer || !this.lumaPipeline || !this.rgbPipeline) {
      throw new Error("WebGPU input packer is not initialized");
    }

    const pipeline =
      packing === "luma_f32_planar" ? this.lumaPipeline : this.rgbPipeline;
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([width, height, 0, 0]),
    );

    const encoder = this.device.createCommandEncoder();
    this.frameRenderer.encodeCapturePass(encoder, video, width, height);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.frameRenderer.getTextureView() },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / GPU_PACK_WORKGROUP_SIZE),
      Math.ceil(height / GPU_PACK_WORKGROUP_SIZE),
    );
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    if (this.paramsBuffer) {
      this.paramsBuffer.destroy();
      this.paramsBuffer = null;
    }
    this.frameRenderer.destroy();
  }
}

class WebGpuRgbCompositor {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private unpackPipeline: GPURenderPipeline | null = null;
  private presentPipeline: GPURenderPipeline | null = null;
  private sampler: GPUSampler | null = null;
  private paramsBuffer: GPUBuffer | null = null;
  private outputTexture: GPUTexture | null = null;
  private outputWidth = 0;
  private outputHeight = 0;
  private hasFrame = false;
  private ready = false;
  private contextState: WebGpuContextState = {
    context: null as unknown as GPUCanvasContext,
    device: null as unknown as GPUDevice,
    format: null as unknown as GPUTextureFormat,
    canvas: null as unknown as HTMLCanvasElement,
    configured: false,
    lastCanvasWidth: 0,
    lastCanvasHeight: 0,
  };

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.canvas = canvas;
    this.device = device;
    this.contextState.canvas = canvas;
    this.contextState.device = device;
  }

  async init(): Promise<void> {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      throw new Error("WebGPU canvas context unavailable");
    }
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.contextState.context = this.context;
    this.contextState.format = this.format;
    this.configureContext();

    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    const unpackModule = this.device.createShaderModule({
      code: rgbUnpackShaderCode,
    });
    this.unpackPipeline = await this.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: unpackModule, entryPoint: "vertexMain" },
      fragment: {
        module: unpackModule,
        entryPoint: "fragmentMain",
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { topology: "triangle-list" },
    });

    const presentModule = this.device.createShaderModule({
      code: presentShaderCode,
    });
    this.presentPipeline = await this.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "vertexMain" },
      fragment: {
        module: presentModule,
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.ready = true;
  }

  uploadRgbFromBuffer(
    gpuBuffer: GPUBuffer,
    width: number,
    height: number,
  ): void {
    if (
      !this.ready ||
      !this.paramsBuffer ||
      !this.unpackPipeline ||
      !this.sampler
    ) {
      throw new Error("WebGPU RGB compositor is not ready");
    }

    this.ensureOutputTexture(width, height);
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([width, height, 0, 0]),
    );

    const bindGroup = this.device.createBindGroup({
      layout: this.unpackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuBuffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.outputTexture!.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.unpackPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.hasFrame = true;
  }

  render(): boolean {
    if (
      !this.ready ||
      !this.context ||
      !this.presentPipeline ||
      !this.sampler ||
      !this.outputTexture ||
      !this.hasFrame
    ) {
      return false;
    }

    this.configureContext();

    const bindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.outputTexture.createView() },
        { binding: 1, resource: this.sampler },
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
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  clear(): void {
    if (!this.context) return;
    this.configureContext();
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
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.hasFrame = false;
  }

  destroy(): void {
    if (this.paramsBuffer) {
      this.paramsBuffer.destroy();
      this.paramsBuffer = null;
    }
    if (this.outputTexture) {
      this.outputTexture.destroy();
      this.outputTexture = null;
    }
    this.ready = false;
    this.hasFrame = false;
    this.contextState.configured = false;
  }

  private configureContext(): void {
    if (!this.context || !this.format) return;
    configureWebGpuContext(this.contextState);
  }

  private ensureOutputTexture(width: number, height: number): void {
    if (
      !this.outputTexture ||
      this.outputWidth !== width ||
      this.outputHeight !== height
    ) {
      if (this.outputTexture) {
        this.outputTexture.destroy();
      }
      this.outputTexture = this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.outputWidth = width;
      this.outputHeight = height;
    }
  }
}

export class EcbsrOnnxUpscaler implements UpscalerImpl {
  private canvas: HTMLCanvasElement;
  private model: OnnxModelDefinition;
  private lumaCompositor: WebGpuLumaCompositor | null = null;
  private rgbCompositor: RgbCanvasCompositor | null = null;
  private gpuRgbCompositor: WebGpuRgbCompositor | null = null;
  private inputPacker: WebGpuInputPacker | null = null;

  private inputCanvas: HTMLCanvasElement;
  private inputContext: CanvasRenderingContext2D;

  private session: OnnxRuntimeWeb.InferenceSession | null = null;
  private inputName = "";
  private outputName = "";
  private pendingInference: Promise<void> | null = null;
  private lastQueuedTime = -1;
  private outputReady = false;
  private failed = false;
  private initError: Error | null = null;

  private gpuDevice: GPUDevice | null = null;

  private inputGpuBuffer: GPUBuffer | null = null;
  private inputTensor: OnnxRuntimeWeb.Tensor | null = null;
  private inputBufferSize = 0;
  private inputTensorWidth = 0;
  private inputTensorHeight = 0;
  private inputTensorChannels = 0;

  private outputGpuBuffer: GPUBuffer | null = null;
  private outputTensor: OnnxRuntimeWeb.Tensor | null = null;
  private outputBufferSize = 0;
  private outputTensorWidth = 0;
  private outputTensorHeight = 0;
  private outputTensorChannels = 0;

  private stats: EcbsrStats = {
    runs: 0,
    preMs: 0,
    inferMs: 0,
    postMs: 0,
    totalMs: 0,
    lastLogAt: 0,
  };
  private lastPendingLogAt = 0;
  private modelInputWidth = 0;
  private modelInputHeight = 0;
  private outputWidth = 0;
  private outputHeight = 0;

  private gpuInputEnabled = false;
  private gpuOutputEnabled = false;
  private activeInputPath: OnnxPathMode = "cpu";
  private activeOutputPath: OnnxPathMode = "cpu";
  private activeCompositePath: OnnxCompositePath = "2d";

  constructor(
    canvas: HTMLCanvasElement,
    options: { modelId?: string } = {},
  ) {
    this.canvas = canvas;
    this.model = getOnnxModelDefinition(options.modelId);
    this.inputCanvas = document.createElement("canvas");
    this.inputContext = this.inputCanvas.getContext("2d", {
      willReadFrequently: true,
    })!;
    void this.init().catch((error: Error) => {
      this.failed = true;
      this.initError = error;
      console.error(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] init failed`,
        error,
      );
    });
  }

  private async init(): Promise<void> {
    if (ECBSR_DEBUG) {
      console.info(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] init start`,
      );
    }

    const ortAdapter = await requestWebGpuAdapter({
      allowSoftware: true,
      preferCompatibility: false,
    });
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.InferenceSession || !ortRuntime.Tensor) {
      throw new Error("onnxruntime-web is not available");
    }
    configureOrtRuntime(ortRuntime, ortAdapter);

    const sessionBundle = await getSharedSession(ortRuntime, this.model);
    this.session = sessionBundle.session;
    this.inputName = sessionBundle.inputName;
    this.outputName = sessionBundle.outputName;

    const sharedDevice = ortRuntime.env.webgpu.device;
    if (!sharedDevice) {
      throw new Error("ONNX Runtime did not expose a WebGPU device");
    }
    this.gpuDevice = sharedDevice as unknown as GPUDevice;

    this.gpuDevice.lost.then((info) => {
      console.error(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU device lost:`,
        info.message,
      );
      this.failed = true;
      this.initError = new Error(`GPU device lost: ${info.message}`);
      void cleanupOrtState();
    });

    try {
      this.inputPacker = new WebGpuInputPacker(this.gpuDevice);
      await this.inputPacker.init();
    } catch (error) {
      this.inputPacker = null;
      console.warn(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU input packer unavailable, using CPU input fallback:`,
        (error as Error).message,
      );
    }

    if (this.model.compositeMode === "luma_replace") {
      this.lumaCompositor = new WebGpuLumaCompositor(this.canvas, this.gpuDevice);
      await this.lumaCompositor.init();
      this.activeCompositePath = "webgpu";
    } else if (this.model.compositeMode === "rgb_replace") {
      try {
        this.gpuRgbCompositor = new WebGpuRgbCompositor(
          this.canvas,
          this.gpuDevice,
        );
        await this.gpuRgbCompositor.init();
        this.activeCompositePath = "webgpu";
      } catch (error) {
        this.gpuRgbCompositor = null;
        console.warn(
          `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU RGB compositor unavailable, using 2D fallback:`,
          (error as Error).message,
        );
      }
      if (!this.gpuRgbCompositor) {
        this.rgbCompositor = new RgbCanvasCompositor(this.canvas);
        this.rgbCompositor.init();
        this.activeCompositePath = "2d";
      }
    } else {
      throw new Error(
        `Unsupported composite mode: ${this.model.compositeMode} for model ${this.model.id}`,
      );
    }

    if (ECBSR_DEBUG) {
      console.info(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] session ready input="${this.inputName}" output="${this.outputName}"`,
      );
    }
  }

  render(video: HTMLVideoElement, _settings: Settings): boolean {
    if (this.failed) {
      throw this.initError ?? new Error("ONNX model initialization failed");
    }
    if (!this.session) {
      return false;
    }

    const width = Math.max(1, video.videoWidth || 1);
    const height = Math.max(1, video.videoHeight || 1);
    this.ensureWorkingSize(width, height);

    if (!this.pendingInference && video.currentTime !== this.lastQueuedTime) {
      this.lastQueuedTime = video.currentTime;
      this.pendingInference = this.runInference(video).finally(() => {
        this.pendingInference = null;
      });
    } else if (this.pendingInference) {
      const now = performance.now();
      if (now - this.lastPendingLogAt > 2000) {
        if (ECBSR_DEBUG) {
          console.info(
            `[Video GPU Super Resolution][ONNX][${this.model.id}] inference still pending currentTime=${video.currentTime.toFixed(3)} lastQueued=${this.lastQueuedTime.toFixed(3)}`,
          );
        }
        this.lastPendingLogAt = now;
      }
    }

    if (!this.outputReady) {
      return false;
    }

    return this.lumaCompositor
      ? this.lumaCompositor.render(video)
      : this.gpuRgbCompositor
        ? this.gpuRgbCompositor.render()
        : this.rgbCompositor?.render() ?? false;
  }

  private ensureWorkingSize(width: number, height: number): void {
    const scale = this.model.sizePolicy.fixedScale ?? this.model.scale;
    const targetOutputWidth = Math.max(
      1,
      Math.min(this.canvas.width || width * scale, width * scale),
    );
    const targetOutputHeight = Math.max(
      1,
      Math.min(this.canvas.height || height * scale, height * scale),
    );
    const scaleByDisplay = Math.max(
      0.25,
      Math.min(
        1,
        targetOutputWidth / (width * scale),
        targetOutputHeight / (height * scale),
      ),
    );
    const pixelScale = Math.min(
      1,
      Math.sqrt(ECBSR_MAX_INPUT_PIXELS / Math.max(1, width * height)),
    );
    const workingScale = Math.max(0.25, Math.min(scaleByDisplay, pixelScale));
    const rawWidth = Math.max(1, Math.round(width * workingScale));
    const rawHeight = Math.max(1, Math.round(height * workingScale));
    const inputWidth = alignDimension(
      rawWidth,
      this.model.sizePolicy.widthAlign,
      this.model.sizePolicy.minInputWidth ?? 1,
    );
    const inputHeight = alignDimension(
      rawHeight,
      this.model.sizePolicy.heightAlign,
      this.model.sizePolicy.minInputHeight ?? 1,
    );

    this.modelInputWidth = inputWidth;
    this.modelInputHeight = inputHeight;

    if (this.inputCanvas.width !== inputWidth) this.inputCanvas.width = inputWidth;
    if (this.inputCanvas.height !== inputHeight) {
      this.inputCanvas.height = inputHeight;
    }

    this.outputWidth = inputWidth * this.model.scale;
    this.outputHeight = inputHeight * this.model.scale;

    this.gpuInputEnabled = this.tryEnsureInputTensor(
      inputWidth,
      inputHeight,
      this.model.inputChannels,
    );
    this.gpuOutputEnabled = this.tryEnsureOutputTensor(
      this.outputWidth,
      this.outputHeight,
      this.model.outputChannels,
    );
  }

  private tryEnsureInputTensor(
    width: number,
    height: number,
    channels: number,
  ): boolean {
    if (!this.gpuDevice || !this.inputPacker) {
      return false;
    }
    try {
      this.ensureGpuTensorResource("input", width, height, channels);
      return true;
    } catch (error) {
      console.warn(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU input tensor unavailable, using CPU fallback:`,
        (error as Error).message,
      );
      return false;
    }
  }

  private tryEnsureOutputTensor(
    width: number,
    height: number,
    channels: number,
  ): boolean {
    if (!this.gpuDevice) {
      return false;
    }
    try {
      this.ensureGpuTensorResource("output", width, height, channels);
      return true;
    } catch (error) {
      console.warn(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU output tensor unavailable, using CPU fallback:`,
        (error as Error).message,
      );
      return false;
    }
  }

  private ensureGpuTensorResource(
    kind: "input" | "output",
    width: number,
    height: number,
    channels: number,
  ): void {
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor || !this.gpuDevice) {
      throw new Error("onnxruntime-web Tensor API is unavailable");
    }

    const size = width * height * channels * 4;
    const alignedSize = Math.ceil(size / 16) * 16;

    const currentBuffer =
      kind === "input" ? this.inputGpuBuffer : this.outputGpuBuffer;
    const currentTensor = kind === "input" ? this.inputTensor : this.outputTensor;
    const currentWidth =
      kind === "input" ? this.inputTensorWidth : this.outputTensorWidth;
    const currentHeight =
      kind === "input" ? this.inputTensorHeight : this.outputTensorHeight;
    const currentChannels =
      kind === "input" ? this.inputTensorChannels : this.outputTensorChannels;
    const currentSize =
      kind === "input" ? this.inputBufferSize : this.outputBufferSize;

    if (
      currentBuffer &&
      currentTensor &&
      currentWidth === width &&
      currentHeight === height &&
      currentChannels === channels &&
      currentSize >= size
    ) {
      return;
    }

    if (kind === "input") {
      this.inputGpuBuffer?.destroy();
      this.inputGpuBuffer = this.gpuDevice.createBuffer({
        size: alignedSize,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      this.inputTensor = ortRuntime.Tensor.fromGpuBuffer(this.inputGpuBuffer, {
        dataType: "float32",
        dims: [1, channels, height, width],
      });
      this.inputBufferSize = size;
      this.inputTensorWidth = width;
      this.inputTensorHeight = height;
      this.inputTensorChannels = channels;
      return;
    }

    this.outputGpuBuffer?.destroy();
    this.outputGpuBuffer = this.gpuDevice.createBuffer({
      size: alignedSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.outputTensor = ortRuntime.Tensor.fromGpuBuffer(this.outputGpuBuffer, {
      dataType: "float32",
      dims: [1, channels, height, width],
    });
    this.outputBufferSize = size;
    this.outputTensorWidth = width;
    this.outputTensorHeight = height;
    this.outputTensorChannels = channels;
  }

  private async runInference(video: HTMLVideoElement): Promise<void> {
    const startedAt = performance.now();
    const width = this.modelInputWidth;
    const height = this.modelInputHeight;
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor || !this.session) {
      throw new Error("onnxruntime-web session is unavailable");
    }

    const preStartedAt = performance.now();
    const inputTensor = await this.createInputTensor(
      ortRuntime,
      video,
      width,
      height,
    );
    const preEndedAt = performance.now();

    let useGpuOutput = Boolean(
      this.gpuOutputEnabled && this.outputTensor && this.outputGpuBuffer,
    );
    this.activeOutputPath = useGpuOutput ? "gpu" : "cpu";

    const inferStartedAt = performance.now();
    let results: Record<string, OnnxRuntimeWeb.Tensor>;
    if (useGpuOutput) {
      try {
        results = await this.session.run(
          { [this.inputName]: inputTensor },
          { [this.outputName]: this.outputTensor! },
        );
      } catch (error) {
        console.warn(
          `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU output binding failed, falling back to CPU output:`,
          (error as Error).message,
        );
        useGpuOutput = false;
        this.activeOutputPath = "cpu";
        results = await this.session.run({ [this.inputName]: inputTensor });
      }
    } else {
      results = await this.session.run({ [this.inputName]: inputTensor });
    }
    const inferEndedAt = performance.now();

    const postStartedAt = performance.now();
    if (this.model.compositeMode === "luma_replace") {
      await this.handleLumaOutput(results, useGpuOutput);
    } else if (this.model.compositeMode === "rgb_replace") {
      await this.handleRgbOutput(results, useGpuOutput);
    } else {
      throw new Error(
        `Unsupported composite mode: ${this.model.compositeMode} for model ${this.model.id}`,
      );
    }

    const finishedAt = performance.now();
    this.outputReady = true;

    this.recordStats({
      preMs: preEndedAt - preStartedAt,
      inferMs: inferEndedAt - inferStartedAt,
      postMs: finishedAt - postStartedAt,
      totalMs: finishedAt - startedAt,
      width,
      height,
      outputWidth: this.outputWidth,
      outputHeight: this.outputHeight,
      inputPath: this.activeInputPath,
      outputPath: this.activeOutputPath,
      compositePath: this.activeCompositePath,
    });
  }

  private async handleLumaOutput(
    results: Record<string, OnnxRuntimeWeb.Tensor>,
    useGpuOutput: boolean,
  ): Promise<void> {
    if (!this.lumaCompositor) {
      throw new Error("Luma compositor is unavailable");
    }

    this.activeCompositePath = "webgpu";
    if (useGpuOutput && this.outputGpuBuffer) {
      this.lumaCompositor.uploadLumaFromBuffer(
        this.outputGpuBuffer,
        this.outputWidth,
        this.outputHeight,
      );
      return;
    }

    const outputTensor = await this.resolveOutputTensor(results);
    const data = toFloat32Array(await getTensorDataAsync(outputTensor));
    this.lumaCompositor.uploadLumaFromData(
      data,
      this.outputWidth,
      this.outputHeight,
    );
  }

  private async handleRgbOutput(
    results: Record<string, OnnxRuntimeWeb.Tensor>,
    useGpuOutput: boolean,
  ): Promise<void> {
    if (useGpuOutput && this.outputGpuBuffer && this.gpuRgbCompositor) {
      this.gpuRgbCompositor.uploadRgbFromBuffer(
        this.outputGpuBuffer,
        this.outputWidth,
        this.outputHeight,
      );
      this.activeCompositePath = "webgpu";
      return;
    }

    const outputTensor = await this.resolveOutputTensor(results);
    const data = await getTensorDataAsync(outputTensor);
    if (!this.rgbCompositor) {
      throw new Error("RGB fallback compositor is unavailable");
    }
    this.rgbCompositor.uploadPlanarRgbData(
      data,
      this.outputWidth,
      this.outputHeight,
    );
    this.activeCompositePath = "2d";
  }

  private async resolveOutputTensor(
    results: Record<string, OnnxRuntimeWeb.Tensor>,
  ): Promise<OnnxRuntimeWeb.Tensor> {
    const outputTensor = results[this.outputName] ?? this.outputTensor;
    if (!outputTensor) {
      throw new Error(`Missing ONNX output tensor "${this.outputName}"`);
    }
    return outputTensor;
  }

  private async createInputTensor(
    ortRuntime: OrtRuntime,
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Promise<OnnxRuntimeWeb.Tensor> {
    if (
      this.gpuInputEnabled &&
      this.inputPacker &&
      this.inputGpuBuffer &&
      this.inputTensor
    ) {
      try {
        await this.inputPacker.packVideoToBuffer(
          video,
          width,
          height,
          this.model.inputPacking,
          this.inputGpuBuffer,
        );
        this.activeInputPath = "gpu";
        return this.inputTensor;
      } catch (error) {
        console.warn(
          `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU input pack failed, using CPU fallback:`,
          (error as Error).message,
        );
        this.gpuInputEnabled = false;
      }
    }

    this.activeInputPath = "cpu";
    if (this.model.inputPacking === "rgb_f32_planar") {
      const rgbData = this.extractRgbCpuPath(video, width, height);
      return new ortRuntime.Tensor("float32", rgbData, [1, 3, height, width]);
    }

    const yData = this.extractLumaCpuPath(video, width, height);
    return new ortRuntime.Tensor("float32", yData, [1, 1, height, width]);
  }

  private extractLumaCpuPath(
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Float32Array {
    this.inputContext.drawImage(video, 0, 0, width, height);
    const sourceImage = this.inputContext.getImageData(0, 0, width, height);
    return extractLumaCpu(sourceImage.data);
  }

  private extractRgbCpuPath(
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Float32Array {
    this.inputContext.drawImage(video, 0, 0, width, height);
    const sourceImage = this.inputContext.getImageData(0, 0, width, height);
    return extractRgbCpu(sourceImage.data);
  }

  destroy(): void {
    this.pendingInference = null;
    this.session = null;
    this.outputReady = false;
    this.lumaCompositor?.clear();
    this.lumaCompositor?.destroy();
    this.gpuRgbCompositor?.clear();
    this.gpuRgbCompositor?.destroy();
    this.rgbCompositor?.clear();
    this.rgbCompositor?.destroy();
    this.lumaCompositor = null;
    this.gpuRgbCompositor = null;
    this.rgbCompositor = null;
    this.inputPacker?.destroy();
    this.inputPacker = null;
    this.inputGpuBuffer?.destroy();
    this.outputGpuBuffer?.destroy();
    this.inputGpuBuffer = null;
    this.outputGpuBuffer = null;
    this.inputTensor = null;
    this.outputTensor = null;
    this.gpuDevice = null;
    this.gpuInputEnabled = false;
    this.gpuOutputEnabled = false;
  }

  private recordStats(sample: EcbsrStatsSample): void {
    this.stats.runs += 1;
    this.stats.preMs += sample.preMs;
    this.stats.inferMs += sample.inferMs;
    this.stats.postMs += sample.postMs;
    this.stats.totalMs += sample.totalMs;

    const now = performance.now();
    if (now - this.stats.lastLogAt < 2000 && this.stats.runs < 6) {
      return;
    }

    const count = this.stats.runs;
    console.info(
      `[Video GPU Super Resolution][ONNX][${this.model.id}] ${sample.width}x${sample.height} -> ${sample.outputWidth}x${sample.outputHeight} avg pre=${(this.stats.preMs / count).toFixed(1)}ms infer=${(this.stats.inferMs / count).toFixed(1)}ms post=${(this.stats.postMs / count).toFixed(1)}ms total=${(this.stats.totalMs / count).toFixed(1)}ms runs=${count} inputPath=${sample.inputPath} outputPath=${sample.outputPath} compositePath=${sample.compositePath}`,
    );
    this.stats.lastLogAt = now;
  }
}

function configureOrtRuntime(ortNs: OrtRuntime, adapter: GPUAdapter): void {
  ortNs.env.logLevel = ECBSR_DEBUG ? "verbose" : "warning";
  ortNs.env.wasm.proxy = false;
  ortNs.env.wasm.numThreads = 1;
  const urls = getOrtAssetUrls();
  ortNs.env.wasm.wasmPaths = {
    mjs: urls.mjs,
    wasm: urls.wasm,
  };
  ortNs.env.webgpu.adapter = adapter as unknown as GPUAdapter;
  ortNs.env.webgpu.powerPreference = "low-power";
  ortNs.env.webgpu.forceFallbackAdapter = false;
  ORT_STATE.configured = true;
  ORT_STATE.adapter = adapter;
}

async function getSharedSession(
  ortNs: OrtRuntime,
  model: OnnxModelDefinition,
): Promise<SessionBundle> {
  const existing = ORT_STATE.sessionPromises.get(model.id);
  if (existing) {
    return existing;
  }

  const modelUrl = chrome.runtime.getURL(model.modelPath);
  if (ECBSR_DEBUG) {
    console.info(
      `[Video GPU Super Resolution][ONNX][${model.id}] create session model=${modelUrl} executionProvider=${model.executionProvider}`,
    );
  }

  const sessionPromise = ortNs.InferenceSession.create(modelUrl, {
    executionProviders: [{ name: model.executionProvider }],
    graphOptimizationLevel: "all",
  })
    .then((session) => {
      ORT_STATE.sessionCount += 1;
      return {
        session,
        inputName: session.inputNames[0] || "input",
        outputName: session.outputNames[0] || "output",
      };
    })
    .catch((error: Error) => {
      ORT_STATE.sessionPromises.delete(model.id);
      console.error(
        `[Video GPU Super Resolution][ONNX][${model.id}] session creation failed`,
        error,
      );
      throw error;
    });

  ORT_STATE.sessionPromises.set(model.id, sessionPromise);
  return sessionPromise;
}

function alignDimension(
  value: number,
  align: number,
  minimum: number,
): number {
  const safeAlign = Math.max(1, align);
  const safeMinimum = Math.max(1, minimum);
  return Math.max(
    safeMinimum,
    Math.ceil(Math.max(safeMinimum, value) / safeAlign) * safeAlign,
  );
}

async function getTensorDataAsync(
  tensor: OnnxRuntimeWeb.Tensor,
): Promise<SupportedTensorData> {
  const getData = (tensor as OnnxRuntimeWeb.Tensor & {
    getData?: () => Promise<unknown>;
  }).getData;

  const data = typeof getData === "function" ? await getData.call(tensor) : tensor.data;
  if (
    data instanceof Float32Array ||
    data instanceof Float64Array ||
    data instanceof Uint8Array
  ) {
    return data;
  }
  throw new Error(`Unsupported ONNX tensor data type: ${tensor.type}`);
}

function toFloat32Array(data: SupportedTensorData): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }
  return Float32Array.from(data);
}

function toByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

function getOrtRuntime(): OrtRuntime {
  const runtime = window.ort;
  if (!runtime) {
    throw new Error("onnxruntime-web runtime script is not loaded");
  }
  return runtime;
}
