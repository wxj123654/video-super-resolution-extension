import type * as OnnxRuntimeWeb from "onnxruntime-web";
import type { Settings, UpscalerImpl } from "../upscaler/types";
import { requestOnnxWebGpuAdapter } from "./onnx-webgpu-utilities";
import lumaShaderCode from "../shaders/onnx-luma.wgsl?raw";
import compositeShaderCode from "../shaders/onnx-composite.wgsl?raw";

type OrtRuntime = typeof OnnxRuntimeWeb;

const ORT_STATE = {
  configured: false,
  sessionPromise: null as Promise<OnnxRuntimeWeb.InferenceSession> | null,
  adapter: null as GPUAdapter | null,
  sessionCount: 0,
};
const ECBSR_DEBUG = false;
const ECBSR_MAX_INPUT_PIXELS = 1920 * 1080;
const ORT_JSEP_MJS_URL = chrome.runtime.getURL(
  "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs",
);
const ORT_JSEP_WASM_URL = chrome.runtime.getURL(
  "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm",
);

function extractLumaCpu(rgba: Uint8ClampedArray): Float32Array {
  const pixels = rgba.length >> 2;
  const luma = new Float32Array(pixels);
  const src = new Uint32Array(rgba.buffer, rgba.byteOffset, pixels);
  for (let i = 0; i < pixels; i++) {
    const p = src[i];
    luma[i] =
      (76 * (p & 0xff) + 150 * ((p >> 8) & 0xff) + 29 * ((p >> 16) & 0xff)) /
      25500;
  }
  return luma;
}

class WebGpuCompositor {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private lumaTexture: GPUTexture | null = null;
  private lumaWidth = 0;
  private lumaHeight = 0;
  private sampler: GPUSampler | null = null;
  ready = false;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.canvas = canvas;
    this.device = device;
  }

  async init(): Promise<void> {
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) throw new Error("WebGPU canvas context unavailable");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });

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
    if (
      !this.lumaTexture ||
      this.lumaWidth !== width ||
      this.lumaHeight !== height
    ) {
      if (this.lumaTexture) this.lumaTexture.destroy();
      this.lumaTexture = this.device.createTexture({
        size: [width, height],
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.lumaWidth = width;
      this.lumaHeight = height;
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToTexture(
      { buffer: gpuBuffer, bytesPerRow: width * 4 },
      { texture: this.lumaTexture },
      { width, height },
    );
    this.device.queue.submit([encoder.finish()]);
  }

  render(video: HTMLVideoElement): boolean {
    if (!this.ready || !this.lumaTexture) return false;

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
  }
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
}

export class EcbsrOnnxUpscaler implements UpscalerImpl {
  private canvas: HTMLCanvasElement;
  private compositor: WebGpuCompositor | null = null;

  private inputCanvas: HTMLCanvasElement;
  private inputContext: CanvasRenderingContext2D;

  private session: OnnxRuntimeWeb.InferenceSession | null = null;
  private inputName = "";
  private outputName = "";
  private pendingInference: Promise<void> | null = null;
  private lastDrawnTime = -1;
  private lastQueuedTime = -1;
  private lastRenderWidth = 0;
  private lastRenderHeight = 0;
  private outputReady = false;
  private failed = false;
  private initError: Error | null = null;

  private gpuDevice: GPUDevice | null = null;
  private lumaPipeline: GPURenderPipeline | null = null;
  private lumaSampler: GPUSampler | null = null;
  private lumaTexture: GPUTexture | null = null;
  private lumaTexWidth = 0;
  private lumaTexHeight = 0;
  private lumaGpuBuffer: GPUBuffer | null = null;
  private lumaGpuBufferSize = 0;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private gpuReady = false;

  private outputGpuBuffer: GPUBuffer | null = null;
  private outputGpuBufferSize = 0;
  private outputTensor: OnnxRuntimeWeb.Tensor | null = null;
  private outputWidth = 0;
  private outputHeight = 0;

  private stats: EcbsrStats = {
    runs: 0,
    preMs: 0,
    inferMs: 0,
    postMs: 0,
    totalMs: 0,
    lastLogAt: 0,
  };
  private runId = 0;
  private lastPendingLogAt = 0;
  private modelInputWidth = 0;
  private modelInputHeight = 0;
  private lastSourceData: Uint8ClampedArray | null = null;
  private initPromise: Promise<void>;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.inputCanvas = document.createElement("canvas");
    this.inputContext = this.inputCanvas.getContext("2d", {
      willReadFrequently: true,
    })!;
    this.initPromise = this.init().catch((error: Error) => {
      this.failed = true;
      this.initError = error;
      console.error("[Video GPU Super Resolution][ECBSR] init failed", error);
    });
  }

  private async init(): Promise<void> {
    if (ECBSR_DEBUG) {
      console.info("[Video GPU Super Resolution][ECBSR] init start");
    }

    const ortAdapter = await requestOnnxWebGpuAdapter({
      allowSoftware: true,
      preferCompatibility: false,
    });
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.InferenceSession || !ortRuntime.Tensor) {
      throw new Error("onnxruntime-web is not available");
    }
    if (typeof ortRuntime.Tensor.fromGpuBuffer !== "function") {
      throw new Error("ort.Tensor.fromGpuBuffer not available");
    }
    configureOrtRuntime(ortRuntime, ortAdapter);
    this.session = await getSharedSession(ortRuntime);
    this.inputName = this.session.inputNames[0] || "input";
    this.outputName = this.session.outputNames[0] || "output";

    const sharedDevice = ortRuntime.env.webgpu.device;
    if (!sharedDevice) {
      throw new Error("ONNX Runtime did not expose a WebGPU device");
    }
    this.gpuDevice = sharedDevice as unknown as GPUDevice;

    try {
      const module = (this.gpuDevice as GPUDevice).createShaderModule({
        code: lumaShaderCode,
      });
      this.lumaPipeline = await (
        this.gpuDevice as GPUDevice
      ).createRenderPipelineAsync({
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: {
          module,
          entryPoint: "fragmentMain",
          targets: [{ format: "r32float" }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.bindGroupLayout = this.lumaPipeline.getBindGroupLayout(0);
      this.lumaSampler = (this.gpuDevice as GPUDevice).createSampler({
        magFilter: "linear",
        minFilter: "linear",
      });
      this.gpuReady = true;
    } catch (error) {
      console.warn(
        "[Video GPU Super Resolution][ECBSR] GPU luma pipeline failed, using CPU fallback:",
        (error as Error).message,
      );
      this.gpuReady = false;
    }

    this.compositor = new WebGpuCompositor(
      this.canvas,
      this.gpuDevice as GPUDevice,
    );
    await this.compositor.init();

    if (ECBSR_DEBUG) {
      console.info(
        `[Video GPU Super Resolution][ECBSR] session ready input="${this.inputName}" output="${this.outputName}" gpuLuma=${this.gpuReady}`,
      );
    }
  }

  render(video: HTMLVideoElement, _settings: Settings): boolean {
    if (this.failed) {
      throw this.initError ?? new Error("ECBSR initialization failed");
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
            `[Video GPU Super Resolution][ECBSR] inference still pending currentTime=${video.currentTime.toFixed(3)} lastQueued=${this.lastQueuedTime.toFixed(3)}`,
          );
        }
        this.lastPendingLogAt = now;
      }
    }

    if (!this.outputReady) {
      return false;
    }

    const rendered = this.compositor!.render(video);
    if (rendered) {
      this.lastRenderWidth = this.canvas.width;
      this.lastRenderHeight = this.canvas.height;
    }
    return rendered;
  }

  private ensureWorkingSize(width: number, height: number): void {
    const targetOutputWidth = Math.max(
      1,
      Math.min(this.canvas.width || width * 2, width * 2),
    );
    const targetOutputHeight = Math.max(
      1,
      Math.min(this.canvas.height || height * 2, height * 2),
    );
    const scaleByDisplay = Math.max(
      0.25,
      Math.min(
        1,
        targetOutputWidth / (width * 2),
        targetOutputHeight / (height * 2),
      ),
    );
    const pixelScale = Math.min(
      1,
      Math.sqrt(ECBSR_MAX_INPUT_PIXELS / Math.max(1, width * height)),
    );
    const scale = Math.max(0.25, Math.min(scaleByDisplay, pixelScale));
    const rawWidth = Math.max(1, Math.round(width * scale));
    const inputHeight = Math.max(1, Math.round(height * scale));
    const inputWidth = Math.max(32, Math.ceil(rawWidth / 32) * 32);

    this.modelInputWidth = inputWidth;
    this.modelInputHeight = inputHeight;

    if (this.inputCanvas.width !== inputWidth)
      this.inputCanvas.width = inputWidth;
    if (this.inputCanvas.height !== inputHeight)
      this.inputCanvas.height = inputHeight;

    const outputWidth = inputWidth * 2;
    const outputHeight = inputHeight * 2;
    const ortRuntime = getOrtRuntime();
    if (
      ortRuntime.Tensor &&
      (this.outputWidth !== outputWidth || this.outputHeight !== outputHeight)
    ) {
      const bufferSize = outputWidth * outputHeight * 4;
      if (this.outputGpuBuffer) this.outputGpuBuffer.destroy();
      this.outputGpuBuffer = (
        this.gpuDevice as GPUDevice
      ).createBuffer({
        size: Math.ceil(bufferSize / 16) * 16,
        usage:
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.STORAGE,
      });
      this.outputTensor = ortRuntime.Tensor.fromGpuBuffer(this.outputGpuBuffer, {
        dataType: "float32",
        dims: [1, 1, outputHeight, outputWidth],
      });
      this.outputWidth = outputWidth;
      this.outputHeight = outputHeight;
    }
  }

  private async runInference(video: HTMLVideoElement): Promise<void> {
    this.runId += 1;
    const startedAt = performance.now();
    const width = this.modelInputWidth;
    const height = this.modelInputHeight;

    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor || !this.session || !this.outputTensor) {
      throw new Error("onnxruntime-web session is unavailable");
    }

    const preStartedAt = performance.now();

    let inputTensor: OnnxRuntimeWeb.Tensor | null = null;
    let useGpuBuffer = false;

    if (this.gpuReady) {
      const gpuBuffer = await this.extractLumaToGpuBuffer(video, width, height);
      if (gpuBuffer) {
        try {
          inputTensor = ortRuntime.Tensor.fromGpuBuffer(gpuBuffer, {
            dims: [1, 1, height, width],
            dataType: "float32",
          });
          useGpuBuffer = true;
        } catch (e) {
          if (ECBSR_DEBUG) {
            console.warn(
              "[Video GPU Super Resolution][ECBSR] fromGpuBuffer failed:",
              (e as Error).message,
            );
          }
        }
      }

      if (!useGpuBuffer && gpuBuffer) {
        const yData = await this.readGpuBufferToCpu(gpuBuffer, width, height);
        inputTensor = new ortRuntime.Tensor("float32", yData, [
          1,
          1,
          height,
          width,
        ]);
      } else if (!useGpuBuffer) {
        const yData = this.extractLumaCpuPath(video, width, height);
        inputTensor = new ortRuntime.Tensor("float32", yData, [
          1,
          1,
          height,
          width,
        ]);
      }
    } else {
      const yData = this.extractLumaCpuPath(video, width, height);
      inputTensor = new ortRuntime.Tensor("float32", yData, [
        1,
        1,
        height,
        width,
      ]);
    }

    const preEndedAt = performance.now();

    if (!inputTensor) {
      throw new Error("ECBSR input tensor preparation failed");
    }

    const inferStartedAt = performance.now();
    await this.session.run(
      { [this.inputName]: inputTensor },
      { [this.outputName]: this.outputTensor },
    );
    const inferEndedAt = performance.now();

    const postStartedAt = performance.now();
    this.compositor!.uploadLumaFromBuffer(
      this.outputGpuBuffer!,
      this.outputWidth,
      this.outputHeight,
    );

    const finishedAt = performance.now();
    this.outputReady = true;
    this.lastDrawnTime = video.currentTime;

    this.recordStats({
      preMs: preEndedAt - preStartedAt,
      inferMs: inferEndedAt - inferStartedAt,
      postMs: finishedAt - postStartedAt,
      totalMs: finishedAt - startedAt,
      width,
      height,
    });
  }

  private extractLumaCpuPath(
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Float32Array {
    this.inputContext.drawImage(video, 0, 0, width, height);
    const sourceImage = this.inputContext.getImageData(0, 0, width, height);
    this.lastSourceData = sourceImage.data;
    return extractLumaCpu(sourceImage.data);
  }

  private async extractLumaToGpuBuffer(
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Promise<GPUBuffer | null> {
    const device = this.gpuDevice as GPUDevice;

    if (
      !this.lumaTexture ||
      this.lumaTexWidth !== width ||
      this.lumaTexHeight !== height
    ) {
      if (this.lumaTexture) this.lumaTexture.destroy();
      this.lumaTexture = device.createTexture({
        size: [width, height],
        format: "r32float",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.lumaTexWidth = width;
      this.lumaTexHeight = height;
    }

    const bufferSize = width * height * 4;
    if (!this.lumaGpuBuffer || this.lumaGpuBufferSize < bufferSize) {
      if (this.lumaGpuBuffer) this.lumaGpuBuffer.destroy();
      this.lumaGpuBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      });
      this.lumaGpuBufferSize = bufferSize;
    }

    const externalTexture = device.importExternalTexture({ source: video });
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.lumaSampler! },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.lumaTexture.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.lumaPipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();

    encoder.copyTextureToBuffer(
      { texture: this.lumaTexture },
      { buffer: this.lumaGpuBuffer, bytesPerRow: width * 4 },
      { width, height },
    );

    device.queue.submit([encoder.finish()]);
    return this.lumaGpuBuffer;
  }

  private async readGpuBufferToCpu(
    gpuBuffer: GPUBuffer,
    width: number,
    height: number,
  ): Promise<Float32Array> {
    const device = this.gpuDevice as GPUDevice;
    const size = width * height * 4;
    const readBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(gpuBuffer, 0, readBuffer, 0, size);
    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();
    readBuffer.destroy();
    return data;
  }

  destroy(): void {
    this.pendingInference = null;
    this.session = null;
    this.outputReady = false;
    if (this.compositor) {
      this.compositor.clear();
      this.compositor.destroy();
    }
    if (this.lumaTexture) {
      this.lumaTexture.destroy();
      this.lumaTexture = null;
    }
    if (this.lumaGpuBuffer) {
      this.lumaGpuBuffer.destroy();
      this.lumaGpuBuffer = null;
    }
    if (this.outputGpuBuffer) {
      this.outputGpuBuffer.destroy();
      this.outputGpuBuffer = null;
    }
    this.gpuDevice = null;
    this.gpuReady = false;
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
      `[Video GPU Super Resolution][ECBSR] ${sample.width}x${sample.height} avg pre=${(this.stats.preMs / count).toFixed(1)}ms infer=${(this.stats.inferMs / count).toFixed(1)}ms post=${(this.stats.postMs / count).toFixed(1)}ms total=${(this.stats.totalMs / count).toFixed(1)}ms runs=${count}`,
    );
    this.stats.lastLogAt = now;
  }
}

function configureOrtRuntime(
  ortNs: OrtRuntime,
  adapter: GPUAdapter,
): void {
  ortNs.env.logLevel = ECBSR_DEBUG ? "verbose" : "warning";
  ortNs.env.wasm.proxy = false;
  ortNs.env.wasm.numThreads = 1;
  ortNs.env.wasm.wasmPaths = {
    mjs: ORT_JSEP_MJS_URL,
    wasm: ORT_JSEP_WASM_URL,
  };
  ortNs.env.webgpu.adapter = adapter as unknown as GPUAdapter;
  ortNs.env.webgpu.powerPreference = "low-power";
  ortNs.env.webgpu.forceFallbackAdapter = false;
  ORT_STATE.configured = true;
  ORT_STATE.adapter = adapter;
}

async function getSharedSession(
  ortNs: OrtRuntime,
): Promise<OnnxRuntimeWeb.InferenceSession> {
  if (ORT_STATE.sessionPromise) {
    return ORT_STATE.sessionPromise;
  }

  const modelUrl = chrome.runtime.getURL("models/ecbsr_x2_m4c8_y.onnx");
  if (ECBSR_DEBUG) {
    console.info(
      `[Video GPU Super Resolution][ECBSR] create session model=${modelUrl} executionProvider=webgpu`,
    );
  }
  ORT_STATE.sessionPromise = ortNs.InferenceSession.create(modelUrl, {
    executionProviders: [{ name: "webgpu" }],
    graphOptimizationLevel: "all",
  })
    .then((session) => {
      ORT_STATE.sessionCount += 1;
      if (ECBSR_DEBUG) {
        console.info(
          `[Video GPU Super Resolution][ECBSR] session created inputs=${session.inputNames.join(",")} outputs=${session.outputNames.join(",")}`,
        );
      }
      return session;
    })
    .catch((error: Error) => {
      ORT_STATE.sessionPromise = null;
      console.error(
        "[Video GPU Super Resolution][ECBSR] session creation failed",
        error,
      );
      throw error;
    });

  return ORT_STATE.sessionPromise;
}

function getOrtRuntime(): OrtRuntime {
  const runtime = window.ort;
  if (!runtime) {
    throw new Error("onnxruntime-web runtime script is not loaded");
  }
  return runtime;
}
