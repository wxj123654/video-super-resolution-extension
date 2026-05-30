import type * as OnnxRuntimeWeb from "onnxruntime-web";
import type {
  Settings,
  UpscalerImpl,
  OnnxModelDefinition,
} from "../upscaler/types";
import {
  requestWebGpuAdapter,
} from "../upscaler/webgpu-utilities";
import { getOnnxModelDefinition, resolvePrecisionModel } from "./onnx-models";
import { resolveModelBuffer, isModelCached } from "./model-cache";
import { supportsFp16 } from "./onnx-webgpu-utilities";
import { GpuPipeline } from "./gpu-pipeline";

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

let _ortAssetUrls: { mjs: string; wasm: string } | null = null;

function getOrtAssetUrls() {
  if (!_ortAssetUrls) {
    _ortAssetUrls = {
      mjs: chrome.runtime.getURL("vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs"),
      wasm: chrome.runtime.getURL("vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm"),
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

function extractRgbCpu(rgba: Uint8ClampedArray, normScale: number, normBias: number): Float32Array {
  const pixels = rgba.length >> 2;
  const rgb = new Float32Array(pixels * 3);
  let rOffset = 0;
  let gOffset = pixels;
  let bOffset = pixels * 2;

  for (let i = 0; i < rgba.length; i += 4) {
    rgb[rOffset++] = (rgba[i] / 255) * normScale + normBias;
    rgb[gOffset++] = (rgba[i + 1] / 255) * normScale + normBias;
    rgb[bOffset++] = (rgba[i + 2] / 255) * normScale + normBias;
  }

  return rgb;
}

export class EcbsrOnnxUpscaler implements UpscalerImpl {
  private canvas: HTMLCanvasElement;
  private model: OnnxModelDefinition;
  private inputCanvas: HTMLCanvasElement;
  private inputContext: CanvasRenderingContext2D;

  private session: OnnxRuntimeWeb.InferenceSession | null = null;
  private inputName = "input";
  private outputName = "output";

  private pipeline: GpuPipeline | null = null;

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

  private outputReady = false;
  private failed = false;
  private initComplete = false;
  private aborted = false;
  private initError: Error | null = null;

  private pendingInference: Promise<void> | null = null;
  private lastQueuedTime = -1;
  private lastPendingLogAt = 0;

  private gpuInputEnabled = false;
  private gpuOutputEnabled = false;
  private gpuTiledFailed = false;
  private activeInputPath: OnnxPathMode = "cpu";
  private activeOutputPath: OnnxPathMode = "cpu";
  private activeCompositePath: OnnxCompositePath = "2d";

  private stats: EcbsrStats = {
    runs: 0,
    preMs: 0,
    inferMs: 0,
    postMs: 0,
    totalMs: 0,
    lastLogAt: 0,
  };
  private modelInputWidth = 0;
  private modelInputHeight = 0;
  private outputWidth = 0;
  private outputHeight = 0;

  // GPU tiled inference resources
  private tileInputGpuBuffer: GPUBuffer | null = null;
  private tileOutputGpuBuffer: GPUBuffer | null = null;
  private tiledFullOutputGpuBuffer: GPUBuffer | null = null;
  private tiledBufferTileSize = 0;
  private tiledBufferChannels = 0;
  private tiledFullOutputW = 0;
  private tiledFullOutputH = 0;

  // Batched tiled inference resources
  private batchedTileInputBuffers: GPUBuffer[] = [];
  private batchedTileOutputBuffers: GPUBuffer[] = [];
  private batchedExtractParamBuffers: GPUBuffer[] = [];
  private batchedCopyParamBuffers: GPUBuffer[] = [];
  private batchedPoolSize = 0;
  private batchedBufferTileSize = 0;
  private batchedBufferChannels = 0;
  private batchedBufferScale = 0;

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

    if (this.model.precisionAlternatives) {
      this.model = resolvePrecisionModel(this.model.id, supportsFp16(ortAdapter));
    }

    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.InferenceSession || !ortRuntime.Tensor) {
      throw new Error("onnxruntime-web is not available");
    }
    if (!ORT_STATE.configured) {
      configureOrtRuntime(ortRuntime, ortAdapter);
    }

    const sessionBundle = await getSharedSession(ortRuntime, this.model);
    this.session = sessionBundle.session;
    this.inputName = sessionBundle.inputName;
    this.outputName = sessionBundle.outputName;

    const sharedDevice = ortRuntime.env.webgpu?.device;
    if (sharedDevice) {
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
    } else {
      console.warn(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] No WebGPU device from ORT, using CPU-only path`,
      );
    }

    if (this.gpuDevice) {
      try {
        const pipeline = new GpuPipeline(this.canvas);
        await pipeline.init(this.gpuDevice, this.model);
        this.pipeline = pipeline;
        this.gpuTiledFailed = false;
        this.activeCompositePath = "webgpu";
      } catch (error) {
        this.pipeline = null;
        this.activeCompositePath = "2d";
        console.warn(
          `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU pipeline unavailable, using CPU fallback:`,
          (error as Error).message,
        );
      }
    }

    if (ECBSR_DEBUG) {
      console.info(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] session ready input="${this.inputName}" output="${this.outputName}"`,
      );
    }

    this.initComplete = true;
  }

  render(video: HTMLVideoElement, _settings: Settings): boolean {
    if (this.failed) {
      throw this.initError ?? new Error("ONNX model initialization failed");
    }
    if (!this.session || !this.initComplete) {
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

    if (this.pipeline) {
      this.pipeline.composite(video);
      return this.pipeline.render();
    }
    return false;
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

    const inputChannels = this.model.input.channels;
    const outputChannels = this.model.output.channels;

    this.gpuInputEnabled = this.tryEnsureTensor(
      "input", inputWidth, inputHeight, inputChannels,
    );
    this.gpuOutputEnabled = this.tryEnsureTensor(
      "output", this.outputWidth, this.outputHeight, outputChannels,
    );
  }

  private tryEnsureTensor(
    kind: "input" | "output",
    width: number,
    height: number,
    channels: number,
  ): boolean {
    if (!this.gpuDevice) return false;
    if (kind === "input" && !this.pipeline) return false;
    try {
      this.ensureGpuTensorResource(kind, width, height, channels);
      return true;
    } catch (error) {
      console.warn(
        `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU ${kind} tensor unavailable:`,
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

    const currentBuffer = kind === "input" ? this.inputGpuBuffer : this.outputGpuBuffer;
    const currentTensor = kind === "input" ? this.inputTensor : this.outputTensor;
    const currentWidth = kind === "input" ? this.inputTensorWidth : this.outputTensorWidth;
    const currentHeight = kind === "input" ? this.inputTensorHeight : this.outputTensorHeight;
    const currentChannels = kind === "input" ? this.inputTensorChannels : this.outputTensorChannels;
    const currentSize = kind === "input" ? this.inputBufferSize : this.outputBufferSize;

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

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    if (kind === "input") {
      this.inputGpuBuffer?.destroy();
      this.inputGpuBuffer = this.gpuDevice.createBuffer({ size: alignedSize, usage });
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
    this.outputGpuBuffer = this.gpuDevice.createBuffer({ size: alignedSize, usage });
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
    if (this.model.tileSize && this.model.tileSize > 0) {
      return this.runTiledInference(video);
    }
    return this.runFullInference(video);
  }

  private async runFullInference(video: HTMLVideoElement): Promise<void> {
    const startedAt = performance.now();
    const width = this.modelInputWidth;
    const height = this.modelInputHeight;
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor || !this.session) {
      throw new Error("onnxruntime-web session is unavailable");
    }

    const preStartedAt = performance.now();
    const inputTensor = await this.createInputTensor(ortRuntime, video, width, height);
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
    await this.handleOutput(results, useGpuOutput);
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

  private async runTiledInference(video: HTMLVideoElement): Promise<void> {
    const width = this.modelInputWidth;
    const height = this.modelInputHeight;
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor || !this.session) {
      throw new Error("onnxruntime-web session is unavailable");
    }

    const tileSize = this.model.tileSize!;
    const scale = this.model.scale;
    const channels = this.model.input.channels;
    const overlap = Math.max(8, Math.floor(tileSize * 0.1));

    const tiles: Array<{ x: number; y: number; w: number; h: number }> = [];
    const step = tileSize - overlap;
    const seen = new Set<string>();
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const tx = x + tileSize > width ? Math.max(0, width - tileSize) : x;
        const ty = y + tileSize > height ? Math.max(0, height - tileSize) : y;
        const key = `${tx},${ty}`;
        if (!seen.has(key)) {
          seen.add(key);
          tiles.push({ x: tx, y: ty, w: tileSize, h: tileSize });
        }
      }
    }

    const outW = width * scale;
    const outH = height * scale;

    if (!this.gpuTiledFailed) {
      const canGpu = this.ensureTiledGpuResources(outW, outH, channels);
      if (canGpu) {
        try {
          return await this.runTiledInferenceGpu(video, width, height, tiles, channels, scale, outW, outH);
        } catch (error) {
          this.gpuTiledFailed = true;
          console.warn(
            `[Video GPU Super Resolution][ONNX][${this.model.id}] GPU tiled inference failed, will use CPU path on next frame:`,
            (error as Error).message,
          );
          return;
        }
      }
    }
    return this.runTiledInferenceCpu(video, width, height, tiles, channels, scale, outW, outH);
  }

  private getTilePoolSize(): number {
    return this.model.tilePoolSize ?? 64;
  }

  private ensureBatchedTiledResources(
    tileCount: number,
    tileSize: number,
    channels: number,
    scale: number,
  ): boolean {
    if (!this.gpuDevice || !this.pipeline) return false;
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor) return false;

    const poolSize = Math.min(tileCount, this.getTilePoolSize());

    if (
      this.batchedPoolSize >= poolSize &&
      this.batchedBufferTileSize >= tileSize &&
      this.batchedBufferChannels >= channels &&
      this.batchedBufferScale >= scale
    ) {
      return true;
    }

    try {
      this.destroyBatchedTileResources();

      const bufferUsage: GPUBufferUsageFlags =
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const paramUsage: GPUBufferUsageFlags =
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

      const inputSize = Math.ceil(tileSize * tileSize * channels * 4 / 16) * 16;
      const outputSize = Math.ceil(tileSize * scale * tileSize * scale * channels * 4 / 16) * 16;

      for (let i = 0; i < poolSize; i++) {
        this.batchedTileInputBuffers.push(
          this.gpuDevice.createBuffer({ size: inputSize, usage: bufferUsage }),
        );
        this.batchedTileOutputBuffers.push(
          this.gpuDevice.createBuffer({ size: outputSize, usage: bufferUsage }),
        );
        this.batchedExtractParamBuffers.push(
          this.gpuDevice.createBuffer({ size: 28, usage: paramUsage }),
        );
        this.batchedCopyParamBuffers.push(
          this.gpuDevice.createBuffer({ size: 32, usage: paramUsage }),
        );
      }
      this.batchedPoolSize = poolSize;
      this.batchedBufferTileSize = tileSize;
      this.batchedBufferChannels = channels;
      this.batchedBufferScale = scale;
      return true;
    } catch {
      this.destroyBatchedTileResources();
      return false;
    }
  }

  private destroyBatchedTileResources(): void {
    for (const buf of this.batchedTileInputBuffers) buf.destroy();
    for (const buf of this.batchedTileOutputBuffers) buf.destroy();
    for (const buf of this.batchedExtractParamBuffers) buf.destroy();
    for (const buf of this.batchedCopyParamBuffers) buf.destroy();
    this.batchedTileInputBuffers = [];
    this.batchedTileOutputBuffers = [];
    this.batchedExtractParamBuffers = [];
    this.batchedCopyParamBuffers = [];
    this.batchedPoolSize = 0;
    this.batchedBufferTileSize = 0;
    this.batchedBufferChannels = 0;
    this.batchedBufferScale = 0;
  }

  private ensureTiledGpuResources(
    outW: number,
    outH: number,
    channels: number,
  ): boolean {
    if (!this.gpuDevice || !this.pipeline) return false;
    const ortRuntime = getOrtRuntime();
    if (!ortRuntime.Tensor) return false;

    const tileSize = this.model.tileSize!;
    const scale = this.model.scale;

    if (
      !this.tileInputGpuBuffer ||
      this.tiledBufferTileSize < tileSize ||
      this.tiledBufferChannels < channels
    ) {
      this.tileInputGpuBuffer?.destroy();
      const inputSize = Math.ceil(tileSize * tileSize * channels * 4 / 16) * 16;
      this.tileInputGpuBuffer = this.gpuDevice.createBuffer({
        size: inputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      this.tileOutputGpuBuffer?.destroy();
      const outputSize = Math.ceil(tileSize * scale * tileSize * scale * channels * 4 / 16) * 16;
      this.tileOutputGpuBuffer = this.gpuDevice.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

      this.tiledBufferTileSize = tileSize;
      this.tiledBufferChannels = channels;
    }

    if (this.tiledFullOutputW !== outW || this.tiledFullOutputH !== outH) {
      this.tiledFullOutputGpuBuffer?.destroy();
      const fullSize = Math.ceil(outW * outH * channels * 4 / 16) * 16;
      this.tiledFullOutputGpuBuffer = this.gpuDevice.createBuffer({
        size: fullSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.tiledFullOutputW = outW;
      this.tiledFullOutputH = outH;
    }

    return true;
  }

  private async runTiledInferenceGpu(
    video: HTMLVideoElement,
    width: number,
    height: number,
    tiles: Array<{ x: number; y: number; w: number; h: number }>,
    channels: number,
    scale: number,
    outW: number,
    outH: number,
  ): Promise<void> {
    const canBatch = this.ensureBatchedTiledResources(
      tiles.length, tiles[0]?.w ?? this.model.tileSize ?? 256, channels, scale,
    );
    if (!canBatch) {
      return this.runTiledInferenceGpuSequential(video, width, height, tiles, channels, scale, outW, outH);
    }

    const startedAt = performance.now();
    const poolSize = this.batchedPoolSize;
    const ortRuntime = getOrtRuntime();

    console.info(
      `[VSR][tiled-gpu] start tiles=${tiles.length} poolSize=${poolSize} input=${width}x${height} output=${outW}x${outH} tileSize=${tiles[0]?.w} channels=${channels}`,
    );

    const preStartedAt = performance.now();
    this.pipeline!.packInput(video, this.inputGpuBuffer!, width, height);
    this.activeInputPath = "gpu";

    const clearEncoder = this.gpuDevice!.createCommandEncoder();
    clearEncoder.clearBuffer(this.tiledFullOutputGpuBuffer!, 0);
    this.gpuDevice!.queue.submit([clearEncoder.finish()]);
    const preEndedAt = performance.now();

    const inferStartedAt = performance.now();

    for (let chunkStart = 0; chunkStart < tiles.length; chunkStart += poolSize) {
      if (this.aborted) return;
      const chunkEnd = Math.min(chunkStart + poolSize, tiles.length);

      // Phase 1: Batch extract all tiles in this chunk
      const extractEncoder = this.gpuDevice!.createCommandEncoder();
      for (let t = chunkStart; t < chunkEnd; t++) {
        const tile = tiles[t];
        const poolIdx = t - chunkStart;
        this.pipeline!.encodeExtractTile(
          extractEncoder,
          this.inputGpuBuffer!,
          this.batchedTileInputBuffers[poolIdx],
          {
            srcWidth: width, srcHeight: height, channels,
            tileX: tile.x, tileY: tile.y, tileW: tile.w, tileH: tile.h,
          },
          this.batchedExtractParamBuffers[poolIdx],
        );
      }
      this.gpuDevice!.queue.submit([extractEncoder.finish()]);

      // Phase 2: Run inferences sequentially
      for (let t = chunkStart; t < chunkEnd; t++) {
        if (this.aborted) return;
        const tile = tiles[t];
        const poolIdx = t - chunkStart;
        const tileOutW = tile.w * scale;
        const tileOutH = tile.h * scale;

        const tileInputTensor = ortRuntime.Tensor.fromGpuBuffer(
          this.batchedTileInputBuffers[poolIdx],
          { dataType: "float32", dims: [1, channels, tile.h, tile.w] },
        );
        const tileOutputTensor = ortRuntime.Tensor.fromGpuBuffer(
          this.batchedTileOutputBuffers[poolIdx],
          { dataType: "float32", dims: [1, channels, tileOutH, tileOutW] },
        );

        const runT0 = performance.now();
        await this.session!.run(
          { [this.inputName]: tileInputTensor },
          { [this.outputName]: tileOutputTensor },
        );
        const runMs = performance.now() - runT0;

        if (t < 5 || t % 50 === 0) {
          console.info(
            `[VSR][tiled-gpu] tile ${t}/${tiles.length} run=${runMs.toFixed(1)}ms pos=(${tile.x},${tile.y})`,
          );
        }
      }

      // Phase 3: Batch copy all tiles in this chunk
      const copyEncoder = this.gpuDevice!.createCommandEncoder();
      for (let t = chunkStart; t < chunkEnd; t++) {
        const tile = tiles[t];
        const poolIdx = t - chunkStart;
        this.pipeline!.encodeCopyTile(
          copyEncoder,
          this.batchedTileOutputBuffers[poolIdx],
          this.tiledFullOutputGpuBuffer!,
          {
            dstWidth: outW, dstHeight: outH, channels,
            dstX: tile.x * scale, dstY: tile.y * scale,
            tileW: tile.w * scale, tileH: tile.h * scale,
          },
          this.batchedCopyParamBuffers[poolIdx],
        );
      }
      this.gpuDevice!.queue.submit([copyEncoder.finish()]);
    }

    await this.gpuDevice!.queue.onSubmittedWorkDone();

    const inferEndedAt = performance.now();
    console.info(
      `[VSR][tiled-gpu] all tiles done in ${(inferEndedAt - inferStartedAt).toFixed(1)}ms`,
    );

    const postStartedAt = performance.now();
    this.uploadToPipeline(this.tiledFullOutputGpuBuffer!, outW, outH, true);
    const finishedAt = performance.now();
    this.outputReady = true;

    this.recordStats({
      preMs: preEndedAt - preStartedAt,
      inferMs: inferEndedAt - inferStartedAt,
      postMs: finishedAt - postStartedAt,
      totalMs: finishedAt - startedAt,
      width,
      height,
      outputWidth: outW,
      outputHeight: outH,
      inputPath: "gpu",
      outputPath: "gpu",
      compositePath: this.activeCompositePath,
    });
  }

  private async runTiledInferenceGpuSequential(
    video: HTMLVideoElement,
    width: number,
    height: number,
    tiles: Array<{ x: number; y: number; w: number; h: number }>,
    channels: number,
    scale: number,
    outW: number,
    outH: number,
  ): Promise<void> {
    const startedAt = performance.now();
    console.info(
      `[VSR][tiled-gpu] start tiles=${tiles.length} input=${width}x${height} output=${outW}x${outH} tileSize=${tiles[0]?.w} channels=${channels}`,
    );

    const preStartedAt = performance.now();
    this.pipeline!.packInput(video, this.inputGpuBuffer!, width, height);
    this.activeInputPath = "gpu";
    const preEndedAt = performance.now();
    console.info(`[VSR][tiled-gpu] pack done in ${(preEndedAt - preStartedAt).toFixed(1)}ms`);

    const clearEncoder = this.gpuDevice!.createCommandEncoder();
    clearEncoder.clearBuffer(this.tiledFullOutputGpuBuffer!, 0);
    this.gpuDevice!.queue.submit([clearEncoder.finish()]);

    const inferStartedAt = performance.now();
    const ortRuntime = getOrtRuntime();

    for (let t = 0; t < tiles.length; t++) {
      if (this.aborted) return;
      const tile = tiles[t];
      const tileT0 = performance.now();

      this.pipeline!.extractTile(
        this.inputGpuBuffer!,
        this.tileInputGpuBuffer!,
        {
          srcWidth: width, srcHeight: height, channels,
          tileX: tile.x, tileY: tile.y, tileW: tile.w, tileH: tile.h,
        },
      );

      const tileOutW = tile.w * scale;
      const tileOutH = tile.h * scale;
      const dstX = tile.x * scale;
      const dstY = tile.y * scale;

      const tileInputTensor = ortRuntime.Tensor.fromGpuBuffer(
        this.tileInputGpuBuffer!,
        { dataType: "float32", dims: [1, channels, tile.h, tile.w] },
      );

      const tileOutputTensor = ortRuntime.Tensor.fromGpuBuffer(
        this.tileOutputGpuBuffer!,
        { dataType: "float32", dims: [1, channels, tileOutH, tileOutW] },
      );

      const runT0 = performance.now();
      await this.session!.run(
        { [this.inputName]: tileInputTensor },
        { [this.outputName]: tileOutputTensor },
      );
      const runMs = performance.now() - runT0;

      this.pipeline!.copyTile(
        this.tileOutputGpuBuffer!,
        this.tiledFullOutputGpuBuffer!,
        {
          dstWidth: outW, dstHeight: outH, channels,
          dstX, dstY, tileW: tileOutW, tileH: tileOutH,
        },
      );

      const totalMs = performance.now() - tileT0;
      if (t < 5 || t % 50 === 0) {
        console.info(
          `[VSR][tiled-gpu] tile ${t}/${tiles.length} run=${runMs.toFixed(1)}ms total=${totalMs.toFixed(1)}ms pos=(${tile.x},${tile.y})`,
        );
      }
    }

    await this.gpuDevice!.queue.onSubmittedWorkDone();

    const inferEndedAt = performance.now();
    console.info(
      `[VSR][tiled-gpu] all tiles done in ${(inferEndedAt - inferStartedAt).toFixed(1)}ms`,
    );

    const postStartedAt = performance.now();
    this.uploadToPipeline(this.tiledFullOutputGpuBuffer!, outW, outH, true);
    const finishedAt = performance.now();
    this.outputReady = true;

    this.recordStats({
      preMs: preEndedAt - preStartedAt,
      inferMs: inferEndedAt - inferStartedAt,
      postMs: finishedAt - postStartedAt,
      totalMs: finishedAt - startedAt,
      width,
      height,
      outputWidth: outW,
      outputHeight: outH,
      inputPath: "gpu",
      outputPath: "gpu",
      compositePath: this.activeCompositePath,
    });
  }

  private async runTiledInferenceCpu(
    video: HTMLVideoElement,
    width: number,
    height: number,
    tiles: Array<{ x: number; y: number; w: number; h: number }>,
    channels: number,
    scale: number,
    outW: number,
    outH: number,
  ): Promise<void> {
    const startedAt = performance.now();
    const ortRuntime = getOrtRuntime();

    const preStartedAt = performance.now();
    const { normalization } = this.model.input;
    const inputData = this.extractCpuInput(video, width, height, channels, normalization);
    const preEndedAt = performance.now();

    const outputPlaneSize = outW * outH;
    const outputData = new Float32Array(outputPlaneSize * channels);

    const inferStartedAt = performance.now();

    for (let t = 0; t < tiles.length; t++) {
      if (this.aborted) return;
      const tile = tiles[t];
      const tilePlaneSize = tile.w * tile.h;
      const tileInput = new Float32Array(tilePlaneSize * channels);

      for (let c = 0; c < channels; c++) {
        for (let ty = 0; ty < tile.h; ty++) {
          const srcOffset = c * width * height + (tile.y + ty) * width + tile.x;
          const dstOffset = c * tilePlaneSize + ty * tile.w;
          tileInput.set(
            inputData.subarray(srcOffset, srcOffset + tile.w),
            dstOffset,
          );
        }
      }

      const tileTensor = new ortRuntime.Tensor(
        "float32",
        tileInput,
        [1, channels, tile.h, tile.w],
      );

      const tileOutW = tile.w * scale;
      const tileOutH = tile.h * scale;

      const tileOutData = new Float32Array(tileOutW * tileOutH * channels);
      const tileOutTensor = new ortRuntime.Tensor(
        "float32",
        tileOutData,
        [1, channels, tileOutH, tileOutW],
      );

      await this.session!.run(
        { [this.inputName]: tileTensor },
        { [this.outputName]: tileOutTensor },
      );

      const tileOutPlaneSize = tileOutW * tileOutH;
      const dstX = tile.x * scale;
      const dstY = tile.y * scale;

      for (let c = 0; c < channels; c++) {
        for (let oy = 0; oy < tileOutH; oy++) {
          const srcOff = c * tileOutPlaneSize + oy * tileOutW;
          const dstOff = c * outputPlaneSize + (dstY + oy) * outW + dstX;
          outputData.set(
            tileOutData.subarray(srcOff, srcOff + tileOutW),
            dstOff,
          );
        }
      }
    }

    const inferEndedAt = performance.now();

    const postStartedAt = performance.now();
    if (this.gpuDevice && this.pipeline) {
      const alignedSize = Math.ceil(outputData.byteLength / 16) * 16;
      const tempBuffer = this.gpuDevice.createBuffer({
        size: alignedSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.gpuDevice.queue.writeBuffer(tempBuffer, 0, outputData);
      this.uploadToPipeline(tempBuffer, outW, outH, true);
      tempBuffer.destroy();
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
      outputWidth: outW,
      outputHeight: outH,
      inputPath: this.activeInputPath,
      outputPath: "cpu",
      compositePath: this.activeCompositePath,
    });
  }

  private async createInputTensor(
    ortRuntime: OrtRuntime,
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): Promise<OnnxRuntimeWeb.Tensor> {
    const channels = this.model.input.channels;

    if (
      this.gpuInputEnabled &&
      this.pipeline &&
      this.inputGpuBuffer &&
      this.inputTensor
    ) {
      try {
        this.pipeline.packInput(video, this.inputGpuBuffer, width, height);
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
    const { normalization } = this.model.input;
    const data = this.extractCpuInput(video, width, height, channels, normalization);
    return new ortRuntime.Tensor("float32", data, [1, channels, height, width]);
  }

  private async handleOutput(
    results: Record<string, OnnxRuntimeWeb.Tensor>,
    useGpuOutput: boolean,
  ): Promise<void> {
    if (!this.pipeline) {
      throw new Error("Pipeline is unavailable for output handling");
    }

    const mode = this.model.composite.mode;

    if (useGpuOutput && this.outputGpuBuffer) {
      if (mode === "luma_inject") {
        this.pipeline.uploadLumaFromBuffer(
          this.outputGpuBuffer,
          this.outputWidth,
          this.outputHeight,
        );
      } else {
        this.pipeline.unpackOutput(
          this.outputGpuBuffer,
          this.outputWidth,
          this.outputHeight,
        );
      }
      this.activeCompositePath = "webgpu";
      return;
    }

    const outputTensor = await this.resolveOutputTensor(results);
    const data = toFloat32Array(await getTensorDataAsync(outputTensor));

    if (mode === "luma_inject") {
      this.pipeline.uploadLumaFromData(data, this.outputWidth, this.outputHeight);
    } else {
      // For CPU output, write to a temp GPU buffer and unpack
      if (this.gpuDevice) {
        const alignedSize = Math.ceil(data.byteLength / 16) * 16;
        const tempBuffer = this.gpuDevice.createBuffer({
          size: alignedSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.gpuDevice.queue.writeBuffer(tempBuffer, 0, data);
        this.pipeline.unpackOutput(tempBuffer, this.outputWidth, this.outputHeight);
        tempBuffer.destroy();
      }
    }
    this.activeCompositePath = "webgpu";
  }

  private uploadToPipeline(
    buffer: GPUBuffer,
    width: number,
    height: number,
    _isGpuBuffer: boolean,
  ): void {
    if (!this.pipeline) return;

    const mode = this.model.composite.mode;
    if (mode === "luma_inject") {
      this.pipeline.uploadLumaFromBuffer(buffer, width, height);
    } else {
      this.pipeline.unpackOutput(buffer, width, height);
    }
  }

  private extractCpuInput(
    video: HTMLVideoElement,
    width: number,
    height: number,
    channels: number,
    normalization: { scale: number; bias: number },
  ): Float32Array {
    this.inputContext.drawImage(video, 0, 0, width, height);
    const sourceImage = this.inputContext.getImageData(0, 0, width, height);
    if (channels === 1) {
      return extractLumaCpu(sourceImage.data);
    }
    return extractRgbCpu(sourceImage.data, normalization.scale, normalization.bias);
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

  destroy(): void {
    this.aborted = true;
    this.pendingInference = null;
    this.session = null;
    this.outputReady = false;
    this.pipeline?.clear();
    this.pipeline?.destroy();
    this.pipeline = null;
    this.inputGpuBuffer?.destroy();
    this.outputGpuBuffer?.destroy();
    this.inputGpuBuffer = null;
    this.outputGpuBuffer = null;
    this.inputTensor = null;
    this.outputTensor = null;
    this.tileInputGpuBuffer?.destroy();
    this.tileOutputGpuBuffer?.destroy();
    this.tiledFullOutputGpuBuffer?.destroy();
    this.tileInputGpuBuffer = null;
    this.tileOutputGpuBuffer = null;
    this.tiledFullOutputGpuBuffer = null;
    this.destroyBatchedTileResources();
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

async function fetchModelBuffer(model: OnnxModelDefinition): Promise<ArrayBuffer> {
  const url = chrome.runtime.getURL(model.modelPath);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load bundled model ${model.id}: ${response.status}`);
  }
  return response.arrayBuffer();
}

async function downloadModelBuffer(model: OnnxModelDefinition): Promise<ArrayBuffer> {
  const onProgress = (loaded: number, total: number): void => {
    try {
      chrome.runtime.sendMessage({
        type: "VSR_MODEL_STATUS",
        modelStatus: { modelId: model.id, state: "downloading" as const, progress: loaded, total },
      });
    } catch { /* popup may be closed */ }
  };

  const cached = await isModelCached(model.id);
  if (!cached) {
    try {
      chrome.runtime.sendMessage({
        type: "VSR_MODEL_STATUS",
        modelStatus: { modelId: model.id, state: "downloading" as const, progress: 0, total: model.source?.fileSize ?? 0 },
      });
    } catch { /* ignore */ }
  }

  try {
    const buffer = await resolveModelBuffer(model, onProgress);
    try {
      chrome.runtime.sendMessage({
        type: "VSR_MODEL_STATUS",
        modelStatus: { modelId: model.id, state: "ready" as const },
      });
    } catch { /* ignore */ }
    return buffer;
  } catch (downloadError) {
    try {
      chrome.runtime.sendMessage({
        type: "VSR_MODEL_STATUS",
        modelStatus: {
          modelId: model.id,
          state: "error" as const,
          error: downloadError instanceof Error ? downloadError.message : String(downloadError),
        },
      });
    } catch { /* ignore */ }
    throw downloadError;
  }
}

async function createSessionWithFallback(
  ortNs: OrtRuntime,
  buffer: ArrayBuffer,
  preferredEp: string,
): Promise<OnnxRuntimeWeb.InferenceSession> {
  const providers = buildEpFallbackChain(preferredEp);
  const errors: string[] = [];

  for (const ep of providers) {
    try {
      return await ortNs.InferenceSession.create(buffer, {
        executionProviders: [{ name: ep }],
        graphOptimizationLevel: "all" as const,
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      errors.push(`${ep}: ${msg}`);
      console.warn(
        `[Video GPU Super Resolution][ONNX] EP "${ep}" failed, trying next fallback`,
        msg,
      );
    }
  }

  throw new Error(
    `All execution providers failed for model.\n${errors.map((e, i) => `  ${i + 1}. ${e}`).join("\n")}`,
  );
}

function buildEpFallbackChain(preferred: string): string[] {
  const chain = [preferred];
  if (preferred === "webgpu") {
    chain.push("wasm");
  }
  if (!chain.includes("wasm")) {
    chain.push("wasm");
  }
  return chain;
}

async function getSharedSession(
  ortNs: OrtRuntime,
  model: OnnxModelDefinition,
): Promise<SessionBundle> {
  const existing = ORT_STATE.sessionPromises.get(model.id);
  if (existing) {
    return existing;
  }

  if (ECBSR_DEBUG) {
    console.info(
      `[Video GPU Super Resolution][ONNX][${model.id}] create session source=${model.source?.type ?? "bundled"} executionProvider=${model.executionProvider}`,
    );
  }

  const sessionPromise = (async () => {
    const modelBuffer = !model.source || model.source.type === "bundled"
      ? await fetchModelBuffer(model)
      : await downloadModelBuffer(model);

    const session = await createSessionWithFallback(ortNs, modelBuffer, model.executionProvider);

    ORT_STATE.sessionCount += 1;
    return {
      session,
      inputName: session.inputNames[0] || "input",
      outputName: session.outputNames[0] || "output",
    };
  })()
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
  const directData = tensor.data;
  if (
    directData instanceof Float32Array ||
    directData instanceof Float64Array ||
    directData instanceof Uint8Array
  ) {
    return directData;
  }

  const getData = (tensor as OnnxRuntimeWeb.Tensor & {
    getData?: () => Promise<unknown>;
  }).getData;

  if (typeof getData === "function") {
    const data = await getData.call(tensor);
    if (
      data instanceof Float32Array ||
      data instanceof Float64Array ||
      data instanceof Uint8Array
    ) {
      return data;
    }
  }

  if (ArrayBuffer.isView(directData) || directData instanceof ArrayBuffer) {
    return new Float32Array(
      directData instanceof ArrayBuffer ? directData : directData.buffer,
      directData instanceof ArrayBuffer ? 0 : (directData as ArrayBufferView).byteOffset,
      directData instanceof ArrayBuffer ? directData.byteLength / 4 : (directData as ArrayBufferView).byteLength / 4,
    );
  }

  throw new Error(
    `Unsupported ONNX tensor data type: ${tensor.type} (data constructor: ${directData?.constructor?.name ?? "null"})`,
  );
}

function toFloat32Array(data: SupportedTensorData): Float32Array {
  if (data instanceof Float32Array) {
    return data;
  }
  return Float32Array.from(data);
}

function getOrtRuntime(): OrtRuntime {
  const runtime = window.ort;
  if (!runtime) {
    throw new Error("onnxruntime-web runtime script is not loaded");
  }
  return runtime;
}
