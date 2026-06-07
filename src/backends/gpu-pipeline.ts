import type {
  OnnxModelDefinition,
} from "../upscaler/types";
import type {
  TileExtractParams,
  TileCopyParams,
  TileBlendParams,
  TileDescData,
} from "./gpu-pipeline-types";
import { configureWebGpuContext, type WebGpuContextState } from "../upscaler/webgpu-utilities";

import packShaderCode from "../shaders/onnx-pack.wgsl?raw";
import unpackShaderCode from "../shaders/onnx-unpack.wgsl?raw";
import compositeShaderCode from "../shaders/onnx-composite.wgsl?raw";
import tileExtractShaderCode from "../shaders/onnx-tile-extract.wgsl?raw";
import tileCopyShaderCode from "../shaders/onnx-tile-copy.wgsl?raw";
import tileBlendShaderCode from "../shaders/onnx-tile-blend.wgsl?raw";
import videoCopyShaderCode from "../shaders/onnx-video-copy.wgsl?raw";
import presentShaderCode from "../shaders/onnx-present.wgsl?raw";

const WORKGROUP_SIZE = 8;

// --- Video frame capture renderer ---

class VideoFrameTextureRenderer {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private sampler: GPUSampler | null = null;
  private frameTexture: GPUTexture | null = null;
  private frameWidth = 0;
  private frameHeight = 0;
  private cachedTextureView: GPUTextureView | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<void> {
    const module = this.device.createShaderModule({ code: videoCopyShaderCode });
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

    const externalTexture = this.device.importExternalTexture({ source: video });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.sampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.frameTexture!.createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  getTextureView(): GPUTextureView {
    if (!this.frameTexture) throw new Error("Video frame texture is unavailable");
    if (!this.cachedTextureView) {
      this.cachedTextureView = this.frameTexture.createView();
    }
    return this.cachedTextureView;
  }

  destroy(): void {
    this.cachedTextureView = null;
    if (this.frameTexture) {
      this.frameTexture.destroy();
      this.frameTexture = null;
    }
  }

  private ensureFrameTexture(width: number, height: number): void {
    if (!this.frameTexture || this.frameWidth !== width || this.frameHeight !== height) {
      this.cachedTextureView = null;
      if (this.frameTexture) this.frameTexture.destroy();
      this.frameTexture = this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.frameWidth = width;
      this.frameHeight = height;
    }
  }
}

// --- GPU Pipeline ---

export class GpuPipeline {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private model: OnnxModelDefinition;

  // Canvas context
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;
  private contextState: WebGpuContextState;

  // Video frame capture
  private frameRenderer: VideoFrameTextureRenderer;

  // Pack pipeline (video → planar buffer)
  private packPipeline: GPUComputePipeline | null = null;
  private packParamsBuffer: GPUBuffer | null = null;
  private packBindGroup: GPUBindGroup | null = null;
  private packBindGroupInputBuffer: GPUBuffer | null = null;

  // Unpack pipeline (planar buffer → texture)
  private unpackPipeline: GPURenderPipeline | null = null;
  private unpackParamsBuffer: GPUBuffer | null = null;
  private unpackOutputTexture: GPUTexture | null = null;
  private unpackOutputWidth = 0;
  private unpackOutputHeight = 0;
  private unpackBindGroup: GPUBindGroup | null = null;
  private unpackBindGroupOutputBuffer: GPUBuffer | null = null;

  // Composite pipeline (model texture + video → canvas)
  private compositePipeline: GPURenderPipeline | null = null;
  private compositeParamsBuffer: GPUBuffer | null = null;
  private compositeSampler: GPUSampler | null = null;
  private modelTexture: GPUTexture | null = null;
  private modelTextureWidth = 0;
  private modelTextureHeight = 0;
  private compositeBindGroup: GPUBindGroup | null = null;

  // Present pipeline (for "replace" mode, just show model output)
  private presentPipeline: GPURenderPipeline | null = null;
  private presentSampler: GPUSampler | null = null;
  private presentBindGroup: GPUBindGroup | null = null;

  // Luma texture (for luma_inject mode)
  private lumaTexture: GPUTexture | null = null;
  private lumaWidth = 0;
  private lumaHeight = 0;

  // Blend params buffer (reused across calls)
  private blendParamsBuffer: GPUBuffer | null = null;

  // Tile pipelines
  private tileExtractPipeline: GPUComputePipeline | null = null;
  private tileCopyPipeline: GPUComputePipeline | null = null;
  private tileBlendPipeline: GPUComputePipeline | null = null;
  private tileExtractParamsBuffer: GPUBuffer | null = null;
  private tileCopyParamsBuffer: GPUBuffer | null = null;

  private hasFrame = false;
  private ready = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.device = null as unknown as GPUDevice;
    this.model = null as unknown as OnnxModelDefinition;
    this.frameRenderer = null as unknown as VideoFrameTextureRenderer;
    this.contextState = {
      context: null as unknown as GPUCanvasContext,
      device: null as unknown as GPUDevice,
      format: null as unknown as GPUTextureFormat,
      canvas: this.canvas,
      configured: false,
      lastCanvasWidth: 0,
      lastCanvasHeight: 0,
    };
  }

  async init(device: GPUDevice, model: OnnxModelDefinition): Promise<void> {
    this.device = device;
    this.model = model;
    this.contextState.device = device;

    // Canvas context
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) throw new Error("WebGPU canvas context unavailable");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.contextState.context = this.context;
    this.contextState.format = this.format;
    this.configureContext();

    // Video frame renderer
    this.frameRenderer = new VideoFrameTextureRenderer(device);
    await this.frameRenderer.init();

    // Pack pipeline
    const packModule = device.createShaderModule({ code: packShaderCode });
    this.packPipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: packModule, entryPoint: "computeMain" },
    });
    this.packParamsBuffer = device.createBuffer({
      size: 32, // 8 floats: width, height, channels, normScale, normBias, colorWeightR, colorWeightG, colorWeightB
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Unpack pipeline
    const unpackModule = device.createShaderModule({ code: unpackShaderCode });
    this.unpackPipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: unpackModule, entryPoint: "vertexMain" },
      fragment: {
        module: unpackModule,
        entryPoint: "fragmentMain",
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.unpackParamsBuffer = device.createBuffer({
      size: 24, // width, height, channels, denormScale, denormBias + pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Composite pipeline
    const compositeModule = device.createShaderModule({ code: compositeShaderCode });
    this.compositePipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: compositeModule, entryPoint: "vertexMain" },
      fragment: {
        module: compositeModule,
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.compositeParamsBuffer = device.createBuffer({
      size: 20, // blendMode(u32) + 4 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.compositeSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Present pipeline (for replace mode — just blit model output to canvas)
    const presentModule = device.createShaderModule({ code: presentShaderCode });
    this.presentPipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "vertexMain" },
      fragment: {
        module: presentModule,
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.presentSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Tile pipelines (only if model uses tiling)
    if (model.tileSize && model.tileSize > 0) {
      const tileExtractModule = device.createShaderModule({ code: tileExtractShaderCode });
      this.tileExtractPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: tileExtractModule, entryPoint: "computeMain" },
      });
      const tileCopyModule = device.createShaderModule({ code: tileCopyShaderCode });
      this.tileCopyPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: tileCopyModule, entryPoint: "computeMain" },
      });
      const tileBlendModule = device.createShaderModule({ code: tileBlendShaderCode });
      this.tileBlendPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: tileBlendModule, entryPoint: "computeMain" },
      });
      this.tileExtractParamsBuffer = device.createBuffer({
        size: 28, // 7 u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.tileCopyParamsBuffer = device.createBuffer({
        size: 32, // 8 u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }

    this.ready = true;
  }

  // --- Input: pack video frame into NCHW planar GPU buffer ---

  packInput(
    video: HTMLVideoElement,
    inputBuffer: GPUBuffer,
    width: number,
    height: number,
  ): void {
    if (!this.packPipeline || !this.packParamsBuffer) {
      throw new Error("GpuPipeline pack is not initialized");
    }

    const { channels, colorWeights, normalization } = this.model.input;
    const weights = channels === 1
      ? (colorWeights ?? [0.299, 0.587, 0.114])
      : [1, 0, 0]; // unused for RGB but fill anyway

    const params = new ArrayBuffer(32);
    const u32 = new Uint32Array(params, 0, 3);
    const f32 = new Float32Array(params, 12, 5);
    u32[0] = width;
    u32[1] = height;
    u32[2] = channels;
    f32[0] = normalization.scale;
    f32[1] = normalization.bias;
    f32[2] = weights[0];
    f32[3] = weights[1];
    f32[4] = weights[2];
    this.device.queue.writeBuffer(this.packParamsBuffer, 0, params);

    const encoder = this.device.createCommandEncoder();
    this.frameRenderer.encodeCapturePass(encoder, video, width, height);

    if (!this.packBindGroup || this.packBindGroupInputBuffer !== inputBuffer) {
      this.packBindGroup = this.device.createBindGroup({
        layout: this.packPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.frameRenderer.getTextureView() },
          { binding: 1, resource: { buffer: inputBuffer } },
          { binding: 2, resource: { buffer: this.packParamsBuffer } },
        ],
      });
      this.packBindGroupInputBuffer = inputBuffer;
    }

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.packPipeline);
    pass.setBindGroup(0, this.packBindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / WORKGROUP_SIZE),
      Math.ceil(height / WORKGROUP_SIZE),
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // --- Output: unpack NCHW planar GPU buffer to intermediate texture ---

  unpackOutput(outputBuffer: GPUBuffer, width: number, height: number): void {
    if (!this.unpackPipeline || !this.unpackParamsBuffer) {
      throw new Error("GpuPipeline unpack is not initialized");
    }

    this.ensureModelTexture(width, height);

    const { channels } = this.model.output;
    const denorm = this.model.output.denormalization ?? { scale: 1.0, bias: 0.0 };

    const params = new ArrayBuffer(24);
    const u32 = new Uint32Array(params, 0, 3);
    const f32 = new Float32Array(params, 12, 2);
    u32[0] = width;
    u32[1] = height;
    u32[2] = channels;
    f32[0] = denorm.scale;
    f32[1] = denorm.bias;
    this.device.queue.writeBuffer(this.unpackParamsBuffer, 0, params);

    if (!this.unpackBindGroup || this.unpackBindGroupOutputBuffer !== outputBuffer) {
      this.unpackBindGroup = this.device.createBindGroup({
        layout: this.unpackPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: { buffer: this.unpackParamsBuffer } },
        ],
      });
      this.unpackBindGroupOutputBuffer = outputBuffer;
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.modelTexture!.createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.unpackPipeline);
    pass.setBindGroup(0, this.unpackBindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.hasFrame = true;
  }

  // --- Combined unpack + composite for reduced command submissions ---

  unpackAndComposite(
    outputBuffer: GPUBuffer,
    width: number,
    height: number,
    video: HTMLVideoElement,
  ): void {
    if (!this.unpackPipeline || !this.unpackParamsBuffer) {
      throw new Error("GpuPipeline unpack is not initialized");
    }
    if (!this.ready) return;

    this.ensureModelTexture(width, height);
    this.configureContext();

    const { channels } = this.model.output;
    const denorm = this.model.output.denormalization ?? { scale: 1.0, bias: 0.0 };

    const params = new ArrayBuffer(24);
    const u32 = new Uint32Array(params, 0, 3);
    const f32 = new Float32Array(params, 12, 2);
    u32[0] = width;
    u32[1] = height;
    u32[2] = channels;
    f32[0] = denorm.scale;
    f32[1] = denorm.bias;
    this.device.queue.writeBuffer(this.unpackParamsBuffer, 0, params);

    if (!this.unpackBindGroup || this.unpackBindGroupOutputBuffer !== outputBuffer) {
      this.unpackBindGroup = this.device.createBindGroup({
        layout: this.unpackPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: { buffer: this.unpackParamsBuffer } },
        ],
      });
      this.unpackBindGroupOutputBuffer = outputBuffer;
    }

    const encoder = this.device.createCommandEncoder();

    // Pass 1: unpack
    const unpackPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.modelTexture!.createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    unpackPass.setPipeline(this.unpackPipeline);
    unpackPass.setBindGroup(0, this.unpackBindGroup);
    unpackPass.draw(3);
    unpackPass.end();

    // Pass 2: present (replace mode)
    if (this.model.composite.mode === "replace") {
      if (this.presentPipeline && this.presentSampler) {
        if (!this.presentBindGroup) {
          this.presentBindGroup = this.device.createBindGroup({
            layout: this.presentPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: this.modelTexture!.createView() },
              { binding: 1, resource: this.presentSampler },
            ],
          });
        }
        const presentPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.context!.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
          }],
        });
        presentPass.setPipeline(this.presentPipeline);
        presentPass.setBindGroup(0, this.presentBindGroup);
        presentPass.draw(3);
        presentPass.end();
      }
    } else {
      // composite mode (luma_inject / overlay)
      if (this.compositePipeline && this.compositeSampler && this.compositeParamsBuffer) {
        const { params: cp } = this.model.composite;
        const blendMode = this.model.composite.mode === "luma_inject" ? 1 : 2;
        const paramsData = new ArrayBuffer(20);
        const view = new DataView(paramsData);
        view.setUint32(0, blendMode, true);
        view.setFloat32(4, cp?.lumaClampMin ?? 0.55, true);
        view.setFloat32(8, cp?.lumaClampMax ?? 1.8, true);
        view.setFloat32(12, cp?.colorDeviation ?? 0.12, true);
        view.setFloat32(16, cp?.blendStrength ?? 1.0, true);
        this.device.queue.writeBuffer(this.compositeParamsBuffer, 0, paramsData);

        const externalTexture = this.device.importExternalTexture({ source: video });
        const modelTex = this.model.composite.mode === "luma_inject" && this.lumaTexture
          ? this.lumaTexture
          : this.modelTexture!;

        this.compositeBindGroup = this.device.createBindGroup({
          layout: this.compositePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: externalTexture },
            { binding: 1, resource: this.compositeSampler },
            { binding: 2, resource: modelTex.createView() },
            { binding: 3, resource: this.compositeSampler },
            { binding: 4, resource: { buffer: this.compositeParamsBuffer } },
          ],
        });

        const compositePass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.context!.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
          }],
        });
        compositePass.setPipeline(this.compositePipeline);
        compositePass.setBindGroup(0, this.compositeBindGroup);
        compositePass.draw(3);
        compositePass.end();
      }
    }

    this.device.queue.submit([encoder.finish()]);
    this.hasFrame = true;
  }

  // --- Upload luma data from GPU buffer (for luma_inject mode, 1-channel output) ---

  uploadLumaFromBuffer(gpuBuffer: GPUBuffer, width: number, height: number): void {
    this.ensureLumaTexture(width, height);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToTexture(
      { buffer: gpuBuffer, bytesPerRow: width * 4 },
      { texture: this.lumaTexture! },
      { width, height },
    );
    this.device.queue.submit([encoder.finish()]);
    this.hasFrame = true;
  }

  uploadLumaFromData(data: Float32Array, width: number, height: number): void {
    this.ensureLumaTexture(width, height);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    this.device.queue.writeTexture(
      { texture: this.lumaTexture! },
      bytes,
      { bytesPerRow: width * 4 },
      { width, height },
    );
    this.hasFrame = true;
  }

  // --- Composite: blend upscaled result with original video and present ---

  composite(video: HTMLVideoElement): void {
    if (!this.ready || !this.hasFrame) return;
    this.configureContext();

    const { mode, params: cp } = this.model.composite;

    if (mode === "replace") {
      this.presentModelOutput();
      return;
    }

    if (!this.compositePipeline || !this.compositeParamsBuffer || !this.compositeSampler) {
      return;
    }

    const blendMode = mode === "luma_inject" ? 1 : 2;
    const lumaClampMin = cp?.lumaClampMin ?? 0.55;
    const lumaClampMax = cp?.lumaClampMax ?? 1.8;
    const colorDeviation = cp?.colorDeviation ?? 0.12;
    const blendStrength = cp?.blendStrength ?? 1.0;

    const paramsData = new ArrayBuffer(20);
    const view = new DataView(paramsData);
    view.setUint32(0, blendMode, true);
    view.setFloat32(4, lumaClampMin, true);
    view.setFloat32(8, lumaClampMax, true);
    view.setFloat32(12, colorDeviation, true);
    view.setFloat32(16, blendStrength, true);
    this.device.queue.writeBuffer(this.compositeParamsBuffer, 0, paramsData);

    const externalTexture = this.device.importExternalTexture({ source: video });
    const modelTex = mode === "luma_inject" && this.lumaTexture
      ? this.lumaTexture
      : this.modelTexture!;

    // External textures are ephemeral — bind group must be recreated each frame
    this.compositeBindGroup = this.device.createBindGroup({
      layout: this.compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.compositeSampler },
        { binding: 2, resource: modelTex.createView() },
        { binding: 3, resource: this.compositeSampler },
        { binding: 4, resource: { buffer: this.compositeParamsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context!.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, this.compositeBindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  render(): boolean {
    return this.hasFrame;
  }

  // --- Tile operations ---

  extractTile(
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    params: TileExtractParams,
  ): void {
    if (!this.tileExtractPipeline || !this.tileExtractParamsBuffer) {
      throw new Error("Tile extract pipeline not initialized");
    }
    const data = new Uint32Array([
      params.srcWidth, params.srcHeight, params.channels,
      params.tileX, params.tileY, params.tileW, params.tileH,
    ]);
    this.device.queue.writeBuffer(this.tileExtractParamsBuffer, 0, data);

    const bindGroup = this.device.createBindGroup({
      layout: this.tileExtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuffer } },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: this.tileExtractParamsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.tileExtractPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.tileW / WORKGROUP_SIZE),
      Math.ceil(params.tileH / WORKGROUP_SIZE),
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  copyTile(
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    params: TileCopyParams,
  ): void {
    if (!this.tileCopyPipeline || !this.tileCopyParamsBuffer) {
      throw new Error("Tile copy pipeline not initialized");
    }
    const data = new Uint32Array([
      params.dstWidth, params.dstHeight, params.channels,
      params.dstX, params.dstY, params.tileW, params.tileH, 0,
    ]);
    this.device.queue.writeBuffer(this.tileCopyParamsBuffer, 0, data);

    const bindGroup = this.device.createBindGroup({
      layout: this.tileCopyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuffer } },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: this.tileCopyParamsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.tileCopyPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.tileW / WORKGROUP_SIZE),
      Math.ceil(params.tileH / WORKGROUP_SIZE),
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  encodeExtractTile(
    encoder: GPUCommandEncoder,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    params: TileExtractParams,
    paramsBuffer: GPUBuffer,
  ): void {
    if (!this.tileExtractPipeline) {
      throw new Error("Tile extract pipeline not initialized");
    }
    const data = new Uint32Array([
      params.srcWidth, params.srcHeight, params.channels,
      params.tileX, params.tileY, params.tileW, params.tileH,
    ]);
    this.device.queue.writeBuffer(paramsBuffer, 0, data);

    const bindGroup = this.device.createBindGroup({
      layout: this.tileExtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuffer } },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.tileExtractPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.tileW / WORKGROUP_SIZE),
      Math.ceil(params.tileH / WORKGROUP_SIZE),
    );
    pass.end();
  }

  encodeCopyTile(
    encoder: GPUCommandEncoder,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    params: TileCopyParams,
    paramsBuffer: GPUBuffer,
  ): void {
    if (!this.tileCopyPipeline) {
      throw new Error("Tile copy pipeline not initialized");
    }
    const data = new Uint32Array([
      params.dstWidth, params.dstHeight, params.channels,
      params.dstX, params.dstY, params.tileW, params.tileH, 0,
    ]);
    this.device.queue.writeBuffer(paramsBuffer, 0, data);

    const bindGroup = this.device.createBindGroup({
      layout: this.tileCopyPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: srcBuffer } },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.tileCopyPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.tileW / WORKGROUP_SIZE),
      Math.ceil(params.tileH / WORKGROUP_SIZE),
    );
    pass.end();
  }

  hasTileBlendPipeline(): boolean {
    return this.tileBlendPipeline !== null;
  }

  blendTiles(
    dstBuffer: GPUBuffer,
    tilesBuffer: GPUBuffer,
    tileDescsBuffer: GPUBuffer,
    params: TileBlendParams,
  ): void {
    if (!this.tileBlendPipeline) {
      throw new Error("Tile blend pipeline not initialized");
    }
    // BlendParams uniform: outWidth, outHeight, channels, tileCount, overlap, pad, pad, pad
    const data = new Uint32Array([
      params.outWidth, params.outHeight, params.channels,
      params.tileCount, params.overlap, 0, 0, 0,
    ]);

    if (!this.blendParamsBuffer) {
      this.blendParamsBuffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.blendParamsBuffer, 0, data);

    const bindGroup = this.device.createBindGroup({
      layout: this.tileBlendPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dstBuffer } },
        { binding: 1, resource: { buffer: tilesBuffer } },
        { binding: 2, resource: { buffer: tileDescsBuffer } },
        { binding: 3, resource: { buffer: this.blendParamsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.tileBlendPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(params.outWidth / WORKGROUP_SIZE),
      Math.ceil(params.outHeight / WORKGROUP_SIZE),
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  clear(): void {
    if (!this.context) return;
    this.configureContext();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.hasFrame = false;
  }

  destroy(): void {
    this.frameRenderer?.destroy();
    this.packParamsBuffer?.destroy();
    this.unpackParamsBuffer?.destroy();
    this.compositeParamsBuffer?.destroy();
    this.tileExtractParamsBuffer?.destroy();
    this.tileCopyParamsBuffer?.destroy();
    this.modelTexture?.destroy();
    this.lumaTexture?.destroy();
    this.blendParamsBuffer?.destroy();
    this.packParamsBuffer = null;
    this.unpackParamsBuffer = null;
    this.compositeParamsBuffer = null;
    this.tileExtractParamsBuffer = null;
    this.tileCopyParamsBuffer = null;
    this.modelTexture = null;
    this.lumaTexture = null;
    this.blendParamsBuffer = null;
    this.packBindGroup = null;
    this.packBindGroupInputBuffer = null;
    this.unpackBindGroup = null;
    this.unpackBindGroupOutputBuffer = null;
    this.compositeBindGroup = null;
    this.presentBindGroup = null;
    this.hasFrame = false;
    this.ready = false;
    this.contextState.configured = false;
  }

  // --- Private helpers ---

  private configureContext(): void {
    if (!this.context || !this.format) return;
    configureWebGpuContext(this.contextState);
  }

  private ensureModelTexture(width: number, height: number): void {
    if (!this.modelTexture || this.modelTextureWidth !== width || this.modelTextureHeight !== height) {
      if (this.modelTexture) this.modelTexture.destroy();
      this.modelTexture = this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
      });
      this.modelTextureWidth = width;
      this.modelTextureHeight = height;
      this.compositeBindGroup = null;
      this.presentBindGroup = null;
    }
  }

  private ensureLumaTexture(width: number, height: number): void {
    if (!this.lumaTexture || this.lumaWidth !== width || this.lumaHeight !== height) {
      if (this.lumaTexture) this.lumaTexture.destroy();
      this.lumaTexture = this.device.createTexture({
        size: [width, height],
        format: "r32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.lumaWidth = width;
      this.lumaHeight = height;
    }
  }

  private presentModelOutput(): void {
    if (!this.presentPipeline || !this.presentSampler || !this.modelTexture) return;

    if (!this.presentBindGroup) {
      this.presentBindGroup = this.device.createBindGroup({
        layout: this.presentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.modelTexture.createView() },
          { binding: 1, resource: this.presentSampler },
        ],
      });
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context!.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, this.presentBindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
