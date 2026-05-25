(function attachVideoGpuSuperResolutionOnnx() {
  const ns = window.VideoGpuSuperResolutionInternal;
  const { requestWebGpuAdapter, getAdapterInfo, isSoftwareAdapter } = ns;
  const ORT_STATE = {
    configured: false,
    sessionPromise: null,
    adapter: null,
    sessionCount: 0
  };
  const ECBSR_DEBUG = false;
  const ECBSR_MAX_INPUT_PIXELS = 1920 * 1080;

  const LUMA_SHADER = `
	struct VertexOutput {
	  @builtin(position) position: vec4f,
	  @location(0) uv: vec2f,
	};

	@vertex
	fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
	  var positions = array<vec2f, 3>(
	    vec2f(-1.0, -1.0),
	    vec2f(3.0, -1.0),
	    vec2f(-1.0, 3.0)
	  );
	  var uvs = array<vec2f, 3>(
	    vec2f(0.0, 1.0),
	    vec2f(2.0, 1.0),
	    vec2f(0.0, -1.0)
	  );
	  var output: VertexOutput;
	  output.position = vec4f(positions[index], 0.0, 1.0);
	  output.uv = uvs[index];
	  return output;
	}

	@group(0) @binding(0) var videoFrame: texture_external;
	@group(0) @binding(1) var videoSampler: sampler;

	@fragment
	fn fragmentMain(input: VertexOutput) -> @location(0) f32 {
	  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
	  let color = textureSampleBaseClampToEdge(videoFrame, videoSampler, uv);
	  return dot(color.rgb, vec3f(0.299, 0.587, 0.114));
	}`;

  const COMPOSITE_SHADER = `
	struct VertexOutput {
	  @builtin(position) position: vec4f,
	  @location(0) uv: vec2f,
	};

	@vertex
	fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
	  var positions = array<vec2f, 3>(
	    vec2f(-1.0, -1.0),
	    vec2f(3.0, -1.0),
	    vec2f(-1.0, 3.0)
	  );
	  var uvs = array<vec2f, 3>(
	    vec2f(0.0, 1.0),
	    vec2f(2.0, 1.0),
	    vec2f(0.0, -1.0)
	  );
	  var output: VertexOutput;
	  output.position = vec4f(positions[index], 0.0, 1.0);
	  output.uv = uvs[index];
	  return output;
	}

	@group(0) @binding(0) var videoTex: texture_external;
	@group(0) @binding(1) var videoSampler: sampler;
	@group(0) @binding(2) var lumaTex: texture_2d<f32>;

	fn sampleLumaBilinear(uv: vec2f) -> f32 {
	  let dims = textureDimensions(lumaTex);
	  let pixCoord = uv * vec2f(dims) - 0.5;
	  let tl = vec2i(floor(pixCoord));
	  let f = fract(pixCoord);
	  let d1 = vec2i(dims) - 1;
	  let c00 = textureLoad(lumaTex, clamp(tl, vec2i(0), d1), 0).r;
	  let c10 = textureLoad(lumaTex, clamp(tl + vec2i(1, 0), vec2i(0), d1), 0).r;
	  let c01 = textureLoad(lumaTex, clamp(tl + vec2i(0, 1), vec2i(0), d1), 0).r;
	  let c11 = textureLoad(lumaTex, clamp(tl + vec2i(1, 1), vec2i(0), d1), 0).r;
	  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
	}


	fn lumaFromRgb(color: vec3f) -> f32 {
	  return dot(color, vec3f(0.299, 0.587, 0.114));
	}

	@fragment
	fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
	  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
	  let base = textureSampleBaseClampToEdge(videoTex, videoSampler, uv).rgb;
	  let baseY = lumaFromRgb(base);
	  let modelY = clamp(sampleLumaBilinear(uv), 0.0, 1.0);
	  let y = mix(baseY, modelY, 1.0);
	  let gain = y / max(baseY, 0.0001);
	  var color = base * clamp(gain, 0.55, 1.8);
	  let minColor = max(base - vec3f(0.12), vec3f(0.0));
	  let maxColor = min(base + vec3f(0.12), vec3f(1.0));
	  return vec4f(clamp(color, minColor, maxColor), 1.0);
	}`;

  function extractLumaCpu(rgba) {
    const pixels = rgba.length >> 2;
    const luma = new Float32Array(pixels);
    const src = new Uint32Array(rgba.buffer, rgba.byteOffset, pixels);
    for (let i = 0; i < pixels; i++) {
      const p = src[i];
      luma[i] = (76 * (p & 0xFF) + 150 * ((p >> 8) & 0xFF) + 29 * ((p >> 16) & 0xFF)) / 25500;
    }
    return luma;
  }

  class WebGpuCompositor {
    constructor(canvas, device) {
      this.canvas = canvas;
      this.device = device;
      this.context = null;
      this.format = null;
      this.pipeline = null;
      this.lumaTexture = null;
      this.lumaWidth = 0;
      this.lumaHeight = 0;
      this.sampler = null;
      this.ready = false;
    }

    async init() {
      this.context = this.canvas.getContext("webgpu");
      if (!this.context) throw new Error("WebGPU canvas context unavailable");
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "opaque"
      });

      const module = this.device.createShaderModule({ code: COMPOSITE_SHADER });
      this.pipeline = await this.device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module, entryPoint: "vertexMain" },
        fragment: {
          module,
          entryPoint: "fragmentMain",
          targets: [{ format: this.format }]
        },
        primitive: { topology: "triangle-list" }
      });

      this.sampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });
      this.ready = true;
    }

    uploadLumaFromBuffer(gpuBuffer, width, height) {
      if (!this.lumaTexture || this.lumaWidth !== width || this.lumaHeight !== height) {
        if (this.lumaTexture) this.lumaTexture.destroy();
        this.lumaTexture = this.device.createTexture({
          size: [width, height],
          format: "r32float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        this.lumaWidth = width;
        this.lumaHeight = height;
      }
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToTexture(
        { buffer: gpuBuffer, bytesPerRow: width * 4 },
        { texture: this.lumaTexture },
        { width, height }
      );
      this.device.queue.submit([encoder.finish()]);
    }

    render(video) {
      if (!this.ready || !this.lumaTexture) return false;

      const externalTexture = this.device.importExternalTexture({ source: video });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: externalTexture },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.lumaTexture.createView() }
        ]
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return true;
    }

    clear() {
      if (!this.context) return;
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
      if (this.lumaTexture) { this.lumaTexture.destroy(); this.lumaTexture = null; }
      this.ready = false;
    }
  }

  class EcbsrOnnxUpscaler {
    constructor(canvas) {
      this.canvas = canvas;
      this.compositor = null;

      this.inputCanvas = document.createElement("canvas");
      this.inputContext = this.inputCanvas.getContext("2d", { willReadFrequently: true });

      this.session = null;
      this.inputName = "";
      this.outputName = "";
      this.pendingInference = null;
      this.lastDrawnTime = -1;
      this.lastQueuedTime = -1;
      this.lastRenderWidth = 0;
      this.lastRenderHeight = 0;
      this.outputReady = false;
      this.failed = false;
      this.initError = null;

      this.gpuDevice = null;
      this.lumaPipeline = null;
      this.lumaSampler = null;
      this.lumaTexture = null;
      this.lumaTexWidth = 0;
      this.lumaTexHeight = 0;
      this.lumaGpuBuffer = null;
      this.lumaGpuBufferSize = 0;
      this.bindGroupLayout = null;
      this.gpuReady = false;

      this.outputGpuBuffer = null;
      this.outputGpuBufferSize = 0;
      this.outputTensor = null;
      this.outputWidth = 0;
      this.outputHeight = 0;

      this.stats = {
        runs: 0,
        preMs: 0,
        inferMs: 0,
        postMs: 0,
        totalMs: 0,
        lastLogAt: 0
      };
      this.runId = 0;
      this.lastPendingLogAt = 0;
      this.modelInputWidth = 0;
      this.modelInputHeight = 0;
      this.initPromise = this.init().catch((error) => {
        this.failed = true;
        this.initError = error;
        console.error("[Video GPU Super Resolution][ECBSR] init failed", error);
      });
    }

    async init() {
      if (ECBSR_DEBUG) {
        console.info("[Video GPU Super Resolution][ECBSR] init start");
      }

      const ortAdapter = await requestWebGpuAdapter({ allowSoftware: true, preferCompatibility: false });
      const ort = window.ort;
      if (!ort?.InferenceSession || !ort?.Tensor) {
        throw new Error("onnxruntime-web is not available");
      }
      if (typeof ort.Tensor.fromGpuBuffer !== "function") {
        throw new Error("ort.Tensor.fromGpuBuffer not available");
      }
      configureOrtRuntime(ort, ortAdapter);
      this.session = await getSharedSession(ort);
      this.inputName = this.session.inputNames[0] || "input";
      this.outputName = this.session.outputNames[0] || "output";

      const sharedDevice = ort.env.webgpu.device;
      if (!sharedDevice) {
        throw new Error("ONNX Runtime did not expose a WebGPU device");
      }
      this.gpuDevice = sharedDevice;

      try {
        const module = sharedDevice.createShaderModule({ code: LUMA_SHADER });
        this.lumaPipeline = await sharedDevice.createRenderPipelineAsync({
          layout: "auto",
          vertex: { module, entryPoint: "vertexMain" },
          fragment: {
            module,
            entryPoint: "fragmentMain",
            targets: [{ format: "r32float" }]
          },
          primitive: { topology: "triangle-list" }
        });
        this.bindGroupLayout = this.lumaPipeline.getBindGroupLayout(0);
        this.lumaSampler = sharedDevice.createSampler({
          magFilter: "linear",
          minFilter: "linear"
        });
        this.gpuReady = true;
      } catch (error) {
        console.warn("[Video GPU Super Resolution][ECBSR] GPU luma pipeline failed, using CPU fallback:", error.message);
        this.gpuReady = false;
      }

      this.compositor = new WebGpuCompositor(this.canvas, sharedDevice);
      await this.compositor.init();

      if (ECBSR_DEBUG) {
        console.info(
          `[Video GPU Super Resolution][ECBSR] session ready input="${this.inputName}" output="${this.outputName}" gpuLuma=${this.gpuReady}`
        );
      }
    }

    render(video) {
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
              `[Video GPU Super Resolution][ECBSR] inference still pending currentTime=${video.currentTime.toFixed(3)} lastQueued=${this.lastQueuedTime.toFixed(3)}`
            );
          }
          this.lastPendingLogAt = now;
        }
      }

      if (!this.outputReady) {
        return false;
      }

      const rendered = this.compositor.render(video);
      if (rendered) {
        this.lastRenderWidth = this.canvas.width;
        this.lastRenderHeight = this.canvas.height;
      }
      return rendered;
    }

    ensureWorkingSize(width, height) {
      const targetOutputWidth = Math.max(1, Math.min(this.canvas.width || width * 2, width * 2));
      const targetOutputHeight = Math.max(1, Math.min(this.canvas.height || height * 2, height * 2));
      const scaleByDisplay = Math.max(0.25, Math.min(1, targetOutputWidth / (width * 2), targetOutputHeight / (height * 2)));
      const pixelScale = Math.min(1, Math.sqrt(ECBSR_MAX_INPUT_PIXELS / Math.max(1, width * height)));
      const scale = Math.max(0.25, Math.min(scaleByDisplay, pixelScale));
      const rawWidth = Math.max(1, Math.round(width * scale));
      const inputHeight = Math.max(1, Math.round(height * scale));
      const inputWidth = Math.max(32, Math.ceil(rawWidth / 32) * 32);

      this.modelInputWidth = inputWidth;
      this.modelInputHeight = inputHeight;

      if (this.inputCanvas.width !== inputWidth) this.inputCanvas.width = inputWidth;
      if (this.inputCanvas.height !== inputHeight) this.inputCanvas.height = inputHeight;

      const outputWidth = inputWidth * 2;
      const outputHeight = inputHeight * 2;
      const ort = window.ort;
      if (ort?.Tensor && (this.outputWidth !== outputWidth || this.outputHeight !== outputHeight)) {
        const bufferSize = outputWidth * outputHeight * 4;
        if (this.outputGpuBuffer) this.outputGpuBuffer.destroy();
        this.outputGpuBuffer = this.gpuDevice.createBuffer({
          size: Math.ceil(bufferSize / 16) * 16,
          usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
        });
        this.outputTensor = ort.Tensor.fromGpuBuffer(this.outputGpuBuffer, {
          dataType: "float32",
          dims: [1, 1, outputHeight, outputWidth]
        });
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
      }
    }

    async runInference(video) {
      this.runId += 1;
      const runId = this.runId;
      const startedAt = performance.now();
      const width = this.modelInputWidth;
      const height = this.modelInputHeight;
      const ort = window.ort;
      if (!ort?.Tensor || !this.session || !this.outputTensor) {
        throw new Error("onnxruntime-web session is unavailable");
      }

      if (ECBSR_DEBUG) {
        console.info(
          `[Video GPU Super Resolution][ECBSR] run#${runId} start frameTime=${video.currentTime.toFixed(3)} input=${width}x${height} source=${video.videoWidth || 0}x${video.videoHeight || 0}`
        );
      }

      const preStartedAt = performance.now();

      let inputTensor;
      let useGpuBuffer = false;

      if (this.gpuReady) {
        const gpuBuffer = await this.extractLumaToGpuBuffer(video, width, height);
        if (gpuBuffer) {
          try {
            inputTensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
              dims: [1, 1, height, width],
              dataType: "float32"
            });
            useGpuBuffer = true;
          } catch (e) {
            if (ECBSR_DEBUG) {
              console.warn("[Video GPU Super Resolution][ECBSR] fromGpuBuffer failed:", e.message);
            }
          }
        }

        if (!useGpuBuffer) {
          const yData = await this.readGpuBufferToCpu(gpuBuffer, width, height);
          inputTensor = new ort.Tensor("float32", yData, [1, 1, height, width]);
        }
      } else {
        const yData = this.extractLumaCpu(video, width, height);
        inputTensor = new ort.Tensor("float32", yData, [1, 1, height, width]);
      }

      const preEndedAt = performance.now();

      const inferStartedAt = performance.now();
      await this.session.run(
        { [this.inputName]: inputTensor },
        { [this.outputName]: this.outputTensor }
      );
      const inferEndedAt = performance.now();

      const postStartedAt = performance.now();
      this.compositor.uploadLumaFromBuffer(this.outputGpuBuffer, this.outputWidth, this.outputHeight);

      const finishedAt = performance.now();
      this.outputReady = true;
      this.lastDrawnTime = video.currentTime;
      if (ECBSR_DEBUG) {
        console.info(
          `[Video GPU Super Resolution][ECBSR] run#${runId} done gpuBuffer=${useGpuBuffer} pre=${(preEndedAt - preStartedAt).toFixed(1)}ms infer=${(inferEndedAt - inferStartedAt).toFixed(1)}ms post=${(finishedAt - postStartedAt).toFixed(1)}ms total=${(finishedAt - startedAt).toFixed(1)}ms`
        );
      }
      this.recordStats({
        preMs: preEndedAt - preStartedAt,
        inferMs: inferEndedAt - inferStartedAt,
        postMs: finishedAt - postStartedAt,
        totalMs: finishedAt - startedAt,
        width,
        height
      });
    }

    extractLumaCpu(video, width, height) {
      this.inputContext.drawImage(video, 0, 0, width, height);
      const sourceImage = this.inputContext.getImageData(0, 0, width, height);
      this.lastSourceData = sourceImage.data;
      return extractLumaCpu(sourceImage.data);
    }

    async extractLumaToGpuBuffer(video, width, height) {
      const device = this.gpuDevice;

      if (!this.lumaTexture || this.lumaTexWidth !== width || this.lumaTexHeight !== height) {
        if (this.lumaTexture) this.lumaTexture.destroy();
        this.lumaTexture = device.createTexture({
          size: [width, height],
          format: "r32float",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        });
        this.lumaTexWidth = width;
        this.lumaTexHeight = height;
      }

      const bufferSize = width * height * 4;
      if (!this.lumaGpuBuffer || this.lumaGpuBufferSize < bufferSize) {
        if (this.lumaGpuBuffer) this.lumaGpuBuffer.destroy();
        this.lumaGpuBuffer = device.createBuffer({
          size: bufferSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
        });
        this.lumaGpuBufferSize = bufferSize;
      }

      const externalTexture = device.importExternalTexture({ source: video });
      const bindGroup = device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: externalTexture },
          { binding: 1, resource: this.lumaSampler }
        ]
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.lumaTexture.createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });
      pass.setPipeline(this.lumaPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0);
      pass.end();

      encoder.copyTextureToBuffer(
        { texture: this.lumaTexture },
        { buffer: this.lumaGpuBuffer, bytesPerRow: width * 4 },
        { width, height }
      );

      device.queue.submit([encoder.finish()]);
      return this.lumaGpuBuffer;
    }

    async readGpuBufferToCpu(gpuBuffer, width, height) {
      const device = this.gpuDevice;
      const size = width * height * 4;
      const readBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
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

    destroy() {
      this.pendingInference = null;
      this.session = null;
      this.outputReady = false;
      if (this.compositor) { this.compositor.clear(); this.compositor.destroy(); }
      if (this.lumaTexture) { this.lumaTexture.destroy(); this.lumaTexture = null; }
      if (this.lumaGpuBuffer) { this.lumaGpuBuffer.destroy(); this.lumaGpuBuffer = null; }
      if (this.outputGpuBuffer) { this.outputGpuBuffer.destroy(); this.outputGpuBuffer = null; }
      this.gpuDevice = null;
      this.gpuReady = false;
    }

    recordStats(sample) {
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
      const avgPre = this.stats.preMs / count;
      const avgInfer = this.stats.inferMs / count;
      const avgPost = this.stats.postMs / count;
      const avgTotal = this.stats.totalMs / count;
      console.info(
        `[Video GPU Super Resolution][ECBSR] ${sample.width}x${sample.height} avg pre=${avgPre.toFixed(1)}ms infer=${avgInfer.toFixed(1)}ms post=${avgPost.toFixed(1)}ms total=${avgTotal.toFixed(1)}ms runs=${count}`
      );
      this.stats.lastLogAt = now;
    }
  }

  function configureOrtRuntime(ort, adapter) {
    const mjsPath = chrome.runtime.getURL("src/vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs");
    const wasmPath = chrome.runtime.getURL("src/vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm");
    ort.env.logLevel = ECBSR_DEBUG ? "verbose" : "warning";
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      mjs: mjsPath,
      wasm: wasmPath
    };
    ort.env.webgpu.adapter = adapter;
    ort.env.webgpu.powerPreference = "low-power";
    ort.env.webgpu.forceFallbackAdapter = false;
    ORT_STATE.configured = true;
    ORT_STATE.adapter = adapter;
  }

  async function getSharedSession(ort) {
    if (ORT_STATE.sessionPromise) {
      return ORT_STATE.sessionPromise;
    }

    const modelUrl = chrome.runtime.getURL("src/models/ecbsr_x2_m4c8_y.onnx");
    if (ECBSR_DEBUG) {
      console.info(
        `[Video GPU Super Resolution][ECBSR] create session model=${modelUrl} executionProvider=webgpu`
      );
    }
    ORT_STATE.sessionPromise = ort.InferenceSession.create(
      modelUrl,
      {
        executionProviders: [{ name: "webgpu" }],
        graphOptimizationLevel: "all"
      }
    )
      .then((session) => {
        ORT_STATE.sessionCount += 1;
        if (ECBSR_DEBUG) {
          console.info(
            `[Video GPU Super Resolution][ECBSR] session created inputs=${session.inputNames.join(",")} outputs=${session.outputNames.join(",")}`
          );
        }
        return session;
      })
      .catch((error) => {
        ORT_STATE.sessionPromise = null;
        console.error("[Video GPU Super Resolution][ECBSR] session creation failed", error);
        throw error;
      });

    return ORT_STATE.sessionPromise;
  }

  ns.EcbsrOnnxUpscaler = EcbsrOnnxUpscaler;
})();
