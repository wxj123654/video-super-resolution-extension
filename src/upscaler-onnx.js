(function attachVideoGpuSuperResolutionOnnx() {
  const ns = window.VideoGpuSuperResolutionInternal;
  const { requestWebGpuAdapter, getAdapterInfo, isSoftwareAdapter } = ns;
  const ORT_STATE = {
    configured: false,
    sessionPromise: null,
    adapter: null,
    sessionCount: 0
  };

  class EcbsrOnnxUpscaler {
    constructor(canvas) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d", { alpha: false, desynchronized: true });
      if (!this.context) {
        throw new Error("Unable to create 2D canvas context for ECBSR");
      }

      this.inputCanvas = document.createElement("canvas");
      this.inputContext = this.inputCanvas.getContext("2d", { willReadFrequently: true });
      this.baseCanvas = document.createElement("canvas");
      this.baseContext = this.baseCanvas.getContext("2d", { willReadFrequently: true });
      this.outputCanvas = document.createElement("canvas");
      this.outputContext = this.outputCanvas.getContext("2d", { willReadFrequently: true });
      if (!this.inputContext || !this.baseContext || !this.outputContext) {
        throw new Error("Unable to create ECBSR working canvases");
      }

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
      this.initPromise = this.init().catch((error) => {
        this.failed = true;
        this.initError = error;
        console.error("[Video GPU Super Resolution][ECBSR] init failed", error);
      });
    }

    async init() {
      console.info("[Video GPU Super Resolution][ECBSR] init start");
      const adapter = await requestWebGpuAdapter({ allowSoftware: true, preferCompatibility: true });
      const ort = window.ort;
      if (!ort?.InferenceSession || !ort?.Tensor) {
        throw new Error("onnxruntime-web is not available");
      }

      const adapterInfo = getAdapterInfo(adapter);
      console.info(
        `[Video GPU Super Resolution][ECBSR] adapter vendor="${adapterInfo.vendor}" architecture="${adapterInfo.architecture}" device="${adapterInfo.device}" description="${adapterInfo.description}" fallback=${Boolean(adapter.isFallbackAdapter)} software=${isSoftwareAdapter(adapter, adapterInfo)}`
      );
      configureOrtRuntime(ort, adapter);
      this.session = await getSharedSession(ort);

      this.inputName = this.session.inputNames[0] || "input";
      this.outputName = this.session.outputNames[0] || "output";
      console.info(
        `[Video GPU Super Resolution][ECBSR] session ready input="${this.inputName}" output="${this.outputName}" sessionCount=${ORT_STATE.sessionCount} webgpuDevice=${Boolean(ort.env.webgpu.device)}`
      );
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
          console.info(
            `[Video GPU Super Resolution][ECBSR] inference still pending currentTime=${video.currentTime.toFixed(3)} lastQueued=${this.lastQueuedTime.toFixed(3)}`
          );
          this.lastPendingLogAt = now;
        }
      }

      if (!this.outputReady) {
        return false;
      }

      const outputWidth = this.outputCanvas.width;
      const outputHeight = this.outputCanvas.height;
      if (outputWidth !== this.lastRenderWidth || outputHeight !== this.lastRenderHeight) {
        this.context.imageSmoothingEnabled = true;
        this.context.imageSmoothingQuality = "high";
        this.lastRenderWidth = outputWidth;
        this.lastRenderHeight = outputHeight;
      }

      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.context.drawImage(this.outputCanvas, 0, 0, this.canvas.width, this.canvas.height);
      return true;
    }

    ensureWorkingSize(width, height) {
      if (this.inputCanvas.width !== width) this.inputCanvas.width = width;
      if (this.inputCanvas.height !== height) this.inputCanvas.height = height;

      const outputWidth = width * 2;
      const outputHeight = height * 2;
      if (this.baseCanvas.width !== outputWidth) this.baseCanvas.width = outputWidth;
      if (this.baseCanvas.height !== outputHeight) this.baseCanvas.height = outputHeight;
      if (this.outputCanvas.width !== outputWidth) this.outputCanvas.width = outputWidth;
      if (this.outputCanvas.height !== outputHeight) this.outputCanvas.height = outputHeight;
    }

    async runInference(video) {
      this.runId += 1;
      const runId = this.runId;
      const startedAt = performance.now();
      const width = this.inputCanvas.width;
      const height = this.inputCanvas.height;
      const ort = window.ort;
      if (!ort?.Tensor || !this.session) {
        throw new Error("onnxruntime-web session is unavailable");
      }

      console.info(
        `[Video GPU Super Resolution][ECBSR] run#${runId} start frameTime=${video.currentTime.toFixed(3)} input=${width}x${height}`
      );

      const preStartedAt = performance.now();
      this.inputContext.drawImage(video, 0, 0, width, height);
      const sourceImage = this.inputContext.getImageData(0, 0, width, height);
      const yData = extractLuma(sourceImage.data);
      const preEndedAt = performance.now();

      const tensor = new ort.Tensor("float32", yData, [1, 1, height, width]);
      const inferStartedAt = performance.now();
      const result = await this.session.run({ [this.inputName]: tensor });
      const inferEndedAt = performance.now();
      const outputTensor = result[this.outputName];
      if (!outputTensor?.data) {
        throw new Error("ECBSR inference produced no output");
      }

      const postStartedAt = performance.now();
      this.baseContext.clearRect(0, 0, this.baseCanvas.width, this.baseCanvas.height);
      this.baseContext.imageSmoothingEnabled = true;
      this.baseContext.imageSmoothingQuality = "high";
      this.baseContext.drawImage(video, 0, 0, this.baseCanvas.width, this.baseCanvas.height);
      const baseImage = this.baseContext.getImageData(0, 0, this.baseCanvas.width, this.baseCanvas.height);
      const combined = mergeLuma(baseImage.data, outputTensor.data);
      this.outputContext.putImageData(new ImageData(combined, this.outputCanvas.width, this.outputCanvas.height), 0, 0);
      const finishedAt = performance.now();
      this.outputReady = true;
      this.lastDrawnTime = video.currentTime;
      console.info(
        `[Video GPU Super Resolution][ECBSR] run#${runId} done outputLocation=${outputTensor.location || "unknown"} outputShape=${Array.isArray(outputTensor.dims) ? outputTensor.dims.join("x") : "unknown"} pre=${(preEndedAt - preStartedAt).toFixed(1)}ms infer=${(inferEndedAt - inferStartedAt).toFixed(1)}ms post=${(finishedAt - postStartedAt).toFixed(1)}ms total=${(finishedAt - startedAt).toFixed(1)}ms`
      );
      this.recordStats({
        preMs: preEndedAt - preStartedAt,
        inferMs: inferEndedAt - inferStartedAt,
        postMs: finishedAt - postStartedAt,
        totalMs: finishedAt - startedAt,
        width,
        height
      });
    }

    destroy() {
      this.pendingInference = null;
      this.session = null;
      this.outputReady = false;
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
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
    ort.env.logLevel = "verbose";
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      mjs: mjsPath,
      wasm: wasmPath
    };
    ort.env.webgpu.adapter = adapter;
    ort.env.webgpu.powerPreference = "low-power";
    ort.env.webgpu.forceFallbackAdapter = true;
    ORT_STATE.configured = true;
    ORT_STATE.adapter = adapter;
    console.info(
      `[Video GPU Super Resolution][ECBSR] ort configured logLevel=${ort.env.logLevel} wasmProxy=${ort.env.wasm.proxy} wasmThreads=${ort.env.wasm.numThreads} wasmMjs=${mjsPath} wasmWasm=${wasmPath}`
    );
  }

  async function getSharedSession(ort) {
    if (ORT_STATE.sessionPromise) {
      console.info("[Video GPU Super Resolution][ECBSR] reuse existing session promise");
      return ORT_STATE.sessionPromise;
    }

    const modelUrl = chrome.runtime.getURL("src/models/ecbsr_x2_m4c8_y.onnx");
    console.info(
      `[Video GPU Super Resolution][ECBSR] create session model=${modelUrl} executionProvider=webgpu`
    );
    ORT_STATE.sessionPromise = ort.InferenceSession.create(
      modelUrl,
      {
        executionProviders: [{ name: "webgpu" }],
        graphOptimizationLevel: "all"
      }
    )
      .then((session) => {
        ORT_STATE.sessionCount += 1;
        console.info(
          `[Video GPU Super Resolution][ECBSR] session created inputs=${session.inputNames.join(",")} outputs=${session.outputNames.join(",")}`
        );
        return session;
      })
      .catch((error) => {
        ORT_STATE.sessionPromise = null;
        console.error("[Video GPU Super Resolution][ECBSR] session creation failed", error);
        throw error;
      });

    return ORT_STATE.sessionPromise;
  }

  function extractLuma(rgba) {
    const pixels = rgba.length / 4;
    const luma = new Float32Array(pixels);
    for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
      const r = rgba[i] / 255;
      const g = rgba[i + 1] / 255;
      const b = rgba[i + 2] / 255;
      luma[p] = clamp01(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return luma;
  }

  function mergeLuma(baseRgba, modelLuma) {
    const output = new Uint8ClampedArray(baseRgba.length);
    for (let i = 0, p = 0; i < baseRgba.length; i += 4, p += 1) {
      const r = baseRgba[i] / 255;
      const g = baseRgba[i + 1] / 255;
      const b = baseRgba[i + 2] / 255;
      const y = clamp01(modelLuma[p]);
      const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5;
      const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5;
      output[i] = Math.round(clamp01(y + 1.402 * (cr - 0.5)) * 255);
      output[i + 1] = Math.round(clamp01(y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)) * 255);
      output[i + 2] = Math.round(clamp01(y + 1.772 * (cb - 0.5)) * 255);
      output[i + 3] = baseRgba[i + 3];
    }
    return output;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }

  ns.EcbsrOnnxUpscaler = EcbsrOnnxUpscaler;
})();
