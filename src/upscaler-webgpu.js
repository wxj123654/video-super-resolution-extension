(function attachVideoGpuSuperResolutionWebGpu() {
  const ns = window.VideoGpuSuperResolutionInternal;
  const { requestWebGpuAdapter, getAdapterInfo, webGpuShader, modeToInt } = ns;

  class WebGpuUpscaler {
    constructor(canvas) {
      this.canvas = canvas;
      this.adapter = null;
      this.device = null;
      this.context = null;
      this.pipeline = null;
      this.sampler = null;
      this.uniformBuffer = null;
      this.bindGroupLayout = null;
      this.format = null;
      this.configured = false;
      this.failed = false;
      this.initError = null;
      this.lastCanvasWidth = 0;
      this.lastCanvasHeight = 0;
      this.initPromise = this.init().catch((error) => {
        this.failed = true;
        this.initError = error;
      });
    }

    async init() {
      this.adapter = await requestWebGpuAdapter();
      this.device = await this.adapter.requestDevice();
      this.context = this.canvas.getContext("webgpu");
      if (!this.context) {
        throw new Error("Unable to create WebGPU canvas context");
      }

      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.sampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });
      this.uniformBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      const module = this.device.createShaderModule({ code: webGpuShader });
      this.pipeline = await this.device.createRenderPipelineAsync({
        layout: "auto",
        vertex: {
          module,
          entryPoint: "vertexMain"
        },
        fragment: {
          module,
          entryPoint: "fragmentMain",
          targets: [{ format: this.format }]
        },
        primitive: {
          topology: "triangle-list"
        }
      });
      this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
      this.configureContext();
    }

    render(video, settings) {
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

      const width = Math.max(1, video.videoWidth || this.canvas.width || 1);
      const height = Math.max(1, video.videoHeight || this.canvas.height || 1);
      const params = new Float32Array([
        1 / width,
        1 / height,
        Number(settings.sharpness) || 0,
        modeToInt(settings.mode)
      ]);
      this.device.queue.writeBuffer(this.uniformBuffer, 0, params);

      const externalTexture = this.device.importExternalTexture({ source: video });
      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: externalTexture },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.uniformBuffer } }
        ]
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store"
          }
        ]
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return true;
    }

    configureContext() {
      if (!this.context || !this.device) return;
      const width = Math.max(1, this.canvas.width || 1);
      const height = Math.max(1, this.canvas.height || 1);
      if (this.configured && width === this.lastCanvasWidth && height === this.lastCanvasHeight) {
        return;
      }

      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "opaque"
      });
      this.lastCanvasWidth = width;
      this.lastCanvasHeight = height;
      this.configured = true;
    }

    destroy() {
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

  ns.WebGpuUpscaler = WebGpuUpscaler;
  ns.getWebGpuAdapterInfo = getAdapterInfo;
})();
