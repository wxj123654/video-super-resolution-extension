(function attachVideoGpuSuperResolution() {
  if (window.VideoGpuSuperResolution) {
    return;
  }

  const ns = window.VideoGpuSuperResolutionInternal;

  class Upscaler {
    constructor(canvas, settings = {}) {
      this.backend = getBackend(settings.engine);

      this.impl = createBackend(canvas, this.backend);
    }

    render(video, settings) {
      return this.impl.render(video, settings);
    }

    destroy() {
      this.impl.destroy();
    }
  }

  function getBackend(engine) {
    if (engine === "webgpu") return "webgpu";
    if (engine === "ecbsr") return "ecbsr";
    if (engine === "tiny-cnn") return "tiny-cnn";
    return "tiny-cnn";
  }

  function createBackend(canvas, backend) {
    if (backend === "webgpu") return new ns.WebGpuUpscaler(canvas);
    if (backend === "ecbsr") return new ns.EcbsrOnnxUpscaler(canvas);
    return new ns.TinyCnnUpscaler(canvas);
  }

  window.VideoGpuSuperResolution = { Upscaler };
})();
