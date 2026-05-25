(function attachVideoGpuSuperResolutionWebGl() {
  const ns = window.VideoGpuSuperResolutionInternal;
  const {
    vertexShader,
    tinyCnnFeatureShader,
    tinyCnnRefineShader,
    tinyCnnReconstructShader,
    createProgram,
    modeToInt
  } = ns;

  class TinyCnnUpscaler {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl2", {
        alpha: false,
        antialias: false,
        preserveDrawingBuffer: false,
        powerPreference: "high-performance"
      });

      if (!this.gl) {
        throw new Error("WebGL2 is not available");
      }
      if (!vertexShader || !tinyCnnFeatureShader || !tinyCnnRefineShader || !tinyCnnReconstructShader) {
        throw new Error("Tiny CNN shaders are not loaded");
      }

      this.featureProgram = createProgram(this.gl, vertexShader, tinyCnnFeatureShader);
      this.refineProgram = createProgram(this.gl, vertexShader, tinyCnnRefineShader);
      this.reconstructProgram = createProgram(this.gl, vertexShader, tinyCnnReconstructShader);
      this.videoTexture = this.gl.createTexture();
      this.featureTexture = this.gl.createTexture();
      this.refinedTexture = this.gl.createTexture();
      this.featureFramebuffer = this.gl.createFramebuffer();
      this.refinedFramebuffer = this.gl.createFramebuffer();
      this.buffer = this.gl.createBuffer();
      this.targetWidth = 0;
      this.targetHeight = 0;

      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
      this.gl.bufferData(
        this.gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
        this.gl.STATIC_DRAW
      );

      for (const texture of [this.videoTexture, this.featureTexture, this.refinedTexture]) {
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
      }
    }

    render(video, settings) {
      const gl = this.gl;
      this.ensureTargets();
      gl.disable(gl.BLEND);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.featureFramebuffer);
      gl.viewport(0, 0, this.targetWidth, this.targetHeight);
      this.drawFeaturePass(video);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.refinedFramebuffer);
      gl.viewport(0, 0, this.targetWidth, this.targetHeight);
      this.drawRefinePass(settings);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      this.drawReconstructPass(video, settings);
      return true;
    }

    ensureTargets() {
      const width = Math.max(1, this.canvas.width);
      const height = Math.max(1, this.canvas.height);
      if (width === this.targetWidth && height === this.targetHeight) return;

      this.targetWidth = width;
      this.targetHeight = height;
      this.allocateTarget(this.featureTexture, this.featureFramebuffer);
      this.allocateTarget(this.refinedTexture, this.refinedFramebuffer);
    }

    allocateTarget(texture, framebuffer) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.targetWidth, this.targetHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("Tiny model framebuffer is incomplete");
      }
    }

    drawFeaturePass(video) {
      const gl = this.gl;
      gl.useProgram(this.featureProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
      gl.uniform1i(gl.getUniformLocation(this.featureProgram, "u_video"), 0);
      gl.uniform2f(gl.getUniformLocation(this.featureProgram, "u_textureSize"), video.videoWidth || 1, video.videoHeight || 1);
      this.drawFullscreen(this.featureProgram);
    }

    drawRefinePass(settings) {
      const gl = this.gl;
      gl.useProgram(this.refineProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.featureTexture);
      gl.uniform1i(gl.getUniformLocation(this.refineProgram, "u_features"), 0);
      gl.uniform2f(gl.getUniformLocation(this.refineProgram, "u_targetSize"), this.targetWidth, this.targetHeight);
      gl.uniform1i(gl.getUniformLocation(this.refineProgram, "u_mode"), modeToInt(settings.mode));
      this.drawFullscreen(this.refineProgram);
    }

    drawReconstructPass(video, settings) {
      const gl = this.gl;
      gl.useProgram(this.reconstructProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
      gl.uniform1i(gl.getUniformLocation(this.reconstructProgram, "u_video"), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.refinedTexture);
      gl.uniform1i(gl.getUniformLocation(this.reconstructProgram, "u_features"), 1);
      gl.uniform2f(gl.getUniformLocation(this.reconstructProgram, "u_textureSize"), video.videoWidth || 1, video.videoHeight || 1);
      gl.uniform2f(gl.getUniformLocation(this.reconstructProgram, "u_targetSize"), this.targetWidth, this.targetHeight);
      gl.uniform1f(gl.getUniformLocation(this.reconstructProgram, "u_sharpness"), Number(settings.sharpness) || 0);
      gl.uniform1i(gl.getUniformLocation(this.reconstructProgram, "u_mode"), modeToInt(settings.mode));
      this.drawFullscreen(this.reconstructProgram);
    }

    drawFullscreen(program) {
      const gl = this.gl;
      const position = gl.getAttribLocation(program, "a_position");
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    destroy() {
      const gl = this.gl;
      for (const texture of [this.videoTexture, this.featureTexture, this.refinedTexture]) {
        gl.deleteTexture(texture);
      }
      gl.deleteFramebuffer(this.featureFramebuffer);
      gl.deleteFramebuffer(this.refinedFramebuffer);
      gl.deleteBuffer(this.buffer);
      gl.deleteProgram(this.featureProgram);
      gl.deleteProgram(this.refineProgram);
      gl.deleteProgram(this.reconstructProgram);
    }
  }

  ns.TinyCnnUpscaler = TinyCnnUpscaler;
})();
