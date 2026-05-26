import type { Settings, UpscalerImpl } from "../upscaler/types";
import {
  vertexShader,
  createProgram,
  modeToInt,
} from "../upscaler/webgl-utilities";
import tinyCnnFeatureShader from "../shaders/tiny-cnn-feature.glsl?raw";
import tinyCnnRefineShader from "../shaders/tiny-cnn-refine.glsl?raw";
import tinyCnnReconstructShader from "../shaders/tiny-cnn-reconstruct.glsl?raw";

export class TinyCnnUpscaler implements UpscalerImpl {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private featureProgram: WebGLProgram;
  private refineProgram: WebGLProgram;
  private reconstructProgram: WebGLProgram;
  private videoTexture: WebGLTexture;
  private featureTexture: WebGLTexture;
  private refinedTexture: WebGLTexture;
  private featureFramebuffer: WebGLFramebuffer;
  private refinedFramebuffer: WebGLFramebuffer;
  private buffer: WebGLBuffer;
  private targetWidth = 0;
  private targetHeight = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });

    if (!gl) {
      throw new Error("WebGL2 is not available");
    }
    if (
      !vertexShader ||
      !tinyCnnFeatureShader ||
      !tinyCnnRefineShader ||
      !tinyCnnReconstructShader
    ) {
      throw new Error("Tiny CNN shaders are not loaded");
    }

    this.gl = gl;
    this.featureProgram = createProgram(gl, vertexShader, tinyCnnFeatureShader);
    this.refineProgram = createProgram(gl, vertexShader, tinyCnnRefineShader);
    this.reconstructProgram = createProgram(
      gl,
      vertexShader,
      tinyCnnReconstructShader,
    );
    this.videoTexture = gl.createTexture()!;
    this.featureTexture = gl.createTexture()!;
    this.refinedTexture = gl.createTexture()!;
    this.featureFramebuffer = gl.createFramebuffer()!;
    this.refinedFramebuffer = gl.createFramebuffer()!;
    this.buffer = gl.createBuffer()!;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    for (const texture of [
      this.videoTexture,
      this.featureTexture,
      this.refinedTexture,
    ]) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }
  }

  render(video: HTMLVideoElement, settings: Settings): boolean {
    const gl = this.gl;
    this.ensureTargets();
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      video,
    );

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

  private ensureTargets(): void {
    const width = Math.max(1, this.canvas.width);
    const height = Math.max(1, this.canvas.height);
    if (width === this.targetWidth && height === this.targetHeight) return;

    this.targetWidth = width;
    this.targetHeight = height;
    this.allocateTarget(this.featureTexture, this.featureFramebuffer);
    this.allocateTarget(this.refinedTexture, this.refinedFramebuffer);
  }

  private allocateTarget(
    texture: WebGLTexture,
    framebuffer: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      this.targetWidth,
      this.targetHeight,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("Tiny model framebuffer is incomplete");
    }
  }

  private drawFeaturePass(video: HTMLVideoElement): void {
    const gl = this.gl;
    gl.useProgram(this.featureProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.uniform1i(
      gl.getUniformLocation(this.featureProgram, "u_video"),
      0,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.featureProgram, "u_textureSize"),
      video.videoWidth || 1,
      video.videoHeight || 1,
    );
    this.drawFullscreen(this.featureProgram);
  }

  private drawRefinePass(settings: Settings): void {
    const gl = this.gl;
    gl.useProgram(this.refineProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.featureTexture);
    gl.uniform1i(
      gl.getUniformLocation(this.refineProgram, "u_features"),
      0,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.refineProgram, "u_targetSize"),
      this.targetWidth,
      this.targetHeight,
    );
    gl.uniform1i(
      gl.getUniformLocation(this.refineProgram, "u_mode"),
      modeToInt(settings.mode),
    );
    this.drawFullscreen(this.refineProgram);
  }

  private drawReconstructPass(
    video: HTMLVideoElement,
    settings: Settings,
  ): void {
    const gl = this.gl;
    gl.useProgram(this.reconstructProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.uniform1i(
      gl.getUniformLocation(this.reconstructProgram, "u_video"),
      0,
    );
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.refinedTexture);
    gl.uniform1i(
      gl.getUniformLocation(this.reconstructProgram, "u_features"),
      1,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.reconstructProgram, "u_textureSize"),
      video.videoWidth || 1,
      video.videoHeight || 1,
    );
    gl.uniform2f(
      gl.getUniformLocation(this.reconstructProgram, "u_targetSize"),
      this.targetWidth,
      this.targetHeight,
    );
    gl.uniform1f(
      gl.getUniformLocation(this.reconstructProgram, "u_sharpness"),
      Number(settings.sharpness) || 0,
    );
    gl.uniform1i(
      gl.getUniformLocation(this.reconstructProgram, "u_mode"),
      modeToInt(settings.mode),
    );
    this.drawFullscreen(this.reconstructProgram);
  }

  private drawFullscreen(program: WebGLProgram): void {
    const gl = this.gl;
    const position = gl.getAttribLocation(program, "a_position");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  destroy(): void {
    const gl = this.gl;
    for (const texture of [
      this.videoTexture,
      this.featureTexture,
      this.refinedTexture,
    ]) {
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
