import { vertexShader, createProgram } from "../upscaler/webgl-utilities";
import ecbsrCompositeShader from "../shaders/ecbsr-post.glsl?raw";

export class EcbsrCompositeRenderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private baseTexture: WebGLTexture;
  private lumaTexture: WebGLTexture;
  private buffer: WebGLBuffer;
  private baseWidth = 0;
  private baseHeight = 0;
  private lumaWidth = 0;
  private lumaHeight = 0;
  private floatLinear: OES_texture_float_linear | null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });

    if (!gl) {
      throw new Error("WebGL2 is not available for ECBSR compositing");
    }
    if (!vertexShader || !ecbsrCompositeShader) {
      throw new Error("ECBSR compositing shaders are not loaded");
    }

    this.gl = gl;
    this.program = createProgram(gl, vertexShader, ecbsrCompositeShader);
    this.baseTexture = gl.createTexture()!;
    this.lumaTexture = gl.createTexture()!;
    this.buffer = gl.createBuffer()!;
    this.floatLinear = gl.getExtension("OES_texture_float_linear");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    gl.bindTexture(gl.TEXTURE_2D, this.lumaTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      this.floatLinear ? gl.LINEAR : gl.NEAREST,
    );
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MAG_FILTER,
      this.floatLinear ? gl.LINEAR : gl.NEAREST,
    );
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  uploadBaseFrame(
    data: ArrayBufferView,
    width: number,
    height: number,
  ): void {
    const gl = this.gl;
    this.baseWidth = width;
    this.baseHeight = height;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      data,
    );
  }

  uploadBaseFrameFromVideo(video: HTMLVideoElement): void {
    const gl = this.gl;
    this.baseWidth = video.videoWidth || 1;
    this.baseHeight = video.videoHeight || 1;
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      video,
    );
  }

  uploadLuma(data: Float32Array, width: number, height: number): void {
    const gl = this.gl;
    this.lumaWidth = width;
    this.lumaHeight = height;
    gl.bindTexture(gl.TEXTURE_2D, this.lumaTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      width,
      height,
      0,
      gl.RED,
      gl.FLOAT,
      data,
    );
  }

  render(): boolean {
    if (!this.baseWidth || !this.baseHeight || !this.lumaWidth || !this.lumaHeight) {
      return false;
    }

    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.baseTexture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_video"), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lumaTexture);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_luma"), 1);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_mix"), 1);

    const position = gl.getAttribLocation(this.program, "a_position");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    return true;
  }

  clear(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteTexture(this.baseTexture);
    gl.deleteTexture(this.lumaTexture);
    gl.deleteBuffer(this.buffer);
    gl.deleteProgram(this.program);
  }
}
