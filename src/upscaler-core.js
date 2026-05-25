(function attachVideoGpuSuperResolutionCore() {
  const ns = window.VideoGpuSuperResolutionInternal ||= {};

  const vertexShader = `#version 300 es
    in vec2 a_position;
    out vec2 v_uv;

    void main() {
      v_uv = a_position * 0.5 + 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  function modeToInt(mode) {
    if (mode === "quality") return 1;
    if (mode === "performance") return 2;
    return 0;
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    const program = gl.createProgram();
    const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "Unable to link shader program");
    }

    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    return program;
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "Unable to compile shader");
    }

    return shader;
  }

  async function requestWebGpuAdapter(options = {}) {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available");
    }

    const { allowSoftware = false, preferCompatibility = false } = options;
    const attempts = preferCompatibility
      ? [
          ["compatibility", { powerPreference: "low-power", featureLevel: "compatibility", forceFallbackAdapter: false }],
          ["default", { forceFallbackAdapter: false }],
          ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }]
        ]
      : [
          ["high-performance", { powerPreference: "high-performance", forceFallbackAdapter: false }],
          ["default", { forceFallbackAdapter: false }],
          ["compatibility", { powerPreference: "high-performance", featureLevel: "compatibility", forceFallbackAdapter: false }]
        ];
    const failures = [];

    for (const [name, options] of attempts) {
      try {
        const adapter = await navigator.gpu.requestAdapter(options);
        if (adapter) {
          const info = getAdapterInfo(adapter);
          if (!allowSoftware && isSoftwareAdapter(adapter, info)) {
            failures.push(`${name}: software adapter (${info.description || info.architecture || "unknown"})`);
            continue;
          }
          return adapter;
        }
        failures.push(`${name}: null`);
      } catch (error) {
        failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    const label = allowSoftware ? "WebGPU adapter" : "hardware WebGPU adapter";
    throw new Error(`Unable to request ${label}. ${failures.join("; ")}`);
  }

  function getAdapterInfo(adapter) {
    const info = adapter.info || {};
    return {
      vendor: info.vendor || "",
      architecture: info.architecture || "",
      device: info.device || "",
      description: info.description || ""
    };
  }

  function isSoftwareAdapter(adapter, info = getAdapterInfo(adapter)) {
    if (adapter.isFallbackAdapter) return true;
    const text = `${info.vendor} ${info.architecture} ${info.device} ${info.description}`.toLowerCase();
    return text.includes("swiftshader") || text.includes("software");
  }

  Object.assign(ns, {
    vertexShader,
    modeToInt,
    createProgram,
    compileShader,
    requestWebGpuAdapter,
    getAdapterInfo,
    isSoftwareAdapter
  });
})();
