export const CONTENT_SCRIPT_FILE = "content.js";
export const CONTENT_STYLE_FILE = "styles/overlay.css";
export const ORT_RUNTIME_FILES = [
  "vendor/onnxruntime/ort.webgpu.min.js",
  "vendor/onnxruntime/ort-wasm-simd-threaded.wasm",
  "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm",
  "vendor/onnxruntime/ort-wasm-simd-threaded.mjs",
  "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs",
] as const;
export const ORT_RUNTIME_SCRIPT_FILES = [ORT_RUNTIME_FILES[0]] as const;
