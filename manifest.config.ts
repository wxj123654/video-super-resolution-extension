import { defineManifest } from "@crxjs/vite-plugin";

export default defineManifest({
  manifest_version: 3,
  name: "Video GPU Super Resolution",
  description:
    "Realtime WebGL/WebGPU super-resolution for HTML5 videos on the current page.",
  version: "0.3.0",
  minimum_chrome_version: "113",
  action: {
    default_title: "Video GPU Super Resolution",
    default_popup: "popup.html",
  },
  permissions: ["activeTab", "scripting", "storage"],
  host_permissions: ["<all_urls>"],
  web_accessible_resources: [
    {
      resources: [
        "models/*",
        "vendor/onnxruntime/ort.webgpu.min.js",
        "vendor/onnxruntime/ort-wasm-simd-threaded.wasm",
        "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm",
        "vendor/onnxruntime/ort-wasm-simd-threaded.mjs",
        "vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs",
        "styles/overlay.css",
        "content.js",
      ],
      matches: ["<all_urls>"],
    },
  ],
});
