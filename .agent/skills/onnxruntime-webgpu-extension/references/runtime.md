# ORT Runtime Layout For Extensions

## Stable Asset Layout

Recommended extension-root layout:

- `vendor/onnxruntime/ort.webgpu.min.js`
- `vendor/onnxruntime/ort-wasm-simd-threaded.wasm`
- `vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm`
- `vendor/onnxruntime/ort-wasm-simd-threaded.mjs`
- `vendor/onnxruntime/ort-wasm-simd-threaded.jsep.mjs`
- `models/*.onnx`

Expose these through `web_accessible_resources`.

## Runtime Access Pattern

- Inject ORT runtime before the content script.
- Access it from content code via `window.ort`.
- Use `chrome.runtime.getURL()` for model and WASM paths.

## Model Registry Pattern

Each model definition should carry enough metadata to drive runtime behavior:

- `id`
- `label`
- `modelPath`
- `scale`
- `inputChannels`
- `outputChannels`
- `inputPacking`
- `outputPacking`
- `compositeMode`
- `sizePolicy`

This keeps UI, controller logic, and backend execution aligned.
