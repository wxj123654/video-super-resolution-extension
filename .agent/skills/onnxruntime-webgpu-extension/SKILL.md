---
name: onnxruntime-webgpu-extension
description: Use this skill when integrating ONNX Runtime Web into a browser extension, especially Chrome MV3 extensions that need static runtime assets, WebGPU execution, model registries, GPU-first preprocessing or compositing, and graceful CPU fallback. Helpful for `window.ort` runtime injection, `web_accessible_resources`, ORT WebGPU setup, GPU buffer tensors, and multi-model ONNX pipelines.
---

# ONNX Runtime WebGPU Extension

Use this skill for browser-extension ONNX stacks that must survive MV3 packaging, static asset rules, and mixed GPU/CPU fallback behavior.

## What This Skill Covers

- Static ORT runtime injection in an extension
- `web_accessible_resources` setup for runtime files, WASM, and models
- `window.ort` / vendor-runtime access from content scripts
- Model registries and per-model metadata
- GPU-first ONNX input/output handling with staged fallback

## Recommended Workflow

1. Treat runtime assets as extension resources.
   Prefer shipping ORT runtime files under a stable path such as `vendor/onnxruntime/` and expose them explicitly in the manifest.
2. Separate model metadata from execution logic.
   Define model ID, label, path, scale, input/output channels, packing mode, and compositing mode in one registry.
3. Configure ORT once.
   Set WebGPU adapter, WASM paths, thread count, and log level in one runtime configuration function.
4. Cache sessions by model.
   Use model-aware keys so switching models does not corrupt shared state.
5. Make GPU-first a policy, not a hard requirement.
   Try GPU input packing, GPU output tensors, and GPU compositing first. Fall back stage by stage instead of failing the whole engine.

## Good Patterns

- `manifest.web_accessible_resources` includes:
  - ORT runtime JS
  - ORT WASM and helper modules
  - model files
- Model registry is small, declarative, and separate from renderer code.
- Performance logs include:
  - input path (`gpu|cpu`)
  - output path (`gpu|cpu`)
  - composite path (`webgpu|2d`)
  - `pre/infer/post/total`

## Common Pitfalls

- Bundler rewrites ORT asset paths and breaks runtime fetches.
- Content script accidentally imports ESM chunks and becomes non-injectable.
- RGB models spend most of their time in JS pixel conversion, not inference.
- ORT GPU tensor support differs across browsers; hard GPU-only assumptions make the extension brittle.
- Model selection UI exists, but engine and model state are not normalized together.

## When To Read References

- Read `references/runtime.md` for ORT asset layout, static injection, and manifest guidance.
- Read `references/gpu-pipeline.md` for GPU-first preprocessing, output handling, and staged fallback.
