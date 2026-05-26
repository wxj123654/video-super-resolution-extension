# ONNX Model Catalog

This extension currently ships with two built-in ONNX models under `public/models/`.

| ID | File | Input | Output | Scale | Compositor | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ecbsr_x2_m4c8_y` | `ecbsr_x2_m4c8_y.onnx` | `NCHW`, 1-channel luma | `NCHW`, 1-channel luma | `2x` | `luma_replace` | Default ECBSR mobile luma model. Width is aligned to 32 pixels. |
| `rgb_bicubic_x2` | `rgb_bicubic_x2.onnx` | `NCHW`, 3-channel RGB | `NCHW`, 3-channel RGB | `2x` | `rgb_replace` | Lightweight RGB baseline model used to validate the full RGB ONNX path. |

When adding a new model:

1. Add the model file to `public/models/`.
2. Register it in `src/backends/onnx-models.ts`.
3. Set the correct `inputChannels`, `outputChannels`, `inputColorSpace`, and `compositor`.
4. Update `sizePolicy` so the runtime can align inputs without hardcoded assumptions.
