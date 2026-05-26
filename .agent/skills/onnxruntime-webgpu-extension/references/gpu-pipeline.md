# GPU-First ONNX Pipeline

## Preferred Resource Flow

1. `texture_external` samples the video frame
2. GPU preprocessing packs input into a buffer layout ORT accepts
3. ORT runs with WebGPU execution provider
4. Output stays on GPU when possible
5. GPU compositing writes the final frame to the visible canvas

## Fallback Strategy

Do not treat GPU support as all-or-nothing.

- GPU input pack fails -> use CPU `getImageData()` / tensor packing
- GPU output tensor path fails -> let ORT produce CPU-readable output
- GPU compositor fails -> use 2D canvas fallback

Keep the fallback stage explicit in logs.

## Performance Interpretation

- high `pre` usually means video readback or CPU tensor packing
- high `infer` means model/runtime cost
- high `post` usually means tensor-to-image conversion or CPU compositing

For RGB models, `post` often dominates if output is unpacked in JS pixel loops.
