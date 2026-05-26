# Video Selection And State

## Candidate Scoring Pattern

Useful scoring inputs:

- selector priority
- renderable readiness
- currently playing bonus
- visible area

This avoids brittle “first match wins” behavior.

## Pipeline Key Pattern

If the renderer can vary by engine and model, derive a single pipeline key such as:

- `webgpu`
- `tiny-cnn`
- `ecbsr:model-id`

Use it to decide when to rebuild the backend and when to clear failure state.

## Error Mapping

Map low-level backend errors into page-level user messages:

- ONNX initialization failed
- hardware WebGPU unavailable
- cross-origin video frame access blocked
- rendering failed and source video restored
