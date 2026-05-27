# MV3 Injection Flow

## Default Injection Sequence

1. Resolve the active tab with `chrome.tabs.query({active: true, currentWindow: true})`.
2. Load persisted popup settings.
3. Try a lightweight ping message to the content script.
4. If ping fails:
   - inject CSS
   - inject vendor runtime scripts, if any
   - inject the content entrypoint
5. Send the real update message after the page is connected.

## Why Root-Relative Asset Paths Matter

Popup bundles usually live under `dist/assets/...`. Injected files usually live at fixed extension-root paths such as:

- `content.js`
- `styles/overlay.css`
- `vendor/onnxruntime/ort.webgpu.min.js`

If popup code computes paths from `import.meta.url`, it often resolves to `/assets/...` and breaks `executeScript()`.

## Message Contract Pattern

- `PING`: confirms the content script is alive
- `UPDATE`: applies settings and returns current state
- `RESCAN`: re-detects the page target

Keep the response shape stable and compact. The popup should not infer page state from DOM directly.
