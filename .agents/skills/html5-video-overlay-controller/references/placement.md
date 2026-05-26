# Overlay Placement

## Placement Modes

Two useful modes:

- global overlay attached to `document.documentElement`
- local overlay attached near the player root

Local overlay is better for custom player stacks and fullscreen containers.

## Placement Inputs

Per-site profile can define:

- `videoSelectors`
- `overlayRootSelector`
- `localOverlay`
- `insertAfterVideo`
- `hideSource`
- `canvasOpacity`

## Rescan Triggers

Use a small, explicit set:

- `ResizeObserver`
- `MutationObserver`
- `loadedmetadata`
- `loadeddata`
- `playing`
- `fullscreenchange`

Debounce rescans so player UI churn does not create constant backend restarts.
