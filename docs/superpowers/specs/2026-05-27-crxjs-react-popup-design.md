# CRXJS 2.4.0 Upgrade And React Popup Design

## Context

The project currently uses `@crxjs/vite-plugin` `^2.0.0-beta.28` with a root-level `popup.html` + `popup.ts` popup implementation and a manually injected `content.js` content script. The popup is implemented with direct DOM manipulation, the diagnostics UI is mixed into the popup, and the current build layout leaves extension pages and injected runtime code loosely organized.

There is one critical build constraint that governs the redesign:

- `content` is injected through `chrome.scripting.executeScript({ files: [...] })`
- the injected script must remain a single classic script file
- the final emitted content script cannot contain top-level `import`

This constraint exists because shared modules imported by both popup and content can trigger Rollup/Vite code splitting, which leaves `import` statements in the emitted content script and causes runtime failures when Chrome injects it as a non-module script.

## Goal

Upgrade the extension to `@crxjs/vite-plugin@2.4.0`, replace the existing popup with a React implementation using `shadcn/ui`, add a formal options page for diagnostics and advanced settings, and reorganize the build so that extension pages can benefit from modern Vite/React workflows without breaking the injected content script runtime.

## Non-Goals

- No sidepanel in this round
- No devtools panel in this round
- No heavy global state framework unless strictly necessary
- No broad architectural rewrite of the video upscaler runtime unrelated to the popup/options migration

## Requirements

### Functional

- Upgrade `@crxjs/vite-plugin` to `2.4.0`
- Replace the existing popup with a React-based popup
- Use `shadcn/ui` for the new extension UI
- Keep the existing popup control semantics, but improve information architecture and styling
- Move the current WebGPU diagnostics area out of popup into a formal options page
- Add a formal options page that includes settings, diagnostics, model/engine information, experimental toggles, and about/version information
- Preserve current popup-to-content messaging behavior for enable/disable, engine changes, display mode, scale, sharpness, overlay opacity, mode, target FPS, and rescan

### Build And Runtime

- Keep the injected content script as a single-file output suitable for `chrome.scripting.executeScript`
- Prevent popup/options shared code from causing content output chunking
- Keep a stable injected content script filename or equivalent deterministic injection target
- Continue exposing only the required `web_accessible_resources`

## Recommended Approach

Use a split architecture:

- React extension pages (`popup`, `options`) use normal Vite page entry behavior
- injected content runtime is treated as a dedicated single-file script output, not as a normal page-like entry that can share emitted chunks with React pages
- shared code is limited to safe cross-context modules such as message definitions, storage helpers, defaults, types, and small pure helpers

This is the recommended middle path between a minimal migration and an overbuilt full UI platform rewrite.

## Alternatives Considered

### Option 1: Minimal Migration

Upgrade CRXJS, add React, keep the popup entry style mostly unchanged, and add a small options page.

Pros:

- Lowest immediate risk
- Smallest code delta

Cons:

- Preserves root-level entry sprawl
- Does not properly address current build structure weaknesses
- Makes future extension page growth harder

### Option 2: UI Layer Refactor

Upgrade CRXJS, move extension pages into a dedicated UI subtree, rebuild popup in React with `shadcn/ui`, add a formal options page, and isolate the content script build boundary.

Pros:

- Solves the actual structural problem
- Improves long-term maintainability
- Matches requested scope

Cons:

- Larger one-time migration
- Requires setting up React page infrastructure and `shadcn/ui`

### Option 3: Full UI Platformization

Add routing shells, broad shared UI infrastructure, sidepanel-ready layout, and a deeper app-like architecture.

Pros:

- Maximum long-term extensibility

Cons:

- Overshoots current needs
- Adds unnecessary complexity for this upgrade

Selected option: **Option 2**

## Target Structure

```text
manifest.config.ts
vite.config.ts
src/
  content/
    index.ts
    controller.ts
    debug.ts
    site-profiles.ts
    video-utils.ts
  shared/
    extension/
      defaults.ts
      messages.ts
      storage.ts
    models/
      onnx-models.ts
    types/
      upscaler.ts
  ui/
    popup/
      index.html
      main.tsx
      app.tsx
      components/
      sections/
    options/
      index.html
      main.tsx
      app.tsx
      components/
      sections/
    lib/
      chrome.ts
      hooks.ts
      format.ts
    styles/
      globals.css
      tokens.css
```

Legacy root-level `popup.html`, `popup.ts`, and `styles/popup.css` are removed after migration.

## Entry And Build Boundaries

### Extension Pages

`popup` and `options` are standard Vite HTML entry pages backed by React.

These pages may:

- use React and `react-dom`
- use `@vitejs/plugin-react`
- use `shadcn/ui`
- share chunks with each other
- use normal page-oriented HMR during development

### Injected Content Script

The content runtime is not allowed to rely on emitted shared ESM chunks.

Implementation rule:

- content may import source modules during development/build authoring
- final emitted injected artifact must remain a single classic script file
- the content artifact must not be emitted as a chunk graph that leaves top-level `import`

Recommended implementation direction:

- generate the injected content artifact as a dedicated single-file script output
- use CRXJS script emission features such as `?script&iife` where appropriate for injection-safe output
- keep popup/options runtime code out of the content output boundary

### Shared Code Rules

Allowed shared code:

- message contracts
- storage access helpers
- default settings
- pure types
- small pure formatting or validation helpers

Avoid sharing:

- React hooks
- UI state containers
- large runtime modules with context-specific side effects
- extension page-specific orchestration code

The goal is to enable source reuse without allowing the content output to become code-split.

## Popup Information Architecture

The new popup is a fast control panel, not a diagnostics console.

### Popup Sections

1. Status card
   - current tab connection state
   - whether a video is detected
   - active engine

2. Primary control
   - enable/disable toggle

3. Core controls
   - engine selector
   - display mode selector
   - scale selector

4. Fine tuning
   - sharpness slider
   - overlay opacity slider
   - target FPS selector
   - mode selector

5. Quick actions
   - rescan videos
   - open full settings

### Popup UI Principles

- preserve current behavior while improving layout and readability
- make high-frequency actions accessible within one screen
- remove long-form diagnostics output from popup
- use `shadcn/ui` primitives such as `Card`, `Switch`, `Select`, `Slider`, `Badge`, and `Button`

## Options Page Information Architecture

The options page is the formal configuration and diagnostics center.

### Options Sections

1. `General`
   - full settings editor
   - same underlying settings schema as popup

2. `Models & Engines`
   - engine descriptions
   - model descriptions
   - limitations and usage notes

3. `Diagnostics`
   - WebGPU diagnostics
   - environment information
   - runtime status details
   - visible error reporting

4. `Experimental`
   - future toggles for advanced behavior
   - reserved space for non-default features

5. `About`
   - extension version
   - build metadata
   - documentation links

## Data Flow

### Storage

`chrome.storage.sync` remains the persistence layer.

Access should be centralized through a shared storage helper instead of inline page-specific `chrome.*` calls spread across the codebase.

### Popup Lifecycle

When popup opens:

1. read persisted settings
2. resolve the active tab
3. ensure the injected content runtime is available on the tab
4. send `VSR_PING` and/or `VSR_UPDATE`
5. render connection and controller state

### Options Lifecycle

When options opens:

1. load persisted settings
2. render full settings UI
3. only run diagnostics when the diagnostics area is entered or explicitly requested

This avoids making the settings page unnecessarily heavy on initial load.

## Styling Strategy

Use `shadcn/ui` with a project-owned style layer rather than a generic default look.

Styling direction:

- clean light-first UI
- explicit tokens for surface, border, accent, and status colors
- popup optimized for compact extension dimensions
- options page designed as a full-width settings surface
- avoid the current mixed handwritten CSS approach for page-level UI

## Upgrade And Migration Steps

1. Upgrade dependencies and Vite/CRXJS integration
2. Add React page infrastructure and `@vitejs/plugin-react`
3. Add `shadcn/ui` baseline setup
4. Establish new `src/ui/popup` and `src/ui/options` entries
5. Isolate content output so it remains injection-safe and single-file
6. Move shared defaults/messages/storage/types into a safe shared layer
7. Implement the React popup
8. Implement the options page and move diagnostics there
9. Remove legacy popup files
10. Verify extension loading and runtime behavior

## Risks

### Content Output Regresses To ESM Chunk Graph

This is the highest-risk failure mode. If content is allowed to share emitted runtime chunks with popup/options, Chrome injection can fail due to top-level `import` in the emitted script.

Mitigation:

- treat content output as a dedicated single-file injection artifact
- verify final emitted content file does not contain top-level `import`

### Popup And Options Diverge In Settings Behavior

Mitigation:

- use one shared settings schema
- centralize defaults and storage access

### Diagnostics Discoverability Drops

Mitigation:

- keep a prominent "full settings" or equivalent CTA inside popup
- make diagnostics a visible first-class section inside options

## Verification

Minimum verification after implementation:

1. `pnpm run typecheck`
2. `pnpm run build`
3. confirm `dist` contains popup and options page entries
4. confirm the injected content artifact remains safe for `chrome.scripting.executeScript`
5. confirm the emitted content artifact contains no top-level `import`
6. load the built extension in Chrome
7. verify popup opens and controls work
8. verify options page opens and diagnostics run
9. verify settings remain consistent between popup and options

## Implementation Boundaries For The Next Step

In the implementation plan, the work should cover:

- dependency upgrades
- React popup replacement
- options page creation
- content build boundary hardening
- manifest and Vite configuration updates
- migration of diagnostics from popup to options
- cleanup of legacy popup assets

It should not expand into unrelated runtime refactors beyond what is necessary to preserve existing behavior and enforce the content build constraint.
