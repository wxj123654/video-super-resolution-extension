---
name: html5-video-overlay-controller
description: Use this skill when building a page-level controller that finds the right HTML5 video on a site, attaches an overlay canvas, tracks SPA or fullscreen changes, and drives a rendering backend. Helpful for browser extensions or injected page tools that need video selection heuristics, site-specific profiles, overlay placement, rescan logic, render scheduling, and failure-state handling.
compatibility: Browser environment with HTML5 video elements. Used in Chrome extensions or injected page scripts.
---

# HTML5 Video Overlay Controller

Use this skill for browser code that needs to discover the correct `<video>`, attach a rendering surface, and keep it aligned while the page mutates.

## What This Skill Covers

- Candidate video discovery and scoring
- Site-profile based selector overrides
- Overlay canvas creation and placement
- Render scheduling based on video time or target FPS
- Rescan logic for SPA pages and dynamically replaced players

## Recommended Workflow

1. Build a small site-profile system.
   Keep host matching, selectors, overlay root, and hide-source rules in data instead of scattered conditionals.
2. Pick one active video deterministically.
   Score candidates by selector priority, renderability, playing state, and visible area.
3. Separate controller from backend.
   Controller owns page lifecycle; backend owns rendering.
4. Keep one overlay canvas and recreate only when backend state truly changes.
5. Add rescan hooks for:
   - mutations
   - fullscreen changes
   - metadata/playing events
   - route changes in SPA players

## Good Patterns

- `Controller.update()` applies settings and returns a compact state object.
- `Controller.rescan()` handles site/player replacement without rebuilding everything.
- `scheduleRescan()` debounces mutation storms.
- Failure state is keyed by active pipeline, not just by engine, so model changes can recover cleanly.

## Common Pitfalls

- Choosing the first `<video>` instead of the most relevant visible player.
- Overlay inserted into the wrong stacking context.
- Replace-mode hiding the source video too early.
- Mutation observers causing endless restart loops after an engine failure.
- Non-engine settings being treated as active when the current backend ignores them.

## When To Read References

- Read `references/selection.md` for video selection and pipeline-key patterns.
- Read `references/placement.md` for overlay placement, fullscreen, and local-vs-global canvas rules.
