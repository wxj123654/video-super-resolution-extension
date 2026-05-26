---
name: chrome-mv3-active-tab-injection
description: Use this skill when building or debugging a Chrome Manifest V3 extension that injects CSS or content scripts from a popup or action button into the active tab. Helpful for popup-driven `chrome.scripting.insertCSS()` / `executeScript()` flows, `activeTab` permission usage, content-script ping/handshake design, extension reload edge cases, and user-facing diagnostics when a page cannot be injected.
---

# Chrome MV3 Active Tab Injection

Use this skill for popup-driven MV3 injection, especially when the extension only injects code on demand instead of declaring global `content_scripts`.

## What This Skill Covers

- Active-tab injection from popup or action UI
- `chrome.scripting.insertCSS()` and `chrome.scripting.executeScript()` ordering
- Ping-before-inject handshake (`VSR_PING`-style flow)
- User-facing error states for unsupported pages
- Extension reload / stale content-script edge cases

## Recommended Workflow

1. Inspect the injection chain first.
   Check `manifest`, popup entrypoint, content entrypoint, and any runtime message handshake before changing code.
2. Prefer root-relative extension asset paths.
   Use extension root paths such as `content.js` or `styles/overlay.css` in `executeScript()` / `insertCSS()`. Do not derive injected asset paths from `import.meta.url` in popup bundles.
3. Ping before injecting.
   Try `chrome.tabs.sendMessage(tabId, {type: "PING"})` first. Inject only if that fails.
4. Keep injection order explicit.
   Inject CSS first, then any vendor runtime files, then the content entrypoint.
5. Differentiate user-visible failure modes.
   Treat `chrome://`, extension store pages, new-tab pages, and missing active tab as separate statuses.

## Good Patterns

- Popup owns the connection workflow and persistent settings.
- Content script owns page state and responds to `PING`, `UPDATE`, `RESCAN`-style messages.
- Runtime logs include both user-facing status and raw error detail.
- Debug builds keep sourcemaps and readable console output.

## Common Pitfalls

- Injected file path resolves relative to a bundled popup chunk instead of the extension root.
- Content script bundle accidentally becomes ESM and fails under classic-script injection.
- Duplicate-init guard throws too early and prevents replacing stale content scripts after extension reload.
- Popup swallows the original `sendMessage` / `executeScript` error and only shows a vague failure message.
- Settings UI implies a field is active even when the current engine ignores it.

## When To Read References

- Read `references/flow.md` when designing or repairing an MV3 popup-to-content handshake.
- Read `references/diagnostics.md` when you need better debug UX, stale-context handling, or unsupported-page messaging.
