# Diagnostics And Failure Handling

## User-Facing States

Map low-level failures to clear statuses:

- no active tab
- page does not allow injection
- content script not connected
- extension was reloaded and page context is stale
- runtime feature unavailable

## Recommended Logging

Prefix logs by scope, for example:

- `[EXT][popup]`
- `[EXT][content]`
- `[EXT][controller]`

Log:

- active tab summary
- ping result
- injected asset list
- raw `sendMessage()` and `executeScript()` errors
- final user-facing status

## Extension Reload Edge Case

After reloading the extension, old content scripts may still exist on already-open pages.

Symptoms:

- `chrome-extension://invalid/`
- stale `chrome.runtime.getURL()` results
- duplicate init guards blocking reinjection

Preferred handling:

- detect stale context
- replace the old instance if possible
- otherwise tell the user to refresh the page
