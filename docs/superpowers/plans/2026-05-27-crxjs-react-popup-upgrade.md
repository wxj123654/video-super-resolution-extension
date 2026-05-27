# CRXJS React Popup Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the extension to `@crxjs/vite-plugin@2.4.0`, rebuild the popup with React, add a formal options page, and harden the content-script build so the injected artifact always remains a single classic script with no top-level `import`.

**Architecture:** The extension keeps one injected runtime (`content`) and two React-backed extension pages (`popup`, `options`). Shared extension code moves into `src/shared/extension`, while page UI lives under `src/ui`. Build regression tests guard the non-negotiable constraint that the emitted content artifact must stay injection-safe even when popup and options share source modules.

**Tech Stack:** TypeScript, Vite 5, `@crxjs/vite-plugin@2.4.0`, React, Vitest, Testing Library, Tailwind CSS, local `shadcn/ui`-style primitives.

---

## File Structure Map

### Existing files to modify

- `package.json`
  - Add React/build/test dependencies and scripts.
- `tsconfig.json`
  - Expand includes for new UI/shared/test paths.
- `manifest.config.ts`
  - Point popup to the new HTML entry and add `options_page`.
  - Keep only required `web_accessible_resources`.
- `vite.config.ts`
  - Add React plugin, content single-file output rules, and test config.
- `src/content/index.ts`
  - Keep controller/message entry, but ensure it builds as the dedicated injected runtime.
- `README.md`
  - Update developer instructions and page entry descriptions after migration.

### Existing files to delete

- `popup.html`
- `popup.ts`
- `styles/popup.css`

### New shared files

- `src/shared/extension/defaults.ts`
  - Default settings and ONNX model option metadata used by popup/options.
- `src/shared/extension/settings.ts`
  - `normalizeSettings`, validation helpers, and UI-friendly labels.
- `src/shared/extension/messages.ts`
  - Message type helpers and lightweight runtime guards.
- `src/shared/extension/storage.ts`
  - Centralized `chrome.storage.sync` read/write helpers.
- `src/shared/extension/injection.ts`
  - Exports the emitted content script file name and page injection helpers.
- `src/shared/extension/diagnostics.ts`
  - Shared formatting helpers for diagnostics output.

### New UI files

- `src/ui/styles/globals.css`
  - Tailwind base/components/utilities and design tokens.
- `src/ui/lib/utils.ts`
  - `cn()` helper for class composition.
- `src/ui/lib/chrome.ts`
  - Thin typed wrappers around `chrome.tabs`, `chrome.scripting`, and messaging.
- `src/ui/lib/popup-controller.ts`
  - Popup orchestration logic separated from React rendering.
- `src/ui/lib/options-controller.ts`
  - Options page load/save/diagnostics orchestration.
- `src/ui/components/ui/button.tsx`
- `src/ui/components/ui/card.tsx`
- `src/ui/components/ui/switch.tsx`
- `src/ui/components/ui/select.tsx`
- `src/ui/components/ui/slider.tsx`
- `src/ui/components/ui/badge.tsx`
  - Local `shadcn/ui`-style primitives needed by popup/options.
- `src/ui/popup/index.html`
- `src/ui/popup/main.tsx`
- `src/ui/popup/app.tsx`
- `src/ui/options/index.html`
- `src/ui/options/main.tsx`
- `src/ui/options/app.tsx`

### New test files

- `src/shared/extension/settings.test.ts`
  - Unit coverage for settings normalization.
- `src/ui/popup/app.test.tsx`
  - Component coverage for popup information architecture.
- `src/ui/options/app.test.tsx`
  - Component coverage for options sections and diagnostics placement.
- `tests/build/extension-build.test.ts`
  - Regression test that builds the extension and asserts the emitted content script is still a single classic script without top-level `import`.
- `tests/setup.ts`
  - Testing Library and DOM matcher setup.

## Task 1: Add Shared Extension Modules And Test Harness

**Files:**
- Create: `src/shared/extension/defaults.ts`
- Create: `src/shared/extension/settings.ts`
- Create: `src/shared/extension/messages.ts`
- Create: `src/shared/extension/storage.ts`
- Create: `src/shared/extension/settings.test.ts`
- Create: `tests/setup.ts`
- Modify: `package.json`
- Modify: `tsconfig.json`
- Modify: `vite.config.ts`

- [ ] **Step 1: Write the failing shared-settings test**

```ts
// src/shared/extension/settings.test.ts
import { describe, expect, it } from "vitest";

import {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  normalizeSettings,
} from "./settings";

describe("normalizeSettings", () => {
  it("falls back to defaults for unsupported engine and model", () => {
    expect(
      normalizeSettings({
        enabled: true,
        engine: "unknown",
        modelId: "missing-model",
      }),
    ).toMatchObject({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: DEFAULT_SETTINGS.engine,
      modelId: DEFAULT_ONNX_MODEL_ID,
    });
  });

  it("preserves valid engine and model values", () => {
    expect(
      normalizeSettings({
        enabled: true,
        engine: "ecbsr",
        modelId: "rgb_bicubic_x2",
      }),
    ).toMatchObject({
      ...DEFAULT_SETTINGS,
      enabled: true,
      engine: "ecbsr",
      modelId: "rgb_bicubic_x2",
    });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm exec vitest run src/shared/extension/settings.test.ts`

Expected: FAIL with `Cannot find module './settings'` or missing `vitest` command, because the shared module and test tooling do not exist yet.

- [ ] **Step 3: Add React, Tailwind, and test dependencies**

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "build:debug": "tsc --noEmit && vite build --mode debug",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@radix-ui/react-select": "^2.1.0",
    "@radix-ui/react-slider": "^1.2.0",
    "@radix-ui/react-switch": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "lucide-react": "^0.469.0",
    "onnxruntime-web": "^1.21.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "tailwind-merge": "^2.5.5"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.2",
    "@crxjs/vite-plugin": "^2.4.0",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.15",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 4: Implement shared settings, storage, message helpers, and test config**

```ts
// src/shared/extension/defaults.ts
import type { EngineType, Settings } from "@src/upscaler/types";

export const ONNX_MODEL_OPTIONS = [
  { id: "ecbsr_x2_m4c8_y", label: "ECBSR Y-only x2" },
  { id: "rgb_bicubic_x2", label: "RGB Bicubic x2" },
] as const;

export const DEFAULT_ONNX_MODEL_ID = ONNX_MODEL_OPTIONS[0].id;

export const VALID_ENGINES: ReadonlySet<EngineType> = new Set([
  "webgpu",
  "tiny-cnn",
  "ecbsr",
]);

export const DEFAULT_SETTINGS: Settings = {
  enabled: false,
  scale: 1.5,
  sharpness: 0.65,
  mode: "balanced",
  overlayOpacity: 0.8,
  displayMode: "overlay",
  engine: "tiny-cnn",
  modelId: DEFAULT_ONNX_MODEL_ID,
  targetFps: "auto",
};
```

```ts
// src/shared/extension/settings.ts
import type { EngineType, Settings } from "@src/upscaler/types";

import {
  DEFAULT_ONNX_MODEL_ID,
  DEFAULT_SETTINGS,
  ONNX_MODEL_OPTIONS,
  VALID_ENGINES,
} from "./defaults";

const VALID_ONNX_MODEL_IDS = new Set(ONNX_MODEL_OPTIONS.map((option) => option.id));

export { DEFAULT_ONNX_MODEL_ID, DEFAULT_SETTINGS, ONNX_MODEL_OPTIONS };

export function normalizeOnnxModelId(modelId: unknown): string {
  return typeof modelId === "string" && VALID_ONNX_MODEL_IDS.has(modelId)
    ? modelId
    : DEFAULT_ONNX_MODEL_ID;
}

export function normalizeSettings(
  value: Partial<Settings> & { engine?: string } = {},
): Settings {
  return {
    ...DEFAULT_SETTINGS,
    ...value,
    engine: VALID_ENGINES.has(value.engine as EngineType)
      ? (value.engine as EngineType)
      : DEFAULT_SETTINGS.engine,
    modelId: normalizeOnnxModelId(value.modelId),
  };
}
```

```ts
// src/shared/extension/messages.ts
import type { Settings, VsrMessage, VsrMessageType } from "@src/upscaler/types";

export function createMessage(
  type: VsrMessageType,
  settings?: Settings,
): VsrMessage {
  return settings ? { type, settings } : { type };
}
```

```ts
// src/shared/extension/storage.ts
import type { Settings } from "@src/upscaler/types";

import { DEFAULT_SETTINGS, normalizeSettings } from "./settings";

export async function loadSettings(): Promise<Settings> {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return normalizeSettings({ ...DEFAULT_SETTINGS, ...stored });
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.sync.set(settings);
}
```

```ts
// tests/setup.ts
import "@testing-library/jest-dom/vitest";
```

```ts
// vite.config.ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
  },
});
```

```json
// tsconfig.json
{
  "include": [
    "src/**/*",
    "tests/**/*",
    "manifest.config.ts",
    "vite.config.ts"
  ]
}
```

- [ ] **Step 5: Run the shared-settings test to verify it passes**

Run: `pnpm exec vitest run src/shared/extension/settings.test.ts`

Expected: PASS with `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add package.json tsconfig.json vite.config.ts src/shared/extension tests/setup.ts
git commit -m "test: add shared extension settings foundation"
```

## Task 2: Guard The Content Build Boundary And Introduce New Page Entries

**Files:**
- Create: `tests/build/extension-build.test.ts`
- Create: `src/shared/extension/injection.ts`
- Create: `src/ui/popup/index.html`
- Create: `src/ui/options/index.html`
- Create: `src/ui/styles/globals.css`
- Modify: `manifest.config.ts`
- Modify: `vite.config.ts`

- [ ] **Step 1: Write the failing build regression test**

```ts
// tests/build/extension-build.test.ts
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const fullPath = join(dir, entry);
    return statSync(fullPath).isDirectory() ? walk(fullPath) : [fullPath];
  });
}

describe("extension build output", () => {
  it("emits popup, options, and a classic injected content bundle", () => {
    rmSync("dist", { recursive: true, force: true });

    execFileSync("pnpm", ["run", "build"], {
      cwd: process.cwd(),
      stdio: "inherit",
      shell: true,
    });

    const manifest = readFileSync("dist/manifest.json", "utf8");
    const iifeFiles = walk("dist").filter((file) => file.endsWith(".iife.js"));
    const content = readFileSync(iifeFiles[0], "utf8");

    expect(manifest).toContain("\"default_popup\": \"popup.html\"");
    expect(manifest).toContain("\"options_page\": \"options.html\"");
    expect(iifeFiles).toHaveLength(1);
    expect(content).not.toMatch(/^\s*import\s/m);
  });
});
```

- [ ] **Step 2: Run the build regression test to verify it fails**

Run: `pnpm exec vitest run tests/build/extension-build.test.ts`

Expected: FAIL because `options.html` is not emitted yet and the build layout has not been migrated to the new page entries.

- [ ] **Step 3: Add the dedicated injection module, new HTML entries, and build rules**

```ts
// src/shared/extension/injection.ts
import contentScriptFile from "@src/content/index.ts?script&iife";

export const CONTENT_SCRIPT_FILE = contentScriptFile;
export const CONTENT_STYLE_FILE = "styles/overlay.css";
export const ORT_RUNTIME_FILES = [
  "vendor/onnxruntime/ort.webgpu.min.js",
] as const;
```

```html
<!-- src/ui/popup/index.html -->
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Video GPU Super Resolution</title>
    <script type="module" src="./main.tsx"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

```html
<!-- src/ui/options/index.html -->
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Video GPU Super Resolution Settings</title>
    <script type="module" src="./main.tsx"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

```css
/* src/ui/styles/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --background: 46 33% 97%;
  --foreground: 204 22% 15%;
  --card: 0 0% 100%;
  --muted: 210 16% 92%;
  --muted-foreground: 208 12% 40%;
  --border: 35 20% 82%;
  --accent: 164 56% 33%;
  color-scheme: light;
}

body {
  @apply bg-[hsl(var(--background))] text-[hsl(var(--foreground))];
}
```

```ts
// manifest.config.ts
import { defineManifest } from "@crxjs/vite-plugin";

export default defineManifest({
  manifest_version: 3,
  name: "Video GPU Super Resolution",
  description:
    "Realtime WebGL/WebGPU super-resolution for HTML5 videos on the current page.",
  version: "0.3.0",
  minimum_chrome_version: "113",
  action: {
    default_title: "Video GPU Super Resolution",
    default_popup: "popup.html",
  },
  options_page: "options.html",
  permissions: ["activeTab", "scripting", "storage"],
  host_permissions: ["<all_urls>"],
  web_accessible_resources: [
    {
      resources: ["models/*", "styles/overlay.css"],
      matches: ["<all_urls>"],
    },
  ],
});
```

```ts
// vite.config.ts
import react from "@vitejs/plugin-react";
import { crx } from "@crxjs/vite-plugin";
import { resolve } from "node:path";
import { defineConfig } from "vite";

import manifest from "./manifest.config";

export default defineConfig(({ mode }) => {
  const isDebugBuild = mode === "debug";

  return {
    build: {
      minify: isDebugBuild ? false : undefined,
      sourcemap: isDebugBuild,
      target: "chrome113",
      rollupOptions: {
        input: {
          popup: resolve(__dirname, "src/ui/popup/index.html"),
          options: resolve(__dirname, "src/ui/options/index.html"),
        },
        output: {
          entryFileNames: (chunkInfo) =>
            chunkInfo.name === "index"
              ? "assets/[name]-[hash].js"
              : "assets/[name]-[hash].js",
          chunkFileNames: "assets/[name]-[hash].js",
          assetFileNames: "assets/[name]-[hash][extname]",
        },
      },
    },
    resolve: {
      alias: {
        "@src": "/src",
      },
    },
    define: {
      __VSR_DEBUG__: JSON.stringify(isDebugBuild),
    },
    plugins: [react(), crx({ manifest })],
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./tests/setup.ts"],
    },
  };
});
```

- [ ] **Step 4: Run the build regression test to verify it passes**

Run: `pnpm exec vitest run tests/build/extension-build.test.ts`

Expected: PASS with `1 passed`, confirming that the build emits popup/options entries and the injected content artifact still has no top-level `import`.

- [ ] **Step 5: Commit**

```bash
git add manifest.config.ts vite.config.ts src/shared/extension/injection.ts src/ui/popup/index.html src/ui/options/index.html src/ui/styles/globals.css tests/build/extension-build.test.ts
git commit -m "test: guard extension build outputs"
```

## Task 3: Replace The Legacy Popup With A React Control Panel

**Files:**
- Create: `src/ui/lib/utils.ts`
- Create: `src/ui/lib/chrome.ts`
- Create: `src/ui/lib/popup-controller.ts`
- Create: `src/ui/components/ui/button.tsx`
- Create: `src/ui/components/ui/card.tsx`
- Create: `src/ui/components/ui/switch.tsx`
- Create: `src/ui/components/ui/select.tsx`
- Create: `src/ui/components/ui/slider.tsx`
- Create: `src/ui/components/ui/badge.tsx`
- Create: `src/ui/popup/main.tsx`
- Create: `src/ui/popup/app.tsx`
- Create: `src/ui/popup/app.test.tsx`
- Delete: `popup.html`
- Delete: `popup.ts`
- Delete: `styles/popup.css`

- [ ] **Step 1: Write the failing popup UI test**

```tsx
// src/ui/popup/app.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PopupApp } from "./app";

describe("PopupApp", () => {
  it("renders core controls and routes diagnostics to full settings", () => {
    render(
      <PopupApp
        state={{
          statusText: "Connected",
          hasVideo: true,
          engineLabel: "ECBSR",
        }}
        onOpenOptions={vi.fn()}
        onRescan={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: /video gpu super resolution/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /full settings/i })).toBeInTheDocument();
    expect(screen.queryByText(/webgpu diagnostics/i)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the popup UI test to verify it fails**

Run: `pnpm exec vitest run src/ui/popup/app.test.tsx`

Expected: FAIL with `Cannot find module './app'`.

- [ ] **Step 3: Implement popup controller and local UI primitives**

```ts
// src/ui/lib/chrome.ts
import type { Settings, VsrMessage } from "@src/upscaler/types";

import { CONTENT_SCRIPT_FILE, CONTENT_STYLE_FILE, ORT_RUNTIME_FILES } from "@src/shared/extension/injection";

export async function queryActiveTab(): Promise<chrome.tabs.Tab | undefined> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

export async function ensureInjected(tabId: number): Promise<void> {
  await chrome.scripting.insertCSS({
    target: { tabId },
    files: [CONTENT_STYLE_FILE],
  });

  await chrome.scripting.executeScript({
    target: { tabId },
    files: [...ORT_RUNTIME_FILES, CONTENT_SCRIPT_FILE],
  });
}

export async function sendToTab(tabId: number, message: VsrMessage): Promise<unknown> {
  return chrome.tabs.sendMessage(tabId, message);
}

export async function openOptionsPage(): Promise<void> {
  await chrome.runtime.openOptionsPage();
}
```

```ts
// src/ui/lib/popup-controller.ts
import type { Settings } from "@src/upscaler/types";

import { createMessage } from "@src/shared/extension/messages";
import { loadSettings, saveSettings } from "@src/shared/extension/storage";
import { normalizeSettings } from "@src/shared/extension/settings";

import { ensureInjected, queryActiveTab, sendToTab } from "./chrome";

export async function bootstrapPopup() {
  const tab = await queryActiveTab();
  const settings = normalizeSettings(await loadSettings());

  if (!tab?.id) {
    return { tabId: null, settings, state: null, statusText: "No active tab" };
  }

  try {
    await ensureInjected(tab.id);
    const state = await sendToTab(tab.id, createMessage("VSR_UPDATE", settings));
    return { tabId: tab.id, settings, state, statusText: "Connected" };
  } catch {
    return { tabId: tab.id, settings, state: null, statusText: "Injection failed" };
  }
}

export async function updatePopupSettings(tabId: number, settings: Settings) {
  const next = normalizeSettings(settings);
  await saveSettings(next);
  return sendToTab(tabId, createMessage("VSR_UPDATE", next));
}
```

```ts
// src/ui/lib/utils.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

```tsx
// src/ui/popup/app.tsx
import { Badge } from "@src/ui/components/ui/badge";
import { Button } from "@src/ui/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@src/ui/components/ui/card";

type PopupAppProps = {
  state: {
    statusText: string;
    hasVideo: boolean;
    engineLabel: string;
  };
  onOpenOptions: () => void;
  onRescan: () => void;
};

export function PopupApp({ state, onOpenOptions, onRescan }: PopupAppProps) {
  return (
    <main className="w-[360px] p-4">
      <Card className="border-[hsl(var(--border))] bg-[hsl(var(--card))]">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base font-semibold">
                Video GPU Super Resolution
              </CardTitle>
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                {state.statusText}
              </p>
            </div>
            <Badge>{state.engineLabel}</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between rounded-lg bg-[hsl(var(--muted))] px-3 py-2">
            <span className="text-sm">Video detected</span>
            <span className="text-sm font-medium">
              {state.hasVideo ? "Yes" : "No"}
            </span>
          </div>
          <div className="grid gap-2">
            <Button onClick={onRescan}>Rescan videos</Button>
            <Button variant="outline" onClick={onOpenOptions}>
              Full settings
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
```

```tsx
// src/ui/popup/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";

import "@src/ui/styles/globals.css";

import { PopupApp } from "./app";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <PopupApp
      state={{ statusText: "Loading...", hasVideo: false, engineLabel: "Tiny CNN" }}
      onOpenOptions={() => chrome.runtime.openOptionsPage()}
      onRescan={() => {}}
    />
  </React.StrictMode>,
);
```

- [ ] **Step 4: Run the popup UI test to verify it passes**

Run: `pnpm exec vitest run src/ui/popup/app.test.tsx`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ui src/shared/extension/injection.ts
git rm popup.html popup.ts styles/popup.css
git commit -m "feat: migrate popup to react"
```

## Task 4: Build The Options Page And Move Diagnostics Out Of Popup

**Files:**
- Create: `src/shared/extension/diagnostics.ts`
- Create: `src/ui/lib/options-controller.ts`
- Create: `src/ui/options/main.tsx`
- Create: `src/ui/options/app.tsx`
- Create: `src/ui/options/app.test.tsx`
- Modify: `README.md`

- [ ] **Step 1: Write the failing options-page test**

```tsx
// src/ui/options/app.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { OptionsApp } from "./app";

describe("OptionsApp", () => {
  it("renders diagnostics in options instead of popup", () => {
    render(
      <OptionsApp
        diagnosticsText="adapter: Intel"
        onRunDiagnostics={() => Promise.resolve()}
      />,
    );

    expect(screen.getByRole("heading", { name: /diagnostics/i })).toBeInTheDocument();
    expect(screen.getByText(/adapter: Intel/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run diagnostics/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the options-page test to verify it fails**

Run: `pnpm exec vitest run src/ui/options/app.test.tsx`

Expected: FAIL with `Cannot find module './app'`.

- [ ] **Step 3: Implement diagnostics helpers and the options page**

```ts
// src/shared/extension/diagnostics.ts
export function formatDiagnosticsSection(
  title: string,
  data: Record<string, unknown>,
): string {
  const lines = [title];

  for (const [key, value] of Object.entries(data)) {
    lines.push(`${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`);
  }

  return lines.join("\n");
}
```

```ts
// src/ui/lib/options-controller.ts
import { loadSettings, saveSettings } from "@src/shared/extension/storage";
import { normalizeSettings } from "@src/shared/extension/settings";

export async function loadOptionsModel() {
  const settings = normalizeSettings(await loadSettings());
  return { settings, diagnosticsText: "Diagnostics have not run yet." };
}

export async function persistOptionsModel(settings: Parameters<typeof normalizeSettings>[0]) {
  const next = normalizeSettings(settings);
  await saveSettings(next);
  return next;
}
```

```tsx
// src/ui/options/app.tsx
import { Button } from "@src/ui/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@src/ui/components/ui/card";

type OptionsAppProps = {
  diagnosticsText: string;
  onRunDiagnostics: () => Promise<void>;
};

export function OptionsApp({ diagnosticsText, onRunDiagnostics }: OptionsAppProps) {
  return (
    <main className="mx-auto max-w-5xl p-8">
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>General</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-[hsl(var(--muted-foreground))]">
              Manage engines, scaling, display mode, and target frame rate here.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Diagnostics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={() => void onRunDiagnostics()}>Run diagnostics</Button>
            <pre className="rounded-lg bg-[hsl(var(--muted))] p-4 text-xs">
              {diagnosticsText}
            </pre>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
```

```tsx
// src/ui/options/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";

import "@src/ui/styles/globals.css";

import { OptionsApp } from "./app";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <OptionsApp
      diagnosticsText="Diagnostics have not run yet."
      onRunDiagnostics={async () => {}}
    />
  </React.StrictMode>,
);
```

```md
<!-- README.md -->
## Architecture

- `src/content/` contains the injected page runtime and controller logic.
- `src/shared/extension/` contains shared settings, messages, storage, and diagnostics helpers.
- `src/ui/popup/` contains the React popup entry.
- `src/ui/options/` contains the React options page entry.
```

- [ ] **Step 4: Run the options-page test and the full build regression suite**

Run: `pnpm exec vitest run src/ui/options/app.test.tsx tests/build/extension-build.test.ts`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Run the full verification commands**

Run: `pnpm run typecheck`

Expected: PASS with no TypeScript errors.

Run: `pnpm run build`

Expected: PASS and emit `dist/popup.html`, `dist/options.html`, one injected `*.iife.js` file, and asset files.

Run: `Get-ChildItem dist -Recurse -Filter *.iife.js | Select-Object -ExpandProperty FullName`

Expected: exactly one emitted injected script path.

Run: `Select-String -Path (Get-ChildItem dist -Recurse -Filter *.iife.js | Select-Object -ExpandProperty FullName) -Pattern '^\s*import '`

Expected: no output, proving the emitted content script still has no top-level `import`.

- [ ] **Step 6: Commit**

```bash
git add README.md src/shared/extension/diagnostics.ts src/ui/options
git commit -m "feat: add options diagnostics page"
```

## Self-Review Checklist

- Spec coverage:
  - CRXJS upgrade is covered in Task 1 dependency updates.
  - React popup migration is covered in Task 3.
  - Options page and diagnostics relocation are covered in Task 4.
  - Content single-file build constraint is covered in Task 2 and Task 4 verification.
- Placeholder scan:
  - No `TODO`, `TBD`, or "implement later" placeholders remain.
- Type consistency:
  - Shared settings helpers use `Settings` from `src/upscaler/types.ts`.
  - Message helpers keep `VsrMessage` / `VsrMessageType`.
  - Popup/options controllers both route through the same storage normalization layer.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-crxjs-react-popup-upgrade.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
